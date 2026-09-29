"""Сравнение распознавателей на страницах отчётов: какая модель вернее читает цифры и таблицы.

Эталон — спрятанный текст PDF (ActualText, сохранённый Word/PowerPoint при экспорте): он не привязан к
координатам, но цифры в нём исходные, не распознанные. Каждый распознаватель читает картинку страницы;
считаем, какая доля эталонных чисел нашлась в его выводе и сколько чисел он «придумал» (нет в эталоне).

Запуск: .venv/bin/python src/ocr_bench.py 2026_q2 2 [модель ...]
  (номер страницы с нуля; модели — ключи LM Studio или "tesseract")
"""
import base64
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
LMS = str(Path.home() / ".lmstudio/bin/lms")
API = "http://localhost:50255/v1/chat/completions"
OUT = ROOT / "notes" / "ocr_bench"
PROMPT = ("Распознай страницу полностью. Таблицы выведи в формате Markdown со всеми строками и столбцами, "
          "заголовки столбцов сохрани. Цифры переписывай точно, как на странице. Ничего не добавляй от себя.")


def numbers(text):
    """Числа с дробной частью в русской записи: «1 066,8», «20,9». Пробелы вокруг запятой убираем.

    Целые без запятой не считаем: там слишком много мусора (номера страниц, сносок, годов).
    """
    t = re.sub(r"(\d)\s*,\s*(\d)", r"\1,\2", text.replace(" ", " "))
    t = re.sub(r"(?<![,\d])(\d{1,3})\s(\d{3}),", r"\1\2,", t)  # «1 066,8» → «1066,8», но не хвост дроби «511,2 422,9»
    return Counter(re.findall(r"(?<![\d,])\d{1,4},\d(?!\d)", t))


def gold(pdf, page_no):
    return numbers(pymupdf.open(pdf)[page_no].get_text("text"))


def page_png(pdf, page_no, dpi):
    return pymupdf.open(pdf)[page_no].get_pixmap(dpi=dpi).tobytes("png")


def run_tesseract(png):
    r = subprocess.run(["tesseract", "stdin", "stdout", "-l", "rus+eng", "--psm", "3"],
                       input=png, capture_output=True, check=True)
    return r.stdout.decode()


def run_lms(model, png, max_tokens=6000):
    subprocess.run([LMS, "unload", "--all"], capture_output=True, timeout=120)
    r = subprocess.run([LMS, "load", model, "-c", "16384", "-y"], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"не загрузилась: {(r.stdout + r.stderr)[-300:]}")
    b64 = base64.b64encode(png).decode()
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "stream": False,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
    if "qwen" in model.lower():  # размышления выключаем, как в боевом режиме
        body["messages"].append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})
    resp = httpx.post(API, json=body, timeout=900).json()
    if "choices" not in resp:
        raise RuntimeError(str(resp)[:300])
    return resp["choices"][0]["message"].get("content") or ""


def score(g, got):
    found = sum(min(c, got[n]) for n, c in g.items())
    extra = sum(c for n, c in got.items() if n not in g)
    return found / max(1, sum(g.values())), extra


def main():
    rep, page_no = sys.argv[1], int(sys.argv[2])
    models = sys.argv[3:] or ["tesseract"]
    pdf = ROOT / "pdf" / f"{rep}.pdf"
    g = gold(pdf, page_no)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{rep}, стр. {page_no + 1}: эталонных чисел {sum(g.values())} (разных {len(g)})")
    for m in models:
        t0 = time.time()
        try:
            png = page_png(pdf, page_no, 300 if m == "tesseract" else 150)
            text = run_tesseract(png) if m == "tesseract" else run_lms(m, png)
        except Exception as e:  # noqa: BLE001 — одна модель не должна ронять сравнение
            print(f"  {m:40} ОШИБКА: {e}")
            continue
        sec = time.time() - t0
        (OUT / f"{rep}_p{page_no + 1}__{re.sub(r'[^\w.-]+', '-', m)}.md").write_text(text)
        rec, extra = score(g, numbers(text))
        print(f"  {m:40} нашла {rec:6.1%} эталонных чисел | лишних {extra:3} | {sec:5.0f} с | {len(text)} симв.")
    subprocess.run([LMS, "unload", "--all"], capture_output=True, timeout=120)


def selftest():
    assert numbers("Чистая прибыль 511,2 422,9 20, 9 %") == Counter({"511,2": 1, "422,9": 1, "20,9": 1})
    assert numbers("1 066, 8 и 2 051,1") == Counter({"1066,8": 1, "2051,1": 1})
    assert numbers("стр. 3, 2026 год") == Counter()
    assert score(Counter({"1,0": 2, "2,0": 1}), Counter({"1,0": 1, "9,9": 1})) == (1 / 3, 1)
    print("ocr_bench: самопроверка пройдена")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else main()
