"""Текст отчётов v2: страницу читает нейросеть со зрением, каждую цифру сверяем со спрятанным текстом PDF.

Зачем (19.09): Tesseract читает таблицу-мозаику и плитки инфографики построчно поперёк столбцов — подписи
отрываются от значений («511,2 mapa Р 22,3 P 24,0 % 29,1%» без понимания, что к чему). Сравнение на 2026_q2
(`src/ocr_bench.py`): Qwen 27B собирает плитки в таблицу «значение ↔ подпись» и не портит слова;
GLM-OCR быстрее, но подменяет слова правдоподобными («увеличились» → «уведомились»).

Проверка: в PDF под картинками спрятан исходный текст (ActualText от Word/PowerPoint) — цифры в нём исходные,
но без раскладки по столбцам. Каждое число с запятой из спрятанного текста страницы обязано найтись в
распознанной странице; числа, которых нет нигде в спрятанном тексте отчёта, — в список «не подтверждены»
(это либо крупные цифры инфографики, которых в спрятанном тексте нет, либо ошибка распознавания).

Старый текст (text/) не трогаем: прошлые прогоны остаются сравнимыми. Новый — text_v2/, со своими хешами.
Уже распознанные отчёты не перезаписываются (текст замораживается).

Запуск: .venv/bin/python src/ocr_v2.py [отчёт ...]      без аргументов — все отчёты из pdf/
        .venv/bin/python src/ocr_v2.py selftest
"""
import base64
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_bench import numbers  # noqa: E402 — тот же разбор чисел, что в сравнении распознавателей

ROOT = Path(__file__).resolve().parent.parent
PDF, OUT, CHECK = ROOT / "pdf", ROOT / "text_v2", ROOT / "notes" / "ocr_v2_check"
HASHES = OUT / "hashes.json"
LMS = str(Path.home() / ".lmstudio/bin/lms")
API = "http://localhost:50255/v1/chat/completions"
MODEL_KEY, MODEL_ID = "qwen/qwen3.8-27b", "ocr-qwen3.8-27b"
DPI = 150
PROMPT = ("Перепиши эту страницу пресс-релиза банка полностью, в порядке чтения. "
          "Таблицы и блоки с показателями выведи таблицами Markdown: каждое значение — в своей ячейке, "
          "подпись показателя — в той же строке или том же столбце, что и его значение; заголовки столбцов сохрани. "
          "Номера сносок пиши надстрочными символами (¹ ² ³), не сливая их с числами. "
          "Цифры, знаки, единицы и проценты переписывай в точности как на странице. Ничего не добавляй и не пропускай.")
NO_THINK = "<think>\n\n</think>\n\n"


def strip_fences(text):
    """Модель иногда оборачивает ответ в ```markdown … ```."""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*\n", "", t)
    t = re.sub(r"\n```\s*$", "", t)
    return t.strip()


def load():
    subprocess.run([LMS, "unload", "--all"], capture_output=True, timeout=120)
    r = subprocess.run([LMS, "load", MODEL_KEY, "-c", "16384", "--parallel", "1", "--identifier", MODEL_ID, "-y"],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"модель не загрузилась: {(r.stdout + r.stderr)[-300:]}")


def read_page(page):
    b64 = base64.b64encode(page.get_pixmap(dpi=DPI).tobytes("png")).decode()
    body = {"model": MODEL_ID, "temperature": 0, "top_k": 1, "top_p": 1, "max_tokens": 8000, "stream": False,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]},
                {"role": "assistant", "content": NO_THINK}]}
    resp = httpx.post(API, json=body, timeout=1800).json()
    ch = resp["choices"][0]
    return strip_fences(ch["message"].get("content") or ""), ch.get("finish_reason")


def verify(page_hidden, page_read, report_hidden):
    """Сверка страницы. Возвращает (доля найденных чисел, пропущенные, неподтверждённые)."""
    g, got = numbers(page_hidden), numbers(page_read)
    # в спрятанном тексте соседние ячейки слипаются («215,3195,0»), поэтому подтверждение — подстрокой без пробелов
    flat = re.sub(r"\s", "", report_hidden)
    missing = sorted(n for n, c in g.items() if got[n] < c)
    unconfirmed = sorted(n for n in got if n not in flat)
    found = sum(min(c, got[n]) for n, c in g.items())
    return (found / sum(g.values()) if g else 1.0), missing, unconfirmed


def ocr_report(rep):
    doc = pymupdf.open(PDF / f"{rep}.pdf")
    hidden = [p.get_text("text") for p in doc]
    report_hidden = "\n".join(hidden)
    pages, rows, t0 = [], [], time.time()
    for i, page in enumerate(doc):
        text, finish = read_page(page)
        rec, missing, unconf = verify(hidden[i], text, report_hidden)
        pages.append(text)
        rows.append(dict(page=i + 1, recall=round(rec, 3), missing=missing, unconfirmed=unconf,
                         truncated=finish == "length", chars=len(text)))
        print(f"  {rep} стр.{i + 1}: чисел найдено {rec:.0%}, пропущено {len(missing)}, "
              f"не подтверждено {len(unconf)}{', ОБРЕЗАНО' if finish == 'length' else ''}", flush=True)
    return "\n\f\n".join(pages), rows, round(time.time() - t0)


def write_check(rep, rows, sec):
    CHECK.mkdir(parents=True, exist_ok=True)
    lines = [f"# Проверка распознавания {rep} (text_v2)\n",
             f"Модель {MODEL_KEY}, {DPI} dpi, {sec} с. Эталон — спрятанный текст PDF.\n",
             "| стр. | чисел найдено | пропущены | не подтверждены | обрезано |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['page']} | {r['recall']:.0%} | {', '.join(r['missing']) or '—'} | "
                     f"{', '.join(r['unconfirmed']) or '—'} | {'ДА' if r['truncated'] else ''} |")
    (CHECK / f"{rep}.md").write_text("\n".join(lines) + "\n")


def main(reports):
    OUT.mkdir(exist_ok=True)
    hashes = json.loads(HASHES.read_text()) if HASHES.exists() else {}
    todo = [r for r in reports if not (OUT / f"{r}.txt").exists()]
    print(f"к распознаванию: {len(todo)}, уже заморожено: {len(hashes)}", flush=True)
    if not todo:
        return
    load()
    try:
        for rep in todo:
            text, rows, sec = ocr_report(rep)
            out = OUT / f"{rep}.txt"
            out.write_text(text)
            write_check(rep, rows, sec)
            recall = sum(r["recall"] for r in rows) / len(rows)
            hashes[rep] = {"sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                           "engine": f"{MODEL_KEY} vision, T=0, {DPI} dpi", "pages": len(rows), "chars": len(text),
                           "mean_page_recall": round(recall, 3),
                           "pages_not_full": [r["page"] for r in rows if r["recall"] < 1 or r["truncated"]],
                           "created": dt.datetime.now().isoformat(timespec="seconds")}
            HASHES.write_text(json.dumps(hashes, ensure_ascii=False, indent=1))
            print(f"{rep}: готово за {sec} с, средняя доля чисел {recall:.1%}", flush=True)
    finally:
        subprocess.run([LMS, "unload", "--all"], capture_output=True, timeout=120)


def selftest():
    assert strip_fences("```markdown\n# А\n| 1 |\n```") == "# А\n| 1 |"
    assert strip_fences("текст") == "текст"
    rec, miss, unc = verify("Чистая прибыль 511,2 422,9", "| 511,2 | 422,9 | 99,9 |", "511,2 422,9 и 20,9")
    assert rec == 1.0 and miss == [] and unc == ["99,9"], (rec, miss, unc)
    # слипшиеся ячейки спрятанного текста: верно прочитанное число не должно попасть в «не подтверждены»
    _, _, unc = verify("", "| 215,3 | 195,0 | 1 066,8 |", "Чистые комиссионные доходы215,3195, 0 и 1 066, 8")
    assert unc == [], unc
    rec, miss, _ = verify("12,5 и 12,5", "12,5", "12,5")
    assert rec == 0.5 and miss == ["12,5"]  # повтор числа на странице тоже должен найтись
    print("ocr_v2: самопроверка пройдена")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        main(sys.argv[1:] or sorted(p.stem for p in PDF.glob("*.pdf")))
