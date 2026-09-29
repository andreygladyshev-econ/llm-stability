"""PDF → текст (Tesseract) → text/{report}.txt + text/hashes.json.

Уже распознанные отчёты не перезаписываются: текст замораживается.
Запуск: .venv/bin/python src/ocr.py
"""
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
PDF, TEXT = ROOT / "pdf", ROOT / "text"
HASHES = TEXT / "hashes.json"
DPI = 300

# Tesseract путает похожие латинские и русские буквы. Чиним только однозначные случаи.
WORD_FIX = {"Ha": "на", "HA": "НА", "He": "не", "3a": "за", "Bo": "Во"}
ZERO_FIX = re.compile(r"(?<![\wА-Яа-яЁё])[OoОо](?=[,.]\d)")  # «O,8» → «0,8»


def fix(text):
    n = 0

    def word(m):
        nonlocal n
        n += 1
        return WORD_FIX[m.group(0)]

    text = re.sub(r"\b(" + "|".join(map(re.escape, WORD_FIX)) + r")\b", word, text)
    text, z = ZERO_FIX.subn("0", text)
    return text, n + z


def ocr_page(page):
    png = page.get_pixmap(dpi=DPI).tobytes("png")
    r = subprocess.run(["tesseract", "stdin", "stdout", "-l", "rus+eng", "--psm", "3"],
                       input=png, capture_output=True, check=True)
    return r.stdout.decode()


def ocr_report(pdf):
    doc = pymupdf.open(pdf)
    text, fixes = fix("\n\f\n".join(ocr_page(p) for p in doc))  # \f — граница страниц
    return pdf.stem, text, fixes, len(doc)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    hashes = json.loads(HASHES.read_text()) if HASHES.exists() else {}
    todo = [p for p in sorted(PDF.glob("*.pdf")) if not (TEXT / f"{p.stem}.txt").exists()]
    print(f"к распознаванию: {len(todo)}, уже заморожено: {len(hashes)}")
    with cf.ThreadPoolExecutor(4) as ex:
        for rid, text, fixes, pages in ex.map(ocr_report, todo):
            out = TEXT / f"{rid}.txt"
            out.write_text(text)
            hashes[rid] = {"sha256": sha(out), "engine": "tesseract rus+eng psm3", "dpi": DPI,
                           "pages": pages, "chars": len(text), "letter_fixes": fixes,
                           "created": dt.datetime.now().isoformat(timespec="seconds")}
            print(rid, pages, "стр.", len(text), "симв., исправлений", fixes)
            HASHES.write_text(json.dumps(hashes, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
