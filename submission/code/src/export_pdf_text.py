"""Текстовый слой PDF → text_pdf/{отчёт}.txt (29.09).

Сканы Сбера содержат скрытый текстовый слой. Для чтения моделью он не годится (таблицы идут потоком, числа
разбиты пробелами: «6 , 7 %»), но буквы и цифры в нём те же, что в отчёте. По нему сверяются цитаты сдачи:
src/pdf_quotes.py берёт из него символы, а из распознанного текста (text_v2/) — пробелы и границы ячеек таблиц.
Слой выгружается один раз и хранится в репозитории вместе с контрольными суммами, чтобы пересборка сдачи не
зависела от версии PyMuPDF. Запуск: .venv/bin/python src/export_pdf_text.py
"""
import hashlib
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "text_pdf"
MAPPING = json.loads((ROOT / "notes" / "pdf_mapping.json").read_text())
# PDF заказчика лежат в папке assignment/ в корне репозитория: рядом с рабочим проектом или на уровень выше папки кода
PDF_DIR = next(p for p in (ROOT.parent / "assignment" / "SBER", ROOT.parent.parent / "assignment" / "SBER")
               if p.is_dir())


def main():
    OUT.mkdir(exist_ok=True)
    hashes = {}
    sources = {rep: PDF_DIR / name for rep, name in MAPPING.items()}
    # отчёты ВТБ и Т-Технологий для проверки переноса — текстовые PDF с сайтов банков (pdf_other/)
    sources.update({p.stem: p for p in sorted((ROOT / "pdf_other").glob("*.pdf"))})
    for rep, path in sorted(sources.items()):
        name = path.name
        text = "\n".join(page.get_text() for page in pymupdf.open(path))
        (OUT / f"{rep}.txt").write_text(text)
        hashes[rep] = {"pdf": name, "sha256": hashlib.sha256(text.encode()).hexdigest()}
        print(f"{rep}: {len(text)} знаков")
    (OUT / "hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
