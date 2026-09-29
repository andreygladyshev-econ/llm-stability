"""Версии замороженного текста отчётов.

v1 — Tesseract по картинке страницы (text/): таблицы и плитки инфографики читаются поперёк столбцов.
v2 — нейросеть со зрением (Qwen 27B), каждая цифра сверена со спрятанным текстом PDF (text_v2/, src/ocr_v2.py).
Прогоны на разных версиях несравнимы напрямую: версия пишется в meta.text_version и в имя варианта (-tx2).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = {"v1": ROOT / "text", "v2": ROOT / "text_v2"}
# Пересобранные файлы сдачи пишутся в code/rebuild/, чтобы не затирать сданные и их можно было сравнить побайтно.
SUBMIT = ROOT / "rebuild"


def text_dir(version="v1"):
    return DIRS[version]


def hashes(version="v1"):
    p = DIRS[version] / "hashes.json"
    return json.loads(p.read_text()) if p.exists() else {}
