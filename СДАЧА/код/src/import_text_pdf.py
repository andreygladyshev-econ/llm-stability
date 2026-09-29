"""PDF с настоящим текстовым слоем (например, печать веб-страницы) → text_v2/, без распознавания.

Пресс-релизы некоторых банков выложены только страницей сайта. Напечатанная в PDF страница содержит обычный
текст, поэтому читать её моделью со зрением незачем: цифры и так исходные. В hashes.json это помечается
отдельным engine, чтобы в записке не перепутать способ чтения.

Запуск: .venv/bin/python src/import_text_pdf.py "pdf_other/Новости ВТБ.pdf" vtb_2026_q1
"""
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "text_v2"


def main(src, name):
    doc = pymupdf.open(src)
    text = "\n\f\n".join(p.get_text("text") for p in doc)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    assert len(text) > 3000, f"текста слишком мало ({len(text)}): похоже, страница всё-таки картинка"
    (OUT / f"{name}.txt").write_text(text)
    h = json.loads((OUT / "hashes.json").read_text())
    h[name] = {"sha256": hashlib.sha256(text.encode()).hexdigest(),
               "engine": "текстовый слой PDF (печать веб-страницы), распознавание не требуется",
               "pages": len(doc), "chars": len(text), "source": str(src),
               "created": dt.datetime.now().isoformat(timespec="seconds")}
    (OUT / "hashes.json").write_text(json.dumps(h, ensure_ascii=False, indent=1))
    print(f"{name}: {len(text)} символов, {len(doc)} стр.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
