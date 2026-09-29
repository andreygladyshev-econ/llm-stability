"""Проверка распознавания №2: стоят ли числа в СВОИХ столбцах (перестановка столбцов = перепутанные периоды).

Первая проверка (`ocr_v2.verify`) отвечает «есть ли число на странице». Эта — «в том ли порядке». В скрытом
текстовом слое PDF числа строки таблицы идут в порядке чтения. Для каждой строки таблицы распознанного текста
с двумя и более числами проверяем: встречаются ли её числа в скрытом слое страницы в том же порядке
(как подпоследовательность). Нарушение порядка — кандидат на переставленные столбцы, смотрится глазами.
Заодно печатается доля чисел скрытого слоя, найденных в распознанном тексте (та же мера, что `ocr_v2.verify`):
по каждому отчёту и в целом (29.09: чтобы число в записке воспроизводилось без повторного распознавания).
PDF берутся из pdf/ рабочего проекта, а если её нет — из папки assignment/SBER по notes/pdf_mapping.json.
Запуск: .venv/bin/python src/ocr_order_check.py
"""
import json
import re
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ocr_v2  # noqa: E402
import texts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAPPING = json.loads((ROOT / "notes" / "pdf_mapping.json").read_text())


def pdf_path(rep):
    own = ROOT / "pdf" / f"{rep}.pdf"
    if own.exists():
        return own
    sber = next(p for p in (ROOT.parent / "assignment" / "SBER", ROOT.parent.parent / "assignment" / "SBER")
                if p.is_dir())
    return sber / MAPPING[rep]
NUM = re.compile(r"-?\d[\d ]*,\d+")                 # числа с десятичной запятой: так же, как первая проверка


def nums(s):
    return [re.sub(r"\s", "", n).lstrip("-") for n in NUM.findall(s)]


def in_order(seq, hidden_flat):
    """Все числа seq встречаются в hidden_flat слева направо (каждое после предыдущего)."""
    pos = 0
    for n in seq:
        i = hidden_flat.find(n, pos)
        if i < 0:
            return False, n
        pos = i + len(n)
    return True, None


def check(rep):
    pages = (texts.text_dir("v2") / f"{rep}.txt").read_text().split("\n\f\n")
    doc = pymupdf.open(pdf_path(rep))
    rows = bad = found = total = 0
    issues = []
    for pno, (page_text, page) in enumerate(zip(pages, doc), 1):
        g, got = ocr_v2.numbers(page.get_text("text")), ocr_v2.numbers(page_text)
        total += sum(g.values()); found += sum(min(c, got[n]) for n, c in g.items())
        hidden = re.sub(r"\s", "", page.get_text("text")).replace("−", "-").replace("-", "")
        for line in page_text.splitlines():
            if not line.lstrip().startswith("|"):
                continue
            seq = [n for n in nums(line) if n in hidden]       # числа, которых нет в слое, проверяет первая проверка
            if len(seq) < 2:
                continue
            rows += 1
            ok, where = in_order(seq, hidden)
            if not ok:
                bad += 1
                issues.append((pno, line.strip()[:150], where))
    return rows, bad, issues, found, total


def main():
    total_rows = total_bad = all_found = all_total = 0
    shares = []
    for rep in sorted(MAPPING):
        rows, bad, issues, found, total = check(rep)
        total_rows += rows; total_bad += bad; all_found += found; all_total += total
        shares.append(found / total)
        print(f"{rep}: чисел скрытого слоя найдено {found / total:.1%}; строк таблиц {rows}, порядок нарушен в {bad}")
        for pno, line, where in issues[:6]:
            print(f"   стр.{pno}: {line}   [сбой на {where}]")
    print(f"\nИТОГО: чисел скрытого слоя найдено {all_found / all_total:.1%} (по отчётам от {min(shares):.1%} до "
          f"{max(shares):.1%}); строк таблиц с 2+ числами {total_rows}, порядок нарушен в {total_bad} "
          f"({total_bad / max(1, total_rows):.1%})")


if __name__ == "__main__":
    main()
