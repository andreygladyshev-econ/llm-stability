"""Цитата сдачи, дословная по тексту PDF (29.09).

Цитату модели программа находит в распознанном тексте (text_v2/) с допуском на ошибки распознавания; раньше в
таблицу шёл этот фрагмент как есть. Аудит сверил все цитаты со скрытым текстовым слоем PDF: в семи из них
распознавание исказило слово («показал» вместо «показав», «Ber» вместо «Sber»), числа везде верные.

Здесь фрагмент выравнивается по текстовому слою PDF (text_pdf/, src/export_pdf_text.py): буквы и цифры берутся
из PDF, из распознанного текста — только пробелы и границы ячеек таблицы «|», которых в слое PDF нет или они
случайны («6 , 7 %»). Дефис, который есть только в PDF, — перенос строки («докумен - тарного»), он опускается.
Номер сноски пишется надстрочным знаком, как в распознанном тексте.

Проверка: цитата без пробелов, «|», дефисов и с обычными цифрами вместо надстрочных — подстрока текста PDF,
приведённого к тому же виду (verified_in_pdf).
"""
import difflib
import re

from rapidfuzz import fuzz

SUP = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789"))
TO_SUP = {v: k for k, v in SUP.items()}
SEPARATORS = set(" \t\n\r  |")
MARKUP = set("*#_`")


def key(c):
    """Символ для сравнения: регистр, «ё», кавычки, тире и латиница-двойник кириллицы не различаются."""
    c = SUP.get(c, c).lower()
    return {"ё": "е", "«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "−": "-", "–": "-", "—": "-",
            "h": "н", "m": "м", "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
            "k": "к", "b": "в", "t": "т"}.get(c, c)


def pdf_chars(text):
    """Символы слоя PDF без пробелов и служебных знаков: [(ключ, символ, пробел перед ним в PDF)]."""
    out, gap = [], False
    for c in text:
        if c.isspace() or "" <= c <= "" or c == "­":
            gap = True
            continue
        out.append((key(c), c, gap))
        gap = False
    return out


def ocr_chars(frag):
    """Символы распознанного фрагмента: [(ключ, символ, разделитель перед ним: '', ' ' или ' | ')]."""
    out, sep = [], ""
    for c in frag:
        if c in SEPARATORS:
            sep = " | " if (c == "|" or sep == " | ") else " "
            continue
        if c in MARKUP:
            continue
        out.append((key(c), c, sep))
        sep = ""
    return out


def is_sup(c):
    return c in SUP


def styled(pdf_c, ocr_c, prev_out, sep=""):
    """Цифра сноски — надстрочная: так было в распознанном тексте или она продолжает надстрочный номер."""
    if pdf_c.isdigit() and (is_sup(ocr_c or "") or (not sep and prev_out and is_sup(prev_out[-1]))):
        return TO_SUP[pdf_c]
    return pdf_c


def footnote(chars, prev_out):
    """Вставка из PDF — номер сноски: одна-две цифры сразу после слова, знака «%», «.», «)» или другой сноски."""
    return (len(chars) <= 2 and all(c[1].isdigit() for c in chars) and bool(prev_out)
            and (prev_out[-1].isalpha() or prev_out[-1] in "%.,)" or is_sup(prev_out[-1])))


def align(qk, pk, lo, hi):
    """Опкоды на окне PDF чуть шире найденного; вставки из PDF по краям окна отбрасываются."""
    lo, hi = max(0, lo - 6), min(len(pk), hi + 6)
    ops = difflib.SequenceMatcher(None, qk, pk[lo:hi], autojunk=False).get_opcodes()
    while ops and ops[0][0] == "insert":
        ops = ops[1:]
    while ops and ops[-1][0] == "insert":
        ops = ops[:-1]
    return lo, ops


def to_pdf(frag, pdf_text):
    """Фрагмент распознанного текста → тот же фрагмент буквами PDF. None, если в PDF его не найти."""
    q = ocr_chars(frag)
    p = pdf_chars(pdf_text)
    if not q:
        return None
    qk, pk = "".join(x[0] for x in q), "".join(x[0] for x in p)
    a = fuzz.partial_ratio_alignment(qk, pk, score_cutoff=85)
    if a is None:
        return None
    lo, ops = align(qk, pk, a.dest_start, a.dest_end)
    seg = p[lo:]
    out, carry = "", None                     # carry — разделитель для следующего символа вместо его собственного
    for op, i1, i2, j1, j2 in ops:
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                sep = q[i1 + k][2] if carry is None else carry
                carry = None
                out += sep + styled(seg[j1 + k][1], q[i1 + k][1], out, sep)
        elif op == "delete":                  # есть только в распознанном тексте: выбрасываем, разделитель переносим
            if q[i1][2] and not carry:
                carry = q[i1][2]
        else:                                 # вставка из PDF или замена разной длины
            chars = seg[j1:j2]
            nq = q[i2] if op == "replace" and i2 < len(q) else (q[i1] if i1 < len(q) else None)
            if op == "insert" and all(c[1] in "-‐" for c in chars):
                continue                      # перенос строки в PDF
            if op == "insert" and footnote(chars, out):
                out += "".join(TO_SUP[c[1]] for c in chars)
                continue
            own = q[i1][2] if op == "replace" else (nq[2] if nq else "")
            sep = own if carry is None else carry
            carry = None
            if op == "insert" and chars[0][1] in ".,;:!?%)»":
                sep = ""
            for k, (_, c, gap) in enumerate(chars):
                inner = " " if (k and gap and c.isalnum() and chars[k - 1][1].isalnum()) else ""
                out += (sep if k == 0 else inner) + styled(c, q[i1][1] if op == "replace" else None, out,
                                                           sep if k == 0 else inner)
            if op == "insert" and nq is not None:
                if chars[-1][1].isalnum() and nq[1].isalnum():   # вставка внутри слова или перед словом — как в PDF
                    carry = " " if (j2 < len(seg) and seg[j2][2]) else ""
                elif chars[-1][1] in "(«„":
                    carry = ""
    return out.strip()


def flat(s):
    """Вид для проверки: без пробелов, разметки, «|», дефисов; надстрочные цифры обычные."""
    return "".join(key(c) for c in s if not (c in SEPARATORS or c in MARKUP or c in "-‐–—−" or c.isspace()
                                          or "" <= c <= "" or c == "­"))


def verified_in_pdf(quote, pdf_text):
    return bool(quote) and flat(quote) in flat(pdf_text)
