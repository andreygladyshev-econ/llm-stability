"""H3: баллы по извлечённым числам считает Python, а не модель.

Числовые показатели (17 из 24) считаются по порогам спеки; качественные (7) остаются за моделью.
Единственное место, где пороги живут в коде. Самопроверка: python src/score_rules.py
"""
import re

# kind: money (пороги 3/15 %), ratio (0,3/1,5 пп), cor (0,2/0,5 пп), opex (разрыв ±1/±10 пп)
# inverted: рост цифры = минус для акционера
NUMERIC = {
    "net_profit": ("money", False), "eps": ("money", False), "nii": ("money", False),
    "fee_income": ("money", False), "provisions": ("money", True),
    "corporate_loans": ("money", False), "retail_loans": ("money", False),
    "customer_funds": ("money", False), "book_value_per_share": ("money", False),
    "active_clients": ("money", False), "digital_metrics": ("money", False),
    "roe": ("ratio", False), "nim": ("ratio", False), "cir": ("ratio", True),
    "capital_adequacy": ("ratio", False),
    "cor": ("cor", True),
    "opex": ("opex", False),
}
JUDGMENT = ["guidance", "portfolio_quality", "market_share", "dividends",
            "tech_development", "external_conditions", "ceo_tone"]
THRESHOLDS = {"money": (3.0, 15.0), "ratio": (0.3, 1.5), "cor": (0.2, 0.5)}
NEG_WORDS = ("сниж", "паден", "сократ", "уменьш", "минус", "ниже")


SUPERSCRIPT = str.maketrans("", "", "⁰¹²³⁴⁵⁶⁷⁸⁹")
NORM_NAME = re.compile(r"(?<![A-Za-zА-Яа-яЁё])[НнHh]\s?\d{1,2}(?:[.,]\d)?(?![\d,.])")  # «Н20.0», «H1.2» — имя норматива, а не число


def number(s, words=True):
    """«511,2 млрд руб.» → 511.2; «-0,3 п.п.» → -0.3; «нет» → None.

    words=True: слова «снижение», «ниже» и т. п. дают минус — так записывают изменение («снизилась на 1,1 пп»).
    words=False: для уровня («27,9%, ниже целевого») слова знак не меняют, минус только явный.
    """
    if not s:
        return None
    # 22.09: GPT-OSS пишет разряды узким неразрывным пробелом (U+202F) — падало на float(). Любые пробельные
    # символы Юникода считаем разделителем разрядов.
    s = re.sub(r"[\s\u00a0\u202f\u2007\u2009]+", " ", str(s).replace("−", "-").replace("–", "-")).translate(SUPERSCRIPT)
    s = NORM_NAME.sub(" ", s)  # 29.09: «Н20.0: 13,7%» читалось как 20,0
    m = re.search(r"[-+]?\d[\d\s]*(?:[.,]\d+)?", s)
    if not m:
        return None
    v = float(re.sub(r"\s", "", m.group(0)).replace(",", "."))
    if v > 0 and (s.strip().startswith("-") or (words and any(w in s.lower() for w in NEG_WORDS))):
        v = -v
    return v


def is_bp(s):
    """Базисные пункты: «116 бп» = 1,16 пп."""
    return bool(s) and bool(re.search(r"\bбп\b|б\.\s?п\.|базисн|\bbps?\b", str(s).lower()))


def in_pp(s, words=True):
    """Число в процентных пунктах: «-3 бп» → -0,03; «-0,03 пп» → -0,03."""
    v = number(s, words)
    return None if v is None else (v / 100 if is_bp(s) else v)


def is_pp(s):
    return bool(s) and bool(re.search(r"п\.?\s?п|процентн\w* пункт|\bpp\b", str(s).lower()))


SCALE = (("трлн", 1e12), ("млрд", 1e9), ("млн", 1e6), ("тыс", 1e3))


def scale(s):
    s = str(s or "").lower()
    return next((k for w, k in SCALE if w in s), None)


def multiple(s):
    """Кратное изменение → проценты: «4,5x», «×4,5», «в 4,5 раза» → +350; «снизились в 2 раза» → −50. Иначе None."""
    t = str(s or "").lower()
    if not re.search(r"\d\s*[xх×]|[xх×]\s*\d|\bраз", t):
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", t)
    if not m:
        return None
    k = float(m.group(0).replace(",", "."))
    if k <= 0:
        return None
    down = any(w in t for w in NEG_WORDS + ("сниз", "упал")) or t.strip().startswith("-")
    return (1 / k - 1) * 100 if down else (k - 1) * 100


def change_of(indicator, value, base, change):
    """Изменение показателя: берём названное в отчёте, иначе считаем из value и base.

    Ночь 5 показала три способа, которыми модель раскладывает числа не так, как ждёт код:
    расходы из таблиц с минусом («-555,4» / «-141,5»), потерянный минус в change («52,3%» при падении
    с 555 до 265), темп прироста в поле value («29,3%» при пустых base и change).
    """
    kind = NUMERIC[indicator][0]
    lo = THRESHOLDS[kind][0]
    # value и base — уровни: слова рядом с числом («ниже целевого») знак уровня не меняют (29.09)
    v, b = ((in_pp(value, False), in_pp(base, False)) if kind in ("ratio", "cor")
            else (number(value, False), number(base, False)))
    from_amounts = None
    # темп («29,3%») в одном поле и сумма («16,1 трлн») в другом — несравнимы (ночь 6: знак переворачивался)
    if v is not None and b not in (None, 0) and ("%" in str(value)) == ("%" in str(base)):
        sv, sb = scale(value), scale(base)
        if sv and sb:  # «26 547 млрд» против «23,3 трлн»
            v, b = v * sv, b * sb
        if kind == "money" and v < 0 and b < 0:  # в таблицах расходы записаны с минусом
            v, b = -v, -b
        from_amounts = v - b if kind in ("ratio", "cor") else (v - b) / abs(b) * 100
    # «+0,2 млн» — это не изменение в процентах: такие значения считаем сами из value и base
    stated = in_pp(change) if change and (is_pp(change) or is_bp(change) or "%" in str(change)) else None
    mult = multiple(change)
    if stated is None and mult is not None and kind == "money":
        stated = mult  # 23.09: «+4,5x», «в 4,5 раза» читалось как +4,5% (2024_q4, резервы: −1 вместо −2)
    if stated is None and kind == "money" and v is not None and not b and change:
        # база не выписана, но названо изменение в тех же единицах: «109,5 млн» и «+0,5 млн» → +0,46%
        c, sc, sv = number(change), scale(change), scale(value)
        if c not in (None, 0) and "%" not in str(change) and (sc == sv or not sc):
            prev = v - c
            if prev:
                stated = c / abs(prev) * 100
    if stated is None and kind == "money" and not base and "%" in str(value) and not is_pp(value):
        stated = number(value)  # темп прироста попал в поле value
    if stated is not None:
        # названный темп приоритетнее (он бывает без валютной переоценки), но знак сверяем с суммами
        if from_amounts is not None and abs(from_amounts) >= lo and stated * from_amounts < 0:
            stated = -stated
        return stated
    return from_amounts


def score(indicator, value="", base="", change=""):
    """Балл по извлечённым числам. None — считать нельзя, нужен балл модели."""
    if indicator not in NUMERIC:
        return None
    kind, inverted = NUMERIC[indicator]
    if kind == "opex":
        # разрыв = рост операционного дохода до резервов − рост операционных расходов, в п.п.
        # рост расходов — в value (в старых ответах бывал в change); change может держать уже готовый разрыв — он не нужен
        opex_growth, income_growth = (number(value) if value else number(change)), number(base)
        if opex_growth is None:
            return None  # модель не выписала рост расходов — считать нечем, берём её балл
        if income_growth is None:
            return 0  # правило спеки: операционного дохода до резервов нет — 0
        gap = round(income_growth - opex_growth, 6)
        return 0 if abs(gap) <= 1 else (1 if gap <= 10 else 2) if gap > 0 else (-1 if gap >= -10 else -2)
    c = change_of(indicator, value, base, change)
    if c is None:
        return None  # не молчаливый ноль: показатель найден, но числа не разобрать — остаётся балл модели
    c = round(c, 6)  # 29.09: 14,6 − 14,3 = 0,29999… без округления давало 0 вместо ±1
    lo, hi = THRESHOLDS[kind]
    mag = 0 if abs(c) < lo else (1 if abs(c) <= hi else 2)
    sign = (1 if c > 0 else -1) * (-1 if inverted else 1)
    return mag * sign


# v7: модель выписывает обе ячейки периода, выбирает код (правила «Какой период» в спеке).
# На дату — сначала «с начала года»; за период — сначала «квартал год к году».
BALANCE = {"corporate_loans", "retail_loans", "customer_funds", "capital_adequacy", "book_value_per_share",
           "active_clients", "digital_metrics"}
SLOT_FIELDS = ("quote", "value", "base", "change")


def pick(indicator, x):
    """x — ответ с ячейками q_*/y_* → (ячейка, балл) по первой ячейке, где балл считается; (None, None) — нечем."""
    for slot in (("y", "q") if indicator in BALANCE else ("q", "y")):
        c = score(indicator, *(x.get(f"{slot}_{k}", "") for k in ("value", "base", "change")))
        if c is not None:
            return slot, c
    return None, None


def selftest():
    assert multiple("+4,5x") == 350 and multiple("в 1,8 раз") > 79 and multiple("снизились в 2 раза") == -50
    assert multiple("+4,5%") is None and score("provisions", "135,4 млрд руб.", "", "+4,5x") == -2
    assert number("1\u202f508.6 млрд руб.") == 1508.6 and number("26 547 млрд") == 26547
    assert score("net_profit", "511,2 млрд руб.", "423,0 млрд руб.") == 2
    assert score("net_profit", change="+20,9%") == 2
    assert score("net_profit", change="+4,1%") == 1
    assert score("net_profit", change="+2,0%") == 0
    assert score("net_profit", change="снижение на 20,9%") == -2
    assert score("roe", "24,0%", "23,0%") == 1            # +1,0 пп
    assert score("roe", "24,0%", "22,0%") == 2            # +2,0 пп
    assert score("cir", "28,4%", "29,5%") == 1            # снижение CIR — плюс
    assert score("cir", "29,5%", "28,4%") == -1
    assert score("provisions", change="+20%") == -2       # рост резервов — минус
    assert score("provisions", change="-20%") == 2
    assert score("cor", "1,4%", "1,1%") == -1             # +0,3 пп, рост — минус
    assert score("cor", "1,8%", "1,1%") == -2             # +0,7 пп
    assert score("cor", "1,2%", "1,1%") == 0              # +0,1 пп — шум
    assert score("opex", change="+12%", base="+11%") == 0      # разрыв −1 пп
    assert score("opex", change="+12%", base="+18%") == 1      # разрыв +6 пп
    assert score("opex", change="+12%", base="+30%") == 2
    assert score("opex", change="+30%", base="+12%") == -2
    assert score("ceo_tone") is None and score("market_share") is None
    assert score("net_profit", "", "", "") is None          # считать нечем — балл модели, а не ноль
    # реальные случаи ночи 5 (19.09)
    assert score("provisions", "-555,4", "-141,5") == -2     # расходы из таблицы с минусом
    assert score("provisions", "265,0 млрд руб.", "555,4 млрд руб.", "52,3%") == 2  # потерянный минус
    assert score("retail_loans", "29,3%", "", "") == 2       # темп прироста в поле value
    assert score("corporate_loans", "+13,8%", "", "") == 1
    assert score("corporate_loans", "26 547 млрд руб.", "23,3 трлн руб.") == 1  # разные единицы
    assert score("customer_funds", "36 694 млрд руб.", "29 876 млрд руб.", "+23,8%") == 2
    assert score("roe", "24,0%", "23,0%") == 1               # у коэффициентов «%» в value — уровень, не темп
    assert score("opex", "", "", "") is None
    # единицы: «+0,2 млн» — не проценты, считаем из value и base (110,4 → 110,6 = +0,18%)
    assert score("active_clients", "110,6 млн человек", "110,4 млн человек", "+0,2 млн") == 0
    assert score("active_clients", "115,0 млн человек", "110,0 млн человек", "+5,0 млн") == 1
    assert score("nii", "1 066,8 млрд руб.", "841,8 млрд руб.", "+26,7%") == 2
    # ночь 6: темп попал в base, сумма в value — не сравнивать суммы, верить названному темпу
    assert score("retail_loans", "16,1 трлн руб.", "29,3%", "+29,3%") == 2
    # ночь 7: три дырки, найденные на плавающих клетках v7
    assert score("opex", "14,7%", "12,3%", "-2,4 пп") == -1        # разрыв считаем сами, готовый в change игнорируем
    assert score("opex", change="+12%", base="+18%") == 1           # старый формат (рост расходов в change) жив
    assert score("cor", "116 бп", "50 бп") == score("cor", "1,16%", "0,5%") == -2   # базисные пункты
    assert score("cor", "0,9%", "0,93%", "-3 бп") == 0
    assert score("active_clients", "109,5 млн", "", "+0,5 млн") == 0   # база не выписана, изменение в штуках
    assert score("active_clients", "109,5 млн", "", "+5,5 млн") == 1
    assert score("net_profit", "511,2 млрд руб.", "", "+88,2 млрд руб.") == 2
    # v7: выбор ячейки периода кодом
    x = {"q_value": "411,1 млрд руб.", "q_change": "-0,1%", "y_value": "1227,2 млрд руб.", "y_change": "+6,8%"}
    assert pick("net_profit", x) == ("q", 0)               # за период — квартал год к году
    assert pick("corporate_loans", {"q_change": "+2,0%", "y_change": "+13,8%"}) == ("y", 1)  # на дату — с начала года
    assert pick("net_profit", {"y_change": "+6,8%"}) == ("y", 1)  # квартала нет — нарастающий итог
    assert pick("net_profit", {}) == (None, None)
    # 29.09: скрытые ошибки разбора, найденные при аудите (в итоговых ответах не встречались)
    assert score("capital_adequacy", "14,6%", "14,3%") == 1          # ровно 0,3 пп — уже ±1
    assert score("capital_adequacy", "Н20.0: 13,7%", "13,4%") == 1   # «Н20.0» — имя норматива
    assert score("cir", "27,9% (ниже целевого)", "29,0%") == 1       # слово рядом с уровнем знак не меняет
    assert score("capital_adequacy", "13,7%²", "13,3%") == 1         # надстрочная сноска не цифра
    assert score("net_profit", "снижение на 5,0%", "", "") == -1      # темп в поле value: слово знак меняет
    assert number("СберБанк Онлайн 78,6 млн") == 78.6 and number("трлн 20,3") == 20.3  # «н» в конце слова — не норматив
    assert BALANCE <= set(NUMERIC) and len(NUMERIC) + len(JUDGMENT) == 24
    print("score_rules: самопроверка пройдена, показателей в коде", len(NUMERIC), "у модели", len(JUDGMENT))


if __name__ == "__main__":
    selftest()
