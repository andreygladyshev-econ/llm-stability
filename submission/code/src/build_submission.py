"""Сборка файлов сдачи (23.09): векторы (CSV, JSON), таблица оценок с обоснованием и дословной цитатой (MD),
замеры устойчивости (CSV, MD, Excel — src/stability_xlsx.py), оценки всех прогонов (runs.csv).

Источник — боевой режим v7: 5 прогонов на отчёт при T=0 с разным порядком показателей; итог клетки — медиана
голосов (совпадает с модой задания во всех клетках — `src/audit.py`, A3).

Цитата (29.09): фрагмент распознанного текста, найденный по цитате модели, переписывается буквами текстового слоя
PDF (src/pdf_quotes.py), поэтому совпадает с отчётом посимвольно; проверку повторяет `src/audit.py`, A7. Если
оценку посчитала программа, а в цитате модели нет ни базы сравнения, ни названного изменения, в таблицу ставится
строка отчёта с этими числами (для операционных расходов добавляется строка с ростом операционного дохода).
Примечания ручной проверки (notes/примечания_проверки.json) и расхождения с эталоном выводятся после обоснования;
оценок они не меняют. Запуск: .venv/bin/python src/build_submission.py
"""
import collections
import csv
import json
import re
import statistics
import sys
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_table as ft  # noqa: E402
import pdf_quotes as pq  # noqa: E402
import report  # noqa: E402
import score_rules as sr  # noqa: E402
import silver_gold as sg  # noqa: E402
import stability_xlsx  # noqa: E402
import texts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = texts.SUBMIT
PREFIX = "qwen3.8-27b-mlx-4bit__v7_extract__t0-shuf-ex2-tx2"
PDF = json.loads((ROOT / "notes" / "pdf_mapping.json").read_text())
SBER_PDF = set(PDF)  # сканы заказчика: assignment/SBER/{файл}
# отчёты других банков — текстовые PDF с сайтов банков (code/pdf_other/)
PDF.update({r: f"{r}.pdf" for r in ("vtb_2026_q1", "tbank_2024_q4", "tbank_2026_q1", "tbank_2026_q2")})
TEMPLATE = list(csv.DictReader(open(sg.GOLD / "2026_q2_CLAUDE.csv")))
ORDER = [r["id"] for r in TEMPLATE]
NAME = {r["id"]: (r["блок"], r["показатель"]) for r in TEMPLATE}
BANK = {"vtb": "ВТБ", "tbank": "Т-Технологии"}
PDF_TEXT = ROOT / "text_pdf"
NOTES = {k: v for k, v in json.loads((ROOT / "notes" / "примечания_проверки.json").read_text()).items()
         if not k.startswith("_")}
GOLD = {rep: {i: v[0] for i, v in sg.read_gold(rep).items()} for rep in sg.REPORTS}
# слова, по которым строка отчёта относится к показателю (для подтверждающей строки с базой сравнения)
KEYWORDS = {"net_profit": ("чистая прибыль",), "eps": ("на обыкновенную акцию", "на акцию"),
            "roe": ("рентабельность капитала",), "nii": ("процентные доходы",), "fee_income": ("комиссионные доходы",),
            "nim": ("процентная маржа",), "cir": ("расходов к",), "cor": ("стоимость риска",),
            "provisions": ("резерв", "кредитного качества"), "corporate_loans": ("корпоративн",), "retail_loans": ("розничн",),
            "customer_funds": ("средств",), "capital_adequacy": ("достаточност",),
            "book_value_per_share": ("балансовая стоимость",), "active_clients": ("клиент",),
            "digital_metrics": ("mau", "пользовател"), "opex": ("операционные расходы",),
            "opex_income": ("операционн", "доход")}

JUDG = {  # правило спеки v7 для балла модели по суждению
    "guidance": {1: "подтверждение цели на год", 2: "повышение прогноза", -2: "снятие или понижение прогноза",
                 -1: "ослабление прогноза", 0: "прогноза в отчёте нет"},
    "portfolio_quality": {1: "качество портфеля улучшилось (доля проблемных кредитов снизилась)",
                          2: "существенное улучшение качества портфеля",
                          -1: "качество портфеля ухудшилось (доля проблемных кредитов выросла)",
                          -2: "существенное ухудшение качества портфеля", 0: "качество без изменений или не описано"},
    "market_share": {1: "растущих долей рынка больше, чем падающих", -1: "падающих долей рынка больше, чем растущих",
                     0: "изменения долей не названы или поровну"},
    "dividends": {2: "«рекордные» дивиденды", 1: "выплата или готовность платить дивиденды",
                  -2: "отказ от выплаты или перенос", 0: "дивиденды не упоминаются"},
    "tech_development": {1: "измеримый факт внедрения (дата, число пользователей, доля процессов)",
                         0: "нет проверяемого факта — только общие формулировки"},
    "external_conditions": {2: "существенное улучшение внешней среды", 1: "внешняя среда улучшилась",
                            -1: "внешняя среда ухудшилась", -2: "существенное ухудшение внешней среды",
                            0: "внешняя среда не описана или нейтральна"},
    "ceo_tone": {2: "в речи председателя «рекорд» или конкретное обязательство с цифрой",
                 1: "факты роста без оговорок", 0: "в речи есть и плюсы, и оговорки",
                 -1: "оговорки, признание проблем", -2: "антикризисная риторика"},
}


def source_link(r):
    """Ссылка на исходный PDF из папки сдачи."""
    if r["id"] in SBER_PDF:
        return f"[{r['файл']}](../assignment/SBER/{r['файл']})"
    return f"[{r['файл']}](code/pdf_other/{r['файл']}), текстовый PDF с сайта банка"


def sgn(x):
    """Целое со знаком по-русски: +12, −2, 0."""
    return f"{x:+d}".replace("-", "−") if x else "0"


def ru(x, unit=""):
    """+20,9% / −2,4 п.п. — русская запись числа."""
    t = f"{x:+.1f}".replace(".", ",").replace("-", "−")
    return t + ("%" if unit == "%" else f" {unit}" if unit else "")


def period_word(ind, slot):
    bal = ind in sr.BALANCE
    return {("q", False): "квартал к тому же кварталу прошлого года", ("y", False): "с начала года к тому же периоду",
            ("q", True): "за квартал", ("y", True): "с начала года"}[(slot, bal)]


# ─── дословная цитата ─────────────────────────────────────────────────────────────────────────────────────────
def light(s):
    return "".join(" " if c in "*#|  \n\t\r" else ("е" if c == "ё" else c) for c in s.lower())


def collapse(s):
    kept, idx = [], []
    for i, c in enumerate(s):
        if c == " " and kept and kept[-1] == " ":
            continue
        kept.append(c); idx.append(i)
    return "".join(kept), idx


def verbatim(quote, raw):
    """Фрагмент исходного текста, соответствующий цитате модели; разметка распознавания убрана. None — не найден."""
    if not quote.strip():
        return None
    t_l = light(raw)
    if len(t_l) != len(raw):
        return None
    t_c, idx = collapse(t_l)
    q_c, _ = collapse(light(quote).strip())
    a = fuzz.partial_ratio_alignment(q_c, t_c, score_cutoff=report.QUOTE_THRESHOLD)
    if a is None:
        return None
    s, e = idx[a.dest_start], idx[a.dest_end - 1] + 1
    while s > 0 and raw[s - 1].isalnum():          # не резать слово или число посередине
        s -= 1
    while e < len(raw) and raw[e].isalnum():
        e += 1
    frag = re.sub(r"\*\*|#+", "", raw[s:e])
    frag = re.sub(r"\s*\|\s*", " | ", frag)
    frag = re.sub(r"\s+", " ", frag).strip(" |")
    return frag


_pdf_cache = {}


def in_pdf(frag, rep):
    """Фрагмент распознанного текста → тот же фрагмент буквами текстового слоя PDF (без слоя — как есть)."""
    if not frag:
        return frag
    if rep not in _pdf_cache:
        p = PDF_TEXT / f"{rep}.txt"
        _pdf_cache[rep] = p.read_text() if p.exists() else None
    pdf = _pdf_cache[rep]
    if pdf is None:
        print(f"  нет текстового слоя PDF для {rep}: цитата оставлена по распознанному тексту")
        return frag
    q = pq.to_pdf(frag, pdf)
    assert q and pq.verified_in_pdf(q, pdf), (rep, frag)
    return q


NUM = re.compile(r"\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?")


def nums(s):
    """Числа строки: «1 508,6» — одно число, ячейки таблицы «397,4 | 357,2» — два (report.numbers их склеивает)."""
    s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]", "", s or "")
    return {re.sub(r"[ \u00a0\u202f]", "", n).replace(".", ",") for n in NUM.findall(s)}


def covers(need, have):
    """Все нужные числа есть в строке; «555» совпадает с «555,4», если у одного из двух нет дробной части."""
    same = lambda n, m: n == m or ("," not in n and m.split(",")[0] == n) or ("," not in m and n.split(",")[0] == m)
    return all(any(same(n, m) for m in have) for n in need)


def basis(ind, x):
    """Числа, на которых стоит оценка программы: (значение, база, названное изменение) — множества строк."""
    slot, code = sr.pick(ind, x)
    if code is None:
        return None
    f = lambda k: nums(x.get(f"{slot}_{k}", ""))
    return f("value"), f("base"), f("change")


def supported(quote, ind, x):
    """Видно ли по цитате, откуда взялась оценка: есть база сравнения или названное изменение."""
    b = basis(ind, x)
    if b is None or not quote:
        return True
    v, base, ch = b
    qn = nums(quote)
    if ind == "opex":                       # нужен и рост расходов, и рост дохода
        return (not v or covers(v, qn)) and (not base or covers(base, qn))
    return bool(base and covers(base, qn)) or bool(ch and covers(ch, qn))


def support_line(key, need, raw):
    """Самая короткая строка или предложение отчёта со словами показателя и всеми нужными числами."""
    best = None
    for line in raw.split("\n"):
        for part in [line] + re.split(r"(?<=[.!?])\s+", line):
            low = light(part)
            words = all if key == "opex_income" else any
            if need and covers(need, nums(part)) and words(k in low for k in KEYWORDS[key]):
                clean = re.sub(r"\s+", " ", re.sub(r"\s*\|\s*", " | ", re.sub(r"\*\*|#+|^\s*[-*]\s+", "", part))).strip(" |*")
                if clean and (best is None or len(clean) < len(best)):
                    best = clean
    return best


# ─── обоснование ──────────────────────────────────────────────────────────────────────────────────────────────
def phrase_sentence(ind, raw):
    """Предложение отчёта, где рядом с показателем стоит оценочная фраза (как ищет код правила фраз)."""
    for sent in re.split(r"(?<=[.!?])\s+|\n", raw):
        low = light(sent)
        if any(w in low for w in report.STATE_WORDS[ind]) and any(ph in low for ph in report.STATE_PHRASES[ind]):
            return re.sub(r"\s+", " ", re.sub(r"\*\*|#+", "", sent)).strip(" |*")
    return None


def justify(ind, x, final, phrase, raw_nums=None):
    if ind not in sr.NUMERIC:
        return JUDG.get(ind, {}).get(final, "оценка модели по правилу спеки"), "модель"
    if phrase is not None and phrase == final:
        return "оценочная фраза в отчёте рядом с показателем — итоговое состояние важнее направления цифры", "код (фраза)"
    slot, code = sr.pick(ind, x)
    if code is None:
        cells = [x.get(f"{s}_{k}") for s in "qy" for k in ("value", "base", "change")]
        if final == 0 and not any(cells):
            return "не упоминается", "код"
        if final == 0:
            return "названо значение без сравнения с прошлым периодом — ноль по правилу спеки", "код"
        return "код не смог посчитать по выписанным числам — балл модели", "модель (запасной выход)"
    v, b, c = (x.get(f"{slot}_{k}", "") for k in ("value", "base", "change"))
    kind, inv = sr.NUMERIC[ind]
    if kind == "opex":
        if sr.number(b) is None:
            return (f"{period_word(ind, slot)}: расходы {v or c}; операционного дохода до резервов в тексте нет — "
                    "сравнить не с чем, 0 по правилу спеки"), "код"
        gap = sr.number(b) - (sr.number(v) if v else sr.number(c))
        return (f"{period_word(ind, slot)}: рост расходов {v or c}, рост операционного дохода до резервов {b} → разрыв "
                f"{ru(gap, 'п.п.')} (в пределах ±1 — 0; до ±10 — ±1; больше — ±2)"), "код"
    ch = sr.change_of(ind, v, b, c)
    unit = "п.п." if kind in ("ratio", "cor") else "%"
    lo, hi = sr.THRESHOLDS[kind]
    stated = bool(c) and sr.number(c) is not None
    # 29.09: названное изменение приоритетнее; если база из ответа с ним не сходится (другой период или номинал
    # вместо «без учёта валютной переоценки»), базу в обосновании не показываем — она в расчёт не вошла
    by_amounts = sr.change_of(ind, v, b, "") if (stated and b) else None
    tol = 0.15 if kind in ("ratio", "cor") else max(1.5, 0.25 * abs(ch))
    in_text = lambda s: raw_nums is None or covers(nums(s), raw_nums)
    # база, которой нет в тексте отчёта, — пересчёт модели из процента: в обоснование не идёт
    show_base = b and in_text(b) and (by_amounts is None or abs(by_amounts - ch) <= tol)
    if not stated and b and not in_text(b):
        print(f"  ВНИМАНИЕ: оценка {ind} посчитана по базе, которой нет в тексте отчёта: {b}")
    head = f"{period_word(ind, slot)}: {v}" + (f" против {b}" if show_base else "")
    rate_only = not stated and not b and kind == "money" and "%" in str(v)   # темп прироста из отчёта в поле значения
    if rate_only:
        stated = True
    g = lambda x: f"{x:g}".replace(".", ",")
    rule = f"пороги {g(lo)} и {g(hi)}{'%' if unit == '%' else ' п.п.'}" + ("; рост — минус для акционера" if inv else "")
    how = "посчитано кодом"
    if stated:
        how = "по отчёту"
        if sr.multiple(c) is not None:
            how += f": «{c.strip().lstrip('+')}»"
        elif b and in_text(b) and not show_base:
            how += (", без учёта валютной переоценки" if kind == "money" and ind in sr.BALANCE
                    else "; база в ответе модели относится к другому периоду и в расчёт не вошла")
    if rate_only:
        return f"{period_word(ind, slot)}: изменение {ru(ch, unit)} ({how}; {rule})", "код"
    return f"{head}, изменение {ru(ch, unit)} ({how}; {rule})", "код"


# ─── сборка ───────────────────────────────────────────────────────────────────────────────────────────────────
def period_label(rep):
    if "_" not in rep or not rep[:4].isdigit():
        b, y, q = rep.split("_")[0], rep.split("_")[1], rep.split("_")[2]
        return f"{BANK.get(b, b)}, {q[1]} кв. {y}"
    y, q = rep.split("_")
    return {"q1": f"1 кв. {y}", "q2": f"2 кв. {y} (6 мес.)", "q3": f"3 кв. {y} (9 мес.)", "q4": f"4 кв. и 12 мес. {y}"}[q]


def build():
    runs = collections.defaultdict(list)
    raw_text = {}
    for p in sorted((ROOT / "raw").glob(f"{PREFIX}__*.json")):
        d = json.loads(p.read_text()); m = d["meta"]
        rep = m["report"]
        if m["run"] > 5 or rep not in ft.SBER + ft.OTHER:
            continue
        if rep not in raw_text:
            raw_text[rep] = (texts.text_dir("v2") / f"{rep}.txt").read_text()
        parsed = report.apply_code_scores(report.parse(d["content"]), report.norm(raw_text[rep]))
        runs[rep].append((m["run"], parsed))
    # сверка с основным модулем: итог клетки тот же
    _, items = report.load()
    g = items[(items.config == "qwen3.8-27b-mlx@4bit | v7_extract | t0-shuf-ex2-tx2") & items.run.isin(range(1, 6))]
    ref = g.groupby(["report", "indicator"]).score.apply(lambda s: statistics.median_low(s.tolist()))
    t07 = items[(items.config == "qwen3.8-27b-mlx@4bit | v7_extract | t0.7-ex2-mp0.05-tx2") & items.run.isin(range(1, 6))]
    t07v = t07.groupby(["report", "indicator"]).score.apply(list)
    reports = []
    for rep in ft.SBER + ft.OTHER:
        rr = sorted(runs[rep], key=lambda t: t[0])
        assert len(rr) == 5, (rep, len(rr))
        tnorm = report.norm(raw_text[rep])
        rows = []
        for ind in ORDER:
            votes = [p[ind]["score"] for _, p in rr]
            final = statistics.median_low(votes)
            assert final == ref[(rep, ind)], (rep, ind, final, ref[(rep, ind)])
            phrase = report.phrase_score(ind, tnorm) if report.PHRASE_RULE else None
            cands = [p[ind] for _, p in rr if p[ind]["score"] == final]
            # представитель: чаще встречающийся период и подтверждённая цитата
            best, quote = cands[0], None
            for x in sorted(cands, key=lambda x: -sum(y.get("period") == x.get("period") for y in cands)):
                if final and x["quote"].strip():
                    q = verbatim(x["quote"], raw_text[rep])
                    if q and report.check_quote(q, final, tnorm)[0] == "verified":
                        best, quote = x, q
                        break
            why, who = justify(ind, best, final, phrase, nums(raw_text[rep]))
            if who == "код (фраза)":
                quote = phrase_sentence(ind, raw_text[rep]) or quote
            extra = None
            if who == "код" and final and ind in sr.NUMERIC and not supported(quote, ind, best):
                v, b, ch = basis(ind, best)
                if ind == "opex":                          # строка с ростом операционного дохода — вторая цитата
                    extra = support_line("opex_income", b, raw_text[rep]) if b else None
                else:
                    quote = (support_line(ind, v | b, raw_text[rep]) if b else None) or \
                            (support_line(ind, v | ch, raw_text[rep]) if ch else None) or quote
            if final == 0 and quote is None:               # ноль по порогу — цитата с цифрой тоже полезна
                for x in cands:
                    slot, _ = sr.pick(ind, x) if ind in sr.NUMERIC else (None, None)
                    q0 = x.get(f"{slot}_quote") if slot else ""
                    q = verbatim(q0, raw_text[rep]) if q0 else None
                    if q and report.check_quote(q, 1, tnorm)[0] == "verified":
                        quote = q
                        break
            notes = [NOTES[f"{rep}:{ind}"]] if f"{rep}:{ind}" in NOTES else []
            if rep in GOLD and GOLD[rep][ind] != final:
                notes.append(f"Эталонная разметка: {sgn(GOLD[rep][ind])}.")
            row = {"id": ind, "блок": NAME[ind][0], "показатель": NAME[ind][1], "оценка": final, "голоса": votes,
                   "хрупкая": len(set(votes)) > 1, "кто_ставит": who, "обоснование": why,
                   "цитата": in_pdf(quote, rep) or ""}
            if extra:
                row["цитата_доп"] = in_pdf(extra, rep)
            if notes:
                row["примечание"] = " ".join(notes)
            rows.append(row)
        sums = [sum(p[i]["score"] for i in ORDER) for _, p in rr]
        total = sum(r["оценка"] for r in rows)
        s07 = [t07v[(rep, i)] for i in ORDER if (rep, i) in t07v.index and len(t07v[(rep, i)]) == 5]
        reports.append({
            "id": rep, "банк": "Сбер" if rep in ft.SBER else BANK[rep.split("_")[0]], "период": period_label(rep),
            "файл": PDF.get(rep, ""), "сумма": total, "вывод": report.verdict(total),
            "устойчивость": {"совпало_во_всех_5": sum(not r["хрупкая"] for r in rows),
                             "вывод_совпал_во_всех_5": len({report.verdict(s) for s in sums}) == 1,
                             "сумма_по_прогонам": sums, "сумма_min": min(sums), "сумма_max": max(sums),
                             "разошлись": [f"{r['id']} ({' '.join(f'{v:+d}' for v in r['голоса'])})" for r in rows if r["хрупкая"]],
                             "совпало_при_T07": (sum(len(set(s)) == 1 for s in s07) if len(s07) == 24 else None)},
            "показатели": rows})
    return reports


def write(reports):
    OUT.mkdir(exist_ok=True)
    sber = [r for r in reports if r["банк"] == "Сбер"]
    other = [r for r in reports if r["банк"] != "Сбер"]
    # векторы CSV
    for name, rs in (("vectors.csv", sber), ("vectors_other_banks.csv", other)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["report", "period", "source_pdf", *ORDER, "sum", "verdict", "stable_indicators_of_24"])
            for r in rs:
                sc = {x["id"]: x["оценка"] for x in r["показатели"]}
                w.writerow([r["id"], r["период"], r["файл"], *[sc[i] for i in ORDER], r["сумма"], r["вывод"],
                            r["устойчивость"]["совпало_во_всех_5"]])
    # векторы JSON
    meta = {"задание": "15 пресс-релизов Сбера по МСФО, 24 показателя, оценки от −2 до +2",
            "модель": "Qwen 3.8 27B, 4 бита (MLX), локально на MacBook в LM Studio",
            "спецификация": "v7_extract (файл СПЕЦИФИКАЦИЯ.md)",
            "режим": "5 прогонов на отчёт при T=0, в каждом прогоне свой порядок показателей; итоговая оценка клетки "
                     "равна медиане оценок и во всех клетках совпадает с самым частым значением; сумму и вывод "
                     "считает программа",
            "вывод": "strong, если сумма больше 12 (среднее больше 0,5); weak, если сумма меньше −12; иначе mixed",
            "кто_ставит": "код: оценку посчитала программа по выписанным моделью числам; код (фраза): оценку дала "
                          "оценочная фраза отчёта; модель: суждение модели; модель (запасной выход): программа не "
                          "смогла посчитать оценку, взята оценка модели",
            "цитаты": "цитата совпадает с текстовым слоем PDF посимвольно (без учёта пробелов и границ ячеек «|»); "
                      "если оценку посчитала программа, цитата содержит базу сравнения или названное изменение; "
                      "цитата_доп — строка с ростом операционного дохода для операционных расходов",
            "примечание": "примечание ручной проверки или расхождение с эталонной разметкой; оценку не меняет"}
    (OUT / "vectors.json").write_text(json.dumps({"описание": meta, "отчёты": reports}, ensure_ascii=False, indent=1))
    # замеры устойчивости
    with open(OUT / "stability.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["report", "period", "stable_of_24", "stable_share", "verdict_same_in_all_5", "sum_min", "sum_max",
                    "sums_by_run", "diverged", "stable_of_24_at_T0.7"])
        for r in reports:
            s = r["устойчивость"]
            w.writerow([r["id"], r["период"], s["совпало_во_всех_5"], f"{s['совпало_во_всех_5'] / 24:.3f}",
                        s["вывод_совпал_во_всех_5"], s["сумма_min"], s["сумма_max"], " ".join(map(str, s["сумма_по_прогонам"])),
                        "; ".join(s["разошлись"]), s["совпало_при_T07"]])
    esc = lambda s: (s or "—").replace("|", "\\|")
    lines = ["# Таблица оценок: 15 пресс-релизов Сбера и 4 отчёта других банков для проверки переноса", "",
             "Для каждого показателя приведены блок, оценка, обоснование и дословная цитата из отчёта, как требует "
             "спецификация. Колонка «Прогоны» содержит оценки пяти прогонов с разным порядком показателей при T=0; "
             "итоговая оценка равна их медиане и во всех клетках совпадает с самым частым значением, как требует "
             "задание. Жирным выделены хрупкие клетки, в которых оценки прогонов разошлись: их нужно проверить "
             "человеку.", "",
             "Каждая цитата сверена программой с текстовым слоем PDF и совпадает с ним посимвольно (без учёта "
             "пробелов и границ ячеек «|», которыми в таблице разделены столбцы отчёта). Если оценку посчитала "
             "программа, цитата содержит базу сравнения или изменение, по которым она посчитана. «Примечание» — "
             "замечание ручной проверки или расхождение с эталонной разметкой; оценок примечания не меняют.", "",
             "Колонка «Кто ставит» показывает источник оценки: «код» означает, что оценку посчитала программа по "
             "выписанным моделью числам; «код (фраза)» означает, что оценку дала оценочная фраза отчёта по правилу "
             "спецификации; «модель» означает суждение модели по правилу спецификации; «модель (запасной выход)» "
             "означает, что программа не смогла разобрать выписанное число и взята оценка модели.", "",
             "Таблица собирается скриптом [code/src/build_submission.py](code/src/build_submission.py).", ""]
    for r in reports:
        s = r["устойчивость"]
        lines += [f"## {r['период']}: сумма {sgn(r['сумма'])}, {r['вывод']}", "",
                  f"Исходный файл: {source_link(r)}. Во всех 5 прогонах совпали **{s['совпало_во_всех_5']} из 24** показателей; "
                  f"вывод {'одинаков во всех 5 прогонах' if s['вывод_совпал_во_всех_5'] else '**различался**'}; сумма по "
                  f"прогонам от {sgn(s['сумма_min'])} до {sgn(s['сумма_max'])}." + (" Разошлись: " + "; ".join(
                      f"{x['показатель']} ({' '.join(f'{v:+d}' if v else '0' for v in x['голоса'])})"
                      for x in r["показатели"] if x["хрупкая"]) + "." if s["разошлись"] else ""), "",
                  "| Блок | Показатель | Оценка | Обоснование | Цитата из отчёта | Кто ставит | Прогоны |",
                  "|---|---|:-:|---|---|---|---|"]
        for x in r["показатели"]:
            sc = f"{x['оценка']:+d}" if x["оценка"] else "0"
            if x["хрупкая"]:
                sc = f"**{sc}**"
            why = esc(x["обоснование"]) + (f". *Примечание:* {esc(x['примечание'])}" if x.get("примечание") else "")
            quote = "; ".join(f"«{esc(q)}»" for q in (x["цитата"], x.get("цитата_доп")) if q) or "—"
            lines.append(f"| {x['блок']} | {x['показатель']} | {sc} | {why} | {quote} | {x['кто_ставит']} | "
                         f"{' '.join(f'{v:+d}' if v else '0' for v in x['голоса'])} |")
        lines.append("")
    (OUT / "ТАБЛИЦА_ОЦЕНОК.md").write_text("\n".join(lines))
    # все голоса: строки — показатели, столбцы — 5 прогонов (как просит задание для Excel)
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["report", "period", "block", "indicator", "name", "run1", "run2", "run3", "run4", "run5", "final",
                    "fragile", "scored_by"])
        for r in reports:
            for x in r["показатели"]:
                w.writerow([r["id"], r["период"], x["блок"], x["id"], x["показатель"], *x["голоса"], x["оценка"],
                            x["хрупкая"], x["кто_ставит"]])
    write_stability_md(sber, other)
    stability_xlsx.write(OUT / "ЗАМЕРЫ_УСТОЙЧИВОСТИ.xlsx", sber, other, {i: NAME[i][1] for i in ORDER})
    # сводка
    miss = [(r["id"], x["id"]) for r in reports for x in r["показатели"] if x["оценка"] and not x["цитата"]]
    st = sum(r["устойчивость"]["совпало_во_всех_5"] for r in sber)
    print(f"Сбер: совпали во всех 5 — {st}/360 = {st / 360:.1%}; вывод одинаков — "
          f"{sum(r['устойчивость']['вывод_совпал_во_всех_5'] for r in sber)}/15; ненулевых без дословной цитаты: {miss or 0}")
    print("суммы:", {r["id"]: r["сумма"] for r in sber})


def write_stability_md(sber, other):
    """ЗАМЕРЫ_УСТОЙЧИВОСТИ.md — пункт 2 сдачи: по каждому отчёту совпадения, вывод, разброс суммы, разошедшиеся."""
    sg_ = lambda x: f"{x:+d}".replace("-", "−") if x else "0"

    def table(rs):
        out = ["| Отчёт | Совпали во всех 5 | Вывод одинаков | Сумма: итог (min…max) | Суммы по прогонам | "
               "Разошлись: показатель (5 прогонов) | Контроль при T=0,7: совпали |", "|---|:-:|:-:|:-:|---|---|:-:|"]
        for r in rs:
            s = r["устойчивость"]
            by = {x["id"]: x for x in r["показатели"]}
            div = "; ".join(f"{by[i]['показатель']} ({' '.join(map(sg_, by[i]['голоса']))})"
                            for i in ORDER if by[i]["хрупкая"]) or "—"
            out.append(f"| {r['период']} | {s['совпало_во_всех_5']} из 24 ({s['совпало_во_всех_5'] / 24:.0%}) | "
                       f"{'да' if s['вывод_совпал_во_всех_5'] else '**нет**'} | {sg_(r['сумма'])} "
                       f"({sg_(s['сумма_min'])}…{sg_(s['сумма_max'])}) | {' '.join(map(sg_, s['сумма_по_прогонам']))} | {div} | "
                       f"{s['совпало_при_T07'] if s['совпало_при_T07'] is not None else '—'} из 24 |")
        return out

    def total(rs):
        st = sum(r["устойчивость"]["совпало_во_всех_5"] for r in rs)
        t07 = [r["устойчивость"]["совпало_при_T07"] for r in rs if r["устойчивость"]["совпало_при_T07"] is not None]
        wide = [r["период"] for r in rs if r["устойчивость"]["сумма_max"] - r["устойчивость"]["сумма_min"] > 2]
        return (st, 24 * len(rs), sum(r["устойчивость"]["вывод_совпал_во_всех_5"] for r in rs), len(rs), wide,
                sum(t07), 24 * len(t07))

    st, n, vs, nr, wide, s07, n07 = total(sber)
    per_ind = collections.Counter(x["id"] for r in sber for x in r["показатели"] if x["хрупкая"])
    pct = lambda a, b: f"{100 * a / b:.1f}%".replace(".", ",")
    lines = [
        "# Замеры устойчивости: 15 пресс-релизов Сбера", "",
        "Каждый отчёт прогонялся пять раз итоговым методом (спецификация v7, модель Qwen 3.8 27B, 4 бита, локально). "
        "При температуре 0 простой повтор запроса на ноутбуке даёт побайтно тот же ответ и ничего не проверяет, "
        "поэтому в каждом прогоне менялся порядок 24 показателей в запросе, а текст отчёта оставался прежним. "
        "Итоговая оценка клетки равна медиане пяти оценок; во всех 360 клетках она совпадает с самым частым "
        "значением, ничьих нет. Сумму и вывод считает программа. Правая колонка таблицы содержит результат "
        "контрольного прогона того же метода при температуре 0,7 (5 прогонов).", "",
        "## Итог", "",
        f"- Показатели, совпавшие во всех 5 прогонах: **{st} из {n} ({pct(st, n)})**.",
        f"- Отчёты, у которых итоговый вывод одинаков во всех 5 прогонах: **{vs} из {nr}**.",
        f"- Разброс суммы больше 2 пунктов у {len(wide)} отчётов из {nr}" + (": " + "; ".join(wide) if wide else "") + ".",
        f"- Контрольный прогон при T=0,7: {s07} из {n07} ({pct(s07, n07)}).",
        "- Для сравнения на 4 отчётах, использованных при настройке, при одинаковой случайности (T=0,7, 5 прогонов): "
        "исходная спецификация заказчика даёт 61,5%, итоговая спецификация v7 с расчётами в программе 88,5% "
        "(записка, рисунок 1).", "",
        table(sber)[0], table(sber)[1], *table(sber)[2:], "",
        "Хрупкие клетки не скрываются: в таблице оценок они выделены жирным, рядом приведены все пять оценок; такие "
        "клетки нужно проверить человеку. Итоговая оценка в них по правилу задания равна самому частому значению.", "",
        "## Какие показатели расходились (15 отчётов Сбера)", "",
        "| Показатель | В скольких отчётах разошёлся |", "|---|:-:|",
        *[f"| {NAME[i][1]} | {c} |" for i, c in per_ind.most_common()], "",
        "## Четыре отчёта других банков (проверка переноса, в основную сдачу не входят)", "",
    ]
    st2, n2, vs2, nr2, wide2, s072, n072 = total(other)
    lines += [f"Во всех 5 прогонах совпали {st2} из {n2} ({pct(st2, n2)}); вывод одинаков в {vs2} отчётах из {nr2}; "
              f"контрольный прогон при T=0,7: {s072} из {n072} ({pct(s072, n072)}).", "", *table(other), "",
              "Эти же замеры в виде таблицы Excel, которую просит задание (на каждый отчёт лист: 24 показателя × 5 "
              "прогонов, совпадения, суммы и выводы посчитаны формулами), — [ЗАМЕРЫ_УСТОЙЧИВОСТИ.xlsx]"
              "(ЗАМЕРЫ_УСТОЙЧИВОСТИ.xlsx). Оценки всех пяти прогонов по каждой клетке в одном файле — [runs.csv](runs.csv), "
              "итоговые векторы — [vectors.csv](vectors.csv) и [vectors.json](vectors.json), замеры в машинном виде — "
              "[stability.csv](stability.csv). Файлы собирает скрипт "
              "[code/src/build_submission.py](code/src/build_submission.py).", ""]
    (OUT / "ЗАМЕРЫ_УСТОЙЧИВОСТИ.md").write_text("\n".join(lines))


if __name__ == "__main__":
    write(build())
