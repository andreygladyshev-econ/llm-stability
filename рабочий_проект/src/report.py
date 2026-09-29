"""raw/*.json → metrics/*.csv + reports/latest.html (утренний отчёт).

Модель не считает ничего: суммы, выводы, устойчивость и проверка цитат — здесь.
Запуск: .venv/bin/python src/report.py        (или src/report.py selftest)
"""
import datetime as dt
import itertools
import statistics
import hashlib
import html
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_rules
import texts  # noqa: E402
from indicators import IDS, INDICATORS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# папка Андрея лежит рядом с проектом; если код запущен из папки сдачи, отчёт пишется в код/пересборка
ANDREY = ROOT.parent / "_АНДРЕЙ" if (ROOT.parent / "_АНДРЕЙ").is_dir() else texts.SUBMIT
RAW, TEXT, METRICS, REPORTS, LOGS = (ROOT / d for d in ("raw", "text", "metrics", "reports", "logs"))
NAMES = {i[0]: i[2] for i in INDICATORS}

VERDICT_THRESHOLD = 0.5      # из спецификации: среднее > 0.5 strong, < −0.5 weak
QUOTE_THRESHOLD = 92         # ponytail: порог на глаз из CLAUDE.md, откалибровать на выборке (_АНДРЕЙ/2_план_и_гипотезы/ИСХОДНЫЙ_ПЛАН_14.09.md, раздел 5)
POSITION_BUCKET = 300        # символов: цитаты ближе этого считаются «одним местом» текста


# ---------- разбор ответа ----------

def parse(content):
    """→ {id: {quote, reasoning, score}} или None. Понимает и объект по id, и список items."""
    # 23.09: Opus 5 отвечает списком объектов с «id» в обёртке ```json — формально верный JSON, но не словарь.
    # Берём самый внешний JSON: список, если он начинается раньше объекта.
    i, j = content.find("["), content.find("{")
    try:
        if 0 <= i < (j if j >= 0 else len(content)):
            d = json.loads(content[i:content.rindex("]") + 1])
        else:
            d = json.loads(content[j:content.rindex("}") + 1])
    except ValueError:
        return None
    if isinstance(d, list):
        d = {x.get("id"): x for x in d if isinstance(x, dict)}
    if not isinstance(d, dict):
        return None
    # обёртка бывает любой: «items», «indicators», просто список (Opus 5 меняет форму от запроса к запросу)
    if isinstance(d, dict) and not any(i in d for i in IDS):
        lst = next((v for v in d.values() if isinstance(v, list) and any(isinstance(x, dict) for x in v)), None)
        if lst is not None:
            d = {x.get("id"): x for x in lst if isinstance(x, dict)}
    out = {}
    for i in IDS:
        x = d.get(i)
        if not isinstance(x, dict) or not isinstance(x.get("score"), int) or not -2 <= x["score"] <= 2:
            return None
        out[i] = {"quote": str(x.get("quote") or ""), "reasoning": str(x.get("reasoning") or ""), "score": x["score"],
                  **{k: str(x.get(k) or "") for k in ("period", "value", "base", "change") if k in x},
                  **{k: str(x.get(k) or "") for k in x if k[:2] in ("q_", "y_")}}  # v7: ячейки периода
    return out


# 21.09 (сверка с эталоном): код считал пороги у показателей состояния и терял исключение из спеки —
# «если рядом с показателем есть прямая оценочная фраза, итоговое состояние важнее направления».
# Пример: CIR вырос на 0,7 пп (по порогам −1), но отчёт пишет «оставаясь ниже целевых значений» (по спеке +1).
# Первая версия (фраза в цитате модели) не сработала: модель выносит фразу в цитату в 6 случаях из 21 и только
# добавляет разброс. Поэтому фразу ищет КОД в самом тексте отчёта — детерминированно. Список фраз — из спеки.
# Выключается PHRASE_RULE=0 (для сравнения).
PHRASE_RULE = __import__("os").environ.get("PHRASE_RULE", "1") == "1"
STATE_WORDS = {"cir": ("отношение операционных расходов к", "расходов к доходам"),
               "cor": ("стоимость риска",),
               "capital_adequacy": ("достаточност",)}
STATE_PHRASES = {"cir": {"ниже целев": 1, "выше целев": -1},
                 "cor": {"на низком уровне": 1, "под давлением": -1},
                 "capital_adequacy": {"выше регуляторного минимума": 1, "выше минимум": 1, "комфортн": 1,
                                      "под давлением": -1, "ниже целев": -1, "ниже минимум": -1}}


def phrase_score(indicator, text):
    """±1 по оценочной фразе в предложении о показателе; None — фразы нет или фразы противоречат друг другу."""
    if indicator not in STATE_WORDS:
        return None
    found = set()
    for sent in re.split(r"(?<=[.!?])\s+|\n", text.lower()):
        if any(w in sent for w in STATE_WORDS[indicator]):
            found |= {v for ph, v in STATE_PHRASES[indicator].items() if ph in sent}
    return found.pop() if len(found) == 1 else None


def apply_code_scores(parsed, text=""):
    """H3: у числовых показателей балл считает Python по извлечённым числам; у качественных остаётся модельный."""
    for i, x in parsed.items():
        x["model_score"] = x["score"]
        if "q_value" in x:  # v7: код выбирает ячейку периода; цитата — из выбранной ячейки
            slot, code = score_rules.pick(i, x)
            x["period"] = slot or ""
            x["quote"] = x.get(f"{slot}_quote", "") if slot else (x.get("q_quote") or x.get("y_quote", ""))
        else:
            code = score_rules.score(i, x.get("value", ""), x.get("base", ""), x.get("change", ""))
        x["code_score"] = code
        ph = phrase_score(i, text) if PHRASE_RULE else None
        if ph is not None:
            code = x["code_score"] = ph        # оценочная фраза в отчёте важнее направления цифры (спека)
            x["phrase_override"] = True
        if code is not None:
            x["score"] = code
    return parsed


# ---------- проверка цитат ----------

def norm(s):
    s = s.lower().replace("ё", "е").replace(" ", " ").replace("\f", " ")
    s = re.sub(r"[‐-―−–—-]", "-", s)
    s = re.sub(r"[«»\"“”„'’`]", "", s)
    # 23.09: разметка распознанного текста — жирный «**448,3**», заголовки «#», границы ячеек «|». Модель цитирует
    # строку таблицы без них, и верная цитата помечалась «выдуманной» (8 итоговых клеток v7 — все такие).
    s = re.sub(r"[*#|]", " ", s)
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)          # «1 019,1» → «1019,1»
    s = re.sub(r"(?<=\d)\s*[,.]\s*(?=\d)", ",", s)  # «18 , 6», «18.6» → «18,6»: модель и OCR путают разделитель
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def numbers(s):
    return {n.replace(".", ",") for n in re.findall(r"\d+(?:[.,]\d+)?", norm(s))}


def check_quote(quote, score, text_norm):
    """→ (статус, позиция в тексте или None)."""
    if not quote.strip():
        return ("empty" if score == 0 else "empty_nonzero"), None
    q = norm(quote)
    a = fuzz.partial_ratio_alignment(q, text_norm, score_cutoff=QUOTE_THRESHOLD)
    if a is None:
        return "fabricated", None
    window = text_norm[max(0, a.dest_start - 20):a.dest_end + 20]
    if not numbers(q) <= numbers(window):
        return "number_mismatch", a.dest_start
    return "verified", a.dest_start


# ---------- устойчивость ----------

def verdict(total):
    mean = total / len(IDS)
    return "strong" if mean > VERDICT_THRESHOLD else "weak" if mean < -VERDICT_THRESHOLD else "mixed"


def disagreement(values):
    s = set(values)
    if len(s) == 1:
        return "none"
    if any(v > 0 for v in s) and any(v < 0 for v in s):
        return "sign"        # −1 против +1: смысл перевернулся
    if 0 in s:
        return "zero"        # 0 против ±1: «есть сигнал или нет»
    return "amplitude"       # −1 против −2: сила, знак тот же


def mode(values):
    c = Counter(values).most_common()
    tie = len(c) > 1 and c[0][1] == c[1][1]
    return c[0][0], tie


def alpha(matrix):
    """matrix: прогоны × показатели. Если все оценки одинаковы, α не определён."""
    m = np.array(matrix, dtype=float)
    if len(m) < 2 or len(np.unique(m[~np.isnan(m)])) < 2:
        return math.nan
    return krippendorff.alpha(reliability_data=m, level_of_measurement="ordinal")


def entropy(labels):
    c = Counter(labels)
    n = sum(c.values())
    return -sum(k / n * math.log2(k / n) for k in c.values()) if n else math.nan


# ---------- сбор ----------

def load():
    runs, items = [], []
    report_texts = {}
    for f in sorted(RAW.glob("*.json")):
        rec = json.loads(f.read_text())
        meta = rec["meta"]
        rep, tv = meta["report"], meta.get("text_version", "v1")
        if (rep, tv) not in report_texts:  # цитату сверяем с той версией текста, которую видела модель
            report_texts[rep, tv] = norm((texts.text_dir(tv) / f"{rep}.txt").read_text())
        parsed = parse(rec["content"])
        if parsed and meta.get("mode") in ("extract", "extract2"):
            parsed = apply_code_scores(parsed, report_texts[rep, tv])
        config = f"{meta['model']} | {meta['spec_version']} | {meta['variant']}"
        row = {**meta, "config": config, "parse_ok": parsed is not None,
               "content_sha": hashlib.sha256(rec["content"].encode()).hexdigest()[:16]}
        if parsed:
            row["sum"] = sum(x["score"] for x in parsed.values())
            row["verdict"] = verdict(row["sum"])
            for i, x in parsed.items():
                status, pos = check_quote(x["quote"], x["score"], report_texts[rep, tv])
                reason_nums_ok = numbers(x["reasoning"]) <= numbers(x["quote"]) if x["quote"] else True
                items.append({"config": config, "report": rep, "run": meta["run"], "indicator": i,
                              "score": x["score"], "model_score": x.get("model_score"),
                              "code_score": x.get("code_score"), "quote_status": status, "position": pos,
                              "reasoning_numbers_in_quote": reason_nums_ok,
                              "quote": x["quote"], "reasoning": x["reasoning"]})
        runs.append(row)
    return pd.DataFrame(runs), pd.DataFrame(items)


def pair_agreement(values):
    """Доля совпавших пар прогонов. Шумит вдвое меньше, чем «единогласно» (замер ночи 4)."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    pairs = [(a, b) for i, a in enumerate(vals) for b in vals[i + 1:]]
    return sum(a == b for a, b in pairs) / len(pairs)


def stability(runs, items):
    by_report, by_indicator, final = [], [], []
    for (config, rep), g in items.groupby(["config", "report"]):
        pivot = g.pivot_table(index="run", columns="indicator", values="score", aggfunc="first").reindex(columns=IDS)
        n_runs = len(pivot)
        types = {i: disagreement(pivot[i].dropna().astype(int).tolist()) for i in IDS}
        sums = runs[(runs.config == config) & (runs.report == rep) & runs.parse_ok]["sum"]
        verdicts = runs[(runs.config == config) & (runs.report == rep) & runs.parse_ok]["verdict"]
        modal = {i: mode(pivot[i].dropna().astype(int).tolist()) for i in IDS}
        modal_sum = sum(v for v, _ in modal.values())
        by_report.append({
            "config": config, "report": rep, "runs": n_runs,
            "unanimous_share": round(sum(t == "none" for t in types.values()) / len(IDS), 3),
            "pair_agreement": round(sum(a for i in IDS
                                        if (a := pair_agreement(pivot[i].dropna().astype(int).tolist())) is not None)
                                    / max(1, sum(1 for i in IDS
                                                 if pair_agreement(pivot[i].dropna().astype(int).tolist()) is not None)), 3),
            "alpha": round(alpha(pivot.values), 3),
            "sign_flips": sum(t == "sign" for t in types.values()),
            "zero_flips": sum(t == "zero" for t in types.values()),
            "amplitude_flips": sum(t == "amplitude" for t in types.values()),
            "sum_min": sums.min(), "sum_max": sums.max(), "modal_sum": modal_sum,
            "verdict_modal": verdict(modal_sum), "verdict_stable": verdicts.nunique() == 1,
            "margin_to_threshold": round(abs(abs(modal_sum) - VERDICT_THRESHOLD * len(IDS)), 1),
            "fabricated_share": round((g.quote_status == "fabricated").mean(), 3),
            "number_mismatch_share": round((g.quote_status == "number_mismatch").mean(), 3),
            "not_mentioned_share": round((g.quote_status == "empty").mean(), 3),
            "unstable": ", ".join(i for i, t in types.items() if t != "none"),
        })
        for i in IDS:
            gi = g[g.indicator == i]
            buckets = [p // POSITION_BUCKET for p in gi.position.dropna()]
            by_indicator.append({"config": config, "report": rep, "indicator": i, "type": types[i],
                                 "values": " ".join(map(str, pivot[i].dropna().astype(int))),
                                 "evidence_entropy": round(entropy(buckets), 2),
                                 "hidden_instability": types[i] == "none" and entropy(buckets) > 0})
            final.append({"config": config, "report": rep, "indicator": i,
                          "score": modal[i][0], "tie": modal[i][1]})
    return pd.DataFrame(by_report), pd.DataFrame(by_indicator), pd.DataFrame(final)


def config_summary(runs, items, by_report, by_ind):
    rows = []
    for config, g in by_report.groupby("config"):
        r = runs[runs.config == config]
        it = items[items.config == config]
        pooled = []
        for rep in g.report:
            p = it[it.report == rep].pivot_table(index="run", columns="indicator", values="score",
                                                 aggfunc="first").reindex(columns=IDS)
            pooled.append(p.values)
        max_runs = max(len(p) for p in pooled)
        stacked = np.hstack([np.vstack([p, np.full((max_runs - len(p), len(IDS)), np.nan)]) for p in pooled])
        rows.append({
            "config": config, "reports": len(g), "runs": len(r),
            "unanimous_share": round(g.unanimous_share.mean(), 3),
            "alpha_pooled": round(alpha(stacked), 3),
            "sign_flips": int(g.sign_flips.sum()), "zero_flips": int(g.zero_flips.sum()),
            "amplitude_flips": int(g.amplitude_flips.sum()),
            "verdict_stable_share": round(g.verdict_stable.mean(), 3),
            "verified_share": round((it.quote_status == "verified").mean(), 3),
            "fabricated_share": round((it.quote_status == "fabricated").mean(), 3),
            "number_mismatch_share": round((it.quote_status == "number_mismatch").mean(), 3),
            "parse_fail": int((~r.parse_ok).sum()), "truncated": int(r.truncated.sum()),
            "cache_contaminated": int(r.get("cache_contaminated", pd.Series(dtype=bool)).fillna(False).sum()),
            "sec_median": r.sec.median(), "load_sec_median": r.get("load_sec", pd.Series(dtype=float)).median(),
            "battery_min": r.get("battery_pct", pd.Series(dtype=float)).min(),
            "thermal_warnings": int(r.get("thermal_warning", pd.Series(dtype=bool)).fillna(False).sum()),
            # у облачных прогонов окружения нет — pandas даёт NaN, а он «истинный», поэтому проверяем тип
            "engine": ", ".join(sorted({str(e.get("backends")) for e in r.get("env", []) if isinstance(e, dict)})),
            "prompt_tokens": r.prompt_tokens.median(),
            "completion_tokens": r.completion_tokens.median(), "tok_per_sec": r.tok_per_sec.median(),
            "hidden_instability": int(by_ind[by_ind.config == config].hidden_instability.sum()),
        })
    return pd.DataFrame(rows)


def spec_comparison(items, max_run=3):
    """Версии спеки на одной модели и температуре, поровну прогонов: иначе больше прогонов = больше расхождений."""
    it = items[items.run <= max_run].copy()
    parts = it.config.str.split(" | ", regex=False)
    # «t0.7-rf» → группа «модель | t0.7», версия «v2_all-rf»: режимы запроса сравниваются с версиями спеки в одной таблице
    temp, suffix = parts.str[2].str.split("-", n=1).str[0], parts.str[2].str.split("-", n=1).str[1]
    it["model_variant"] = parts.str[0] + " | " + temp
    it["spec"] = parts.str[1] + suffix.fillna("").map(lambda s: f"-{s}" if s else "")
    rows = []
    for (mv, spec, rep, ind), g in it.groupby(["model_variant", "spec", "report", "indicator"]):
        if g.run.nunique() == max_run:
            rows.append({"model_variant": mv, "spec": spec, "report": rep, "indicator": ind,
                         "type": disagreement(g.score.tolist())})
    d = pd.DataFrame(rows)
    if d.empty:
        return d, d
    per_spec = (d.assign(ok=d.type == "none", sign=d.type == "sign", zero=d.type == "zero")
                .groupby(["model_variant", "spec"])
                .agg(reports=("report", "nunique"), unanimous_share=("ok", "mean"),
                     sign_flips=("sign", "sum"), zero_flips=("zero", "sum"))
                .round(3).reset_index())
    per_ind = (d.assign(unstable=d.type != "none")
               .pivot_table(index=["model_variant", "indicator"], columns="spec", values="unstable", aggfunc="sum")
               .reset_index())
    return per_spec, per_ind


def sanity(runs, items, final):
    rows = []
    # Барьер: при temperature=0 и пустом кэше прогоны обязаны совпасть побайтово.
    zero = runs[runs.temperature == 0]
    for (config, rep), g in zero.groupby(["config", "report"]):
        if len(g) > 1:
            rows.append({"config": config, "check": f"T=0 прогоны {rep} совпали побайтово",
                         "ok": g.content_sha.nunique() == 1,
                         "detail": f"разных ответов: {g.content_sha.nunique()} из {len(g)}; "
                                   f"из кэша токенов: {int(g.get('cache_cached_tokens', pd.Series([0])).fillna(0).sum())}"})
    for config, f in final.groupby("config"):
        sums = f.groupby("report").score.sum()
        if "2022_q4" in sums and len(sums) > 1:
            rows.append({"config": config, "check": "2022_q4 — самая низкая сумма",
                         "ok": sums.idxmin() == "2022_q4", "detail": f"2022_q4={sums['2022_q4']}, min={sums.min()}"})
        if "2026_q2" in sums and len(sums) > 2:
            rank = sums.rank(ascending=False)["2026_q2"]
            rows.append({"config": config, "check": "2026_q2 — в верхней трети",
                         "ok": rank <= math.ceil(len(sums) / 3), "detail": f"место {int(rank)} из {len(sums)}"})
        it = items[items.config == config]
        bad_prov = it[(it.indicator == "provisions") & (it.score > 0) & it.quote.str.contains("вырос|увелич|рост", case=False)]
        rows.append({"config": config, "check": "резервы выросли → не плюс", "ok": bad_prov.empty,
                     "detail": f"подозрительных оценок: {len(bad_prov)}"})
        bad_cir = it[(it.indicator == "cir") & (it.score < 0) & it.quote.str.contains("сниз|ниже", case=False)]
        rows.append({"config": config, "check": "CIR снизился → не минус", "ok": bad_cir.empty,
                     "detail": f"подозрительных оценок: {len(bad_cir)}"})
    return pd.DataFrame(rows)


# ---------- HTML ----------

def table(df, max_rows=200):
    return df.head(max_rows).to_html(index=False, escape=True, na_rep="—", border=0, classes="t")


def section(title, explain, body):
    return f"<h2>{html.escape(title)}</h2><p class='x'>{html.escape(explain)}</p>{body}"


def night_block():
    if not (LOGS / "status.json").exists():
        return "<p>Статус ночи не найден.</p>"
    s = json.loads((LOGS / "status.json").read_text())
    failed = sorted((RAW / "_failed").glob("*.json")) if (RAW / "_failed").exists() else []
    info = {k: s.get(k) for k in ("started", "finished", "total", "already_done", "done", "failed",
                                  "truncated", "parse_failed", "skipped_models", "last_error")}
    ev = "".join(f"<li>{html.escape(e)}</li>" for e in s.get("events", [])[-40:])
    fl = "".join(f"<li>{html.escape(f.read_text())}</li>" for f in failed[:30])
    return (f"<pre>{html.escape(json.dumps(info, ensure_ascii=False, indent=1))}</pre>"
            f"<details><summary>События ночи</summary><ul>{ev}</ul></details>"
            f"<details><summary>Незавершённые задания ({len(failed)})</summary><ul>{fl}</ul></details>")


def build_html(runs, items, by_report, by_ind, final, summary, checks, spec_cmp=None, spec_cmp_ind=None):
    parts = [f"<h1>Утренний отчёт · {dt.datetime.now():%d.%m.%Y %H:%M}</h1>"]
    parts.append(section("1. Как прошла ночь", "Сколько заданий сделано, что сломалось, какие модели пропущены и почему.",
                         night_block()))
    if spec_cmp is not None and not spec_cmp.empty:
        parts.append(section(
            "1б. Версии спецификации: что дала каждая правка",
            "Только первые 3 прогона у каждой версии, чтобы сравнение было честным. unanimous_share — доля пар "
            "«отчёт × показатель», одинаковых во всех 3 прогонах. Ниже — в скольких отчётах показатель разошёлся "
            "при каждой версии спеки (меньше — лучше).",
            table(spec_cmp) + table(spec_cmp_ind, 400)))
    parts.append(section(
        "2. Сводка по конфигурациям",
        "Конфигурация = модель | версия спеки | режим. unanimous_share — доля показателей, одинаковых во всех "
        "прогонах (главная метрика задания). alpha — согласие прогонов с поправкой на случайность, ≥0,8 надёжно, "
        "пусто = все прогоны одинаковы. sign — переворот знака (плохо), zero — ноль против ±1, amplitude — сила при "
        "том же знаке. verified/fabricated — цитата найдена в тексте / не найдена. sec_median — минут×60 на прогон.",
        table(summary)))
    parts.append(section("3. Проверки здравого смысла",
                         "Соотношения, которые обязана дать любая осмысленная система (из задания).", table(checks)))
    parts.append(section("4. Устойчивость по отчётам",
                         "margin_to_threshold — насколько сумма далека от порога strong/mixed/weak: "
                         "чем меньше, тем легче вывод перевернуть. unstable — какие показатели разошлись.",
                         table(by_report)))
    ind = (by_ind.assign(unstable=by_ind.type != "none")
           .groupby(["config", "indicator"])
           .agg(unstable_reports=("unstable", "sum"), reports=("unstable", "size"),
                sign=("type", lambda t: (t == "sign").sum()), zero=("type", lambda t: (t == "zero").sum()),
                amplitude=("type", lambda t: (t == "amplitude").sum()),
                hidden=("hidden_instability", "sum"))
           .reset_index().sort_values(["config", "unstable_reports"], ascending=[True, False]))
    ind["indicator"] = ind.indicator.map(lambda i: f"{i} · {NAMES[i]}")
    parts.append(section("5. Капризные показатели",
                         "В скольких отчётах показатель разошёлся между прогонами и как. hidden — оценка одна и та же, "
                         "но цитаты из разных мест текста: скрытая нестабильность.", table(ind)))
    bad = items[items.quote_status.isin(["fabricated", "number_mismatch", "empty_nonzero"])]
    parts.append(section(
        "6. Подозрительные цитаты",
        "fabricated — фразы нет в тексте отчёта; number_mismatch — фраза есть, но число в ней другое; "
        "empty_nonzero — оценка не ноль, а цитаты нет. Первые 60, полный список в metrics/items.csv.",
        table(bad[["config", "report", "run", "indicator", "score", "quote_status", "quote", "reasoning"]], 60)))
    for config, f in final.groupby("config"):
        wide = f.pivot(index="indicator", columns="report", values="score").reindex(IDS)
        wide.loc["СУММА"] = wide.sum()
        wide.loc["ВЫВОД"] = [verdict(s) for s in wide.loc["СУММА"]]
        wide.index = [f"{i} · {NAMES.get(i, '')}".strip(" ·") for i in wide.index]
        parts.append(section(f"7. Итоговая таблица: {config}",
                             "Мода по прогонам. Сумму и вывод считает Python, не модель.",
                             wide.reset_index().rename(columns={"index": "показатель"}).to_html(index=False, border=0, classes="t")))
    css = ("body{font:14px/1.45 -apple-system,system-ui,sans-serif;margin:24px;color:#1b1b1b;background:#fafaf8}"
           "h1{font-size:22px}h2{font-size:17px;margin-top:32px;border-top:1px solid #ddd;padding-top:16px}"
           ".x{color:#555;max-width:900px}.t{border-collapse:collapse;font-size:12px;margin:8px 0}"
           ".t td,.t th{padding:4px 8px;border-bottom:1px solid #e5e5e5;text-align:left;vertical-align:top;max-width:420px}"
           ".t th{background:#f0f0ec;position:sticky;top:0}pre{background:#f0f0ec;padding:10px;overflow-x:auto}")
    return f"<meta charset='utf-8'><title>Утренний отчёт</title><style>{css}</style>" + "".join(parts)


def results_journal(runs, items):
    """Накопительный журнал: строка на конфигурацию, все ночи сразу. → _АНДРЕЙ/ЖУРНАЛ_РЕЗУЛЬТАТОВ.md"""
    rows = []
    for cfg, g in items.groupby("config"):
        r = runs[runs.config == cfg]
        # честное сравнение: «единогласно» считается по ВСЕМ тройкам прогонов, а не только по первой.
        # Ночь 4: один замер по 3 прогонам гуляет на ±7 п.п., поэтому важны среднее и размах.
        piv = g.pivot_table(index=["report", "indicator"], columns="run", values="score").dropna(axis=1, how="any")
        combos = list(itertools.combinations(piv.columns, 3))[:120] if piv.shape[1] >= 3 else []
        unan = [(piv[list(c)].nunique(axis=1) == 1).mean() for c in combos]
        agree = [statistics.mean([(piv[a] == piv[b]).mean() for a, b in itertools.combinations(c, 2)]) for c in combos]
        cells = [x for _, x in g[g.run <= 3].groupby(["report", "indicator"])]
        types = [disagreement(c.score.tolist()) for c in cells if c.run.nunique() == 3]
        nz = g[g.score != 0]
        sums = g.groupby(["report", "run"]).score.sum()
        rows.append({
            "config": cfg.replace(" | ", " · "),  # «|» сломал бы разметку таблицы
            "прогонов": len(r), "отчётов": g.report.nunique(),
            # если по отчётам разное число разобранных прогонов, метрики несопоставимы — это должно быть видно
            "прогонов/отчёт": (f"{per.min()}–{per.max()}" if (per := g.groupby("report").run.nunique()).nunique() > 1
                               else str(per.min())),
            "единогласно": round(statistics.mean(unan), 3) if unan else None,
            "разброс": f"{min(unan):.3f}–{max(unan):.3f}" if len(unan) > 1 else "",
            "согласие пар": round(statistics.mean(agree), 3) if agree else None,
            # альфа Криппендорфа (порядковая): поправлена на случайность и сравнима при разном числе прогонов
            "альфа": round(alpha(g.pivot_table(index="run", columns=["report", "indicator"], values="score").values), 3),
            "знак": sum(t == "sign" for t in types), "0↔±1": sum(t == "zero" for t in types),
            "сила": sum(t == "amplitude" for t in types),
            "цитата найдена": round((nz.quote_status == "verified").mean(), 3) if len(nz) else None,
            "сумма min..max": f"{int(sums.min())}..{int(sums.max())}" if len(sums) else None,
            "мин/прогон": round(r.sec.median() / 60, 1) if len(r) else None,
            "непригодных": int((~r.parse_ok).sum() + r.truncated.sum()),
            "с кэшем": int(r.get("cache_contaminated", pd.Series(dtype=bool)).fillna(False).sum()),
            "даты": f"{pd.to_datetime(r.started).min():%d.%m}–{pd.to_datetime(r.started).max():%d.%m}" if len(r) else None,
        })
    def md(df, index=False):  # без внешних зависимостей
        cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
        out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for idx, row in df.iterrows():
            vals = ([str(idx).replace(" | ", " · ")] if index else []) + ["" if pd.isna(v) else str(v) for v in row]
            out.append("| " + " | ".join(vals) + " |")
        return "\n".join(out)

    j = pd.DataFrame(rows).sort_values(["единогласно", "прогонов"], ascending=False)
    zero_t = items[items.config.str.contains(r"\| t0(?:-\w+)?$", regex=True) & (items.run == 1)]
    t0 = (zero_t.groupby(["config", "report"]).score.sum().unstack().astype("Int64") if len(zero_t) else pd.DataFrame())
    text = [f"# Журнал результатов\n",
            f"Обновляется автоматически при каждом запуске `src/report.py`. Последнее обновление: "
            f"{dt.datetime.now():%d.%m.%Y %H:%M}. Всего прогонов: {len(runs)}.\n",
            "«единогласно» — доля показателей, одинаковых во всех 3 прогонах, усреднённая по всем тройкам прогонов "
            "(«разброс» — от худшей тройки к лучшей: это шум самого замера); «согласие пар» — доля совпавших пар "
            "прогонов, она шумит вдвое меньше; «альфа» — согласие всех прогонов с поправкой на случайность (порядковая альфа "
            "Криппендорфа: соседние баллы — меньшая ошибка, чем смена знака), сравнима при разном числе прогонов; «знак» — перевороты знака; «цитата найдена» — доля ненулевых оценок с дословной цитатой.\n",
            md(j), "\n\n## Боевой режим T=0: суммы 24 оценок (>12 strong, <−12 weak)\n",
            md(t0, index=True) if len(t0) else "нет прогонов при T=0", "\n"]
    (ANDREY / "ЖУРНАЛ_РЕЗУЛЬТАТОВ.md").write_text("\n".join(text))
    return j


def main():
    runs, items = load()
    if runs.empty:
        sys.exit("raw/ пуст")
    METRICS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    ANDREY.mkdir(exist_ok=True)
    ok_runs = runs[runs.parse_ok]
    if items.empty:
        by_report = by_ind = final = summary = checks = spec_cmp = spec_cmp_ind = pd.DataFrame()
    else:
        by_report, by_ind, final = stability(ok_runs, items)
        summary = config_summary(runs, items, by_report, by_ind)
        checks = sanity(runs, items, final)
        spec_cmp, spec_cmp_ind = spec_comparison(items)
    for name, df in [("runs", runs), ("items", items), ("stability_by_report", by_report),
                     ("stability_by_indicator", by_ind), ("final_table", final),
                     ("summary", summary), ("sanity", checks),
                     ("spec_comparison", spec_cmp), ("spec_comparison_by_indicator", spec_cmp_ind)]:
        df.to_csv(METRICS / f"{name}.csv", index=False)
    if not items.empty:
        results_journal(runs, items)  # накопительный журнал всех конфигураций → _АНДРЕЙ/ЖУРНАЛ_РЕЗУЛЬТАТОВ.md
    page = build_html(runs, items, by_report, by_ind, final, summary, checks, spec_cmp, spec_cmp_ind)
    (REPORTS / "latest.html").write_text(page)
    (ANDREY / "ОТЧЁТ.html").write_text(page)
    (REPORTS / f"report_{dt.datetime.now():%Y-%m-%d_%H%M}.html").write_text(page)
    print(f"прогонов {len(runs)}, разобрано {len(ok_runs)} → reports/latest.html, metrics/*.csv")


def selftest():
    t = norm("| **Чистая прибыль** | **448,3** | **411,1** | **9,0%** |")
    assert check_quote("Чистая прибыль | 448,3 | 411,1 | 9,0%", 1, t)[0] == "verified"
    text = norm("Во 2 квартале Сбер заработал 511,2 млрд руб. чистой прибыли, показав рост на 20,9% год к году. "
                "Рентабельность капитала составила 24 , 2 % за 1 полугодие.")
    assert check_quote("Сбер заработал 511,2 млрд руб. чистой прибыли, показав рост на 20,9% год к году", 2, text)[0] == "verified"
    assert check_quote("Сбер заработал 611,2 млрд руб. чистой прибыли, показав рост на 20,9% год к году", 2, text)[0] == "number_mismatch"
    assert check_quote("Рентабельность капитала составила 24,2%", 1, text)[0] == "verified"
    assert check_quote("Дивиденды выросли в два раза по сравнению с прошлым годом", 2, text)[0] == "fabricated"
    assert check_quote("", 0, text)[0] == "empty" and check_quote("", 1, text)[0] == "empty_nonzero"
    assert [disagreement(v) for v in ([1, 1], [-1, -2], [0, 1], [-1, 1, 0])] == ["none", "amplitude", "zero", "sign"]
    assert verdict(13) == "strong" and verdict(12) == "mixed" and verdict(-13) == "weak"
    assert mode([-1, -1, 0, -1, -1]) == (-1, False) and mode([1, 1, 0, 0])[1] is True
    assert math.isnan(alpha([[1, 1], [1, 1]]))
    assert parse('{"items":[' + ",".join(f'{{"id":"{i}","quote":"","reasoning":"","score":0}}' for i in IDS) + "]}")
    assert parse('{"net_profit":{"score":3}}') is None
    print("selftest ok")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else main()
