"""Самоаудит: не ошибка ли наш «успех». Запуск: .venv/bin/python src/selfaudit.py

Проверяет не модель, а НАШ стек: метрики, верификатор цитат, парсер, полноту данных,
честность замера детерминизма. Каждая проверка либо ОК, либо печатает, что именно не так.
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report as R
import texts  # noqa: E402
import score_rules  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
fails = []


def check(name, ok, detail=""):
    print(f"{'ОК  ' if ok else 'СБОЙ'} · {name}{' — ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def metrics_on_synthetic():
    """Метрики должны давать известный ответ на выдуманных данных."""
    check("метрика «согласие пар»: все ответы одинаковы → 1.0", R.pair_agreement([1, 1, 1, 1]) == 1.0)
    check("метрика «согласие пар»: все разные → 0.0", R.pair_agreement([1, 2, 0, -1]) == 0.0)
    check("метрика «согласие пар»: 3 из 4 пар совпали", abs(R.pair_agreement([1, 1, 1, 2]) - 0.5) < 1e-9,
          f"= {R.pair_agreement([1, 1, 1, 2]):.3f} (пар 6, совпало 3)")
    check("тип расхождения: знак", R.disagreement([1, -1]) == "sign")
    check("тип расхождения: ноль", R.disagreement([0, 1]) == "zero")
    check("тип расхождения: сила", R.disagreement([1, 2]) == "amplitude")
    check("тип расхождения: нет", R.disagreement([2, 2, 2]) == "none")


def verifier_on_synthetic():
    """Верификатор цитат обязан ловить выдумку и подмену чисел."""
    text = R.norm("Во 2 квартале банк заработал 511,2 млрд руб. чистой прибыли, показав рост на 20,9% год к году.")
    cases = [
        ("дословная цитата → verified", "банк заработал 511,2 млрд руб. чистой прибыли", 2, "verified"),
        ("выдуманная цитата → fabricated", "прибыль утроилась благодаря продаже активов", 2, "fabricated"),
        ("подменённое число → number_mismatch", "банк заработал 611,2 млрд руб. чистой прибыли", 2, "number_mismatch"),
        ("пустая цитата при нуле → empty", "", 0, "empty"),
        ("пустая цитата при ненулевой оценке → empty_nonzero", "", 2, "empty_nonzero"),
    ]
    for name, quote, score, expect in cases:
        status, _ = R.check_quote(quote, score, text)
        check(f"верификатор: {name}", status == expect, f"получили {status}")


def parser_rejects_broken():
    check("парсер: мусор вместо JSON → None", R.parse("это не json") is None)
    check("парсер: не хватает показателей → None", R.parse('{"net_profit": {"score": 1}}') is None)
    good = json.dumps({i: {"quote": "q", "reasoning": "r", "score": 0} for i in R.IDS}, ensure_ascii=False)
    check("парсер: полный ответ → разбирается", R.parse(good) is not None)
    bad = json.dumps({i: {"quote": "q", "reasoning": "r", "score": 7} for i in R.IDS}, ensure_ascii=False)
    check("парсер: балл вне шкалы → None", R.parse(bad) is None)


def texts_match_hashes():
    bad, n = [], 0
    for tv in texts.DIRS:
        d, h = texts.text_dir(tv), texts.hashes(tv)
        n += len(h)
        bad += [f"{k} ({tv})" for k, v in h.items()
                if (d / f"{k}.txt").exists() and hashlib.sha256((d / f"{k}.txt").read_bytes()).hexdigest() != v["sha256"]]
    check("тексты отчётов не менялись после заморозки", not bad, f"разошлись: {bad}" if bad else f"проверено {n}")


def raw_matches_text_hash():
    """Прогон, сделанный на другой версии текста, нельзя сравнивать с остальными."""
    hs = {tv: texts.hashes(tv) for tv in texts.DIRS}
    bad = []
    for f in sorted((ROOT / "raw").glob("*.json")):
        m = json.loads(f.read_text())["meta"]
        cur = hs[m.get("text_version", "v1")].get(m["report"], {}).get("sha256")
        if cur and m.get("text_hash") and m["text_hash"] != cur:
            bad.append(f.name)
    check("прогоны сделаны на текущей версии текстов", not bad, f"устарели: {len(bad)} — {bad[:3]}" if bad else "")


def determinism_is_real(runs):
    """T=0 «совпало побайтово» не должно быть артефактом: это должны быть разные запросы с пустым кэшем."""
    t0 = runs[(runs.temperature == 0) & runs.parse_ok & (runs.variant == "t0")]
    ok, detail = True, []
    for (cfg, rep), g in t0.groupby(["config", "report"]):
        if len(g) < 2:
            continue
        same_answer = g.content_sha.nunique() == 1
        different_calls = g.started.nunique() == len(g)
        empty_cache = (g.cache_cached_tokens.fillna(0) == 0).all()
        if same_answer and not (different_calls and empty_cache):
            ok = False
            detail.append(f"{rep}: разные запросы={different_calls}, кэш пуст={empty_cache}")
    check("детерминизм T=0 — настоящий (разные запросы, кэш пуст)", ok, "; ".join(detail))


def equal_runs_per_report(items):
    """Если у отчёта меньше прогонов, «единогласно» завышается — сравнение станет нечестным."""
    bad = []
    for cfg, g in items.groupby("config"):
        n = g.groupby("report").run.nunique()
        if n.nunique() > 1:
            bad.append(f"{cfg.split(' | ', 1)[1]}: {n.to_dict()}")
    if bad:
        print(f"ВНИМ · у {len(bad)} конфигураций разное число прогонов по отчётам — их метрики несопоставимы "
              f"с остальными (в журнале это видно в колонке «прогонов/отчёт»):")
        for b in bad:
            print(f"       {b}")
    else:
        check("в каждой конфигурации у отчётов одинаковое число прогонов", True)


def code_scores_only_where_expected(items):
    if "code_score" not in items.columns:
        return
    ex = items[items.code_score.notna()]
    if ex.empty:
        return
    wrong = sorted(set(ex.indicator) - set(score_rules.NUMERIC))
    check("Python считает балл только у числовых показателей", not wrong, f"лишние: {wrong}")


def main():
    print("=== метрики на выдуманных данных ===")
    metrics_on_synthetic()
    print("\n=== верификатор цитат ===")
    verifier_on_synthetic()
    print("\n=== парсер ответов ===")
    parser_rejects_broken()
    print("\n=== расчёт баллов кодом ===")
    score_rules.selftest()
    print("\n=== данные проекта ===")
    texts_match_hashes()
    raw_matches_text_hash()
    runs = pd.read_csv(ROOT / "metrics" / "runs.csv")
    items = pd.read_csv(ROOT / "metrics" / "items.csv")
    determinism_is_real(runs)
    equal_runs_per_report(items)
    code_scores_only_where_expected(items)
    print(f"\nИтог: проверок пройдено {'все' if not fails else f'не все — сбои: {fails}'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
