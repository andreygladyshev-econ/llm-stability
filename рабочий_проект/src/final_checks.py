"""Проверка выводов записки на достаточность данных (23.09, после финальной ночи и облака).

Каждый блок — один тезис записки и честная проверка: хватает ли данных, какой интервал, против какой нулевой модели.
  A. Облако «недетерминировано» — меняются ли БАЛЛЫ, а не только текст ответа.
  B. Хрупкость — свойство клетки: пересечение хрупких клеток при разных возмущениях против случайного (гипергеом.).
  C. Сколько перестановок нужно: рост числа хрупких клеток с 3 → 5 → 10.
  D. Сравнение моделей: хрупкие на отчёт с бутстрап-интервалом по отчётам, эталон с интервалом Клоппера–Пирсона.
  E. Та же Qwen 27B: bf16 против fp8 в облаке против 4 бит на ноутбуке.
  F. Порог отказа по предрегистрации (18.09) — теперь на 5 размеченных отчётах (120 клеток).
  G. Кластеризация: насколько клетки одного отчёта зависимы (эффективный объём выборки).
Запуск: .venv/bin/python src/final_checks.py [A B ...]  — без аргументов все блоки.
"""
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstain  # noqa: E402
import report  # noqa: E402
import silver_gold as sg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
M = "qwen3.8-27b-mlx@4bit"
DEV = ["2026_q2", "2022_q4", "2024_q3", "2023_q4"]
BATTLE = ["2026_q2", "2022_q4", "2024_q4", "2025_q3", "vtb_2026_q1", "tbank_2026_q2"]
GOLD4 = ["2026_q2", "2022_q4", "2024_q4", "2025_q3"]
CLOUD = ["deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3", "z-ai/glm-5.3-flash", "xiaomi/mimo-v2.6-flash",
         "google/gemma-4-31b-it", "qwen/qwen3.8-27b", "qwen/qwen3-235b-a22b-2507", "qwen/qwen3.6-35b-a3b",
         "nvidia/nemotron-3-super-120b-a12b", "openai/gpt-oss-120b", "nvidia/nemotron-3-nano-30b-a3b",
         "openai/gpt-5.6-sol", "openai/gpt-6-sol", "openai/gpt-6-luna"]
random.seed(1)


def cfg(model, variant, spec="v7_extract"):
    return f"{model} | {spec} | {variant}"


def votes(items, config, reports, runs=None):
    g = items[(items.config == config) & items.report.isin(reports)]
    if runs is not None:
        g = g[g.run.isin(runs)]
    return g.groupby(["report", "indicator"]).score.apply(list)


def fragile_set(v):
    return {k for k, s in v.items() if len(set(s)) > 1}


def hypergeom_sf(k, N, K, n):
    """P(X ≥ k): из N клеток K «хрупких по A», берём n «хрупких по B» случайно — сколько совпадёт."""
    return sum(math.comb(K, i) * math.comb(N - K, n - i) for i in range(k, min(K, n) + 1)) / math.comb(N, n)


def mcnemar_exact(b, c):
    """Точный двусторонний тест Макнемара: b клеток стали верными, c — неверными."""
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(max(b, c), n + 1)) / 2 ** n) if n else 1.0


def boot_mean(xs, b=4000):
    """90% бутстреп-интервал среднего (уровень как в предрегистрации). Зерно фиксировано — числа воспроизводятся."""
    rng = random.Random(0)
    ms = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(b))
    return ms[int(0.05 * b)], ms[int(0.95 * b)]


def cp_interval(x, n, conf=0.90):
    a = (1 - conf) / 2
    lo = 0.0 if x == 0 else _cp_bound(x, n, a, lower=True)
    hi = 1.0 if x == n else abstain.cp_upper(x, n, a)
    return lo, hi


def _cp_bound(x, n, a, lower):
    lo, hi = 0.0, x / n
    for _ in range(60):   # нижняя граница: P(X ≥ x | p) = a
        mid = (lo + hi) / 2
        if 1 - abstain.binom_cdf(x - 1, n, mid) < a:
            lo = mid
        else:
            hi = mid
    return lo


# ────────────────────────────────────────────────────────────────────────────────────────────────────
def block_a(items):
    print("\nA. ОБЛАКО: одинаковый запрос 5 раз при T=0 — меняются ли баллы, а не только текст")
    print(f"{'модель':36}{'ответов':>8}{'разных текстов':>16}{'клеток с разным баллом':>24}")
    for m in CLOUD + ["anthropic/claude-opus-5", "anthropic/claude-opus-5.5", "x-ai/grok-4.7"]:
        v = votes(items, cfg(m, "t0-ex2-det-tx2"), ["2026_q2"])
        if v.empty:
            continue
        n = min(len(s) for s in v)
        raw = [json.loads(p.read_text())["content"] for p in (ROOT / "raw").glob(
            f"{m.replace('/', '-')}__v7_extract__t0-ex2-det-tx2__2026_q2__run*.json")]
        diff = sum(len(set(s)) > 1 for s in v)
        print(f"{m:36}{n:>8}{len(set(raw)):>16}{diff:>18} из 24")
    v = votes(items, cfg(M, "t0-shuf-ex2-tx2"), ["2026_q2"])
    print(f"{'для сравнения — ноутбук, повтор':36}{'2':>8}{'1':>16}{'0':>18} из 24  (байт в байт, 21.09 и 22.09)")


def block_b(items):
    print("\nB. ХРУПКОСТЬ — СВОЙСТВО КЛЕТКИ: пересечение хрупких клеток при разных возмущениях (4 отладочных отчёта, 96 клеток)")
    anon = votes(items, cfg(M, "t0-shuf-ex2-tx2"), ["anon_" + r for r in DEV])
    anon.index = anon.index.set_levels([lvl.str.replace("anon_", "") for lvl in anon.index.levels[:1]] +
                                       list(anon.index.levels[1:]))
    sets = {
        "T=0,7 (10 прогонов)": fragile_set(votes(items, cfg(M, "t0.7-ex2-mp0.05-tx2"), DEV, range(1, 11))),
        "T=0,3 (9 прогонов)": fragile_set(votes(items, cfg(M, "t0.3-ex2-mp0.05-tx2"), DEV, range(1, 10))),
        "порядок, T=0 (5 перест.)": fragile_set(votes(items, cfg(M, "t0-shuf-ex2-tx2"), DEV, range(1, 6))),
        "обезличенный текст, T=0 (5)": fragile_set(anon),
    }
    N = 96
    for name, s in sets.items():
        print(f"  {name:30} хрупких {len(s):>2}")
    ref_name = "T=0,7 (10 прогонов)"
    ref = sets[ref_name]
    print(f"\n  совпадение с «{ref_name}» ({len(ref)} клеток) против случайного выбора:")
    for name, s in sets.items():
        if name == ref_name or not s:
            continue
        k = len(s & ref)
        exp = len(s) * len(ref) / N
        print(f"  {name:30} {k:>2} из {len(s):>2} (случайно ожидалось {exp:4.1f}); P(≥{k}) = {hypergeom_sf(k, N, len(ref), len(s)):.1e}")
    union = set().union(*sets.values())
    print(f"\n  хрупкая хотя бы при одном возмущении: {len(union)} из {N}; ни при одном: {N - len(union)} ({(N - len(union)) / N:.0%})")
    return sets


def block_c(items):
    print("\nC. СКОЛЬКО ПЕРЕСТАНОВОК НУЖНО (боевой режим, T=0, 4 отладочных отчёта)")
    for k in (3, 5, 7, 10):
        v = votes(items, cfg(M, "t0-shuf-ex2-tx2"), DEV, range(1, k + 1))
        n = min(len(s) for s in v)
        fr = Counter(r for (r, _), s in v.items() if len(set(s)) > 1)
        med = {key: statistics.median_low(s) for key, s in v.items()}
        print(f"  {k:>2} перестановок (есть {n}): хрупких {sum(fr.values()):>2} = {sum(fr.values()) / 4:.2f} на отчёт  {dict(fr)}")
        if k == 5:
            med5 = med
        if k == 10:
            changed = sum(med[key] != med5[key] for key in med)
            print(f"     итог клетки (медиана) при 10 против 5 перестановок изменился в {changed} клетках из 96")


def block_d(items):
    print("\nD. СРАВНЕНИЕ МОДЕЛЕЙ (боевой режим: 6 отчётов × 5 перестановок; эталон — 4 отчёта, 96 клеток)")
    gold = {(r, i): v for r in GOLD4 for i, v in sg.read_gold(r).items()}
    rows = [("НАША Qwen 27B, ноутбук", cfg(M, "t0-shuf-ex2-tx2"), range(1, 6))]
    rows += [(m, cfg(m, "t0-shuf-ex2-or-tx2"), None) for m in CLOUD]
    print(f"{'модель':36}{'хрупких/отчёт':>14}{'90% интервал':>16}{'эталон':>8}{'90% интервал':>16}")
    res = {}
    for name, c, runs in rows:
        v = votes(items, c, BATTLE, runs)
        if v.empty:
            continue
        per = [sum(len(set(s)) > 1 for (r, _), s in v.items() if r == rep) for rep in BATTLE]
        lo, hi = boot_mean(per)
        med = {k: statistics.median_low(s) for k, s in v.items()}
        keys = [k for k in gold if k in med]
        x = sum(med[k] == gold[k][0] for k in keys)
        glo, ghi = cp_interval(x, len(keys))
        res[name] = (per, x, len(keys))
        print(f"{name:36}{statistics.mean(per):>14.2f}{f'[{lo:.1f}; {hi:.1f}]':>16}{f'{x}/{len(keys)}':>8}"
              f"{f'[{glo:.0%}; {ghi:.0%}]':>16}")
    base = res["НАША Qwen 27B, ноутбук"][0]
    print("\n  парно с ноутбуком (разница хрупких на отчёт, облако − ноутбук, 90% бутстрап по 6 отчётам):")
    for m in ("deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3", "openai/gpt-5.6-sol", "qwen/qwen3.8-27b"):
        if m in res:
            d = [a - b for a, b in zip(res[m][0], base)]
            lo, hi = boot_mean(d)
            print(f"  {m:34} {statistics.mean(d):+.2f}  [{lo:+.2f}; {hi:+.2f}]  {'различие есть' if hi < 0 or lo > 0 else 'в пределах шума'}")


def block_e(items):
    print("\nE. ТА ЖЕ QWEN 27B: ноутбук 4 бита / облако fp8 / облако bf16 (частично: 21 ответ)")
    bf = [json.loads(p.read_text()) for p in (ROOT / "raw_qwen_bf16").glob("*.json")]
    det = [report.parse(d["content"]) for d in bf if "-det-" in d["meta"]["variant"]]
    det = [d for d in det if d]
    if det:
        sc = [[report.apply_code_scores(dict(d)) if False else None] for d in det]
    from collections import defaultdict
    by = defaultdict(list)
    for d in bf:
        if "-det-" in d["meta"]["variant"]:
            by["det"].append(d["content"])
    print(f"  bf16, повтор одного запроса ×{len(by['det'])}: разных текстов {len(set(by['det']))}")
    fp8 = [json.loads(p.read_text())["content"] for p in (ROOT / "raw").glob("qwen-qwen3.8-27b__v7_extract__t0-ex2-det-tx2__*.json")]
    print(f"  fp8,  повтор одного запроса ×{len(fp8)}: разных текстов {len(set(fp8))}")
    print("  ноутбук 4 бита: повтор байт в байт (21.09 → 22.09)")


def block_f(items):
    print("\nF. ПОРОГ ОТКАЗА ПО ПРЕДРЕГИСТРАЦИИ 18.09 — 5 размеченных отчётов (120 клеток), 10 прогонов при T=0,7")
    gold = {(r, i): v for r in sg.REPORTS for i, v in sg.read_gold(r).items()}
    v = votes(items, cfg(M, "t0.7-ex2-mp0.05-tx2"), sg.REPORTS, range(1, 11))
    keys = [k for k in gold if k in v.index and len(v[k]) >= 10]
    print(f"  клеток с 10 прогонами: {len(keys)} из {len(gold)}")
    unc, err_s, err_d, modal = [], [], [], {}
    for k in keys:
        c = Counter(v[k]); top, cnt = c.most_common(1)[0]
        modal[k] = top
        unc.append(1 - cnt / len(v[k]))
        err_s.append(int(abstain.err_strict(top, gold[k][0])))
        err_d.append(int(abstain.err_direction(top, gold[k][0])))
    print(f"  ошибок без отказа: строгих {sum(err_s)} из {len(keys)} ({sum(err_s) / len(keys):.1%}), "
          f"по направлению {sum(err_d)} ({sum(err_d) / len(keys):.1%})")
    delta = 0.10 / 3   # три определения ошибки — Бонферрони, как в предрегистрации
    for name, errs in (("по направлению (основное)", err_d), ("строгое", err_s)):
        for alpha in (0.05, 0.10, 0.20):
            lam, cov, ucb = abstain.calibrate(unc, errs, alpha, delta)
            verdict = ("—" if lam is None else f"порог неуверенности ≤ {lam:.2f}, отвечаем на {cov:.0%}, граница риска {ucb:.3f}")
            print(f"  {name:28} alpha={alpha:.2f}: {verdict}")
    # проверка предпосылки: связана ли неуверенность с ошибкой
    hi = [e for u, e in zip(unc, err_s) if u > 0]; lo = [e for u, e in zip(unc, err_s) if u == 0]
    print(f"  предпосылка: строгих ошибок среди клеток без разброса {sum(lo)}/{len(lo)} = {sum(lo) / max(len(lo), 1):.1%}, "
          f"среди клеток с разбросом {sum(hi)}/{len(hi)} = {sum(hi) / max(len(hi), 1):.1%}")
    # вердикты — только описательно (предрегистрация: сертифицировать на 5 отчётах нельзя)
    ok = 0
    for r in sg.REPORTS:
        tot = sum(modal.get((r, i), 0) for i in report.IDS)
        gtot = sum(gold[(r, i)][0] for i in report.IDS)
        ok += report.verdict(tot) == report.verdict(gtot)
    print(f"  вывод strong/mixed/weak по моде совпал с эталоном в {ok} из {len(sg.REPORTS)} отчётов (описательно, без гарантии)")
    return unc, err_s, keys


def block_g(items):
    print("\nG. НЕЗАВИСИМЫ ЛИ КЛЕТКИ: ошибки внутри одного отчёта (боевой режим v7, 5 размеченных отчётов)")
    gold = {(r, i): v for r in sg.REPORTS for i, v in sg.read_gold(r).items()}
    v = votes(items, cfg(M, "t0-shuf-ex2-tx2"), sg.REPORTS, range(1, 6))
    per = {r: [int(statistics.median_low(v[(r, i)]) != gold[(r, i)][0]) for i in report.IDS if (r, i) in v.index]
           for r in sg.REPORTS}
    allx = [x for xs in per.values() for x in xs]
    p = statistics.mean(allx); m = 24; k = len(per)
    msb = m * sum((statistics.mean(xs) - p) ** 2 for xs in per.values()) / (k - 1)
    msw = sum(sum((x - statistics.mean(xs)) ** 2 for x in xs) for xs in per.values()) / (k * (m - 1))
    icc = max(0.0, (msb - msw) / (msb + (m - 1) * msw)) if msb + (m - 1) * msw else 0.0
    deff = 1 + (m - 1) * icc
    print(f"  ошибок по отчётам: {[sum(xs) for xs in per.values()]} из 24; внутриклассовая корреляция ICC = {icc:.3f}")
    print(f"  эффект дизайна = {deff:.2f} → эффективный объём {len(allx) / deff:.0f} клеток вместо {len(allx)}")


if __name__ == "__main__":
    assert abs(hypergeom_sf(0, 10, 3, 3) - 1) < 1e-12 and hypergeom_sf(3, 10, 3, 3) < 0.01
    lo, hi = cp_interval(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    want = set(sys.argv[1:]) or set("ABCDEFG")
    _, items = report.load()
    for b in "ABCDEFG":
        if b in want:
            globals()[f"block_{b.lower()}"](items)
