#!/usr/bin/env python3
"""L12: оценка правильности без эталона (Dawid–Skene, одномонетная версия).

Идея (arXiv 2608.18294): если есть три и более независимых несовершенных измерения
одной и той же величины, то по их расхождениям можно восстановить и скрытую истину,
и точность каждого измерения — без единой эталонной метки.

Здесь измерения — разные модели (разные семейства = разные механизмы ошибок).
Внутри модели прогоны сворачиваются в моду: несколько прогонов одной модели —
это НЕ независимые измерения.

Модель: у оценщика j с вероятностью a_j балл верный, иначе равномерно по остальным
четырём. Параметров 5+4 против 96 показателей — считается устойчиво, в отличие от
полных матриц ошибок 5x5 на каждого оценщика.

Запуск:
    python src/no_gold.py            # расчёт на metrics/items.csv
    python src/no_gold.py selftest   # проверка метода на синтетике с известной истиной
"""
from __future__ import annotations

import collections
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
ITEMS = ROOT / "metrics" / "items.csv"
OUT = ROOT / "metrics" / "no_gold.csv"
CLASSES = [-2, -1, 0, 1, 2]
K = len(CLASSES)
IDX = {c: i for i, c in enumerate(CLASSES)}

# Оценщики: конфигурации на одной и той же спеке и температуре, разные модели.
# Семейство — чтобы проверить чувствительность к нарушению независимости ошибок.
RATERS = {
    "qwen3.8-27b-mlx@4bit | v2_inplace | t0.7": ("Qwen 27B", "qwen"),
    "t-pro-it-2.1 | v2_inplace | t0.7": ("T-pro", "qwen"),  # дообученная Qwen
    "gigachat3.1-10b-a1.8b | v2_inplace | t0.7": ("GigaChat", "gigachat"),
    "google/gemma-4-31b-qat | v2_inplace | t0.7": ("Gemma", "gemma"),
    "qwen3.8-4b-distill | v2_inplace | t0.7": ("Qwen 4B", "qwen"),
}


def load_votes(path=ITEMS):
    """(rater, report, indicator) -> список баллов по прогонам."""
    votes = collections.defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        cfg = RATERS.get(r["config"])
        if not cfg:
            continue
        try:
            s = int(r["score"])
        except (ValueError, TypeError):
            continue
        if s in IDX:
            votes[(cfg[0], r["report"], r["indicator"])].append(s)
    return votes


def collapse(votes):
    """Прогоны одной модели -> одна оценка (мода) + доля прогонов за неё."""
    out = {}
    for key, ss in votes.items():
        cnt = collections.Counter(ss)
        top = max(cnt.values())
        # при ничьей берём ближайший к нулю: не завышаем уверенность
        best = min((c for c, n in cnt.items() if n == top), key=abs)
        out[key] = (best, top / len(ss), len(ss))
    return out


def build_matrix(collapsed, raters):
    """Матрица «показатель x оценщик» из баллов; только полные строки."""
    items = sorted({(rep, ind) for (r, rep, ind) in collapsed if r in raters})
    rows, kept = [], []
    for rep, ind in items:
        row = [collapsed.get((r, rep, ind)) for r in raters]
        if any(v is None for v in row):
            continue
        rows.append([IDX[v[0]] for v in row])
        kept.append((rep, ind))
    return np.array(rows, dtype=int), kept


def dawid_skene(Y, iters=300, tol=1e-9):
    """EM. Y[i,j] — класс, выставленный оценщиком j показателю i.

    Возвращает (точности оценщиков, апостериорные вероятности классов, априор).
    """
    n, m = Y.shape
    onehot = np.zeros((n, m, K))
    onehot[np.arange(n)[:, None], np.arange(m)[None, :], Y] = 1
    q = onehot.mean(axis=1)  # старт — голосование большинством
    q /= q.sum(axis=1, keepdims=True)
    a = np.full(m, 0.7)
    prior = q.mean(axis=0)
    prev = None
    for _ in range(iters):
        # M-шаг
        a = np.clip((q[:, None, :] * onehot).sum(axis=(0, 2)) / n, 1.0 / K + 1e-6, 1 - 1e-6)
        prior = np.clip(q.mean(axis=0), 1e-9, None)
        prior /= prior.sum()
        # E-шаг в логарифмах
        hit, miss = np.log(a), np.log((1 - a) / (K - 1))
        logq = np.log(prior)[None, :] + (onehot * hit[None, :, None] + (1 - onehot) * miss[None, :, None]).sum(axis=1)
        logq -= logq.max(axis=1, keepdims=True)
        q = np.exp(logq)
        q /= q.sum(axis=1, keepdims=True)
        ll = float(logq.sum())
        if prev is not None and abs(ll - prev) < tol:
            break
        prev = ll
    return a, q, prior


def majority_accuracy(Y):
    """Наивная оценка: «точность» = совпадение с большинством. Смещена вверх."""
    n, m = Y.shape
    acc = np.zeros(m)
    for i in range(n):
        cnt = collections.Counter(Y[i])
        top = max(cnt.values())
        win = [c for c, v in cnt.items() if v == top]
        for j in range(m):
            acc[j] += Y[i, j] in win
    return acc / n


def bootstrap(Y, reps=400, seed=0):
    rng = np.random.default_rng(seed)
    n = Y.shape[0]
    out = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        try:
            a, _, _ = dawid_skene(Y[idx])
            out.append(a)
        except Exception:
            pass
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


def report(raters, label):
    votes = load_votes()
    collapsed = collapse(votes)
    Y, items = build_matrix(collapsed, raters)
    if Y.shape[0] == 0:
        print(f"{label}: нет общих показателей")
        return None
    a, q, prior = dawid_skene(Y)
    naive = majority_accuracy(Y)
    lo, hi = bootstrap(Y)
    print(f"\n### {label}: {Y.shape[0]} показателей, {Y.shape[1]} оценщиков")
    print(f"{'оценщик':<12} {'точность (без эталона)':>24} {'дов. интервал':>18} {'совпадение с большинством':>27}")
    for j, r in enumerate(raters):
        print(f"{r:<12} {a[j]:>24.3f} {f'{lo[j]:.2f}–{hi[j]:.2f}':>18} {naive[j]:>27.3f}")
    conf = q.max(axis=1)
    print(f"\nсредняя уверенность в восстановленной истине: {conf.mean():.3f}")
    for thr in (0.99, 0.95, 0.9, 0.8):
        cov = (conf >= thr).mean()
        print(f"  показателей с уверенностью >= {thr:.2f}: {cov:.0%}")
    return Y, items, a, q, naive, list(raters)


def main():
    res = report([n for n, _ in RATERS.values()], "Все пять оценщиков")
    # чувствительность: убрать модели одного семейства с Qwen 27B (ошибки коррелируют)
    indep = [n for c, (n, fam) in RATERS.items() if fam != "qwen" or c.startswith("qwen3.8-27b")]
    report(indep, "Только разные семейства (Qwen 27B, GigaChat, Gemma)")
    if res:
        Y, items, a, q, naive, raters = res
        with open(OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["report", "indicator", "истина_оценка", "уверенность"] + raters)
            for i, (rep, ind) in enumerate(items):
                w.writerow([rep, ind, CLASSES[int(q[i].argmax())], round(float(q[i].max()), 4)]
                           + [CLASSES[c] for c in Y[i]])
        print(f"\nпо показателям: {OUT.relative_to(ROOT)}")


def selftest():
    """Метод проверяется там, где истина известна: синтетика с заданными точностями."""
    rng = np.random.default_rng(1)
    true_a = np.array([0.90, 0.70, 0.60, 0.85])
    n = 400
    z = rng.integers(0, K, n)
    Y = np.empty((n, len(true_a)), dtype=int)
    for j, aj in enumerate(true_a):
        ok = rng.random(n) < aj
        wrong = (z + rng.integers(1, K, n)) % K
        Y[:, j] = np.where(ok, z, wrong)
    a, q, _ = dawid_skene(Y)
    naive = majority_accuracy(Y)
    err = np.abs(a - true_a).max()
    assert err < 0.06, f"EM не восстановил точности: {a} против {true_a}"
    assert (q.argmax(axis=1) == z).mean() > 0.9, "плохо восстановлена истина"
    assert (naive - true_a).mean() > 0.03, "голосование большинством должно завышать точность"
    print("восстановленные точности:", np.round(a, 3), "истинные:", true_a)
    print("голосование большинством:", np.round(naive, 3), f"— завышает в среднем на {(naive - true_a).mean():.3f}")
    # три оценщика — минимум, при котором задача решаема
    a3, _, _ = dawid_skene(Y[:, :3])
    assert np.abs(a3 - true_a[:3]).max() < 0.08, f"на трёх оценщиках развалилось: {a3}"
    print("на трёх оценщиках:", np.round(a3, 3))
    print("selftest OK")


if __name__ == "__main__":
    selftest() if len(sys.argv) > 1 and sys.argv[1] == "selftest" else main()
