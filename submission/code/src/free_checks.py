"""Проверки, которые считаются на уже собранных прогонах — прогоны не нужны (H10, H31, H17, L3).

H10: помогает ли выбросить самые капризные показатели — устойчивее ли становится итоговый вывод.
H31: насколько вывод strong/mixed/weak держится, если менять веса блоков и порог — или он на волоске.
H17: согласие двух семейств моделей (Qwen и Gemma) против согласия прогонов одной модели.
L3:  среднее по прогонам вместо моды — меняется ли вывод.

Запуск: .venv/bin/python src/free_checks.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report  # noqa: E402
from indicators import IDS, INDICATORS  # noqa: E402

V7 = "qwen3.8-27b-mlx@4bit | v7_extract | t0.7-ex2-mp0.05-tx2"
QWEN = "qwen3.8-27b-mlx@4bit | v2_inplace | t0.7-mp0.05-tx2"
GEMMA = "google/gemma-4-31b-qat | v2_inplace | t0.7-mp0.05-tx2"
BLOCK = {i[0]: i[1] for i in INDICATORS}


def cube(items, config):
    """(отчёт, прогон, показатель) → балл."""
    g = items[items.config == config]
    return g.pivot_table(index=["report", "run"], columns="indicator", values="score").reindex(columns=IDS)


def verdicts(c, keep=IDS, weights=None, thr=report.VERDICT_THRESHOLD):
    w = pd.Series(weights or {i: 1.0 for i in keep}).reindex(keep).fillna(1.0)
    mean = (c[keep] * w).sum(axis=1) / w.sum()
    return mean.map(lambda m: "strong" if m > thr else "weak" if m < -thr else "mixed")


def share_stable(v):
    """Доля отчётов, где вывод одинаков во всех прогонах."""
    g = v.groupby(level="report").nunique()
    return float((g == 1).mean())


def h10(c):
    """Выбрасываем самые капризные показатели по одному — следим за устойчивостью вывода."""
    caprice = c.groupby(level="report").nunique().mean().sort_values(ascending=False)
    rows, keep = [], list(IDS)
    for n in range(0, 7):
        if n:
            keep.remove(caprice.index[n - 1])
        rows.append({"выброшено": n, "последний выброшенный": caprice.index[n - 1] if n else "—",
                     "вывод одинаков во всех прогонах": round(share_stable(verdicts(c, keep)), 3)})
    return pd.DataFrame(rows)


def h31(c):
    """Устойчивость вывода к весам блоков и порогу."""
    rng = np.random.default_rng(0)
    base = verdicts(c)
    rows = [{"вариант": "как в спеке (все веса 1, порог 0,5)", "вывод одинаков во всех прогонах": round(share_stable(base), 3),
             "совпало с базой": 1.0}]
    for thr in (0.3, 0.7):
        v = verdicts(c, thr=thr)
        rows.append({"вариант": f"порог {thr}", "вывод одинаков во всех прогонах": round(share_stable(v), 3),
                     "совпало с базой": round(float((v == base).mean()), 3)})
    same = []
    for _ in range(200):  # случайные веса блоков ±30%
        wb = {b: float(rng.uniform(0.7, 1.3)) for b in set(BLOCK.values())}
        v = verdicts(c, weights={i: wb[BLOCK[i]] for i in IDS})
        same.append(float((v == base).mean()))
    rows.append({"вариант": "веса блоков ±30% (200 случайных наборов)",
                 "вывод одинаков во всех прогонах": "—", "совпало с базой": round(float(np.mean(same)), 3)})
    return pd.DataFrame(rows)


def l3(c):
    """Среднее по прогонам вместо моды: меняется ли итоговая таблица и вывод."""
    rows = []
    for rep, g in c.groupby(level="report"):
        mode = g.mode().iloc[0]
        mean = g.mean().round()
        rows.append({"отчёт": rep, "показателей разошлось": int((mode != mean).sum()),
                     "вывод по моде": verdicts(pd.DataFrame([mode])).iloc[0],
                     "вывод по среднему": verdicts(pd.DataFrame([mean])).iloc[0]})
    return pd.DataFrame(rows)


def h17(items):
    """Согласие семейств: Qwen против Gemma — и согласие прогонов внутри каждой модели."""
    q, g = cube(items, QWEN), cube(items, GEMMA)
    reps = sorted(set(q.index.get_level_values("report")) & set(g.index.get_level_values("report")))
    if not reps:
        return "нет общих отчётов"
    rows = []
    for name, c in (("Qwen сам с собой", q), ("Gemma сама с собой", g)):
        acc = [float((c.loc[r].iloc[a] == c.loc[r].iloc[b]).mean())
               for r in reps for a in range(len(c.loc[r])) for b in range(a + 1, len(c.loc[r]))]
        rows.append({"пара": name, "совпадение баллов": round(float(np.mean(acc)), 3)})
    cross = [float((q.loc[r].iloc[a] == g.loc[r].iloc[b]).mean())
             for r in reps for a in range(len(q.loc[r])) for b in range(len(g.loc[r]))]
    rows.append({"пара": "Qwen против Gemma", "совпадение баллов": round(float(np.mean(cross)), 3)})
    return pd.DataFrame(rows)


def main():
    _, items = report.load()
    c = cube(items, V7)
    c = c.loc[[r for r in c.index.get_level_values("report").unique() if not r.startswith(("cf_", "anon_"))]]
    print(f"данные: конвейер v7, отчётов {c.index.get_level_values('report').nunique()}, ответов {len(c)}\n")
    print("H10 — выбрасываем капризные показатели:"); print(h10(c).to_string(index=False))
    print("\nH31 — устойчивость вывода к весам и порогу:"); print(h31(c).to_string(index=False))
    print("\nL3 — среднее вместо моды:"); print(l3(c).to_string(index=False))
    print("\nH17 — согласие семейств (спека v2_inplace, новый текст):"); print(h17(items))


if __name__ == "__main__":
    main()
