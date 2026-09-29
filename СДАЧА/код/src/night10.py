"""Разбор ночи 21→22.09: N7 (контроль 5 бит), N8 (перестановки при T=0), разложение разброса (2608.16253)."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eocr  # noqa: E402
import report  # noqa: E402
import silver_gold as sg  # noqa: E402

M = "qwen3.8-27b-mlx@4bit | v7_extract |"
SBER = ["2023_q1", "2023_q3", "2024_q1", "2024_q2", "2025_q1", "2025_q2", "2025_q4", "2026_q1"] + eocr.DEV
BANKS = ["vtb_2026_q1", "tbank_2026_q2", "tbank_2026_q1", "tbank_2024_q4"]


def n7(items):
    arms = {"4 бита, контекст 32768": f"{M} t0.7-ex2-mp0.05-tx2", "4 бита, контекст 20480": f"{M} t0.7-ex2-mp0.05-c20480-tx2",
            "5 бит, контекст 20480": "qwen3.8-27b-mlx@5bit | v7_extract | t0.7-ex2-mp0.05-tx2"}
    return pd.DataFrame({n: eocr.metrics(eocr.arm_items(items, c, range(1, 20), 5)) for n, c in arms.items()}).T


def n8(items, gold):
    sh = items[items.config == f"{M} t0-shuf-ex2-tx2"]
    samp = items[items.config == f"{M} t0.7-ex2-mp0.05-tx2"]
    rows, fragile_all = [], {}
    for rep in SBER + BANKS:
        g = sh[sh.report == rep]
        runs = sorted(g.run.unique())
        if len(runs) < 3:
            continue
        piv = g.pivot_table(index="run", columns="indicator", values="score")
        f3 = set(piv.loc[runs[:3]].columns[piv.loc[runs[:3]].nunique() > 1])
        f5 = set(piv.columns[piv.nunique() > 1]) if len(runs) >= 5 else None
        fragile_all[rep] = f3
        s = samp[samp.report == rep]
        fs = set(s.groupby("indicator").score.nunique().loc[lambda v: v > 1].index) if s.run.nunique() >= 3 else None
        row = {"отчёт": rep, "перест.": len(runs), "хрупких (3 перест.)": len(f3),
               "хрупких (5 перест.)": len(f5) if f5 is not None else "—",
               "из них плавает при T=0,7": f"{len(f3 & fs)}/{len(f3)}" if fs is not None and f3 else "—"}
        if rep in sg.REPORTS:
            mode3 = piv.loc[runs[:3]].mode().iloc[0]
            row["мода 3 перест. = эталон"] = f"{sum(int(mode3[i]) == gold[(rep, i)][0] for i in report.IDS)}/24"
        rows.append(row)
    return pd.DataFrame(rows).fillna("")


def decomposition(items):
    """Несмещённое разложение (фиксированный набор порядков, 2608.16253, формулы 10–12):
    U_within — случайность вызова (два вызова при одном порядке расходятся),
    U_order  — вклад порядка показателей, U_total = U_within + U_order."""
    cells = {}
    base = items[items.config == f"{M} t0.7-ex2-mp0.05-tx2"]                    # порядок из спеки
    orders = {"спека": base[base.run <= 2]}
    for k in (1, 2, 3):
        orders[f"o{k}"] = items[items.config == f"{M} t0.7-shuf-ex2-mp0.05-o{k}-tx2"]
    out = []
    for rep in eocr.DEV:
        for ind in report.IDS:
            laws = []
            for g in orders.values():
                v = g[(g.report == rep) & (g.indicator == ind)].score.astype(int).tolist()
                if len(v) >= 2:
                    laws.append(v)
            K = len(laws)
            if K < 2:
                continue
            a_within = np.mean([np.mean([x == y for i, x in enumerate(v) for j, y in enumerate(v) if i != j])
                                for v in laws])
            a_across = np.mean([np.mean([x == y for x in laws[k] for y in laws[l]])
                                for k in range(K) for l in range(K) if k != l])
            u_within = 1 - a_within
            u_order = (K - 1) / K * (a_within - a_across)
            out.append({"отчёт": rep, "показатель": ind, "порядков": K, "U_вызов": u_within, "U_порядок": u_order,
                        "числовой": ind in report.score_rules.NUMERIC})
    return pd.DataFrame(out)


def main():
    _, items = report.load()
    gold = {(r, i): v for r in sg.REPORTS for i, v in sg.read_gold(r).items()}
    print("=== N7: контроль к 5 битам (по 5 прогонов на 4 отчёта)")
    print(n7(items).drop(columns=["прогонов/отч"]).to_string())
    print("\n=== N8: боевой режим T=0, перестановки порядка показателей")
    t = n8(items, gold)
    print(t.to_string(index=False))
    f3 = pd.to_numeric(t["хрупких (3 перест.)"])
    print(f"среднее хрупких на отчёт: Сбер {f3[t.отчёт.isin(SBER)].mean():.1f} из 24, чужие банки {f3[t.отчёт.isin(BANKS)].mean():.1f}")
    d = decomposition(items)
    print(f"\n=== Разложение разброса, T=0,7, 4 отладочных отчёта ({len(d)} клеток, порядков на клетку: {d.порядков.min()}–{d.порядков.max()})")
    for name, g in (("все", d), ("числовые", d[d.числовой]), ("суждения", d[~d.числовой])):
        tot = g.U_вызов.mean() + g.U_порядок.mean()
        print(f"  {name:9}: случайность вызова {g.U_вызов.mean():.4f} | порядок {g.U_порядок.mean():+.4f} | "
              f"всего {tot:.4f} | доля порядка {g.U_порядок.mean() / tot:.0%}")
    top = d.assign(tot=d.U_вызов + d.U_порядок).sort_values("tot", ascending=False).head(8)
    print("  самые неустойчивые клетки:", ", ".join(f"{r[-7:]}:{i}" for r, i in zip(top.отчёт, top.показатель)))
    print(f"  клеток с ненулевым разбросом: {int(((d.U_вызов + d.U_порядок) > 1e-9).sum())} из {len(d)}")


if __name__ == "__main__":
    main()
