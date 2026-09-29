"""Итоговая таблица задания: 15 отчётов Сбера × 24 показателя, боевой режим (T=0, пять перестановок порядка).

Итог клетки — медиана пяти голосов (для шкалы −2..+2 при нечётном числе голосов определена всегда; на 384 клетках
v7 ни одной ничьей, медиана везде совпала с модой). Клетка «хрупкая», если голоса разошлись, — её в итоговом
файле видно по звёздочке и её надо отдать человеку.
По отчёту: сумма, вывод strong/mixed/weak, разброс суммы по перестановкам, и главное — **устойчив ли вывод**:
если каждую хрупкую клетку сдвинуть в худшую и в лучшую из наблюдённых сторон, меняется ли вывод.
Столбец T=0,7 — доля клеток, не сменивших балл ни разу за 5 сэмплированных прогонов (где такие прогоны есть).

Запуск: .venv/bin/python src/final_table.py v7_extract      (или v10_extract)
Пишет reports/final_<spec>_cells.csv (длинная), _matrix.csv (15×24, «*» — хрупкая), _reports.csv (по отчётам).
"""
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
M = "qwen3.8-27b-mlx@4bit"
SBER = ["2022_q4", "2023_q1", "2023_q2", "2023_q3", "2023_q4", "2024_q1", "2024_q2", "2024_q3", "2024_q4",
        "2025_q1", "2025_q2", "2025_q3", "2025_q4", "2026_q1", "2026_q2"]
OTHER = ["vtb_2026_q1", "tbank_2024_q4", "tbank_2026_q1", "tbank_2026_q2"]
RUNS = range(1, 6)


def cells(items, spec, variant="t0-shuf-ex2-tx2", runs=RUNS):
    g = items[(items.config == f"{M} | {spec} | {variant}") & items.run.isin(runs)]
    votes = g.groupby(["report", "indicator"]).score.apply(lambda s: [int(x) for x in s])
    out = pd.DataFrame({"votes": votes})
    out["n"] = out.votes.map(len)
    out["score"] = out.votes.map(lambda v: int(statistics.median_low(v)))
    out["fragile"] = out.votes.map(lambda v: len(set(v)) > 1)
    out["agree"] = out.votes.map(lambda v: max(v.count(x) for x in set(v)) / len(v))
    return out.reset_index(), g


def per_report(c, g):
    rows = []
    for rep, x in c.groupby("report"):
        total = int(x.score.sum())
        lo = total + int(sum(min(v) - s for v, s in zip(x.votes, x.score)))
        hi = total + int(sum(max(v) - s for v, s in zip(x.votes, x.score)))
        run_sums = g[g.report == rep].groupby("run").score.sum()
        verdicts = run_sums.map(report.verdict)
        rows.append({"отчёт": rep, "перестановок": int(x.n.min()), "сумма": total, "вывод": report.verdict(total),
                     "хрупких клеток": int(x.fragile.sum()),
                     "сумма по перестановкам": f"{int(run_sums.min())}..{int(run_sums.max())}",
                     "вывод по перестановкам совпал": f"{int((verdicts == report.verdict(total)).sum())}/{len(verdicts)}",
                     "худший..лучший случай": f"{lo}..{hi}",
                     "вывод устойчив к хрупким": report.verdict(lo) == report.verdict(hi) == report.verdict(total)})
    return pd.DataFrame(rows)


def sampling_col(items, spec):
    c, _ = cells(items, spec, "t0.7-ex2-mp0.05-tx2")
    c = c[c.n >= 5]
    return (1 - c.groupby("report").fragile.mean()).round(3).rename("устойчивых клеток при T=0,7")


def main(spec):
    _, items = report.load()
    c, g = cells(items, spec)
    c = c[c.report.isin(SBER + OTHER)]
    r = per_report(c, g).set_index("отчёт").join(sampling_col(items, spec)).reset_index()
    r["банк"] = r["отчёт"].map(lambda x: "Сбер" if x in SBER else x.split("_")[0])
    missing = [x for x in SBER if x not in set(c.report)]
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    c.assign(votes=c.votes.map(lambda v: " ".join(map(str, v)))).to_csv(out / f"final_{spec}_cells.csv", index=False)
    mat = c.assign(v=[f"{s:+d}{'*' if f else ''}" for s, f in zip(c.score, c.fragile)]).pivot(
        index="report", columns="indicator", values="v").reindex(columns=report.IDS)
    mat.to_csv(out / f"final_{spec}_matrix.csv")
    r.to_csv(out / f"final_{spec}_reports.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.to_string(index=False))
    s = r[r["банк"] == "Сбер"]
    print(f"\n{spec}: отчётов Сбера {len(s)} из 15{' — нет: ' + ', '.join(missing) if missing else ''}; "
          f"хрупких клеток {c[c.report.isin(SBER)].fragile.mean():.1%}; "
          f"вывод устойчив к хрупким в {int(s['вывод устойчив к хрупким'].sum())} из {len(s)}")


def selftest():
    df = pd.DataFrame({"config": [f"{M} | vX | t0-shuf-ex2-tx2"] * 5, "report": ["r"] * 5, "indicator": ["roe"] * 5,
                       "run": [1, 2, 3, 4, 5], "score": [1, 1, 2, 1, 2]})
    c, _ = cells(df, "vX")
    assert c.score[0] == 1 and c.fragile[0] and c.agree[0] == 0.6
    df.loc[:, "score"] = [-1, -1, 2, 2, 0]  # ничья 2:2 → медиана 0, а не наименьшее
    assert cells(df, "vX")[0].score[0] == 0


if __name__ == "__main__":
    selftest()
    main(sys.argv[1] if len(sys.argv) > 1 else "v7_extract")
