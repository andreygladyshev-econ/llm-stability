"""E-OCR, ночь 6: старый текст против нового (и база против извлечения) — одинаковое число прогонов на серию.

Серии (все Qwen 27B, T=0.7, min_p=0.05, 4 отладочных отчёта):
  A  база, старый текст (Tesseract)   A0 — прогоны 1–5 утра 19.09, A — прогоны этой ночи (6+)
  B  база, новый текст (v2)
  C  извлечение, новый текст
  D  извлечение, старый текст (утро 19.09) — четвёртая клетка 2×2
  V7 две ячейки периода, выбор кодом, новый текст (с ночи 19→20.09)
Сравнение A0 с A = мера шума (та же серия, разные прогоны).
Запуск: .venv/bin/python src/eocr.py [k]   k — сколько прогонов брать на отчёт (по умолчанию максимум общий)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report  # noqa: E402

DEV = ["2022_q4", "2026_q2", "2024_q3", "2023_q4"]
JUDGMENT = {"ceo_tone", "external_conditions", "tech_development", "guidance", "dividends", "market_share",
            "portfolio_quality"}
M = "qwen3.8-27b-mlx@4bit"
ARMS = {  # имя: (config, какие номера прогонов)
    "A0 база/старый (утро)": (f"{M} | v2_inplace | t0.7-mp0.05", range(1, 6)),
    "A  база/старый (ночь)": (f"{M} | v2_inplace | t0.7-mp0.05", range(6, 16)),
    "B  база/новый": (f"{M} | v2_inplace | t0.7-mp0.05-tx2", range(1, 11)),
    "C  извлеч./новый": (f"{M} | v5_extract | t0.7-ex-mp0.05-tx2", range(1, 11)),
    "D  извлеч./старый": (f"{M} | v5_extract | t0.7-ex-mp0.05", range(1, 11)),
    "V7 две ячейки/новый": (f"{M} | v7_extract | t0.7-ex2-mp0.05-tx2", range(1, 11)),
}


def arm_items(items, config, runs, k):
    g = items[(items.config == config) & items.report.isin(DEV) & items.run.isin(list(runs))]
    keep = []
    for rep in DEV:  # первые k прогонов каждого отчёта
        rr = sorted(g[g.report == rep].run.unique())[:k]
        keep.append(g[(g.report == rep) & g.run.isin(rr)])
    import pandas as pd
    return pd.concat(keep)


def metrics(g):
    cells = g.groupby(["report", "indicator"]).score.agg(lambda s: s.nunique())
    types = g.groupby(["report", "indicator"]).score.agg(lambda s: report.disagreement(s.astype(int).tolist()))
    judg = cells.index.get_level_values("indicator").isin(JUDGMENT)
    g = g.assign(rn=g.groupby("report").run.rank(method="dense"))
    a = report.alpha(g.pivot_table(index="rn", columns=["report", "indicator"], values="score").values)
    return {"прогонов/отч": int(g.rn.max()),
            "альфа": round(a, 3),
            "плав. числ. (из 68)": int((cells[~judg] > 1).sum()),
            "плав. сужд. (из 28)": int((cells[judg] > 1).sum()),
            "смена знака": int((types == "sign").sum()),
            "цитаты подтв.": round((g.quote_status == "verified").mean(), 3),
            "выдуманы": round((g.quote_status == "fabricated").mean(), 3),
            "цифра не та": round((g.quote_status == "number_mismatch").mean(), 3)}


def main():
    _, items = report.load()
    items = items[items.config.str.startswith(M)]
    avail = {n: arm_items(items, c, r, 99).groupby("report").run.nunique() for n, (c, r) in ARMS.items()}
    k = int(sys.argv[1]) if sys.argv[1:] else int(min(v.min() for v in avail.values() if len(v) == 4))
    print(f"прогонов на отчёт в каждой серии: k={k}  (есть: " +
          ", ".join(f"{n.split()[0]}={int(v.min()) if len(v) == 4 else 0}" for n, v in avail.items()) + ")\n")
    import pandas as pd
    rows = {n: metrics(arm_items(items, c, r, k)) for n, (c, r) in ARMS.items()
            if len(avail[n]) == 4 and avail[n].min() >= k}
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
