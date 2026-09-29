"""Серебряный эталон (разметка Claude вслепую по тексту и спеке) против конвейеров — ось «правильность».

Эталон НЕ независим полностью: его ставила тоже языковая модель, к тому же соавтор спеки. Поэтому:
  · цитаты эталона проверяются дословно по тексту (сам себя проверяю тем же верификатором, что модели);
  · спорные клетки помечены и считаются отдельно;
  · Андрей проверяет случайную выборку клеток (`_АНДРЕЙ/4_разметка/АУДИТ_ЭТАЛОНА.md`) — по ней оценивается
    ошибка самого эталона.
Запуск: .venv/bin/python src/silver_gold.py
"""
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstain  # noqa: E402
import report  # noqa: E402
import texts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# эталон: рядом с кодом сдачи (СДАЧА/эталон), в репозитории для рабочего проекта, у автора — в личной папке
GOLD = next(p for p in (ROOT.parent / "эталон", ROOT.parent / "СДАЧА" / "эталон", ROOT.parent / "_АНДРЕЙ" / "4_разметка")
            if p.is_dir())
REPORTS = ["2026_q2", "2022_q4", "2023_q2", "2024_q4", "2025_q3"]  # + холдаут, размечен вслепую 22.09
M = "qwen3.8-27b-mlx@4bit"
CONFIGS = {
    "v1 исходная спека (старый текст)": f"{M} | v1_baseline | t0.7",
    "v2_inplace (старый текст)": f"{M} | v2_inplace | t0.7",
    "v2_inplace (новый текст)": f"{M} | v2_inplace | t0.7-mp0.05-tx2",
    "v5 извлечение": f"{M} | v5_extract | t0.7-ex-mp0.05-tx2",
    "v7 T=0,7": f"{M} | v7_extract | t0.7-ex2-mp0.05-tx2",
    "v7 T=0,3": f"{M} | v7_extract | t0.3-ex2-mp0.05-tx2",
    "v7 T=0": f"{M} | v7_extract | t0-ex2-tx2",
    "Gemma v2_inplace (новый текст)": "google/gemma-4-31b-qat | v2_inplace | t0.7-mp0.05-tx2",
}


def read_gold(rep):
    rows = list(csv.DictReader(open(GOLD / f"{rep}_CLAUDE.csv")))
    return {r["id"]: (int(r["ОЦЕНКА (-2..2)"]), r["ЦИТАТА из отчёта"], r["комментарий"].startswith("СПОРНО"))
            for r in rows}


def self_check():
    bad = []
    for rep in REPORTS:
        text = report.norm((texts.text_dir("v2") / f"{rep}.txt").read_text())
        for i, (s, q, _) in read_gold(rep).items():
            st, _ = report.check_quote(q, s, text)
            if st not in ("verified", "empty"):
                bad.append((rep, i, st))
    return bad


def main():
    bad = self_check()
    print(f"самопроверка цитат эталона: {'все подтверждены' if not bad else bad}\n")
    gold = {(rep, i): v for rep in REPORTS for i, v in read_gold(rep).items()}
    _, items = report.load()
    rows = []
    for name, cfg in CONFIGS.items():
        g = items[(items.config == cfg) & items.report.isin(REPORTS)]
        if g.empty or g.groupby("report").run.nunique().size < len(REPORTS):
            continue
        modal = g.groupby(["report", "indicator"]).score.agg(lambda s: s.mode().iloc[0])
        exact = strict_err = dir_err = n = 0
        exact_clear = n_clear = 0
        for key, (gs, _, disputed) in gold.items():
            ms = int(modal[key]); n += 1
            exact += ms == gs; dir_err += abstain.err_direction(ms, gs)
            if not disputed:
                n_clear += 1; exact_clear += ms == gs
        verdicts = sum(report.verdict(sum(int(modal[(r, i)]) for i in report.IDS)) ==
                       report.verdict(sum(gold[(r, i)][0] for i in report.IDS)) for r in REPORTS)
        rows.append({"конвейер": name, "прогонов": int(g.run.max()),
                     "совпало с эталоном": f"{exact}/{n} = {exact / n:.0%}",
                     "без спорных": f"{exact_clear}/{n_clear} = {exact_clear / n_clear:.0%}",
                     "грубые ошибки (знак или ≥2)": dir_err,
                     "верхняя граница грубых (CP, δ=0,1)": round(abstain.cp_upper(dir_err, n, 0.1), 3),
                     "вывод совпал": f"{verdicts}/{len(REPORTS)}"})
    print(pd.DataFrame(rows).to_string(index=False))
    for rep in REPORTS:
        print(f"эталон {rep}: сумма {sum(gold[(rep, i)][0] for i in report.IDS)}, "
              f"вывод {report.verdict(sum(gold[(rep, i)][0] for i in report.IDS))}")


if __name__ == "__main__":
    main()
