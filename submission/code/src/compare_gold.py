"""Ручная разметка Андрея против прогонов модели → вторая ось: правильность.

Запуск: .venv/bin/python src/compare_gold.py [отчёт]
Читает ручную разметку {отчёт}_ANDREY.csv из рабочей папки автора (_АНДРЕЙ/4_разметка рядом с проектом, в репозиторий
не входит) и metrics/items.csv, печатает и пишет metrics/gold_{отчёт}.csv. Эталон сдачи читает silver_gold.py.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from indicators import IDS, INDICATORS

ROOT = Path(__file__).resolve().parent.parent
ANDREY = ROOT.parent / "_АНДРЕЙ"  # папка Андрея лежит рядом с проектом, не внутри
NAMES = {i[0]: i[2] for i in INDICATORS}


def read_gold(report):
    path = ANDREY / "4_разметка" / f"{report}_ANDREY.csv"
    gold, notes = {}, {}
    with path.open() as f:
        for row in csv.DictReader(f):
            raw = (row.get("ОЦЕНКА (-2..2)") or "").strip().replace("+", "").replace("−", "-")
            if raw:
                gold[row["id"]] = int(raw)
                notes[row["id"]] = (row.get("комментарий") or "").strip()
    missing = [i for i in IDS if i not in gold]
    assert not missing, f"не заполнены: {', '.join(missing)}"
    return gold, notes


def mode(values):
    c = Counter(values).most_common()
    return c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else None


def main():
    report = sys.argv[1] if len(sys.argv) > 1 else "2026_q2"
    gold, notes = read_gold(report)
    items = pd.read_csv(ROOT / "metrics" / "items.csv")
    items = items[items.report == report]
    p = items.config.str.split(" | ", regex=False)
    items["model"], items["spec"], items["v"] = p.str[0], p.str[1], p.str[2]
    rows, detail = [], []
    for (model, spec, v), g in items.groupby(["model", "spec", "v"]):
        vec = {i: (g[(g.indicator == i) & (g.run == 1)].score.iloc[0] if v == "t0"
                   else mode(g[g.indicator == i].score.tolist())) for i in IDS if (g.indicator == i).any()}
        if len(vec) < len(IDS):
            continue
        hits = [i for i in IDS if vec[i] == gold[i]]
        near = [i for i in IDS if abs(vec[i] - gold[i]) == 1]
        signs = [i for i in IDS if vec[i] * gold[i] < 0]
        rows.append({"модель": model.split("/")[-1], "спека": spec, "режим": v,
                     "совпало с эталоном": round(len(hits) / len(IDS), 3),
                     "мимо на 1": len(near), "знак противоположный": len(signs),
                     "сумма модели": sum(vec.values()), "сумма эталона": sum(gold.values()),
                     "расходится": ", ".join(i for i in IDS if vec[i] != gold[i])})
        for i in IDS:
            detail.append({"спека": spec, "режим": v, "показатель": i, "название": NAMES[i],
                           "эталон": gold[i], "модель": vec[i], "разница": vec[i] - gold[i],
                           "комментарий Андрея": notes.get(i, "")})
    df = pd.DataFrame(rows).sort_values("совпало с эталоном", ascending=False)
    pd.set_option("display.width", 250)
    print(f"\nОтчёт {report}. Сумма эталона: {sum(gold.values())}\n")
    print(df.drop(columns=["расходится"]).to_string(index=False))
    print("\nПоказатели, где модель систематически расходится с эталоном (во скольких конфигурациях):")
    d = pd.DataFrame(detail)
    bad = d[d.разница != 0].groupby(["показатель", "название"]).size().sort_values(ascending=False)
    print(bad.to_string())
    d.to_csv(ROOT / "metrics" / f"gold_{report}.csv", index=False)
    df.to_csv(ROOT / "metrics" / f"gold_{report}_summary.csv", index=False)
    print(f"\n→ metrics/gold_{report}.csv, metrics/gold_{report}_summary.csv")


if __name__ == "__main__":
    main()
