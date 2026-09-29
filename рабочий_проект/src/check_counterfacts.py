"""H26: изменилась ли оценка так, как обязана была. Запуск: .venv/bin/python src/check_counterfacts.py

Для каждого контрфакта сравниваем моду баллов на подменённом тексте с модой на исходном отчёте.
Это проверка по пунктам: шум замера ей почти не мешает, в отличие от средних цифр.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CFG = "qwen3.8-27b-mlx@4bit | v2_inplace | t0.7"


def mode(scores):
    c = Counter(scores)
    top = max(c.values())
    best = sorted(v for v, n in c.items() if n == top)
    return best[0] if len(best) == 1 else best[len(best) // 2]  # при ничьей — середина


def main():
    items = pd.read_csv(ROOT / "metrics" / "items.csv")
    facts = json.loads((ROOT / "notes" / "counterfacts.json").read_text())
    rows = []
    for f in facts:
        base = items[(items.config == CFG) & (items.report == f["base"]) & (items.indicator == f["indicator"])]
        cf = items[(items.config == CFG) & (items.report == f["name"]) & (items.indicator == f["indicator"])]
        if base.empty or cf.empty:
            rows.append({**f, "было": None, "стало": None, "прогонов": len(cf), "вердикт": "нет прогонов"})
            continue
        b, c = mode(base.score.tolist()), mode(cf.score.tolist())
        rows.append({"контрфакт": f["name"].replace("cf_", ""), "показатель": f["indicator"],
                     "ожидание": f["expect"], "было": b, "стало": c, "изменилось": "да" if b != c else "НЕТ",
                     "прогонов": cf.run.nunique(), "разброс на контрфакте": sorted(set(cf.score))})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    changed = (df.get("изменилось") == "да").sum() if "изменилось" in df else 0
    print(f"\nоценка изменилась в {changed} из {len(df)} подмен")
    print("Правило работает, если изменение совпало с ожиданием из колонки «ожидание» "
          "(знак и направление), а не просто «что-то поменялось».")
    df.to_csv(ROOT / "metrics" / "counterfacts.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
