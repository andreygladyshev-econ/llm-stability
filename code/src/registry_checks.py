"""Бесплатные проверки реестра спорных мест (22.09) на готовых прогонах v7, боевой режим T=0, 5 перестановок.

П. 2 (H29): абсолютные пороги стоимости риска 0,2 / 0,5 пп подогнаны под банк с CoR ≈ 1%. Относительные:
    изменение больше 20% / 50% от прошлого уровня. Сколько клеток меняют балл и у каких банков.
П. 10: операционные расходы получают 0, когда операционного дохода до резервов в тексте нет. Сколько таких клеток,
    и у скольких доход можно было бы восстановить из расходов и отношения расходов к доходам (CIR).
Мощность машины: время и энергия на один отчёт.
Запуск: .venv/bin/python src/registry_checks.py
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_rules as sr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def rel_cor(v, b):
    """Относительные пороги стоимости риска: |Δ| / прошлый уровень < 20% → 0, до 50% → ±1, больше → ±2."""
    if v is None or not b:
        return None
    r = (v - b) / abs(b)
    mag = 0 if abs(r) < 0.2 else 1 if abs(r) <= 0.5 else 2
    return -mag if r > 0 else mag


def runs(spec="v7_extract", variant="t0-shuf-ex2-tx2"):
    for f in sorted((ROOT / "raw").glob(f"*__{spec}__{variant}__*.json")):
        d = json.loads(f.read_text())
        try:
            yield d["meta"], json.loads(d["content"])
        except (ValueError, TypeError):
            continue


def main():
    cor = collections.defaultdict(list)
    opex = collections.Counter()
    t, e = [], []
    for m, c in runs():
        rep = m["report"]
        t.append(m["sec"] + (m.get("load_sec") or 0))
        if m.get("load_w"):
            e.append(m["load_w"] * m["sec"] / 3600)
        x = c.get("cor") or {}
        slot, s_abs = sr.pick("cor", x)
        if slot:
            v, b = sr.in_pp(x.get(f"{slot}_value")), sr.in_pp(x.get(f"{slot}_base"))
            ch = sr.in_pp(x.get(f"{slot}_change")) if x.get(f"{slot}_change") else None
            if b is None and v is not None and ch is not None:
                b = v - ch
            cor[rep].append((v, b, s_abs, rel_cor(v, b)))
        o = c.get("opex") or {}
        oslot, _ = sr.pick("opex", o)
        if oslot and not o.get(f"{oslot}_base"):
            cir = c.get("cir") or {}
            opex[(rep, bool(cir.get("q_value") or cir.get("y_value")))] += 1
    print("П. 2 — стоимость риска: абсолютные пороги против относительных (уровень, база, балл абс., балл отн.)")
    changed = 0
    for rep, xs in sorted(cor.items()):
        v, b, a, r = xs[0]
        diff = sum(1 for x in xs if x[2] != x[3] and x[3] is not None)
        changed += diff > 0
        print(f"  {rep:14} CoR {v} против {b}: абс {a:+d}, отн {r if r is None else f'{r:+d}'}"
              f"{'   ← МЕНЯЕТСЯ' if diff else ''}")
    print(f"  отчётов, где балл меняется: {changed} из {len(cor)}")
    print("\nП. 10 — операционные расходы без дохода до резервов (прогонов с нулём по правилу «сравнить не с чем»):")
    for (rep, has_cir), n in sorted(opex.items()):
        print(f"  {rep:14} {n} из 5; CIR в отчёте {'есть — доход восстановим' if has_cir else 'нет'}")
    print(f"\nМашина: {len(t)} прогонов, в среднем {sum(t) / len(t) / 60:.1f} мин на отчёт"
          + (f", {sum(e) / len(e):.1f} Вт·ч на отчёт" if e else ""))


if __name__ == "__main__":
    assert rel_cor(4.7, 6.7) == 1 and rel_cor(1.2, 0.3) == -2 and rel_cor(1.0, 0.9) == 0
    main()
