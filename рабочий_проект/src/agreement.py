#!/usr/bin/env python3
"""Согласие с поправкой на случайность (альфа Криппендорфа).

Зачем: наша метрика «единогласно» — это доля точных совпадений. Она не вычитает
согласие, которое возникло бы само собой. По arXiv 2606.19544 (21 модель, 541 тыс.
оценок) точное совпадение завышает согласие на 34–41 п.п. по сравнению с
поправленным на случайность. Нужно знать, насколько завышает у нас.

Порядковая версия (ordinal) учитывает, что перепутать +2 и +1 — не то же самое,
что перепутать +2 и -2. Для шкалы -2..+2 это правильный вариант; номинальную
считаем рядом для сравнения.

    python src/agreement.py            # по всем конфигурациям
    python src/agreement.py selftest
"""
from __future__ import annotations

import collections
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ITEMS = ROOT / "metrics" / "items.csv"


def alpha(units, ordinal=True):
    """units — список списков оценок одной единицы разными прогонами.

    Возвращает альфу Криппендорфа: 1 — полное согласие, 0 — как при случайном
    выставлении, ниже нуля — систематическое расхождение.
    """
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    vals = sorted({v for u in units for v in u})
    o = collections.Counter()
    for u in units:
        m = len(u)
        for i, a in enumerate(u):
            for j, b in enumerate(u):
                if i != j:
                    o[(a, b)] += 1 / (m - 1)
    n_c = {c: sum(o[(c, k)] for k in vals) for c in vals}
    n = sum(n_c.values())
    if n < 2:
        return float("nan")

    if ordinal:
        # порядковая метрика различия: цена ошибки растёт с расстоянием по шкале
        order = {v: i for i, v in enumerate(vals)}

        def d2(c, k):
            lo, hi = sorted((order[c], order[k]))
            s = sum(n_c[vals[g]] for g in range(lo, hi + 1))
            return (s - (n_c[c] + n_c[k]) / 2) ** 2
    else:
        def d2(c, k):
            return 0.0 if c == k else 1.0

    do = sum(o[(c, k)] * d2(c, k) for c in vals for k in vals) / n
    de = sum(n_c[c] * n_c[k] * d2(c, k) for c in vals for k in vals) / (n * (n - 1))
    return 1.0 if de == 0 else 1 - do / de


def unanimity(units):
    """Наша нынешняя метрика: доля единиц, где все прогоны совпали."""
    units = [u for u in units if len(u) >= 2]
    return sum(len(set(u)) == 1 for u in units) / len(units) if units else float("nan")


def main():
    by_cfg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(ITEMS, encoding="utf-8")):
        try:
            by_cfg[r["config"]][(r["report"], r["indicator"])].append(int(r["score"]))
        except (ValueError, TypeError):
            continue
    rows = []
    for cfg, units in by_cfg.items():
        u = list(units.values())
        if len(u) < 24 or max(len(x) for x in u) < 3:
            continue
        rows.append((cfg, unanimity(u), alpha(u, ordinal=True), alpha(u, ordinal=False), len(u)))
    rows.sort(key=lambda x: -x[2])
    print(f"{'конфигурация':<46}{'единогласно':>12}{'альфа (порядк.)':>17}{'альфа (номин.)':>16}{'завышение':>11}")
    for cfg, un, ao, an, k in rows:
        print(f"{cfg:<46}{un:>12.3f}{ao:>17.3f}{an:>16.3f}{un - an:>11.3f}")
    print("\n«завышение» — на сколько наша метрика больше согласия с поправкой на случайность (номинального).")


def selftest():
    # Канонический пример Криппендорфа (три наблюдателя, пропуски).
    # Ожидаемое значение посчитано вручную по матрице совпадений:
    # 10 единиц с 2+ оценками, n=28, несовпадающих пар 7 -> Do=7/28=0.25;
    # суммы по баллам 5,10,8,3,2 -> De=(28^2-202)/(28*27)=0.7698; альфа=1-0.25/0.7698.
    A = [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None]
    B = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3]
    C = [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None]
    units = [[x for x in col if x is not None] for col in zip(A, B, C)]
    a = alpha(units, ordinal=False)
    assert abs(a - 0.6753) < 0.001, f"номинальная альфа {a:.4f}, ожидалось 0.6753"
    assert abs(alpha([[1, 1, 1], [2, 2, 2], [0, 0, 0]], ordinal=False) - 1.0) < 1e-9, "полное согласие должно давать 1"
    # порядковая метрика мягче к соседним баллам, чем номинальная
    near = [[1, 2], [2, 1], [0, 1], [1, 0], [2, 2], [0, 0]]
    assert alpha(near, ordinal=True) > alpha(near, ordinal=False)
    # систематическое расхождение уходит ниже нуля
    assert alpha([[0, 2], [2, 0], [0, 2], [2, 0]], ordinal=False) < 0
    print(f"канонический пример: {a:.4f} (ожидалось 0.6753) — selftest OK")


if __name__ == "__main__":
    selftest() if len(sys.argv) > 1 and sys.argv[1] == "selftest" else main()
