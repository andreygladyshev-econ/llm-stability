#!/usr/bin/env python3
"""L11: порог отказа с гарантией на долю ошибок (математика).

Задача. Модель выдаёт оценку и меру неуверенности s (у нас — расхождение между
прогонами). Принимаем оценку, если s <= lam. Тогда:
    выборочный риск R(lam) = P(оценка неверна | принято)
    покрытие        C(lam) = P(принято)
Нужен порог lam, при котором R(lam) <= alpha с вероятностью не ниже 1-delta по
случайности размеченной выборки, и при этом C(lam) как можно больше.

Правило (Learn-then-Test, Angelopoulos и др., Ann. Appl. Stat. 2025):
берём наибольший lam из кандидатов, у которого ВЕРХНЯЯ доверительная граница
доли ошибок среди принятых не превышает alpha. Кандидатов n штук, поэтому
уровень каждой границы delta/n (поправка Бонферрони) — иначе гарантия не держится,
потому что порог выбран по тем же данным.

Граница — точная биномиальная (Клоппер–Пирсон). Она тем и важна, что на маленьких
выборках Хёфдинг не сертифицирует ничего: при нуле ошибок среди k принятых
    Хёфдингу нужно      k >= ln(1/delta') / (2 alpha^2)
    Клопперу–Пирсону    k >= ln(1/delta') / ln(1/(1-alpha))
Второе меньше примерно в 1/(2 alpha) раз. Отсюда же считается, сколько нужно
разметить руками, чтобы вообще что-то гарантировать.

    python src/abstain.py            # сколько разметки нужно под наши размеры
    python src/abstain.py selftest   # проверка границ и симуляция нарушений
"""
from __future__ import annotations

import math
import random
import sys


def binom_cdf(x, k, p):
    """P(X <= x) для X ~ Binom(k, p)."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if x < k else 1.0
    return sum(math.comb(k, i) * p**i * (1 - p) ** (k - i) for i in range(x + 1))


def cp_upper(x, k, delta):
    """Точная верхняя граница Клоппера–Пирсона на долю ошибок: x ошибок из k."""
    if k == 0:
        return 1.0
    if x >= k:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):  # бисекция: ищем p, при котором P(X<=x|p) = delta
        mid = (lo + hi) / 2
        if binom_cdf(x, k, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def hoeffding_upper(x, k, delta):
    if k == 0:
        return 1.0
    return min(1.0, x / k + math.sqrt(math.log(1 / delta) / (2 * k)))


def min_clean_k(alpha, delta_prime, bound="cp"):
    """Сколько принятых показателей БЕЗ ЕДИНОЙ ошибки нужно, чтобы граница дошла до alpha."""
    if bound == "cp":
        return math.ceil(math.log(1 / delta_prime) / math.log(1 / (1 - alpha)))
    return math.ceil(math.log(1 / delta_prime) / (2 * alpha**2))


# --- Три определения ошибки. Зафиксированы 18.09.2026 ДО разметки. ---
# Записаны в submission/gold_labels/ПРЕДРЕГИСТРАЦИЯ_ПОРОГА.md до расчёта и после него не менялись.

def err_strict(score, gold):
    """Строгое: любое отличие от эталона."""
    return int(score != gold)


def err_direction(score, gold):
    """По направлению (основное): смена знака или расхождение на 2+ балла.

    Ноль знака не имеет: 0 против +1 — не смена знака, это расхождение на 1 балл.
    Согласовано с нашей же классификацией переворотов (знак / ноль / амплитуда).
    """
    if (score > 0 and gold < 0) or (score < 0 and gold > 0):
        return 1
    return int(abs(score - gold) >= 2)


def err_verdict(total, gold_total, rule):
    """По вердикту: неверен итоговый вывод strong / mixed / weak. rule — функция суммы."""
    return int(rule(total) != rule(gold_total))


LOSSES = {"строгое": err_strict, "по направлению": err_direction}


def calibrate(scores, errors, alpha, delta):
    """Порог по размеченной выборке. scores — неуверенность, errors — 1 если оценка неверна.

    Возвращает (порог, покрытие, граница риска) либо (None, 0, 1) — «сертифицировать нечего».
    """
    pairs = sorted(zip(scores, errors))
    n = len(pairs)
    dp = delta / max(n, 1)
    best = (None, 0.0, 1.0)
    for k in range(1, n + 1):
        lam = pairs[k - 1][0]
        if k < n and pairs[k][0] == lam:
            continue  # порог должен захватывать все равные значения
        x = sum(e for _, e in pairs[:k])
        ucb = cp_upper(x, k, dp)
        if ucb <= alpha:
            best = (lam, k / n, ucb)
    return best


def plan():
    print("Сколько показателей нужно разметить руками, чтобы порог вообще можно было")
    print("сертифицировать. Условие: среди принятых нет ни одной ошибки (лучший случай).")
    print("delta = 0.1 — вероятность, что гарантия не сработает.\n")
    head = f"{'разметка':<22}" + "".join(f"{f'alpha={a}':>24}" for a in (0.05, 0.10, 0.20))
    print(head)
    print(f"{'':<22}" + "".join(f"{'Клоппер-П.':>12}{'Хёфдинг':>12}" for _ in range(3)))
    for n, note in ((24, "1 отчёт"), (48, "2 отчёта"), (96, "4 отчёта"), (240, "10 отчётов"), (360, "15 отчётов")):
        dp = 0.1 / n
        cells = ""
        for a in (0.05, 0.10, 0.20):
            cp = min_clean_k(a, dp, "cp")
            ho = min_clean_k(a, dp, "hoeffding")
            cells += f"{cp if cp <= n else '—':>12}{ho if ho <= n else '—':>12}"
        print(f"{note} ({n} показ.)".ljust(22) + cells)
    print("\n«—» значит: при таком объёме разметки гарантия недостижима, даже если модель")
    print("не ошиблась ни разу. Числа — минимальный размер принятой части без ошибок.")
    print("\nЧитать так: на 96 размеченных показателях (4 отчёта) Клоппер–Пирсон позволяет")
    print("обещать «ошибок не больше 10%», если модель уверенно и верно закрывает хотя бы")
    print("66 из них. Хёфдингу на тех же данных не хватит никогда.")

    print("\n\nЦена того, что определений ошибки три, а не одно.")
    print("Чтобы гарантия держалась сразу для всех трёх, delta делится на 3 (поправка Бонферрони).")
    print(f"\n{'разметка':<22}{'alpha=0.1, одно':>18}{'alpha=0.1, три':>16}{'разница':>10}")
    for n, note in ((48, "2 отчёта"), (96, "4 отчёта"), (240, "10 отчётов")):
        one = min_clean_k(0.10, 0.1 / n, "cp")
        three = min_clean_k(0.10, 0.1 / 3 / n, "cp")
        fmt = lambda v: str(v) if v <= n else "—"
        print(f"{note} ({n} показ.)".ljust(22) + f"{fmt(one):>18}{fmt(three):>16}{three - one:>10}")
    print("\nПлата небольшая — около 10 показателей: требование растёт как логарифм.")


def selftest():
    # определения ошибки
    assert err_strict(2, 1) == 1 and err_strict(1, 1) == 0
    assert err_direction(2, 1) == 0, "соседние баллы одного знака — не ошибка по направлению"
    assert err_direction(1, -1) == 1, "смена знака — ошибка"
    assert err_direction(0, 1) == 0, "ноль против +1 — расхождение на 1 балл, не смена знака"
    assert err_direction(0, 2) == 1, "расхождение на 2 балла — ошибка"
    assert err_direction(-2, 2) == 1
    assert err_direction(-1, 0) == 0 and err_direction(-2, 0) == 1
    # строгое определение обязано быть не мягче, чем по направлению
    for s in range(-2, 3):
        for g in range(-2, 3):
            assert err_strict(s, g) >= err_direction(s, g), (s, g)

    # точная граница: известные значения Клоппера–Пирсона
    assert abs(cp_upper(0, 10, 0.05) - 0.2589) < 1e-3, cp_upper(0, 10, 0.05)
    assert abs(cp_upper(2, 20, 0.05) - 0.2834) < 1e-3, cp_upper(2, 20, 0.05)
    assert cp_upper(0, 100, 0.05) < cp_upper(0, 10, 0.05), "больше данных — граница уже"
    assert cp_upper(3, 30, 0.05) > 0.1, "граница обязана быть выше наблюдённой доли 0.1"
    # закрытая формула согласована с самой границей
    for a in (0.05, 0.1, 0.2):
        k = min_clean_k(a, 0.001, "cp")
        assert cp_upper(0, k, 0.001) <= a and cp_upper(0, k - 1, 0.001) > a, f"формула неточна при alpha={a}"
        assert min_clean_k(a, 0.001, "cp") < min_clean_k(a, 0.001, "hoeffding")

    # симуляция: сертифицированное правило нарушает бюджет не чаще delta,
    # а «подобрал порог по выборке» (naive) — гораздо чаще. Воспроизводим 2606.15153.
    rng = random.Random(7)
    alpha, delta, n_cal, n_te = 0.2, 0.1, 200, 4000

    def draw(m):
        s, e = [], []
        for _ in range(m):
            bad = rng.random() < 0.3
            s.append(rng.gauss(1.5 if bad else 0.0, 1.0))  # неуверенность выше у неверных
            e.append(int(bad))
        return s, e

    def naive(scores, errors, a):
        pairs = sorted(zip(scores, errors))
        best = (None, 0.0)
        for k in range(1, len(pairs) + 1):
            if sum(e for _, e in pairs[:k]) / k <= a:
                best = (pairs[k - 1][0], k / len(pairs))
        return best

    viol_cp = viol_naive = 0
    trials = 60
    for _ in range(trials):
        sc, er = draw(n_cal)
        ts, te = draw(n_te)
        for rule, lam in (("cp", calibrate(sc, er, alpha, delta)[0]), ("naive", naive(sc, er, alpha)[0])):
            if lam is None:
                continue
            acc = [e for s, e in zip(ts, te) if s <= lam]
            if acc and sum(acc) / len(acc) > alpha:
                if rule == "cp":
                    viol_cp += 1
                else:
                    viol_naive += 1
    print(f"нарушений бюджета из {trials}: сертифицированное правило {viol_cp}, подбор по выборке {viol_naive}")
    assert viol_cp <= delta * trials + 2, "сертификат обязан нарушаться не чаще delta"
    assert viol_naive > viol_cp, "подбор порога по выборке обязан нарушать чаще"
    print("selftest OK")


if __name__ == "__main__":
    selftest() if len(sys.argv) > 1 and sys.argv[1] == "selftest" else plan()
