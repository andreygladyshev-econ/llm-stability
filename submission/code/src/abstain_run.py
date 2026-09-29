"""Порог отказа на реальных прогонах: мера неуверенности по ОДНОМУ прогону → сертифицированный порог.

Смысл процедуры и результат — ПРИЛОЖЕНИЯ.md, раздел Е; заранее записанные правила — gold_labels/ПРЕДРЕГИСТРАЦИЯ_ПОРОГА.md.
Здесь — только вычисление.

Что считается (гарантия без ручной разметки):
  ошибка = «повтор того же прогона дал бы другой балл». Формально для клетки z = (отчёт, показатель)
  прогоны — независимые одинаково распределённые исходы (это свойство протокола: модель перезагружается,
  кэш пуст, история пуста). Риск принятого ответа R(lam) = P(Y¹ ≠ Y² | S¹ ≤ lam), где S — мера неуверенности,
  посчитанная по тому же первому прогону.

Разделение данных (иначе порог подгоняется):
  · 4 отладочных отчёта × 10 прогонов — «обучение» меры S (веса логистической регрессии);
  · 8 нетронутых кварталов × 3 прогона — калибровка порога, эти данные при обучении не видны.

Две границы риска, и разница между ними — честный ответ на вопрос «сколько данных нужно»:
  · по клеткам (Клоппер–Пирсон): узкая, но требует независимости клеток внутри отчёта — это допущение;
  · по отчётам (эмпирический Бернштейн, Маурер–Понтил): допущений почти нет, но нужно много отчётов.
Обе печатаются. Поправка на перебор порогов — последовательная проверка (fixed sequence), FWER контролируется порядком.

Запуск: .venv/bin/python src/abstain_run.py [--alpha 0.1] [--delta 0.1]
        .venv/bin/python src/abstain_run.py selftest
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstain  # noqa: E402  — cp_upper, binom_cdf
import report  # noqa: E402
import score_rules  # noqa: E402

CONFIG = "qwen3.8-27b-mlx@4bit | v7_extract | t0.7-ex2-mp0.05-tx2"
FIT = ["2022_q4", "2023_q4", "2024_q3", "2026_q2"]                      # отладочные: обучение меры S
CAL = ["2023_q1", "2023_q3", "2024_q1", "2024_q2", "2025_q1", "2025_q2", "2025_q4", "2026_q1"]  # калибровка
FEATURES = ["judgment", "no_code", "disagree", "fallback", "boundary", "quote_bad"]


# ---------- мера неуверенности по одному прогону ----------

def boundary_closeness(indicator, x):
    """Насколько посчитанное изменение близко к порогу значимости: 1 — ровно на границе, 0 — далеко."""
    kind = score_rules.NUMERIC.get(indicator, (None,))[0]
    if kind not in score_rules.THRESHOLDS:
        return 0.0
    slot = x.get("period")
    c = score_rules.change_of(indicator, *(x.get(f"{slot}_{k}", "") for k in ("value", "base", "change"))) \
        if slot in ("q", "y") else score_rules.change_of(indicator, x.get("value", ""), x.get("base", ""),
                                                         x.get("change", ""))
    if c is None:
        return 0.0
    rel = min(abs(abs(c) - t) / t for t in score_rules.THRESHOLDS[kind])
    return float(max(0.0, 1.0 - min(rel, 1.0)))


def features(row):
    """Признаки одного ответа по одному показателю. Ни один не подсматривает в другие прогоны."""
    num = row["indicator"] in score_rules.NUMERIC
    code, model = row.get("code_score"), row.get("model_score")
    both = code is not None and model is not None and not pd.isna(code) and not pd.isna(model)
    return {
        "judgment": 0.0 if num else 1.0,                                    # балл ставила модель, проверить нечем
        "no_code": 1.0 if num and (code is None or pd.isna(code)) else 0.0,  # числа не разобрались
        "disagree": float(min(abs(model - code), 4)) / 4 if both else 0.0,   # модель и код разошлись
        "fallback": 1.0 if num and row.get("period") == ("q" if row["indicator"] in score_rules.BALANCE
                                                         else "y") else 0.0,  # взята не первая ячейка периода
        "boundary": row["_boundary"],                                        # изменение у порога значимости
        "quote_bad": 0.0 if row["quote_status"] in ("verified", "empty") else 1.0,
    }


def logistic_fit(X, y, iters=4000, lr=0.5, l2=1e-3, seed=0):
    """Логистическая регрессия на numpy: веса меры неуверенности. Без sklearn — одна формула."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, X.shape[1] + 1)
    Xb = np.c_[X, np.ones(len(X))]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        grad = Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0]
        w -= lr * grad
    return w


def logistic_score(X, w):
    return 1 / (1 + np.exp(-(np.c_[X, np.ones(len(X))] @ w)))


# ---------- данные ----------

def table(items, reports):
    """(отчёт, показатель, прогон) → балл, мера неуверенности, признаки."""
    g = items[(items.config == CONFIG) & items.report.isin(reports)].copy()
    g["_boundary"] = [boundary_closeness(r["indicator"], r) for _, r in g.iterrows()]
    feat = pd.DataFrame([features(r) for _, r in g.iterrows()], index=g.index)
    out = pd.concat([g[["report", "indicator", "run", "score"]], feat], axis=1)
    # метка для обучения: балл этого прогона ≠ большинству ОСТАЛЬНЫХ прогонов той же клетки
    lab = []
    for (rep, ind), cell in out.groupby(["report", "indicator"]):
        for i, r in cell.iterrows():
            others = cell.drop(i)["score"]
            lab.append((i, int(len(others) and r["score"] != others.mode().iloc[0])))
    out["unstable"] = pd.Series(dict(lab))
    return out


def pairs(cal):
    """Все упорядоченные пары прогонов внутри клетки: (мера первого, различие баллов, отчёт)."""
    rows = []
    for (rep, ind), cell in cal.groupby(["report", "indicator"]):
        v = cell[["S", "score"]].to_numpy()
        for a in range(len(v)):
            for b in range(len(v)):
                if a != b:
                    rows.append((rep, ind, v[a][0], int(v[a][1] != v[b][1])))
    return pd.DataFrame(rows, columns=["report", "indicator", "S", "err"])


# ---------- сертификация ----------

def emp_bernstein_upper(x, delta):
    """Верхняя граница среднего по n независимым величинам из отрезка длины R (Маурер–Понтил, 2009)."""
    n, R = len(x), float(np.max(x) - np.min(x)) or 1.0
    if n < 2:
        return np.inf
    v = float(np.var(x, ddof=1))
    return float(np.mean(x) + np.sqrt(2 * v * np.log(2 / delta) / n) + 7 * R * np.log(2 / delta) / (3 * (n - 1)))


def certify(pr, alpha, delta, by_report=True):
    """Learn then Test, последовательная проверка порогов (fixed sequence) — поправка на перебор не нужна.

    Пороги упорядочены заранее, от самого строгого к самому щедрому; начинаем с первого, при котором принятых
    хватает, чтобы граница в принципе могла опуститься до alpha (k_min = ln(1/delta)/ln(1/(1-alpha)): при нуле
    ошибок среди k_min принятых граница Клоппера–Пирсона 1 − delta^(1/k) не выше alpha). Идём вверх, пока
    гипотеза «риск > alpha» отвергается точной биномиальной границей (Клоппер–Пирсон) на уровне delta;
    останавливаемся на первой неудаче. FWER контролируется порядком проверок, а не делением уровня
    (Learn then Test, Angelopoulos и др.).
    """
    best = None
    k_min = abstain.min_clean_k(alpha, delta, "cp")
    # сетка порогов объявлена заранее: децили меры неуверенности. Так на каждом шаге принятых заведомо много,
    # и проверка не обрывается на случайной ошибке среди первых десяти ответов.
    for lam in np.unique(np.quantile(pr.S.to_numpy(), np.arange(1, 11) / 10)):
        acc = pr[pr.S <= lam]
        k, x = len(acc), int(acc.err.sum())
        if k < k_min:                       # столько принятых не сертифицирует даже при нуле ошибок
            continue
        cp = abstain.cp_upper(x, k, delta)  # граница по клеткам (нужна независимость клеток внутри отчёта)
        if cp > alpha:                      # гипотеза не отвергнута — дальше пороги только щедрее, стоп
            break
        per_report = np.array([g.err.mean() - alpha for _, g in acc.groupby("report")])
        eb = emp_bernstein_upper(per_report, delta) if by_report else -np.inf   # граница по отчётам
        best = dict(lam=float(lam), accepted=k, errors=x, observed=x / k, cp_upper=cp,
                    eb_ok=bool(eb <= 0), eb_value=float(eb), reports=len(set(acc.report)))
    return best


def main(argv):
    alpha = float(argv[argv.index("--alpha") + 1]) if "--alpha" in argv else 0.10
    delta = float(argv[argv.index("--delta") + 1]) if "--delta" in argv else 0.10
    _, items = report.load()
    fit, cal = table(items, FIT), table(items, CAL)
    w = logistic_fit(fit[FEATURES].to_numpy(float), fit.unstable.to_numpy(float))
    print("веса меры неуверенности (обучены на 4 отладочных отчётах, калибровочные не видели):")
    for name, val in zip(FEATURES + ["сдвиг"], w):
        print(f"  {name:10} {val:+.2f}")
    cal = cal.assign(S=logistic_score(cal[FEATURES].to_numpy(float), w))
    pr = pairs(cal)
    base = pr.err.mean()
    print(f"\nкалибровка: {len(set(cal.report))} отчёта × {cal.run.nunique()} прогона, "
          f"{len(cal)} ответов, {len(pr)} пар прогонов")
    print(f"риск без отказов (принимаем всё): {base:.3f}")
    rows = []
    for a in (0.05, 0.10, 0.20):
        c = certify(pr, a, delta)
        if not c:
            rows.append({"alpha": a, "порог": "—", "покрытие": 0.0, "ошибок среди принятых": "—",
                         "граница по клеткам": "—", "по отчётам": "—"})
            continue
        rows.append({"alpha": a, "порог": round(c["lam"], 3), "покрытие": round(c["accepted"] / len(pr), 3),
                     "ошибок среди принятых": f"{c['errors']}/{c['accepted']} = {c['observed']:.3f}",
                     "граница по клеткам": round(c["cp_upper"], 3),
                     "по отчётам": "проходит" if c["eb_ok"] else f"не проходит ({c['eb_value']:+.3f})"})
    # различает ли мера вообще: риск по группам неуверенности + AUC (доля правильно упорядоченных пар)
    q = pd.qcut(pr.S, 5, duplicates="drop")
    grp = pr.groupby(q, observed=True).err.agg(["mean", "size"])
    pos, neg = pr.S[pr.err == 1].to_numpy(), pr.S[pr.err == 0].to_numpy()
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    print("\nриск по пятиням меры неуверенности (снизу вверх):")
    print("  " + "  ".join(f"{r['mean']:.3f} (n={int(r['size'])})" for _, r in grp.iterrows()))
    print(f"  AUC меры: {auc:.3f} (0,5 — мера бесполезна; 1,0 — идеально отделяет)")
    print("\nсертифицированный порог (delta =", delta, "):")
    print(pd.DataFrame(rows).to_string(index=False))
    # диагностика зависимости клеток внутри отчёта: во сколько раз дисперсия больше независимой
    acc = pr[pr.S <= (rows[1]["порог"] if isinstance(rows[1]["порог"], float) else pr.S.max())]
    m = acc.groupby("report").err.mean()
    ind_var = base * (1 - base) / (len(acc) / max(1, len(m)))
    print(f"\nразброс доли ошибок по отчётам: {m.std(ddof=1):.3f}; "
          f"если бы клетки были независимы, ожидался бы {np.sqrt(ind_var):.3f} "
          f"(во сколько раз больше: {m.std(ddof=1) / np.sqrt(ind_var):.1f})")


def selftest():
    # мера неуверенности обязана быть монотонной: чем выше S, тем чаще расхождение
    # риск растёт с мерой неуверенности; данных должно хватать — иначе правило честно ничего не сертифицирует
    rng = np.random.default_rng(0)
    n = 3000
    s = rng.uniform(0, 1, n)
    err = (rng.uniform(0, 1, n) < s ** 2).astype(int)
    pr = pd.DataFrame({"report": rng.integers(0, 8, n), "indicator": "x", "S": s, "err": err})
    c = certify(pr, 0.10, 0.10)
    assert c and c["cp_upper"] <= 0.10 and 0.3 < c["lam"] < 0.65, c
    # истинный риск среди принятых здесь известен (среднее s) — он не должен заметно превышать alpha
    true_risk = (s[s <= c["lam"]] ** 2).mean()          # истинный риск среди принятых известен по построению
    assert true_risk <= 0.10, true_risk
    # мало данных — правило обязано отказаться от сертификации, а не выдать порог наугад
    assert certify(pd.DataFrame({"report": 0, "indicator": "x", "S": s[:30], "err": err[:30]}), 0.10, 0.10) is None
    # ни одного принятого при заведомо шумных данных
    pr2 = pd.DataFrame({"report": 0, "indicator": "x", "S": np.ones(50), "err": np.ones(50, int)})
    assert certify(pr2, 0.10, 0.10) is None
    assert emp_bernstein_upper(np.zeros(10), 0.1) > 0
    print("abstain_run: самопроверка пройдена")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else main(sys.argv[1:])
