"""Глубокий анализ, часть 2 (23.09): строгие версии главных тестов и проверки «что было бы, если».

 10. Хрупкость — свойство клетки или показателя? Перестановочный тест с сохранением трудности каждого показателя.
 11. «Запасной выход»: если код не смог посчитать числовой балл, берётся балл модели. Что было бы при правиле кода.
 12. Как сводить 24 оценки в вывод: сумма (спека) против альтернатив — достижим ли «weak», устойчив ли вывод.
 13. Порог отказа только на отложенных отчётах (без отладочных, которые видела настройка).
 14. Размышление без DeepSeek (у него база и опыт шли через разных провайдеров).
 15. Стресс-замер итогового конвейера: T=0,7, 5 прогонов, все 15 отчётов.
Все пересчёты — на готовых ответах, без новых прогонов. Вывод помечается как ретроспективный.
"""
import collections
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstain  # noqa: E402
import final_checks as fc  # noqa: E402
import final_table as ft  # noqa: E402
import report  # noqa: E402
import score_rules as sr  # noqa: E402
import silver_gold as sg  # noqa: E402

M = fc.M
ROOT = Path(__file__).resolve().parent.parent
PROD = fc.cfg(M, "t0-shuf-ex2-tx2")
BLOCKS = {  # блоки спеки: 8 блоков, 24 показателя
    "прибыльность": ["net_profit", "eps", "roe", "guidance"],
    "доходы": ["nii", "fee_income", "nim"],
    "эффективность": ["opex", "cir"],
    "риск": ["cor", "provisions", "portfolio_quality"],
    "баланс": ["corporate_loans", "retail_loans", "customer_funds", "market_share"],
    "капитал": ["capital_adequacy", "dividends", "book_value_per_share"],
    "клиенты": ["active_clients", "digital_metrics", "tech_development"],
    "среда": ["external_conditions", "ceo_tone"],
}
random.seed(7)


def fragile_flags(v):
    return {k: len(set(s)) > 1 for k, s in v.items()}


def stratified_overlap(ref, other, keys, n_iter=20000):
    """P(перекрытие ≥ наблюдаемого), если хрупкие клетки второго возмущения случайно переставить между отчётами
    ВНУТРИ каждого показателя (число хрупких у каждого показателя сохраняется)."""
    obs = sum(ref[k] and other[k] for k in keys)
    by_ind = collections.defaultdict(list)
    for k in keys:
        by_ind[k[1]].append(k)
    hits = 0
    exp = []
    rng = random.Random(0)                     # своё зерно: p не зависит от того, что считали до этого
    for _ in range(n_iter):
        ov = 0
        for ind, ks in by_ind.items():
            flags = [other[k] for k in ks]
            rng.shuffle(flags)
            ov += sum(f and ref[k] for f, k in zip(flags, ks))
        exp.append(ov); hits += ov >= obs
    return obs, statistics.mean(exp), (hits + 1) / (n_iter + 1)


def b10(items):
    print("\n10. СВОЙСТВО КЛЕТКИ ИЛИ ПОКАЗАТЕЛЯ? Перестановка хрупкости внутри каждого показателя (строже гипергеометрии)")
    dev = fc.DEV
    anon = fc.votes(items, PROD, ["anon_" + r for r in dev])
    anon.index = anon.index.set_levels([anon.index.levels[0].str.replace("anon_", "")] + list(anon.index.levels[1:]))
    ref = fragile_flags(fc.votes(items, fc.cfg(M, "t0.7-ex2-mp0.05-tx2"), dev, range(1, 11)))
    sets = {"T=0,3": fragile_flags(fc.votes(items, fc.cfg(M, "t0.3-ex2-mp0.05-tx2"), dev, range(1, 10))),
            "порядок, T=0": fragile_flags(fc.votes(items, PROD, dev, range(1, 6))),
            "обезличенный текст": fragile_flags(anon)}
    keys = [k for k in ref if all(k in s for s in sets.values())]
    for name, s in sets.items():
        obs, exp, p = stratified_overlap(ref, s, keys)
        print(f"  {name:20} совпало с T=0,7: {obs:>2}; ожидалось при той же трудности показателей {exp:4.1f}; p = {p:.4f}")
    # то же для «межмодельных» расхождений (блок 3 первой части)
    others = ["deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3", "openai/gpt-5.6-sol", "openai/gpt-6-sol"]
    vloc = fc.votes(items, PROD, fc.BATTLE, range(1, 6))
    t07 = fc.votes(items, fc.cfg(M, "t0.7-ex2-mp0.05-tx2"), fc.BATTLE, range(1, 11))
    meds = []
    for m in others:
        v = fc.votes(items, fc.cfg(m, "t0-shuf-ex2-or-tx2"), fc.BATTLE)
        meds.append({k: statistics.median_low(s) for k, s in v.items()})
    keys = [k for k in vloc.index if all(k in md for md in meds) and k in t07.index]
    selfany = {k: len(set(vloc[k])) > 1 or len(set(t07[k])) > 1 for k in keys}
    cross = {k: len({md[k] for md in meds}) > 1 for k in keys}
    obs, exp, p = stratified_overlap(selfany, cross, keys)
    print(f"  расхождения 4 сильных моделей ∩ хрупкие у нашей: {obs}; ожидалось при той же трудности показателей {exp:.1f}; p = {p:.4f}")


def whatif_scores(g, rule):
    num = g.indicator.isin(sr.NUMERIC)
    code = g.code_score
    if rule == "как есть":
        return g.score
    if rule == "код не посчитал → 0":
        return g.score.where(~num, code.fillna(0))
    if rule == "код не посчитал → ±1 по знаку модели":
        return g.score.where(~num, code.fillna(g.model_score.clip(-1, 1)))


def b11(items):
    print("\n11. «ЗАПАСНОЙ ВЫХОД»: код не смог посчитать числовой балл → сейчас берётся балл модели (ретроспективно)")
    g = items[(items.config == PROD) & items.run.isin(range(1, 6)) & items.report.isin(ft.SBER + ft.OTHER)]
    num = g.indicator.isin(sr.NUMERIC)
    fb = g[num & g.code_score.isna()]
    print(f"  доля числовых ответов, где код не посчитал: {len(fb) / num.sum():.1%} ({len(fb)} из {num.sum()})")
    gold = {(r, i): v[0] for r in sg.REPORTS for i, v in sg.read_gold(r).items()}
    for rule in ("как есть", "код не посчитал → 0", "код не посчитал → ±1 по знаку модели"):
        s = whatif_scores(g, rule)
        v = g.assign(s=s).groupby(["report", "indicator"]).s.apply(list)
        sb = [k for k in v.index if k[0] in ft.SBER]
        fr = sum(len(set(v[k])) > 1 for k in sb)
        frn = sum(len(set(v[k])) > 1 for k in sb if k[1] in sr.NUMERIC)
        med = {k: statistics.median_low(v[k]) for k in v.index}
        ok = sum(med[k] == gold[k] for k in gold if k in med)
        print(f"  {rule:40} совпали во всех 5: {1 - fr / len(sb):.1%}  (хрупких числовых {frn}, всего {fr});  эталон {ok}/120")


def verdict_sum(scores):
    return report.verdict(sum(scores.values()))


def b12(items):
    print("\n12. КАК СВОДИТЬ 24 ОЦЕНКИ В ВЫВОД (итоговая таблица v7, 15 отчётов Сбера)")
    v = fc.votes(items, PROD, ft.SBER, range(1, 6))
    med = {k: statistics.median_low(s) for k, s in v.items()}
    runs = items[(items.config == PROD) & items.report.isin(ft.SBER) & items.run.isin(range(1, 6))]

    def agg(scores, how):
        if how == "сумма (спека): > +12 / < −12":
            s = sum(scores.values()); return "s" if s > 12 else "w" if s < -12 else "m"
        if how == "среднее по ненулевым: > +0,5 / < −0,5":
            nz = [x for x in scores.values() if x]; m = sum(nz) / len(nz) if nz else 0
            return "s" if m > 0.5 else "w" if m < -0.5 else "m"
        if how == "среднее блоков: > +0,5 / < −0,5":
            m = statistics.mean(statistics.mean(scores[i] for i in ids) for ids in BLOCKS.values())
            return "s" if m > 0.5 else "w" if m < -0.5 else "m"
    for how in ("сумма (спека): > +12 / < −12", "среднее по ненулевым: > +0,5 / < −0,5", "среднее блоков: > +0,5 / < −0,5"):
        out, stable = [], 0
        for rep in ft.SBER:
            fin = agg({i: med[(rep, i)] for i in report.IDS}, how)
            per = [agg(dict(zip(x.indicator, x.score)), how) for _, x in runs[runs.report == rep].groupby("run")]
            stable += len(set(per)) == 1
            out.append(f"{rep[2:]}:{fin}")
        print(f"  {how:40} {' '.join(out)}   | вывод одинаков во всех 5 перестановках: {stable}/15")


def b13(items):
    print("\n13. ПОРОГ ОТКАЗА ТОЛЬКО НА ОТЛОЖЕННЫХ ОТЧЁТАХ (72 клетки; отладочные исключены)")
    hold = ["2023_q2", "2024_q4", "2025_q3"]
    gold = {(r, i): v[0] for r in hold for i, v in sg.read_gold(r).items()}
    v = fc.votes(items, fc.cfg(M, "t0.7-ex2-mp0.05-tx2"), hold, range(1, 11))
    rows = []
    for k in gold:
        c = collections.Counter(v[k]); top, cnt = c.most_common(1)[0]
        rows.append((1 - cnt / len(v[k]), int(abstain.err_direction(top, gold[k])), int(top != gold[k])))
    for label, idx in (("по направлению", 1), ("строгое", 2)):
        errs = [r[idx] for r in rows]
        for alpha in (0.10, 0.15, 0.20):
            lam, cov, ucb = abstain.calibrate([r[0] for r in rows], errs, alpha, 0.10 / 3)
            # вторичный: фиксированная последовательность порогов (без поправки на перебор)
            fs = None
            for t in sorted({r[0] for r in rows}):
                sel = [r[idx] for r in rows if r[0] <= t]
                if abstain.cp_upper(sum(sel), len(sel), 0.10 / 3) > alpha:
                    break
                fs = (t, len(sel) / len(rows))
            pr = "—" if lam is None else f"{cov:.0%} (граница {ucb:.3f})"
            print(f"  {label:15} alpha={alpha:.2f}: предрегистр. {pr:22} | фикс. последовательность {'—' if fs is None else f'{fs[1]:.0%}'}")
    print(f"  ошибок по направлению: {sum(r[1] for r in rows)} из 72; строгих: {sum(r[2] for r in rows)} из 72")


def b14(items):
    print("\n14. РАЗМЫШЛЕНИЕ БЕЗ DEEPSEEK (у DeepSeek база — провайдер DeepInfra, опыт — CoreWeave; у двух других провайдер один)")
    gold = {(r, i): g[0] for r in fc.GOLD4 for i, g in sg.read_gold(r).items()}
    for models in (["qwen/qwen3.8-27b", "z-ai/glm-5.3-flash"], ["deepseek/deepseek-v4.1-flash", "qwen/qwen3.8-27b", "z-ai/glm-5.3-flash"]):
        up = down = 0; dfr = []
        for m in models:
            va = fc.votes(items, fc.cfg(m, "t0-shuf-ex2-or-tx2"), fc.BATTLE)
            vb = fc.votes(items, fc.cfg(m, "t0-shuf-ex2-think-tx2"), fc.BATTLE)
            a = {k: statistics.median_low(s) for k, s in va.items()}; b = {k: statistics.median_low(s) for k, s in vb.items()}
            for k in gold:
                if k in a and k in b:
                    up += b[k] == gold[k] and a[k] != gold[k]; down += a[k] == gold[k] and b[k] != gold[k]
            for rep in fc.BATTLE:
                dfr.append(sum(len(set(s)) > 1 for (r, _), s in vb.items() if r == rep) -
                           sum(len(set(s)) > 1 for (r, _), s in va.items() if r == rep))
        n = up + down
        p = fc.mcnemar_exact(up, down)
        lo, hi = fc.boot_mean(dfr)
        print(f"  {' + '.join(x.split('/')[1] for x in models):55} эталон +{up}/−{down} (двусторонний p={p:.3f}); хрупких {statistics.mean(dfr):+.2f} [{lo:+.2f}; {hi:+.2f}]")


def b15(items):
    print("\n15. СТРЕСС-ЗАМЕР ИТОГОВОГО КОНВЕЙЕРА: T=0,7, первые 5 прогонов")
    for name, reps in (("15 отчётов Сбера", ft.SBER), ("4 отладочных", fc.DEV), ("чужие банки", ft.OTHER)):
        v = fc.votes(items, fc.cfg(M, "t0.7-ex2-mp0.05-tx2"), reps, range(1, 6))
        v = v[[len(s) == 5 for s in v]]
        ok = sum(len(set(s)) == 1 for s in v)
        print(f"  {name:18} совпали во всех 5: {ok / len(v):.1%} ({ok} из {len(v)} клеток, отчётов {v.index.get_level_values(0).nunique()})")
    v = fc.votes(items, PROD, ft.SBER, range(1, 6))
    print(f"  для сравнения боевой режим (T=0, 5 перестановок), 15 Сбера: {sum(len(set(s)) == 1 for s in v) / len(v):.1%}")


def b16(items):
    print("\n16. ТРИ СИЛЬНЫЕ ОТКРЫТЫЕ МОДЕЛИ НА ВСЕХ 19 ОТЧЁТАХ (15 Сбера + 4 чужих; 5 перестановок, T=0)")
    reps = ft.SBER + ft.OTHER
    gold = {(r, i): v[0] for r in sg.REPORTS for i, v in sg.read_gold(r).items()}
    runs = {"ноутбук Qwen 27B": (PROD, range(1, 6)), "DeepSeek V4.1 Flash": (fc.cfg("deepseek/deepseek-v4.1-flash", "t0-shuf-ex2-or-tx2"), None),
            "GLM-5.3": (fc.cfg("z-ai/glm-5.3", "t0-shuf-ex2-or-tx2"), None)}
    per, med = {}, {}
    for name, (c, rr) in runs.items():
        v = fc.votes(items, c, reps, rr)
        assert v.index.get_level_values(0).nunique() == 19, (name, v.index.get_level_values(0).nunique())
        per[name] = [sum(len(set(v[(r, i)])) > 1 for i in report.IDS) for r in reps]
        med[name] = {k: statistics.median_low(s) for k, s in v.items()}
    base = per["ноутбук Qwen 27B"]
    for name in runs:
        ok = sum(med[name][k] == g for k, g in gold.items())
        lo, hi = fc.cp_interval(ok, len(gold))
        d = "" if name.startswith("ноутбук") else "разница с ноутбуком {:+.2f} [{:+.2f}; {:+.2f}]".format(
            statistics.mean(x - y for x, y in zip(per[name], base)), *fc.boot_mean([x - y for x, y in zip(per[name], base)]))
        print(f"  {name:20} хрупких на отчёт {statistics.mean(per[name]):.2f}  эталон {ok}/{len(gold)} [{lo:.0%}; {hi:.0%}]  {d}")
    names = list(runs)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b_ = med[names[i]], med[names[j]]
            same = sum(a[k] == b_[k] for k in a)
            vs = sum(report.verdict(sum(a[(r, x)] for x in report.IDS)) == report.verdict(sum(b_[(r, x)] for x in report.IDS)) for r in reps)
            print(f"  {names[i]} ~ {names[j]}: клетки совпали {same}/{len(a)} = {same / len(a):.0%}; вывод совпал в {vs} из 19")
    allv = sum(len({report.verdict(sum(med[n][(r, x)] for x in report.IDS)) for n in names}) == 1 for r in reps)
    print(f"  вывод одинаков у всех трёх: {allv} из 19")
    for r in reps:
        sums = {n: sum(med[n][(r, x)] for x in report.IDS) for n in names}
        if len({report.verdict(v) for v in sums.values()}) > 1:
            print(f"   расходится {r}: суммы {sums} — от порога 12 не дальше {max(abs(v - 12) for v in sums.values())}")


def b17(items):
    print("\n17. ИТОГОВЫЙ РЕЖИМ (T=0, 5 перестановок): ОШИБКИ ПРОТИВ ЭТАЛОНА В ХРУПКИХ И УСТОЙЧИВЫХ КЛЕТКАХ (5 отчётов, 120 клеток)")
    gold = {(r, i): v[0] for r in sg.REPORTS for i, v in sg.read_gold(r).items()}
    v = fc.votes(items, PROD, sg.REPORTS, range(1, 6))
    fr = {k for k in gold if len(set(v[k])) > 1}
    err = {k for k in gold if statistics.median_low(v[k]) != gold[k]}
    ed = {k for k in err if abstain.err_direction(statistics.median_low(v[k]), gold[k])}
    auto = len(gold) - len(fr)
    print(f"  хрупких (уходят человеку): {len(fr)} из {len(gold)} ({len(fr) / len(gold):.1%}); ошибок всего {len(err)}, "
          f"из них в хрупких {len(err & fr)}; ошибок знака или на 2+ балла: {len(ed)}")
    print(f"  ошибок среди принятых автоматически: {len(err - fr)} из {auto} ({len(err - fr) / auto:.1%}; верхняя граница 90% "
          f"{abstain.cp_upper(len(err - fr), auto, 0.10):.3f}); среди хрупких: {len(err & fr)} из {len(fr)}")
    for k in sorted(err):
        print(f"   {k[0]} {k[1]:20} метод {statistics.median_low(v[k]):+d}, эталон {gold[k]:+d}, {'хрупкая' if k in fr else 'устойчивая'}")


def b18(items):
    print("\n18. v7 ПРОТИВ v10 (итоговый режим; отложенные отчёты 2023_q2, 2024_q4, 2025_q3)")
    hold = ["2023_q2", "2024_q4", "2025_q3"]
    gold = {(r, i): g for r in hold for i, g in sg.read_gold(r).items()}
    und = [k for k, g in gold.items() if not g[2]]
    for spec in ("v7_extract", "v10_extract"):
        v = fc.votes(items, fc.cfg(M, "t0-shuf-ex2-tx2", spec), ft.SBER + ft.OTHER, range(1, 6))
        med = {k: statistics.median_low(s) for k, s in v.items()}
        reps = sorted({r for r, _ in v.index})
        fr = [sum(len(set(v[(r, i)])) > 1 for i in report.IDS) for r in reps]
        sb = [k for k in v.index if k[0] in ft.SBER]
        print(f"  {spec:12} эталон {sum(med[k] == g[0] for k, g in gold.items())}/72; без спорных "
              f"{sum(med[k] == gold[k][0] for k in und)}/{len(und)}; хрупких на отчёт ({len(reps)} отч.) {statistics.mean(fr):.2f}; "
              f"Сбер совпали во всех 5: {sum(len(set(v[k])) == 1 for k in sb)}/{len(sb)}")


def b19(items):
    print("\n19. OPUS И GROK: СОКРАЩЁННЫЙ ПРОТОКОЛ (в сравнение моделей не входят)")
    for m in ("anthropic/claude-opus-5", "anthropic/claude-opus-5.5", "x-ai/grok-4.7"):
        v = fc.votes(items, fc.cfg(m, "t0-shuf-ex2-or-tx2"), fc.BATTLE)
        reps = sorted({r for r, _ in v.index})
        fr = [sum(len(set(v[(r, i)])) > 1 for i in report.IDS if (r, i) in v.index) for r in reps]
        print(f"  {m:28} отчёты {reps}; голосов в клетке {min(len(s) for s in v)}–{max(len(s) for s in v)}; "
              f"хрупких на отчёт {statistics.mean(fr):.2f}")


def b20(items):
    import json
    print("\n20. ОБЪЁМ РАБОТЫ")
    loc, cloud_models = [], set()
    for p_ in (ROOT / "raw").glob("*.json"):
        m = json.loads(p_.read_text())["meta"]
        (cloud_models.add(m["model"]) if m.get("cloud") else loc.append(m))
    wh = [x["load_w"] * x["sec"] / 3600 for x in loc if x.get("load_w")]
    print(f"  локальных прогонов: {len(loc)}; машинных часов: {sum(x.get('sec') or 0 for x in loc) / 3600:.0f}; "
          f"энергия ≈ {statistics.mean(wh) * len(loc) / 1000:.1f} кВт·ч (замер мощности есть у {len(wh)} прогонов, "
          f"в среднем {statistics.mean(wh):.1f} Вт·ч на прогон)")
    print(f"  версий спецификации в локальных прогонах: {len({x['spec_version'] for x in loc})}")
    led = ROOT / "logs" / "cloud_ledger.jsonl"
    if led.exists():
        rows = [json.loads(x) for x in led.read_text().splitlines() if x.strip()]
        print(f"  облако: запросов {len(rows)}, моделей с сохранёнными ответами {len(cloud_models)}, расход ${sum(r['cost'] for r in rows):.2f}")


def b21(items):
    import figures
    print("\n21. ЛЕСТНИЦА СТАБИЛИЗАЦИИ (4 отладочных отчёта, по 5 прогонов на шаг) — данные рисунка 1")
    for name, un, fab, t in figures.ladder_data(items):
        print(f"  {name:52} {t:6} совпали во всех 5: {un:5.1f}%   выдуманных цитат: {fab:4.1f}%")


def b22(items):
    print("\n22. ФОРМА ЗАПИСИ ПРАВИЛ (ночь 2): доля показателей, одинаковых во всех 3 первых прогонах при T=0,7, "
          "4 отладочных отчёта, старое распознавание текста")
    forms = [("v2_inplace", "t0.7", "правила вписаны в текст"), ("v2_all", "t0.7-rf", "дописаны в конец, отчёт перед спекой"),
             ("v2_compact", "t0.7", "только правила, сжато"), ("v2_state", "t0.7", "модуль «состояние»"),
             ("v2_all", "t0.7", "правила дописаны в конец"), ("v2_market", "t0.7", "модуль «доля рынка»"),
             ("v1_baseline", "t0.7", "исходная спецификация"), ("v2_period", "t0.7", "модуль «период»"),
             ("v2_top", "t0.7", "уточнения в начале")]
    for spec, var, name in forms:
        v = fc.votes(items, fc.cfg(M, var, spec), fc.DEV, range(1, 4))
        v = v[[len(x) == 3 for x in v]]
        if v.empty:
            continue
        print(f"  {spec + ' ' + var:22} {name:38} {sum(len(set(x)) == 1 for x in v) / len(v):.1%} ({len(v)} клеток)")
    print("  в скольких отчётах из 4 показатель расходился (3 первых прогона): исходная спецификация → правила вписаны в текст")
    base = fc.votes(items, fc.cfg(M, "t0.7", "v1_baseline"), fc.DEV, range(1, 4))
    new = fc.votes(items, fc.cfg(M, "t0.7", "v2_inplace"), fc.DEV, range(1, 4))
    for ind in ("market_share", "digital_metrics", "capital_adequacy", "cor", "active_clients", "opex", "cir"):
        n = lambda v: sum(len(set(v[(r, ind)])) > 1 for r in fc.DEV if (r, ind) in v.index)
        print(f"   {ind:18} {n(base)} → {n(new)}")


if __name__ == "__main__":
    want = sys.argv[1:] or [str(i) for i in range(10, 23)]
    _, items = report.load()
    for b in want:
        globals()[f"b{b}"](items)
