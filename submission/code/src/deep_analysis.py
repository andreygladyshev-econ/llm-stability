"""Глубокий анализ для записки (23.09): те же данные с разных сторон.

  1. Модели группами: локальная / открытые в облаке / закрытые; ступени железа (32 → 512 ГБ).
  2. Какие показатели трудны для ВСЕХ сильных моделей (свойство спеки), а какие — для одной (свойство модели).
  3. Где сильные модели расходятся между собой — там же, где наша модель расходится сама с собой?
  4. Разбор ошибок против эталона: чьи ошибки общие, какие клетки неверны почти у всех.
  5. Почему плавают числовые клетки: разная цифра, код не посчитал, граница порога.
  6. Сбер 2022–2026: суммы, нули, хрупкость; почему «weak» недостижим.
  7. Скрытая потеря информации в двухшаговой схеме: «ложные нули» против эталона.
  8. Подгонка: отладочные против отложенных отчётов.
  9. Цена и время одного отчёта в боевом режиме на разном железе.
Запуск: .venv/bin/python src/deep_analysis.py [номера блоков]
"""
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_checks as fc  # noqa: E402
import final_table as ft  # noqa: E402
import report  # noqa: E402
import score_rules as sr  # noqa: E402
import silver_gold as sg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
M = fc.M
BATTLE, GOLD4 = fc.BATTLE, fc.GOLD4
# (веса, параметров всего, активных — млрд, архитектура) — из карточек OpenRouter и Hugging Face (23.09).
# Ступень железа не пишется руками, а выводится из размера (функция hw): 22.09 две модели стояли не на своей ступени.
MODELS = {
    "ноутбук Qwen 27B 4 бита": ("открытые", 27, 27, "плотная"),
    "qwen/qwen3.8-27b": ("открытые", 27, 27, "плотная"),
    "google/gemma-4-31b-it": ("открытые", 30.7, 30.7, "плотная"),
    "qwen/qwen3.6-35b-a3b": ("открытые", 35, 3, "смесь экспертов"),
    "nvidia/nemotron-3-nano-30b-a3b": ("открытые", 30, 3, "смесь экспертов"),
    "openai/gpt-oss-120b": ("открытые", 117, 5.1, "смесь экспертов"),
    "nvidia/nemotron-3-super-120b-a12b": ("открытые", 120, 12, "смесь экспертов"),
    "qwen/qwen3-235b-a22b-2507": ("открытые", 235, 22, "смесь экспертов"),
    "z-ai/glm-5.3-flash": ("открытые", 321.3, None, "не раскрыта"),
    "xiaomi/mimo-v2.6-flash": ("открытые", 309, 15, "смесь экспертов"),
    "deepseek/deepseek-v4.1-flash": ("открытые", 763.2, None, "смесь экспертов"),  # по описанию 8–16 млрд активных
    "z-ai/glm-5.3": ("открытые", 753.3, None, "смесь экспертов"),
    "openai/gpt-5.6-sol": ("закрытые", None, None, "—"),
    "openai/gpt-6-sol": ("закрытые", None, None, "—"),
    "openai/gpt-6-luna": ("закрытые", None, None, "—"),
}
GB_PER_B = 0.55                     # ГБ под веса на млрд параметров при 4 битах
HW = (32, 64, 128, 256, 512)        # классы железа, ГБ


def hw(params):
    """Наименьший класс железа, куда веса при 4 битах влезают с запасом 15% на контекст."""
    return next((h for h in HW if params * GB_PER_B * 1.15 <= h), None) if params else None


LOCAL = "ноутбук Qwen 27B 4 бита"


def cfg_of(name):
    return (fc.cfg(M, "t0-shuf-ex2-tx2"), range(1, 6)) if name == LOCAL else (fc.cfg(name, "t0-shuf-ex2-or-tx2"), None)


def medians(items, name, reps=BATTLE):
    c, r = cfg_of(name)
    v = fc.votes(items, c, reps, r)
    return v, {k: statistics.median_low(s) for k, s in v.items()}


def gold_all():
    return {(r, i): g for r in sg.REPORTS for i, g in sg.read_gold(r).items()}


def meta(name):
    if name == LOCAL:
        ms = [json.loads(p.read_text())["meta"] for p in sorted((ROOT / "raw").glob("qwen3.8-27b-mlx-4bit__v7_extract__t0-shuf-ex2-tx2__*.json"))]
    else:
        ms = [json.loads(p.read_text())["meta"] for p in sorted((ROOT / "raw").glob(f"{name.replace('/', '-')}__v7_extract__t0-shuf-ex2-or-tx2__*.json"))]
    return ms


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def b1(items):
    print("\n1. МОДЕЛИ ГРУППАМИ (6 отчётов × 5 перестановок, T=0; эталон — 4 отчёта, 96 клеток)")
    gold = gold_all()
    rows = []
    for name, (w, params, active, arch) in MODELS.items():
        tier = f"{hw(params)} ГБ" if params else "—"
        v, med = medians(items, name)
        if v.empty:
            continue
        per = [sum(len(set(s)) > 1 for (r, _), s in v.items() if r == rep) for rep in BATTLE]
        keys = [k for k in gold if k[0] in GOLD4 and k in med]
        ok = sum(med[k] == gold[k][0] for k in keys)
        grave = sum(abs(med[k] - gold[k][0]) >= 2 or med[k] * gold[k][0] < 0 for k in keys)
        ms = meta(name)
        sec = statistics.median(m["sec"] for m in ms) if ms else 0
        cost = statistics.mean(m.get("cost_usd") or 0 for m in ms) if ms else 0
        dv = fc.votes(items, fc.cfg(name, "t0-ex2-det-tx2"), ["2026_q2"]) if name != LOCAL else None
        det = sum(len(set(s)) > 1 for s in dv) if dv is not None and not dv.empty else 0
        g = items[(items.config == cfg_of(name)[0]) & items.report.isin(BATTLE)]
        q = g[g.quote_status.isin(["verified", "fabricated", "number_mismatch"])]
        fab = (q.quote_status == "fabricated").mean()
        rows.append((name, w, tier, f"{params:g}/{active:g}" if active else f"{params:g}" if params else "—",
                     statistics.mean(per), ok, grave, det, fab, sec, cost))
    print(f"{'модель':34}{'веса':>9}{'железо':>12}{'парам.':>14}{'хрупк':>7}{'эталон':>8}{'грубых':>7}"
          f"{'повтор':>7}{'выдум':>7}{'сек':>6}{'$/зап':>8}")
    for r in sorted(rows, key=lambda x: x[4]):
        print(f"{r[0]:34}{r[1]:>9}{r[2]:>12}{r[3]:>14}{r[4]:>7.2f}{r[5]:>5}/96{r[6]:>7}{r[7]:>7}{r[8]:>7.1%}{r[9]:>6.0f}{r[10]:>8.4f}")
    for grp in ("открытые", "закрытые"):
        sub = [r for r in rows if r[1] == grp and r[4] < 4]
        if sub:
            print(f"  сильные {grp}: хрупких {statistics.mean(r[4] for r in sub):.2f}, эталон {statistics.mean(r[5] for r in sub):.1f}/96, "
                  f"повтор {statistics.mean(r[7] for r in sub):.1f} кл., n={len(sub)}")
    print("\n  ступени железа (лучшая модель ступени):")
    tiers = collections.defaultdict(list)
    for r in rows:
        if r[2] != "—":
            tiers[r[2]].append(r)
    for t in sorted(tiers, key=lambda t: int(t.split()[0])):
        best = min(tiers[t], key=lambda x: x[4])
        print(f"  {t:12} лучшая: {best[0]:32} хрупких {best[4]:.2f}, эталон {best[5]}/96;  "
              f"все модели ступени: {', '.join(f'{x[0].split(chr(47))[-1]} {x[4]:.1f}' for x in tiers[t])}")
    return rows


def b2(items):
    print("\n2. КАКИЕ ПОКАЗАТЕЛИ ТРУДНЫ ДЛЯ ВСЕХ СИЛЬНЫХ МОДЕЛЕЙ (хрупкость по показателю, 6 отчётов)")
    strong = [LOCAL, "deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3", "openai/gpt-5.6-sol", "openai/gpt-6-sol",
              "google/gemma-4-31b-it", "z-ai/glm-5.3-flash", "qwen/qwen3.8-27b"]
    frag = collections.defaultdict(list)
    for name in strong:
        v, _ = medians(items, name)
        for (r, i), s in v.items():
            frag[i].append(len(set(s)) > 1)
    rank = sorted(((i, sum(x) / len(x), len(x)) for i, x in frag.items()), key=lambda t: -t[1])
    for i, p, n in rank[:10]:
        kind = "код" if i in sr.NUMERIC else "модель"
        print(f"  {i:22} хрупкая в {p:5.1%} случаев «модель × отчёт» ({n})   балл ставит: {kind}")
    num = [p for i, p, n in rank if i in sr.NUMERIC]
    jud = [p for i, p, n in rank if i not in sr.NUMERIC]
    print(f"  среднее: числовые (код) {statistics.mean(num):.1%}, суждения (модель) {statistics.mean(jud):.1%}")
    # согласие рангов: трудность показателя у нашей модели против остальных
    loc = collections.Counter(); oth = collections.Counter()
    for name in strong:
        v, _ = medians(items, name)
        tgt = loc if name == LOCAL else oth
        for (r, i), s in v.items():
            tgt[i] += len(set(s)) > 1
    ids = report.IDS
    a = [loc[i] for i in ids]; b = [oth[i] for i in ids]
    ra = {i: sorted(a).index(x) for i, x in zip(ids, a)}; rb = {i: sorted(b).index(x) for i, x in zip(ids, b)}
    n = len(ids); d2 = sum((ra[i] - rb[i]) ** 2 for i in ids)
    print(f"  ранговая корреляция трудности показателей: наша модель против 7 других сильных ρ ≈ {1 - 6 * d2 / (n * (n * n - 1)):.2f}")


def b3(items):
    print("\n3. ГДЕ СИЛЬНЫЕ МОДЕЛИ РАСХОДЯТСЯ МЕЖДУ СОБОЙ — ТАМ ЖЕ, ГДЕ НАША МОДЕЛЬ РАСХОДИТСЯ САМА С СОБОЙ?")
    others = ["deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3", "openai/gpt-5.6-sol", "openai/gpt-6-sol"]
    reps = BATTLE
    vloc, mloc = medians(items, LOCAL, reps)
    meds = [medians(items, m, reps)[1] for m in others]
    keys = [k for k in mloc if all(k in md for md in meds)]
    cross = {k for k in keys if len({md[k] for md in meds}) > 1}
    selff = {k for k in keys if len(set(vloc[k])) > 1}
    # хрупкие у нас по сэмплированию тоже — объединение возмущений на тех же отчётах
    t07 = fc.votes(items, fc.cfg(M, "t0.7-ex2-mp0.05-tx2"), reps, range(1, 11))
    self07 = {k for k, s in t07.items() if k in set(keys) and len(set(s)) > 1}
    selfany = selff | self07
    N = len(keys); k = len(cross & selfany)
    print(f"  клеток {N}; расходятся 4 сильные модели: {len(cross)}; хрупкие у нашей модели (перестановки или T=0,7): {len(selfany)}")
    print(f"  пересечение {k} (случайно ожидалось {len(cross) * len(selfany) / N:.1f}); P(≥{k}) = {fc.hypergeom_sf(k, N, len(selfany), len(cross)):.1e}")
    print(f"  доля межмодельных расхождений, которые наша модель сама помечает хрупкими: {k / max(len(cross), 1):.0%}")
    print("  расхождения моделей, которых наша модель НЕ помечает:", sorted(cross - selfany)[:12])


def b4(items):
    print("\n4. ОШИБКИ ПРОТИВ ЭТАЛОНА: у нашей таблицы и общие для моделей")
    gold = gold_all()
    _, med = medians(items, LOCAL, sg.REPORTS)
    errs = [(k, med[k], gold[k][0], gold[k][2]) for k in gold if k in med and med[k] != gold[k][0]]
    print(f"  наша таблица v7: {len(errs)} расхождений из {len(gold)}")
    for (r, i), m, g, d in errs:
        print(f"   {r} {i:20} модель {m:+d}  эталон {g:+d}  {'СПОРНО' if d else ''}")
    names = [LOCAL, "deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3", "openai/gpt-5.6-sol", "openai/gpt-6-sol",
             "z-ai/glm-5.3-flash", "google/gemma-4-31b-it", "xiaomi/mimo-v2.6-flash"]
    wrong = collections.Counter(); seen = collections.Counter()
    for n in names:
        _, md = medians(items, n, GOLD4)
        for k in gold:
            if k in md:
                seen[k] += 1; wrong[k] += md[k] != gold[k][0]
    common = [(k, wrong[k], seen[k]) for k in wrong if wrong[k] >= 0.6 * seen[k] and seen[k] >= 5]
    print(f"  клетки, где ошибаются ≥60% из {len(names)} сильных моделей (4 отчёта): {len(common)}")
    for (r, i), w, s in sorted(common, key=lambda t: -t[1]):
        print(f"   {r} {i:20} неверно у {w} из {s}; эталон {gold[(r, i)][0]:+d} {'СПОРНО' if gold[(r, i)][2] else ''}")


def b5(items):
    print("\n5. ПОЧЕМУ ПЛАВАЮТ ЧИСЛОВЫЕ КЛЕТКИ (боевой режим v7, 19 отчётов): разбор по ответам")
    reasons = collections.Counter(); ex = collections.defaultdict(list)
    for f in sorted((ROOT / "raw").glob("qwen3.8-27b-mlx-4bit__v7_extract__t0-shuf-ex2-tx2__*.json")):
        pass
    by = collections.defaultdict(list)
    for f in sorted((ROOT / "raw").glob("qwen3.8-27b-mlx-4bit__v7_extract__t0-shuf-ex2-tx2__*.json")):
        d = json.loads(f.read_text()); m = d["meta"]
        if m["run"] > 5 or m["report"].startswith("anon"):
            continue
        p = report.parse(d["content"]) or {}
        for i in sr.NUMERIC:
            x = p.get(i) or {}
            slot, sc = sr.pick(i, x)
            val = (slot, x.get(f"{slot}_value", "") if slot else "", x.get(f"{slot}_base", "") if slot else "",
                   x.get(f"{slot}_change", "") if slot else "")
            by[(m["report"], i)].append((sc, val, x.get("score")))
    for key, runs in by.items():
        scores = [(s if s is not None else ("м", ms)) for s, _, ms in runs]
        final = [s if s is not None else ms for s, _, ms in runs]
        if len(set(final)) <= 1:
            continue
        slots = {v[0] for _, v, _ in runs}
        nums = {(sr.number(v[1]), sr.number(v[2])) for _, v, _ in runs if v[0]}
        if any(s is None for s, _, _ in runs):
            why = "код не смог посчитать в части прогонов → балл модели"
        elif len(slots) > 1:
            why = "разный период (квартал / с начала года)"
        elif len(nums) > 1:
            why = "разная цифра в том же периоде (база, подсегмент, строка)"
        else:
            why = "та же цифра, разное поле изменения (названное / посчитанное)"
        reasons[why] += 1; ex[why].append(key)
    tot = sum(reasons.values())
    for why, n in reasons.most_common():
        print(f"  {n:>2} из {tot}  {why}   напр.: {ex[why][:3]}")


def b6(items):
    print("\n6. СБЕР 2022–2026: суммы, нули, хрупкость (итоговая таблица v7)")
    _, med = medians(items, LOCAL, ft.SBER)
    v, _ = medians(items, LOCAL, ft.SBER)
    for rep in ft.SBER:
        sc = [med[(rep, i)] for i in report.IDS]
        fr = sum(len(set(v[(rep, i)])) > 1 for i in report.IDS)
        print(f"  {rep}  сумма {sum(sc):+3d}  среднее {sum(sc) / 24:+.2f}  нулей {sc.count(0):>2}  плюс {sum(x > 0 for x in sc):>2}  минус {sum(x < 0 for x in sc):>2}  хрупких {fr}")
    sums = [sum(med[(r, i)] for i in report.IDS) for r in ft.SBER]
    print(f"  нижняя граница weak — сумма < −12; самый слабый отчёт даёт {min(sums):+d}. Нулей в среднем {statistics.mean(sum(med[(r, i)] == 0 for i in report.IDS) for r in ft.SBER):.1f} из 24")
    for thr in (0.5, 0.3, 0.25):
        ver = [report.verdict(s) if thr == 0.5 else ("strong" if s / 24 > thr else "weak" if s / 24 < -thr else "mixed") for s in sums]
        print(f"   порог среднего ±{thr}: " + " ".join(f"{r[2:]}:{x[0]}" for r, x in zip(ft.SBER, ver)))
    # вывод относительно собственной истории банка: z-оценка суммы
    mu, sd = statistics.mean(sums), statistics.pstdev(sums)
    print("   относительно истории банка (z-оценка суммы, |z|>1 — выделяется): " +
          " ".join(f"{r[2:]}:{(s - mu) / sd:+.1f}" for r, s in zip(ft.SBER, sums)))


def b7(items):
    print("\n7. ТИХАЯ ПОТЕРЯ В ДВУХШАГОВОЙ СХЕМЕ: итог 0, а эталон ≠ 0 (5 размеченных отчётов)")
    gold = gold_all()
    _, med = medians(items, LOCAL, sg.REPORTS)
    fz = [k for k in gold if k in med and med[k] == 0 and gold[k][0] != 0]
    fnz = [k for k in gold if k in med and med[k] != 0 and gold[k][0] == 0]
    nz = sum(g[0] != 0 for g in gold.values())
    print(f"  ложных нулей: {len(fz)} из {nz} ненулевых в эталоне ({len(fz) / nz:.1%}): {fz}")
    print(f"  лишних ненулей: {len(fnz)} из {len(gold) - nz} нулевых в эталоне: {fnz}")


def b8(items):
    print("\n8. ПОДГОНКА: отладочные против отложенных (эталон и хрупкость)")
    gold = gold_all()
    _, med = medians(items, LOCAL, sg.REPORTS)
    v, _ = medians(items, LOCAL, ft.SBER)
    for name, reps in (("отладочные 2026_q2, 2022_q4", ["2026_q2", "2022_q4"]), ("отложенные 2023_q2, 2024_q4, 2025_q3", ["2023_q2", "2024_q4", "2025_q3"])):
        keys = [k for k in gold if k[0] in reps]
        ok = sum(med[k] == gold[k][0] for k in keys)
        fr = statistics.mean(sum(len(set(v[(r, i)])) > 1 for i in report.IDS) for r in reps)
        print(f"  {name:40} эталон {ok}/{len(keys)} = {ok / len(keys):.1%}; хрупких на отчёт {fr:.1f}")
    dev4 = fc.DEV
    rest = [r for r in ft.SBER if r not in dev4 and r not in ("2023_q2", "2024_q4", "2025_q3")]
    for name, reps in (("4 отладочных (вся настройка)", dev4), ("8 нетронутых кварталов", rest), ("3 отложенных", ["2023_q2", "2024_q4", "2025_q3"])):
        print(f"  {name:32} хрупких на отчёт {statistics.mean(sum(len(set(v[(r, i)])) > 1 for i in report.IDS) for r in reps):.2f}")


def b9(items):
    print("\n9. ЦЕНА И ВРЕМЯ ОДНОГО ОТЧЁТА В БОЕВОМ РЕЖИМЕ (5 перестановок)")
    ms = meta(LOCAL)
    s = statistics.median(m["sec"] + (m.get("load_sec") or 0) for m in ms)
    wh = statistics.median(m["load_w"] * m["sec"] / 3600 for m in ms if m.get("load_w"))
    print(f"  ноутбук (Qwen 27B, 4 бита): {s / 60:.1f} мин на прогон → {5 * s / 60:.0f} мин на отчёт, ~{5 * wh:.0f} Вт·ч; 15 отчётов ≈ {75 * s / 3600:.1f} ч")
    for name in ("deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3", "openai/gpt-6-sol", "openai/gpt-6-luna", "qwen/qwen3.8-27b"):
        ms = meta(name)
        if not ms:
            continue
        s = statistics.median(m["sec"] for m in ms); c = statistics.mean(m.get("cost_usd") or 0 for m in ms)
        print(f"  {name:32} {s:>4.0f} с на запрос → {5 * s / 60:.1f} мин на отчёт подряд (или {s / 60:.1f} мин параллельно), ${5 * c:.3f} на отчёт, ${75 * c:.2f} на 15 отчётов")


if __name__ == "__main__":
    want = sys.argv[1:] or [str(i) for i in range(1, 10)]
    _, items = report.load()
    for b in want:
        globals()[f"b{b}"](items)
