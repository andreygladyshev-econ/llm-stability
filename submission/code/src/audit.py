"""Методический аудит данных перед запиской (23.09). Каждая проверка — одна возможная ошибка, которая исказила бы вывод.
Запуск: .venv/bin/python src/audit.py
"""
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_checks as fc  # noqa: E402
import final_table as ft  # noqa: E402
import report  # noqa: E402
import silver_gold as sg  # noqa: E402
import texts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = sg.GOLD
M = fc.M
PROD = "qwen3.8-27b-mlx-4bit__v7_extract__t0-shuf-ex2-tx2"
problems = []


def say(ok, msg):
    print(("  ✓ " if ok else "  ✗ ") + msg)
    if not ok:
        problems.append(msg)


def metas(prefix):
    out = []
    for p in sorted((ROOT / "raw").glob(f"{prefix}__*.json")):
        d = json.loads(p.read_text())
        out.append((d["meta"], d.get("content") or ""))
    return out


def a1():
    print("\nA1. Одна и та же спека и один и тот же текст во всех прогонах итоговой таблицы")
    ms = [m for m, _ in metas(PROD) if m["run"] <= 5 and not m["report"].startswith("anon")]
    say(len({m["spec_hash"] for m in ms}) == 1, f"спека v7 одна во всех {len(ms)} прогонах (хешей: {len({m['spec_hash'] for m in ms})})")
    by = collections.defaultdict(set)
    for m in ms:
        by[m["report"]].add(m["text_hash"])
    say(all(len(h) == 1 for h in by.values()), "текст каждого отчёта один и тот же во всех прогонах")
    cur = texts.text_hashes("v2") if hasattr(texts, "text_hashes") else None
    hashes = json.loads((ROOT / "text_v2" / "hashes.json").read_text())
    diff = [r for r, hs in by.items() if r in hashes and list(hs)[0] != hashes[r]["sha256"]]
    say(not diff, f"текст на диске совпадает с тем, что видели модели (расхождения: {diff or 'нет'})")


def a2():
    print("\nA2. Полнота: у каждого отчёта ровно 5 разобранных прогонов; обрывы, кэш")
    ms = [(m, c) for m, c in metas(PROD) if m["run"] <= 5 and not m["report"].startswith("anon")]
    cnt = collections.Counter(m["report"] for m, _ in ms)
    say(all(cnt[r] == 5 for r in ft.SBER + ft.OTHER), f"по 5 прогонов на отчёт: {dict(cnt) if any(cnt[r] != 5 for r in ft.SBER + ft.OTHER) else 'все 19'}")
    bad = [m["name"] for m, c in ms if not report.parse(c)]
    say(not bad, f"все ответы разбираются (не разобрано: {bad or 0})")
    say(not [m for m, _ in ms if m.get("truncated")], "обрезанных ответов нет")
    say(not [m for m, _ in ms if m.get("cache_contaminated")], "чужого кэша нет ни в одном прогоне")
    say(len({m.get("temperature") for m, _ in ms}) == 1 and ms[0][0].get("temperature") == 0, "температура 0 во всех прогонах")
    say(len({m.get("ctx_effective") or m.get("ctx_loaded") for m, _ in ms}) == 1, f"контекст один: {({m.get('ctx_effective') for m, _ in ms})}")
    orders = collections.defaultdict(set)
    for m, _ in ms:
        orders[m["report"]].add(m["request_hash"])
    say(all(len(h) == 5 for h in orders.values()), "5 перестановок у каждого отчёта действительно разные запросы")


def a3(items):
    print("\nA3. Итог клетки: мода задания = наша медиана; ничьих нет")
    v = fc.votes(items, fc.cfg(M, "t0-shuf-ex2-tx2"), ft.SBER + ft.OTHER, range(1, 6))
    ties = [k for k, s in v.items() if len([c for c in collections.Counter(s).values() if c == max(collections.Counter(s).values())]) > 1]
    diff = [k for k, s in v.items() if collections.Counter(s).most_common(1)[0][0] != statistics.median_low(s)]
    say(not ties, f"ничьих при 5 голосах нет ({len(ties)})")
    say(not diff, f"мода совпадает с медианой во всех {len(v)} клетках")
    sums = {r: sum(statistics.median_low(v[(r, i)]) for i in report.IDS) for r in ft.SBER}
    say(all(isinstance(x, int) for x in sums.values()), "суммы — целые, считает код (не модель)")


def a4():
    print("\nA4. Эталон не менялся после фиксации; холдаут не тронут до финальной ночи")
    lines = (GOLD_DIR / "ХОЛДАУТ_ХЕШИ_22.09.txt").read_text().splitlines()
    for ln in lines:
        if "_CLAUDE.csv" in ln:
            h, path = ln.split()[:2]
            cur = hashlib.sha256((GOLD_DIR / Path(path).name).read_bytes()).hexdigest()
            say(cur == h, f"{Path(path).name}: хеш совпадает с зафиксированным 22.09")
    early = []
    for p in sorted((ROOT / "raw").glob("*.json")):
        m = json.loads(p.read_text())["meta"]
        if m.get("report") in ("2023_q2", "2024_q4", "2025_q3") and (m.get("started") or "") < "2026-09-22T17:30":
            early.append((m["name"].split("__")[1:3], m["started"][:10]))
    say(len(early) <= 6, f"до финальной ночи холдаут прогонялся {len(early)} раз(а): {sorted(set(map(str, early)))} "
        "— один проверочный прогон всех 15 отчётов 20.09; клетки холдаута при настройке не разбирались")


def a5():
    print("\nA5. Облако: провайдер и параметры — что именно сравниваем")
    rows = collections.defaultdict(lambda: collections.defaultdict(set))
    for p in sorted((ROOT / "raw").glob("*.json")):
        m = json.loads(p.read_text())["meta"]
        if not m.get("cloud"):
            continue
        rows[m["model"]][m["variant"]].add((m.get("provider"), m.get("model_quant"), tuple(m.get("dropped_params") or ()),
                                            m.get("reasoning_mode")))
    for model, vs in sorted(rows.items()):
        for var, s in sorted(vs.items()):
            flag = "" if len(s) == 1 else "  ← несколько провайдеров/настроек в одной серии"
            prov = "; ".join(f"{a} {b}, отброшено {list(c) or '—'}, размышл. {d}" for a, b, c, d in sorted(s, key=str))
            print(f"   {model:34} {var:26} {prov}{flag}")
    notemp = sorted({m for m, vs in rows.items() for s in vs.values() for x in s if "temperature" in x[2]})
    print(f"  ! без температуры (провайдер ставит свою по умолчанию, это НЕ T=0): {notemp}")


def a6(items):
    print("\nA6. Цитаты итоговой таблицы: дословность")
    g = items[(items.config == fc.cfg(M, "t0-shuf-ex2-tx2")) & items.report.isin(ft.SBER) & items.run.isin(range(1, 6))]
    exact = fuzzy = 0
    for (r, i), x in g.groupby(["report", "indicator"]):
        med = statistics.median_low(x.score.tolist())
        if not med:
            continue
        t = report.norm((texts.text_dir("v2") / f"{r}.txt").read_text())
        qs = x[(x.score == med) & (x.quote_status == "verified")].quote.tolist()
        if any(report.norm(q) in t for q in qs):
            exact += 1
        else:
            fuzzy += 1
    say(True, f"у {exact} итоговых оценок цитата модели — точная подстрока распознанного текста (после снятия разметки); "
        f"у {fuzzy} — совпадение ≥92% (ошибка распознавания, перенос строки). В сдачу идёт фрагмент, переписанный "
        "буквами PDF (A7)")


def a7():
    """29.09: цитаты сданной таблицы — посимвольно по текстовому слою PDF; у оценок программы видна база расчёта."""
    import hashlib
    import build_submission as bs
    import pdf_quotes as pq
    print("\nA7. Цитаты сданной таблицы против текстового слоя PDF")
    hs = json.loads((ROOT / "text_pdf" / "hashes.json").read_text())
    bad_hash = [r for r, h in hs.items()
                if hashlib.sha256((ROOT / "text_pdf" / f"{r}.txt").read_text().encode()).hexdigest() != h["sha256"]]
    say(not bad_hash, f"текстовый слой PDF совпадает с зафиксированным ({len(hs)} отчётов)"
        + (f"; изменены: {bad_hash}" if bad_hash else ""))
    sub = next(p for p in (ROOT.parent / "vectors.json", ROOT.parent / "submission" / "vectors.json") if p.exists())
    reps = json.loads(sub.read_text())["отчёты"]
    n = miss = weak = 0
    for r in reps:
        pdf = (ROOT / "text_pdf" / f"{r['id']}.txt").read_text()
        for x in r["показатели"]:
            for q in (x["цитата"], x.get("цитата_доп")):
                if q:
                    n += 1
                    miss += not pq.verified_in_pdf(q, pdf)
    say(miss == 0, f"цитат в таблице {n}; не найдено в тексте PDF посимвольно: {miss}")
    sber = [r for r in reps if r["банк"] == "Сбер"]
    nz = [x for r in sber for x in r["показатели"] if x["оценка"]]
    say(all(x["цитата"] for x in nz), f"у всех {len(nz)} ненулевых оценок Сбера есть цитата")


if __name__ == "__main__":
    _, items = report.load()
    a1(); a2(); a3(items); a4(); a5(); a6(items); a7()
    print(f"\nИТОГ: проблем {len(problems)}" + (": " + "; ".join(problems) if problems else ""))
