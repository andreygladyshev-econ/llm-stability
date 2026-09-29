"""Опыт 2 (23.09): помогает ли размышление — та же модель, тот же набор, размышление выключено/минимально против включённого.
Запуск: .venv/bin/python src/think_compare.py <модель> [<серия-база> <серия-опыт>]"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_checks as fc  # noqa: E402
import report  # noqa: E402
import silver_gold as sg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def row(items, model, battle, det):
    v = fc.votes(items, fc.cfg(model, battle), fc.BATTLE)
    per = [sum(len(set(s)) > 1 for (r, _), s in v.items() if r == rep) for rep in fc.BATTLE]
    med = {k: statistics.median_low(s) for k, s in v.items()}
    gold = {(r, i): g[0] for r in fc.GOLD4 for i, g in sg.read_gold(r).items()}
    x = sum(med.get(k) == g for k, g in gold.items())
    d = fc.votes(items, fc.cfg(model, det), ["2026_q2"])
    metas = [json.loads(p.read_text())["meta"] for p in sorted((ROOT / "raw").glob(f"{model.replace('/', '-')}__v7_extract__{battle}__*.json"))]
    ok = [m for m in metas if m.get("parse_ok")]
    return {"хрупких/отчёт": round(statistics.mean(per), 2), "эталон": f"{x}/{len(gold)}",
            "повтор: клеток с разным баллом": sum(len(set(s)) > 1 for s in d) if not d.empty else "—",
            "без ответа": f"{len(metas) - len(ok)}/{len(metas)}",
            "сек/запрос": round(statistics.median(m["sec"] for m in metas)) if metas else "—",
            "$/запрос": round(statistics.mean(m.get("cost_usd") or 0 for m in metas), 4) if metas else "—",
            "размышление, ток": round(statistics.median(m.get("reasoning_tokens") or 0 for m in metas)) if metas else "—"}, per


if __name__ == "__main__":
    model = sys.argv[1]
    _, items = report.load()
    a, pa = row(items, model, "t0-shuf-ex2-or-tx2", "t0-ex2-det-tx2")
    b, pb = row(items, model, "t0-shuf-ex2-think-tx2", "t0-ex2-det-think-tx2")
    print(f"{model}\n{'':34}{'без/мин.':>12}{'включено':>12}")
    for k in a:
        print(f"  {k:32}{str(a[k]):>12}{str(b[k]):>12}")
    lo, hi = fc.boot_mean([y - x for x, y in zip(pa, pb)])
    print(f"  разница хрупких (вкл − выкл): {statistics.mean(pb) - statistics.mean(pa):+.2f}, 90% [{lo:+.2f}; {hi:+.2f}]")
