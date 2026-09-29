"""Разбор облачных прогонов — после КАЖДОЙ модели, до решения «идём дальше».

По модели: сколько запросов и денег (факт против прогноза), разбираются ли ответы, обрывы, провайдер и точность
весов, детерминизм при T=0 (одинаковые запросы — одинаковые ответы?), хрупкие клетки боевого режима,
совпадение с эталоном (2026_q2, 2022_q4), устойчивость при T=0,7. Рядом — наша локальная Qwen для сравнения.

Запуск: .venv/bin/python src/cloud_report.py [модель]      без модели — сводка по всем прогнанным
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eocr  # noqa: E402
import night  # noqa: E402
import report  # noqa: E402
import silver_gold as sg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOCAL = "qwen3.8-27b-mlx@4bit"


def cfg(model, variant):
    return f"{model} | v7_extract | {variant}"


def model_summary(model, runs, items, gold):
    r = runs[runs.model == model]
    if r.empty:
        return None
    raw = [json.loads(p.read_text())["meta"] for p in (ROOT / "raw").glob(f"{night.safe(model)}__*.json")]
    cost = sum(m.get("cost_usd") or 0 for m in raw)
    prov = sorted({f"{m.get('provider')} ({m.get('model_quant') or '?'})" for m in raw if m.get("cloud")})
    out = {"модель": model, "ответов": len(r), "разобрано": f"{int(r.parse_ok.sum())}/{len(r)}",
           "обрывов": int(r.get("truncated", pd.Series(dtype=bool)).fillna(False).sum()) if "truncated" in r else 0,
           "потрачено": round(cost, 3), "провайдер": ", ".join(prov) or "локально"}
    # детерминизм: 5 (или 3) одинаковых запросов при T=0 — совпали ли ответы байт в байт
    det = r[r.variant.str.contains("-det")]
    out["детерминизм T=0"] = (f"{det.content_sha.nunique()} разных из {len(det)}" if len(det) else "—")
    # боевой режим: хрупкие клетки (перестановки расходятся) и мода против эталона
    b = items[items.config == cfg(model, "t0-shuf-ex2-or-tx2")]
    if not b.empty:
        fr = b.groupby(["report", "indicator"]).score.nunique()
        out["хрупких на отчёт"] = round(float((fr > 1).groupby(level=0).sum().mean()), 1)
        modal = b.groupby(["report", "indicator"]).score.agg(lambda s: s.mode().iloc[0])
        keys = [k for k in gold if k in modal.index]
        if keys:
            out["эталон"] = f"{sum(int(modal[k]) == gold[k][0] for k in keys)}/{len(keys)}"
    # сэмплирование T=0,7: те же метрики, что у локальных серий
    s = cfg(model, "t0.7-ex2-or-tx2")
    if not items[items.config == s].empty:
        m = eocr.metrics(eocr.arm_items(items, s, range(1, 20), 5))
        out.update({"альфа T=0,7": m["альфа"], "плав. числ.": m["плав. числ. (из 68)"],
                    "цитаты подтв.": m["цитаты подтв."], "выдуманы": m["выдуманы"]})
    return out


def main(only=None):
    runs, items = report.load()
    gold = {(r, i): v for r in sg.REPORTS for i, v in sg.read_gold(r).items()}
    cloud_models = sorted(runs[runs["cloud"].fillna(False).astype(bool)].model.unique()) if "cloud" in runs else []
    models = [only] if only else cloud_models
    rows = [x for x in (model_summary(m, runs, items, gold) for m in models) if x]
    if not rows:
        print("облачных ответов пока нет"); return
    # локальная точка отсчёта: наша Qwen 4 бита
    ref = {"модель": "НАША: Qwen 27B, 4 бита, ноутбук"}
    lb = items[items.config == f"{LOCAL} | v7_extract | t0-shuf-ex2-tx2"]
    fr = lb[lb.report.isin(["2026_q2", "2022_q4", "2024_q4", "2025_q3", "vtb_2026_q1", "tbank_2026_q2"])].groupby(["report", "indicator"]).score.nunique()
    ref["хрупких на отчёт"] = round(float((fr > 1).groupby(level=0).sum().mean()), 1)
    m = eocr.metrics(eocr.arm_items(items, f"{LOCAL} | v7_extract | t0.7-ex2-mp0.05-tx2", range(1, 20), 5))
    ref.update({"альфа T=0,7": m["альфа"], "плав. числ.": m["плав. числ. (из 68)"], "цитаты подтв.": m["цитаты подтв."],
                "выдуманы": m["выдуманы"], "детерминизм T=0": "1 из N (байт в байт)"})
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows + [ref]).fillna("").to_string(index=False))
    led = ROOT / "logs" / "cloud_ledger.jsonl"
    if led.exists():
        tot = sum(json.loads(x)["cost"] for x in led.read_text().splitlines())
        print(f"\nвсего потрачено облачным раннером: ${tot:.2f} (общий стоп $18, лимит ключа $20)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
