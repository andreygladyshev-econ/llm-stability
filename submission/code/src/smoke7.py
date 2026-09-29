"""Проверка пробного ответа v7: разбирается, код выбирает ячейки периода. Код выхода 1 — очередь не запускать."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report  # noqa: E402
import score_rules  # noqa: E402

f = Path(__file__).resolve().parent.parent / "raw" / "qwen3.8-27b-mlx-4bit__v7_extract__t0.7-ex2-mp0.05-tx2__2024_q3__run1.json"
if not f.exists():
    sys.exit("нет ответа пробного прогона")
p = report.parse(json.loads(f.read_text())["content"])
if not p:
    sys.exit("ответ не разбирается")
p = report.apply_code_scores(p)
picked = [i for i in score_rules.NUMERIC if p[i]["period"]]
for i in ("net_profit", "eps", "roe", "cor", "cir", "retail_loans"):
    x = p[i]
    print(f"{i:14} ячейка={x['period'] or '—'} код={x['code_score']} модель={x['model_score']} | "
          f"q: {x.get('q_value')} / {x.get('q_change')} | y: {x.get('y_value')} / {x.get('y_change')}")
print(f"код выбрал ячейку у {len(picked)} из {len(score_rules.NUMERIC)} числовых показателей")
sys.exit(0 if len(picked) >= 10 else "код выбрал ячейку меньше чем у 10 показателей — схема не работает")
