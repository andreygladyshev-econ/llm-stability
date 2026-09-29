"""Бесплатная проверка 22.09: берёт ли модель базу сравнения из столбца «прошлый квартал» вместо «год назад».

Найдено на T-Банке (v10, перестановка 1): в строке таблицы «2кв26 2кв25 изм. 1кв26 изм. 6м26 6м25 изм.» модель
взяла базой 1кв26 → прибыль +13% вместо −15%. Эвристика: в цитате-строке база стоит ПОСЛЕ значения и между ними
уже был столбец изменения (%, п.п.) — значит, база перескочила через годовое сравнение. Ложные срабатывания
возможны у отчётов, где год назад стоит после квартального сравнения, — такие смотреть глазами.
Запуск: .venv/bin/python src/qoq_check.py
"""
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM = re.compile(r"[-+−]?\d+(?:[  ]\d{3})*(?:,\d+)?\s*(?:%|п\.\s?п\.|пп|б\.\s?п\.)?")
CORE = re.compile(r"\d+(?:[  ]\d{3})*(?:,\d+)?")


def core(s):
    m = CORE.search(s or "")
    return m.group(0).replace(" ", " ") if m else None


def suspicious(it):
    """True, если база сравнения квартала взята через столбец изменения (подозрение на «к прошлому кварталу»)."""
    v, b = core(it.get("q_value")), core(it.get("q_base"))
    toks = [m.group(0).strip() for m in NUM.finditer(it.get("q_quote") or "")]
    nums = [core(x) for x in toks]
    if not (v and b and v in nums and b in nums):
        return None
    iv, ib = nums.index(v), nums.index(b)
    return ib > iv + 1 and any(("%" in x or "п" in x) for x in toks[iv + 1:ib])


def main():
    hits, tot, ex = collections.Counter(), collections.Counter(), []
    for f in (ROOT / "raw").glob("*.json"):
        d = json.loads(f.read_text()); m = d["meta"]
        if m.get("mode") != "extract2" or not d.get("content"):
            continue
        try:
            c = json.loads(d["content"])
        except ValueError:
            continue
        bank = "сбер" if m["report"][0].isdigit() else m["report"].split("_")[0]
        for ind, it in (c.items() if isinstance(c, dict) else []):
            s = suspicious(it) if isinstance(it, dict) else None
            if s is None:
                continue
            tot[(m["spec_version"], bank)] += 1
            if s:
                hits[(m["spec_version"], bank)] += 1
                ex.append((m["name"].split("__", 1)[1], ind, (it.get("q_quote") or "")[:100], it["q_value"], it["q_base"]))
    for k in sorted(tot):
        print(f"{k[0]:12} {k[1]:6} подозрительных {hits[k]} из {tot[k]} ячеек с базой в строке")
    for e in ex:
        print("  ", e)


if __name__ == "__main__":
    assert suspicious({"q_value": "39,5", "q_base": "35,0", "q_quote": "Чистая прибыль 39,5 46,7 -15% 35,0 13%"})
    assert not suspicious({"q_value": "39,5", "q_base": "46,7", "q_quote": "Чистая прибыль 39,5 46,7 -15% 35,0 13%"})
    main()
