"""Очередь ночи 6 (длинная: Андрей сам запускает и ставит на паузу) — собирается после распознавания text_v2 (в неё попадают только отчёты с готовым v2).

Главный опыт — E-OCR: влияет ли чтение отчёта на устойчивость. Три серии в одном запуске, вперемешку по кругу
(урок ночи 5: на 3–5 прогонах альфа гуляет на ±0,05, серии из разных дней несравнимы):
  А. база v2_inplace на старом тексте (Tesseract)        — прогоны 6–15 серии min_p=0.05 (1–5 сделаны утром 19.09)
  Б. база v2_inplace на новом тексте (Qwen-зрение, v2)   — прогоны 1–10
  В. извлечение v5_extract на новом тексте              — прогоны 1–10
Предсказание записано до прогона: см. комментарий в очереди.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEV = ["2022_q4", "2026_q2", "2024_q3", "2023_q4"]
UNTOUCHED = ["2023_q1", "2023_q3", "2024_q1", "2024_q2", "2025_q1", "2025_q2", "2025_q4", "2026_q1"]
MODEL = "qwen3.8-27b-mlx@4bit"

HEAD = '''# Ночь 6 — собрана src/make_queue6.py после распознавания text_v2.
#
# E-OCR: влияет ли чтение отчёта на устойчивость. Три серии вперемешку в одном запуске, по 10 прогонов.
# ПРЕДСКАЗАНИЕ (записано 19.09 до прогона):
#   1. На новом тексте (Б) числовых плавающих клеток меньше, чем на старом (А), минимум на треть;
#   2. у 7 показателей-суждений (тон, внешние условия, технологии, прогноз, дивиденды, доля рынка,
#      качество портфеля) разница в пределах шума — они от чтения таблиц не зависят (встроенный контроль);
#   3. доля проверяемых цитат на новом тексте выше.
#   Если 1 и 2 выполнены — финальная таблица делается на text_v2. Если 1 не выполнено — чтение не главная
#   причина плавания, и это вывод для записки.
# Сравнение: альфа, плавающие клетки, смена знака — только по прогонам этой ночи.
'''


def job(spec, runs, text=None, mode=None, reports=DEV, temperature=0.7, min_p=0.05, model=MODEL, ctx=32768):
    lines = ["[[job]]", f'model = "{model}"', f'spec = "{spec}"',
             "reports = [" + ", ".join(f'"{r}"' for r in reports) + "]",
             f"runs = {runs}", f"temperature = {temperature}"]
    if min_p is not None:
        lines.append(f"min_p = {min_p}")
    if mode:
        lines.append(f'mode = "{mode}"')
    if text:
        lines.append(f'text = "{text}"')
    lines.append(f"ctx = {ctx}")
    return "\n".join(lines) + "\n"


def main():
    ready = set(texts.hashes("v2"))
    dev = [r for r in DEV if r in ready]
    if len(dev) < len(DEV):
        sys.exit(f"нет text_v2 для отладочных отчётов: {sorted(set(DEV) - ready)} — опыт E-OCR невозможен")
    out = [HEAD]
    out.append("# ══════ E-OCR: 10 кругов по три серии ══════\n")
    for r in range(1, 11):
        out.append(f"# круг {r}\n")
        out.append(job("v2_inplace", 5 + r))                          # А: старый текст, прогоны 6–15
        out.append(job("v2_inplace", r, text="v2"))                   # Б: новый текст
        out.append(job("v5_extract", r, text="v2", mode="extract"))   # В: извлечение на новом тексте
    all15 = sorted(ready)
    out.append("\n# ══════ Боевой режим T=0 на новом тексте — все отчёты с готовым v2 ══════\n")
    out.append(job("v2_inplace", 1, text="v2", reports=all15, temperature=0, min_p=None))
    untouched = [r for r in UNTOUCHED if r in ready]
    if "--stage1" in sys.argv:  # марафон: сначала только главный опыт, потом распознавание остальных отчётов
        return write(out, ready, untouched)
    if untouched:
        out.append("\n# ══════ H28: нетронутые кварталы на новом тексте, по 3 прогона ══════\n")
        out.append(job("v2_inplace", 3, text="v2", reports=untouched))
    out.append("\n# ══════ Другие семейства на новом тексте — для правильности без эталона (L12) ══════\n"
               "# Gemma: контекст 20480 — предел памяти; новый текст на 2–4% длиннее, возможны обрывы (раньше 2 из 12).\n"
               "# Если обрывается подряд — раннер сам пропустит. Ответы ~10 мин.\n")
    out.append(job("v2_inplace", 5, text="v2", model="google/gemma-4-31b-qat", ctx=20480))
    out.append(job("v2_inplace", 5, text="v2", model="t-pro-it-2.1", ctx=20480))
    if untouched:
        out.append("\n# ══════ H28: добор нетронутых кварталов до 5 прогонов ══════\n")
        out.append(job("v2_inplace", 5, text="v2", reports=untouched))
    write(out, ready, untouched)


def write(out, ready, untouched):
    path = ROOT / "queues" / "night6.toml"
    path.write_text("\n".join(out))
    print(f"очередь: {path} | отчётов с v2: {len(ready)} | нетронутых в запасе: {len(untouched)}"
          f"{' | только этап 1' if '--stage1' in sys.argv else ''}")


if __name__ == "__main__":
    main()
