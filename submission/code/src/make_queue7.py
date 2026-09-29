"""Очередь 19→20.09 (Андрей: «с запасом до 14:00, остановлю сам около 11–12»).

Промежуточный итог E-OCR (19.09 21:15, по 3 прогона на серию, src/eocr.py): новый текст не уменьшил плавание
чисел в базе (14 клеток против 14), но цитаты точнее; извлечение на новом тексте плавает сильнее, чем на старом
(9 против 3). Разбор: половина плавающих числовых клеток — модель по-разному выбирает ПЕРИОД (квартал против
9 месяцев), хотя правило в спеке есть. Чистый текст показывает обе цифры рядом — выбор становится случайным.
Кандидат v7_extract: модель выписывает обе ячейки периода, выбирает код (score_rules.pick).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts  # noqa: E402
from make_queue6 import DEV, UNTOUCHED, job  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

HEAD = '''# Ночь 19→20.09 — собрана src/make_queue7.py.
#
# ГЛАВНОЕ: v7_extract (две ячейки периода, выбор кодом) против v5_extract на том же новом тексте, по 10 прогонов.
# ПРЕДСКАЗАНИЕ (записано 19.09 21:10 до прогона):
#   1. у v7 плавающих числовых клеток меньше, чем у v5 на новом тексте, минимум вдвое (v5: 9 из 68 на 3 прогонах);
#   2. плавание из-за выбора периода (eps, net_profit, roe, cor, cir) у v7 исчезает;
#   3. доля подтверждённых цитат не ниже, чем у v5 (0,913).
#   Если 1 не выполнено — модель путает и раскладку по ячейкам; тогда выбор периода надо делать до модели
#   (подать ей только одну колонку), и это вывод для записки.
# ДОБОР E-OCR: база старый/новый текст и извлечение на старом — до 6 прогонов на отчёт (было предрегистрировано 10;
#   сокращено, потому что главный вопрос сместился к выбору периода; отклонение записано в ночь_6.md).
# Затем T=0 (боевой режим) на всех 15 отчётах: v7 и база на новом тексте. Запас: нетронутые кварталы v7, Gemma.
'''


def main():
    ready = set(texts.hashes("v2"))
    assert set(DEV) <= ready, "нет text_v2 для отладочных отчётов"
    out = [HEAD, "# ══════ v7 против v5 (извлечение, новый текст) + добор E-OCR — вперемешку ══════\n"]
    for r in range(1, 11):
        out.append(f"# круг {r}\n")
        out.append(job("v7_extract", r, text="v2", mode="extract2"))           # кандидат, прогоны 1–10
        if r <= 7:
            out.append(job("v5_extract", r + 3, text="v2", mode="extract"))    # контроль, прогоны 4–10
        if r <= 3:
            out.append(job("v2_inplace", 8 + r))                               # A: база, старый текст → до 11
            out.append(job("v2_inplace", 3 + r, text="v2"))                    # B: база, новый текст → до 6
            out.append(job("v5_extract", 3 + r, mode="extract"))               # D: извлечение, старый текст → до 6
    all15 = sorted(ready)
    out.append("\n# ══════ Боевой режим T=0, все 15 отчётов на новом тексте ══════\n")
    out.append(job("v7_extract", 1, text="v2", mode="extract2", reports=all15, temperature=0, min_p=None))
    out.append(job("v2_inplace", 1, text="v2", reports=all15, temperature=0, min_p=None))
    out.append("\n# ══════ Запас: H28 — нетронутые кварталы, v7 по 3 прогона ══════\n")
    out.append(job("v7_extract", 3, text="v2", mode="extract2", reports=UNTOUCHED))
    out.append("\n# ══════ Глубокий запас: Gemma на новом тексте (правильность без эталона, L12) ══════\n")
    out.append(job("v2_inplace", 5, text="v2", model="google/gemma-4-31b-qat", ctx=20480))
    path = ROOT / "queues" / "night7.toml"
    path.write_text("\n".join(out))
    print(f"очередь: {path}")


if __name__ == "__main__":
    main()
