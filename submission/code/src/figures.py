"""Графики для записки и приложений (23.09, переработано 29.09). Все данные считаются здесь же из готовых ответов.

Рисунки 1–3 входят в записку, 4–10 — в приложения; номер файла совпадает с номером рисунка в тексте.
Оформление — по правилам dataviz: палитра проверена валидатором (две категории: синий и оранжевый; упорядоченная
шкала синего из трёх шагов), тонкие отметки, подписи текстовым цветом, одна ось, сетка — сплошные волосяные линии.
Шрифт: Helvetica Neue на macOS, Arial или метрически совместимый Liberation Sans в других системах; от шрифта
зависят только пиксели рисунка, не данные. Запуск: .venv/bin/python src/figures.py [номера рисунков]
"""
import collections
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstain  # noqa: E402
import eocr  # noqa: E402
import final_checks as fc  # noqa: E402
import final_table as ft  # noqa: E402
import report  # noqa: E402
import score_rules as sr  # noqa: E402
import silver_gold as sg  # noqa: E402
import texts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = texts.SUBMIT / "figures"
M = fc.M
# палитра (светлая тема, справочный экземпляр dataviz)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#f0efec", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95"]          # 0…4 — последовательная шкала (синяя)
DIV = {-2: "#b3302f", -1: "#ec9a99", 0: "#f0efec", 1: "#86b6ef", 2: "#1c5cab"}  # расходящаяся: красный ↔ синий
GRAY = "#b5b3ac"
LIGHT, DARK = "#86b6ef", "#184f95"        # упорядоченная шкала синего: шаги 250 и 600 (валидатор, --ordinal)
try:                                                   # десятичная запятая на осях
    import locale
    locale.setlocale(locale.LC_NUMERIC, "ru_RU.UTF-8")
    plt.rcParams["axes.formatter.use_locale"] = True
except Exception:
    pass
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 9.5, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2, "axes.facecolor": SURF,
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "xtick.major.size": 0, "ytick.major.size": 0, "axes.titlesize": 10.5,
})


def header(fig, title, subtitle=None, source=None):
    """Заголовок, подзаголовок (переносится по ширине фигуры), примечание снизу. → доля высоты, где кончается шапка."""
    import textwrap
    w_in, h_in = fig.get_size_inches()
    line = 1 / (h_in * 72 / 13.5)                       # высота строки 9,5 pt в долях фигуры
    fig.text(0.012, 1 - 0.18 / h_in, title, ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK)
    y = 1 - 0.18 / h_in - 0.27 / h_in
    if subtitle:
        lines = textwrap.wrap(subtitle, width=int(w_in * 13.2))
        fig.text(0.012, y, "\n".join(lines), ha="left", va="top", fontsize=9.5, color=INK2, linespacing=1.35)
        y -= len(lines) * line * 1.35
    if source:
        fig.text(0.012, 0.06 / h_in, source, ha="left", va="bottom", fontsize=7.8, color=MUTED)
    return y


def ru(x, fmt):
    """Число по-русски: десятичная запятая, настоящий минус."""
    return format(x, fmt).replace(".", ",").replace("-", "−")


def sci(x):
    """Малое p по-русски: 4,9e-04 → «4,9·10⁻⁴»."""
    m, e = f"{x:.1e}".replace(".", ",").split("e")
    return f"{m}·10" + str(int(e)).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


def hgrid(ax, axis="x"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def hbar(ax, y, value, color, thick=0.52, r_px=4):
    """Горизонтальный столбец одним контуром: 4-пиксельное скругление на конце, прямой угол у нулевой линии."""
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    if value <= 0:
        return
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent()
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    dx = (x1 - x0) / bb.width; dy = abs(y1 - y0) / bb.height
    rx = min(r_px * dx, value / 2); ry = min(r_px * dy, thick / 2)
    b, t, w = y - thick / 2, y + thick / 2, value
    verts = [(0, b), (w - rx, b), (w, b), (w, b + ry), (w, t - ry), (w, t), (w - rx, t), (0, t), (0, b)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, linewidth=0, zorder=3))


def hbar_from(ax, y, x0, x1, color, thick=0.52, r_px=4):
    """Плавающий горизонтальный отрезок каскадной диаграммы: скругление 4 px на правом конце."""
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    w = x1 - x0
    if w <= 0:
        return
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent()
    xa, xb = ax.get_xlim(); ya, yb = ax.get_ylim()
    dx = (xb - xa) / bb.width; dy = abs(yb - ya) / bb.height
    rx = min(r_px * dx, w / 2); ry = min(r_px * dy, thick / 2)
    b, t = y - thick / 2, y + thick / 2
    verts = [(x0, b), (x1 - rx, b), (x1, b), (x1, b + ry), (x1, t - ry), (x1, t), (x1 - rx, t), (x0, t), (x0, b)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, linewidth=0, zorder=3))


def comma_ticks(fig):
    """Десятичная запятая и настоящий минус на числовых осях — без зависимости от системной локали."""
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if not isinstance(axis.get_major_formatter(), ScalarFormatter):
                continue                                # подписи, заданные вручную, не трогаем
            locs = [x for x in axis.get_ticklocs() if abs(x) < 1e6]
            nd = max((len(f"{x:g}".split(".")[1]) if "." in f"{x:g}" else 0) for x in locs) if locs else 0
            axis.set_major_formatter(FuncFormatter(lambda v, _p, nd=nd: f"{v:.{nd}f}".replace(".", ",").replace("-", "−")))


def save(fig, name):
    comma_ticks(fig)
    fig.canvas.draw()                                   # проверка: ни один текст не вылезает за край рисунка
    for t in fig.texts + [t for ax in fig.axes for t in ax.texts]:
        bb = t.get_window_extent()
        assert bb.x0 >= -1 and bb.x1 <= fig.bbox.width + 1, f"{name}: текст за краем — «{t.get_text()[:50]}»"
    for ax in fig.axes:                                  # подписи осей и всё нарисованное внутри области графика
        bb = ax.get_tightbbox(fig.canvas.get_renderer())
        assert bb.x0 >= -1 and bb.x1 <= fig.bbox.width + 1, f"{name}: подпись оси за краем рисунка"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("  ✓", name)


# ─── данные ──────────────────────────────────────────────────────────────────────────────────────────────────
def ladder_data(items):
    steps = [("Исходная спецификация заказчика", f"{M} | v1_baseline | t0.7", "T=0,7"),
             ("Правила переписаны в тексте спецификации", f"{M} | v2_inplace | t0.7-mp0.05", "T=0,7"),
             ("+ новое распознавание отчётов", f"{M} | v2_inplace | t0.7-mp0.05-tx2", "T=0,7"),
             ("+ оценки 17 числовых показателей считает программа", f"{M} | v5_extract | t0.7-ex-mp0.05-tx2", "T=0,7"),
             ("+ период выбирает программа", f"{M} | v7_extract | t0.7-ex2-mp0.05-tx2", "T=0,7"),
             ("+ температура 0,3", f"{M} | v7_extract | t0.3-ex2-mp0.05-tx2", "T=0,3"),
             ("Итоговый режим: T=0, 5 перестановок порядка", f"{M} | v7_extract | t0-shuf-ex2-tx2", "T=0")]
    rows = []
    for name, cfg, t in steps:
        g = eocr.arm_items(items, cfg, range(1, 16), 5)
        v = g.groupby(["report", "indicator"]).score.apply(list)
        m = eocr.metrics(g)
        rows.append((name, 100 * sum(len(set(s)) == 1 for s in v) / len(v), 100 * m["выдуманы"], t))
    return rows


def model_rows(items):
    import deep_analysis as da
    gold = da.gold_all()
    out = []
    for name, (w, params, active, arch) in da.MODELS.items():
        v, med = da.medians(items, name)
        if v.empty:
            continue
        per = [sum(len(set(s)) > 1 for (r, _), s in v.items() if r == rep) for rep in fc.BATTLE]
        keys = [k for k in gold if k[0] in fc.GOLD4 and k in med]
        ok = sum(med[k] == gold[k][0] for k in keys)
        ms = da.meta(name)
        cost = statistics.mean(m.get("cost_usd") or 0 for m in ms) * 5 if ms else 0
        out.append({"name": name, "weights": w, "params": params, "active": active, "arch": arch,
                    "fragile": statistics.mean(per), "gold": 100 * ok / len(keys), "cost": cost})
    return out


SHORT = {"ноутбук Qwen 27B 4 бита": "Ноутбук: Qwen 27B, 4 бита", "qwen/qwen3.8-27b": "Qwen 27B (облако, fp8)",
         "google/gemma-4-31b-it": "Gemma 4 31B", "qwen/qwen3.6-35b-a3b": "Qwen 3.6 35B-A3B",
         "nvidia/nemotron-3-nano-30b-a3b": "Nemotron Nano 30B", "openai/gpt-oss-120b": "GPT-OSS 120B",
         "nvidia/nemotron-3-super-120b-a12b": "Nemotron Super 120B", "qwen/qwen3-235b-a22b-2507": "Qwen3 235B",
         "z-ai/glm-5.3-flash": "GLM-5.3 Flash", "xiaomi/mimo-v2.6-flash": "MiMo V2.6 Flash",
         "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash", "z-ai/glm-5.3": "GLM-5.3",
         "openai/gpt-5.6-sol": "GPT-5.6 Sol", "openai/gpt-6-sol": "GPT-6 Sol", "openai/gpt-6-luna": "GPT-6 Luna"}


# ─── рисунки ─────────────────────────────────────────────────────────────────────────────────────────────────
IND = {"net_profit": "Чистая прибыль", "eps": "Прибыль на акцию", "roe": "Рентабельность капитала",
       "guidance": "Прогноз менеджмента", "nii": "Процентные доходы", "fee_income": "Комиссионные доходы",
       "nim": "Процентная маржа", "opex": "Операционные расходы", "cir": "Расходы / доходы", "cor": "Стоимость риска",
       "provisions": "Расходы на резервы", "portfolio_quality": "Качество портфеля",
       "corporate_loans": "Корпоративные кредиты", "retail_loans": "Розничные кредиты",
       "customer_funds": "Средства клиентов", "market_share": "Доля рынка", "capital_adequacy": "Достаточность капитала",
       "dividends": "Дивиденды", "book_value_per_share": "Балансовая стоимость акции",
       "active_clients": "Активные клиенты", "digital_metrics": "Цифровые метрики",
       "tech_development": "Технологии", "external_conditions": "Внешние условия", "ceo_tone": "Тональность председателя"}
QLAB = {"q1": "1 кв.", "q2": "2 кв.", "q3": "3 кв.", "q4": "4 кв."}


def order():
    import csv
    return [r["id"] for r in csv.DictReader(open(sg.GOLD / "2026_q2_CLAUDE.csv"))]


def axes_at(fig, left, bottom, right, top):
    return fig.add_axes([left, bottom, right - left, top - bottom])


def swatch_row(fig, y, items, x0=0.012, box=None):
    """Строка легенды в шапке: цветной квадрат + подпись текстовым цветом."""
    w_in, h_in = fig.get_size_inches()
    x = x0
    for color, label, edge in items:
        fig.patches.append(Rectangle((x, y - 0.055 / h_in), 0.14 / w_in, 0.14 / h_in, transform=fig.transFigure,
                                     facecolor=color if color else "none", edgecolor=edge or color, linewidth=1.1))
        t = fig.text(x + 0.2 / w_in, y + 0.015 / h_in, label, fontsize=8.6, color=INK2, va="center")
        fig.canvas.draw()
        x += (t.get_window_extent().width + 0.42 * fig.dpi) / (w_in * fig.dpi)


def f1_waterfall(items):
    """Рисунок 1: из чего сложился прирост устойчивости (каскад) и доля выдуманных цитат на тех же шагах."""
    # значения округляются до десятых сразу: приращения на рисунке складываются из тех же чисел, что в подписях
    base, text, ocr, code, period, _t03, final = [(n, round(u, 1), round(f, 1), t) for n, u, f, t in ladder_data(items)]
    rows = [("Спецификация заказчика", 0, base[1], GRAY, base[2], "total"),
            ("Правила переписаны в тексте спецификации", base[1], text[1], LIGHT, text[2], "step"),
            ("Сканы распознаны заново моделью со зрением", text[1], ocr[1], LIGHT, ocr[2], "step"),
            ("17 числовых оценок считает программа", ocr[1], code[1], BLUE, code[2], "step"),
            ("Период сравнения выбирает программа", code[1], period[1], BLUE, period[2], "step"),
            ("Итог при той же случайности, T=0,7", 0, period[1], DARK, period[2], "total"),
            ("Итоговый режим: T=0, 5 порядков показателей", 0, final[1], DARK, final[2], "total")]
    prog = (code[1] - ocr[1]) + (period[1] - code[1])
    fig = plt.figure(figsize=(10.4, 5.0))
    top = header(fig, f"Устойчивость выросла с {ru(base[1], '.1f')}% до {ru(period[1], '.1f')}% при той же случайности модели",
                 "Доля показателей, совпавших во всех пяти прогонах: четыре отчёта Сбера, использованные при настройке, "
                 f"модель Qwen 3.8 27B на ноутбуке. Два шага, на которых решение перешло к программе, дали "
                 f"{ru(prog, '.1f')} п.п.; правки текста спецификации вместе — {ru(text[1] - base[1], '.1f')} п.п.",
                 "Разброс замера на 5 прогонах ±5,6 п.п.: отдельные правки текста его не превышают и показаны одним шагом. "
                 "Шаги 1–2 на старом распознавании. Расчёт: code/src/figures.py")
    swatch_row(fig, top - 0.035, [(GRAY, "исходная спецификация", None), (LIGHT, "правки текста и распознавания", None),
                                  (BLUE, "решение передано программе", None), (DARK, "итог", None)])
    a1 = axes_at(fig, 0.345, 0.12, 0.765, top - 0.14)
    a2 = axes_at(fig, 0.815, 0.12, 0.985, top - 0.14)
    n = len(rows)
    ys = [n - 1 - k + (0.35 if k == n - 1 else 0) for k in range(n)]
    ys[-1] = -0.45                                                  # итоговый режим — ниже разделителя
    for ax, lim in ((a1, (0, 100)), (a2, (0, 10))):
        ax.set_xlim(*lim); ax.set_ylim(-1.05, n - 0.4)
        hgrid(ax); ax.spines["left"].set_visible(False)
    for (name, x0, x1, col, fab, kind), y in zip(rows, ys):
        hbar_from(a1, y, x0, x1, col)
        if kind == "step":
            a1.plot([x0, x0], [y + 0.26, y + 1 - 0.26], color=AXIS, linewidth=0.8, zorder=2)   # связка с шагом выше
            a1.text(x1 + 1.2, y, "+" + ru(x1 - x0, ".1f"), va="center", fontsize=9, color=INK2)
        else:
            a1.text(x1 + 1.2, y, ru(x1, ".1f") + "%", va="center", fontsize=9.2, color=INK,
                    fontweight="bold" if col == DARK else None)
        hbar(a2, y, fab, ORANGE)
        a2.text(fab + 0.25, y, ru(fab, ".1f") + "%", va="center", fontsize=9, color=INK)
    for ax in (a1, a2):
        ax.axhline(0.28, color=AXIS, linewidth=0.8, zorder=1)
    a1.text(0.5, 0.18, "другой способ повтора: при T=0 ответ совпадает побайтно, меняется порядок показателей",
            fontsize=7.4, color=MUTED, va="top")
    a1.set_yticks(ys); a1.set_yticklabels([r[0] for r in rows], fontsize=9)
    a2.set_yticks([])
    a1.set_xticks([0, 25, 50, 75, 100]); a2.set_xticks([0, 5, 10])
    a1.set_title("Показателей, совпавших во всех 5 прогонах, %", loc="left", color=INK2, fontsize=9.5, pad=8)
    a2.set_title("Выдуманных цитат, %", loc="left", color=INK2, fontsize=9.5, pad=8)
    save(fig, "f1_прирост_устойчивости")


def f3_leaks(items):
    """Рисунок 3: где оценки расходятся между прогонами и насколько чаще там ошибки."""
    ords = order()
    v = fc.votes(items, fc.cfg(M, "t0-shuf-ex2-tx2"), ft.SBER, range(1, 6))
    cnt = {i: sum(len(set(v[(r, i)])) > 1 for r in ft.SBER) for i in ords}
    judg = set(sr.JUDGMENT)
    rows = sorted([(i, c) for i, c in cnt.items() if c], key=lambda t: (t[1], t[0] not in judg, IND[t[0]]))
    sj = sum(cnt[i] for i in judg) / (len(judg) * len(ft.SBER))
    sn = sum(cnt[i] for i in ords if i not in judg) / ((len(ords) - len(judg)) * len(ft.SBER))
    gold = {(r, i): g[0] for r in sg.REPORTS for i, g in sg.read_gold(r).items()}

    def err(var, runs):
        vv = fc.votes(items, fc.cfg(M, var), sg.REPORTS, runs)
        top_ = {k: collections.Counter(vv[k]).most_common(1)[0][0] for k in gold}
        fr = [k for k in gold if len(set(vv[k])) > 1]
        st = [k for k in gold if len(set(vv[k])) == 1]
        e = lambda ks: sum(top_[k] != gold[k] for k in ks)
        return (e(fr), len(fr)), (e(st), len(st))
    hot = err("t0.7-ex2-mp0.05-tx2", range(1, 11))
    prod = err("t0-shuf-ex2-tx2", range(1, 6))
    fig = plt.figure(figsize=(10.4, 5.6))
    top = header(fig, "Оценки расходятся там, где нужно суждение, и именно там чаще ошибки",
                 f"Слева — 15 отчётов Сбера в итоговом режиме: в показателях, которые оценивает модель, хрупких клеток "
                 f"{sj:.0%}, в показателях, которые считает программа, {sn:.0%}. Справа — 5 отчётов с эталонной "
                 f"разметкой: ошибки собраны в хрупких клетках, поэтому сомнительные клетки находятся без эталона, "
                 "одними повторными прогонами",
                 "Хрупкая клетка — оценки прогонов разошлись. Ошибка — самая частая оценка прогонов не равна эталонной. "
                 "Расчёт: code/src/figures.py, code/src/deep_analysis2.py (блок 24)")
    swatch_row(fig, top - 0.035, [(ORANGE, "оценивает модель (суждение)", None), (BLUE, "считает программа", None)])
    a1 = axes_at(fig, 0.215, 0.10, 0.575, top - 0.13)
    a2 = axes_at(fig, 0.715, 0.10, 0.985, top - 0.13)
    hgrid(a1); a1.spines["left"].set_visible(False)
    a1.set_xlim(0, 5); a1.set_ylim(-0.7, len(rows) - 0.3)
    for y, (i, c) in enumerate(rows):
        hbar(a1, y, c, ORANGE if i in judg else BLUE, thick=0.6)
        a1.text(c + 0.08, y, str(c), va="center", fontsize=8.6, color=INK)
    a1.set_yticks(range(len(rows))); a1.set_yticklabels([IND[i] for i, _ in rows], fontsize=8.6)
    a1.set_xticks(range(0, 6))
    a1.set_title("В скольких отчётах из 15 оценка разошлась", loc="left", color=INK2, fontsize=9.5, pad=8)
    hgrid(a2); a2.spines["left"].set_visible(False)
    a2.set_xlim(0, 62); a2.set_ylim(-0.7, 4.3)
    pos = [(3.6, hot[0], "хрупкие", "#52514e"), (2.8, hot[1], "устойчивые", "#c3c2b7"),
           (1.0, prod[0], "хрупкие", "#52514e"), (0.2, prod[1], "устойчивые", "#c3c2b7")]
    for y, (e, n_), lab, col in pos:
        hbar(a2, y, 100 * e / n_, col, thick=0.6)
        a2.text(100 * e / n_ + 1, y, f"{ru(100 * e / n_, '.0f' if e / n_ > 0.1 else '.1f')}% ({e} из {n_})", va="center",
                fontsize=8.6, color=INK)
    a2.set_yticks([p[0] for p in pos]); a2.set_yticklabels([p[2] for p in pos], fontsize=8.6)
    a2.text(0, 4.15, "10 прогонов при T=0,7", fontsize=8.6, color=INK2, fontweight="bold", va="center")
    a2.text(0, 1.55, "итоговый режим, 5 прогонов", fontsize=8.6, color=INK2, fontweight="bold", va="center")
    a2.set_xticks([0, 20, 40, 60])
    a2.set_title("Доля оценок, не совпавших с эталоном, %", loc="left", color=INK2, fontsize=9.5, pad=8)
    save(fig, "f3_где_расходятся_оценки")


def f5_cells(items):
    dev = fc.DEV
    anon = fc.votes(items, fc.cfg(M, "t0-shuf-ex2-tx2"), ["anon_" + r for r in dev])
    anon.index = anon.index.set_levels([anon.index.levels[0].str.replace("anon_", "")] + list(anon.index.levels[1:]))
    sets = [fc.votes(items, fc.cfg(M, "t0.7-ex2-mp0.05-tx2"), dev, range(1, 11)),
            fc.votes(items, fc.cfg(M, "t0.3-ex2-mp0.05-tx2"), dev, range(1, 10)),
            fc.votes(items, fc.cfg(M, "t0-shuf-ex2-tx2"), dev, range(1, 6)), anon]
    import deep_analysis2 as da2
    ords = order()
    reps = ["2022_q4", "2023_q4", "2024_q3", "2026_q2"]
    cnt = {(r, i): sum(len(set(v[(r, i)])) > 1 for v in sets if (r, i) in v.index) for r in reps for i in ords}
    flags = [da2.fragile_flags(v) for v in sets]
    keys = [k for k in flags[0] if all(k in f for f in flags)]
    N, ref = len(keys), {k for k in keys if flags[0][k]}
    over, exp_, pv, ps = [], [], [], []
    for f in flags[1:]:
        s_ = {k for k in keys if f[k]}
        over.append(f"{len(s_ & ref)} из {len(s_)}"); exp_.append(len(s_) * len(ref) / N)
        pv.append(fc.hypergeom_sf(len(s_ & ref), N, len(ref), len(s_))); ps.append(da2.stratified_overlap(flags[0], f, keys)[2])
    never = sum(c == 0 for c in cnt.values()) / len(cnt)
    fig = plt.figure(figsize=(10.4, 5.0))
    top = header(fig, "Одни и те же клетки нестабильны при любом способе возмущения",
                 "Четыре отчёта, использованные при настройке, × 24 показателя. Способы возмущения: случайность генерации "
                 f"при T=0,7 и T=0,3, другой порядок показателей при T=0 и текст без названия банка. {never:.0%} клеток не "
                 "изменились ни при одном способе",
                 f"Совпадения с хрупкими клетками при T=0,7: {', '.join(over)} при случайном ожидании {ru(min(exp_), '.1f')}–"
                 f"{ru(max(exp_), '.1f')} (p от {sci(min(pv))} до {sci(max(pv))}); с учётом трудности показателей "
                 f"p ≤ {ru(max(ps), '.3f')}. Расчёт: code/src/figures.py")
    swatch_row(fig, top - 0.035, [(SEQ[0], "ни при одном", AXIS), (SEQ[1], "при 1 способе", None),
                                  (SEQ[2], "при 2", None), (SEQ[3], "при 3", None), (SEQ[4], "при всех 4", None)])
    ax = axes_at(fig, 0.085, 0.33, 0.985, top - 0.10)
    for x, i in enumerate(ords):
        for y, r in enumerate(reps[::-1]):
            c = cnt[(r, i)]
            ax.add_patch(Rectangle((x + 0.04, y + 0.06), 0.92, 0.88, facecolor=SEQ[c], edgecolor=SURF, linewidth=1.5))
            if c:
                ax.text(x + 0.5, y + 0.5, str(c), ha="center", va="center", fontsize=8.5,
                        color="white" if c >= 3 else INK, fontweight="bold")
    ax.set_xlim(0, len(ords)); ax.set_ylim(0, len(reps))
    ax.set_xticks([x + 0.5 for x in range(len(ords))])
    ax.set_xticklabels([IND[i] for i in ords], rotation=55, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticks([y + 0.5 for y in range(len(reps))])
    ax.set_yticklabels([QLAB[r[5:]] + " " + r[:4] for r in reps[::-1]], fontsize=8.8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "f5_хрупкие_клетки")


def f6_models(items):
    rows = model_rows(items)
    col = {"открытые": BLUE, "закрытые": ORANGE}
    fig = plt.figure(figsize=(10.6, 5.2))
    top = header(fig, "Сильные модели дают близкие результаты независимо от размера и цены",
                 "Каждая точка соответствует модели, работающей по одной и той же схеме (спецификация v7, 6 отчётов × 5 "
                 "перестановок). По горизонтали совпадение с эталоном на 4 размеченных отчётах, по вертикали число хрупких "
                 "клеток на отчёт. Модели OpenAI не принимают параметр температуры",
                 "Модели Claude и Grok не показаны: эталон размечен моделью Claude, у Opus и Grok сокращённая схема. "
                 "Расчёт: code/src/figures.py")
    a1 = axes_at(fig, 0.065, 0.13, 0.43, top - 0.06)
    a2 = axes_at(fig, 0.50, 0.13, 0.985, top - 0.06)
    X2, Y2 = (89, 100.9), (4.15, 1.15)
    for ax in (a1, a2):
        hgrid(ax, "both")
        ax.set_xlabel("Совпадение с эталоном, % из 96 клеток", fontsize=8.8)
    for r in rows:
        loc = r["name"].startswith("ноутбук")
        c = AQUA if loc else col[r["weights"]]
        for ax, zoom in ((a1, False), (a2, True)):
            if zoom and not (X2[0] <= r["gold"] <= X2[1] and Y2[1] <= r["fragile"] <= Y2[0]):
                continue
            ax.scatter(r["gold"], r["fragile"], s=110 if loc else 64, color=c, edgecolor=SURF, linewidth=2, zorder=4)
    a1.set_xlim(55, 101.5); a1.set_ylim(17.5, 0)
    a2.set_xlim(*X2); a2.set_ylim(*Y2)
    a1.set_ylabel("Хрупких клеток на отчёт (меньше — устойчивее)", fontsize=8.8)
    a1.add_patch(Rectangle((X2[0], Y2[1]), X2[1] - X2[0], Y2[0] - Y2[1], fill=False, edgecolor=MUTED, linewidth=0.9, zorder=2))
    a1.text(X2[0] - 0.4, Y2[0] + 0.5, "увеличено справа", fontsize=7.8, color=MUTED, ha="right", va="top")
    for r in rows:
        n = SHORT[r["name"]]
        if r["fragile"] > 4.2:
            a1.text(r["gold"] + 1.0, r["fragile"], n, fontsize=8.2, color=INK2, va="center")
    pos = {"DeepSeek V4.1 Flash": (-0.22, 0, "right"), "GPT-5.6 Sol": (0, -0.15, "center"),
           "GPT-6 Sol": (0, -0.15, "center"), "GLM-5.3": (0.22, 0, "left"),
           "Ноутбук: Qwen 27B, 4 бита": (0.25, 0, "left"), "Gemma 4 31B": (-0.22, 0, "right"),
           "Qwen 27B (облако, fp8)": (0.22, 0, "left"), "Qwen3 235B": (0.22, 0, "left"),
           "GPT-6 Luna": (0, -0.15, "center"), "GLM-5.3 Flash": (0.22, 0, "left"), "MiMo V2.6 Flash": (0.22, 0, "left")}
    for r in rows:
        n = SHORT[r["name"]]
        if n in pos and X2[0] <= r["gold"] <= X2[1] and Y2[1] <= r["fragile"] <= Y2[0]:
            dx, dy, ha = pos[n]
            loc = n.startswith("Ноутбук")
            a2.text(r["gold"] + dx, r["fragile"] + dy, "Ноутбук: Qwen 27B" if loc else n, fontsize=8.2,
                    color=INK if loc else INK2, ha=ha, va="center", fontweight="bold" if loc else None)
    for c, lab in ((BLUE, "открытые веса, облако"), (ORANGE, "закрытые веса, облако"), (AQUA, "ноутбук, открытые веса")):
        a2.scatter([], [], s=50, color=c, edgecolor=SURF, linewidth=2, label=lab)
    a2.legend(loc="lower left", frameon=False, fontsize=8.2, handletextpad=0.2, borderaxespad=0.3)
    save(fig, "f6_карта_моделей")


def f7_repeat(items):
    rows = []
    for m in ["deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3", "z-ai/glm-5.3-flash", "xiaomi/mimo-v2.6-flash",
              "google/gemma-4-31b-it", "qwen/qwen3.8-27b", "qwen/qwen3-235b-a22b-2507", "qwen/qwen3.6-35b-a3b",
              "nvidia/nemotron-3-super-120b-a12b", "openai/gpt-oss-120b", "nvidia/nemotron-3-nano-30b-a3b",
              "openai/gpt-5.6-sol", "openai/gpt-6-sol", "openai/gpt-6-luna"]:
        v = fc.votes(items, fc.cfg(m, "t0-ex2-det-tx2"), ["2026_q2"])
        if not v.empty:
            rows.append((SHORT[m], sum(len(set(s)) > 1 for s in v)))
    rows.append(("Ноутбук: Qwen 27B, 4 бита", 0))
    rows.sort(key=lambda t: (t[1], t[0]))
    fig = plt.figure(figsize=(8.8, 5.6))
    top = header(fig, "Облачные модели не повторяют ответ, но у сильных оценки почти не меняются",
                 "Один отчёт (2 кв. 2026), 5 одинаковых запросов к одному провайдеру; T=0 и seed=1 там, где провайдер их "
                 "принимает (модели OpenAI не принимают температуру, для MiMo и GLM-5.3 Flash seed не передавался). Текст "
                 "ответа в облаке различался у всех моделей, на ноутбуке совпадал побайтно, в том числе через сутки",
                 "Сервер объединяет запросы разных пользователей в пакеты, а от размера пакета зависит порядок вычислений "
                 "(Thinking Machines Lab, 2025). Расчёт: code/src/figures.py")
    ax = axes_at(fig, 0.29, 0.14, 0.97, top - 0.04)
    ax.set_xlim(0, 24); ax.set_ylim(-0.7, len(rows) - 0.3)
    hgrid(ax); ax.spines["left"].set_visible(False)
    for y, (n, k) in enumerate(rows):
        hbar(ax, y, k, AQUA if n.startswith("Ноутбук") else BLUE, thick=0.55)
        ax.text(k + 0.3, y, "0, ответ совпадает побайтно" if n.startswith("Ноутбук") else str(k), va="center", fontsize=8.6,
                color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8.8)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("Показателей из 24, у которых балл меняется на одном и том же запросе", fontsize=8.8)
    save(fig, "f7_повторяемость")


def f8_reasoning(items):
    gold = {(r, i): g[0] for r in fc.GOLD4 for i, g in sg.read_gold(r).items()}

    def cost(m, var):
        ms = [json.loads(p.read_text())["meta"] for p in sorted((ROOT / "raw").glob(f"{m.replace('/', '-')}__v7_extract__{var}__*.json"))]
        return statistics.mean(x.get("cost_usd") or 0 for x in ms if x["report"] in fc.BATTLE)

    rows, flips, price = [], {}, []
    for m, lab in (("deepseek/deepseek-v4.1-flash", "DeepSeek V4.1 Flash:\nвыключено → низкий уровень"),
                   ("qwen/qwen3.8-27b", "Qwen 27B (облако):\nвыключено → низкий уровень"),
                   ("z-ai/glm-5.3-flash", "GLM-5.3 Flash:\nминимальный → высокий")):
        vals, meds = [], []
        for var in ("t0-shuf-ex2-or-tx2", "t0-shuf-ex2-think-tx2"):
            v = fc.votes(items, fc.cfg(m, var), fc.BATTLE)
            med = {k: statistics.median_low(s) for k, s in v.items()}
            fr = statistics.mean(sum(len(set(s)) > 1 for (r, _), s in v.items() if r == rep) for rep in fc.BATTLE)
            vals.append((sum(med.get(k) == g for k, g in gold.items()), fr)); meds.append(med)
        a, b = meds
        flips[m] = (sum(b.get(k) == g != a.get(k) for k, g in gold.items()), sum(a.get(k) == g != b.get(k) for k, g in gold.items()))
        price.append(cost(m, "t0-shuf-ex2-think-tx2") / cost(m, "t0-shuf-ex2-or-tx2"))
        rows.append((lab, vals))
    up, down = map(sum, zip(*flips.values()))
    up2, down2 = map(sum, zip(*(f for m, f in flips.items() if not m.startswith("deepseek"))))
    dfr = statistics.mean(vals[1][1] - vals[0][1] for _, vals in rows)
    p = lambda b, c: "p\u00a0=\u00a0" + ru(fc.mcnemar_exact(b, c), ".3f")      # неразрывно: перенос не рвёт «p = …»
    fig = plt.figure(figsize=(10.2, 4.6))
    hi = [json.loads(q.read_text()) for q in sorted((ROOT / "raw").glob(
        "deepseek-deepseek-v4.1-flash__v7_extract__t0-ex2-det-think-high-tx2__*.json"))]
    fail = sum(not (x.get("content") or "").strip() for x in hi)
    top = header(fig, "Умеренное размышление повышает правильность и устойчивость",
                 "Та же модель, те же 6 отчётов × 5 перестановок, меняется только уровень размышления (у DeepSeek при "
                 f"этом сменился провайдер, DeepInfra → CoreWeave). По трём моделям вместе верными стали {up} клеток, "
                 f"неверными {down} (точный двусторонний тест Макнемара, {p(up, down)}); без DeepSeek {up2} и {down2}, "
                 f"{p(up2, down2)}. Хрупких клеток стало на {ru(-dfr, '.1f')} меньше на отчёт",
                 f"На максимальном уровне DeepSeek не ответил в {fail} запросах из {len(hi)}: весь лимит в "
                 f"{hi[0]['meta']['max_tokens'] // 1000} тыс. токенов ушёл на рассуждение. Умеренное размышление дороже в "
                 f"{ru(min(price), '.1f')}–{ru(max(price), '.1f')} раза. Расчёт: code/src/figures.py")
    h = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=GRAY, markeredgecolor=SURF),
         plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=BLUE, markeredgecolor=SURF)]
    fig.legend(h, ["без размышления / минимальное", "с размышлением"], loc="upper left", bbox_to_anchor=(0.005, top + 0.01),
               ncol=2, frameon=False, fontsize=8.6, handletextpad=0.3, columnspacing=1.6)
    a1 = axes_at(fig, 0.21, 0.13, 0.58, top - 0.13)
    a2 = axes_at(fig, 0.62, 0.13, 0.985, top - 0.13)
    for ax, idx, lim, title in ((a1, 0, (86, 96), "Совпало с эталоном, клеток из 96"), (a2, 1, (0, 4), "Хрупких клеток на отчёт")):
        hgrid(ax); ax.spines["left"].set_visible(False)
        ax.set_xlim(*lim); ax.set_ylim(-0.6, len(rows) - 0.4)
        for y, (lab, vals) in enumerate(rows[::-1]):
            a, b = vals[0][idx], vals[1][idx]
            ax.plot([a, b], [y, y], color=GRAY, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter([a], [y], s=60, color=GRAY, edgecolor=SURF, linewidth=2, zorder=3)
            ax.scatter([b], [y], s=70, color=BLUE, edgecolor=SURF, linewidth=2, zorder=4)
            fmt = (lambda x: f"{x:.0f}") if idx == 0 else (lambda x: f"{x:.2f}".replace(".", ","))
            ax.text(b, y + 0.22, fmt(b), ha="center", fontsize=8.6, color=INK)
            ax.text(a, y + 0.22, fmt(a), ha="center", fontsize=8.4, color=MUTED)
        ax.set_title(title, loc="left", color=INK2, fontsize=9.5, pad=6)
    a1.set_yticks(range(len(rows))); a1.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8.6)
    a2.set_yticks([])
    save(fig, "f8_размышление")


def f10_certificate(items):
    gold = {(r, i): v[0] for r in sg.REPORTS for i, v in sg.read_gold(r).items()}
    v = fc.votes(items, fc.cfg(M, "t0.7-ex2-mp0.05-tx2"), sg.REPORTS, range(1, 11))
    unc, ed, es = [], [], []
    for k in gold:
        c = collections.Counter(v[k]); top_, cnt = c.most_common(1)[0]
        unc.append(1 - cnt / len(v[k])); ed.append(int(abstain.err_direction(top_, gold[k]))); es.append(int(top_ != gold[k]))
    delta, n = 0.10 / 3, len(unc)

    def curve(errs):
        """Точная кривая Learn-then-Test: покрытие как функция бюджета; меняется только на границах кандидатов-порогов."""
        pairs = sorted(zip(unc, errs))
        pts = [(abstain.cp_upper(sum(e for _, e in pairs[:k]), k, delta / n), k / n)
               for k in range(1, n + 1) if k == n or pairs[k][0] != pairs[k - 1][0]]
        cov = lambda a: max([c for u, c in pts if u <= a], default=0.0)
        for a in (0.05, 0.10, 0.20):
            assert abs(cov(a) - abstain.calibrate(unc, errs, a, delta)[1]) < 1e-12, a
        return cov, sorted(u for u, _ in pts)

    cov_d, bp_d = curve(ed)
    cov_s, bp_s = curve(es)
    all_d = min(u for u in bp_d if cov_d(u) == 1)                  # бюджет, при котором закрыто всё
    hold = [i for i, k in enumerate(gold) if k[0] not in fc.DEV]    # только отложенные отчёты
    ph = sorted(zip([unc[i] for i in hold], [ed[i] for i in hold]))
    assert not sum(e for _, e in ph), "на отложенных есть ошибки по знаку — подпись ниже станет неверной"
    all_h = abstain.cp_upper(0, len(ph), delta / len(ph))           # 0 ошибок: граница, когда закрыто всё
    first_s = min(u for u in bp_s if cov_s(u) > 0)                  # первый бюджет, где строгое что-то закрывает
    pc = lambda x: ru(100 * x, ".1f").replace(",0", "")
    fig = plt.figure(figsize=(9.0, 4.8))
    top = header(fig, "Доля клеток, принимаемых автоматически при гарантированном потолке ошибок",
                 f"5 размеченных отчётов, {n} клеток. Для гарантии используются 10 прогонов при T=0,7: итоговая оценка "
                 "равна самой частой, неуверенность равна доле прогонов с другой оценкой. Граница Клоппера–Пирсона с "
                 "поправками на перебор порогов и на три определения ошибки, уровень доверия 90%. Непринятые клетки "
                 "передаются человеку",
                 f"Процедура и бюджет 10% записаны заранее, 18.09. На 3 отложенных отчётах ({len(ph)} клетки) ошибок по "
                 f"знаку нет, граница {pc(all_h)}%. Расчёт: code/src/figures.py")
    ax = axes_at(fig, 0.10, 0.15, 0.975, top - 0.05)
    hgrid(ax, "both")
    xs = [0.02] + [u for u in sorted(set(bp_d + bp_s)) if 0.02 < u < 0.30] + [0.30]
    for cov, c, lab in ((cov_d, BLUE, "ошибка знака или на 2 балла и больше"), (cov_s, ORANGE, "любое отличие от эталона")):
        ax.step([100 * a for a in xs], [100 * cov(a) for a in xs], where="post", color=c, linewidth=2, label=lab)
    ax.text(100 * all_d + 0.5, 110, f"ошибок знака и на 2+ балла нет: все {n} клеток принимаются автоматически "
            f"с гарантией «не больше {pc(all_d)}%»", fontsize=8.4, color=INK2, va="center")
    ax.text(100 * first_s + 0.6, 66, f"строгое определение: при бюджете 10% принято {pc(cov_s(0.10))}% клеток,\n"
            f"при {pc(first_s)}% принято {pc(cov_s(first_s))}%, при 20% принято {pc(cov_s(0.20))}%",
            fontsize=8.4, color=INK2, va="top")
    ax.set_xlim(2, 30); ax.set_ylim(-3, 117); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Гарантированный потолок доли ошибок среди принятых клеток, %", fontsize=8.8)
    ax.set_ylabel("Клеток принято автоматически, %", fontsize=8.8)
    ax.legend(loc="lower right", frameon=False, fontsize=8.4)
    save(fig, "f10_порог_отказа")


def f4_table(items):
    ords = order()
    v = fc.votes(items, fc.cfg(M, "t0-shuf-ex2-tx2"), ft.SBER, range(1, 6))
    same = sum(len(set(s)) == 1 for s in v)
    fig = plt.figure(figsize=(10.8, 7.4))
    top = header(fig, "Итоговые оценки: 15 отчётов Сбера × 24 показателя",
                 "Итоговый режим, спецификация v7: 5 прогонов на отчёт при T=0 с разным порядком показателей. Итоговая "
                 "оценка клетки равна медиане, которая совпадает с самым частым значением. Во всех 5 прогонах совпали "
                 f"{same} из {len(v)} клеток ({ru(100 * same / len(v), '.1f')}%)",
                 "Обоснования и цитаты для каждой оценки: ТАБЛИЦА_ОЦЕНОК.md; векторы: vectors.csv и vectors.json. "
                 "Расчёт: code/src/figures.py")
    swatch_row(fig, top - 0.03, [(DIV[2], "+2", None), (DIV[1], "+1", None), (DIV[0], "0", AXIS), (DIV[-1], "−1", None),
                                 (DIV[-2], "−2", None), (None, "хрупкая клетка: оценки прогонов разошлись, нужна проверка человеком", INK)])
    ax = axes_at(fig, 0.105, 0.22, 0.86, top - 0.085)
    reps = ft.SBER[::-1]
    for y, r in enumerate(reps):
        tot = 0
        for x, i in enumerate(ords):
            s = v[(r, i)]; med = statistics.median_low(s); tot += med
            ax.add_patch(Rectangle((x + 0.04, y + 0.06), 0.92, 0.88, facecolor=DIV[med], edgecolor=SURF, linewidth=1.5))
            if med:
                ax.text(x + 0.5, y + 0.5, f"{med:+d}".replace("-", "−"), ha="center", va="center", fontsize=7.4,
                        color="white" if abs(med) == 2 else INK)
            if len(set(s)) > 1:
                ax.add_patch(Rectangle((x + 0.1, y + 0.12), 0.8, 0.76, fill=False, edgecolor=INK, linewidth=1.2))
        ax.text(len(ords) + 0.35, y + 0.5, f"{tot:+d}".replace("-", "−"), va="center", ha="left", fontsize=8.6, color=INK)
        ax.text(len(ords) + 1.9, y + 0.5, "strong" if tot > 12 else "mixed" if tot >= -12 else "weak", va="center",
                ha="left", fontsize=8.6, color=INK if tot > 12 else INK2, fontweight="bold" if tot > 12 else None)
    ax.text(len(ords) + 0.35, len(reps) + 0.25, "сумма   вывод", fontsize=8.2, color=MUTED, ha="left", va="bottom")
    ax.set_xlim(0, len(ords)); ax.set_ylim(0, len(reps))
    ax.set_xticks([x + 0.5 for x in range(len(ords))])
    ax.set_xticklabels([IND[i] for i in ords], rotation=55, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticks([y + 0.5 for y in range(len(reps))]); ax.set_yticklabels([QLAB[r[5:]] + " " + r[:4] for r in reps], fontsize=8.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "f4_итоговая_таблица")


def f2_dynamics(items):
    """Рисунок 2: сумма оценок по отчётам Сбера и проверки здравого смысла из задания."""
    v = fc.votes(items, fc.cfg(M, "t0-shuf-ex2-tx2"), ft.SBER, range(1, 6))
    g = items[(items.config == fc.cfg(M, "t0-shuf-ex2-tx2")) & items.report.isin(ft.SBER) & items.run.isin(range(1, 6))]
    per = g.groupby(["report", "run"]).score.sum()
    xs = list(range(len(ft.SBER)))
    tot = [sum(statistics.median_low(v[(r, i)]) for i in report.IDS) for r in ft.SBER]
    near = [r for r, t in zip(ft.SBER, tot) if abs(t - 12) <= 2]
    flip = [r for r in ft.SBER if len({s > 12 for s in per[r]}) > 1]
    assert set(flip) <= set(near), "вывод менялся у отчёта далеко от порога — подпись ниже станет неверной"
    assert tot[0] == min(tot) and tot[-1] >= sorted(tot)[-2], "подписи проверок здравого смысла станут неверными"
    name = lambda r: QLAB[r[5:]] + " " + r[:4]
    fig = plt.figure(figsize=(10.2, 5.0))
    top = header(fig, "Динамика оценок Сбера проходит проверки здравого смысла из задания",
                 "Точка — сумма итоговых оценок 24 показателей (шкала от −48 до +48), серая полоса — разброс суммы по пяти "
                 "прогонам. 2022 год самый слабый, 2 кв. 2026 среди сильных, соседние отчёты не перескакивают через "
                 "всю шкалу",
                 f"{len(near)} отчётов лежат в пределах ±2 от порога 12; у {len(flip)} из них ({', '.join(map(name, flip))}) "
                 "вывод менялся между прогонами. Вывод weak (сумма меньше −12) не достигается. Расчёт: code/src/figures.py")
    h = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=BLUE, markeredgecolor=SURF),
         plt.Line2D([], [], marker="o", linestyle="", markersize=6.5, color=SURF, markeredgecolor=BLUE, markeredgewidth=1.6)]
    fig.legend(h, ["вывод strong", "вывод mixed"], loc="upper left", bbox_to_anchor=(0.005, top + 0.005), ncol=2,
               frameon=False, fontsize=8.6, handletextpad=0.3, columnspacing=1.6)
    ax = axes_at(fig, 0.075, 0.14, 0.985, top - 0.09)
    hgrid(ax, "y")
    ax.axhline(12, color=MUTED, linewidth=1, zorder=1)
    ax.text(10.5, 12.9, "порог strong: сумма больше 12", fontsize=8.2, color=MUTED, ha="center")
    ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
    for x, r in zip(xs, ft.SBER):
        lo, hi = per[r].min(), per[r].max()
        if hi > lo:
            ax.plot([x, x], [lo, hi], color=GRAY, linewidth=6, solid_capstyle="round", zorder=2)
    ax.plot(xs, tot, color=BLUE, linewidth=1.3, zorder=3, alpha=0.55)
    for x, t in zip(xs, tot):
        strong = t > 12
        ax.scatter([x], [t], s=58 if strong else 46, color=BLUE if strong else SURF, edgecolor=BLUE if not strong else SURF,
                   linewidth=1.6 if not strong else 2, zorder=4)
    sg_ = lambda t: f"{t:+d}".replace("-", "−")
    notes = {0: (0.35, -5.2, "left", f"{sg_(tot[0])}: самый слабый отчёт.\nПрибыль −78,3%, стоимость риска\nвыросла втрое"),
             4: (0.35, 30.5, "left", f"{sg_(tot[4])}: годовой отчёт\nсравнивается с кризисным 2022"),
             len(xs) - 1: (-0.35, 26.5, "right", f"{sg_(tot[-1])}: прибыль +20,9%,\nрекордные дивиденды")}
    for x, (dx, ty, ha, txt) in notes.items():
        ax.text(x + dx, ty, txt, ha=ha, va="center", fontsize=8.2, color=INK2, linespacing=1.3)
    ax.text(2.35, 4.6, "+12: в отчётах 2023 года нет цифр\nпрошлого года, много нулей по правилу", ha="center",
            va="center", fontsize=8.2, color=INK2, linespacing=1.3)
    ax.set_xticks(xs); ax.set_xticklabels([QLAB[r[5:]] + "\n" + r[:4] for r in ft.SBER], fontsize=8)
    ax.set_xlim(-0.5, len(xs) - 0.3)
    ax.set_ylabel("Сумма 24 оценок", fontsize=8.8); ax.set_ylim(-10, 35)
    save(fig, "f2_динамика_сбера")


def f9_hardware(items):
    import deep_analysis as da
    rows = [r for r in model_rows(items) if r["params"]]
    ours = next(r for r in rows if r["name"] == da.LOCAL)
    moe = [r["fragile"] / ours["fragile"] for r in rows if r["arch"] == "смесь экспертов" and r["params"] <= 120]
    better = [r for r in rows if r["fragile"] < ours["fragile"]]
    assert better and all(da.hw(r["params"]) == 512 for r in better), "заголовок «устойчивее только 400+ ГБ» станет неверным"
    gap = ours["fragile"] - min(r["fragile"] for r in rows)
    fig = plt.figure(figsize=(9.6, 5.6))
    top = header(fig, "Ноутбук на 32 ГБ близок к лучшему результату, устойчивее только модели на 400+ ГБ",
                 "Каждая точка соответствует открытой модели, работающей по одной схеме (6 отчётов × 5 перестановок). По "
                 "горизонтали память под веса при сжатии до 4 бит; модели левее линии «32» помещаются в ноутбук. Лучшей "
                 f"модели ноутбук уступает {ru(gap, '.1f')} хрупкой клетки на отчёт; у «смесей экспертов» до 120 млрд "
                 f"параметров хрупких клеток в {min(moe):.0f}–{max(moe):.0f} раз больше при любом объёме памяти",
                 "Память: 0,55 ГБ на млрд параметров при 4 битах, без учёта контекста. Размер и тип моделей по карточкам "
                 "OpenRouter и Hugging Face. Расчёт: code/src/figures.py")
    style = {"плотная": dict(color=BLUE, edgecolor=SURF, linewidth=2),
             "смесь экспертов": dict(color=SURF, edgecolor=BLUE, linewidth=1.6),
             "не раскрыта": dict(color=GRAY, edgecolor=SURF, linewidth=2)}
    mk = lambda **k: plt.Line2D([], [], marker="o", linestyle="", markersize=7, **k)
    h = [mk(color=AQUA, markeredgecolor=SURF), mk(color=BLUE, markeredgecolor=SURF),
         mk(color=SURF, markeredgecolor=BLUE, markeredgewidth=1.6), mk(color=GRAY, markeredgecolor=SURF)]
    fig.legend(h, ["ноутбук (плотная модель)", "плотная модель", "смесь экспертов", "тип не раскрыт"], loc="upper left",
               bbox_to_anchor=(0.005, top + 0.005), ncol=4, frameon=False, fontsize=8.6, handletextpad=0.3, columnspacing=1.6)
    ax = axes_at(fig, 0.085, 0.17, 0.985, top - 0.09)
    hgrid(ax, "both")
    ax.set_xscale("log", base=2)
    ax.axvline(32, color=MUTED, linewidth=1.1, zorder=1)
    shift = {"deepseek/deepseek-v4.1-flash": 1.06, "z-ai/glm-5.3": 0.94}       # только разнести слипшиеся точки
    lab = {da.LOCAL: (23, 0.75), "google/gemma-4-31b-it": (23, 1.95), "qwen/qwen3.8-27b": (23, 3.15),
           "qwen/qwen3.6-35b-a3b": (22, None), "nvidia/nemotron-3-nano-30b-a3b": (19, None),
           "openai/gpt-oss-120b": (73, None), "nvidia/nemotron-3-super-120b-a12b": (75, None),
           "qwen/qwen3-235b-a22b-2507": (112, 2.25), "z-ai/glm-5.3-flash": (205, 2.25), "xiaomi/mimo-v2.6-flash": (205, 4.95),
           "deepseek/deepseek-v4.1-flash": (530, 0.95), "z-ai/glm-5.3": (530, 2.4)}
    for r in rows:
        loc = r["name"] == da.LOCAL
        x, y = r["params"] * da.GB_PER_B * shift.get(r["name"], 1), r["fragile"]
        st = dict(color=AQUA, edgecolor=SURF, linewidth=2) if loc else style[r["arch"]]
        ax.scatter([x], [y], s=100 if loc else 58, zorder=4, **st)
        lx, ly = lab[r["name"]]
        ly = y if ly is None else ly
        if ly != y:
            ax.plot([x, lx * (0.97 if lx > x else 1.03)], [y, ly], color=AXIS, linewidth=0.8, zorder=3)
        name = SHORT[r["name"]]
        ax.text(lx, ly, name, fontsize=8.1, color=INK if loc else INK2, va="center", ha="left" if lx > x else "right",
                fontweight="bold" if loc else None, bbox=dict(facecolor=SURF, edgecolor="none", pad=0.5), zorder=5)
    ax.set_xlim(11, 1500); ax.set_ylim(18, 0)
    ax.set_xticks([16, 32, 64, 128, 256, 512])
    ax.set_xticklabels(["16", "32\nноутбук", "64\nMac Studio", "128\n2 видеокарты", "256\nсервер, 4 карты", "512\nкластер"],
                       fontsize=8.3)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Память под веса при 4 битах, ГБ (логарифмическая шкала)", fontsize=8.8)
    ax.set_ylabel("Хрупких клеток на отчёт (меньше — лучше)", fontsize=8.8)
    save(fig, "f9_железо")


if __name__ == "__main__":
    _, items = report.load()
    fns = {"1": f1_waterfall, "2": f2_dynamics, "3": f3_leaks, "4": f4_table, "5": f5_cells, "6": f6_models,
           "7": f7_repeat, "8": f8_reasoning, "9": f9_hardware, "10": f10_certificate}
    for w in sys.argv[1:] or list(fns):
        fns[w](items)
