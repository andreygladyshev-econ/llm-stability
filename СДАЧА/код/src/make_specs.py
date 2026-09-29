"""Модули из specs/modules_v2.md → варианты спецификации specs/{вариант}.md и .prompt.md.

Запуск: .venv/bin/python src/make_specs.py
Перезапись безопасна: если вариант уже гоняли, а текст изменился, night.py остановится на несовпадении хеша.

Ручные варианты (не собираются здесь, только получают .prompt.md): v2_inplace — правки вписаны внутрь текста,
v2_compact — только правила и таблица, v2_en — английский перевод v2_inplace со своим промптом.
"""
import re
from pathlib import Path

SPECS = Path(__file__).resolve().parent.parent / "specs"
PROMPT = "v1_baseline.prompt.md"
ALL = ["period", "market_share", "cor", "opex", "state", "tone", "digital", "quotes"]

# вариант → (основа, модули, куда вставить уточнения)
VARIANTS = {
    "v2_all": ("v1_baseline", ALL, "end"),
    "v2_top": ("v1_baseline", ALL, "start"),
    "v2_noex": ("base_v1_noexamples", ALL, "end"),
    "v2_period": ("v1_baseline", ["period"], "end"),
    "v2_market": ("v1_baseline", ["market_share"], "end"),
    "v2_cor": ("v1_baseline", ["cor"], "end"),
    "v2_opex": ("v1_baseline", ["opex"], "end"),
    "v2_state": ("v1_baseline", ["state"], "end"),
    "v2_tone": ("v1_baseline", ["tone"], "end"),
}
MANUAL = ["v2_inplace", "v2_compact", "v3", "v4", "v2_tbl", "v2p1", "v2p2", "v6"]  # промпт как у v1; у v2_en свой, лежит рядом


def modules():
    text = (SPECS / "modules_v2.md").read_text()
    parts = re.split(r"<!-- module: (\w+) -->", text)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def main():
    mods, prompt = modules(), (SPECS / PROMPT).read_text()
    for name, (base, keys, where) in VARIANTS.items():
        missing = [k for k in keys if k not in mods]
        assert not missing, f"{name}: нет модулей {missing}"
        base_text = (SPECS / f"{base}.md").read_text().rstrip()
        block = f"## Уточнения ({name}) — имеют приоритет над остальными правилами\n\n" + "\n\n".join(mods[k] for k in keys)
        text = f"{block}\n\n---\n\n{base_text}\n" if where == "start" else f"{base_text}\n\n---\n\n{block}\n"
        (SPECS / f"{name}.md").write_text(text)
        (SPECS / f"{name}.prompt.md").write_text(prompt)
        print(f"{name}: {base} + {len(keys)} мод. ({where}) → {len(text)} символов")
    for name in MANUAL:
        (SPECS / f"{name}.prompt.md").write_text(prompt)
        print(f"{name}: ручной, {len((SPECS / f'{name}.md').read_text())} символов")
    assert (SPECS / "v2_en.prompt.md").exists()
    print(f"v2_en: ручной, {len((SPECS / 'v2_en.md').read_text())} символов, свой промпт")


if __name__ == "__main__":
    main()
