#!/bin/bash
# Качает YandexGPT-5-Lite-8B-instruct (GGUF) и сам дописывает задания в очередь дня.
# Зачем: третья семья моделей, русскоязычная и маленькая (8 млрд) — проверяем две вещи сразу:
#   1. тянет ли маленькая русская модель нашу схему извлечения;
#   2. падает ли движок llama.cpp на строгой JSON-схеме и у небольшой модели, или это беда только крупной.
cd "$(dirname "$0")/.." || exit 1
LMS="$HOME/.lmstudio/bin/lms"
stamp() { echo "=== $(date '+%d.%m %H:%M') $*"; }
stamp "качаю YandexGPT-5-Lite-8B-instruct-GGUF"
"$LMS" get "https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct-GGUF" -y || { stamp "закачка не удалась"; exit 1; }
KEY=$("$LMS" ls --json | .venv/bin/python -c "
import json,sys
ms=[m['modelKey'] for m in json.load(sys.stdin) if 'yandexgpt' in str(m.get('modelKey','')).lower()]
print(ms[0] if ms else '')")
[ -z "$KEY" ] && { stamp "модель скачалась, но ключ не найден"; exit 1; }
stamp "ключ модели: $KEY"
cat >> queues/day21b.toml <<TOML

# Третья семья: YandexGPT-5-Lite-8B-instruct (GGUF). Русская, маленькая (8 млрд).
# Пара «со схемой / без схемы» проверяет, падает ли llama.cpp на строгом JSON и у небольшой модели.
[[job]]
model = "$KEY"
spec = "v7_extract"
reports = ["2022_q4", "2026_q2", "2024_q3", "2023_q4"]
runs = 3
temperature = 0.7
min_p = 0.05
mode = "extract2"
text = "v2"
ctx = 32768
prefill = false

[[job]]
model = "$KEY"
spec = "v7_extract"
reports = ["2022_q4", "2026_q2", "2024_q3", "2023_q4"]
runs = 3
temperature = 0.7
min_p = 0.05
mode = "extract2"
text = "v2"
ctx = 32768
prefill = false
schema = false
TOML
stamp "задания дописаны в очередь — раннер подхватит их сам"
