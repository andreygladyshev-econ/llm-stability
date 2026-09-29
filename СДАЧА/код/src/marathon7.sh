#!/bin/bash
# 19.09 вечер: дождаться мягкой остановки night6 → пробный прогон v7 → проверка → очередь night7 без срока.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
stamp() { echo "=== $(date '+%d.%m %H:%M') $*"; }
stamp "жду остановки прошлой очереди (pid $1)"
while kill -0 "$1" 2>/dev/null; do sleep 15; done
stamp "пробный прогон v7"
$PY src/night.py run queues/smoke7.toml || { stamp "пробный прогон упал — стоп"; exit 1; }
$PY src/smoke7.py || { stamp "проверка v7 не прошла — стоп"; exit 1; }
$PY src/night.py start queues/night7.toml || { stamp "очередь не стартовала — стоп"; exit 1; }
stamp "очередь night7 запущена"
