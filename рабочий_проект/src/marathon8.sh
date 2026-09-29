#!/bin/bash
# 20.09: дождаться конца очереди night7 (блок Gemma) → запустить night8 (v8 против v7 + финальная таблица).
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
stamp() { echo "=== $(date '+%d.%m %H:%M') $*"; }
stamp "жду конца очереди night7 (pid $1)"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
$PY src/night.py start queues/night8.toml || { stamp "очередь не стартовала — стоп"; exit 1; }
stamp "очередь night8 запущена"
