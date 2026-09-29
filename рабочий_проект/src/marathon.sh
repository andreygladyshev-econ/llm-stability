#!/bin/bash
# Марафон 19.09 (Андрей: «сначала распознай все отчёты, потом начинай прогоны»; прервёт сам pause/stop).
#   1. дождаться цепочки распознавания (процесс $1: text_v2 всех 15 → make_queue6.py)
#   2. пересобрать полную очередь night6 и запустить прогоны без срока
# Лог: logs/marathon.log. Пауза/продолжение/остановка — обычные команды night.py.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
stamp() { echo "=== $(date '+%d.%m %H:%M') $*"; }

stamp "жду окончания распознавания (цепочка pid $1)"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
stamp "распознано отчётов v2: $($PY -c 'import sys; sys.path.insert(0,"src"); import texts; print(len(texts.hashes("v2")))')"
$PY src/make_queue6.py || { stamp "очередь не собралась — стоп"; exit 1; }
$PY src/night.py start queues/night6.toml || { stamp "прогоны не стартовали — стоп"; exit 1; }
stamp "прогоны запущены"
