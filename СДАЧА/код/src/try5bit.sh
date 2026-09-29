#!/bin/bash
# Замер 5 бит при поднятом пределе памяти: ждём, пока Андрей освободит память, и запускаем.
# Предел поднят до 26 ГиБ ТОЛЬКО на этот замер; после — возвращается 24 (страховка от зависания Mac).
# Раннер всё равно следит за ростом подкачки: если модель не потянет, он сам снимет замер.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
stamp() { echo "=== $(date '+%d.%m %H:%M') $*"; }
free_gib() { $PY -c "
import subprocess,re
o=subprocess.run(['vm_stat'],capture_output=True,text=True).stdout
g=lambda k: int(re.search(k+r':\s+(\d+)',o).group(1))
print(round((g('Pages free')+g('Pages inactive')+g('Pages speculative'))*4096/2**30,1))"; }
stamp "жду, пока освободится память (нужно около 8 ГиБ свободных)"
for i in $(seq 1 240); do
  F=$(free_gib)
  case "$(echo "$F >= 8" | bc -l)" in 1) stamp "свободно $F ГиБ — запускаю"; break;; esac
  sleep 30
done
sed -i '' 's/^MEM_LIMIT_GIB = 24 /MEM_LIMIT_GIB = 26 /' src/night.py
$PY src/night.py start queues/variant5bit.toml && stamp "замер 5 бит запущен"
