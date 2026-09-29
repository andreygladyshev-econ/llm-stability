#!/bin/bash
# H11 (движок) и H6b (глубина сжатия): тот же Qwen 27B в GGUF, 4 и 6 бит.
# Качается параллельно с прогонами: сеть занята, видеокарта нет.
LMS="$HOME/.lmstudio/bin/lms"
for q in q4_k_m q6_k; do
  echo "=== $(date '+%d.%m %H:%M') качаю qwen/qwen3.8-27b@$q"
  "$LMS" get "qwen/qwen3.8-27b@$q" --gguf -y || echo "не вышло: $q"
done
echo "=== $(date '+%d.%m %H:%M') готово"
"$LMS" ls | grep -i qwen3.8-27b
