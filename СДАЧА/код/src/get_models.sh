#!/bin/zsh
# Докачка малых моделей для пилота дообучения (пилот не проводился, в записку не вошёл).
# Можно прерывать и запускать снова — скачанное сохраняется. Лог: logs/models_download.log
cd "${0:A:h}/.."
MODELS=(empero-ai/Qwen3.8-4B-Distill Qwen/Qwen3.5-4B empero-ai/Qwen3.8-9B-Distill Qwen/Qwen3.5-9B openbmb/MiniCPM5-2B)
for m in $MODELS; do
  echo "=== $m $(date +%H:%M)"
  ok=0
  for i in {1..30}; do
    .venv/bin/hf download "$m" --exclude "*.gguf" --exclude "*.pth" >/dev/null 2>>logs/models_download.err && { ok=1; break }
    echo "повтор $i: $(tail -1 logs/models_download.err | cut -c1-100)"; sleep 60
  done
  echo "итог $m ok=$ok $(du -sh ~/.cache/huggingface/hub/models--${m//\//--} | cut -f1)"
done
# обрывки прерванных загрузок не нужны: файлы докачиваются заново целиком
find ~/.cache/huggingface/hub -name "*.incomplete" -delete 2>/dev/null
echo "=== готово $(date +%H:%M)"
