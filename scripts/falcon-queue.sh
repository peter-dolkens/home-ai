#!/usr/bin/env bash
# falcon (Mac mini M2 Pro, 32 GB, ~25 GB free disk) MLX training queue: waits for the running MLX job, then trains each
# config in turn (--train-only: frogmouth fuses, imports into Ollama and evaluates). Disk is tight, so only the next
# job's base model stays in the HF cache. Idempotent: finished runs (<out>/lora) are skipped, interrupted ones resume.
#   nohup scripts/falcon-queue.sh configs/leaf-qwen35-2b.json:homeai-leaf-qwen35-2b > outputs/falcon-queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
. scripts/mlx-env.sh
note() { echo "$(date '+%F %T') $*"; }
while pgrep -f "homeai.train.train_mlx" >/dev/null; do sleep 60; done
keep_only() {  # drop every cached base model except $1 (a repo id). This huggingface_hub keeps the weights in a
  # shared store (hub/blobs/<xx>/<sha> with a .refs list of the model dirs using each), so drop the entries whose
  # refs name only removed models too
  for d in "$HF_HOME"/hub/models--*; do
    [ -d "$d" ] && [ "$(basename "$d")" != "models--${1//\//--}" ] && rm -rf "$d" && note "removed $(basename "$d")"
  done
  for r in "$HF_HOME"/hub/blobs/*/*.refs; do
    [ -f "$r" ] || continue
    live=0; for m in $(cat "$r"); do [ -d "$HF_HOME/hub/${m%%/*}" ] && live=1; done
    [ $live = 0 ] && rm -f "${r%.refs}" "${r%.refs}.lock" "$r" && note "removed blob $(basename "${r%.refs}" | cut -c1-12)"
  done
}
for job in "$@"; do  # CONFIG:OLLAMA_NAME
  set -- "${job%%:*}" "${job##*:}"
  base=$(python -c "import json;print(json.load(open('$1'))['base_model'])")
  keep_only "$base"
  hf download "$base" >/dev/null 2>&1 || { note "download $base FAILED"; continue; }
  note "train $1"
  SET="save_steps=25" HF_HUB_OFFLINE=1 scripts/train-mlx.sh "$1" "$2" --train-only || note "$1 FAILED"
done
note "queue done"
