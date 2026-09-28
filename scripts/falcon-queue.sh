#!/usr/bin/env bash
# falcon (Mac mini M2 Pro, 32 GB, ~25 GB free disk) MLX training queue: waits for the running MLX job, then trains each
# config in turn (--train-only: frogmouth fuses, imports into Ollama and evaluates). Disk is tight, so only the next
# job's base model stays in the HF cache. Idempotent: finished runs (<out>/lora) are skipped, interrupted ones resume.
#   nohup scripts/falcon-queue.sh > outputs/falcon-queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
. scripts/mlx-env.sh
note() { echo "$(date '+%F %T') $*"; }
while pgrep -f "homeai.train.train_mlx" >/dev/null; do sleep 60; done
keep_only() {  # drop every cached base model except $1 (a repo id)
  for d in "$HF_HOME"/hub/models--*; do
    [ -d "$d" ] && [ "$(basename "$d")" != "models--${1//\//--}" ] && rm -rf "$d" && note "removed $(basename "$d")"
  done
}
for job in "configs/qwen35-2b-lora-orch.json homeai-2b-lora-orch" "configs/leaf-qwen35-2b.json homeai-leaf-qwen35-2b"; do
  set -- $job
  base=$(python -c "import json;print(json.load(open('$1'))['base_model'])")
  keep_only "$base"
  hf download "$base" >/dev/null 2>&1 || { note "download $base FAILED"; continue; }
  note "train $1"
  SET="save_steps=25" HF_HUB_OFFLINE=1 scripts/train-mlx.sh "$1" "$2" --train-only || note "$1 FAILED"
done
note "queue done"
