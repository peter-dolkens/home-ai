#!/usr/bin/env bash
# MLX counterpart of scripts/train-leaf-qwen.sh for the Macs (frogmouth; falcon trains only): LoRA with
# homeai.train.train_mlx, fuse to bf16 safetensors (<out>/merged16), import into this Mac's Ollama 0.21.2 with the same
# Modelfile lapwing uses. Idempotent per step; evals run separately (homeai.eval.e2e / homeai.tiered.eval / homeai.leaf.eval).
#   scripts/train-mlx.sh configs/qwen35-2b-lora-orch.json homeai-2b-lora-orch
#   scripts/train-mlx.sh configs/qwen35-0.8b-lora-v5.json homeai-0.8b-lora-v5 --train-only   # falcon: no Ollama there
#   SET="save_steps=25" scripts/train-mlx.sh ...                                               # extra config overrides
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/mlx-env.sh
CFG=$1; NAME=$2; MODE=${3:-}
OUT=$(python -c "import json;print(json.load(open('$CFG'))['out'])"); mkdir -p "$OUT"
# Short Metal command buffers: macOS aborts GPU work that holds up the display for too long ("Impacting Interactivity",
# killed falcon's 0.8B run at step 760); 2 ops / 50 MB per buffer measured no slower. A run that still dies resumes
# from its last checkpoint (save_steps), up to 5 times.
export MLX_MAX_OPS_PER_BUFFER=${MLX_MAX_OPS_PER_BUFFER:-2} MLX_MAX_MB_PER_BUFFER=${MLX_MAX_MB_PER_BUFFER:-50}
for attempt in 1 2 3 4 5; do
  [ -f "$OUT/lora/adapters.safetensors" ] && break
  echo "$(date '+%F %T') train $CFG on $HOMEAI_HOST (attempt $attempt)"
  python -m homeai.train.train_mlx --config "$CFG" ${SET:+--set $SET} >> "$OUT/train.log" 2>&1 && break
  grep -q "Command buffer execution failed" <(tail -5 "$OUT/train.log") || { echo "train failed: $OUT/train.log"; exit 1; }
  sleep 30
done
[ -f "$OUT/lora/adapters.safetensors" ] || { echo "train failed: $OUT/train.log"; exit 1; }
[ "$MODE" = "--train-only" ] && exit 0
if [ ! -f "$OUT/merged16/homeai-merge.json" ]; then
  echo "$(date '+%F %T') fuse"
  python -m homeai.train.train_mlx --config "$CFG" --fuse > "$OUT/merge.log" 2>&1
fi
scripts/mlx-env.sh ollama-serve >/dev/null
if ! "$OLLAMA" list | grep -q "^$NAME[: ]"; then
  # Ollama's built-in renderer/parser per family (LFM2.5: lfm2; Qwen3.5: qwen3.5), from the config's ollama_template_from
  REND=$(python -c "import json;t=json.load(open('$CFG')).get('ollama_template_from','');print('lfm2' if t.startswith('lfm') else 'qwen3.5')")
  printf 'FROM %s/merged16\nTEMPLATE {{ .Prompt }}\nRENDERER %s\nPARSER %s\nPARAMETER temperature 0\n' "$PWD/$OUT" "$REND" "$REND" > "$OUT/Modelfile"
  echo "$(date '+%F %T') ollama create $NAME"
  "$OLLAMA" create "$NAME" --quantize q4_K_M -f "$OUT/Modelfile" > "$OUT/import.log" 2>&1
fi
"$OLLAMA" list | grep "^$NAME[: ]"
