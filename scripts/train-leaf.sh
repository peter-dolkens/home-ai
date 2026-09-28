#!/usr/bin/env bash
# Fine-tune a Needle 3 room leaf (architecture track, designs C/D) on cassowary's GPU, build the .cact and score it
# on the held-out leaf tests with the CPU engine. Idempotent: skips steps whose outputs exist.
#   scripts/train-leaf.sh <name> <train.jsonl> [extra needle finetune args...]
# e.g. scripts/train-leaf.sh generic-r16-e3 data/generated/leaf-generic-train.jsonl --epochs 3
# Needle's own held-out scoring (one generate per example, very slow) is skipped via NEEDLE_SKIP_SCORE (patched into
# .needle-venv); homeai.leaf.eval scores the leaf instead. Local LoRA leaves Needle's confidence head untrained (confidence reports None); see homeai/leaf/.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$1; DATA=$2; shift 2
OUT=outputs/leaf/$NAME; mkdir -p "$OUT"
NEEDLE_DIR=${NEEDLE_DIR:-/ai/models/homeai/needle}   # linux-x86_64 engine + base needle3.cact
if [ ! -x "$NEEDLE_DIR/needle" ] || [ ! -f "$NEEDLE_DIR/needle3.cact" ]; then
  mkdir -p "$NEEDLE_DIR"
  curl -sL -o "$NEEDLE_DIR/needle" https://huggingface.co/Cactus-Compute/needle3/resolve/main/linux-x86_64/needle
  curl -sL -o "$NEEDLE_DIR/needle3.cact" https://huggingface.co/Cactus-Compute/needle3/resolve/main/needle3.cact
  chmod +x "$NEEDLE_DIR/needle"
fi
run() {  # JAX on the GPU; no preallocation so it can share the card with Ollama's resident models
  docker run --rm --gpus all --name homeai-leaf -e NEEDLE_TELEMETRY=0 -e NEEDLE_SKIP_SCORE=1 -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e HF_HOME=/work/.hf-needle -v "$PWD:/work" -w /work --entrypoint /work/.needle-venv/bin/needle unsloth/unsloth:latest "$@"
}
if [ ! -f "$OUT/adapter.safetensors" ]; then
  run finetune "$DATA" --checkpoint-dir "$OUT/ckpt" --out "$OUT/adapter.safetensors" "$@" 2>&1 | tee "$OUT/train.log"
fi
if [ ! -f "$OUT/leaf.cact" ]; then
  run build --lora "$OUT/adapter.safetensors" --out "$OUT/leaf.cact" 2>&1 | tee "$OUT/build.log"
fi
docker run --rm --entrypoint chown -v "$PWD/outputs/leaf:/o" alpine -R "$(id -u):$(id -g)" /o >/dev/null 2>&1 || true
for t in generic bundook; do
  [ -f "$OUT/test-$t.summary.json" ] && continue
  PYTHONPATH=. python3 -m homeai.leaf.eval --data "data/generated/leaf-test-$t.jsonl" --needle "$NEEDLE_DIR/needle" \
    --model "$OUT/leaf.cact" --out "$OUT/test-$t.jsonl" 2>&1 | tail -30 | tee "$OUT/test-$t.txt"
done
