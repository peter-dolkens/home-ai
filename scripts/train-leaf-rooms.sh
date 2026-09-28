#!/usr/bin/env bash
# Per-room Needle 3 leaves for the Bundook house (architecture track, designs C/D; Peter, 26 Sep: many over-fit room
# models instead of one generic leaf). Runs natively on lapwing (RTX 3080, no docker): each room trains on its own data
# (data/generated/leaf-bundook/<area>-train.jsonl) plus a generic sample so phrasing generalises, then the Bundook leaf
# test is scored with every room routed to its own model (homeai.leaf.eval --model-dir). Idempotent per room.
#   scripts/train-leaf-rooms.sh <name> [generic sample size] [extra needle finetune args...]
# e.g. scripts/train-leaf-rooms.sh rooms-r16-e3 800 --epochs 3
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$1; MIX=${2:-800}; shift 2 || true
OUT=outputs/leaf/$NAME; mkdir -p "$OUT/cact"
NEEDLE=${NEEDLE:-.needle-venv/bin/needle}
ENGINE_DIR=${ENGINE_DIR:-needle-engine}   # linux-x86_64 CLI engine + base needle3.cact, for scoring
if [ ! -x "$ENGINE_DIR/needle" ] || [ ! -f "$ENGINE_DIR/needle3.cact" ]; then
  mkdir -p "$ENGINE_DIR"
  curl -sL -o "$ENGINE_DIR/needle" https://huggingface.co/Cactus-Compute/needle3/resolve/main/linux-x86_64/needle
  curl -sL -o "$ENGINE_DIR/needle3.cact" https://huggingface.co/Cactus-Compute/needle3/resolve/main/needle3.cact
  chmod +x "$ENGINE_DIR/needle"
fi
# The 10 GB 3080 fits a step of batch 8 x 640 tokens, 4 x 768 or 2 x 1024 reliably (memory grows with sequence length, not
# batch; 8 x 768 and 4 x 896 fit only sometimes); each room gets the largest batch its longest example allows, so nothing is truncated.
export NEEDLE_TELEMETRY=0 NEEDLE_SKIP_SCORE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
for train in data/generated/leaf-bundook/*-train.jsonl; do
  area=$(basename "$train" -train.jsonl)
  [ -f "$OUT/cact/$area.cact" ] && continue
  d="$OUT/$area"; mkdir -p "$d"
  PYTHONPATH=. python3 - "$train" data/generated/leaf-generic-train.jsonl "$MIX" "$d/data.jsonl" <<'EOF'
import json, random, sys
room, generic, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
from homeai.leaf.tokens import example_tokens
rows = [l for l in open(room) if l.strip()]
cap = max(example_tokens(json.loads(l)) for l in rows)  # keep the room's own length tier (batch size on the 3080)
g = [l for l in open(generic) if l.strip() and example_tokens(json.loads(l)) <= cap]
rows += random.Random(7).sample(g, min(n, len(g)))
random.Random(11).shuffle(rows)
open(out, "w").writelines(r if r.endswith("\n") else r + "\n" for r in rows)
print(f"{out}: {len(rows)} examples")
EOF
  longest=$(PYTHONPATH=. python3 -c "import json,sys; from homeai.leaf.tokens import example_tokens; print(max(example_tokens(json.loads(l)) for l in open(sys.argv[1]) if l.strip()))" "$d/data.jsonl")
  if [ "$longest" -le 632 ]; then bs=8; ml=640; elif [ "$longest" -le 760 ]; then bs=4; ml=768; else bs=2; ml=1024; fi
  echo "$(date '+%F %T') $area: training (longest $longest tokens -> batch $bs, max-len $ml)"
  "$NEEDLE" finetune "$d/data.jsonl" --val-split 0.05 --batch-size $bs --max-len $ml --checkpoint-dir "$d/ckpt" --out "$d/adapter.safetensors" "$@" > "$d/train.log" 2>&1
  "$NEEDLE" build --lora "$d/adapter.safetensors" --out "$OUT/cact/$area.cact" > "$d/build.log" 2>&1
  echo "$(date '+%F %T') $area: $(grep -E 'epoch +[0-9]+/' "$d/train.log" | tail -1 | tr -s ' ')"
done
GENERIC=${GENERIC:-outputs/leaf/generic-r16-e3/leaf.cact}   # fallback for rooms without their own model
[ -f "$OUT/test-bundook.summary.json" ] || PYTHONPATH=. python3 -m homeai.leaf.eval --data data/generated/leaf-test-bundook.jsonl \
  --needle "$ENGINE_DIR/needle" --model "$GENERIC" --model-dir "$OUT/cact" --out "$OUT/test-bundook.jsonl" 2>&1 | tail -30 | tee "$OUT/test-bundook.txt"
