#!/usr/bin/env bash
# lapwing (RTX 3080) job queue for the architecture track: small leaves and fast models. Idempotent: each step skips
# what exists. Run: nohup setsid scripts/lapwing-queue.sh > outputs/lapwing-queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
while pgrep -f "leaf-qwen-finish.sh|train-leaf-qwen.sh" >/dev/null; do sleep 30; done   # the 2B / 0.8B leaves first
note() { echo "$(date '+%F %T') $*"; }
for t in office bedroom bathroom kitchen utility circulation outdoor; do
  note "type leaf $t"
  scripts/train-leaf-qwen.sh configs/leaf-type-$t.json homeai-leaf-type-$t --no-eval || note "type leaf $t: FAILED"
done
# score the Bundook leaf test with every room routed to its type's leaf
OUT=outputs/leaf-types; mkdir -p $OUT
[ -f $OUT/test-bundook.summary.json ] || PYTHONPATH=. python3 -m homeai.leaf.eval --ollama http://127.0.0.1:11500 --model homeai-leaf-qwen35-2b \
  --area-models data/generated/leaf-type/bundook-type-models.json --workers 2 --data data/generated/leaf-test-bundook.jsonl --out $OUT/test-bundook.jsonl > $OUT/test-bundook.txt 2>&1
python3 -c "import json;o=json.load(open('$OUT/test-bundook.summary.json'))['overall'];print('type leaves', {k:o.get(k) for k in ('exact','end_state','refusal','false_refusal','latency_ms_p50')})"
# design E fast model: Qwen3.5-2B as a full assistant on the champion's v5 data (A format)
note "fast 2B full assistant"
scripts/train-leaf-qwen.sh configs/qwen35-2b-lora-v5.json homeai-2b-lora-v5 --no-eval || note "fast 2B: FAILED"
note "queue done"
