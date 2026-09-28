#!/usr/bin/env bash
# Fine-tune a small Qwen as the room leaf on lapwing (RTX 3080, no docker), import it into lapwing's own Ollama and
# score it on the held-out leaf tests (homeai.leaf.eval --ollama). Idempotent per step.
#   scripts/train-leaf-qwen.sh configs/leaf-qwen35-2b.json homeai-leaf-qwen35-2b
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=$1; NAME=$2; NOEVAL=${3:-}   # --no-eval: train, merge and import only
PY=${PY:-.train-venv/bin/python}
export OLLAMA_HOST=${OLLAMA_HOST:-127.0.0.1:11500} OLLAMA_MODELS=${OLLAMA_MODELS:-$HOME/home-ai/ollama-models}
OLLAMA=${OLLAMA:-./ollama/bin/ollama}
OUT=$(python3 -c "import json;print(json.load(open('$CFG'))['out'])"); mkdir -p "$OUT"
pgrep -f "ollama serve" >/dev/null || { nohup setsid "$OLLAMA" serve > outputs/ollama-serve.log 2>&1 < /dev/null & sleep 4; }
if [ ! -d "$OUT/lora" ]; then
  # free the 10 GB card for training: unload every resident Ollama model (this Ollama has no "stop --all")
  "$OLLAMA" ps 2>/dev/null | awk 'NR>1{print $1}' | while read -r m; do "$OLLAMA" stop "$m" >/dev/null 2>&1; done
  sleep 3
  echo "$(date '+%F %T') train $CFG"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m homeai.train.train --config "$CFG" > "$OUT/train.log" 2>&1
fi
# Unsloth copies the base shard into merged16 read-only (0444) and then overwrites it in place: make it writable first
merged_ok() { [ -f "$OUT/merged16/config.json" ] && ! grep -q "Model merge failed" "$OUT/merge.log" 2>/dev/null; }
if ! merged_ok; then
  echo "$(date '+%F %T') merge"
  [ -d "$OUT/merged16" ] && chmod -R u+w "$OUT/merged16" && rm -f "$OUT/merged16/config.json"
  "$PY" scripts/merge16.py "$CFG" > "$OUT/merge.log" 2>&1 || true
  if grep -q "Permission denied" "$OUT/merge.log"; then chmod -R u+w "$OUT/merged16"; rm -f "$OUT/merged16/config.json"; "$PY" scripts/merge16.py "$CFG" > "$OUT/merge.log" 2>&1; fi
  merged_ok || { echo "merge failed: $OUT/merge.log"; exit 1; }
fi
if ! "$OLLAMA" list | grep -q "^$NAME[: ]"; then
  # Ollama's built-in renderer/parser per family (LFM2.5: lfm2; Qwen3.5: qwen3.5), from the config's ollama_template_from
  REND=$(python3 -c "import json;t=json.load(open('$CFG')).get('ollama_template_from','');print('lfm2' if t.startswith('lfm') else 'qwen3.5')")
  printf 'FROM %s/merged16\nTEMPLATE {{ .Prompt }}\nRENDERER %s\nPARSER %s\nPARAMETER temperature 0\n' "$PWD/$OUT" "$REND" "$REND" > "$OUT/Modelfile"
  "$OLLAMA" create "$NAME" --quantize q4_K_M -f "$OUT/Modelfile" > "$OUT/import.log" 2>&1
fi
[ "$NOEVAL" = "--no-eval" ] && exit 0
for t in bundook generic; do
  [ -f "$OUT/test-$t.summary.json" ] && continue
  PYTHONPATH=. python3 -m homeai.leaf.eval --ollama "http://$OLLAMA_HOST" --model "$NAME" --workers 2 \
    --data "data/generated/leaf-test-$t.jsonl" --out "$OUT/test-$t.jsonl" > "$OUT/test-$t.txt" 2>&1
  python3 -c "import json,sys;o=json.load(open(sys.argv[1]))['overall'];print(sys.argv[2], {k:o.get(k) for k in ('exact','end_state','refusal','false_refusal','latency_ms_p50')})" "$OUT/test-$t.summary.json" "$NAME $t"
done
