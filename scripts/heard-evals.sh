#!/usr/bin/env bash
# Outcome evals (homeai.eval.e2e, accuracy only) of the models on the heard sets (homeai/voice/roundtrip.py): the
# written baseline on the same host, parakeet clean/farfield/noisy10, small clean/biased, medium biased, tiny clean,
# then the confusion suite. Per model and set, every condition runs in e2e's lockstep mode (each example under all
# variants back to back, so Ollama reuses the shared system-prompt cache). Resumable: cells with a report.json are
# skipped; a killed lockstep run redoes its unfinished cells.
#   OLLAMA=http://10.24.10.150:11600 scripts/heard-evals.sh sample     # 30 rows per set, confusions-sample: every cell
#   OLLAMA=http://10.24.10.150:11600 scripts/heard-evals.sh full       # the whole sets
#   MODELS="homeai-4b-lora-v5" CONDS="written parakeet-clean" scripts/heard-evals.sh sample
# Outputs: outputs/eval/<model>-<set>-heard-<engine>-<condition>[-s30]/ (written baseline: <model>-<set>-heard-written),
# summarised by python3 -m homeai.eval.heard_summary. HOMEAI_HOST/HARDWARE name the machine serving the model (the
# client's own name would be wrong), so any seconds_per_turn carries the right host.
set -uo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-sample}
OLLAMA=${OLLAMA:-http://10.24.10.150:11600}
MODELS=${MODELS:-"homeai-4b-lora-v5 homeai-2b-lora-v5 homeai-4b-lora-v8"}
CONDS=${CONDS:-"written parakeet-clean parakeet-farfield parakeet-noisy10 small-clean small-biased medium-biased tiny-clean"}
SETS=${SETS:-"testv2 wild wild2"}
NUM_CTX=${NUM_CTX:-}  # e.g. 32768 for design A on the large-house (bigbench-bundook) sets
# 13 Oct: the magpie labels apply only when the model is served from magpie. Before this, every run (including
# cassowary's own localhost:11434/11435 runs) was labelled magpie / apple-m2.
if [[ $OLLAMA == *10.24.10.150* ]]; then
  export HOMEAI_HOST=${HOMEAI_HOST:-magpie} HOMEAI_HARDWARE=${HOMEAI_HARDWARE:-apple-m2}
  export HOMEAI_HOST_NOTE=${HOMEAI_HOST_NOTE:-"standalone ollama 0.21.2 (Metal) on magpie, e2e client elsewhere"}
fi
export PYTHONPATH=.
src() { [ "$1" = testv2 ] && echo test-v2 || echo "$1"; }  # bash 3.2 (macOS): no associative arrays
if [ "$MODE" = sample ]; then SFX=-s30; SAMPLE="--sample 30"; CONF=confusions-sample; else SFX=; SAMPLE=; CONF=confusions; fi
lockstep() {  # model log "data out" pairs...
  local m=$1 log=$2 datas="" outs=""
  shift 2
  while [ $# -gt 0 ]; do
    [ -f "$2/report.json" ] || { datas="$datas,$1"; outs="$outs,$2"; }
    shift 2
  done
  [ -z "$datas" ] && return 0
  echo "$(date '+%F %T') $m: ${outs#,}"
  python3 -m homeai.eval.e2e --model "$m" --ollama "$OLLAMA" --data "${datas#,}" $SAMPLE ${NUM_CTX:+--num-ctx $NUM_CTX} --out "${outs#,}" > "$log" 2>&1 \
    || echo "$(date '+%F %T') FAILED (see $log)"
}
mkdir -p outputs/eval
for m in $MODELS; do
  for s in $SETS; do
    args=()
    for c in $CONDS; do
      if [ "$c" = written ]; then
        args+=("data/generated/$(src $s).jsonl" "outputs/eval/$m-$s-heard-written$SFX")
      else
        args+=("data/generated/heard/$(src $s)-$c.jsonl" "outputs/eval/$m-$s-heard-$c$SFX")
      fi
    done
    lockstep "$m" "outputs/eval/$m-$s-heard$SFX.log" "${args[@]}"
  done
  SAMPLE= lockstep "$m" "outputs/eval/$m-$CONF-heard.log" "data/generated/heard/$CONF.jsonl" "outputs/eval/$m-$CONF-heard"
done
echo "$(date '+%F %T') done $MODE"
