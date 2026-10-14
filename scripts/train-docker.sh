#!/usr/bin/env bash
# Run training / export inside the Unsloth container on cassowary (RTX 3090).
#   scripts/train-docker.sh train  configs/qwen35-4b.json
#   scripts/train-docker.sh export configs/qwen35-4b.json homeai-4b
#   scripts/train-docker.sh shell
#   DETACH=-d scripts/train-docker.sh train configs/qwen35-4b.json   # background; follow with: docker logs -f homeai-train
# The repo is mounted at /work, the HF cache persists in /ai/models/hf, and the
# container reaches Ollama at http://ollama:11434 on the proxy network.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-unsloth/unsloth:latest}
NAME=${NAME:-homeai-train}
HF_CACHE=${HF_CACHE:-/ai/experiments/hf-cache}
# per-config overrides: "hf_cache" (host dir) and "pythonpath" (repo-relative dir prepended, e.g. .spark-libs for
# models that need a different transformers than the image ships)
CFG=${2:-}
if [ -n "$CFG" ] && [ -f "$CFG" ]; then
  _hc=$(python3 -c "import json;print(json.load(open('$CFG')).get('hf_cache',''))")
  _pp=$(python3 -c "import json;print(json.load(open('$CFG')).get('pythonpath',''))")
  [ -n "$_hc" ] && HF_CACHE=$_hc
fi
EXTRA_ENV=()
[ -n "${_pp:-}" ] && EXTRA_ENV=(-e "PYTHONPATH=/work/$_pp:/work")
mkdir -p "$HF_CACHE" "$ROOT/outputs"
run() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  # -d when DETACH=-d; -it only when we actually have a terminal; plain otherwise (cron/driver)
  local mode="${DETACH:-}"
  if [ -z "$mode" ] && [ -t 0 ] && [ -t 1 ]; then mode="-it"; fi
  docker run --rm $mode --name "$NAME" \
    --device nvidia.com/gpu=all \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NVIDIA_VISIBLE_DEVICES=all -e HF_HOME=/root/.cache/huggingface -e TOKENIZERS_PARALLELISM=false \
    "${EXTRA_ENV[@]}" \
    -v "$ROOT":/work -w /work \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -v /ai/models/homeai/runs:/ai/models/homeai/runs \
    --network proxy \
    --shm-size 8g \
    "$IMAGE" "$@"
}
LOG="outputs/$(basename "${2:-configs/qwen35-4b.json}" .json)-${1:-run}.log"
case "${1:-}" in
  train)  run bash -c "python -m homeai.train.train --config '${2:-configs/qwen35-4b.json}' ${*:3} 2>&1 | tee '$LOG'" ;;
  export) run bash -c "python -m homeai.train.export --config '${2:-configs/qwen35-4b.json}' --name '${3:-homeai}' --no-create ${*:4} 2>&1 | tee '$LOG'" ;;
  collect) run python -m homeai.train.export --config "${2:-configs/qwen35-4b.json}" --collect-only ;;
  dpo)    run bash -c "python -m homeai.train.dpo --config '${2}' ${*:3} 2>&1 | tee '$LOG'" ;;
  merge)  run python scripts/merge16.py "${2:-configs/qwen35-4b.json}" ;;
  shell)  run bash ;;
  *) echo "usage: $0 {train|dpo|export|merge|shell} [config] [name]"; exit 1 ;;
esac
