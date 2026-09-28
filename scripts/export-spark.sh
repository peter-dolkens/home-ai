#!/usr/bin/env bash
# Export a Spark X2.5 fine-tune to Ollama 0.34.x (container ollama-next, port 11435):
#   merge LoRA (transformers 4.57 path) -> llama.cpp convert (knows Spark2_5ForCausalLM) -> Q4_K_M -> ollama create
#   with the vendor GGUF's template/stops.   scripts/export-spark.sh configs/spark-x25-4b-v5.json homeai-spark-x25-4b-v5
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=$1; NAME=$2; BASE=${3:-hf.co/XHToken/Spark-X2.5-4B-GGUF:Q4_K_M}; CTR=${CTR:-ollama-next}
OUT=$(python3 -c "import json;print(json.load(open('$CFG'))['out'])")
LLAMA=${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:full}
own() { docker run --rm -v "$PWD/outputs":/o alpine chown -R "$(id -u):$(id -g)" /o >/dev/null 2>&1; }
if docker exec $CTR ollama list | grep -q "^$NAME[: ]"; then echo "exists: $NAME"; exit 0; fi
[ -f "$OUT/merged16/config.json" ] || { NAME=homeai-merge DETACH="" scripts/train-docker.sh merge "$CFG" </dev/null || exit 1; own; }
mkdir -p "$OUT/gguf"
GG="$OUT/gguf/$NAME-q4_K_M.gguf"
if [ ! -f "$GG" ]; then
  docker pull -q "$LLAMA" >/dev/null
  docker run --rm -v "$(readlink -f "$OUT/merged16")":/in:ro -v "$(readlink -f "$OUT/gguf")":/g "$LLAMA" --convert /in --outtype f16 --outfile /g/f16.gguf || exit 1
  docker run --rm -v "$(readlink -f "$OUT/gguf")":/g "$LLAMA" --quantize /g/f16.gguf /g/$(basename "$GG") Q4_K_M || exit 1
  rm -f "$OUT/gguf/f16.gguf"; own
fi
docker exec $CTR ollama show "$BASE" --modelfile | grep -vE '^(FROM|#)' > /tmp/$NAME.Modelfile.body
{ echo "FROM /tmp/$NAME.gguf"; cat /tmp/$NAME.Modelfile.body; } > /tmp/$NAME.Modelfile
docker cp "$(readlink -f "$GG")" $CTR:/tmp/$NAME.gguf && docker cp /tmp/$NAME.Modelfile $CTR:/tmp/$NAME.Modelfile
docker exec $CTR ollama create "$NAME" -f /tmp/$NAME.Modelfile && docker exec $CTR rm -f /tmp/$NAME.gguf
docker exec $CTR ollama show "$NAME" | sed -n '/Capabilities/,/^$/p'
