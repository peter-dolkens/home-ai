#!/usr/bin/env bash
# Convert + quantise a merged 16-bit checkpoint with Ollama on the Mac (32 GB RAM), then ship the
# Ollama-native GGUF back to cassowary and import it there. Needed for the 9B: Ollama's converter
# needs ~24 GB of RAM for it, which OOM-kills the Ollama server on cassowary (31 GB, shared).
#   scripts/convert-on-mac.sh outputs/qwen35-9b-r1 homeai-9b qwen3.5
# IMPORTANT: use the same Ollama version as cassowary (0.21.2). Newer Ollama imports Qwen3.5 as
# per-tensor blobs that 0.21.2 cannot read, and no longer offers q4_K_M for this architecture.
set -euo pipefail
RUN=$1; NAME=$2; RENDERER=${3:-qwen3.5}; QUANT=${QUANT:-q4_K_M}; OLLAMA_VERSION=${OLLAMA_VERSION:-0.21.2}
HOST=${CASSOWARY:-cassowary.dolkens.net}; REMOTE=/ai/experiments/home-ai
S=${SCRATCH:-/tmp/homeai-convert}; mkdir -p "$S/ollama-models" "$S/ollama-$OLLAMA_VERSION"
OLLAMA="$S/ollama-$OLLAMA_VERSION/ollama"
if [ ! -x "$OLLAMA" ]; then
  curl -sL -o "$S/ollama-$OLLAMA_VERSION/ollama-darwin.tgz" "https://github.com/ollama/ollama/releases/download/v$OLLAMA_VERSION/ollama-darwin.tgz"
  tar -xzf "$S/ollama-$OLLAMA_VERSION/ollama-darwin.tgz" -C "$S/ollama-$OLLAMA_VERSION"
fi
export OLLAMA_HOST=127.0.0.1:11436
if ! "$OLLAMA" list >/dev/null 2>&1; then
  (OLLAMA_MODELS="$S/ollama-models" nohup "$OLLAMA" serve > "$S/ollama-serve.log" 2>&1 &); sleep 5
fi
ollama() { "$OLLAMA" "$@"; }
rsync -a --partial "$HOST:$REMOTE/$RUN/merged16/" "$S/$NAME-merged16/"
printf 'FROM %s/%s-merged16\nTEMPLATE {{ .Prompt }}\nRENDERER %s\nPARSER %s\nPARAMETER num_ctx 8192\nPARAMETER temperature 0.2\nPARAMETER top_p 0.9\nPARAMETER top_k 20\n' "$S" "$NAME" "$RENDERER" "$RENDERER" > "$S/$NAME.Modelfile"
ollama create "$NAME" --quantize "$QUANT" -f "$S/$NAME.Modelfile"
BLOB=$(ollama show --modelfile "$NAME" | awk '/^FROM /{print $2}')
GGUF="$S/$NAME-$QUANT.gguf"; cp "$BLOB" "$GGUF"
ssh "$HOST" "mkdir -p $REMOTE/$RUN/gguf"
rsync -a --partial "$GGUF" "$HOST:$REMOTE/$RUN/gguf/"
ssh "$HOST" "cd $REMOTE && python3 -m homeai.train.ollama_import --run $RUN --name $NAME --base ${BASE_TAG:-${RENDERER}:9b} --gguf"
echo "done: $NAME imported on $HOST; portable GGUF at $REMOTE/$RUN/gguf/$(basename "$GGUF")"
