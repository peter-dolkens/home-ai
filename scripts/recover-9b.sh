#!/usr/bin/env bash
# Recover the 8-9B tier champion homeai-9b-v2 (29 Sep): a disk-space prune on cassowary deleted its GGUF and Ollama
# model; its merged16 was rebuilt there and copied to frogmouth. Ollama's converter needs ~24 GB for a 9B, so the MLX
# training on this Mac is paused (SIGSTOP) while it runs and resumed afterwards, whatever happens.
#   nohup setsid scripts/recover-9b.sh > outputs/recover-9b.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=homeai-9b-v2; RUN=outputs/qwen35-9b-v2; QUANT=q4_K_M
SRC=$PWD/.mlx/imports/$NAME-merged16; EXP=$PWD/.mlx/exports/$NAME
S=/private/tmp/claude-501/-Users-peter-dolkens-Projects-home-ai/0ff4226d-5aa5-4b6c-9a81-89369fd665d4/scratchpad/convert
OLLAMA=$S/ollama-0.21.2/ollama; export OLLAMA_HOST=127.0.0.1:11436
HOST=cassowary.dolkens.net; REMOTE=/ai/experiments/home-ai
C=(-oControlPath="$HOME/.ssh/cm-%C" -oControlMaster=auto -oControlPersist=30m)
ts() { date '+%F %T'; }

pids=$(pgrep -f "homeai.train.train_mlx" || true)
resume() { [ -n "$pids" ] && kill -CONT $pids 2>/dev/null && echo "$(ts) resumed MLX training ($pids)"; }
trap resume EXIT
[ -n "$pids" ] && kill -STOP $pids && echo "$(ts) paused MLX training ($pids)"

"$OLLAMA" list >/dev/null 2>&1 || { (OLLAMA_MODELS="$S/ollama-models" nohup "$OLLAMA" serve > "$S/ollama-serve.log" 2>&1 &); sleep 5; }
if ! "$OLLAMA" list | grep -q "^$NAME[: ]"; then
  echo "$(ts) converting $NAME"
  "$OLLAMA" create "$NAME" --quantize "$QUANT" -f "$EXP/$NAME.Modelfile" > "$EXP/create.log" 2>&1 || { echo "$(ts) create FAILED"; tail -c 400 "$EXP/create.log"; exit 1; }
fi
resume; pids=""
BLOB=$("$OLLAMA" show --modelfile "$NAME" | awk '/^FROM /{print $2}')
GGUF=$EXP/$NAME-$QUANT.gguf
[ -f "$GGUF" ] || cp "$BLOB" "$GGUF"
echo "$(ts) gguf $(du -h "$GGUF" | cut -f1) sha256 $(shasum -a 256 "$GGUF" | cut -d' ' -f1)"

# cassowary's root disk also holds Frigate's database: ship only with the reserve intact
free=$(ssh "${C[@]}" $HOST "df -BG --output=avail / | tail -1 | tr -dc 0-9")
if [ "$free" -lt 50 ]; then echo "$(ts) cassowary has ${free}G free (<50G): not shipping"; exit 1; fi
ssh "${C[@]}" $HOST "mkdir -p $REMOTE/$RUN/gguf"
rsync -a --partial -e "ssh ${C[*]}" "$GGUF" "$HOST:$REMOTE/$RUN/gguf/"
ssh "${C[@]}" $HOST "cd $REMOTE && python3 -m homeai.train.ollama_import --run $RUN --name $NAME --base qwen3.5:9b --gguf && rm -rf $RUN/merged16 && docker exec ollama ollama list | grep '^$NAME'" \
  && echo "$(ts) done: $NAME imported on cassowary (merged16 removed there; portable GGUF kept in $RUN/gguf)" \
  || { echo "$(ts) import on cassowary FAILED"; exit 1; }
