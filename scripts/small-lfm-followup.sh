#!/usr/bin/env bash
# 13 Oct (Peter: focus on the small LFMs, time on cassowary, compare on the Mac mini): once lapwing has trained and imported
# a design-B LFM run, fan it out the way the 350M went (run from frogmouth, detached):
#   1. lapwing: house heard evals, 10 conditions (Ollama 0.21.2, accuracy only, as the pick's)
#   2. cassowary: the GGUF into the main Ollama (0.40.3), then outputs/bench-o040.sh on it (B sets + house typed, timed)
#   3. falcon: merged16 -> MLX (scripts/falcon-mlx-build.sh), then the falcon stream (prefix timing + B sets)
#   nohup scripts/small-lfm-followup.sh lfm25-230m-lora-b-v8r256e4 "mlx6 mlx8" > outputs/followup-230m.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=$1; VARIANTS=${2:-mlx6}; M=homeai-$RUN
note() { echo "$(date '+%F %T') $M: $*"; }
# one multiplexed connection per host: many short ssh calls tripped lapwing's sshd MaxStartups (13 Oct)
O=(-oBatchMode=yes -oControlPath=$HOME/.ssh/cm-%C -oControlMaster=auto -oControlPersist=30m)
L=(ssh "${O[@]}" lapwing.dolkens.net); C=(ssh "${O[@]}" cassowary.dolkens.net)
until "${L[@]}" "OLLAMA_HOST=127.0.0.1:11500 ~/home-ai/ollama/bin/ollama list 2>/dev/null | grep -q '^$M:'"; do
  "${L[@]}" "tail -1 ~/home-ai/outputs/lapwing-q55.log" | grep -q "$M: train FAILED" && { note "training failed on lapwing"; exit 1; }
  sleep 300
done
note "imported on lapwing"

# 1. lapwing heard evals (SKIP_HEARD=1: already started, e.g. when re-running steps 2-3)
[ -n "${SKIP_HEARD:-}" ] || "${L[@]}" "cd ~/home-ai && nohup setsid env OLLAMA=http://127.0.0.1:11500 HOMEAI_HOST=lapwing HOMEAI_HARDWARE=rtx-3080 \
  HOMEAI_HOST_NOTE='lapwing (accuracy only), user_context' HOMEAI_PROMPT_LAYOUT=user_context PYTHONPATH=. MODELS=$M SETS=bigbench-b-bundook200 \
  CONDS='written parakeet-clean parakeet-farfield parakeet-noisy10 small-clean small-biased medium-biased tiny-clean medium-farfield-biased medium-noisy10-biased' \
  scripts/heard-evals.sh full > outputs/$M-bundook200-heard.log 2>&1 < /dev/null &" && note "lapwing heard evals started"

# 2. cassowary: same weights and Modelfile parameters as lapwing's import, then the timed benchmark
blob=$("${L[@]}" "OLLAMA_HOST=127.0.0.1:11500 ~/home-ai/ollama/bin/ollama show --modelfile $M | awk '/^FROM /{print \$2}'")
params=$("${L[@]}" "OLLAMA_HOST=127.0.0.1:11500 ~/home-ai/ollama/bin/ollama show --modelfile $M | grep -E '^(RENDERER|PARSER|PARAMETER)'")
"${C[@]}" "mkdir -p /ai/experiments/home-ai/outputs/$RUN/gguf"
scp -3 -q "${O[@]}" "lapwing.dolkens.net:$blob" "cassowary.dolkens.net:/ai/experiments/home-ai/outputs/$RUN/gguf/$M.gguf" || { note "copy to cassowary FAILED"; exit 1; }
printf 'FROM /import/%s/model.gguf\n%s\n' "$M" "$params" | "${C[@]}" "cat > /tmp/Mf.$M"
"${C[@]}" "cd /ai/experiments/home-ai && docker exec ollama mkdir -p /import/$M && docker cp outputs/$RUN/gguf/$M.gguf ollama:/import/$M/model.gguf \
  && docker cp /tmp/Mf.$M ollama:/import/$M/Modelfile && docker exec ollama ollama create $M -f /import/$M/Modelfile >/dev/null 2>&1; docker exec ollama rm -rf /import/$M; \
  docker exec ollama ollama list | grep -q '^$M:' && nohup setsid outputs/bench-o040-next.sh $M > /dev/null 2>&1 < /dev/null &" && note "cassowary import + benchmark queued"

# 3. falcon: MLX from lapwing's merged weights
mkdir -p "outputs/$RUN"
rsync -a -e "ssh ${O[*]}" "lapwing.dolkens.net:home-ai/outputs/$RUN/lora" "lapwing.dolkens.net:home-ai/outputs/$RUN/merged16" "lapwing.dolkens.net:home-ai/outputs/$RUN/config.used.json" "outputs/$RUN/" \
  || { note "fetch of merged weights FAILED"; exit 1; }
# shellcheck disable=SC2086
scripts/falcon-mlx-build.sh "$RUN" "$M" $VARIANTS && scripts/falcon-pull.sh start && note "falcon MLX ($VARIANTS) staged, stream started"
