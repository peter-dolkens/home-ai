#!/usr/bin/env bash
# 13 Oct: when cassowary has trained and merged the LFM2.5 2.6B (outputs/train-2.6b.sh), relay its merged weights through
# frogmouth to lapwing and import them there (Ollama 0.21.2, as the small LFMs); scripts/small-lfm-followup.sh, already
# waiting, then fans it out (heard evals, cassowary benchmark, falcon MLX).
set -uo pipefail
cd "$(dirname "$0")/.."
R=lfm25-2.6b-lora-b-v8r128e4; M=homeai-$R
O=(-oBatchMode=yes -oControlPath=$HOME/.ssh/cm-%C -oControlMaster=auto -oControlPersist=30m)
note() { echo "$(date '+%F %T') $M: $*"; }
until ssh "${O[@]}" cassowary.dolkens.net "test -f /ai/experiments/home-ai/outputs/$R/merged16/config.json && ! pgrep -f outputs/[t]rain-2.6b.sh >/dev/null"; do
  ssh "${O[@]}" cassowary.dolkens.net "grep -q '$R.json: FAILED' /ai/experiments/home-ai/outputs/overnight-status.log" && { note "cassowary run FAILED"; exit 1; }
  sleep 600
done
note "merged on cassowary; relaying"
mkdir -p outputs/$R
rsync -a -e "ssh ${O[*]}" cassowary.dolkens.net:/ai/experiments/home-ai/outputs/$R/{lora,merged16,config.used.json} outputs/$R/ || { note "fetch FAILED"; exit 1; }
ssh "${O[@]}" lapwing.dolkens.net "mkdir -p ~/home-ai/outputs/$R"
rsync -a -e "ssh ${O[*]}" outputs/$R/lora outputs/$R/merged16 outputs/$R/config.used.json lapwing.dolkens.net:home-ai/outputs/$R/ || { note "push FAILED"; exit 1; }
ssh "${O[@]}" lapwing.dolkens.net "cd ~/home-ai && echo 'merged on cassowary (Unsloth)' > outputs/$R/merge.log && \
  OLLAMA_HOST=127.0.0.1:11500 PYTHONPATH=. HOMEAI_PROMPT_LAYOUT=user_context scripts/train-leaf-qwen.sh configs/$R.json $M --no-eval" | tail -2
note "import on lapwing: $(ssh "${O[@]}" lapwing.dolkens.net "OLLAMA_HOST=127.0.0.1:11500 ~/home-ai/ollama/bin/ollama list | grep -c '^$M:'")"
