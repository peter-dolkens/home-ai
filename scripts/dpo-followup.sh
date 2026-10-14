#!/usr/bin/env bash
# 14 Oct: after cassowary's outputs/dpo-train.sh (DPO round 1 on the pick, merged16 on /ai/models): bring the run to
# frogmouth and lapwing, import it into lapwing's Ollama (train-leaf-qwen.sh skips training and the merge when lora/ and
# merged16/ exist), then the same fan-out as the small LFMs (scripts/small-lfm-followup.sh: lapwing heard evals,
# cassowary import + benchmark, falcon MLX 6-bit + stream). Run from frogmouth, detached:
#   nohup scripts/dpo-followup.sh lfm25-1.2b-dpo-b-v1 > outputs/followup-dpo.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=$1; M=homeai-$RUN
note() { echo "$(date '+%F %T') $M: $*"; }
O=(-oBatchMode=yes -oControlPath=$HOME/.ssh/cm-%C -oControlMaster=auto -oControlPersist=30m)
L=(ssh "${O[@]}" lapwing.dolkens.net); C=(ssh "${O[@]}" cassowary.dolkens.net)
until "${C[@]}" "test -f /ai/experiments/home-ai/outputs/$RUN/merged16/config.json"; do
  "${C[@]}" "grep -q 'dpo configs/$RUN.json: FAILED' /ai/experiments/home-ai/outputs/overnight-status.log" && { note "DPO failed on cassowary"; exit 1; }
  sleep 300
done
note "merged on cassowary"
mkdir -p "outputs/$RUN"
rsync -a -e "ssh ${O[*]}" "cassowary.dolkens.net:/ai/experiments/home-ai/outputs/$RUN/" "outputs/$RUN/" || { note "fetch FAILED"; exit 1; }
"${L[@]}" "mkdir -p home-ai/outputs/$RUN"
rsync -a -e "ssh ${O[*]}" "outputs/$RUN/lora" "outputs/$RUN/merged16" "outputs/$RUN/config.used.json" "outputs/$RUN/merge.log" \
  "lapwing.dolkens.net:home-ai/outputs/$RUN/" || { note "copy to lapwing FAILED"; exit 1; }
"${L[@]}" "cd ~/home-ai && scripts/train-leaf-qwen.sh configs/$RUN.json $M --no-eval" > "outputs/$RUN/lapwing-import.log" 2>&1 \
  && note "imported on lapwing" || { note "lapwing import FAILED"; exit 1; }
exec scripts/small-lfm-followup.sh "$RUN" mlx6
