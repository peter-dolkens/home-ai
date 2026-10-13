#!/usr/bin/env bash
# Rebuild the results page from every host's reports (Peter, 14 Oct: update it whenever training or benchmarking completes).
#   scripts/report-refresh.sh <scratchpad dir>      -> <scratchpad>/voice-lab-picks.html (publish it to the same artifact URL)
set -uo pipefail
cd "$(dirname "$0")/.."
S=${1:?scratchpad dir}; W=$S/reportwork3; mkdir -p $W/outputs; ln -sfn "$PWD/configs" $W/configs; ln -sfn "$PWD/data" $W/data
O=(-oBatchMode=yes -oControlPath=$HOME/.ssh/cm-%C -oControlMaster=auto -oControlPersist=30m)
rsync -a -e "ssh ${O[*]}" --include='*/' --include='report.json' --include='latency.json' --include='prefix.json' --exclude='*' cassowary.dolkens.net:/ai/experiments/home-ai/outputs/eval/ $W/outputs/eval/
rsync -a -e "ssh ${O[*]}" --include='*/' --include='report.json' --exclude='*' lapwing.dolkens.net:home-ai/outputs/eval/ $W/outputs/eval/
rsync -a --include='*/' --include='report.json' --exclude='*' outputs/eval/ $W/outputs/eval/
scp -q "${O[@]}" cassowary.dolkens.net:/ai/experiments/home-ai/outputs/CHAMPIONS.json $W/outputs/
rsync -a -e "ssh ${O[*]}" cassowary.dolkens.net:/ai/experiments/home-ai/outputs/reconcile/ $W/outputs/reconcile/
scripts/falcon-pull.sh >/dev/null 2>&1; rm -rf $W/outputs/eval-falcon; cp -r outputs/eval-falcon outputs/FALCON_CHAMPIONS.json outputs/FALCON_CHAMPIONS.md $W/outputs/
cp -r outputs/leaf-results $W/outputs/ 2>/dev/null
cd $W && export PYTHONPATH=$OLDPWD
python3 -m homeai.eval.heard_summary --out outputs/HEARD >/dev/null 2>&1
python3 -m homeai.eval.design_champions >/dev/null 2>&1
python3 -m homeai.eval.report_v2 --out $S/voice-lab-picks.html --findings $S/findings3.json --status $S/status3.json
