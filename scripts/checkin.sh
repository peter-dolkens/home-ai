#!/usr/bin/env bash
# One status snapshot across every machine in the home-ai fleet (Peter, 29 Sep: 15-minute check-ins that don't depend
# on in-session crons). Run from frogmouth; a background loop sleeps and runs this, so its exit wakes the session.
#   scripts/checkin.sh [sleep_seconds]
sleep "${1:-0}"
C=(-oControlPath="$HOME/.ssh/cm-%C" -oControlMaster=auto -oControlPersist=30m -oConnectTimeout=10 -oBatchMode=yes)
date '+%F %T'

echo "== cassowary"
ssh "${C[@]}" cassowary.dolkens.net 'cd /ai/experiments/home-ai
  echo "disk: $(df -BG --output=avail / | tail -1 | tr -d " ") free"
  pgrep -f "^bash scripts/overnight.sh" >/dev/null && echo "driver: running" || echo "driver: NOT running"
  docker ps --format "{{.Names}}: {{.Status}}" | grep -E "^(homeai|frigate)"
  echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
  grep -E "^2026" outputs/overnight.log | grep -vE ": exists$|already done|not rebuilding" | tail -3 | cut -c1-150
  tr "\r" "\n" < "$(ls -t outputs/*-train.log | head -1)" | grep -E "[0-9]+/[0-9]+ \[" | tail -1 | cut -c1-90' 2>&1

echo "== lapwing"
# 6 Oct: lapwing's wired port lost its DHCP lease and it fell back to Wi-Fi (10.24.10.8) while DNS still gives the wired address
LW=()
nc -z -G 3 lapwing.dolkens.net 22 >/dev/null 2>&1 || LW=(-oHostName=10.24.10.8 -oHostKeyAlias=lapwing.dolkens.net)
ssh "${C[@]}" "${LW[@]}" lapwing.dolkens.net 'cd ~/home-ai
  echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
  pgrep -af "homeai.eval.e2e|train-leaf|merge16" | grep -v pgrep | sed -E "s/--data [^ ]*//" | cut -c1-110
  tail -1 outputs/heard-evals.log 2>/dev/null | cut -c1-150' 2>&1

echo "== frogmouth"
ps -Ao etime,args | grep -E "train_mlx|homeai.eval|ollama (create|runner)" | grep -v grep | cut -c1-110
for f in "$HOME"/Projects/home-ai/.mlx/outputs/*/train.log; do [ -f "$f" ] && [ "$(find "$f" -mmin -30)" ] && echo "$(basename "$(dirname "$f")"): $(grep -E "^step" "$f" | tail -1 | cut -c1-90)"; done

echo "== falcon"
ssh "${C[@]}" -oHostName=10.24.1.110 -oHostKeyAlias=falcon.dolkens.net falcon.dolkens.net '
  echo "disk: $(df -h / | tail -1 | awk "{print \$4}") free"
  pgrep -fl "home-ai-falcon|homeai.eval" | grep -v pgrep | cut -c1-110 || echo "nothing of ours running"
  f=$(find ~/home-ai-falcon -maxdepth 3 -name "*.log" -mmin -60 -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | head -1); [ -n "$f" ] && echo "$(basename "$f"): $(tail -1 "$f" | cut -c1-140)"' 2>&1

echo "== magpie (Peter's laptop: expect nothing of ours)"
ssh "${C[@]}" -oHostName=10.24.10.150 magpie.dolkens.net 'pgrep -fl "home-ai|ollama" | grep -v pgrep | cut -c1-90 || echo "clear"' 2>/dev/null || echo "offline"
