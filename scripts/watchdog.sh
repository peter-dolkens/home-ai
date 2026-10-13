#!/usr/bin/env bash
# Machine-level backup watchdog for scripts/overnight.sh (installed in the user's crontab on cassowary).
# - relaunches the pipeline if the driver is not running and it hasn't finished
# - kills a training container whose log has not progressed for 45 minutes (driver then retries)
# Log: outputs/watchdog.log
cd "$(dirname "$0")/.." || exit 1
LOG=outputs/watchdog.log
STATUS=outputs/overnight.status
ts() { date '+%F %T'; }
mkdir -p outputs
# disk guard (28 Sep: our merges filled the disk Frigate's database lives on and stopped the camera recorder).
# Below 25G free: drop rebuildable merged16 dirs of models already in Ollama. Below 10G: stop our merge/training job.
# Runs even when STOP is set; never touches Frigate.
free_gb() { df -BG --output=avail /ai | tail -1 | tr -dc 0-9; }
if [ "$(free_gb)" -lt 25 ]; then
  have=$(docker exec ollama ollama list 2>/dev/null | awk 'NR>1{sub(":latest","",$1); print $1}')
  for d in outputs/*/merged16; do
    [ "$(free_gb)" -ge 25 ] && break
    m=$(python3 -c "import json,os;print(next((k for k,v in json.load(open('configs/models.json')).items() if not k.startswith('_') and v.get('config') and os.path.exists(v['config']) and json.load(open(v['config'])).get('out')=='${d%/merged16}'),''))" 2>/dev/null)
    if [ -n "$m" ] && echo "$have" | grep -qx "$m"; then
      rm -rf "$d"; echo "$(ts) DISK: removed $d ($m is in Ollama), $(free_gb)G free" | tee -a "$LOG" >> "$STATUS"
    fi
  done
fi
if [ "$(free_gb)" -lt 10 ]; then
  echo "$(ts) DISK: only $(free_gb)G free, stopping homeai-merge/homeai-train" | tee -a "$LOG" >> "$STATUS"
  docker stop homeai-merge homeai-train >/dev/null 2>&1
fi
# Stop only on an explicit marker: the plan grows (phase 2, phase 3 ...) and the driver is idempotent,
# so a relaunch after a "finished" line simply runs whatever steps are new and exits again.
if [ -f outputs/STOP ]; then
  echo "$(ts) STOP marker present; nothing to do" >> "$LOG"; exit 0
fi
# stall detection: training container up but no log write for 45 min
if docker ps -q -f name=homeai-train | grep -q .; then
  latest=$(ls -t outputs/*-train.log 2>/dev/null | head -1)
  if [ -n "$latest" ] && [ $(( $(date +%s) - $(stat -c %Y "$latest") )) -gt 2700 ]; then
    echo "$(ts) STALL: $latest silent for >45m, killing homeai-train" | tee -a "$LOG" >> "$STATUS"
    docker rm -f homeai-train >/dev/null 2>&1
  fi
fi
if ! pgrep -af "scripts/overnight.sh" | grep -v -- "--quality-evals" | grep -q .; then   # one-off eval workers are not the driver
  echo "$(ts) driver not running; relaunching overnight.sh" | tee -a "$LOG" >> "$STATUS"
  nohup setsid scripts/overnight.sh >> outputs/overnight.log 2>&1 < /dev/null &
else
  prog=$(tr '\r' '\n' < "$(ls -t outputs/*-train.log 2>/dev/null | head -1)" 2>/dev/null | grep -E "/[0-9]+ \[" | tail -1 | tr -s ' ')
  echo "$(ts) ok: $(tail -1 "$STATUS" 2>/dev/null | cut -c21-80) | ${prog:-no training log}" >> "$LOG"
fi
