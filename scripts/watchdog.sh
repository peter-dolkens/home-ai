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
