#!/usr/bin/env bash
# Start the falcon stream once it is idle and stage/<artifact>.ready is there, so a job whose artifact frogmouth pushes
# into the gap (after the running stream exits on stage/RESTART, which frees its disk) goes first; after MAX_WAIT minutes
# idle it starts regardless. RESUME_PID: a builder paused with kill -STOP (so its push cannot take the room first) is
# resumed with kill -CONT once the stream is started. Detached:
#   nohup scripts/falcon-start-when-ready.sh mlx/homeai-4b-lora-v10-mlx4 > outputs/falcon-start-when-ready.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: $0 <stage-relative artifact>}; MAX_WAIT=${MAX_WAIT:-40}
F=(ssh -oControlPath=$HOME/.ssh/cm-%C -oControlMaster=auto -oControlPersist=30m -oBatchMode=yes -oHostName=10.24.1.110 -oHostKeyAlias=falcon.dolkens.net falcon.dolkens.net)
note() { echo "$(date '+%F %T') $*"; }
idle=0
while sleep 60; do
  if "${F[@]}" "pgrep -f 'homeai[.]eval[.]falcon_stream' >/dev/null"; then idle=0; continue; fi
  idle=$((idle + 1))
  if "${F[@]}" "test -f ~/home-ai-falcon/stage/$NAME.ready && test ! -f ~/home-ai-falcon/stage/HOLD" || [ $idle -ge "$MAX_WAIT" ]; then
    note "stream idle ${idle} min; $NAME $("${F[@]}" "test -f ~/home-ai-falcon/stage/$NAME.ready && echo ready || echo 'not ready (timeout)'"): starting"
    scripts/falcon-pull.sh start
    [ -n "${RESUME_PID:-}" ] && kill -CONT "$RESUME_PID" && note "resumed $RESUME_PID"
    exit 0
  fi
done
