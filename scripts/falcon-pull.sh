#!/usr/bin/env bash
# Falcon low-power stream, frogmouth side (homeai/eval/falcon_stream.py runs on falcon):
#   scripts/falcon-pull.sh sync     # code + test sets -> falcon:~/home-ai-falcon/repo (rsync, no --delete)
#   scripts/falcon-pull.sh start    # start the stream on falcon (or restart it between steps, with the synced code)
#   scripts/falcon-pull.sh          # pull reports back, rebuild outputs/FALCON_CHAMPIONS.* and outputs/reconcile-falcon here
#   scripts/falcon-pull.sh status   # the stream's job list, log tail and footprint
# The stream runs on falcon's /usr/bin/python3 (Xcode's 3.9): code it imports must stay 3.9-compatible (no match, and
# X | None only under from __future__ import annotations).
set -euo pipefail
cd "$(dirname "$0")/.."
E="ssh -oControlPath=$HOME/.ssh/cm-%C -oControlMaster=auto -oControlPersist=30m -oBatchMode=yes -oHostName=10.24.1.110 -oHostKeyAlias=falcon.dolkens.net"
F="$E falcon.dolkens.net"
R=falcon.dolkens.net:home-ai-falcon/repo
case "${1:-pull}" in
  sync)
    rsync -a -e "$E" --exclude __pycache__ homeai "$R/"
    rsync -a -e "$E" --relative data/generated/test-v2.jsonl data/generated/wild.jsonl data/generated/wild2.jsonl \
      data/generated/bigbench-b.jsonl data/generated/test-v2-b.jsonl data/generated/wild-b.jsonl data/generated/wild2-b.jsonl \
      data/house/bundook-project.json data/house/ha-entity-profiles.json data/generated/bigbench-b-bundook.jsonl data/generated/bigbench-bundook.jsonl "$R/" ;;
  start)
    # the new stream waits on the running one's lock (stage/RESTART makes that one exit between steps); the [t] keeps
    # pgrep from matching this command line itself
    $F 'W=~/home-ai-falcon; pgrep -f "falcon_stream --wai[t]" >/dev/null && { echo "a (re)start is already queued"; exit; }
        # only a running stream is told to exit: a RESTART left behind would stop the new one at its first step
        pgrep -f "homeai[.]eval[.]falcon_stream" >/dev/null && touch $W/stage/RESTART; cd $W/repo
        (nohup /usr/bin/python3 -m homeai.eval.falcon_stream --wait >> ../logs/stream.log 2>&1 < /dev/null &)
        echo "stream (re)start queued"' ;;
  status)
    $F 'cd ~/home-ai-falcon/repo && /usr/bin/python3 -m homeai.eval.falcon_stream --status; tail -5 ../logs/stream.log; df -h ~ | tail -1' ;;
  pull)
    mkdir -p outputs/eval-falcon outputs/reconcile-falcon
    rsync -a -e "$E" "$R/outputs/eval-falcon/" outputs/eval-falcon/
    rsync -a -e "$E" "$R/outputs/reconcile-falcon/" outputs/reconcile-falcon/ 2>/dev/null || true
    python3 -m homeai.eval.reconcile_grid --eval-dir outputs/eval-falcon --out outputs/reconcile-falcon > /dev/null
    python3 -m homeai.eval.host_champions --host falcon --eval-dir outputs/eval-falcon ;;
esac
