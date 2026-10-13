#!/usr/bin/env bash
# Falcon MLX artifacts for any archived PEFT run, built on frogmouth and pushed into falcon's stage/ (the per-run sibling of
# scripts/falcon-phase2-build.sh, which builds v5's MLX and GGUF variants; homeai/eval/falcon_stream.py JOBS):
#   1. the run's LoRA adapter + config.used.json from Piper (else cassowary's outputs, read-only) -> outputs/<run>/
#   2. fused into the local base (default .mlx/bases/Qwen3.5-4B) -> outputs/<run>/merged16 (homeai.train.fuse_peft)
#   3. mlx_lm.convert per variant: mlx4 = 4-bit g64 (default), mlx4g32 = 4-bit g32, mlx6 / mlx8 = 6- / 8-bit g64 -> .mlx/exports/falcon/mlx/<name>-<variant>
#   4. pushed to falcon stage/mlx/<name>-<variant> (+ .ready) under the 12 GB cap, between timed steps (stage/HOLD handshake)
# Idempotent: finished pieces are skipped. Run detached (macOS has no setsid):
#   nohup scripts/falcon-mlx-build.sh qwen35-4b-lora-v10 homeai-4b-lora-v10 mlx4 > outputs/falcon-mlx-build-v10.log 2>&1 < /dev/null & disown
# The prompt layout the model was trained on (v10: stable_prefix) is not in the MLX dir: the falcon job carries it (Job.layout).
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/mlx-env.sh
note() { echo "$(date '+%F %T') $*"; }
RUN_ID=${1:?usage: $0 <run> <name> [variant ...]}; NAME=${2:?usage: $0 <run> <name> [variant ...]}; shift 2
VARIANTS=("${@:-mlx4}")
RUN=outputs/$RUN_ID; X=$MLX_ROOT/exports/falcon; mkdir -p "$X/mlx" "$RUN"
E="ssh -oControlPath=$HOME/.ssh/cm-%C -oControlMaster=auto -oControlPersist=30m -oBatchMode=yes -oHostName=10.24.1.110 -oHostKeyAlias=falcon.dolkens.net"
F=($E falcon.dolkens.net)
FW='~/home-ai-falcon'

if [ ! -f "$RUN/lora/adapter_model.safetensors" ]; then
  note "fetch $RUN_ID lora"
  ssh -oBatchMode=yes piper.dolkens.net "tar -C home-ai-archive/runs/$RUN_ID -cf - lora config.used.json" | tar -xf - -C "$RUN" ||
    ssh -oBatchMode=yes cassowary.dolkens.net "tar -C /ai/experiments/home-ai/outputs/$RUN_ID -cf - lora config.used.json" | tar -xf - -C "$RUN"
fi
[ -f "$RUN/lora/adapter_model.safetensors" ] || { note "no adapter for $RUN_ID"; exit 1; }
BASE=${BASE:-$MLX_ROOT/bases/$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['base_model'].split('/')[-1])" "$RUN/config.used.json")}
[ -f "$RUN/merged16/config.json" ] || { note "fuse $RUN_ID on $BASE"; python -m homeai.train.fuse_peft --lora "$RUN/lora" --out "$RUN/merged16" --base "$BASE"; }

for n in "${VARIANTS[@]}"; do
  case $n in
    mlx4) q="-q --q-bits 4 --q-group-size 64" ;;
    mlx4g32) q="-q --q-bits 4 --q-group-size 32" ;;
    mlx6) q="-q --q-bits 6 --q-group-size 64" ;;
    mlx8) q="-q --q-bits 8 --q-group-size 64" ;;
    *) note "unknown variant $n"; exit 1 ;;
  esac
  d="$X/mlx/$NAME-$n"
  [ -f "$d/config.json" ] && continue
  note "mlx_lm.convert $n"
  rm -rf "$d"
  # shellcheck disable=SC2086
  python -m mlx_lm convert --hf-path "$RUN/merged16" --mlx-path "$d" $q
done

push() {  # push <local path> <stage-relative name>: waits for room under the cap and a pause between timed steps
  local src=$1 name=$2 need
  "${F[@]}" "test -f $FW/stage/$name.ready" 2>/dev/null && return 0
  need=$(du -sk "$src" | cut -f1)
  # room under the cap, and one transfer at a time: another builder (falcon-phase2-build.sh) owns stage/HOLD until it is
  # done, and its files count against the cap, so the room is checked again after waiting for it
  until [ $(( $("${F[@]}" "du -sk $FW" 2>/dev/null | cut -f1) + need )) -lt $(( 11500 * 1024 )) ] &&
        ! "${F[@]}" "test -f $FW/stage/HOLD" 2>/dev/null; do sleep 120; done
  "${F[@]}" "touch $FW/stage/HOLD"
  until "${F[@]}" "test -f $FW/stage/HELD || ! pgrep -f 'homeai[.]eval[.]falcon_strea[m]' >/dev/null || grep -q waiting $FW/logs/state.json" 2>/dev/null; do sleep 20; done
  note "push $name ($((need / 1024)) MB)"
  "${F[@]}" "mkdir -p $FW/stage/$(dirname "$name")"
  rsync -a --partial -e "$E" "$src" "falcon.dolkens.net:home-ai-falcon/stage/$(dirname "$name")/"
  "${F[@]}" "touch $FW/stage/$name.ready; rm -f $FW/stage/HOLD"
}
for n in "${VARIANTS[@]}"; do push "$X/mlx/$NAME-$n" "mlx/$NAME-$n"; done
note "$NAME MLX artifacts pushed: ${VARIANTS[*]}"
