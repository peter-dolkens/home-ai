#!/usr/bin/env bash
# Run the voice round trip (homeai.voice.roundtrip) in the homeai-voice container, CPU only (deploy/voice/Dockerfile).
#   scripts/voice-roundtrip.sh --fetch --stt tiny,base,small,parakeet
#   scripts/voice-roundtrip.sh --sets probe --stt base,parakeet --conditions clean,farfield,noisy10,noisy5,lowmic
#   scripts/voice-roundtrip.sh --latency 40 --stt tiny,base,small      # keep the machine otherwise quiet
#   scripts/voice-roundtrip.sh --summary --summary-sets probe
#   VOICE_CACHE=/path/to/cache CPUS=8 MEM_GB=9 scripts/voice-roundtrip.sh ...     # e.g. on magpie (10 GB VM)
# The repo is mounted at /work (outputs land in data/generated/heard), models + audio + transcripts persist in
# $VOICE_CACHE at /cache. CPUS / MEM_GB default to the Docker VM's size minus 1 GB; roundtrip.py sizes its workers
# (processes x threads) to them. HOMEAI_HOST records which Mac measured any timing.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-homeai-voice}
CACHE=${VOICE_CACHE:-$HOME/.cache/homeai-voice}
CPUS=${CPUS:-$(docker info --format '{{.NCPU}}')}
MEM_GB=${MEM_GB:-$(( ($(docker info --format '{{.MemTotal}}') + 536870912) / 1073741824 - 1 ))}
mkdir -p "$CACHE"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" "$ROOT/deploy/voice"
mode=""
if [ -t 0 ] && [ -t 1 ]; then mode="-it"; fi
exec docker run --rm $mode --cpus "$CPUS" -m "${MEM_GB}g" \
  -e HOMEAI_HOST="$(hostname -s)" -e HOMEAI_HOST_LOAD="$(sysctl -n vm.loadavg 2>/dev/null || cat /proc/loadavg)" \
  -e HOMEAI_HARDWARE="$(cd "$ROOT" && python3 -c 'from homeai.host import host_info; print(host_info()["hardware"])')" \
  -e HOMEAI_HOST_NOTE="docker $(docker info --format '{{.OSType}}/{{.Architecture}}') VM, $CPUS cpus, ${MEM_GB} GB" \
  -e HOMEAI_VOICE_MEM_GB="$MEM_GB" \
  -v "$ROOT":/work -w /work -v "$CACHE":/cache \
  "$IMAGE" --cpus "$CPUS" "$@"
