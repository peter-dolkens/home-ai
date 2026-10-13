#!/usr/bin/env bash
# Falcon phase 2 artifacts, built on frogmouth and pushed into falcon's stage/ (homeai/eval/falcon_stream.py JOBS):
#   1. the v5 LoRA adapter (Piper) fused into Qwen3.5-4B -> outputs/qwen35-4b-lora-v5/merged16 (homeai.train.fuse_peft;
#      BASE = a local copy of the HF snapshot, default .mlx/bases/Qwen3.5-4B, copied from cassowary's HF cache)
#   2. quantisations of it: Q4_K_M with Ollama 0.21.2 (.mlx, as cassowary; the control: the rebuild is not bit-identical
#      to the archived Q4_K_M), Q4_0 / IQ4_XS / Q5_K_M with llama.cpp's llama-quantize from Ollama's F16 conversion
#      (Ollama refuses to requantise a Q4 GGUF and 0.21.2 only offers F16 / Q4_K_S / Q4_K_M / Q8_0)
#   3. mlx_lm.convert 4-bit, group size 64 (and 32: on the 0.8B g64 cost 12 points, g32 2)
#   4. push each one to falcon when its stage/ has room under the 12 GB cap, between timed steps (stage/HOLD handshake)
# Idempotent: finished pieces are skipped.
#   nohup scripts/falcon-phase2-build.sh > outputs/falcon-phase2-build.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/mlx-env.sh
note() { echo "$(date '+%F %T') $*"; }
RUN=outputs/qwen35-4b-lora-v5; NAME=homeai-4b-lora-v5; X=$MLX_ROOT/exports/falcon; mkdir -p "$X/mlx"
ARCHIVED=11944e16076c0ee0bd2b9af6cba21780dc79b27bdf68976fcfd9c89729737c2e  # the Q4_K_M cassowary serves
E="ssh -oControlPath=$HOME/.ssh/cm-%C -oControlMaster=auto -oControlPersist=30m -oBatchMode=yes -oHostName=10.24.1.110 -oHostKeyAlias=falcon.dolkens.net"
F=($E falcon.dolkens.net)
FW='~/home-ai-falcon'

[ -f "$RUN/lora/adapter_model.safetensors" ] || { mkdir -p "$RUN"; ssh piper.dolkens.net "tar -C home-ai-archive/runs/qwen35-4b-lora-v5 -cf - lora" | tar -xf - -C "$RUN"; }
[ -f "$RUN/merged16/config.json" ] || { note "fuse"; python -m homeai.train.fuse_peft --lora "$RUN/lora" --out "$RUN/merged16" --base "${BASE:-$MLX_ROOT/bases/Qwen3.5-4B}"; }

scripts/mlx-env.sh ollama-serve >/dev/null
MF() { printf 'FROM %s\nTEMPLATE {{ .Prompt }}\nRENDERER qwen3.5\nPARSER qwen3.5\nPARAMETER num_ctx 8192\nPARAMETER temperature 0.2\nPARAMETER top_k 20\nPARAMETER top_p 0.9\n' "$1"; }
blob() { "$OLLAMA" show --modelfile "$1" | awk '/^FROM /{print $2}'; }
# Ollama 0.21.2 quantises only to F16 / Q4_K_S / Q4_K_M / Q8_0: the Q4_K_M control comes from Ollama itself
if [ ! -f "$X/$NAME-q4_K_M-rebuilt.gguf" ]; then
  MF "$PWD/$RUN/merged16" > "$X/q4_K_M.Modelfile"
  note "ollama create --quantize q4_K_M"
  "$OLLAMA" create "$NAME-falcon-q4km" --quantize q4_K_M -f "$X/q4_K_M.Modelfile" > "$X/q4_K_M.create.log" 2>&1
  cp "$(blob "$NAME-falcon-q4km")" "$X/q4.part" && mv "$X/q4.part" "$X/$NAME-q4_K_M-rebuilt.gguf"
  "$OLLAMA" rm "$NAME-falcon-q4km" >/dev/null
fi
# the rest with llama.cpp's llama-quantize (a release build under .mlx/llama.cpp) from Ollama's own F16 conversion, so the
# tensor names and metadata stay Ollama's: its Q4_K_M from that F16 is byte-identical to Ollama's wherever the two pick
# the same tensor type (32 of 834 tensors get Q6_K vs Q4_K the other way round). Vision tensors stay F16, as Ollama keeps them.
Q=$MLX_ROOT/llama.cpp/llama-${LLAMA_CPP:-b11243}/llama-quantize
if [ ! -x "$Q" ]; then
  mkdir -p "$MLX_ROOT/llama.cpp"
  curl -sSL "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP:-b11243}/llama-${LLAMA_CPP:-b11243}-bin-macos-arm64.tar.gz" | tar -xzf - -C "$MLX_ROOT/llama.cpp"
fi
F16=$X/$NAME-f16-llamacpp.gguf
for q in Q4_0 IQ4_XS Q5_K_M; do
  out="$X/$NAME-$(echo $q | tr '[:upper:]' '[:lower:]').gguf"
  [ -f "$out" ] && continue
  if [ ! -f "$F16" ]; then
    MF "$PWD/$RUN/merged16" > "$X/f16.Modelfile"
    "$OLLAMA" create "$NAME-falcon-f16" --quantize F16 -f "$X/f16.Modelfile" > "$X/f16.create.log" 2>&1
    python3 scripts/gguf_kv.py "$(blob "$NAME-falcon-f16")" "$F16" qwen35.rope.dimension_sections 11,11,10,0
    "$OLLAMA" rm "$NAME-falcon-f16" >/dev/null
  fi
  note "llama-quantize $q"
  "$Q" --tensor-type '^v\..*=f16' "$F16" "$X/tmp-$q.gguf" "$q" 2>&1 | grep -E "quant size|error|fail" || true
  python3 scripts/gguf_kv.py "$X/tmp-$q.gguf" "$out.part" qwen35.rope.dimension_sections 11,11,10 && mv "$out.part" "$out"
  rm -f "$X/tmp-$q.gguf"
done
rm -f "$F16"
for v in "mlx4:-q --q-bits 4 --q-group-size 64" "mlx4g32:-q --q-bits 4 --q-group-size 32"; do
  n=${v%%:*}; d="$X/mlx/$NAME-$n"
  [ -f "$d/config.json" ] && continue
  note "mlx_lm.convert $n"
  # shellcheck disable=SC2086
  python -m mlx_lm convert --hf-path "$RUN/merged16" --mlx-path "$d" ${v#*:}
done

push() {  # push <local path> <stage-relative name>: waits for room under the cap and a pause between timed steps
  local src=$1 name=$2 need
  "${F[@]}" "test -f $FW/stage/$name.ready" 2>/dev/null && return 0
  need=$(du -sk "$src" | cut -f1)
  until [ $(( $("${F[@]}" "du -sk $FW" 2>/dev/null | cut -f1) + need )) -lt $(( 11500 * 1024 )) ]; do sleep 120; done
  "${F[@]}" "touch $FW/stage/HOLD"
  until "${F[@]}" "test -f $FW/stage/HELD || ! pgrep -f 'homeai[.]eval[.]falcon_strea[m]' >/dev/null || grep -q waiting $FW/logs/state.json" 2>/dev/null; do sleep 20; done
  note "push $name ($((need / 1024)) MB)"
  "${F[@]}" "mkdir -p $FW/stage/$(dirname "$name")"
  rsync -a --partial -e "$E" "$src" "falcon.dolkens.net:home-ai-falcon/stage/$(dirname "$name")/"
  "${F[@]}" "touch $FW/stage/$name.ready; rm -f $FW/stage/HOLD"
}
push "$X/mlx/$NAME-mlx4" "mlx/$NAME-mlx4"
push "$X/mlx/$NAME-mlx4g32" "mlx/$NAME-mlx4g32"
if [ -f "$X/$NAME-q4_K_M-rebuilt.gguf.identical" ]; then  # the baseline already is this model: drop the control job
  "${F[@]}" "touch $FW/stage/$NAME-q4_K_M-rebuilt.gguf.skip"
else
  push "$X/$NAME-q4_K_M-rebuilt.gguf" "$NAME-q4_K_M-rebuilt.gguf"
fi
for q in q4_0 iq4_xs q5_k_m; do push "$X/$NAME-$q.gguf" "$NAME-$q.gguf"; done
note "phase 2 artifacts pushed"
