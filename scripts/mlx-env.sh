#!/usr/bin/env bash
# Self-contained MLX training environment for frogmouth (MacBook Air M5, 32 GB): the venv, the Python it runs on, every
# package / model cache and the Ollama model store live under .mlx/ in the repo. Nothing goes to Homebrew, global pip,
# ~/.cache or the shell profile; `rm -rf .mlx` undoes all of it.
#   scripts/mlx-env.sh setup                      # create .mlx/venv (uv-managed Python 3.12) and install the packages
#   scripts/mlx-env.sh python -m homeai.train.train_mlx --config configs/qwen35-2b-lora-v5.json
#   scripts/mlx-env.sh ollama-serve               # this Mac's Ollama 0.21.2 on $OLLAMA_HOST, models in .mlx/ollama-models
#   source scripts/mlx-env.sh                     # the same environment in the current shell
_here=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)
export HOMEAI_ROOT=$_here MLX_ROOT=$_here/.mlx
export UV_CACHE_DIR=$MLX_ROOT/cache/uv UV_PYTHON_INSTALL_DIR=$MLX_ROOT/python UV_PYTHON_BIN_DIR=$MLX_ROOT/bin \
       UV_TOOL_DIR=$MLX_ROOT/uv-tools UV_TOOL_BIN_DIR=$MLX_ROOT/bin UV_NO_MODIFY_PATH=1 UV_PYTHON_PREFERENCE=only-managed
export HF_HOME=$MLX_ROOT/hf HF_HUB_DISABLE_TELEMETRY=1 PIP_CACHE_DIR=$MLX_ROOT/cache/pip TORCH_HOME=$MLX_ROOT/cache/torch \
       XDG_CACHE_HOME=$MLX_ROOT/cache MPLCONFIGDIR=$MLX_ROOT/cache/matplotlib TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$_here${PYTHONPATH:+:$PYTHONPATH}
# Ollama: the same version as cassowary / lapwing (0.21.2, see scripts/convert-on-mac.sh), on its own port and model store
export OLLAMA=${OLLAMA:-$MLX_ROOT/ollama-0.21.2/ollama}
export OLLAMA_HOST=${OLLAMA_HOST:-127.0.0.1:11437} OLLAMA_MODELS=$MLX_ROOT/ollama-models
# recorded in every training log and eval report (homeai/host.py): throughput and latency only compare within one host
export HOMEAI_HOST=${HOMEAI_HOST:-$(scutil --get LocalHostName 2>/dev/null | tr '[:upper:]' '[:lower:]' || hostname -s)}
export HOMEAI_HARDWARE=${HOMEAI_HARDWARE:-$(sysctl -n machdep.cpu.brand_string | tr '[:upper:]' '[:lower:]' | tr -d ' ' | sed 's/^apple/apple-/')}
if [ "$HOMEAI_HOST" = frogmouth ] && [ -z "${HOMEAI_HOST_NOTE:-}" ]; then
  export HOMEAI_HOST_NOTE="Apple GPU (Metal); a Docker speech-to-text job (10 CPUs, 7 GB) shares this machine: latency indicative only, not comparable with the CUDA boxes"
fi
mkdir -p "$MLX_ROOT/bin" "$UV_CACHE_DIR" "$HF_HOME" "$OLLAMA_MODELS"
UV=$MLX_ROOT/bin/uv
[ -f "$MLX_ROOT/venv/bin/activate" ] && . "$MLX_ROOT/venv/bin/activate"

# sourced: stop here with the environment set
(return 0 2>/dev/null) && return 0
set -euo pipefail
case "${1:-}" in
  setup)
    [ -x "$UV" ] || { echo "put a uv binary at $UV (standalone build from github.com/astral-sh/uv/releases)"; exit 1; }
    [ -d "$MLX_ROOT/venv" ] || "$UV" venv --python 3.12 "$MLX_ROOT/venv"
    # torch (CPU / MPS) only for the transformers reference forward pass in the validation step: `setup --no-torch`
    # on a training-only host with little disk (falcon)
    TORCH=torch; [ "${2:-}" = "--no-torch" ] && TORCH=
    VIRTUAL_ENV=$MLX_ROOT/venv "$UV" pip install --python "$MLX_ROOT/venv/bin/python" \
      "mlx>=0.29" "mlx-lm>=0.28" transformers safetensors numpy jinja2 huggingface_hub $TORCH
    ;;
  ollama-serve)
    # GPU discovery has a 30 s timeout that a cold Metal-library compile on a busy machine can miss; the server then
    # runs everything on 4 CPU threads (a minute per call). Restart until it reports Metal (3 tries).
    for try in 1 2 3; do
      if ! "$OLLAMA" list >/dev/null 2>&1; then
        nohup "$OLLAMA" serve > "$MLX_ROOT/ollama-serve.log" 2>&1 < /dev/null &
        until grep -q "inference compute" "$MLX_ROOT/ollama-serve.log" 2>/dev/null; do sleep 2; done
      fi
      grep "inference compute" "$MLX_ROOT/ollama-serve.log" | grep -q "library=Metal" && break
      echo "ollama serve fell back to CPU (try $try): restarting"
      pkill -f "$OLLAMA serve"; sleep 3
    done
    "$OLLAMA" --version
    ;;
  "") echo "usage: scripts/mlx-env.sh setup | ollama-serve | COMMAND..."; exit 2 ;;
  *) exec "$@" ;;
esac
