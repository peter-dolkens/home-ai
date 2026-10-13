"""Which machine a training log or eval report came from. Several hosts of different specs run jobs (cassowary RTX 3090,
lapwing RTX 3080, frogmouth M5, falcon M2 Pro): throughput and latency are only comparable within one host.
scripts/mlx-env.sh sets HOMEAI_HOST / HOMEAI_HARDWARE / HOMEAI_HOST_NOTE (falcon_stream also HOMEAI_OLLAMA_VERSION); elsewhere the hostname and GPU are detected."""
from __future__ import annotations

import os
import platform
import re
import socket
import subprocess


def _hardware() -> str:
    try:
        if platform.system() == "Darwin":  # "Apple M5" -> apple-m5, "Apple M2 Pro" -> apple-m2pro
            brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout
            return re.sub(r"^apple", "apple-", re.sub(r"\s+", "", brand.strip().lower()))
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout
        if gpu.strip():  # "NVIDIA GeForce RTX 3090" -> rtx-3090
            return re.sub(r"\s+", "-", re.sub(r"^(nvidia\s+)?(geforce\s+)?", "", gpu.splitlines()[0].strip().lower()))
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.machine()


def host_info() -> dict:
    info = {"host": os.environ.get("HOMEAI_HOST") or socket.gethostname().split(".")[0],
            "hardware": os.environ.get("HOMEAI_HARDWARE") or _hardware()}
    if os.environ.get("HOMEAI_HOST_NOTE"):
        info["host_note"] = os.environ["HOMEAI_HOST_NOTE"]
    if os.environ.get("HOMEAI_OLLAMA_VERSION"):  # the Ollama build serving the model (falcon runs several side by side)
        info["ollama_version"] = os.environ["HOMEAI_OLLAMA_VERSION"]
    return info
