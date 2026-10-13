"""Falcon low-power stream (Peter, 29 Sep): a separate champion set for models served on falcon (Mac mini M2 Pro, 10 CPU /
16 GPU cores, 32 GB, Metal). Falcon timings are only ever compared with other falcon runs, never with cassowary. Runs ON
falcon, detached and resumable (sessions restart often; macOS has no setsid):

  cd ~/home-ai-falcon/repo && nohup python3 -m homeai.eval.falcon_stream >> ../logs/stream.log 2>&1 < /dev/null & disown
  touch ../stage/RESTART; nohup python3 -m homeai.eval.falcon_stream --wait >> ../logs/stream.log 2>&1 < /dev/null &  # new code
  python3 -m homeai.eval.falcon_stream --status          # jobs, steps done, footprint

Containment: everything lives under ~/home-ai-falcon (W), at most CAP_GB on disk: ollama/ (the standalone Ollama 0.21.2
darwin build, the cassowary version; OLLAMA_MODELS=ollama/models), ollama-<version>/ (other Ollama builds a Job.ollama asks
for, each with its own models/ store; a GGUF already in another store is hard-linked, not copied), mlx/ (uv venv, Python and caches of the MLX server),
repo/ (this repo's code and test sets, rsynced from frogmouth), stage/ (downloads, and artifacts frogmouth pushes), logs/.
Only the system python3 (stdlib) runs the clients; nothing is installed outside W.

Phase 1 (baseline, as cassowary serves): the archived Q4_K_M GGUF (Piper, else cassowary's outputs, read-only), sha256
checked against the blob cassowary's Ollama serves, imported with cassowary's TEMPLATE / RENDERER / PARSER / PARAMETER
lines and server settings (KEEP_ALIVE=-1, NUM_PARALLEL=2); then the idle latency bench (--n 60, as cassowary) and the
outcome evals on test-v2, wild and wild2, one model loaded, nothing else of ours running.
Phase 2 (optimisations, after the v5 / v8 / LFM 1.2B / 2B baselines): rows <model>@falcon-<variant> for the warm prefix
bench, Ollama server and request settings, the MLX server and other quantisations; see JOBS.

A step whose report exists is skipped; an interrupted one reruns from scratch. A job whose artifacts frogmouth has not
pushed yet (stage/<name>.ready) waits while later jobs run. Before every timed step the stream honours stage/HOLD (it
answers with stage/HELD and waits), so frogmouth can copy files in without disturbing a measurement. After every job
homeai.eval.host_champions recomputes the falcon champions and every other model is removed to respect the disk cap.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

W = os.path.expanduser("~/home-ai-falcon")
OLLAMA_STOCK = "0.21.2"  # the cassowary version, at W/ollama; any other version lives at W/ollama-<version>
OLLAMA_URL = "https://github.com/ollama/ollama/releases/download/v{v}/ollama-darwin.tgz"


def ollama_dir(version: str) -> str:
    return f"{W}/ollama" if version == OLLAMA_STOCK else f"{W}/ollama-{version}"


OLLAMA = f"{W}/ollama/ollama"  # the build in use (use_ollama switches binary and store together)
MODELS = f"{W}/ollama/models"
STAGE = f"{W}/stage"
LOGS = f"{W}/logs"
MLX = f"{W}/mlx"
PORT, MLX_PORT, WARM_PORT = 11610, 11611, 11619
URL = f"http://127.0.0.1:{PORT}"
QUIET_LOAD = 4.0  # 1-minute load average a timed step waits for (10 cores; the stream itself adds ~1-2)
CAP_GB = 12.0
EVAL = "outputs/eval-falcon"
SETS = {"testv2": "data/generated/test-v2.jsonl", "wild": "data/generated/wild.jsonl", "wild2": "data/generated/wild2.jsonl"}
# design B (lookup tools, rooms-only prompt): each example's "toolset": "b" selects the B schemas in homeai.eval.e2e
SETS_B = {"bigbench-b": "data/generated/bigbench-b.jsonl", "testv2-b": "data/generated/test-v2-b.jsonl",
          "wild-b": "data/generated/wild-b.jsonl", "wild2-b": "data/generated/wild2-b.jsonl"}
# warm big-house timing for design B: the new house's own plan (3 houses, 7-8 requests each), grouped by house
PREFIX_B = ("--data", "data/generated/bigbench-b-bundook.jsonl", "--group-by", "house", "--groups", "3", "--per-group", "8")
PIPER, PIPER_DIR = "piper.dolkens.net", "home-ai-archive/runs"
CASSOWARY, CASSOWARY_DIR = "cassowary.dolkens.net", "/ai/experiments/home-ai/outputs"
# cassowary's Ollama container runs with these (docker inspect ollama); the baseline matches them
STOCK_ENV = {"OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_NUM_PARALLEL": "2"}
MLX_LM = "mlx-lm==0.31.3"  # the version frogmouth trains and converts with


@dataclasses.dataclass
class Job:
    row: str                            # report name: <model> (baseline) or <model>@falcon-<variant>
    model: str                          # Ollama model name (or the MLX model dir under stage/)
    run: str = ""                       # archive run holding gguf/<model>-q4_K_M.gguf (phase 1)
    like: str = ""                      # reuse this cassowary model's Modelfile lines (variants)
    gguf: str = ""                      # stage/<gguf> pushed from frogmouth instead of the archive
    params: dict = dataclasses.field(default_factory=dict)  # extra Modelfile PARAMETERs (num_batch)
    env: dict = dataclasses.field(default_factory=dict)     # extra Ollama server environment
    num_ctx: int = 8192
    mlx: bool = False                   # serve stage/<model> with mlx_lm.server instead of Ollama
    steps: tuple = ("latency", "e2e")
    layout: str = ""                    # prompt layout the model was trained on (homeai/prompt_layout.py; "" = classic)
    ollama: str = OLLAMA_STOCK          # Ollama build serving it (W/ollama-<version>, its own model store, same port)
    sets: dict = None                   # e2e test sets (default SETS; design B: SETS_B)
    prefix_args: tuple = ()             # extra homeai.eval.prefix_bench arguments (design B: the Bundook set, by house)
    note: str = "Ollama 0.21.2, cassowary settings (KEEP_ALIVE=-1, NUM_PARALLEL=2), Q4_K_M, num_ctx 8192"

    def needs(self) -> list[str]:
        """stage/ artifacts frogmouth pushes for this job (a GGUF or an MLX model dir, each with a .ready marker)"""
        return [x for x in (self.gguf, self.model if self.mlx else "") if x]


def base(model, run, **kw):
    return Job(model, model, run=run, **kw)


V5 = "homeai-4b-lora-v5"
V10 = "homeai-4b-lora-v10"
PREFIX = ("prefix",)
TIMING = ("latency", "prefix")
FULL = ("latency", "prefix", "e2e")
SP = "stable_prefix"
V11 = "homeai-4b-lora-v11"
O35 = "0.35.0"
O35_NOTE = "Ollama 0.35.0 (llama-server backend), cassowary settings (KEEP_ALIVE=-1, NUM_PARALLEL=2), Q4_K_M, num_ctx 8192"
JOBS = [
    # 13 Oct (Peter: focus on the small LFMs, and models for the Mac mini): the pick's recipe (design B, train-b-v8, r256,
    # 4 epochs) on LFM2.5 350M and 230M, and r128 on LFM2.5 2.6B; 6-bit as the pick, 8-bit too for the small ones
    *[Job(f"homeai-lfm25-{s}@falcon-{v}", f"mlx/homeai-lfm25-{s}-{v}", mlx=True, steps=("prefix", "e2e"), layout="user_context",
          sets=SETS_B, prefix_args=PREFIX_B, note=f"{MLX_LM} server, {v[3:]}-bit g64, prompt cache on, design B tools, user_context prompts")
      for s, vs in (("350m-lora-b-v8r256e4", ("mlx6", "mlx8")), ("230m-lora-b-v8r256e4", ("mlx6", "mlx8")),
                    ("350m-lora-b-v8r256e4s2", ("mlx6",)),
                    ("350m-lora-b-v8r256e6", ("mlx6",)), ("230m-lora-b-v8r256e6", ("mlx6",)),
                    ("350m-full-b-v8e3", ("mlx6",)), ("230m-full-b-v8e3", ("mlx6",)), ("1.2b-soup-b-v8r256e4", ("mlx6",)),
                    ("350m-soup-b-v8r256e4", ("mlx6",))) for v in vs],
    # 4 Oct: B-v4w, the 4B B-v2u plus its own verified design-B transcripts of the wild training conversations: 83.9% mean on
    # the 3090 (B-v2u 82.5%); the Mac's accurate design-B option (B-v2u: 83.2% at 3.7 s warm on falcon)
    Job("homeai-4b-lora-b-v4w@falcon-mlx6", "mlx/homeai-4b-lora-b-v4w-mlx6", mlx=True, steps=("prefix", "e2e"), layout="user_context",
        sets=SETS_B, prefix_args=PREFIX_B, note=f"{MLX_LM} server, 6-bit g64, prompt cache on, design B tools, user_context prompts"),
    # 9 Oct: LFM B-v8 r256 at 4 epochs, the house pick (two seeds: heard 81.9%, typed 89.3%, B sets 83.1%); 6-bit, the Mac's
    # deployment build
    Job("homeai-lfm25-1.2b-lora-b-v8r256e4@falcon-mlx6", "mlx/homeai-lfm25-1.2b-lora-b-v8r256e4-mlx6", mlx=True, steps=("prefix", "e2e"),
        layout="user_context", sets=SETS_B, prefix_args=PREFIX_B,
        note=f"{MLX_LM} server, 6-bit g64, prompt cache on, design B tools, user_context prompts"),
    # 6 Oct: LFM B-v7 at LoRA rank 256 (lapwing): best on the house (87.7% typed, 80.5% heard; on turns unseen in training
    # 81.2 / 74.8% vs r128 79.5 / 71.2%) though lower on the random-house B sets (78.5 vs 80.1%); 6-bit
    Job("homeai-lfm25-1.2b-lora-b-v7r256@falcon-mlx6", "mlx/homeai-lfm25-1.2b-lora-b-v7r256-mlx6", mlx=True, steps=("prefix", "e2e"),
        layout="user_context", sets=SETS_B, prefix_args=PREFIX_B,
        note=f"{MLX_LM} server, 6-bit g64, prompt cache on, design B tools, user_context prompts"),
    # 6 Oct: LFM B-v7 at LoRA rank 128 (lapwing): 80.1% B-set mean (r64 79.1), 86.0% typed on the house set (r64 84.6, Spark
    # B-v6n 85.5); 6-bit, as r32 / r64
    Job("homeai-lfm25-1.2b-lora-b-v7r128@falcon-mlx6", "mlx/homeai-lfm25-1.2b-lora-b-v7r128-mlx6", mlx=True, steps=("prefix", "e2e"),
        layout="user_context", sets=SETS_B, prefix_args=PREFIX_B,
        note=f"{MLX_LM} server, 6-bit g64, prompt cache on, design B tools, user_context prompts"),
    # 5 Oct: LFM B-v7 at LoRA rank 64: level with r32 typed (B sets 79.1 vs 79.2%, house 84.6 vs 84.2%) but +2.0 heard on the
    # house set (77.4 vs 75.4%, every condition up; 3 m 71.3 vs 68.6); 6-bit, as r32's Mac deployment
    Job("homeai-lfm25-1.2b-lora-b-v7r64@falcon-mlx6", "mlx/homeai-lfm25-1.2b-lora-b-v7r64-mlx6", mlx=True, steps=("prefix", "e2e"),
        layout="user_context", sets=SETS_B, prefix_args=PREFIX_B,
        note=f"{MLX_LM} server, 6-bit g64, prompt cache on, design B tools, user_context prompts"),
    # 5 Oct: the 2B B-v2u at LoRA rank 32 (lapwing: 79.6% B-set mean vs 78.2%, 80.9% typed on the house set vs 79.4%); the
    # Mac's 2 s pick if it holds heard and serves as fast as r16 (1.77 s warm on the house at 6-bit)
    Job("homeai-2b-lora-b-v2u-r32@falcon-mlx6", "mlx/homeai-2b-lora-b-v2u-r32-mlx6", mlx=True, steps=("prefix", "e2e"), layout="user_context",
        sets=SETS_B, prefix_args=PREFIX_B, note=f"{MLX_LM} server, 6-bit g64, prompt cache on, design B tools, user_context prompts"),
    # 5 Oct: the same at 6-bit (8-bit: 79.1% B-set mean, 0.81 s warm on the house): is a smaller quantisation faster on the
    # Mac at little cost (the 2B and 4B lost ~4 points at 4-bit)?
    Job("homeai-lfm25-1.2b-lora-b-v7r32@falcon-mlx6", "mlx/homeai-lfm25-1.2b-lora-b-v7r32-mlx6", mlx=True, steps=("prefix", "e2e"),
        layout="user_context", sets=SETS_B, prefix_args=PREFIX_B,
        note=f"{MLX_LM} server, 6-bit g64, prompt cache on, design B tools, user_context prompts"),
    # 5 Oct: LFM B-v7 at LoRA rank 32: on lapwing 79.2% B-set mean (r16 76.9), 84.2% typed on the house set (82.9), heard
    # mean 75.4% (74.8); merged, it should serve exactly as fast as B-v7
    Job("homeai-lfm25-1.2b-lora-b-v7r32@falcon-mlx8", "mlx/homeai-lfm25-1.2b-lora-b-v7r32-mlx8", mlx=True, steps=("prefix", "e2e"),
        layout="user_context", sets=SETS_B, prefix_args=PREFIX_B,
        note=f"{MLX_LM} server, 8-bit g64, prompt cache on, design B tools, user_context prompts"),
    # 5 Oct: the 2B B-v6n, the 2B B-v2u's recipe with the hard (far-field / noise) speech-to-text augmentation (lapwing: 77.5%
    # mean on the B sets vs 78.2%); the Mac's 2 s option if it holds up heard on the house set
    Job("homeai-2b-lora-b-v6n@falcon-mlx6", "mlx/homeai-2b-lora-b-v6n-mlx6", mlx=True, steps=("prefix", "e2e"), layout="user_context",
        sets=SETS_B, prefix_args=PREFIX_B, note=f"{MLX_LM} server, 6-bit g64, prompt cache on, design B tools, user_context prompts"),
    # 4 Oct: LFM B-v7, LFM B-v3h plus the hard (far-field / noise) speech-to-text augmentation: on the 200-conversation house
    # set heard (lapwing) 68.6% at 3 m and 63.6% in noise vs B-v3h's 64.3 / 61.0, typed 82.9 vs 82.0
    Job("homeai-lfm25-1.2b-lora-b-v7@falcon-mlx8", "mlx/homeai-lfm25-1.2b-lora-b-v7-mlx8", mlx=True, steps=("prefix", "e2e"),
        layout="user_context", sets=SETS_B, prefix_args=PREFIX_B,
        note=f"{MLX_LM} server, 8-bit g64, prompt cache on, design B tools, user_context prompts"),
    # 3 Oct: LFM B-v3h, the house data on the Mac's fastest design-B model (lapwing: 78.0 vs 76.3% mean, large houses +4.7)
    Job("homeai-lfm25-1.2b-lora-b-v3h@falcon-mlx8", "mlx/homeai-lfm25-1.2b-lora-b-v3h-mlx8", mlx=True, steps=("prefix", "e2e"),
        layout="user_context", sets=SETS_B, prefix_args=PREFIX_B,
        note=f"{MLX_LM} server, 8-bit g64, prompt cache on, design B tools, user_context prompts"),
    # 3 Oct: the 2B B-v3h (B-v2's data + 1,650 Bundook-plan conversations): +1.6 points over the 2B B-v2u on lapwing
    # (79.8 vs 78.2%, wild-b +4), the house data helping the 2B where it did not help the 4B; the Mac's 2 s option
    Job("homeai-2b-lora-b-v3h@falcon-mlx6", "mlx/homeai-2b-lora-b-v3h-mlx6", mlx=True, steps=("prefix", "e2e"), layout="user_context",
        sets=SETS_B, prefix_args=PREFIX_B, note=f"{MLX_LM} server, 6-bit g64, prompt cache on, design B tools, user_context prompts"),
    # 3 Oct: the 4B B-v2u, the Mac's more accurate design-B option (the 2B B-v2u: 80.5% at 1.77 s warm on the Bundook plan)
    Job("homeai-4b-lora-b-v2u@falcon-mlx6", "mlx/homeai-4b-lora-b-v2u-mlx6", mlx=True, steps=("prefix", "e2e"), layout="user_context",
        sets=SETS_B, prefix_args=PREFIX_B, note=f"{MLX_LM} server, 6-bit g64, prompt cache on, design B tools, user_context prompts"),
    # 2 Oct: the fast half of a Mac design-E pair with the 2B B-v2u (both user_context, so both cache the house prompt);
    # LFM 1.2B at 8-bit, as a small model has the least room for quantisation loss
    Job("homeai-lfm25-1.2b-lora-b-v2u@falcon-mlx8", "mlx/homeai-lfm25-1.2b-lora-b-v2u-mlx8", mlx=True, steps=("prefix", "e2e"),
        layout="user_context", sets=SETS_B, prefix_args=PREFIX_B,
        note=f"{MLX_LM} server, 8-bit g64, prompt cache on, design B tools, user_context prompts"),
    # 2 Oct: the Mac's design-B candidate: 2B B-v2 trained in the user_context layout, so the system message is identical
    # per house and the MLX cache can hold it (B-v2's stable_prefix prompt was re-read in full: warm 4.4 s on the Bundook plan)
    Job("homeai-2b-lora-b-v2u@falcon-mlx6", "mlx/homeai-2b-lora-b-v2u-mlx6", mlx=True, steps=("prefix", "e2e"), layout="user_context",
        sets=SETS_B, prefix_args=PREFIX_B, note=f"{MLX_LM} server, 6-bit g64, prompt cache on, design B tools, user_context prompts"),
    # 2 Oct: design A for the whole new house on the Mac, warm: v11 (user_context) keeps the system message identical per
    # house whatever the speaker, so the MLX cache can hold the ~20k-token Bundook prompt; 32k window as design A needs there
    Job("homeai-4b-lora-v11@falcon-mlx4-bundook", f"mlx/{V11}-mlx4", mlx=True, steps=PREFIX, layout="user_context", num_ctx=32768,
        prefix_args=("--data", "data/generated/bigbench-bundook.jsonl", "--group-by", "house", "--groups", "3", "--per-group", "8"),
        note=f"{MLX_LM} server, 4-bit g64, prompt cache on, user_context prompts, Bundook plan by house, num_ctx 32768"),
    # 2 Oct: the 2B design-B model (cassowary: 80.2% mean on the B sets at 1.32 s/request vs the 4B B-v2's 81.5% at 2.09 s)
    # as the Mac's big-house option; 6-bit, since 4-bit cost the 4B B-v2 ~4 points and 6-bit won them back
    Job("homeai-2b-lora-b-v2@falcon-mlx6", "mlx/homeai-2b-lora-b-v2-mlx6", mlx=True, steps=("e2e", "prefix"), layout=SP, sets=SETS_B, prefix_args=PREFIX_B,
        note=f"{MLX_LM} server, 6-bit g64, prompt cache on, design B tools, stable_prefix prompts"),
    # 2 Oct: v12 at 4-bit ties v9 on falcon (91.7 / 82.3 / 86.6 vs 91.4 / 83.6 / 87.5) and answers a warm request in 2.32 s
    # (v9 6.13 s); 6-bit won back B-v2's 4-bit loss, so it is the candidate default for the Mac
    Job("homeai-4b-lora-v12@falcon-mlx6", "mlx/homeai-4b-lora-v12-mlx6", mlx=True, steps=FULL, layout="clock_in_user",
        note=f"{MLX_LM} server, 6-bit g64, prompt cache on, clock_in_user prompts"),
    # 2 Oct: v12 moves only the clock into the user turn ("[now] ..."), so the system prompt is identical per room and
    # presence state: the key low-power test is v9's clarification at v11's warm speed (v11 2.6 s, v9 6.1 s, both mlx4)
    Job("homeai-4b-lora-v12@falcon-mlx4", "mlx/homeai-4b-lora-v12-mlx4", mlx=True, steps=FULL, layout="clock_in_user",
        note=f"{MLX_LM} server, 4-bit g64, prompt cache on, clock_in_user prompts"),
    # 2 Oct, overnight: design B is the new house's design (11,600 entities); its champion (B-v2, stable_prefix) on MLX 4-bit
    # is the low-power option for it. Falcon timings compare only with other falcon runs.
    Job("homeai-4b-lora-b-v2@falcon-mlx4", "mlx/homeai-4b-lora-b-v2-mlx4", mlx=True, steps=("e2e",), layout=SP, sets=SETS_B,
        note=f"{MLX_LM} server, 4-bit g64, prompt cache on, design B tools, stable_prefix prompts"),
    # 2 Oct: 4-bit g64 cost B-v2 ~4 points against cassowary's Q4_K_M (81.2 / 86.8 / 79.7 -> 77.2 / 82.2 / 75.1 on
    # bigbench-b / testv2-b / wild-b: garbled confirmations, dropped safety notes); 6-bit tests whether precision is the cause
    Job("homeai-4b-lora-b-v2@falcon-mlx6", "mlx/homeai-4b-lora-b-v2-mlx6", mlx=True, steps=("e2e",), layout=SP, sets=SETS_B, 
        note=f"{MLX_LM} server, 6-bit g64, prompt cache on, design B tools, stable_prefix prompts"),
    # 1 Oct, overnight: v9 lost ~3 points on 0.35.0 (testv2 87.4 / wild2 84.0 vs cassowary 0.21.2's 90.2 / 87.8) while
    # v5 held (92.0 / 86.5 vs 93.1 / 86.5) and v9 on MLX holds (91.4 / 87.5). The same GGUF on falcon's 0.21.2 separates
    # the Ollama build from the Mac's Metal kernels.
    base("homeai-4b-lora-v9", "qwen35-4b-lora-v9", steps=("e2e",)),
    # 1 Oct: the champion (v9, classic layout) on MLX 4-bit, for the low-power comparison with v11 / v12's cacheable layouts
    Job("homeai-4b-lora-v9@falcon-mlx4", "mlx/homeai-4b-lora-v9-mlx4", mlx=True, steps=FULL,
        note=f"{MLX_LM} server, 4-bit g64, prompt cache on, classic prompts"),
    # 1 Oct: on frogmouth Ollama 0.35.0 evaluates a v5 prompt ~2x faster than 0.21.2 (1.42 s vs 2.88 s for a new house,
    # 1.02 s vs 1.31 s for the same house with a new request). Same archived Q4_K_M and cassowary Modelfile lines, served
    # by 0.35.0 on the same port; v5 must match its 0.21.2 accuracy (same weights), v9 is the current champion.
    Job(f"{V5}@falcon-ollama035", V5, run="qwen35-4b-lora-v5", ollama=O35, steps=FULL, note=O35_NOTE),
    Job("homeai-lfm25-1.2b-lora-v9@falcon-ollama035", "homeai-lfm25-1.2b-lora-v9", run="lfm25-1.2b-lora-v9", ollama=O35,
        steps=FULL, note=O35_NOTE),
    Job("homeai-4b-lora-v9@falcon-ollama035", "homeai-4b-lora-v9", run="qwen35-4b-lora-v9", ollama=O35, steps=FULL, note=O35_NOTE),
    # 1 Oct: v11 puts "[context] Now | Speaker | People" at the start of each user turn (user_context), leaving the system
    # message identical across requests -- the point mlx_lm's hybrid-model prompt cache can resume from (v10's in-system
    # block could not be reused). The MLX job is the measurement; the LFM 1.2B v9 is the new design-E fast model.
    Job(f"{V11}@falcon-mlx4", f"mlx/{V11}-mlx4", mlx=True, steps=FULL, layout="user_context",
        note=f"{MLX_LM} server, 4-bit g64, prompt cache on, user_context prompts"),
    base("homeai-lfm25-1.2b-lora-v9", "lfm25-1.2b-lora-v9"),
    # 30 Sep, ahead of the queue: v10 is v9's recipe with the stable_prefix layout (Now / Speaker / People in a final
    # "Current context:" block), so the whole static prompt is a cacheable prefix. Its row names do not resolve through
    # configs/models.json (falcon has no copy; an MLX dir never would), so the jobs force the layout on every eval client.
    # The MLX job waits for scripts/falcon-mlx-build.sh qwen35-4b-lora-v10 homeai-4b-lora-v10 mlx4 (pushed while the
    # stream is stopped, scripts/falcon-start-when-ready.sh, so it goes first); the baseline needs only the archived GGUF.
    Job(f"{V10}@falcon-mlx4", f"mlx/{V10}-mlx4", mlx=True, steps=FULL, layout=SP,
        note=f"{MLX_LM} server, 4-bit g64, prompt cache on, stable_prefix prompts"),
    base(V10, "qwen35-4b-lora-v10", steps=FULL, layout=SP,
         note="Ollama 0.21.2, cassowary settings (KEEP_ALIVE=-1, NUM_PARALLEL=2), Q4_K_M, num_ctx 8192, stable_prefix prompts"),
    # phase 1, first four (Peter's priority order)
    base(V5, "qwen35-4b-lora-v5"),
    base("homeai-4b-lora-v8", "qwen35-4b-lora-v8"),
    base("homeai-lfm25-1.2b-lora-v5-fix2", "lfm25-1.2b-lora-v5-fix2"),
    base("homeai-2b-lora-v5", "qwen35-2b-lora-v5"),
    # phase 2: warm prefix cache on the stock setup
    base(V5, "qwen35-4b-lora-v5", steps=PREFIX),
    base("homeai-2b-lora-v5", "qwen35-2b-lora-v5", steps=PREFIX),
    base("homeai-lfm25-1.2b-lora-v5-fix2", "lfm25-1.2b-lora-v5-fix2", steps=PREFIX),
    base("homeai-4b-lora-v8", "qwen35-4b-lora-v8", steps=PREFIX),
    # phase 2: mlx-lm's OpenAI-compatible server, 4-bit, group size 64, from the same fused weights
    Job(f"{V5}@falcon-mlx4", f"mlx/{V5}-mlx4", mlx=True, steps=FULL, note=f"{MLX_LM} server, 4-bit g64, prompt cache on"),
    # frogmouth check on the 0.8B v5 (40 test-v2 conversations, 94 turns): Ollama Q4_K_M 84.0%, MLX bf16 84.0%, 8-bit 84.0%,
    # 4-bit g32 81.9%, 4-bit g64 72.3%, mixed 4/6-bit g64 73.4%: the group size is what costs accuracy, so g32 too
    Job(f"{V5}@falcon-mlx4g32", f"mlx/{V5}-mlx4g32", mlx=True, steps=FULL, note=f"{MLX_LM} server, 4-bit g32, prompt cache on"),
    # phase 2: Ollama settings on v5, timing only (same weights; a q8_0 KV cache barely changes the numerics and only the
    # 8 full-attention layers of the 32 keep a KV cache in this hybrid model).
    # Flash attention is already on in the baseline: 0.21.2 enables it by default on Metal ("enabling flash attention",
    # FlashAttention:Enabled in the load request), so OLLAMA_FLASH_ATTENTION=1 alone is the baseline and is not rerun.
    Job(f"{V5}@falcon-kvq8", V5, run="qwen35-4b-lora-v5", env={"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"},
        steps=TIMING, note="Ollama 0.21.2 stock (flash attention on by default) + q8_0 KV cache"),
    Job(f"{V5}@falcon-nb1024", f"{V5}-nb1024", like=V5, run="qwen35-4b-lora-v5", params={"num_batch": 1024}, steps=TIMING,
        note="Ollama 0.21.2 stock, PARAMETER num_batch 1024"),
    Job(f"{V5}@falcon-nb2048", f"{V5}-nb2048", like=V5, run="qwen35-4b-lora-v5", params={"num_batch": 2048}, steps=TIMING,
        note="Ollama 0.21.2 stock, PARAMETER num_batch 2048"),
    Job(f"{V5}@falcon-ctx32k", V5, run="qwen35-4b-lora-v5", num_ctx=32768, steps=TIMING,
        note="Ollama 0.21.2 stock, num_ctx 32768 (vs 8192)"),
    # phase 2: other quantisations of v5, built on frogmouth from the fused 16-bit weights (scripts/falcon-phase2-build.sh:
    # Ollama cannot requantise a Q4 and 0.21.2 only makes F16 / Q4_K_S / Q4_K_M / Q8_0, so Q4_0 / IQ4_XS / Q5_K_M come from
    # llama.cpp's llama-quantize on Ollama's F16). The fuse is not bit-identical to Unsloth's, hence the Q4_K_M control.
    Job(f"{V5}@falcon-q4_k_m-rebuilt", f"{V5}-q4km-rebuilt", like=V5, gguf=f"{V5}-q4_K_M-rebuilt.gguf", steps=FULL,
        note="Ollama 0.21.2 stock, Q4_K_M rebuilt on frogmouth from the fused weights (control for the other quants)"),
    Job(f"{V5}@falcon-q4_0", f"{V5}-q4_0", like=V5, gguf=f"{V5}-q4_0.gguf", steps=FULL, note="Ollama 0.21.2 stock, Q4_0 (llama.cpp b11243)"),
    Job(f"{V5}@falcon-iq4_xs", f"{V5}-iq4_xs", like=V5, gguf=f"{V5}-iq4_xs.gguf", steps=FULL, note="Ollama 0.21.2 stock, IQ4_XS (llama.cpp b11243, no imatrix)"),
    Job(f"{V5}@falcon-q5_k_m", f"{V5}-q5_k_m", like=V5, gguf=f"{V5}-q5_k_m.gguf", steps=FULL, note="Ollama 0.21.2 stock, Q5_K_M (llama.cpp b11243)"),
    # phase 1, the rest
    base("homeai-4b-lora-v7", "qwen35-4b-lora-v7"),
    base("homeai-gemma4-e4b-v8", "gemma4-e4b-v8"),
    base("homeai-9b-v2", "qwen35-9b-v2"),
    base("homeai-0.8b-lora-v5", "qwen35-0.8b-lora-v5"),
    base("homeai-lfm25-350m-lora-v5-fix", "lfm25-350m-lora-v5-fix"),
    # 13 Oct: LFM2.5 2.6B r128 (trained on cassowary overnight): last, because the stream waits in order for staged artifacts
    Job("homeai-lfm25-2.6b-lora-b-v8r128e4@falcon-mlx6", "mlx/homeai-lfm25-2.6b-lora-b-v8r128e4-mlx6", mlx=True, steps=("prefix", "e2e"),
        layout="user_context", sets=SETS_B, prefix_args=PREFIX_B,
        note=f"{MLX_LM} server, 6-bit g64, prompt cache on, design B tools, user_context prompts"),
]


# ---- plumbing ----

def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def sh(cmd: str, check=True, **kw) -> str:
    r = subprocess.run(["/bin/bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True, **kw)
    if check and r.returncode:
        raise RuntimeError(f"{cmd[:160]} -> {r.returncode}: {r.stderr.strip()[-400:]}")
    return r.stdout


def ssh(host: str, cmd: str) -> str:
    return sh(f"ssh -o BatchMode=yes -o ConnectTimeout=15 {host} {json.dumps(cmd)} 2>/dev/null")


def state(**kw):
    json.dump({"time": time.strftime("%F %T"), "pid": os.getpid(), **kw}, open(f"{LOGS}/state.json", "w"))


def footprint_gb() -> float:
    return int(sh(f"du -sk {W}").split()[0]) * 1024 / 1e9  # du counts a hard-linked GGUF (stage/ + blobs/) once


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def hold():
    """frogmouth copies files in only between timed steps: it creates stage/HOLD, waits for stage/HELD, copies, removes HOLD.
    stage/RESTART makes the stream exit here (between steps), so a new version can be started without losing a step."""
    if os.path.exists(f"{STAGE}/RESTART"):
        os.remove(f"{STAGE}/RESTART")
        log("RESTART requested: exiting between steps")
        stop_servers()
        sys.exit(0)
    if not os.path.exists(f"{STAGE}/HOLD"):
        return
    log("HOLD: pausing for a transfer")
    open(f"{STAGE}/HELD", "w").write(str(os.getpid()))
    while os.path.exists(f"{STAGE}/HOLD"):
        time.sleep(15)
    os.remove(f"{STAGE}/HELD")
    log("HOLD released")


def ready(name: str) -> bool:
    return os.path.exists(f"{STAGE}/{name}") and os.path.exists(f"{STAGE}/{name}.ready")


def skipped(job: Job) -> bool:
    """frogmouth marks a job's artifact stage/<name>.skip when it is not needed (a rebuild identical to the baseline)"""
    return any(os.path.exists(f"{STAGE}/{n}.skip") for n in job.needs())


# ---- servers ----

def api(path: str, body: dict | None = None, timeout=10):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def ollama(*args, check=True) -> str:
    env = f"OLLAMA_HOST=127.0.0.1:{PORT} OLLAMA_MODELS={MODELS}"
    return sh(f"{env} {OLLAMA} " + " ".join(json.dumps(a) for a in args), check=check)


def use_ollama(version: str):
    global OLLAMA, MODELS
    OLLAMA, MODELS = f"{ollama_dir(version)}/ollama", f"{ollama_dir(version)}/models"


def ollama_setup(version: str):
    """W/ollama-<version>: the release's darwin tarball, unpacked (it counts against CAP_GB like everything under W)."""
    d = ollama_dir(version)
    if os.path.exists(f"{d}/ollama"):
        return
    if version == OLLAMA_STOCK:
        raise RuntimeError(f"{d}/ollama missing (the stock build is installed by hand)")
    ensure_space(0.8, set())  # ~0.5 GB unpacked (llama-server, MLX libraries) plus the tarball while unpacking
    tgz = f"{STAGE}/ollama-{version}-darwin.tgz.part"
    log(f"install Ollama {version} -> {d}")
    sh(f"curl -fsSL -o {tgz} {OLLAMA_URL.format(v=version)} && mkdir -p {d}.part && tar -C {d}.part -xzf {tgz}")
    os.remove(tgz)
    if not os.path.exists(f"{d}.part/ollama"):
        raise RuntimeError(f"Ollama {version} tarball has no top-level ollama binary")
    os.rename(f"{d}.part", d)
    os.makedirs(f"{d}/models", exist_ok=True)


def stop_servers():
    # every Ollama build under W (serve, runner, 0.35's llama-server children) and our MLX server
    pat = f"{W}/ollama[-.0-9]*/(ollama (serve|runner)|llama-server)|mlx_lm.server.*--port {MLX_PORT}"
    sh(f"pkill -f '{pat}'", check=False)
    for _ in range(30):
        if not sh(f"pgrep -f '{pat}'", check=False).strip():
            return
        time.sleep(1)


_serving: dict | None = None
_version = ""  # what the running Ollama reports (/api/version), recorded in every report as ollama_version


def warm_metal(penv: dict):
    """Compile the Metal library once before `ollama serve`, whose GPU discovery times out after 30 s: 0.21.2 with a bare
    runner, builds that ship llama-server (0.35.0) with llama-server --list-devices."""
    d = os.path.dirname(OLLAMA)
    if os.path.exists(f"{d}/llama-server"):
        sh(f"{d}/llama-server --list-devices > {LOGS}/ollama-warm.log 2>&1", check=False, env=penv, timeout=600)
        return
    warm = subprocess.Popen([OLLAMA, "runner", "--ollama-engine", "--port", str(WARM_PORT)], env=penv,
                            stdout=open(f"{LOGS}/ollama-warm.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            if "Metal" in urllib.request.urlopen(f"http://127.0.0.1:{WARM_PORT}/info", timeout=5).read().decode():
                break
        except OSError:
            pass
        if warm.poll() is not None:
            break
        time.sleep(3)
    warm.kill()


def serve_ollama(extra: dict, version: str = OLLAMA_STOCK):
    """Ollama <version> on 127.0.0.1:PORT with STOCK_ENV + extra, restarted when the build or settings differ. GPU
    discovery has a 30 s timeout and Ollama compiles its Metal library at start-up (scripts/mlx-env.sh): warm the shader
    cache first, then restart until the server reports Metal."""
    global _serving, _version
    env = {**STOCK_ENV, **extra}
    want = {"ollama": version, **env}
    if _serving == want:
        try:
            api("/api/version")
            return
        except OSError:
            pass
    stop_servers()
    ollama_setup(version)
    use_ollama(version)
    os.makedirs(MODELS, exist_ok=True)
    penv = {**os.environ, **env, "OLLAMA_HOST": f"127.0.0.1:{PORT}", "OLLAMA_MODELS": MODELS}
    warm_metal(penv)
    for attempt in range(6):
        logf = f"{LOGS}/ollama-serve.log"
        subprocess.Popen([OLLAMA, "serve"], env=penv, stdout=open(logf, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
        for _ in range(90):
            txt = open(logf, errors="replace").read()
            if "inference compute" in txt:
                break
            time.sleep(2)
        time.sleep(2)
        if "library=Metal" in open(logf, errors="replace").read():
            _serving = want
            _version = api("/api/version").get("version", version)
            log(f"ollama {_version} serve on Metal", " ".join(f"{k}={v}" for k, v in extra.items()) or "(stock)")
            return
        log(f"ollama serve fell back to CPU (try {attempt + 1}): restarting")
        stop_servers()
    raise RuntimeError("ollama serve is not on Metal after 6 tries")


def mlx_setup():
    """uv venv under W/mlx with mlx-lm, its Python, caches and HF home all inside W."""
    py = f"{MLX}/venv/bin/python"
    if os.path.exists(py) and not sh(f"{py} -c 'import mlx_lm' 2>&1", check=False).strip():
        return py
    uv = shutil.which("uv") or "/opt/homebrew/bin/uv"
    env = (f"UV_CACHE_DIR={MLX}/cache UV_PYTHON_INSTALL_DIR={MLX}/python UV_PYTHON_BIN_DIR={MLX}/bin UV_TOOL_DIR={MLX}/tools "
           f"UV_NO_MODIFY_PATH=1 UV_PYTHON_PREFERENCE=only-managed")
    sh(f"mkdir -p {MLX} && {env} {uv} venv --python 3.12 {MLX}/venv && {env} {uv} pip install --python {py} '{MLX_LM}'")
    sh(f"rm -rf {MLX}/cache")  # wheels are not needed after the install
    return py


def serve_mlx(model_dir: str):
    global _serving
    want = {"mlx": model_dir}
    if _serving == want:
        return
    stop_servers()
    _serving = None
    py = mlx_setup()
    env = {**os.environ, "HF_HOME": f"{MLX}/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    subprocess.Popen([py, "-m", "mlx_lm.server", "--model", model_dir, "--host", "127.0.0.1", "--port", str(MLX_PORT),
                      "--chat-template-args", json.dumps({"enable_thinking": False})], env=env,
                     stdout=open(f"{LOGS}/mlx-server.log", "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     start_new_session=True)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{MLX_PORT}/v1/models", timeout=5)
            _serving = want
            log("mlx_lm.server up:", model_dir)
            return
        except OSError:
            time.sleep(3)
    raise RuntimeError("mlx_lm.server did not come up")


# ---- models ----

def cassowary_modelfile(model: str) -> tuple[str, list[str]]:
    """(blob digest, TEMPLATE/RENDERER/PARSER/PARAMETER lines) of what cassowary's Ollama serves, cached in W/modelfiles."""
    p = f"{W}/modelfiles/{model}.Modelfile"
    if not os.path.exists(p):
        txt = ssh(CASSOWARY, f"docker exec ollama ollama show --modelfile {model}")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(txt)
    lines = open(p).read().splitlines()
    frm = next(l for l in lines if l.startswith("FROM "))
    keep = [l for l in lines if l.split(" ", 1)[0] in ("TEMPLATE", "RENDERER", "PARSER", "PARAMETER", "SYSTEM")]
    if any('"""' in l for l in keep):
        raise RuntimeError(f"{model}: multi-line Modelfile directive, not handled")
    return frm.rsplit("sha256-", 1)[-1].strip(), keep


def stores() -> list[str]:
    """Every Ollama model store under W (one per build)."""
    return sorted({f"{ollama_dir(OLLAMA_STOCK)}/models", MODELS} | {f"{W}/{d}/models" for d in os.listdir(W)
                  if d.startswith("ollama-") and os.path.isdir(f"{W}/{d}/models")})


def _manifests(store: str) -> dict:
    """{model name: manifest path} of a store, read from disk (works whether or not its server runs)."""
    root = f"{store}/manifests/registry.ollama.ai/library"
    out = {}
    for name in (os.listdir(root) if os.path.isdir(root) else []):
        for tag in os.listdir(f"{root}/{name}"):
            out[name if tag == "latest" else f"{name}:{tag}"] = f"{root}/{name}/{tag}"
    return out


def _digests(manifest: str) -> set[str]:
    m = json.load(open(manifest))
    return {l["digest"].replace(":", "-") for l in [m.get("config") or {}, *m.get("layers", [])] if l.get("digest")}


def store_rm(store: str, name: str):
    """Remove a model from a store whose server is not running: its manifest, then the blobs no other manifest of that
    store references (what `ollama rm` does; a blob hard-linked into another store lives on there)."""
    mans = _manifests(store)
    if name not in mans:
        return
    path = mans.pop(name)
    mine = _digests(path)
    os.remove(path)
    still = set().union(*(_digests(p) for p in mans.values())) if mans else set()
    for d in mine - still:
        try:
            os.remove(f"{store}/blobs/{d}")
        except OSError:
            pass


def evict(store: str, name: str):
    if store == MODELS and _serving and "ollama" in _serving:
        ollama("rm", name, check=False)
    else:
        store_rm(store, name)


def installed() -> set[str]:
    return {m["name"].split(":")[0] for m in api("/api/tags").get("models", [])}


def installed_safe() -> set[str]:
    try:
        return installed()
    except OSError:  # Ollama not running (MLX jobs): ask the CLI's store directly
        return {l.split()[0].split(":")[0] for l in ollama("list", check=False).splitlines()[1:] if l.strip()}


def ensure_space(need_gb: float, keep: set[str], current: str = ""):
    """Evict Ollama models until need_gb more fits under CAP_GB: others first, then the champions (they come back from
    the archive when the stream ends), never the model being imported."""
    for f in os.listdir(STAGE):
        if f.endswith(".part"):
            os.remove(os.path.join(STAGE, f))
    while footprint_gb() + need_gb > CAP_GB:
        # every store counts (a blob hard-linked into two stores is freed only once both drop it): other stores first
        have = sorted(((st != MODELS, st, m) for st in stores() for m in _manifests(st) if (st, m) != (MODELS, current)), reverse=True)
        victims = [(st, m) for _, st, m in have if m not in keep] + [(st, m) for _, st, m in have if m in keep]
        if not victims:
            raise RuntimeError(f"disk cap: {footprint_gb():.1f} GB used, {need_gb:.1f} GB more needed, nothing left to evict")
        log(f"disk cap: evicting {victims[0][1]} from {os.path.relpath(victims[0][0], W)} ({footprint_gb():.1f} GB used)")
        evict(*victims[0])


def fetch_archived(job: Job, digest: str, keep: set[str]) -> str:
    """The archived Q4_K_M, straight into the blob store (one copy on disk), verified against cassowary's digest."""
    blob = f"{MODELS}/blobs/sha256-{digest}"
    if os.path.exists(blob):
        return blob
    for st in stores():  # another build's store has it: hard-link (one copy on disk), read-only so neither build can
        other = f"{st}/blobs/sha256-{digest}"  # rewrite what the other serves (Ollama only ever creates or unlinks blobs)
        if st != MODELS and os.path.exists(other):
            os.makedirs(os.path.dirname(blob), exist_ok=True)
            os.chmod(other, 0o444)
            os.link(other, blob)
            log(f"blob sha256-{digest[:12]} hard-linked from {os.path.relpath(st, W)}")
            return blob
    name = f"{job.like or job.model}-q4_K_M.gguf"
    for host, root in ((PIPER, PIPER_DIR), (CASSOWARY, CASSOWARY_DIR)):
        try:
            size = int(ssh(host, f"stat -c %s {root}/{job.run}/gguf/{name}").strip())
        except (RuntimeError, ValueError):
            continue
        ensure_space(size / 1e9, keep, job.model)
        part = f"{STAGE}/{name}.part"
        log(f"fetch {name} ({size / 1e9:.2f} GB) from {host}")
        sh(f"ssh -o BatchMode=yes {host} 'tar -C {root}/{job.run} -cf - gguf/{name}' | tar -xOf - > {part}")
        got = sha256(part)
        if got == digest:
            os.makedirs(os.path.dirname(blob), exist_ok=True)
            os.replace(part, blob)
            return blob
        log(f"{host} copy of {name} is not the served blob ({got[:12]} vs {digest[:12]}; still uploading?)")
        os.remove(part)
    raise RuntimeError(f"{name}: no copy matching sha256 {digest[:12]} on Piper or cassowary")


def import_model(job: Job, keep: set[str]):
    """Create job.model in falcon's Ollama with the Modelfile lines cassowary serves (plus job.params)."""
    if job.model in installed():
        return
    digest, lines = cassowary_modelfile(job.like or job.model)
    if job.gguf:  # frogmouth-built quantisation: its own weights, the reference model's template / renderer / parameters
        src = f"{STAGE}/{job.gguf}"
        digest = sha256(src)
        blob = f"{MODELS}/blobs/sha256-{digest}"
        if not os.path.exists(blob):
            os.makedirs(os.path.dirname(blob), exist_ok=True)
            os.link(src, blob)  # same file twice, one copy on disk; stage/ is cleared once the job is done
    else:
        blob = fetch_archived(job, digest, keep)
    lines = [l for l in lines if not any(l.startswith(f"PARAMETER {k} ") for k in job.params)]
    lines += [f"PARAMETER {k} {v}" for k, v in job.params.items()]
    mf = f"{W}/modelfiles/{job.model}.falcon.Modelfile"
    open(mf, "w").write(f"FROM {blob}\n" + "\n".join(lines) + "\n")
    log(f"ollama create {job.model}")
    ollama("create", job.model, "-f", mf)
    served = ollama("show", "--modelfile", job.model)
    if f"sha256-{digest}" not in served:
        raise RuntimeError(f"{job.model}: imported blob is not sha256-{digest[:12]}")


# ---- steps ----

def out_dir(job: Job, step: str, s: str = "") -> str:
    return f"{EVAL}/{job.row}-{s}-e2e" if step == "e2e" else f"{EVAL}/{job.row}-{step}"


def done(job: Job, step: str) -> bool:
    if step == "e2e":
        return all(os.path.exists(f"{out_dir(job, step, s)}/report.json") for s in (job.sets or SETS))
    return os.path.exists(f"{out_dir(job, step)}/{step}.json")


def client(job: Job) -> tuple[str, str, dict]:
    """(--ollama URL, --model, extra env) for the eval clients."""
    env = {"HOMEAI_HOST": "falcon", "HOMEAI_HARDWARE": "apple-m2pro", "HOMEAI_HOST_NOTE": f"falcon stream: {job.note}"}
    if not job.mlx:  # homeai.host records it as ollama_version in every report
        env["HOMEAI_OLLAMA_VERSION"] = _version or job.ollama
    # the layout travels in the environment too, so every client (and anything it spawns) renders the same prompts
    env["HOMEAI_PROMPT_LAYOUT"] = job.layout or "classic"
    if job.mlx:  # the bridge's 512-token cap and no thinking, as Ollama gets them
        env["HOMEAI_OPENAI_EXTRA"] = json.dumps({"max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False}})
        return f"http://127.0.0.1:{MLX_PORT}/v1", f"{STAGE}/{job.model}", env
    return URL, job.model, env


def run_step(job: Job, step: str):
    url, model, env = client(job)
    env = {**os.environ, **env, "PYTHONPATH": "."}
    ctx = ["--num-ctx", str(job.num_ctx), "--prompt-layout", job.layout or "classic"]
    if step == "e2e":
        cmds = [(out_dir(job, step, s), ["-m", "homeai.eval.e2e", "--model", model, "--data", d, "--ollama", url, *ctx])
                for s, d in (job.sets or SETS).items()]
        cmds = [(o, c + ["--out", o]) for o, c in cmds if not os.path.exists(f"{o}/report.json")]
    elif step == "latency":
        o = out_dir(job, step)
        cmds = [(o, ["-m", "homeai.eval.latency", "--model", model, "--data", SETS["testv2"], "--n", "60", "--ollama", url, *ctx, "--out", o])]
    else:
        o = out_dir(job, step)
        cmds = [(o, ["-m", "homeai.eval.prefix_bench", "--model", model, "--ollama", url, *ctx, *job.prefix_args, "--out", o])]
    for o, c in cmds:
        hold()
        # falcon restarted on 29 Sep with a load average of ~430 (post-boot indexing): timings wait for a quiet machine
        waited = 0
        while os.getloadavg()[0] > QUIET_LOAD and waited < 45 * 60:
            if waited == 0:
                log(f"{job.row}: waiting for load {os.getloadavg()[0]:.1f} to drop below {QUIET_LOAD} before {step}")
            time.sleep(60); waited += 60
        state(job=job.row, step=step, out=o)
        busy = sh("pgrep -fl 'homeai[.](eval|train)|mlx_lm[.]convert' | grep -v falcon_stream", check=False).strip()
        log(f"{job.row}: {step} -> {o} (load {os.getloadavg()[0]:.2f}{'; BUSY: ' + busy if busy else ''})")
        t0 = time.time()
        with open(f"{LOGS}/{os.path.basename(o)}.log", "w") as f:
            r = subprocess.run([sys.executable, *c], env=env, stdout=f, stderr=subprocess.STDOUT)
        log(f"{job.row}: {step} exit {r.returncode} after {time.time() - t0:.0f}s")
        if r.returncode:
            raise RuntimeError(f"{step} failed, see {LOGS}/{os.path.basename(o)}.log")


def champions() -> set[str]:
    """Recompute outputs/FALCON_CHAMPIONS.* and the E-pair grid from falcon's own logs; the Ollama models to keep."""
    sh(f"PYTHONPATH=. {sys.executable} -m homeai.eval.reconcile_grid --eval-dir {EVAL} --out outputs/reconcile-falcon >/dev/null", check=False)
    sh(f"PYTHONPATH=. {sys.executable} -m homeai.eval.host_champions --host falcon --eval-dir {EVAL} >/dev/null", check=False)
    try:
        ch = json.load(open("outputs/FALCON_CHAMPIONS.json"))
    except (OSError, ValueError):
        return set()
    rows = {ch.get("accuracy"), ch.get("speed")} - {None}
    return {j.model for j in JOBS if j.row in rows and not j.mlx}


def unload_all():
    for m in api("/api/ps").get("models", []):
        api("/api/generate", {"model": m["name"], "keep_alive": 0}, timeout=60)


def run_job(job: Job, keep: set[str]):
    if job.mlx:
        serve_mlx(f"{STAGE}/{job.model}")
    else:
        serve_ollama(job.env, job.ollama)
        import_model(job, keep | {job.model})
        unload_all()  # one model loaded at a time (KEEP_ALIVE=-1 keeps the last one resident)
    for step in job.steps:
        if not done(job, step):
            run_step(job, step)
    if job.mlx:
        stop_servers()
    else:
        unload_all()
    for n in job.needs():  # pushed artifacts are cleared once measured (a GGUF lives on as the model's blob)
        sh(f"rm -rf {STAGE}/{n} {STAGE}/{n}.ready", check=False)


def status():
    print(f"footprint {footprint_gb():.2f} GB (cap {CAP_GB} GB)")
    for j in JOBS:
        waits = [n for n in j.needs() if not ready(n)]
        print(f"{j.row:44} " + " ".join(f"{s}:{'done' if done(j, s) else '-'}" for s in j.steps) + (f"  [{j.layout}]" if j.layout else "")
              + (f"  [ollama {j.ollama}]" if j.ollama != OLLAMA_STOCK else "")
              + (f"  waits for {waits}" if waits else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", default="", help="comma list of rows to run (default: all JOBS in order)")
    ap.add_argument("--wait", action="store_true", help="wait for a running stream to exit (stage/RESTART), then run")
    a = ap.parse_args()
    for d in (MODELS, STAGE, LOGS):
        os.makedirs(d, exist_ok=True)
    if a.status:
        return status()
    lock = open(f"{LOGS}/stream.pid", "a+")
    try:  # one stream at a time: the lock lives as long as this process
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if a.wait else fcntl.LOCK_NB))
    except OSError:
        return print("falcon_stream already running")
    lock.truncate(0); lock.write(str(os.getpid())); lock.flush()
    jobs = [j for j in JOBS if not a.only or j.row in a.only.split(",")]
    failed: set[int] = set()
    while True:
        todo = [(i, j) for i, j in enumerate(jobs) if i not in failed and not skipped(j) and not all(done(j, s) for s in j.steps)]
        runnable = [(i, j) for i, j in todo if all(ready(n) for n in j.needs()) or j.model in installed_safe()]
        if not todo:
            break
        if not runnable:
            state(waiting=[j.row for _, j in todo])
            log("waiting for artifacts:", ", ".join(n for _, j in todo for n in j.needs()))
            for _ in range(30):  # a restart request is honoured while waiting too (2 Oct: a waiting stream ignored it)
                if os.path.exists(f"{STAGE}/RESTART"):
                    os.remove(f"{STAGE}/RESTART")
                    log("RESTART requested: exiting while waiting for artifacts")
                    stop_servers()
                    return
                time.sleep(10)
            continue
        i, job = runnable[0]
        hold()
        keep = champions()
        nxt = next((j for _, j in runnable[1:]), None)
        try:
            run_job(job, keep)
        except Exception as e:  # noqa: BLE001
            log(f"{job.row}: FAILED {e}")
            failed.add(i)
        keep = champions() | ({nxt.model, nxt.like} - {""} if nxt and not nxt.mlx else set())
        for st in stores():  # every build's store (the next job may hard-link a kept blob into its own)
            for m in sorted(set(_manifests(st)) - keep):
                evict(st, m)
        log(f"{job.row}: done; keeping {sorted(keep)}; footprint {footprint_gb():.2f} GB")
    stop_servers()
    # the champions stay installed on falcon (evicted ones come back from the archive)
    serve_ollama({})
    for j in JOBS:
        if j.model in champions() and not j.gguf and j.run and j.row == j.model:
            try:
                import_model(j, champions())
            except Exception as e:  # noqa: BLE001
                log(f"restore {j.model}: {e}")
    stop_servers()
    state(finished=True, failed=[jobs[i].row for i in failed])
    log("stream finished" + (f"; failed: {[jobs[i].row for i in failed]}" if failed else ""))


if __name__ == "__main__":
    main()
