#!/usr/bin/env bash
# Sequential overnight pipeline on cassowary. Idempotent: each step is skipped if its output exists.
#   nohup setsid scripts/overnight.sh > outputs/overnight.log 2>&1 &
# Status is written to outputs/overnight.status after every step.
set -uo pipefail
cd "$(dirname "$0")/.."
STATUS=outputs/overnight.status
mkdir -p outputs
note() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

wait_for_train() {  # wait until no homeai-train container is running
  while docker ps -q -f name=homeai-train | grep -q .; do sleep 60; done
}

train_run() {  # cfg
  local cfg=$1 out; out=$(python3 -c "import json;print(json.load(open('$cfg'))['out'])")
  if [ -d "$out/lora" ]; then note "train $cfg: already done"; return 0; fi
  wait_for_train
  note "train $cfg: starting"
  docker rm -f homeai-train >/dev/null 2>&1
  DETACH=-d scripts/train-docker.sh train "$cfg" >/dev/null
  sleep 600; fill_quality; wait_for_train   # accuracy evals of earlier models share the GPU while it trains
  own
  if [ -d "$out/lora" ]; then note "train $cfg: done"; else note "train $cfg: FAILED (no lora dir) - see outputs/$(basename "$cfg" .json)-train.log"; return 1; fi
}

own() { docker run --rm -v "$PWD/outputs":/o alpine chown -R "$(id -u):$(id -g)" /o >/dev/null 2>&1; }

model_url() { python3 -c "import json;print(json.load(open('configs/models.json')).get('$1',{}).get('ollama','http://localhost:11434'))" 2>/dev/null || echo http://localhost:11434; }
model_ctr() { python3 -c "import json;print(json.load(open('configs/models.json')).get('$1',{}).get('ollama_container','ollama'))" 2>/dev/null || echo ollama; }

has_model() {  # name -- Ollama's list is briefly empty right after models are unloaded; retry before deciding
  for _ in 1 2 3 4 5 6; do
    local out; out=$(docker exec "$(model_ctr "$1")" ollama list 2>/dev/null)   # the model's own container (Spark: ollama-next)
    if echo "$out" | grep -q "^$1[: ]"; then return 0; fi
    if [ "$(echo "$out" | grep -c .)" -gt 1 ]; then return 1; fi   # list populated, model genuinely absent
    sleep 5
  done
  return 1
}

export_run() {  # cfg name base  -- merge LoRA to 16-bit, let Ollama convert+quantise, keep a portable GGUF
  local cfg=$1 name=$2 base=$3 out; out=$(python3 -c "import json;print(json.load(open('$cfg'))['out'])")
  if has_model "$name"; then note "import $name: exists"; return 0; fi   # merged16 may have been pruned; the model is what counts
  if [ -f "$out/merged16/config.json" ]; then note "merge $cfg: exists"; else
    wait_for_train   # never share the GPU with a training run
    note "merge $cfg: starting"
    docker rm -f homeai-train >/dev/null 2>&1
    NAME=homeai-merge DETACH="" scripts/train-docker.sh merge "$cfg" </dev/null >> outputs/overnight.log 2>&1
    own
    [ -f "$out/merged16/config.json" ] || { note "merge $cfg: FAILED - see outputs/overnight.log"; return 1; }
    note "merge $cfg: done"
  fi
  own
  case $base in *9b*|gemma4*) [ -f "$out/.convert-elsewhere" ] || touch "$out/.convert-elsewhere";; esac   # 9B / Gemma 4 (8B stored) convert OOMs Ollama here
  if [ -f "$out/.convert-elsewhere" ] && ! has_model "$name"; then
    note "import $name: skipped (needs conversion on the Mac: scripts/convert-on-mac.sh $out $name)"; return 1
  fi
  if has_model "$name"; then note "import $name: exists"; else
    python3 -m homeai.train.ollama_import --run "$out" --name "$name" --base "$base" >> outputs/overnight.log 2>&1 \
      && note "import $name: done" || { note "import $name: FAILED"; return 1; }
  fi
}

latency_run() {  # name -- idle-GPU speed bench (seconds per model turn + per request)
  local out="outputs/eval/$1-latency"
  if [ -f "$out/latency.json" ] && grep -q '"training_running": false' "$out/latency.json"; then note "latency $1: exists"; return 0; fi
  wait_for_train
  note "latency $1: starting"
  python3 -m homeai.eval.latency --model "$1" --ollama "$(model_url "$1")" --n 60 --out "$out" >> outputs/overnight.log 2>&1 </dev/null \
    && note "latency $1: done $(python3 -c "import json;r=json.load(open('$out/latency.json'));print(f\"{r['seconds_per_call']:.2f} s/turn, {r['cold_mean']:.2f} s/request\")")" \
    || note "latency $1: FAILED"
  docker exec "$(model_ctr "$1")" ollama stop "$1" >/dev/null 2>&1
}

run_set() {  # model set
  case $2 in
    latency)   latency_run "$1" ;;
    test-v2)   eval_run "$1" data/generated/test-v2.jsonl -testv2 ;;
    wild)      eval_run "$1" data/generated/wild.jsonl -wild ;;
    para)      eval_run "$1" data/generated/test-v2-para.jsonl -para ;;
    wild2)     eval_run "$1" data/generated/wild2.jsonl -wild2 ;;
    wild-e2e)  e2e_run "$1" data/generated/wild.jsonl -wild-e2e ;;
    wild2-e2e) e2e_run "$1" data/generated/wild2.jsonl -wild2-e2e ;;
    test-v2-e2e) e2e_run "$1" data/generated/test-v2.jsonl -testv2-e2e ;;
    rules)     eval_run "$1" data/generated/test-rules.jsonl -rules ;;
  esac
}

fill_missing() {  # every candidate model gets every current test set + a clean speed bench
  local todo line m s
  mapfile -t todo < <(python3 -m homeai.eval.champions --missing)
  for line in "${todo[@]}"; do
    read -r m s <<< "$line"
    has_model "$m" && run_set "$m" "$s"
  done
}

gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc 0-9; }

fill_quality() {  # [skip-regex] accuracy evals only (reports' timings are ignored); safe to run beside training
  local todo line m s skip=${1:-^$}
  # beside a training run only 4B models fit (training + epoch-end validation spike + a 9B OOM'd the 4b-v3 run)
  local only4b=""; docker ps -q -f name=homeai-train | grep -q . && only4b="-4b"
  mapfile -t todo < <(python3 -m homeai.eval.champions --missing | grep -v " latency$" | grep -Ev "$skip" | grep -- "$only4b")
  for line in "${todo[@]}"; do
    [ "$(gpu_free_mb)" -ge 10000 ] || sleep 30   # Ollama may still be unloading the previous model
    [ "$(gpu_free_mb)" -ge 10000 ] || { note "quality evals: paused, GPU memory short ($(gpu_free_mb) MiB free)"; return 0; }
    read -r m s <<< "$line"
    has_model "$m" && run_set "$m" "$s"
  done
}

free_gb() { df -BG --output=avail /ai | tail -1 | tr -dc 0-9; }

prune_if_short() {  # Peter: keep old models unless space runs short. merged16 (rebuildable) first, then non-champion models.
  local cand line m cfg out
  [ "$(free_gb)" -ge 60 ] && return 0
  mapfile -t cand < <(python3 -m homeai.eval.champions --prunable)
  for line in "${cand[@]}"; do
    read -r m cfg <<< "$line"; out=$(python3 -c "import json;print(json.load(open('$cfg'))['out'])")
    if [ -d "$out/merged16" ]; then rm -rf "$out/merged16"; note "prune: removed $out/merged16 ($(free_gb)G free now)"; fi
    [ "$(free_gb)" -ge 60 ] && return 0
  done
  [ "$(free_gb)" -ge 30 ] && return 0
  for line in "${cand[@]}"; do
    read -r m cfg <<< "$line"; out=$(python3 -c "import json;print(json.load(open('$cfg'))['out'])")
    rm -rf "$out/gguf" "$out/checkpoints"; docker exec ollama ollama rm "$m" >/dev/null 2>&1
    note "prune: removed model $m (LoRA adapter + reports kept; $(free_gb)G free now)"
    [ "$(free_gb)" -ge 30 ] && return 0
  done
}

e2e_run() {  # name data suffix -- outcome eval (free-running loop, judged on final state + spoken facts)
  local name=$1 data=$2 tag=$3 out="outputs/eval/$1$3"
  if [ -f "$out/report.json" ]; then note "e2e $name$tag: exists"; return 0; fi
  note "e2e $name$tag ($data): starting"
  python3 -m homeai.eval.e2e --model "$name" --ollama "$(model_url "$name")" --data "$data" --out "$out" >> outputs/overnight.log 2>&1 </dev/null \
    && note "e2e $name$tag: done $(python3 -c "import json;print(f\"pass={json.load(open('$out/report.json'))['pass']:.1%}\")")" \
    || note "e2e $name$tag: FAILED"
  docker exec "$(model_ctr "$name")" ollama stop "$name" >/dev/null 2>&1
}

rft_run() {  # name [select args] -- rejection-sampling reinforcement from the best 4B by outcome score (Peter, 24 Sep)
  local name=$1 n=${1##*-} run="outputs/rft/$1" data cfg base mix; shift; local sel_args="$*"
  data="data/generated/train-$n.jsonl"; cfg="configs/qwen35-4b-$n.json"
  if has_model "$name"; then note "rft $name: exists"; return 0; fi
  mkdir -p "$run"
  if [ -n "${RFT_SAMPLES_FROM:-}" ] && [ ! -f "$run/samples.jsonl" ]; then   # re-select an earlier run's attempts
    cp "$RFT_SAMPLES_FROM/samples.jsonl" "$RFT_SAMPLES_FROM/base.txt" "$RFT_SAMPLES_FROM/pool_ids.json" "$run/"
  fi
  if [ ! -f "$cfg" ]; then
    fill_missing; python3 -m homeai.eval.champions > /dev/null
    if [ -f "$run/base.txt" ]; then base=$(cat "$run/base.txt"); else
      base=$(python3 -m homeai.train.rft pick-base 2>> outputs/rft-$n.log) || { note "rft $name: no 4B with a complete outcome score yet"; return 1; }
      echo "$base" > "$run/base.txt"
    fi
    wait_for_train
    note "rft $name: sampling 4 attempts per training request from $base"
    python3 -m homeai.train.rft sample --model "$base" --out "$run" >> outputs/rft-$n.log 2>&1 </dev/null || { note "rft $name: sampling FAILED - see outputs/rft-$n.log"; return 1; }
    docker exec ollama ollama stop "$base" >/dev/null 2>&1
    mix=$(python3 -c "import json;m=json.load(open('configs/models.json'));print(json.load(open(m['$base']['config']))['train'])")
    python3 -m homeai.train.rft select --run "$run" --out "$data" --mix "$mix" $sel_args >> outputs/rft-$n.log 2>&1 || { note "rft $name: select FAILED"; return 1; }
    note "rft $name: $(python3 -c "import json;s=json.load(open('$run/select.json'));print(f\"solved {s['solve_rate']:.0%} of {s['turns']} turns, {s['mean_calls_chosen']:.2f} calls/turn chosen vs {s['mean_calls_reference_same_turns']:.2f} in the references, {s['examples']} examples\")")"
    python3 -m homeai.train.rft make-config --base "$base" --name "$name" --train "$data" >> outputs/rft-$n.log 2>&1 || { note "rft $name: config FAILED"; return 1; }
  fi
  train_run "$cfg" && export_run "$cfg" "$name" qwen3.5:4b
  fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
}

eval_run() {  # name [data] [suffix]
  local name=$1 data=${2:-data/generated/test.jsonl} tag=${3:-}
  local out="outputs/eval/$name$tag"
  if [ -f "$out/report.json" ]; then note "eval $name$tag: exists"; return 0; fi
  note "eval $name$tag ($data): starting"
  python3 -m homeai.eval.run_eval --model "$name" --ollama "$(model_url "$name")" --data "$data" --think off --out "$out" >> outputs/overnight.log 2>&1 \
    && note "eval $name$tag: done $(python3 -c "import json;print(f\"pass={json.load(open('$out/report.json'))['pass']:.1%}\")")" \
    || note "eval $name$tag: FAILED"
  docker exec "$(model_ctr "$name")" ollama stop "$name" >/dev/null 2>&1
}

baseline_full() {  # tag
  local tag=$1
  local name="baseline-${tag//:/-}-full"
  [ -f "outputs/eval/$name/report.json" ] && return 0
  note "baseline $tag full test set: starting"
  python3 -m homeai.eval.run_eval --model "$tag" --data data/generated/test.jsonl --think off --out "outputs/eval/$name" >> outputs/overnight.log 2>&1 && note "baseline $tag: done" || note "baseline $tag: FAILED"
  docker exec ollama ollama stop "$tag" >/dev/null 2>&1
}

note "=== overnight pipeline start ==="
# one-off: accuracy evals only, e.g. beside a driver that is busy training:  scripts/overnight.sh --quality-evals " para$"
if [ "${1:-}" = "--quality-evals" ]; then fill_quality "${2:-^$}"; note "quality evals (one-off): done"; exit 0; fi
wait_for_train

train_run configs/qwen35-4b.json      && export_run configs/qwen35-4b.json homeai-4b qwen3.5:4b      && eval_run homeai-4b
train_run configs/qwen35-9b.json      && export_run configs/qwen35-9b.json homeai-9b qwen3.5:9b      && eval_run homeai-9b
baseline_full qwen3.5:4b
baseline_full qwen3.5:9b
train_run configs/qwen35-4b-lora.json && export_run configs/qwen35-4b-lora.json homeai-4b-lora qwen3.5:4b && eval_run homeai-4b-lora data/generated/test-v1.jsonl
# ---- phase 2: v2 data (media, appliances, water, vehicle; paraphrased corpus), scored on test v2
if [ -f data/generated/test-v2.jsonl ]; then
  train_run configs/qwen35-4b-v2.json && export_run configs/qwen35-4b-v2.json homeai-4b-v2 qwen3.5:4b && eval_run homeai-4b-v2 data/generated/test-v2.jsonl && eval_run homeai-4b-v2 data/generated/wild.jsonl -wild
  for m in homeai-4b homeai-9b homeai-4b-lora; do has_model $m && eval_run $m data/generated/test-v2.jsonl -testv2; done
  eval_run qwen3.5:4b data/generated/test-v2.jsonl -baseline-testv2
  eval_run qwen3.5:9b data/generated/test-v2.jsonl -baseline-testv2
  train_run configs/qwen35-9b-v2.json && export_run configs/qwen35-9b-v2.json homeai-9b-v2 qwen3.5:9b && eval_run homeai-9b-v2 data/generated/test-v2.jsonl -testv2 && eval_run homeai-9b-v2 data/generated/wild.jsonl -wild
fi
# ---- phase 3: v3 data (held-out paraphrase split + agent-written training conversations), phrasing test = test-v2-para
if [ -f data/generated/train-v3.jsonl ]; then
  # 4b-v3 training was OOM-killed at step 422 on 24 Sep (concurrent 9B eval); skipped in favour of v4 (a superset of its data)
  for m in homeai-4b-v2 homeai-9b-v2 homeai-4b-lora; do has_model $m && eval_run $m data/generated/test-v2-para.jsonl -para; done
fi
# ---- phase 4: v4 data = v3 + second batch of agent-written training conversations (speaker-relative, look-before-acting, mixed, missing)
if [ -f data/generated/train-v4.jsonl ]; then
  train_run configs/qwen35-4b-v4.json && export_run configs/qwen35-4b-v4.json homeai-4b-v4 qwen3.5:4b
  fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
fi
# ---- phase 5: bf16 LoRA on v2 data (best data so far; v4 additions did not help)
if [ -f data/generated/train.jsonl ]; then
  train_run configs/qwen35-4b-lora-v2.json && export_run configs/qwen35-4b-lora-v2.json homeai-4b-lora-v2 qwen3.5:4b
  fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
fi
# ---- phase 6: reinforcement (rejection sampling with the outcome reward) from the best 4B so far
rft_run homeai-4b-rft1
# rft1 learned to unlock/disarm without asking (pre-confirmed wins) and to answer tersely (shortest-reply tie-break):
# same attempts, re-selected with every key fact required, no unconfirmed sensitive actions, length-neutral tie-break
RFT_SAMPLES_FROM=outputs/rft/homeai-4b-rft1 rft_run homeai-4b-rft2 --strict-facts --confirm-sensitive --tiebreak ref-length
# ---- phase 7: v5 data (agent conversations once, normalised to template conventions, + clarify/solar-forecast batches)
if [ -f data/generated/train-v5.jsonl ]; then
  train_run configs/qwen35-4b-v5.json && export_run configs/qwen35-4b-v5.json homeai-4b-v5 qwen3.5:4b
  fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
fi
# ---- phase 8: another base model (Peter, 25 Sep): Gemma 4 E4B on v2 data, QLoRA (bf16 base OOMs on the 3090)
if [ -f configs/gemma4-e4b-v2.json ]; then
  train_run configs/gemma4-e4b-v2.json && export_run configs/gemma4-e4b-v2.json homeai-gemma4-e4b-v2 gemma4:e4b
  fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
fi
# ---- phase 9: champion data (v5) with the champion method (bf16 LoRA)
if [ -f data/generated/train-v5.jsonl ]; then
  train_run configs/qwen35-4b-lora-v5.json && export_run configs/qwen35-4b-lora-v5.json homeai-4b-lora-v5 qwen3.5:4b
  fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
fi
# ---- phase 11: LFM2.5 1.2B / 350M / 230M with the champion recipe (Peter, 25 Sep)
for s_ in 1.2b 350m 230m; do
  if [ -f configs/lfm25-$s_-lora-v5.json ]; then
    train_run configs/lfm25-$s_-lora-v5.json && export_run configs/lfm25-$s_-lora-v5.json homeai-lfm25-$s_-lora-v5 lfm2.5-$s_
  fi
done
fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
# ---- architecture track, design A baseline (Peter, 26 Sep): the champion as deployed (flat inventory prompt, 8k context)
# on the large-house benchmark (3k / 10k / 20k entities + the Bundook plan). Not a champion set.
if [ -f data/generated/bigbench.jsonl ] && has_model homeai-4b-lora-v5; then
  wait_for_train
  e2e_run homeai-4b-lora-v5 data/generated/bigbench.jsonl -bigbench-e2e
fi
# ---- phase 12: Spark X2.5 4B (Peter, 25 Sep): transformers+PEFT backend; exported by hand (Ollama 0.34.4 path)
if [ -f configs/spark-x25-4b-lora-v5.json ]; then
  if train_run configs/spark-x25-4b-lora-v5.json; then
    has_model homeai-spark-x25-4b-lora-v5 || { note "export homeai-spark-x25-4b-lora-v5: starting"; scripts/export-spark.sh configs/spark-x25-4b-lora-v5.json homeai-spark-x25-4b-lora-v5 >> outputs/overnight.log 2>&1 </dev/null && note "export homeai-spark-x25-4b-lora-v5: done" || note "export homeai-spark-x25-4b-lora-v5: FAILED"; }
    fill_missing; python3 -m homeai.eval.champions > /dev/null
  fi
fi
# ---- phase 10: v6 data (confirmation safety + named-device clarification) with the champion method
if [ -f data/generated/train-v6.jsonl ]; then
  train_run configs/qwen35-4b-lora-v6.json && export_run configs/qwen35-4b-lora-v6.json homeai-4b-lora-v6 qwen3.5:4b
  fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
fi
# ---- phase 13: Gemma 4 E4B again, v5 data, tool-response mask fixed (keep <|tool_response> as a label); converted on the Mac
if [ -f configs/gemma4-e4b-v5.json ]; then
  train_run configs/gemma4-e4b-v5.json && export_run configs/gemma4-e4b-v5.json homeai-gemma4-e4b-v5 gemma4:e4b
  fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
fi
# ---- architecture track, designs C/D (Peter, 26 Sep): Needle 3 room leaf, generic (rooms from many houses), LoRA on the GPU
if [ -x .needle-venv/bin/needle ] && [ -f data/generated/leaf-generic-train.jsonl ] && [ ! -f outputs/leaf/generic-r16-e3/test-bundook.summary.json ]; then
  wait_for_train
  note "leaf generic-r16-e3: starting"
  scripts/train-leaf.sh generic-r16-e3 data/generated/leaf-generic-train.jsonl --epochs 3 --val-split 0.05 >> outputs/overnight.log 2>&1 </dev/null \
    && note "leaf generic-r16-e3: done $(python3 -c "import json;s=json.load(open('outputs/leaf/generic-r16-e3/test-bundook.summary.json'));print({k:s[k] for k in list(s)[:4]})" 2>/dev/null)" \
    || note "leaf generic-r16-e3: FAILED - see outputs/leaf/generic-r16-e3/"
fi
# ---- architecture track: design A with a 32k context window (does a bigger window alone fix large houses?)
if [ -f data/generated/bigbench.jsonl ] && has_model homeai-4b-lora-v5 && [ ! -f outputs/eval/homeai-4b-lora-v5-bigbench32k-e2e/report.json ]; then
  wait_for_train
  note "e2e homeai-4b-lora-v5-bigbench32k-e2e: starting"
  python3 -m homeai.eval.e2e --model homeai-4b-lora-v5 --data data/generated/bigbench.jsonl --num-ctx 32768 --out outputs/eval/homeai-4b-lora-v5-bigbench32k-e2e >> outputs/overnight.log 2>&1 </dev/null \
    && note "e2e homeai-4b-lora-v5-bigbench32k-e2e: done $(python3 -c "import json;print(f\"pass={json.load(open('outputs/eval/homeai-4b-lora-v5-bigbench32k-e2e/report.json'))['pass']:.1%}\")")" \
    || note "e2e homeai-4b-lora-v5-bigbench32k-e2e: FAILED"
  docker exec ollama ollama stop homeai-4b-lora-v5 >/dev/null 2>&1
fi
# ---- phase 15: v7 data = v5 + Peter's clarification rules (26 Sep): ask when the speaker's room has several matches, act when the house has one
if [ -f data/generated/train-v7.jsonl ]; then
  train_run configs/qwen35-4b-lora-v7.json && export_run configs/qwen35-4b-lora-v7.json homeai-4b-lora-v7 qwen3.5:4b
  fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
fi
# ---- architecture track, design B (Peter, 26 Sep): rooms-only prompt + find_devices / filtered get_state; outcome evals on
# the B versions of the sets (same requests and houses as A's), compared with homeai-4b-lora-v5 on the A versions
if [ -f configs/qwen35-4b-lora-b.json ] && [ -f data/generated/bigbench-b.jsonl ]; then
  train_run configs/qwen35-4b-lora-b.json && export_run configs/qwen35-4b-lora-b.json homeai-4b-lora-b qwen3.5:4b && {
    e2e_run homeai-4b-lora-b data/generated/bigbench-b.jsonl -bigbench-b-e2e
    e2e_run homeai-4b-lora-b data/generated/test-v2-b.jsonl -testv2-b-e2e
    e2e_run homeai-4b-lora-b data/generated/wild-b.jsonl -wild-b-e2e
    e2e_run homeai-4b-lora-b data/generated/wild2-b.jsonl -wild2-b-e2e
  }
  prune_if_short
fi
# ---- architecture track, designs C/D: the orchestrator (B's tools + ask_rooms routing to room leaves); pipeline evals run
# against room leaves served from lapwing (homeai.tiered.eval), launched separately
if [ -f configs/qwen35-4b-lora-orch.json ] && [ -f data/generated/train-orch.jsonl ]; then
  train_run configs/qwen35-4b-lora-orch.json && export_run configs/qwen35-4b-lora-orch.json homeai-4b-lora-orch qwen3.5:4b
  prune_if_short
fi
# ---- context-length sweep (Peter, 27 Sep): accuracy, speed and GPU memory against num_ctx on the large-house benchmark
ctx_run() {  # model data tag num_ctx
  local m=$1 data=$2 tag=$3 ctx=$4 out="outputs/eval/$1-$3-e2e"
  [ -f "$out/report.json" ] && return 0
  has_model "$m" || return 0
  wait_for_train
  note "e2e $m-$tag-e2e (num_ctx $ctx): starting"
  python3 -m homeai.eval.e2e --model "$m" --ollama "$(model_url "$m")" --data "$data" --num-ctx "$ctx" --out "$out" >> outputs/overnight.log 2>&1 </dev/null \
    && docker exec "$(model_ctr "$m")" ollama ps > "$out/ollama_ps.txt" 2>&1; \
  [ -f "$out/report.json" ] && note "e2e $m-$tag-e2e: done $(python3 -c "import json;r=json.load(open('$out/report.json'));print(f\"pass={r['pass']:.1%} s/turn={r['seconds_per_turn']:.2f}\")") | $(awk 'NR==2{print $3, $4}' "$out/ollama_ps.txt")" || note "e2e $m-$tag-e2e: FAILED"
  docker exec "$(model_ctr "$m")" ollama stop "$m" >/dev/null 2>&1
}
if [ -f data/generated/bigbench.jsonl ]; then
  for c in 12 16 24 48; do ctx_run homeai-4b-lora-v5 data/generated/bigbench.jsonl bigbench${c}k $((c * 1024)); done
  ctx_run homeai-4b-lora-v5 data/generated/bigbench.jsonl bigbench8kps 8192     # 8k again, for its GPU memory
  ctx_run homeai-4b-lora-v5 data/generated/bigbench.jsonl bigbench32kps 32768   # 32k again, for its GPU memory
  [ -f data/generated/bigbench-b.jsonl ] && for c in 8 16; do ctx_run homeai-4b-lora-b data/generated/bigbench-b.jsonl bigbench-b${c}k $((c * 1024)); done
fi
# ---- architecture track (Peter, 27 Sep): other candidates on the large-house benchmark with a 32k window (the regular sets
# never exceed ~4k tokens, so their scores can't change; bigbench turns reach ~36k)
if [ -f data/generated/bigbench.jsonl ]; then
  for m in homeai-4b-lora-v7 homeai-gemma4-e4b-v5 homeai-4b-v5 homeai-9b-v2; do
    has_model "$m" || continue
    out=outputs/eval/$m-bigbench32k-e2e; [ -f "$out/report.json" ] && continue
    wait_for_train
    note "e2e $m-bigbench32k-e2e: starting"
    python3 -m homeai.eval.e2e --model "$m" --ollama "$(model_url "$m")" --data data/generated/bigbench.jsonl --num-ctx 32768 --out "$out" >> outputs/overnight.log 2>&1 </dev/null \
      && note "e2e $m-bigbench32k-e2e: done $(python3 -c "import json;print(f\"pass={json.load(open('$out/report.json'))['pass']:.1%}\")")" \
      || note "e2e $m-bigbench32k-e2e: FAILED"
    docker exec "$(model_ctr "$m")" ollama stop "$m" >/dev/null 2>&1
  done
fi
# ---- Qwen3.8 27B, untuned (Peter, 27 Sep): the only new Qwen3.8 size that fits (Flash-Next is ~180B stored, 2.4T is 2.4T);
# candidate slow model for design E, orchestrator, and a ceiling for what a much larger model buys here
if docker exec ollama ollama list | grep -q "^qwen3.8:27b " || { note "pull qwen3.8:27b: starting"; docker exec ollama ollama pull qwen3.8:27b >/dev/null 2>&1 && note "pull qwen3.8:27b: done"; }; then
  wait_for_train
  eval_run qwen3.8:27b data/generated/test-v2.jsonl -baseline-testv2
  e2e_run qwen3.8:27b data/generated/wild.jsonl -wild-e2e
  ctx_run qwen3.8:27b data/generated/bigbench.jsonl bigbench32k 32768
  docker exec ollama ollama stop qwen3.8:27b >/dev/null 2>&1
fi
# ---- phase 16, traditional fallback track (Peter, 27 Sep): v8 = v7 + large-house conversations in the champion's format,
# 8k sequences; scored on the regular sets like every champion candidate, plus the large-house benchmark at 32k
if [ -f data/generated/train-v8.jsonl ]; then
  train_run configs/qwen35-4b-lora-v8.json && export_run configs/qwen35-4b-lora-v8.json homeai-4b-lora-v8 qwen3.5:4b && {
    fill_missing; python3 -m homeai.eval.champions > /dev/null
    out=outputs/eval/homeai-4b-lora-v8-bigbench32k-e2e
    [ -f "$out/report.json" ] || { note "e2e homeai-4b-lora-v8-bigbench32k-e2e: starting"; python3 -m homeai.eval.e2e --model homeai-4b-lora-v8 --data data/generated/bigbench.jsonl --num-ctx 32768 --out "$out" >> outputs/overnight.log 2>&1 </dev/null && note "e2e homeai-4b-lora-v8-bigbench32k-e2e: done $(python3 -c "import json;print(f\"pass={json.load(open('$out/report.json'))['pass']:.1%}\")")" || note "e2e homeai-4b-lora-v8-bigbench32k-e2e: FAILED"; docker exec ollama ollama stop homeai-4b-lora-v8 >/dev/null 2>&1; }
  }
  prune_if_short
fi
# ---- design E grid (Peter, 28 Sep): per-turn logs (resulting changes + timing) for every fast/slow candidate on the three
# outcome sets, so any fast x slow pair can be reconciled offline (homeai/eval/reconcile.py). Separate dirs, not champion inputs.
for m in homeai-4b-lora-v5 homeai-4b-lora-v7 homeai-4b-lora-v8 homeai-9b-v2 homeai-gemma4-e4b-v5 homeai-4b-v5; do
  has_model "$m" || continue
  for s in test-v2:testv2 wild:wild wild2:wild2; do
    d=${s%%:*}; t=${s##*:}; out=outputs/eval-turns/$m-$t
    [ -f "$out/turns.jsonl" ] && [ -f "$out/report.json" ] && continue
    wait_for_train
    note "turns $m-$t: starting"
    python3 -m homeai.eval.e2e --model "$m" --ollama "$(model_url "$m")" --data data/generated/$d.jsonl --out "$out" >> outputs/overnight.log 2>&1 </dev/null \
      && note "turns $m-$t: done $(python3 -c "import json;r=json.load(open('$out/report.json'));print(f\"pass={r['pass']:.1%} s/call={r['seconds_per_turn']:.2f}\")")" || note "turns $m-$t: FAILED"
  done
  docker exec "$(model_ctr "$m")" ollama stop "$m" >/dev/null 2>&1
done
# ---- Gemma 4 E4B on v8 data (Peter, 28 Sep: revisit earlier models with the newer data): fast-model candidate for design E
if [ -f configs/gemma4-e4b-v8.json ] && [ -f data/generated/train-v8.jsonl ]; then
  train_run configs/gemma4-e4b-v8.json && export_run configs/gemma4-e4b-v8.json homeai-gemma4-e4b-v8 gemma4:e4b
  prune_if_short
fi
# ---- phase 14: champion recipe with rank-32 adapters (capacity only; same v5 data, 2 epochs)
if [ -f configs/qwen35-4b-lora-v5-r32.json ]; then
  train_run configs/qwen35-4b-lora-v5-r32.json && export_run configs/qwen35-4b-lora-v5-r32.json homeai-4b-lora-v5-r32 qwen3.5:4b
  fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
fi
# ---- every model on every current test set + idle-GPU speed, then champions (outputs/CHAMPIONS.md, results.csv)
fill_missing; python3 -m homeai.eval.champions > /dev/null && note "champions: $(python3 -c "import json;c=json.load(open('outputs/CHAMPIONS.json'));print(f\"best={c['best']} best<=1s={c['best_fast']}\")")"
prune_if_short
python3 -m homeai.eval.summarize > /dev/null 2>&1 && note "results: outputs/RESULTS.md written"
note "=== overnight pipeline finished ==="
