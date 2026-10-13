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
  if [ -d "$out/lora" ]; then note "train $cfg: done (by a run this driver waited for)"; return 0; fi
  ensure_disk 8 "train $cfg" || return 1
  unload_all
  note "train $cfg: starting"
  docker rm -f homeai-train >/dev/null 2>&1
  DETACH=-d scripts/train-docker.sh train "$cfg" >/dev/null
  sleep 600; fill_quality; wait_for_train   # accuracy evals of earlier models share the GPU while it trains
  own
  if [ -d "$out/lora" ]; then note "train $cfg: done"; else note "train $cfg: FAILED (no lora dir) - see outputs/$(basename "$cfg" .json)-train.log"; return 1; fi
}

unload_all() {  # stop every model both Ollama containers hold (Frigate's own GPU use is left alone)
  local c m
  for c in ollama ollama-next; do
    for m in $(docker exec "$c" ollama ps 2>/dev/null | awk 'NR>1{print $1}'); do docker exec "$c" ollama stop "$m" >/dev/null 2>&1; done
  done
  sleep 5
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
  # removed by prune_if_short for space: rebuilding it would refill the disk (the 28 Sep loop). rm the marker to rebuild.
  if [ -f "$out/.pruned" ]; then note "import $name: pruned for space, not rebuilding (rm $out/.pruned to rebuild)"; return 0; fi
  if [ -f "$out/merged16/config.json" ]; then note "merge $cfg: exists"; else
    wait_for_train   # never share the GPU with a training run
    case $base in *9b*|gemma4*) ensure_disk 20 "merge $cfg" || return 1;; *2b*|*0.8b*|lfm*) ensure_disk 6 "merge $cfg" || return 1;; *) ensure_disk 10 "merge $cfg" || return 1;; esac
    note "merge $cfg: starting"
    docker rm -f homeai-train homeai-merge >/dev/null 2>&1
    docker network disconnect -f proxy homeai-merge >/dev/null 2>&1   # a stale endpoint failed every merge on 28 Sep
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
  # off-box copy on Piper (Peter, 29 Sep); Ollama keeps its own copy, so the local portable GGUF then goes
  if [ -d "$out/gguf" ] && [ ! -f "$out/.archived" ]; then
    scripts/archive.sh push "$out" >> "$STATUS" 2>&1 && scripts/archive.sh drop-gguf "$out" >> "$STATUS" 2>&1
  fi
  return 0
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

# Frigate's database shares this disk (/ai/frigate/media): a full disk on 28 Sep killed its recorder and the camera
# streams. Every heavy write first reserves its size plus RESERVE_GB, pruning rebuildable outputs, and skips if it can't.
RESERVE_GB=25   # Frigate's DB is ~100 MB; the danger is 0 bytes free, not a small margin

ensure_disk() {  # need_gb what
  local want=$(( $1 + RESERVE_GB ))
  [ "$(free_gb)" -ge "$want" ] && return 0
  prune_if_short "$want" "$want"
  [ "$(free_gb)" -ge "$want" ] && return 0
  note "$2: skipped, disk short ($(free_gb)G free, need $1G + ${RESERVE_GB}G reserve)"; return 1
}

prune_if_short() {  # [soft_gb] [hard_gb] -- Peter: keep old models unless space runs short. Rebuildable merged16 first
  # (models already in Ollama, then prunable ones); below hard_gb, non-champion models themselves.
  local soft=${1:-60} hard=${2:-$((RESERVE_GB + 10))} cand line m cfg out d
  [ "$(free_gb)" -ge "$soft" ] && return 0
  for d in outputs/*/merged16; do
    [ -f "$d/config.json" ] || continue
    m=$(python3 -c "import json,os;print(next((k for k,v in json.load(open('configs/models.json')).items() if not k.startswith('_') and v.get('config') and os.path.exists(v['config']) and json.load(open(v['config'])).get('out')=='${d%/merged16}'),''))" 2>/dev/null)
    [ -n "$m" ] && has_model "$m" || continue
    rm -rf "$d"; note "prune: removed $d, $m is in Ollama ($(free_gb)G free now)"
    [ "$(free_gb)" -ge "$soft" ] && return 0
  done
  mapfile -t cand < <(python3 -m homeai.eval.champions --prunable)
  for line in "${cand[@]}"; do
    read -r m cfg <<< "$line"; out=$(python3 -c "import json;print(json.load(open('$cfg'))['out'])")
    if [ -d "$out/merged16" ]; then rm -rf "$out/merged16"; note "prune: removed $out/merged16 ($(free_gb)G free now)"; fi
    [ "$(free_gb)" -ge "$soft" ] && return 0
  done
  [ "$(free_gb)" -ge "$hard" ] && return 0
  for line in "${cand[@]}"; do
    read -r m cfg <<< "$line"; out=$(python3 -c "import json;print(json.load(open('$cfg'))['out'])")
    has_model "$m" || [ -d "$out/gguf" ] || continue   # already gone
    # only models with a verified copy on Piper leave the box (scripts/archive.sh restore brings one back)
    [ -f "$out/.archived" ] || { [ -d "$out/gguf" ] && scripts/archive.sh push "$out" >> "$STATUS" 2>&1; } || { note "prune: keeping $m (no archived copy)"; continue; }
    rm -rf "$out/gguf" "$out/checkpoints"; docker exec "$(model_ctr "$m")" ollama rm "$m" >/dev/null 2>&1; touch "$out/.pruned"
    note "prune: removed model $m (archived on Piper: scripts/archive.sh restore; $(free_gb)G free now)"
    [ "$(free_gb)" -ge "$hard" ] && return 0
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
    [ -f outputs/spark-x25-4b-lora-v5/.pruned ] || has_model homeai-spark-x25-4b-lora-v5 || { note "export homeai-spark-x25-4b-lora-v5: starting"; scripts/export-spark.sh configs/spark-x25-4b-lora-v5.json homeai-spark-x25-4b-lora-v5 >> outputs/overnight.log 2>&1 </dev/null && note "export homeai-spark-x25-4b-lora-v5: done" || note "export homeai-spark-x25-4b-lora-v5: FAILED"; }
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
if docker exec ollama-next ollama list | grep -q "^qwen3.8:27b " || { ensure_disk 20 "pull qwen3.8:27b" && note "pull qwen3.8:27b: starting" && { docker exec ollama-next ollama pull qwen3.8:27b 2>&1 | tr '\r' '\n' | grep -v "pulling\|verifying" | tail -3 >> outputs/overnight.log; has_model qwen3.8:27b; } && note "pull qwen3.8:27b: done" || { note "pull qwen3.8:27b: FAILED - see outputs/overnight.log"; false; }; }; then
  wait_for_train
  eval_run qwen3.8:27b data/generated/test-v2.jsonl -baseline-testv2
  e2e_run qwen3.8:27b data/generated/wild.jsonl -wild-e2e
  ctx_run qwen3.8:27b data/generated/bigbench.jsonl bigbench32k 32768
  docker exec ollama-next ollama stop qwen3.8:27b >/dev/null 2>&1
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
# ---- design C with a 2B orchestrator (Peter, 29 Sep: train on the GPU boxes, not the Macs; MLX on the M5 needed ~27 h)
if [ -f configs/qwen35-2b-lora-orch.json ] && [ -f data/generated/train-orch.jsonl ]; then
  train_run configs/qwen35-2b-lora-orch.json && export_run configs/qwen35-2b-lora-orch.json homeai-2b-lora-orch qwen3.5:2b
  prune_if_short
fi
# ---- v9 (Peter, 29 Sep: voice robustness): v8 + speech-to-text round trips of its own training turns (Kokoro/Piper ->
# parakeet/whisper), STT-style casing and homophone/near-miss augmentation; leakage-checked against every test set
if [ -f configs/qwen35-4b-lora-v9.json ] && [ -f data/generated/train-v9.jsonl ]; then
  train_run configs/qwen35-4b-lora-v9.json && export_run configs/qwen35-4b-lora-v9.json homeai-4b-lora-v9 qwen3.5:4b
  fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
fi
# ---- v10 (Peter, 29 Sep: prompt caching for low-power hosts): v9's data with the stable-prefix layout -- time, speaker and
# presence moved to a "Current context:" block at the end of the system prompt, so a caching server reuses the static part
if [ -f configs/qwen35-4b-lora-v10.json ] && [ -f data/generated/train-v10.jsonl ]; then
  train_run configs/qwen35-4b-lora-v10.json && export_run configs/qwen35-4b-lora-v10.json homeai-4b-lora-v10 qwen3.5:4b
  fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
fi
# ---- B-v2: design B (lookup tools) with v9's speech-to-text augmentation, stable-prefix layout
if [ -f configs/qwen35-4b-lora-b-v2.json ] && [ -f data/generated/train-b-v2.jsonl ]; then
  train_run configs/qwen35-4b-lora-b-v2.json && export_run configs/qwen35-4b-lora-b-v2.json homeai-4b-lora-b-v2 qwen3.5:4b && {
    e2e_run homeai-4b-lora-b-v2 data/generated/bigbench-b.jsonl -bigbench-b-e2e
    e2e_run homeai-4b-lora-b-v2 data/generated/test-v2-b.jsonl -testv2-b-e2e
    e2e_run homeai-4b-lora-b-v2 data/generated/wild-b.jsonl -wild-b-e2e
    e2e_run homeai-4b-lora-b-v2 data/generated/wild2-b.jsonl -wild2-b-e2e
  }
  prune_if_short
fi
# ---- v11 (30 Sep): v10 regressed on clarification (wild_clarify 72->56) with the context block at the end of the system
# prompt; user_context puts "[context] Now | Speaker | People" at the start of each user turn instead (still cacheable)
if [ -f configs/qwen35-4b-lora-v11.json ]; then
  train_run configs/qwen35-4b-lora-v11.json && export_run configs/qwen35-4b-lora-v11.json homeai-4b-lora-v11 qwen3.5:4b
  fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
fi
# ---- Spark X2.5 4B on v9 data (1 Oct): the fastest 4B, made speech-robust like v9
if [ -f configs/spark-x25-4b-lora-v9.json ]; then
  if train_run configs/spark-x25-4b-lora-v9.json; then
    [ -f outputs/spark-x25-4b-lora-v9/.pruned ] || has_model homeai-spark-x25-4b-lora-v9 || { note "export homeai-spark-x25-4b-lora-v9: starting"; scripts/export-spark.sh configs/spark-x25-4b-lora-v9.json homeai-spark-x25-4b-lora-v9 >> outputs/overnight.log 2>&1 </dev/null && note "export homeai-spark-x25-4b-lora-v9: done" || note "export homeai-spark-x25-4b-lora-v9: FAILED"; }
    fill_missing; python3 -m homeai.eval.champions > /dev/null
  fi
  prune_if_short
fi
# ---- v12 (1 Oct): only the clock moves to the user turn ("[now] ..."); Speaker / People stay where v9 learned them, so
# clarification should hold while the system message stays identical per room and presence state (cacheable)
if [ -f configs/qwen35-4b-lora-v12.json ]; then
  train_run configs/qwen35-4b-lora-v12.json && export_run configs/qwen35-4b-lora-v12.json homeai-4b-lora-v12 qwen3.5:4b
  fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
fi
# ---- 2B B-v2 (2 Oct): design B on Qwen3.5-2B, trained on lapwing (accuracy there: bigbench-b 77.6 vs the 4B's 81.2) and
# imported here; timed on the idle GPU against homeai-4b-lora-b-v2, after the Spark rebuild (outputs/spark-rebuild.sh) lets go
if has_model homeai-2b-lora-b-v2; then
  while pgrep -f "outputs/spark-rebuil[d]" >/dev/null; do sleep 60; done
  wait_for_train; unload_all
  e2e_run homeai-2b-lora-b-v2 data/generated/bigbench-b.jsonl -bigbench-b-e2e
  e2e_run homeai-2b-lora-b-v2 data/generated/test-v2-b.jsonl -testv2-b-e2e
  e2e_run homeai-2b-lora-b-v2 data/generated/wild-b.jsonl -wild-b-e2e
  e2e_run homeai-2b-lora-b-v2 data/generated/wild2-b.jsonl -wild2-b-e2e
fi
# ---- LFM 1.2B B-v2 (2 Oct): the fast half of a design-B E pair (lapwing logs: LFM -> 4B B-v2 80.7% final, first action
# 1.51 s on lapwing's 3080); timed here so the pair's first-action time comes from one GPU
if has_model homeai-lfm25-1.2b-lora-b-v2; then
  while pgrep -f "outputs/spark-rebuil[d]" >/dev/null; do sleep 60; done
  wait_for_train; unload_all
  e2e_run homeai-lfm25-1.2b-lora-b-v2 data/generated/bigbench-b.jsonl -bigbench-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v2 data/generated/test-v2-b.jsonl -testv2-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v2 data/generated/wild-b.jsonl -wild-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v2 data/generated/wild2-b.jsonl -wild2-b-e2e
fi
# ---- v12 retry (2 Oct): its 06:03 merge collided with the Spark rebuild's export (both start when training ends) and
# failed; the adapter is fine, so merge + import now and run its evals before anything else
if [ -f configs/qwen35-4b-lora-v12.json ] && ! has_model homeai-4b-lora-v12; then
  wait_for_train; unload_all
  export_run configs/qwen35-4b-lora-v12.json homeai-4b-lora-v12 qwen3.5:4b
  fill_missing; python3 -m homeai.eval.champions > /dev/null; prune_if_short
fi
# ---- Spark B-v2 (2 Oct): design B on the fastest 4B (Spark v9 rebuilt: 1.08 s/request); served by ollama-next like the
# other Spark models, timed on the four B sets against homeai-4b-lora-b-v2 (81.5% mean, 2.09 s/request)
if [ -f configs/spark-x25-4b-lora-b-v2.json ]; then
  if train_run configs/spark-x25-4b-lora-b-v2.json; then
    [ -f outputs/spark-x25-4b-lora-b-v2/.pruned ] || has_model homeai-spark-x25-4b-lora-b-v2 || { note "export homeai-spark-x25-4b-lora-b-v2: starting"; scripts/export-spark.sh configs/spark-x25-4b-lora-b-v2.json homeai-spark-x25-4b-lora-b-v2 >> outputs/overnight.log 2>&1 </dev/null && note "export homeai-spark-x25-4b-lora-b-v2: done" || note "export homeai-spark-x25-4b-lora-b-v2: FAILED"; }
    if has_model homeai-spark-x25-4b-lora-b-v2; then
      wait_for_train; unload_all
      e2e_run homeai-spark-x25-4b-lora-b-v2 data/generated/bigbench-b.jsonl -bigbench-b-e2e
      e2e_run homeai-spark-x25-4b-lora-b-v2 data/generated/test-v2-b.jsonl -testv2-b-e2e
      e2e_run homeai-spark-x25-4b-lora-b-v2 data/generated/wild-b.jsonl -wild-b-e2e
      e2e_run homeai-spark-x25-4b-lora-b-v2 data/generated/wild2-b.jsonl -wild2-b-e2e
    fi
  fi
  prune_if_short
fi
# ---- 4B B-v2u (2 Oct): B-v2 in the user_context layout, the Mac's more accurate design-B option (the 2B with this layout
# tied the stable_prefix 2B and answered warm on the Bundook plan in 1.77 s on falcon); timed here on the B sets
if [ -f configs/qwen35-4b-lora-b-v2u.json ]; then
  train_run configs/qwen35-4b-lora-b-v2u.json && export_run configs/qwen35-4b-lora-b-v2u.json homeai-4b-lora-b-v2u qwen3.5:4b && {
    wait_for_train; unload_all
    e2e_run homeai-4b-lora-b-v2u data/generated/bigbench-b.jsonl -bigbench-b-e2e
    e2e_run homeai-4b-lora-b-v2u data/generated/test-v2-b.jsonl -testv2-b-e2e
    e2e_run homeai-4b-lora-b-v2u data/generated/wild-b.jsonl -wild-b-e2e
    e2e_run homeai-4b-lora-b-v2u data/generated/wild2-b.jsonl -wild2-b-e2e
  }
  prune_if_short
fi
# ---- Spark B-v2 heard on the Bundook plan (2 Oct): the new B leader (82.5% mean, 85.3% large houses, 1.05 s/request) needs
# its voice numbers beside B-v2's (lapwing); Spark needs Ollama >= 0.34, so it runs here on ollama-next (accuracy only)
if has_model homeai-spark-x25-4b-lora-b-v2 && [ ! -f outputs/eval/homeai-spark-x25-4b-lora-b-v2-bigbench-b-bundook-heard-written/report.json ]; then
  wait_for_train; unload_all
  note "heard homeai-spark-x25-4b-lora-b-v2 bigbench-b-bundook: starting"
  env OLLAMA=http://localhost:11435 HOMEAI_PROMPT_LAYOUT=stable_prefix MODELS=homeai-spark-x25-4b-lora-b-v2 SETS="bigbench-b-bundook" \
    scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard homeai-spark-x25-4b-lora-b-v2 bigbench-b-bundook: done" \
    || note "heard homeai-spark-x25-4b-lora-b-v2 bigbench-b-bundook: FAILED"
  docker exec ollama-next ollama stop homeai-spark-x25-4b-lora-b-v2 >/dev/null 2>&1
fi
# ---- Spark v9 heard on the everyday sets (3 Oct): the rebuilt Spark v9 (93.7% testv2 outcome, 1.08 s/request) has no voice
# numbers beside v9's; only ollama-next can serve Spark (accuracy only)
if has_model homeai-spark-x25-4b-lora-v9 && [ ! -f outputs/eval/homeai-spark-x25-4b-lora-v9-confusions-heard/report.json ]; then
  wait_for_train; unload_all
  note "heard homeai-spark-x25-4b-lora-v9 (everyday sets): starting"
  env OLLAMA=http://localhost:11435 HOMEAI_PROMPT_LAYOUT=classic MODELS=homeai-spark-x25-4b-lora-v9 SETS="testv2 wild wild2" \
    scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard homeai-spark-x25-4b-lora-v9: done" \
    || note "heard homeai-spark-x25-4b-lora-v9: FAILED"
  docker exec ollama-next ollama stop homeai-spark-x25-4b-lora-v9 >/dev/null 2>&1
fi
# ---- B-v3h (3 Oct): design B, house-specific: B-v2's data + 1,650 Bundook-plan conversations (scripts/build-b-v3h.sh),
# the 4B B-v2u recipe (user_context). Timed on the B sets (random houses = generalisation) and run on the Bundook heard set
# (the house itself) beside B-v2 / Spark B-v2
if [ -f configs/qwen35-4b-lora-b-v3h.json ] && [ -f data/generated/train-b-v3h.jsonl ]; then
  train_run configs/qwen35-4b-lora-b-v3h.json && export_run configs/qwen35-4b-lora-b-v3h.json homeai-4b-lora-b-v3h qwen3.5:4b && {
    wait_for_train; unload_all
    e2e_run homeai-4b-lora-b-v3h data/generated/bigbench-b.jsonl -bigbench-b-e2e
    e2e_run homeai-4b-lora-b-v3h data/generated/test-v2-b.jsonl -testv2-b-e2e
    e2e_run homeai-4b-lora-b-v3h data/generated/wild-b.jsonl -wild-b-e2e
    e2e_run homeai-4b-lora-b-v3h data/generated/wild2-b.jsonl -wild2-b-e2e
    if [ ! -f outputs/eval/homeai-4b-lora-b-v3h-confusions-heard/report.json ]; then
      note "heard homeai-4b-lora-b-v3h bigbench-b-bundook: starting"
      env OLLAMA=http://localhost:11434 HOMEAI_PROMPT_LAYOUT=user_context MODELS=homeai-4b-lora-b-v3h SETS="bigbench-b-bundook" \
        scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard homeai-4b-lora-b-v3h: done" || note "heard homeai-4b-lora-b-v3h: FAILED"
      docker exec ollama ollama stop homeai-4b-lora-b-v3h >/dev/null 2>&1
    fi
  }
  prune_if_short
fi
# ---- 2B and LFM B-v2u timed here (3 Oct): the cache-friendly design-B family (lapwing-trained, imported 3 Oct) on the 3090,
# so the LFM -> 4B B-v2u fast + slow pair and the 2B's speed come from one GPU (outputs/reconcile-b via reconcile_grid --sets b)
for m in homeai-2b-lora-b-v2u homeai-lfm25-1.2b-lora-b-v2u; do
  has_model $m || continue
  wait_for_train; unload_all
  e2e_run $m data/generated/bigbench-b.jsonl -bigbench-b-e2e
  e2e_run $m data/generated/test-v2-b.jsonl -testv2-b-e2e
  e2e_run $m data/generated/wild-b.jsonl -wild-b-e2e
  e2e_run $m data/generated/wild2-b.jsonl -wild2-b-e2e
done
# ---- B-v4w (3 Oct): design-B transcripts of the 1,174 wild training conversations by rejection sampling with the 4B B-v2u
# (homeai/train/rft_b.py), added to B-v2's data; B had no wild-style training data and is weakest on wild-b / wild2-b
if [ -f configs/qwen35-4b-lora-b-v4w.json ]; then
  if [ ! -f data/generated/train-b-v4w.jsonl ] && has_model homeai-4b-lora-b-v2u; then
    wait_for_train; unload_all
    note "rft-b wild sample (homeai-4b-lora-b-v2u, k=4): starting"
    python3 -m homeai.train.rft_b sample --model homeai-4b-lora-b-v2u --layout user_context --out outputs/rft-b/wild --k 4 >> outputs/overnight.log 2>&1 </dev/null \
      && python3 -m homeai.train.rft_b select --run outputs/rft-b/wild --out data/generated/train-b-wild.jsonl >> outputs/overnight.log 2>&1 \
      && cat data/generated/train-b-v2.jsonl data/generated/train-b-wild.jsonl > data/generated/train-b-v4w.jsonl.tmp \
      && python3 -m homeai.datagen.leakage data/generated/train-b-v4w.jsonl.tmp --json data/generated/train-b-v4w.leakage.json >> outputs/overnight.log 2>&1 \
      && python3 -c "import json,sys; d=json.load(open('data/generated/train-b-v4w.leakage.json')); d=d[0] if isinstance(d,list) else d; sys.exit(1 if d['augmented_overlap_turns'] else 0)" \
      && mv data/generated/train-b-v4w.jsonl.tmp data/generated/train-b-v4w.jsonl \
      && note "rft-b wild: $(wc -l < data/generated/train-b-wild.jsonl) rows -> train-b-v4w ($(wc -l < data/generated/train-b-v4w.jsonl))" \
      || note "rft-b wild: FAILED"
    docker exec ollama ollama stop homeai-4b-lora-b-v2u >/dev/null 2>&1
  fi
  if [ -f data/generated/train-b-v4w.jsonl ]; then
    train_run configs/qwen35-4b-lora-b-v4w.json && export_run configs/qwen35-4b-lora-b-v4w.json homeai-4b-lora-b-v4w qwen3.5:4b && {
      wait_for_train; unload_all
      e2e_run homeai-4b-lora-b-v4w data/generated/bigbench-b.jsonl -bigbench-b-e2e
      e2e_run homeai-4b-lora-b-v4w data/generated/test-v2-b.jsonl -testv2-b-e2e
      e2e_run homeai-4b-lora-b-v4w data/generated/wild-b.jsonl -wild-b-e2e
      e2e_run homeai-4b-lora-b-v4w data/generated/wild2-b.jsonl -wild2-b-e2e
      if [ ! -f outputs/eval/homeai-4b-lora-b-v4w-confusions-heard/report.json ]; then
        note "heard homeai-4b-lora-b-v4w bigbench-b-bundook: starting"
        env OLLAMA=http://localhost:11434 HOMEAI_PROMPT_LAYOUT=user_context MODELS=homeai-4b-lora-b-v4w SETS="bigbench-b-bundook" \
          scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard homeai-4b-lora-b-v4w: done" || note "heard homeai-4b-lora-b-v4w: FAILED"
        docker exec ollama ollama stop homeai-4b-lora-b-v4w >/dev/null 2>&1
      fi
    }
    prune_if_short
  fi
fi
# ---- Spark B-v4w (4 Oct): Spark on B-v4w's data (the wild transcripts lifted the 4B Qwen from 82.5 to 83.9%); Spark B-v2
# tied 82.5% at 1.05 s/request, so this aims at the GPU box's best and fastest design-B model. ollama-next, like the Spark models
if [ -f configs/spark-x25-4b-lora-b-v4w.json ] && [ -f data/generated/train-b-v4w.jsonl ]; then
  if train_run configs/spark-x25-4b-lora-b-v4w.json; then
    [ -f outputs/spark-x25-4b-lora-b-v4w/.pruned ] || has_model homeai-spark-x25-4b-lora-b-v4w || { note "export homeai-spark-x25-4b-lora-b-v4w: starting"; scripts/export-spark.sh configs/spark-x25-4b-lora-b-v4w.json homeai-spark-x25-4b-lora-b-v4w >> outputs/overnight.log 2>&1 </dev/null && note "export homeai-spark-x25-4b-lora-b-v4w: done" || note "export homeai-spark-x25-4b-lora-b-v4w: FAILED"; }
    if has_model homeai-spark-x25-4b-lora-b-v4w; then
      cp -n configs/spark-x25-4b-lora-b-v4w.json outputs/spark-x25-4b-lora-b-v4w/config.used.json 2>/dev/null
      [ -f outputs/spark-x25-4b-lora-b-v4w/.archived ] || { scripts/archive.sh push outputs/spark-x25-4b-lora-b-v4w >> "$STATUS" 2>&1 && scripts/archive.sh drop-gguf outputs/spark-x25-4b-lora-b-v4w >> "$STATUS" 2>&1; }
      wait_for_train; unload_all
      e2e_run homeai-spark-x25-4b-lora-b-v4w data/generated/bigbench-b.jsonl -bigbench-b-e2e
      e2e_run homeai-spark-x25-4b-lora-b-v4w data/generated/test-v2-b.jsonl -testv2-b-e2e
      e2e_run homeai-spark-x25-4b-lora-b-v4w data/generated/wild-b.jsonl -wild-b-e2e
      e2e_run homeai-spark-x25-4b-lora-b-v4w data/generated/wild2-b.jsonl -wild2-b-e2e
      if [ ! -f outputs/eval/homeai-spark-x25-4b-lora-b-v4w-confusions-heard/report.json ]; then
        note "heard homeai-spark-x25-4b-lora-b-v4w bigbench-b-bundook: starting"
        env OLLAMA=http://localhost:11435 HOMEAI_PROMPT_LAYOUT=stable_prefix MODELS=homeai-spark-x25-4b-lora-b-v4w SETS="bigbench-b-bundook" \
          scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard homeai-spark-x25-4b-lora-b-v4w: done" || note "heard homeai-spark-x25-4b-lora-b-v4w: FAILED"
        docker exec ollama-next ollama stop homeai-spark-x25-4b-lora-b-v4w >/dev/null 2>&1
      fi
    fi
  fi
  prune_if_short
fi
# ---- design B on the new house, larger (4 Oct): bigbench-b-bundook200 (200 conversations on the full Bundook plan, seeds 9.5M,
# held out) for every design-B GPU candidate; the 50-conversation set (~114 turns) was too small to separate them (+-4 points)
for m in homeai-spark-x25-4b-lora-b-v2 homeai-4b-lora-b-v4w homeai-4b-lora-b-v2u homeai-spark-x25-4b-lora-b-v4w homeai-4b-lora-b-v3h \
         homeai-4b-lora-b-v2 homeai-2b-lora-b-v2u homeai-lfm25-1.2b-lora-b-v2u; do
  [ -f data/generated/bigbench-b-bundook200.jsonl ] && has_model $m || continue
  wait_for_train; unload_all
  e2e_run $m data/generated/bigbench-b-bundook200.jsonl -bundook200-b-e2e
done
# ---- the same set heard through speech-to-text (4 Oct, scripts/heard-bundook200.sh): the four GPU design-B candidates
for ml in homeai-spark-x25-4b-lora-b-v4w:stable_prefix homeai-spark-x25-4b-lora-b-v2:stable_prefix homeai-4b-lora-b-v4w:user_context homeai-4b-lora-b-v2u:user_context; do
  m=${ml%%:*}; lay=${ml##*:}
  [ -f data/generated/heard/bigbench-b-bundook200-parakeet-clean.jsonl ] && has_model $m || continue
  [ -f outputs/eval/$m-bigbench-b-bundook200-heard-tiny-clean/report.json ] && continue
  wait_for_train; unload_all
  note "heard $m bundook200: starting"
  env OLLAMA="$(model_url $m)" HOMEAI_PROMPT_LAYOUT=$lay MODELS=$m SETS="bigbench-b-bundook200" \
    scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard $m bundook200: done" || note "heard $m bundook200: FAILED"
  docker exec "$(model_ctr $m)" ollama stop $m >/dev/null 2>&1
done
# ---- LFM B-v3h timed here (4 Oct): the Mac's fastest pick, and the fast half of a GPU pair with Spark B-v2 (the house's
# best heard); its logs on the B sets and the larger Bundook set pair with Spark B-v2's (reconcile_grid --sets b)
if has_model homeai-lfm25-1.2b-lora-b-v3h; then
  wait_for_train; unload_all
  e2e_run homeai-lfm25-1.2b-lora-b-v3h data/generated/bigbench-b.jsonl -bigbench-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v3h data/generated/test-v2-b.jsonl -testv2-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v3h data/generated/wild-b.jsonl -wild-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v3h data/generated/wild2-b.jsonl -wild2-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v3h data/generated/bigbench-b-bundook200.jsonl -bundook200-b-e2e
fi
# ---- Spark B-v6n (4 Oct): Spark B-v2's recipe on B's data with the hard (far-field / TV noise) speech-to-text augmentation
# (scripts/build-b-v6n.sh); judged on the 200-conversation Bundook set heard, where Spark B-v2 falls to 69% at 3 m and 67% in noise
if [ -f configs/spark-x25-4b-lora-b-v6n.json ] && [ -f data/generated/train-b-v6n.jsonl ]; then
  if train_run configs/spark-x25-4b-lora-b-v6n.json; then
    m=homeai-spark-x25-4b-lora-b-v6n; o=outputs/spark-x25-4b-lora-b-v6n
    [ -f $o/.pruned ] || has_model $m || { note "export $m: starting"; scripts/export-spark.sh configs/spark-x25-4b-lora-b-v6n.json $m >> outputs/overnight.log 2>&1 </dev/null && note "export $m: done" || note "export $m: FAILED"; }
    if has_model $m; then
      cp -n configs/spark-x25-4b-lora-b-v6n.json $o/config.used.json 2>/dev/null
      sleep 30   # 4 Oct: an archive push right after export-spark failed its checksum once; let the GGUF settle
      [ -f $o/.archived ] || { scripts/archive.sh push $o >> "$STATUS" 2>&1 || scripts/archive.sh push $o >> "$STATUS" 2>&1; } && scripts/archive.sh drop-gguf $o >> "$STATUS" 2>&1
      wait_for_train; unload_all
      e2e_run $m data/generated/bigbench-b.jsonl -bigbench-b-e2e
      e2e_run $m data/generated/test-v2-b.jsonl -testv2-b-e2e
      e2e_run $m data/generated/wild-b.jsonl -wild-b-e2e
      e2e_run $m data/generated/wild2-b.jsonl -wild2-b-e2e
      e2e_run $m data/generated/bigbench-b-bundook200.jsonl -bundook200-b-e2e
      if [ ! -f outputs/eval/$m-bigbench-b-bundook200-heard-tiny-clean/report.json ]; then
        note "heard $m bundook200: starting"
        env OLLAMA=http://localhost:11435 HOMEAI_PROMPT_LAYOUT=stable_prefix MODELS=$m SETS="bigbench-b-bundook200" \
          scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard $m bundook200: done" || note "heard $m bundook200: FAILED"
        docker exec ollama-next ollama stop $m >/dev/null 2>&1
      fi
    fi
  fi
  prune_if_short
fi
# ---- LFM B-v7 timed here (4 Oct, trained on lapwing): B-v3h plus the hard speech-to-text augmentation; on the
# 200-conversation house set heard it gained 4.3 points at 3 m and 2.6 in noise over LFM B-v3h (lapwing). Its logs pair with
# Spark B-v2's and Spark B-v6n's (reconcile_grid --sets b)
if has_model homeai-lfm25-1.2b-lora-b-v7; then
  wait_for_train; unload_all
  e2e_run homeai-lfm25-1.2b-lora-b-v7 data/generated/bigbench-b.jsonl -bigbench-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v7 data/generated/test-v2-b.jsonl -testv2-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v7 data/generated/wild-b.jsonl -wild-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v7 data/generated/wild2-b.jsonl -wild2-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v7 data/generated/bigbench-b-bundook200.jsonl -bundook200-b-e2e
fi
# ---- Spark B-v9 (5 Oct): Spark B-v6n's data (hard far-field / TV-noise augmentation) plus the 4B's 993 verified wild
# transcripts (B-v4w's addition, +3-4 on Spark's wild sets); B-v6n is the best design-B model typed (84.3% B-set mean)
if [ -f configs/spark-x25-4b-lora-b-v9.json ] && [ -f data/generated/train-b-v9.jsonl ]; then
  if train_run configs/spark-x25-4b-lora-b-v9.json; then
    m=homeai-spark-x25-4b-lora-b-v9; o=outputs/spark-x25-4b-lora-b-v9
    [ -f $o/.pruned ] || has_model $m || { note "export $m: starting"; scripts/export-spark.sh configs/spark-x25-4b-lora-b-v9.json $m >> outputs/overnight.log 2>&1 </dev/null && note "export $m: done" || note "export $m: FAILED"; }
    if has_model $m; then
      cp -n configs/spark-x25-4b-lora-b-v9.json $o/config.used.json 2>/dev/null
      sleep 30
      [ -f $o/.archived ] || { scripts/archive.sh push $o >> "$STATUS" 2>&1 || scripts/archive.sh push $o >> "$STATUS" 2>&1; } && scripts/archive.sh drop-gguf $o >> "$STATUS" 2>&1
      [ -f $o/.archived ] && [ -d $o/merged16 ] && rm -rf $o/merged16   # 5 Oct: rebuildable, and the evals below run for hours near the disk reserve
      wait_for_train; unload_all
      e2e_run $m data/generated/bigbench-b.jsonl -bigbench-b-e2e
      e2e_run $m data/generated/test-v2-b.jsonl -testv2-b-e2e
      e2e_run $m data/generated/wild-b.jsonl -wild-b-e2e
      e2e_run $m data/generated/wild2-b.jsonl -wild2-b-e2e
      e2e_run $m data/generated/bigbench-b-bundook200.jsonl -bundook200-b-e2e
      if [ ! -f outputs/eval/$m-bigbench-b-bundook200-heard-tiny-clean/report.json ]; then
        note "heard $m bundook200: starting"
        env OLLAMA=http://localhost:11435 HOMEAI_PROMPT_LAYOUT=stable_prefix MODELS=$m SETS="bigbench-b-bundook200" \
          scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard $m bundook200: done" || note "heard $m bundook200: FAILED"
        docker exec ollama-next ollama stop $m >/dev/null 2>&1
      fi
    fi
  fi
  prune_if_short
fi
# ---- LFM B-v7 at LoRA rank 32 timed here (5 Oct, trained on lapwing): the Mac's speed pick (lapwing: 79.2% B-set mean,
# 84.2% typed on the 200-conversation house set, heard mean 75.4%); its logs pair with Spark B-v6n's (fast acts, slow corrects)
if has_model homeai-lfm25-1.2b-lora-b-v7r32; then
  wait_for_train; unload_all
  e2e_run homeai-lfm25-1.2b-lora-b-v7r32 data/generated/bigbench-b.jsonl -bigbench-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v7r32 data/generated/test-v2-b.jsonl -testv2-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v7r32 data/generated/wild-b.jsonl -wild-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v7r32 data/generated/wild2-b.jsonl -wild2-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v7r32 data/generated/bigbench-b-bundook200.jsonl -bundook200-b-e2e
fi
# ---- Spark B-v6n at LoRA rank 32 (5 Oct): r32 beat r16 on LFM and the 2B at the same speed; same data and recipe as the
# house's GPU pick otherwise (paged 8-bit Adam keeps the extra memory to ~0.4 GB beside Frigate)
if [ -f configs/spark-x25-4b-lora-b-v6n-r32.json ]; then
  if train_run configs/spark-x25-4b-lora-b-v6n-r32.json; then
    m=homeai-spark-x25-4b-lora-b-v6n-r32; o=outputs/spark-x25-4b-lora-b-v6n-r32
    [ -f $o/.pruned ] || has_model $m || { note "export $m: starting"; scripts/export-spark.sh configs/spark-x25-4b-lora-b-v6n-r32.json $m >> outputs/overnight.log 2>&1 </dev/null && note "export $m: done" || note "export $m: FAILED"; }
    if has_model $m; then
      cp -n configs/spark-x25-4b-lora-b-v6n-r32.json $o/config.used.json 2>/dev/null
      sleep 30
      [ -f $o/.archived ] || { scripts/archive.sh push $o >> "$STATUS" 2>&1 || scripts/archive.sh push $o >> "$STATUS" 2>&1; } && scripts/archive.sh drop-gguf $o >> "$STATUS" 2>&1
      [ -f $o/.archived ] && [ -d $o/merged16 ] && rm -rf $o/merged16   # rebuildable; the evals run for hours
      wait_for_train; unload_all
      e2e_run $m data/generated/bigbench-b.jsonl -bigbench-b-e2e
      e2e_run $m data/generated/test-v2-b.jsonl -testv2-b-e2e
      e2e_run $m data/generated/wild-b.jsonl -wild-b-e2e
      e2e_run $m data/generated/wild2-b.jsonl -wild2-b-e2e
      e2e_run $m data/generated/bigbench-b-bundook200.jsonl -bundook200-b-e2e
      if [ ! -f outputs/eval/$m-bigbench-b-bundook200-heard-tiny-clean/report.json ]; then
        note "heard $m bundook200: starting"
        env OLLAMA=http://localhost:11435 HOMEAI_PROMPT_LAYOUT=stable_prefix MODELS=$m SETS="bigbench-b-bundook200" \
          scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard $m bundook200: done" || note "heard $m bundook200: FAILED"
        docker exec ollama-next ollama stop $m >/dev/null 2>&1
      fi
    fi
  fi
  prune_if_short
fi
# ---- LFM B-v7 at LoRA rank 128 timed here (6 Oct, trained on lapwing): level with Spark B-v6n heard on the 200-conversation
# house set (77.9% both) and ahead typed (86.0 vs 85.5%) on lapwing; its cassowary logs pair with Spark's
if has_model homeai-lfm25-1.2b-lora-b-v7r128; then
  wait_for_train; unload_all
  e2e_run homeai-lfm25-1.2b-lora-b-v7r128 data/generated/bigbench-b.jsonl -bigbench-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v7r128 data/generated/test-v2-b.jsonl -testv2-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v7r128 data/generated/wild-b.jsonl -wild-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v7r128 data/generated/wild2-b.jsonl -wild2-b-e2e
  e2e_run homeai-lfm25-1.2b-lora-b-v7r128 data/generated/bigbench-b-bundook200.jsonl -bundook200-b-e2e
fi
# ---- LFM B-v7 r128 heard on the 200-conversation house set here (6 Oct): its cassowary logs per speech-to-text condition
# pair with Spark B-v6n r32's (fast acts, slow corrects) for the GPU decision on spoken requests
if has_model homeai-lfm25-1.2b-lora-b-v7r128 && [ ! -f outputs/eval/homeai-lfm25-1.2b-lora-b-v7r128-bigbench-b-bundook200-heard-tiny-clean/report.json ]; then
  wait_for_train; unload_all
  note "heard homeai-lfm25-1.2b-lora-b-v7r128 bundook200: starting"
  env OLLAMA=http://localhost:11434 HOMEAI_PROMPT_LAYOUT=user_context MODELS=homeai-lfm25-1.2b-lora-b-v7r128 SETS="bigbench-b-bundook200" \
    scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard homeai-lfm25-1.2b-lora-b-v7r128 bundook200: done" || note "heard homeai-lfm25-1.2b-lora-b-v7r128 bundook200: FAILED"
  docker exec ollama ollama stop homeai-lfm25-1.2b-lora-b-v7r128 >/dev/null 2>&1
fi
# ---- Spark B-v9 at LoRA rank 32 (6 Oct): B-v9's data (hard speech augmentation + wild transcripts, best typed at r16) with rank 32
# (rank 32 lifted Spark B-v6n heard 77.9 -> 78.5% and typed 85.5 -> 86.4%)
if [ -f configs/spark-x25-4b-lora-b-v9-r32.json ]; then
  if train_run configs/spark-x25-4b-lora-b-v9-r32.json; then
    m=homeai-spark-x25-4b-lora-b-v9-r32; o=outputs/spark-x25-4b-lora-b-v9-r32
    [ -f $o/.pruned ] || has_model $m || { note "export $m: starting"; scripts/export-spark.sh configs/spark-x25-4b-lora-b-v9-r32.json $m >> outputs/overnight.log 2>&1 </dev/null && note "export $m: done" || note "export $m: FAILED"; }
    if has_model $m; then
      cp -n configs/spark-x25-4b-lora-b-v9-r32.json $o/config.used.json 2>/dev/null
      sleep 30
      [ -f $o/.archived ] || { scripts/archive.sh push $o >> "$STATUS" 2>&1 || scripts/archive.sh push $o >> "$STATUS" 2>&1; } && scripts/archive.sh drop-gguf $o >> "$STATUS" 2>&1
      [ -f $o/.archived ] && [ -d $o/merged16 ] && rm -rf $o/merged16   # rebuildable; the evals run for hours
      wait_for_train; unload_all
      e2e_run $m data/generated/bigbench-b.jsonl -bigbench-b-e2e
      e2e_run $m data/generated/test-v2-b.jsonl -testv2-b-e2e
      e2e_run $m data/generated/wild-b.jsonl -wild-b-e2e
      e2e_run $m data/generated/wild2-b.jsonl -wild2-b-e2e
      e2e_run $m data/generated/bigbench-b-bundook200.jsonl -bundook200-b-e2e
      if [ ! -f outputs/eval/$m-bigbench-b-bundook200-heard-tiny-clean/report.json ]; then
        note "heard $m bundook200: starting"
        env OLLAMA=http://localhost:11435 HOMEAI_PROMPT_LAYOUT=stable_prefix MODELS=$m SETS="bigbench-b-bundook200" \
          scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard $m bundook200: done" || note "heard $m bundook200: FAILED"
        docker exec ollama-next ollama stop $m >/dev/null 2>&1
      fi
    fi
  fi
  prune_if_short
fi
# ---- Spark B-v6n r32, second seed (6 Oct): the GPU pick retrained with seed 1234, to measure Spark's run-to-run spread
# (heard runs include the names-biased medium Whisper far-field and noisy sets)
if [ -f configs/spark-x25-4b-lora-b-v6n-r32s2.json ]; then
  if train_run configs/spark-x25-4b-lora-b-v6n-r32s2.json; then
    m=homeai-spark-x25-4b-lora-b-v6n-r32s2; o=outputs/spark-x25-4b-lora-b-v6n-r32s2
    [ -f $o/.pruned ] || has_model $m || { note "export $m: starting"; scripts/export-spark.sh configs/spark-x25-4b-lora-b-v6n-r32s2.json $m >> outputs/overnight.log 2>&1 </dev/null && note "export $m: done" || note "export $m: FAILED"; }
    if has_model $m; then
      cp -n configs/spark-x25-4b-lora-b-v6n-r32s2.json $o/config.used.json 2>/dev/null
      sleep 30
      [ -f $o/.archived ] || { scripts/archive.sh push $o >> "$STATUS" 2>&1 || scripts/archive.sh push $o >> "$STATUS" 2>&1; } && scripts/archive.sh drop-gguf $o >> "$STATUS" 2>&1
      [ -f $o/.archived ] && [ -d $o/merged16 ] && rm -rf $o/merged16   # rebuildable; the evals run for hours
      wait_for_train; unload_all
      e2e_run $m data/generated/bigbench-b.jsonl -bigbench-b-e2e
      e2e_run $m data/generated/test-v2-b.jsonl -testv2-b-e2e
      e2e_run $m data/generated/wild-b.jsonl -wild-b-e2e
      e2e_run $m data/generated/wild2-b.jsonl -wild2-b-e2e
      e2e_run $m data/generated/bigbench-b-bundook200.jsonl -bundook200-b-e2e
      if [ ! -f outputs/eval/$m-bigbench-b-bundook200-heard-tiny-clean/report.json ]; then
        note "heard $m bundook200: starting"
        env OLLAMA=http://localhost:11435 HOMEAI_PROMPT_LAYOUT=stable_prefix MODELS=$m SETS="bigbench-b-bundook200" CONDS="written parakeet-clean parakeet-farfield parakeet-noisy10 small-clean small-biased medium-biased tiny-clean medium-farfield-biased medium-noisy10-biased" \
          scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard $m bundook200: done" || note "heard $m bundook200: FAILED"
        docker exec ollama-next ollama stop $m >/dev/null 2>&1
      fi
    fi
  fi
  prune_if_short
fi
# ---- LFM B-v8 r256 timed here (7 Oct, trained on lapwing): the default small model (two-seed means 82.7% B sets, 79.0% heard
# on the house); its cassowary timing on the B sets and the house set, and the medium Whisper names-biased heard sets
if has_model homeai-lfm25-1.2b-lora-b-v8r256; then
  m=homeai-lfm25-1.2b-lora-b-v8r256
  wait_for_train; unload_all
  e2e_run $m data/generated/bigbench-b.jsonl -bigbench-b-e2e
  e2e_run $m data/generated/test-v2-b.jsonl -testv2-b-e2e
  e2e_run $m data/generated/wild-b.jsonl -wild-b-e2e
  e2e_run $m data/generated/wild2-b.jsonl -wild2-b-e2e
  e2e_run $m data/generated/bigbench-b-bundook200.jsonl -bundook200-b-e2e
  if [ ! -f outputs/eval/$m-bigbench-b-bundook200-heard-medium-noisy10-biased/report.json ]; then
    note "heard $m bundook200 (medium names-biased): starting"
    env OLLAMA=http://localhost:11434 HOMEAI_PROMPT_LAYOUT=user_context MODELS=$m SETS="bigbench-b-bundook200" CONDS="written medium-biased medium-farfield-biased medium-noisy10-biased" \
      scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard $m bundook200 (medium names-biased): done" || note "heard $m bundook200: FAILED"
    docker exec ollama ollama stop $m >/dev/null 2>&1
  fi
fi
# ---- Spark B-v7 r32 (8 Oct): the GPU pick's recipe (B-v6n r32) plus the 1,650 cleaned Bundook-plan conversations (train-b-v7)
# (heard runs include the medium Whisper names-biased far-field and noisy sets)
if [ -f configs/spark-x25-4b-lora-b-v7-r32.json ]; then
  if train_run configs/spark-x25-4b-lora-b-v7-r32.json; then
    m=homeai-spark-x25-4b-lora-b-v7-r32; o=outputs/spark-x25-4b-lora-b-v7-r32
    [ -f $o/.pruned ] || has_model $m || { note "export $m: starting"; scripts/export-spark.sh configs/spark-x25-4b-lora-b-v7-r32.json $m >> outputs/overnight.log 2>&1 </dev/null && note "export $m: done" || note "export $m: FAILED"; }
    if has_model $m; then
      cp -n configs/spark-x25-4b-lora-b-v7-r32.json $o/config.used.json 2>/dev/null
      sleep 30
      [ -f $o/.archived ] || { scripts/archive.sh push $o >> "$STATUS" 2>&1 || scripts/archive.sh push $o >> "$STATUS" 2>&1; } && scripts/archive.sh drop-gguf $o >> "$STATUS" 2>&1
      [ -f $o/.archived ] && [ -d $o/merged16 ] && rm -rf $o/merged16   # rebuildable; the evals run for hours
      wait_for_train; unload_all
      e2e_run $m data/generated/bigbench-b.jsonl -bigbench-b-e2e
      e2e_run $m data/generated/test-v2-b.jsonl -testv2-b-e2e
      e2e_run $m data/generated/wild-b.jsonl -wild-b-e2e
      e2e_run $m data/generated/wild2-b.jsonl -wild2-b-e2e
      e2e_run $m data/generated/bigbench-b-bundook200.jsonl -bundook200-b-e2e
      if [ ! -f outputs/eval/$m-bigbench-b-bundook200-heard-tiny-clean/report.json ]; then
        note "heard $m bundook200: starting"
        env OLLAMA=http://localhost:11435 HOMEAI_PROMPT_LAYOUT=stable_prefix MODELS=$m SETS="bigbench-b-bundook200" CONDS="written parakeet-clean parakeet-farfield parakeet-noisy10 small-clean small-biased medium-biased tiny-clean medium-farfield-biased medium-noisy10-biased" \
          scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard $m bundook200: done" || note "heard $m bundook200: FAILED"
        docker exec ollama-next ollama stop $m >/dev/null 2>&1
      fi
    fi
  fi
  prune_if_short
fi
# ---- Spark B-v7 r32, second seed (8 Oct): seed 1234; seed 3407 gave the best house results yet (89.0% typed, 80.5% heard)
# (heard runs include the medium Whisper names-biased far-field and noisy sets)
if [ -f configs/spark-x25-4b-lora-b-v7-r32s2.json ]; then
  if train_run configs/spark-x25-4b-lora-b-v7-r32s2.json; then
    m=homeai-spark-x25-4b-lora-b-v7-r32s2; o=outputs/spark-x25-4b-lora-b-v7-r32s2
    [ -f $o/.pruned ] || has_model $m || { note "export $m: starting"; scripts/export-spark.sh configs/spark-x25-4b-lora-b-v7-r32s2.json $m >> outputs/overnight.log 2>&1 </dev/null && note "export $m: done" || note "export $m: FAILED"; }
    if has_model $m; then
      cp -n configs/spark-x25-4b-lora-b-v7-r32s2.json $o/config.used.json 2>/dev/null
      sleep 30
      [ -f $o/.archived ] || { scripts/archive.sh push $o >> "$STATUS" 2>&1 || scripts/archive.sh push $o >> "$STATUS" 2>&1; } && scripts/archive.sh drop-gguf $o >> "$STATUS" 2>&1
      [ -f $o/.archived ] && [ -d $o/merged16 ] && rm -rf $o/merged16   # rebuildable; the evals run for hours
      wait_for_train; unload_all
      e2e_run $m data/generated/bigbench-b.jsonl -bigbench-b-e2e
      e2e_run $m data/generated/test-v2-b.jsonl -testv2-b-e2e
      e2e_run $m data/generated/wild-b.jsonl -wild-b-e2e
      e2e_run $m data/generated/wild2-b.jsonl -wild2-b-e2e
      e2e_run $m data/generated/bigbench-b-bundook200.jsonl -bundook200-b-e2e
      if [ ! -f outputs/eval/$m-bigbench-b-bundook200-heard-tiny-clean/report.json ]; then
        note "heard $m bundook200: starting"
        env OLLAMA=http://localhost:11435 HOMEAI_PROMPT_LAYOUT=stable_prefix MODELS=$m SETS="bigbench-b-bundook200" CONDS="written parakeet-clean parakeet-farfield parakeet-noisy10 small-clean small-biased medium-biased tiny-clean medium-farfield-biased medium-noisy10-biased" \
          scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard $m bundook200: done" || note "heard $m bundook200: FAILED"
        docker exec ollama-next ollama stop $m >/dev/null 2>&1
      fi
    fi
  fi
  prune_if_short
fi
# ---- Spark B-v7 r32, 3 epochs (9 Oct): extra epochs lifted LFM B-v8 r256 2 points heard on the house; same test for Spark
# (heard runs include the medium Whisper names-biased far-field and noisy sets)
if [ -f configs/spark-x25-4b-lora-b-v7-r32e3.json ]; then
  if train_run configs/spark-x25-4b-lora-b-v7-r32e3.json; then
    m=homeai-spark-x25-4b-lora-b-v7-r32e3; o=outputs/spark-x25-4b-lora-b-v7-r32e3
    [ -f $o/.pruned ] || has_model $m || { note "export $m: starting"; scripts/export-spark.sh configs/spark-x25-4b-lora-b-v7-r32e3.json $m >> outputs/overnight.log 2>&1 </dev/null && note "export $m: done" || note "export $m: FAILED"; }
    if has_model $m; then
      cp -n configs/spark-x25-4b-lora-b-v7-r32e3.json $o/config.used.json 2>/dev/null
      sleep 30
      [ -f $o/.archived ] || { scripts/archive.sh push $o >> "$STATUS" 2>&1 || scripts/archive.sh push $o >> "$STATUS" 2>&1; } && scripts/archive.sh drop-gguf $o >> "$STATUS" 2>&1
      [ -f $o/.archived ] && [ -d $o/merged16 ] && rm -rf $o/merged16   # rebuildable; the evals run for hours
      wait_for_train; unload_all
      e2e_run $m data/generated/bigbench-b.jsonl -bigbench-b-e2e
      e2e_run $m data/generated/test-v2-b.jsonl -testv2-b-e2e
      e2e_run $m data/generated/wild-b.jsonl -wild-b-e2e
      e2e_run $m data/generated/wild2-b.jsonl -wild2-b-e2e
      e2e_run $m data/generated/bigbench-b-bundook200.jsonl -bundook200-b-e2e
      if [ ! -f outputs/eval/$m-bigbench-b-bundook200-heard-tiny-clean/report.json ]; then
        note "heard $m bundook200: starting"
        env OLLAMA=http://localhost:11435 HOMEAI_PROMPT_LAYOUT=stable_prefix MODELS=$m SETS="bigbench-b-bundook200" CONDS="written parakeet-clean parakeet-farfield parakeet-noisy10 small-clean small-biased medium-biased tiny-clean medium-farfield-biased medium-noisy10-biased" \
          scripts/heard-evals.sh full >> outputs/overnight.log 2>&1 </dev/null && note "heard $m bundook200: done" || note "heard $m bundook200: FAILED"
        docker exec ollama-next ollama stop $m >/dev/null 2>&1
      fi
    fi
  fi
  prune_if_short
fi
# ---- design C timed on one box (Peter, 29 Sep: timings compared with earlier results come from cassowary): orchestrator
# (4B or 2B) and the 2B room leaf both served by this Ollama, idle GPU. Earlier routed runs split orchestrator and leaves
# across cassowary and lapwing.
if has_model homeai-leaf-qwen35-2b; then
  for orch in 4b 2b; do
    has_model homeai-$orch-lora-orch || continue
    for s in test-v2-b:testv2 wild-b:wild bigbench-b:bigbench; do d=${s%%:*}; t=${s##*:}; out=outputs/eval/tiered-routed-${orch}orch-2b-local-$t
      [ -f "$out/report.json" ] && continue
      wait_for_train; unload_all
      note "tiered ${orch}orch-2b-local $t: starting"
      python3 -m homeai.tiered.eval --data data/generated/$d.jsonl --ollama http://localhost:11434 --model homeai-$orch-lora-orch \
        --leaf homeai-leaf-qwen35-2b --leaf-url http://localhost:11434 --name tiered-routed-${orch}orch-2b-local --out "$out" >> outputs/overnight.log 2>&1 </dev/null \
        && note "tiered ${orch}orch-2b-local $t: done $(python3 -c "import json;r=json.load(open('$out/report.json'));l=r['latency']['turn_s'];print(f\"pass={r['pass']:.1%} turn mean={l['mean']:.2f}s p50={l['p50']:.2f}s\")")" \
        || note "tiered ${orch}orch-2b-local $t: FAILED"
    done
  done
  unload_all
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
