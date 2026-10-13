#!/usr/bin/env bash
# Build data/generated/train-b-v3h.jsonl (3 Oct): B-v2's training data plus a house-specific supplement, 1,650 design-B
# conversations on the Bundook plan (train_b --big-kinds bundook_full:0:0.7,bundook_projected:0:0.3, seeds 6.5M+,
# data/generated/train-b-bundook-sup.jsonl), given the same speech-to-text augmentation as B-v2 (scripts/build-b-v2.sh,
# homeai/datagen/asr_v9.py); only the supplement is round-tripped, B-v2's rows are reused as built. Then the leakage check
# and configs/qwen35-4b-lora-b-v3h.json (the 4B B-v2u recipe, user_context layout).
#   nohup scripts/build-b-v3h.sh >> outputs/build-b-v3h.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
W=data/generated/v9bh
export PYTHONPATH=. IMAGE=${IMAGE:-homeai-voice:latest}
python3 -m homeai.datagen.asr_v9 plan --src data/generated/train-b-bundook-sup.jsonl --work $W
VR=(scripts/voice-roundtrip.sh --voices train --seed-salt v9 --out $W/heard)
t0=$(date +%s)
for tts in kokoro piper; do
  sets=$(ls $W/rt/${tts}__*.jsonl 2>/dev/null | paste -sd, -)
  [ -n "$sets" ] && "${VR[@]}" --tts $tts --tts-only --tts-workers ${TTS_WORKERS:-5} --sets "$sets"
done
t1=$(date +%s)
for f in $W/rt/*.jsonl; do
  IFS=_ read -r tts _ engine _ cond <<< "$(basename "$f" .jsonl)"
  "${VR[@]}" --tts "$tts" --stt "$engine" --conditions "$cond" --sets "$f" 2>&1 | grep -v cpuid_info
done
t2=$(date +%s)
echo "round trip: tts $((t1 - t0)) s, stt $((t2 - t1)) s (host $(hostname -s))"
python3 -m homeai.datagen.asr_v9 build --work $W --out data/generated/train-b-sup-v2.jsonl --layout stable_prefix
cat data/generated/train-b-v2.jsonl data/generated/train-b-sup-v2.jsonl > data/generated/train-b-v3h.jsonl
python3 -m homeai.datagen.leakage data/generated/train-b-v3h.jsonl --json data/generated/train-b-v3h.leakage.json
python3 - <<'PY'
import json
c = json.load(open("configs/qwen35-4b-lora-b-v2u.json"))
n = sum(1 for _ in open("data/generated/train-b-v3h.jsonl"))
c["_comment"] = (f"Design B, house-specific (3 Oct): B-v2's data plus 1,650 conversations on the Bundook plan (the new house; seeds "
                 f"6.5M+, clear of every test set) with the same speech-to-text augmentation, {n} rows; the 4B B-v2u recipe and "
                 f"user_context layout (cacheable on the Mac, no cost on the GPU: 82.5% mean). The random-house B sets stay the "
                 f"generalisation check; bigbench-b-bundook and its heard sets measure the house itself.")
c["train"] = "data/generated/train-b-v3h.jsonl"
c["out"] = "outputs/qwen35-4b-lora-b-v3h"
json.dump(c, open("configs/qwen35-4b-lora-b-v3h.json", "w"), indent=1)
PY
echo "$(date '+%F %T') build-b-v3h done"
