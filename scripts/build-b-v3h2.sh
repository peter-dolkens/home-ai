#!/usr/bin/env bash
# Build data/generated/train-b-v3h2.jsonl (3 Oct): B-v3h (B-v2 + the first 1,650 Bundook-plan conversations) plus a second,
# independent 1,650 (train_b --seed 320 --big-seed 6600000) with the same speech-to-text augmentation: does more house data keep helping the small models?
# data/generated/train-b-bundook-sup2.jsonl), given the same speech-to-text augmentation as B-v2 (scripts/build-b-v2.sh,
# homeai/datagen/asr_v9.py); only the supplement is round-tripped, B-v2's rows are reused as built. Then the leakage check
# and configs/qwen35-4b-lora-b-v3h.json (the 4B B-v2u recipe, user_context layout).
#   nohup scripts/build-b-v3h.sh >> outputs/build-b-v3h.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
W=data/generated/v9bh2
export PYTHONPATH=. IMAGE=${IMAGE:-homeai-voice:latest}
python3 -m homeai.datagen.asr_v9 plan --src data/generated/train-b-bundook-sup2.jsonl --work $W
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
python3 -m homeai.datagen.asr_v9 build --work $W --out data/generated/train-b-sup2-v2.jsonl --layout stable_prefix
cat data/generated/train-b-v2.jsonl data/generated/train-b-sup-v2.jsonl data/generated/train-b-sup2-v2.jsonl > data/generated/train-b-v3h2.jsonl
python3 -m homeai.datagen.leakage data/generated/train-b-v3h2.jsonl --json data/generated/train-b-v3h2.leakage.json
echo "$(date '+%F %T') build-b-v3h2 done"
