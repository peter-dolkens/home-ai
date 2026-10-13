#!/usr/bin/env bash
# Build data/generated/train-b-v11.jsonl (6 Oct): design B's training data (train-b-split.jsonl) with the "medium" speech-to-text
# augmentation (homeai/datagen/asr_v9.py --preset medium: 45% of conversations round-tripped, transcribed mostly by medium
# Whisper with the house's names as initial_prompt, clean / 3 m / TV noise, a little Parakeet), a fresh draw (salt v9m), plus
# the cleaned Bundook-plan supplement (train-b-sup-v2c), i.e. LFM B-v7's recipe with the speech-to-text the house will run.
# Same steps as scripts/build-b-v6n.sh; audio and transcripts cached by content in $VOICE_CACHE.
#   nohup scripts/build-b-v11.sh >> outputs/build-b-v11.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
W=data/generated/v9bm
export PYTHONPATH=. IMAGE=${IMAGE:-homeai-voice:latest}
python3 -m homeai.datagen.asr_v9 plan --preset medium --salt v9m --src data/generated/train-b-split.jsonl --work $W
VR=(scripts/voice-roundtrip.sh --voices train --seed-salt v9m --out $W/heard)
t0=$(date +%s)
for tts in kokoro piper; do
  sets=$(ls $W/rt/${tts}__*.jsonl | paste -sd, -)
  "${VR[@]}" --tts $tts --tts-only --tts-workers ${TTS_WORKERS:-5} --sets "$sets"
done
t1=$(date +%s)
for f in $W/rt/*.jsonl; do
  IFS=_ read -r tts _ engine _ cond <<< "$(basename "$f" .jsonl)"
  "${VR[@]}" --tts "$tts" --stt "$engine" --conditions "$cond" --sets "$f" 2>&1 | grep -v cpuid_info
done
t2=$(date +%s)
echo "round trip: tts $((t1 - t0)) s, stt $((t2 - t1)) s (host $(hostname -s))"
python3 -m homeai.datagen.asr_v9 build --preset medium --salt v9m --work $W --out data/generated/train-b-hm.jsonl --layout stable_prefix
cat data/generated/train-b-hm.jsonl data/generated/train-b-sup-v2c.jsonl > data/generated/train-b-v11.jsonl
python3 -m homeai.datagen.leakage data/generated/train-b-v11.jsonl --json data/generated/train-b-v11.leakage.json
python3 -c "import json,sys; d=json.load(open('data/generated/train-b-v11.leakage.json')); d=d[0] if isinstance(d,list) else d; sys.exit(1 if d['augmented_overlap_turns'] else 0)"
python3 -m homeai.datagen.relayout check data/generated/train-b-split.jsonl data/generated/train-b-hm.jsonl
echo "$(date "+%F %T") build-b-v11 done"
