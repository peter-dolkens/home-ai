#!/usr/bin/env bash
# Build data/generated/train-v9.jsonl (homeai/datagen/asr_v9.py): plan, speak every round-trip group once per
# synthesiser with parallel TTS workers, transcribe each group under its engine and condition in the homeai-voice
# container (scripts/voice-roundtrip.sh), then build, run the leakage check (stops on any leak) and write
# configs/qwen35-4b-lora-v9.json. Resumable: audio and transcripts are
# cached by content in $VOICE_CACHE.
#   VOICE_CACHE=/path/to/voice-cache nohup python3 -c "import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" \
#     scripts/build-v9.sh >> outputs/build-v9.log 2>&1 &     # detached: survives the launching session
set -euo pipefail
cd "$(dirname "$0")/.."
W=data/generated/v9
export PYTHONPATH=.
python3 -m homeai.datagen.asr_v9 plan
VR=(scripts/voice-roundtrip.sh --voices train --seed-salt v9 --out $W/heard)
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
python3 -m homeai.datagen.asr_v9 build
python3 -m homeai.datagen.leakage data/generated/train-v9.jsonl --json data/generated/train-v9.leakage.json
# the config: v8's with only the data paths and the output dir changed
python3 - <<'PY'
import json
c = json.load(open("configs/qwen35-4b-lora-v8.json"))
c["train"] = "data/generated/train-v9.jsonl"
c["out"] = "outputs/qwen35-4b-lora-v9"
json.dump(c, open("configs/qwen35-4b-lora-v9.json", "w"), indent=2)
open("configs/qwen35-4b-lora-v9.json", "a").write("\n")
PY
echo "$(date '+%F %T') build-v9 done"
