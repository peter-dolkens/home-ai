#!/usr/bin/env bash
# Build data/generated/train-b-v2.jsonl: design B's training data (train-b-split.jsonl) with v9's speech-to-text
# augmentation (homeai/datagen/asr_v9.py: ~35% round trips, Kokoro-heavy, training-only voices, the v9 seed salt; STT-style
# casing; homophone / near-miss turns) and the v10 prompt layout (stable_prefix). Same steps as scripts/build-v9.sh: plan,
# speak every round-trip group once per synthesiser, transcribe each group under its engine and condition in the
# homeai-voice container, build, leakage check (stops on any leak), then configs/qwen35-4b-lora-b-v2.json. Audio and
# transcripts are cached by content in $VOICE_CACHE (shared with v9's run), so a rerun only does what is missing.
#   VOICE_CACHE=/path/to/voice-cache nohup scripts/build-b-v2.sh >> outputs/build-b-v2.log 2>&1 & disown
set -euo pipefail
cd "$(dirname "$0")/.."
W=data/generated/v9b
export PYTHONPATH=. IMAGE=${IMAGE:-homeai-voice:latest}
python3 -m homeai.datagen.asr_v9 plan --src data/generated/train-b-split.jsonl --work $W
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
python3 -m homeai.datagen.asr_v9 build --work $W --out data/generated/train-b-v2.jsonl --layout stable_prefix
python3 -m homeai.datagen.leakage data/generated/train-b-v2.jsonl --json data/generated/train-b-v2.leakage.json
python3 -m homeai.datagen.relayout check data/generated/train-b-split.jsonl data/generated/train-b-v2.jsonl
# the config: B's with only the training file, the output dir, the layout and the comment changed
python3 - <<'PY'
import json
c = json.load(open("configs/qwen35-4b-lora-b.json"))
s = json.load(open("data/generated/train-b-v2.stats.json"))
lk, n = s["leakage"], s["counts"]
c["_comment"] = (f"Design B + voice robustness (Peter, 29 Sep): B's data and recipe (configs/qwen35-4b-lora-b.json) with v9's "
                 f"speech-to-text augmentation (homeai/datagen/asr_v9.py, scripts/build-b-v2.sh): {n.get('roundtrip', 0)} of "
                 f"{s['rows']} conversations' user turns replaced by round trips (Kokoro/Piper -> parakeet/whisper, training-only "
                 f"voices), STT-style casing on {n.get('casing', 0)} more, {n.get('homophone', 0)} homophone/split-name/number "
                 f"near-misses ({n.get('homophone:ambiguous', 0)} ambiguous lock/look -> clarifying question); leakage vs every "
                 f"held-out set incl. the -b and heard sets: {lk['augmented_overlap_turns']}. v10 prompt layout (stable_prefix: "
                 f"Now / Speaker / People in a final 'Current context:' block, homeai/house/snapshot.py relayout).")
c["train"] = "data/generated/train-b-v2.jsonl"
c["out"] = "outputs/qwen35-4b-lora-b-v2"
c["prompt_layout"] = "stable_prefix"
json.dump(c, open("configs/qwen35-4b-lora-b-v2.json", "w"), indent=2)
open("configs/qwen35-4b-lora-b-v2.json", "a").write("\n")
PY
echo "$(date '+%F %T') build-b-v2 done"
