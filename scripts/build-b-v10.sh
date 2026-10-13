#!/usr/bin/env bash
# Build data/generated/train-b-v10.jsonl (5 Oct): LFM B-v7's data (train-b-v7: the hard speech-to-text augmentation +
# the cleaned Bundook-plan supplement) plus a SECOND, independently drawn hard augmentation of design B's data (asr_v9
# --preset hard --salt v9b, voices seeded with v9b), keeping only its augmented rows (ids suffixed -h2). The hard preset
# lifted LFM 4.3 points at 3 m on the 200-conversation house set (B-v7 vs B-v3h); this asks whether twice the noisy
# exposure lifts it further. Same steps as scripts/build-b-v6n.sh; audio and transcripts cached in $VOICE_CACHE.
#   nohup scripts/build-b-v10.sh >> outputs/build-b-v10.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
W=data/generated/v9bn2
export PYTHONPATH=. IMAGE=${IMAGE:-homeai-voice:latest}
python3 -m homeai.datagen.asr_v9 plan --preset hard --salt v9b --src data/generated/train-b-split.jsonl --work $W
VR=(scripts/voice-roundtrip.sh --voices train --seed-salt v9b --out $W/heard)
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
python3 -m homeai.datagen.asr_v9 build --preset hard --salt v9b --work $W --out data/generated/train-b-hard2.jsonl --layout stable_prefix
python3 - <<'PY'
import json
# 5 Oct: only the first draw's conversations are eligible (no user turn among the held-out utterances), so both draws
# augment the same ~3,600; keep the second-draw rows whose user turns differ (43% came back identical, mostly clean STT)
first = {}
for l in open("data/generated/train-b-v6n.jsonl"):
    r = json.loads(l)
    first[r["id"]] = [m["content"] for m in r["messages"] if m["role"] == "user"]
extra = []
for l in open("data/generated/train-b-hard2.jsonl"):
    r = json.loads(l)
    if r.get("augmented") and [m["content"] for m in r["messages"] if m["role"] == "user"] != first.get(r["id"]):
        r["id"] += "-h2"
        extra.append(r)
with open("data/generated/train-b-v10.jsonl", "w") as f:
    for l in open("data/generated/train-b-v7.jsonl"):
        f.write(l)
    for r in extra:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"train-b-v10: train-b-v7 + {len(extra)} second-draw augmented rows")
PY
python3 -m homeai.datagen.leakage data/generated/train-b-v10.jsonl --json data/generated/train-b-v10.leakage.json
python3 -c "import json,sys; d=json.load(open('data/generated/train-b-v10.leakage.json')); d=d[0] if isinstance(d,list) else d; sys.exit(1 if d['augmented_overlap_turns'] else 0)"
echo "$(date "+%F %T") build-b-v10 done"
