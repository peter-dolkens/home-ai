#!/usr/bin/env bash
# Stronger speech-to-text under far-field and TV noise on the 200-conversation Bundook set (6 Oct). The language models now
# score 84-87% typed but 68-74% from 3 m and 67-70% under TV noise, those two measured with Parakeet only; medium Whisper with
# the house's names as initial_prompt ("biased") was the best clean condition (82-85%). Here: names-biased medium and
# large-v3-turbo (and small, the cheap Mac option) on far-field and noisy audio, test voices and Piper as for the other sets.
set -euo pipefail
cd "$(dirname "$0")/.."
S=data/generated/bigbench-b-bundook200.jsonl
VR=(scripts/voice-roundtrip.sh --sets "$S" --voices test)
"${VR[@]}" --stt medium --conditions farfield-biased,noisy10-biased
"${VR[@]}" --stt large-v3-turbo --conditions biased,farfield-biased,noisy10-biased
"${VR[@]}" --stt small --conditions farfield-biased
ls data/generated/heard/ | grep bigbench-b-bundook200 | grep jsonl$
echo "$(date '+%F %T') heard-bundook200-stt done"
