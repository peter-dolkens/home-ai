#!/usr/bin/env bash
# Heard (speech-to-text) versions of data/generated/bigbench-b-bundook200.jsonl (4 Oct): the larger held-out Bundook set in the
# conditions the heard evals use (scripts/heard-evals.sh CONDS), test voices and Piper as for the other heard sets.
set -euo pipefail
cd "$(dirname "$0")/.."
S=data/generated/bigbench-b-bundook200.jsonl
VR=(scripts/voice-roundtrip.sh --sets "$S" --voices test)
"${VR[@]}" --stt parakeet --conditions clean,farfield,noisy10
"${VR[@]}" --stt small --conditions clean,biased
"${VR[@]}" --stt medium --conditions biased
"${VR[@]}" --stt tiny --conditions clean
ls data/generated/heard/ | grep bigbench-b-bundook200 | grep jsonl$
echo "$(date '+%F %T') heard-bundook200 done"
