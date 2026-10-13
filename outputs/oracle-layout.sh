#!/usr/bin/env bash
# One-off (29 Sep): layout checks for v10 / B-v2 - heard sets convert losslessly, then an oracle e2e replay of the
# reference on every -sp set and on two heard conditions per stem, all in the stable_prefix layout.
cd "$(dirname "$0")/.."
export PYTHONPATH=.
python3 -m homeai.datagen.relayout heard
H=data/generated/heard
python3 -m homeai.datagen.relayout oracle --out outputs/eval/oracle-layout \
  data/generated/{test-v2,wild,wild2,test-rules,bigbench,test-v2-b,wild-b,wild2-b,bigbench-b}-sp.jsonl \
  $H/{test-v2,wild,wild2,bigbench-bundook,test-v2-b,wild-b,wild2-b,bigbench-b-bundook}-{small-biased,parakeet-noisy10}.jsonl
echo "oracle exit $? $(date '+%F %T')"
