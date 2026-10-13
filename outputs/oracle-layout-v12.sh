#!/usr/bin/env bash
# One-off (1 Oct): layout checks for v12 (clock_in_user) - train-v9 / val-v3 and the heard sets convert losslessly, then
# an oracle e2e replay of the reference on every -sp set and two heard conditions per stem with the requests rendered
# as clock_in_user at load time, and the tiered oracle on test-v2-b.
cd "$(dirname "$0")/.."
export PYTHONPATH=.
S=$(mktemp -d)
for f in train-v9 val-v3; do
  python3 -m homeai.datagen.relayout convert --layout clock_in_user data/generated/$f.jsonl $S/$f-cu.jsonl
done
rm -rf $S
python3 -m homeai.datagen.relayout heard --layout clock_in_user
H=data/generated/heard
python3 -m homeai.datagen.relayout oracle --layout clock_in_user --out outputs/eval/oracle-layout-cu \
  data/generated/{test-v2,wild,wild2,test-rules,bigbench,test-v2-b,wild-b,wild2-b,bigbench-b}-sp.jsonl \
  $H/{test-v2,wild,wild2,bigbench-bundook,test-v2-b,wild-b,wild2-b,bigbench-b-bundook}-{small-biased,parakeet-noisy10}.jsonl
echo "oracle clock_in_user exit $? $(date '+%F %T')"
PYTHONPATH=. python3 -m homeai.tiered.eval --data data/generated/test-v2-b.jsonl --model oracle --leaf oracle --prompt-layout clock_in_user \
  --out outputs/eval/oracle-layout-cu/tiered-test-v2-b > /dev/null 2>&1
python3 -c "import json;d=json.load(open('outputs/eval/oracle-layout-cu/tiered-test-v2-b/report.json'));print('tiered', {k:d[k] for k in ('pass','turns','errors','history_errors','prompt_layout')})"
echo "all done $(date '+%F %T')"
