#!/usr/bin/env bash
# One-off (30 Sep): layout checks for v11 (user_context) - heard sets convert losslessly, then an oracle e2e replay of the
# reference on every -sp set and two heard conditions per stem with the requests rendered as user_context at load time,
# and a stable_prefix re-run on the small sets (the evals now render requests instead of rewriting rows).
cd "$(dirname "$0")/.."
export PYTHONPATH=.
python3 -m homeai.datagen.relayout heard --layout user_context
H=data/generated/heard
python3 -m homeai.datagen.relayout oracle --layout stable_prefix --out outputs/eval/oracle-layout-sp2 \
  data/generated/{test-v2,wild,wild2,test-rules,test-v2-b,wild-b,wild2-b}-sp.jsonl
echo "oracle stable_prefix exit $?"
python3 -m homeai.datagen.relayout oracle --layout user_context --out outputs/eval/oracle-layout-uc \
  data/generated/{test-v2,wild,wild2,test-rules,bigbench,test-v2-b,wild-b,wild2-b,bigbench-b}-sp.jsonl \
  $H/{test-v2,wild,wild2,bigbench-bundook,test-v2-b,wild-b,wild2-b,bigbench-b-bundook}-{small-biased,parakeet-noisy10}.jsonl
echo "oracle user_context exit $? $(date '+%F %T')"
PYTHONPATH=. python3 -m homeai.tiered.eval --data data/generated/test-v2-b.jsonl --model oracle --leaf oracle --prompt-layout user_context \
  --out outputs/eval/oracle-layout-uc/tiered-test-v2-b > /dev/null 2>&1
python3 -c "import json;d=json.load(open('outputs/eval/oracle-layout-uc/tiered-test-v2-b/report.json'));print('tiered', {k:d[k] for k in ('pass','turns','errors','history_errors','prompt_layout')})"
echo "all done $(date '+%F %T')"
