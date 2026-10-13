"""Score the 200-conversation Bundook set on test turns seen vs unseen (word-for-word, normalised) in training (6 Oct).

46% of its user turns appear verbatim in train-b-v7's Bundook-plan rows; on the same house the same words mean the same action,
so large adapters can partly remember answers. Reports typed and the mean of the eight heard conditions for each split.
    python3 -m homeai.eval.unseen_split
"""
import json, re, os, sys
def norm(t): return re.sub(r"[^a-z0-9 ]+", "", (t or "").lower()).strip()
seen = set()
for l in open("data/generated/train-b-v7.jsonl"):
    r = json.loads(l)
    seen.update(norm(m["content"]) for m in r["messages"] if m["role"] == "user")
# written test turns, keyed (id, turn index)
key = {}
for l in open("data/generated/bigbench-b-bundook200.jsonl"):
    r = json.loads(l)
    us = [m["content"] for m in r["messages"] if m["role"] == "user"]
    for i, u in enumerate(us):
        key[(r["id"], i)] = norm(u) in seen
def score(d):
    p = os.path.join(d, "turns.jsonl")
    if not os.path.exists(p): return None
    rows = [json.loads(l) for l in open(p)]
    s = [r["ok"] for r in rows if key.get((r["id"], r["turn"])) is True]
    u = [r["ok"] for r in rows if key.get((r["id"], r["turn"])) is False]
    return (100 * sum(s) / len(s), len(s), 100 * sum(u) / len(u), len(u)) if s and u else None
models = {"LFM r256": "lfm25-1.2b-lora-b-v7r256", "LFM r128": "lfm25-1.2b-lora-b-v7r128", "LFM r64": "lfm25-1.2b-lora-b-v7r64",
          "LFM r16 (B-v7)": "lfm25-1.2b-lora-b-v7", "Spark B-v6n r32": "spark-x25-4b-lora-b-v6n-r32", "Spark B-v6n": "spark-x25-4b-lora-b-v6n",
          "2B B-v2u r32": "2b-lora-b-v2u-r32"}
conds = ["written", "medium-biased", "small-biased", "small-clean", "parakeet-clean", "tiny-clean", "parakeet-farfield", "parakeet-noisy10"]
for name, m in models.items():
    t = score(f"outputs/eval/homeai-{m}-bundook200-b-e2e")
    h = [score(f"outputs/eval/homeai-{m}-bigbench-b-bundook200-heard-{c}") for c in conds]
    hs = [x for x in h if x]
    line = f"{name:17s}"
    if t: line += f" typed seen {t[0]:5.1f} unseen {t[2]:5.1f} (n={t[1]}/{t[3]})"
    if len(hs) == 8:
        line += f" | heard mean seen {sum(x[0] for x in hs)/8:5.1f} unseen {sum(x[2] for x in hs)/8:5.1f}"
    print(line)
