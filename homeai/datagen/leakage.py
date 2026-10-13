"""Train/test leakage check on user utterances, compared after homeai.voice.vocab's normalisation (case, punctuation,
fillers, number words and apostrophes folded), so "Turn on the light." and "turn on the light" are the same utterance.

    python3 -m homeai.datagen.leakage data/generated/train-v9.jsonl
    python3 -m homeai.datagen.leakage data/generated/train-v9.jsonl --json outputs/leakage-v9.json

The held-out side is every test, wild, wild2, para, rules and bigbench set in data/generated (wildtrain* is training
data), their heard/ transcripts, and the confusion suite. Two counts:
  augmented  user turns an augmentation wrote (they carry "_written", or the row a "_v9" kind): must be 0
  inherited  unaugmented turns that match a held-out turn; these come from the generator's shared templates
             ("yes please", "turn on the light", "we're away for the day, can you check on the house for me") and are
             in every earlier training set as well
Exit status 1 when any augmented turn leaks.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

from ..voice.vocab import norm_words

HELD_OUT = ("test", "wild", "bigbench", "para", "rules")


def norm(text: str) -> str:
    return " ".join(norm_words(text))


def held_out_files(root: str = "data/generated") -> list[str]:
    files = [f for f in glob.glob(f"{root}/*.jsonl") if any(k in os.path.basename(f) for k in HELD_OUT)
             and not os.path.basename(f).startswith(("wildtrain", "train"))]  # train-b-wild.jsonl (3 Oct) is training data
    return sorted(files + glob.glob(f"{root}/heard/*.jsonl"))


def user_texts(ex: dict):
    """Every user-side text of a row: the turn as trained or scored, and the written original when there is one."""
    if "messages" in ex:
        for m in ex["messages"]:
            if m["role"] == "user":
                yield m["content"]
                if m.get("_written"):
                    yield m["_written"]
    elif "query" in ex:
        yield ex["query"]
        if ex.get("_written"):
            yield ex["_written"]


def held_out(root: str = "data/generated") -> dict:
    """normalised utterance -> the held-out files it appears in."""
    out = collections.defaultdict(set)
    for f in held_out_files(root):
        for l in open(f):
            if l.strip():
                for t in user_texts(json.loads(l)):
                    k = norm(t)
                    if k:
                        out[k].add(os.path.basename(f))
    return out


def check(path: str, test: dict) -> dict:
    aug, inh = collections.Counter(), collections.Counter()
    aug_turns = inh_turns = aug_rows = 0
    for l in open(path):
        ex = json.loads(l)
        augmented_row = bool(ex.get("_v9", {}).get("kind"))
        aug_rows += augmented_row
        for m in ex.get("messages", []):
            if m["role"] != "user":
                continue
            k = norm(m["content"])
            is_aug = augmented_row or "_written" in m
            if k in test:
                if is_aug:
                    aug[k] += 1
                    aug_turns += 1
                else:
                    inh[k] += 1
                    inh_turns += 1
            if is_aug and m.get("_written") and norm(m["_written"]) in test:
                aug[norm(m["_written"])] += 1
                aug_turns += 1
    return {"file": path, "held_out_utterances": len(test), "augmented_rows": aug_rows,
            "augmented_overlap_turns": aug_turns, "augmented_overlap_distinct": len(aug),
            "augmented_examples": aug.most_common(10),
            "inherited_overlap_turns": inh_turns, "inherited_overlap_distinct": len(inh),
            "inherited_top": inh.most_common(15)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train", nargs="+")
    ap.add_argument("--json", help="write the report here too")
    a = ap.parse_args()
    test = held_out()
    reps = [check(p, test) for p in a.train]
    for r in reps:
        print(json.dumps({k: v for k, v in r.items() if k != "inherited_top"}))
        print("  inherited, most common:", r["inherited_top"][:8])
    if a.json:
        json.dump(reps, open(a.json, "w"), indent=2)
    raise SystemExit(1 if any(r["augmented_overlap_turns"] for r in reps) else 0)


if __name__ == "__main__":
    main()
