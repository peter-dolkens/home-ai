"""Draw a training set from the corpus shards.

    python -m homeai.datagen.sample --corpus data/corpus/v1 --n 6000 --out data/generated/train.jsonl \
        --boost clarify=2 confirm_sensitive=2 check_house=1.5 --exclude data/generated/test.jsonl

Sampling is by category weight (the registry weights, times any --boost multipliers), without
replacement, and never reuses a house_seed that appears in the excluded files (the fixed test set).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random

from .scenarios import SCENARIOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-out", default=None)
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boost", nargs="*", default=[], help="category=multiplier")
    ap.add_argument("--exclude", nargs="*", default=[], help="jsonl files whose house_seeds must not be sampled")
    ap.add_argument("--extra", nargs="*", default=[], help="extra jsonl files (e.g. agent-written conversations) added in full")
    ap.add_argument("--extra-repeat", type=int, default=1, help="how many times to include each extra example (upweighting)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    banned = set()
    for p in a.exclude:
        for line in open(p):
            banned.add(json.loads(line)["house_seed"])
    pool = collections.defaultdict(list)
    for shard in sorted(glob.glob(f"{a.corpus}/*.jsonl")):
        for line in open(shard):
            ex = json.loads(line)
            if ex["house_seed"] in banned:
                continue
            pool[ex["category"]].append(ex)
    boost = {k: float(v) for k, v in (b.split("=") for b in a.boost)}
    weights = {name: w * boost.get(name, 1.0) for name, _, w in SCENARIOS if name in pool}
    total_w = sum(weights.values())
    picked = []
    for cat, w in weights.items():
        want = round(a.n * w / total_w)
        rows = pool[cat]
        rng.shuffle(rows)
        picked += rows[:want]
        if want > len(rows):
            print(f"warning: {cat}: wanted {want}, corpus has {len(rows)}")
    extras = []
    for p in a.extra:
        for line in open(p):
            ex = json.loads(line)
            if ex["house_seed"] in banned:
                continue
            for k in range(a.extra_repeat):
                cp = dict(ex)
                cp["id"] = f"{ex['id']}-x{k}" if k else ex["id"]
                extras.append(cp)
    if extras:
        print(f"extras: {len(extras)} examples from {len(a.extra)} file(s) (repeat x{a.extra_repeat})")
    rng.shuffle(picked)
    val, train = picked[:a.val] if a.val_out else [], picked[a.val:] if a.val_out else picked
    train = train + extras
    rng.shuffle(train)
    with open(a.out, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.val_out:
        with open(a.val_out, "w") as f:
            for r in val:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = collections.Counter(r["category"] for r in train)
    print(f"train {len(train)} -> {a.out}" + (f", val {len(val)} -> {a.val_out}" if a.val_out else ""))
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:20s} {v}")


if __name__ == "__main__":
    main()
