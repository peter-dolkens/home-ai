"""Generate the synthetic training set.

    python -m homeai.datagen.build --n 3000 --out data/generated

Every example gets its own randomly generated house. Output is JSONL in chat
format (system/user/assistant/tool messages, tool_calls with dict arguments);
the tool schemas live in homeai.tools and are attached at training time.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

from ..house.generator import make_house, pick_speaker
from .common import Conv
from .scenarios import BY_NAME, SCENARIOS


def generate_one(i: int, base_seed: int, only: str | None = None, profile: str | None = None, profile_share: float = 0.0,
                 toolset: str | None = None, strict_b: bool = True):
    """toolset "b": the design-B transcript (homeai/tools_b.py). strict_b: a B transcript whose reference relies on
    something the B tool results did not show counts as a failed attempt (else it is kept with "b_issues")."""
    rng = random.Random(base_seed * 1_000_003 + i)
    house_seed = base_seed * 100_000 + i
    names = [n for n, _, _ in SCENARIOS]
    weights = [w for _, _, w in SCENARIOS]
    use_profile = profile if (profile and rng.random() < profile_share) else None
    for _attempt in range(6):
        name = only or rng.choices(names, weights=weights)[0]
        house = make_house(house_seed, use_profile)
        speaker = pick_speaker(house, rng)
        conv = BY_NAME[name](house, speaker, rng) if not toolset else _with_toolset(toolset, BY_NAME[name], house, speaker, rng)
        if conv is not None and toolset == "b" and conv.b_issues and strict_b:
            conv = None
        if conv is not None:
            ex = conv.build()
            ex["id"] = f"{base_seed}-{i}"
            ex["profile"] = use_profile
            if conv.b_issues:
                ex["b_issues"] = conv.b_issues
            return ex
        if only:
            house_seed += 7_919  # try another house for this scenario
    return None


def _with_toolset(toolset: str, builder, *args):
    old, Conv.toolset = Conv.toolset, toolset
    try:
        return builder(*args)
    finally:
        Conv.toolset = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data/generated")
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--test", type=float, default=0.05)
    ap.add_argument("--only", default=None, help="generate a single scenario type (debugging)")
    ap.add_argument("--profile", default="bundook", help="fixed house profile to mix in (see homeai/house/profiles.py)")
    ap.add_argument("--profile-share", type=float, default=0.4, help="fraction of examples built on the profile house")
    ap.add_argument("--single", default=None, help="write all examples to this one JSONL (corpus shard) instead of train/val/test splits")
    ap.add_argument("--toolset", default=None, choices=["a", "b"], help="b: design-B transcripts (rooms-only prompt, homeai/tools_b.py)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    examples = []
    for i in range(args.n):
        ex = generate_one(i, args.seed, args.only, args.profile, args.profile_share, args.toolset)
        if ex:
            examples.append(ex)
    if args.single:
        os.makedirs(os.path.dirname(args.single) or ".", exist_ok=True)
        with open(args.single, "w") as f:
            for r in examples:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"shard: {len(examples)} -> {args.single}")
        return
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_test = int(len(examples) * args.test)
    n_val = int(len(examples) * args.val)
    splits = {"test": examples[:n_test], "val": examples[n_test:n_test + n_val], "train": examples[n_test + n_val:]}
    for name, rows in splits.items():
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} -> {path}")
    cats = collections.Counter(e["category"] for e in examples)
    print(f"profile houses: {sum(1 for e in examples if e.get('profile'))} / {len(examples)}")
    turns = collections.Counter(sum(1 for m in e["messages"] if m["role"] == "assistant") for e in examples)
    chars = sum(len(json.dumps(e["messages"])) for e in examples) / max(1, len(examples))
    print("\ncategories:")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22s} {v}")
    print(f"\nassistant turns per example: {dict(sorted(turns.items()))}")
    print(f"avg chars per example: {chars:.0f}  (~{chars / 3.6:.0f} tokens)")


if __name__ == "__main__":
    sys.exit(main())
