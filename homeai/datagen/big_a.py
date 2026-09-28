"""Large-house training conversations in the champion's own format (design A: flat inventory prompt, get_state /
call_service), for the traditional fallback model (Peter, 27 Sep): the champion never trained on 3k-20k-entity houses
or the Bundook plan and loses ~10 points on them even with a 32k window.

Same builders and house kinds as design B's big-house share (homeai/datagen/train_b.py), toolset "a", category mix
from the v5 template data. Seeds from 7,000,000 (bigbench uses 9,000,000+, design B's big houses 6,000,000+).

    PYTHONPATH=. python3 -m homeai.datagen.big_a --n 1500 --out data/generated/big-a.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

from .common import Conv
from .train_b import BIG_KINDS, PER_HOUSE, SMALL_ONLY, mix_counts
from .bigbench import build_one


def _house(job):
    kind, n, seed, cats, first = job
    Conv.toolset = "a"
    out = []
    for k, cat in enumerate(cats):
        rng = random.Random(seed * 101 + k)
        ex = None
        for attempt in range(4):  # a neighbouring house when the scenario cannot be built there
            ex = build_one(kind, n, seed + attempt * 104_729, cat, rng, first + k)
            if ex is not None:
                break
        if ex is not None:
            ex["id"] = f"abig-{kind}-{n or 'plan'}-{first + k}"
            ex.pop("scenario", None)
            out.append(ex)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--mix", default="data/generated/train-v5.jsonl")
    ap.add_argument("--seed", type=int, default=7_000_000)
    ap.add_argument("--out", default="data/generated/big-a.jsonl")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    counts = {c: v for c, v in mix_counts(a.mix).items() if c not in SMALL_ONLY}
    total = sum(counts.values())
    rng = random.Random(a.seed)
    cats = [c for c, v in counts.items() for _ in range(round(v / total * a.n))]
    rng.shuffle(cats)
    jobs, first = [], 0
    for kind, n, share in BIG_KINDS:
        want = round(a.n * share)
        mine, cats = cats[:want], cats[want:]
        for h in range(0, len(mine), PER_HOUSE):
            jobs.append((kind, n, a.seed + len(jobs) * 1_009, mine[h:h + PER_HOUSE], first))
            first += PER_HOUSE
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for out in ex.map(_house, jobs):
            rows.extend(out)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{a.out}: {len(rows)} examples; kinds", collections.Counter(r["big"]["kind"] + str(r["big"].get("n") or "") for r in rows))


if __name__ == "__main__":
    main()
