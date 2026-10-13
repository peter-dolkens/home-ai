"""Design-B training set: the template part of a reference training file, category for category, rebuilt as design-B
transcripts (homeai/tools_b.py), with a share on big houses and the usual paraphrased first utterances.

    PYTHONPATH=. python3 -m homeai.datagen.train_b --mix data/generated/train-v5.jsonl \
        --add clarify_in_room=144 act_only_one=150 --out data/generated/train-b.jsonl

--mix       category counts come from this file's template rows (agent-written "wild" rows cannot be rebuilt: left out)
--add       extra categories (any builder in homeai/datagen/bigbench.py BUILDERS), e.g. the v7 rule batches
--big-share fraction of examples on big houses (3k-20k entities, random and the Bundook plan); missing and smalltalk
            stay on small houses (big houses have most "missing" things, small talk needs no house)
Small houses use build.generate_one (seeds --seed*100_000 + i, 40% on the Bundook profile as in the corpus); big houses
use bigbench.build_one on seeds from --big-seed, ten examples per house. Seeds stay clear of the test sets (100_000+),
wild sets (800_000-920_011), the rule batches (7.1M-8.4M), the v2 corpus (20M-21M) and bigbench (9M+). A transcript
whose reference relies on something the B results did not show ("b_issues") is rebuilt on another house."""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

from ..house.bighouse import house_for
from .augment import load_bank, paraphrase
from .bigbench import BUILDERS, build_one
from .build import generate_one
from .common import Conv
from .scenarios import BY_NAME

SMALL_ONLY = {"missing", "smalltalk"}
# (kind, n entities, share of the big-house examples)
BIG_KINDS = [("random", 3000, 0.15), ("random", 6000, 0.15), ("random", 10000, 0.2), ("random", 15000, 0.15),
             ("random", 20000, 0.15), ("bundook_full", 0, 0.1), ("bundook_projected", 0, 0.1)]
PER_HOUSE = 10


def mix_counts(path: str) -> collections.Counter:
    return collections.Counter(json.loads(l)["category"] for l in open(path) if l.strip() and not json.loads(l).get("wild"))


def _small(job):
    cat, i, seed = job
    BY_NAME.setdefault(cat, BUILDERS[cat])  # --add may name a bigbench-only builder (anything_on, open_check, top_power)
    ex = generate_one(i, seed, cat, "bundook", 0.4, toolset="b")
    if ex:
        ex["id"] = f"b{seed}-{i}"
    return ex


def _big_house(job):
    """All examples planned on one big house (one process keeps its pristine-house cache warm)."""
    kind, n, seed, cats, first = job
    Conv.toolset = "b"
    out = []
    for k, cat in enumerate(cats):
        rng = random.Random(seed * 101 + k)
        ex = None
        for attempt in range(4):  # another neighbouring house when the scenario cannot be built or the B reference is flagged
            ex = build_one(kind, n, seed + attempt * 104_729, cat, rng, first + k)
            if ex is not None and not ex.get("b_issues"):
                break
            ex = None
        if ex is not None:
            ex["id"] = f"bbig-{kind}-{n or 'plan'}-{first + k}"
            ex.pop("scenario", None)
            out.append(ex)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="data/generated/train-v5.jsonl")
    ap.add_argument("--add", nargs="*", default=[], help="category=count")
    ap.add_argument("--out", default="data/generated/train-b.jsonl")
    ap.add_argument("--big-share", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=300, help="small houses: house_seed = seed * 100_000 + i")
    ap.add_argument("--big-seed", type=int, default=6_000_000)
    ap.add_argument("--bank", default="data/paraphrases/v1-train.json", help="paraphrase bank ('' for none)")
    ap.add_argument("--para-share", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every count (quick runs)")
    ap.add_argument("--big-kinds", default="", help="override BIG_KINDS: kind:n:share,... (e.g. the house-specific "
                    "supplement 'bundook_full:0:0.7,bundook_projected:0:0.3')")
    a = ap.parse_args()
    global BIG_KINDS
    if a.big_kinds:
        BIG_KINDS = [(k, int(n), float(f)) for k, n, f in (x.split(":") for x in a.big_kinds.split(","))]

    counts = mix_counts(a.mix)
    for kv in a.add:
        k, v = kv.split("=")
        counts[k] += int(v)
    counts = collections.Counter({k: max(1, round(v * a.scale)) for k, v in counts.items()})
    unknown = [k for k in counts if k not in BUILDERS]
    if unknown:
        raise SystemExit(f"no builder for {unknown}")
    total = sum(counts.values())
    share = a.big_share * total / max(1, total - sum(v for k, v in counts.items() if k in SMALL_ONLY))
    big_n = {k: 0 if k in SMALL_ONLY else round(v * share) for k, v in counts.items()}
    print(f"{total} examples, {sum(big_n.values())} on big houses")

    # small houses, in waves until every category has its count (some builders rarely apply: clarify_in_room needs a
    # satellite in a room with two devices one word can mean)
    small, left, i, mult = [], {k: v - big_n[k] for k, v in counts.items()}, 0, 1.0
    with ProcessPoolExecutor(a.workers) as pool:
        for _wave in range(6):
            jobs = []
            for cat, v in sorted(left.items()):
                for _ in range(round(v * mult) if v else 0):
                    jobs.append((cat, i, a.seed))
                    i += 1
            if not jobs:
                break
            got = [ex for ex in pool.map(_small, jobs, chunksize=64) if ex]
            for ex in got:
                if left[ex["category"]] > 0:
                    left[ex["category"]] -= 1
                    small.append(ex)
            mult = min(40.0, mult * 4)
    print(f"small houses: {len(small)}" + (f", short: {({k: v for k, v in left.items() if v})}" if any(left.values()) else ""), flush=True)

    # big houses: shuffle the planned categories over houses of each kind, PER_HOUSE per house; a second round on
    # new houses tops up what could not be built
    rng = random.Random(a.big_seed)
    big, hseed = [], a.big_seed
    todo = collections.Counter(big_n)
    with ProcessPoolExecutor(a.workers) as pool:
        for _round in range(3):
            plan = [cat for cat, v in sorted(todo.items()) for _ in range(v)]
            if not plan:
                break
            rng.shuffle(plan)
            hjobs, pos = [], 0
            for kind, n, frac in BIG_KINDS:
                want = round(len(plan) * frac) if (kind, n) != BIG_KINDS[-1][:2] else len(plan) - pos
                cats = plan[pos:pos + want]
                for j in range(0, len(cats), PER_HOUSE):
                    hjobs.append((kind, n, hseed, cats[j:j + PER_HOUSE], len(big) + pos + j + 100_000 * _round))
                    hseed += 10
                pos += want
            for batch in pool.map(_big_house, hjobs):
                for ex in batch:
                    todo[ex["category"]] -= 1
                    big.append(ex)
    print(f"big houses: {len(big)} of {sum(big_n.values())}", flush=True)

    rows = small + big
    if a.bank:
        bank = load_bank(a.bank)
        prng = random.Random(a.seed)
        hit = 0
        for ex in rows:
            if prng.random() >= a.para_share:
                continue
            p = paraphrase(ex, bank, prng, house_for(ex) if ex.get("big") else None)
            if p:
                hit += 1
                next(m for m in ex["messages"] if m["role"] == "user")["content"] = p
                ex["augmented"] = True
        print(f"paraphrased first utterances: {hit} of {len(rows)}")
    random.Random(a.seed).shuffle(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(rows)} -> {a.out}")
    got = collections.Counter(r["category"] for r in rows)
    gbig = collections.Counter(r["category"] for r in rows if r.get("big"))
    print(f"\n{'category':22s} {'want':>5s} {'got':>5s} {'big':>5s}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{k:22s} {v:5d} {got[k]:5d} {gbig[k]:5d}")
    kinds = collections.Counter((r["big"]["kind"], r["big"]["n"]) for r in rows if r.get("big"))
    print("\nbig houses:", ", ".join(f"{k} {n or ''}: {v}" for (k, n), v in sorted(kinds.items())))
    print(f"bundook profile (small): {sum(1 for r in rows if r.get('profile') == 'bundook')}")


if __name__ == "__main__":
    main()
