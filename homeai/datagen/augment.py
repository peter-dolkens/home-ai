"""Apply paraphrases to a corpus: rewrite the first user turn of each example using the paraphrase
bank (built from slot skeletons, see skeleton.py) while leaving every tool call and answer intact.

    python -m homeai.datagen.augment --corpus data/corpus/v1 --bank data/paraphrases/v1.json \
        --out data/corpus/v1-aug --share 0.6 --copies 1

--share    fraction of examples whose first utterance is paraphrased (the rest keep the template form)
--copies   extra paraphrased copies to emit per example (0 = rewrite in place; 1 = keep original + add one)
Paraphrases that lose a slot value, a number, or a polarity word (on/off/open/close/lock/unlock) are rejected.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

from .common import voiceify
from .skeleton import fill, normalise, skeleton_for_example

POLARITY = [("on", "off"), ("open", "close"), ("lock", "unlock"), ("up", "down"), ("warm", "cool")]


def polarity_ok(orig: str, para: str) -> bool:
    o, p = orig.lower(), para.lower()
    for a, b in POLARITY:
        ra, rb = re.compile(rf"\b{a}\b"), re.compile(rf"\b{b}\b")
        if bool(ra.search(o)) != bool(ra.search(p)) and bool(rb.search(o)) != bool(rb.search(p)):
            return False
        # one-sided flips: original has 'on' only, paraphrase has 'off' only
        if ra.search(o) and not rb.search(o) and rb.search(p) and not ra.search(p):
            return False
        if rb.search(o) and not ra.search(o) and ra.search(p) and not rb.search(p):
            return False
    return True


def load_bank(path: str) -> dict[str, list[str]]:
    bank = json.load(open(path))
    return {normalise(k): v for k, v in bank.items()}


def paraphrase(ex: dict, bank: dict, rng: random.Random, house=None):
    sk, mapping = skeleton_for_example(ex, house)
    key = normalise(sk)
    options = bank.get(key)
    if not options:
        return None
    rng.shuffle(options)
    orig_first = next(m["content"] for m in ex["messages"] if m["role"] == "user")
    for p in options:
        if any(tok not in p for tok in mapping):
            continue
        if re.search(r"\{[a-z0-9]+\}", re.sub("|".join(re.escape(t) for t in mapping) or r"$^", "", p)):
            continue  # stray slot token not in mapping
        text = fill(p, mapping)
        if not polarity_ok(orig_first, text):
            continue
        kind = "q" if orig_first.rstrip().endswith("?") or text.lower().split(" ")[0] in ("is", "are", "where", "who", "what", "when", "how", "can", "will", "do", "does", "did", "any", "has", "have", "should") else "cmd"
        return voiceify(text[:1].upper() + text[1:] + ("?" if kind == "q" and not text.endswith("?") else ""), rng, kind)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--share", type=float, default=0.6)
    ap.add_argument("--copies", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    bank = load_bank(a.bank)
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    total = hit = 0
    for shard in sorted(glob.glob(os.path.join(a.corpus, "*.jsonl"))):
        out_path = os.path.join(a.out, os.path.basename(shard))
        with open(shard) as f, open(out_path, "w") as g:
            for line in f:
                ex = json.loads(line)
                total += 1
                if a.copies == 0:
                    if rng.random() < a.share:
                        p = paraphrase(ex, bank, rng)
                        if p:
                            hit += 1
                            for m in ex["messages"]:
                                if m["role"] == "user":
                                    m["content"] = p
                                    break
                            ex["augmented"] = True
                    g.write(json.dumps(ex, ensure_ascii=False) + "\n")
                else:
                    g.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    for k in range(a.copies):
                        if rng.random() >= a.share:
                            continue
                        p = paraphrase(ex, bank, rng)
                        if not p:
                            continue
                        hit += 1
                        cp = json.loads(json.dumps(ex))
                        for m in cp["messages"]:
                            if m["role"] == "user":
                                m["content"] = p
                                break
                        cp["id"] = f"{ex['id']}-p{k}"
                        cp["augmented"] = True
                        g.write(json.dumps(cp, ensure_ascii=False) + "\n")
    print(f"{total} examples read; {hit} paraphrased utterances written to {a.out}")


if __name__ == "__main__":
    main()
