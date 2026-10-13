"""v9 training data: train-v8 made robust to what speech-to-text actually hands the model.

Three augmentations of train-v8's conversations, each replacing user turns in place (the original stays in
"_written", the row gets "_v9": {"kind": ...}); assistant targets are untouched unless noted:
  roundtrip  ~35% of conversations: every user turn spoken (Kokoro ~70%, Piper ~30%, training voices only, salted
             seeds; homeai/voice/roundtrip.py --voices train --seed-salt v9) and transcribed under a mix of engines and
             conditions (MIX). A turn whose transcript cannot be acted on (unrecoverable(): WER over 0.6, a lost house
             term and half its words, two lost terms, or a changed number) keeps its written text: training the reference action on
             it would teach guessing.
  homophone  ~5%: one near-miss per conversation, built from the conversation's own house vocabulary: lights/likes,
             lock/look, off/of, on/an, fan/van, heater/eater; compound names split ("en suite", "mud room",
             "al fresco", "down lights"); possessives flattened ("Petes office"); digits said as words. "look the front
             door" for a lock request is ambiguous, so that turn's target becomes a short clarifying question and the
             conversation ends there; every other near-miss keeps its target.
  casing     ~30% of the remaining eligible conversations: user turns in STT style (common.stt_case).
Only conversations with no user turn in any held-out set (homeai/datagen/leakage.py) are augmented, and an
augmented turn that lands on a held-out utterance is reverted, so the augmentations add no leakage.

    python3 -m homeai.datagen.asr_v9 plan           # data/generated/v9/plan.json + rt/<tts>__<engine>__<condition>.jsonl
    scripts/build-v9.sh                              # plan, round trip in the homeai-voice container, build, check
    python3 -m homeai.datagen.asr_v9 build          # -> data/generated/train-v9.jsonl + train-v9.stats.json

The same recipe on other training data (design B-v2, scripts/build-b-v2.sh): --src / --out / --work, and --layout to
write the rows in a system prompt layout (homeai/house/snapshot.py relayout; B-v2: stable_prefix).
"""
from __future__ import annotations

import argparse
import collections
import copy
import glob
import hashlib
import json
import os
import re

from ..prompt_layout import apply as apply_layout
from ..voice import vocab
from . import leakage
from .common import stt_case

SRC = "data/generated/train-v8.jsonl"
OUT = "data/generated/train-v9.jsonl"
WORK = "data/generated/v9"
SALT = "v9"
SHARES = {"roundtrip": 0.35, "homophone": 0.05, "casing": 0.30}  # casing: of the eligible rest
KOKORO_SHARE = 0.7
MIX = {  # engine, condition -> share of round-tripped conversations
    ("parakeet", "clean"): 0.22, ("small", "biased"): 0.20, ("small", "clean"): 0.15, ("tiny", "clean"): 0.08,
    ("medium", "biased"): 0.08, ("base", "clean"): 0.03, ("parakeet", "farfield"): 0.07,
    ("small", "farfield-biased"): 0.05, ("parakeet", "noisy10"): 0.07, ("small", "noisy10-biased"): 0.05,
}
# "hard" (4 Oct): more conversations round-tripped and over half of them in far-field / TV-noise conditions; on the
# Bundook plan the best design-B model fell from 84% typed to 69% at 3 m and 67% under TV noise (pass --preset to both
# plan and build)
PRESETS = {
    "v9": (dict(SHARES), dict(MIX)),
    "hard": ({"roundtrip": 0.45, "homophone": 0.05, "casing": 0.30},
             {("parakeet", "clean"): 0.12, ("small", "biased"): 0.10, ("small", "clean"): 0.08, ("tiny", "clean"): 0.05,
              ("medium", "biased"): 0.05, ("parakeet", "farfield"): 0.15, ("small", "farfield-biased"): 0.12,
              ("parakeet", "noisy10"): 0.15, ("small", "noisy10-biased"): 0.12, ("parakeet", "noisy5"): 0.06}),
    # 6 Oct: the speech-to-text the house is likely to run (medium Whisper with the house's names as initial_prompt: 11-12%
    # word error from 3 m / under TV noise vs Parakeet's 18-29%), so the model learns that engine's error style
    "medium": ({"roundtrip": 0.45, "homophone": 0.05, "casing": 0.30},
               {("medium", "biased"): 0.25, ("medium", "farfield-biased"): 0.35, ("medium", "noisy10-biased"): 0.30,
                ("parakeet", "farfield"): 0.05, ("parakeet", "noisy10"): 0.05}),
}
HOMOPHONE_KINDS = {"homophone": 0.5, "split": 0.3, "number": 0.2}
SWAPS = {"lights": "likes", "light": "like", "lock": "look", "locked": "looked", "off": "of", "on": "an",
         "fan": "van", "fans": "vans", "heater": "eater", "heaters": "eaters"}
# pieces a compound house word can be heard as (the split must use two of these)
PARTS = set("en suite mud room al fresco down up lights light car port home pod bath bed side hall way drive door bell "
            "work shop sun store dish washer air con wash out lamp wall pool play fire place power point gate sky tool "
            "shed book shelf wine green house club gym dress robe spot flood strip ceiling night stand floor foot path "
            "back front yard gar age laundry chute rain water tank solar panel boot head board".split())
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def stable(*parts) -> float:
    return int(hashlib.sha1("|".join(map(str, (SALT,) + parts)).encode()).hexdigest()[:12], 16) / 16 ** 12


def said(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n == 100:
        return "one hundred"
    return str(n)


def user_idx(ex: dict) -> list[int]:
    return [j for j, m in enumerate(ex["messages"]) if m["role"] == "user"]


def pick_weighted(x: float, weights: dict):
    acc = 0.0
    for k, w in weights.items():
        acc += w
        if x < acc:
            return k
    return k


# ---------------------------------------------------------------------- plan
def plan(a):
    rows = [json.loads(l) for l in open(a.src)]
    test = leakage.held_out()
    eligible = [r for r in rows if not any(leakage.norm(r["messages"][j]["content"]) in test for j in user_idx(r))]
    order = sorted(eligible, key=lambda r: stable("order", r["id"]))
    n_rt = round(SHARES["roundtrip"] * len(rows))
    rt, rest = order[:n_rt], order[n_rt:]
    groups = collections.defaultdict(list)
    kinds = {}
    for r in rt:
        tts = "kokoro" if stable("tts", r["id"]) < KOKORO_SHARE else "piper"
        engine, cond = pick_weighted(stable("stt", r["id"]), MIX)
        groups[(tts, engine, cond)].append(r)
        kinds[r["id"]] = {"kind": "roundtrip", "tts": tts, "stt": engine, "condition": cond}
    for r in rest:
        kinds[r["id"]] = {"kind": None}
    os.makedirs(f"{a.work}/rt", exist_ok=True)
    for f in glob.glob(f"{a.work}/rt/*.jsonl"):
        os.remove(f)
    for (tts, engine, cond), rs in sorted(groups.items()):
        with open(f"{a.work}/rt/{tts}__{engine}__{cond}.jsonl", "w") as fh:
            for r in rs:
                fh.write(json.dumps(r) + "\n")
    json.dump({"source": a.src, "rows": len(rows), "eligible": len(eligible), "held_out_utterances": len(test),
               "kinds": kinds}, open(f"{a.work}/plan.json", "w"))
    print(f"{len(rows)} rows, {len(eligible)} eligible (no held-out user turn), {len(rt)} to round-trip in "
          f"{len(groups)} groups:")
    for (tts, engine, cond), rs in sorted(groups.items()):
        print(f"  {tts:7s} {engine:9s} {cond:16s} {len(rs):5d} conversations, "
              f"{sum(len(user_idx(r)) for r in rs):5d} turns")


# ---------------------------------------------------------------------- near misses
def swap_word(text: str, word: str, heard: str):
    m = re.search(r"(?<![A-Za-z'])" + re.escape(word) + r"(?![A-Za-z'])", text, re.I)
    if not m:
        return None
    w = m.group(0)
    h = heard.capitalize() if w[:1].isupper() else heard
    return text[:m.start()] + h + text[m.end():]


def splits(house_terms: vocab.Terms) -> dict:
    """{compound word: 'part part'} for the single-word room and device names of this house that split in two."""
    words = {w for ph in house_terms.rooms | house_terms.devices for w in ph}
    out = {}
    for w in words:
        for i in range(2, len(w) - 1):
            if w[:i] in PARTS and w[i:] in PARTS:
                out[w] = f"{w[:i]} {w[i:]}"
                break
    return out


def near_misses(ex: dict, j: int, terms: vocab.Terms) -> list[tuple[str, str, str]]:
    """[(kind, heard text, note)] for user turn j."""
    text = ex["messages"][j]["content"]
    out = []
    for w, h in SWAPS.items():
        t = swap_word(text, w, h)
        if t:
            out.append(("homophone", t, f"{w}->{h}"))
    for w, h in splits(terms).items():
        t = swap_word(text, w, h)
        if t:
            out.append(("split", t, f"{w}->{h}"))
    for p in terms.people:
        t = swap_word(text, p + "'s", p + "s")
        if t:
            out.append(("split", t, f"{p}'s->{p}s"))
    m = re.search(r"\b(\d{1,3})(\s?%| percent| degrees?|°)?", text)
    if m and int(m.group(1)) <= 100:
        unit = m.group(2) or ""
        sp = " percent" if "%" in unit or "percent" in unit else " degrees" if unit else ""
        out.append(("number", text[:m.start()] + said(int(m.group(1))) + sp + text[m.end():], f"{m.group(0)}->said"))
    return out


def lock_target(ex: dict, j: int, house) -> str | None:
    """The clarifying question when turn j asked to lock something (the swap makes it "look ...")."""
    seg = []
    for m in ex["messages"][j + 1:]:
        if m["role"] == "user":
            break
        seg += [tc["function"] for tc in m.get("tool_calls") or []]
    for c in seg:
        args = c["arguments"] if isinstance(c["arguments"], dict) else json.loads(c["arguments"])
        if c["name"] == "call_service" and args.get("service") == "lock":
            ids = (args.get("target") or {}).get("entity_id") or []
            ids = [ids] if isinstance(ids, str) else ids
            if len(ids) == 1 and ids[0] in house.entities:
                return f"Did you want the {house.entities[ids[0]].name.lower()} locked, or just checked?"
            return "Did you want that locked, or just checked?"
    return None


def homophone_row(ex: dict, test: set, quota: collections.Counter, terms: vocab.Terms, house):
    """The conversation with one near-miss user turn, or None."""
    for j in user_idx(ex):
        options = [o for o in near_misses(ex, j, terms) if leakage.norm(o[1]) not in test and quota[o[0]] > 0]
        if not options:
            continue
        options.sort(key=lambda o: (-quota[o[0]], stable("nm", ex["id"], o[2])))
        kind, heard, note = options[0]
        row = copy.deepcopy(ex)
        u = row["messages"][j]
        u["_written"], u["content"] = u["content"], heard
        meta = {"kind": "homophone", "sub": kind, "change": note, "turn": user_idx(ex).index(j)}
        if note == "lock->look":
            q = lock_target(ex, j, house)
            if q:  # ambiguous: ask, and end the conversation on the question
                row["messages"] = row["messages"][:j + 1] + [{"role": "assistant", "content": q}]
                meta["sub"] = "ambiguous"
        quota[kind] -= 1
        row["_v9"] = meta
        return row
    return None


# ---------------------------------------------------------------------- build
def unrecoverable(t: dict, written: str, heard: str) -> bool:
    """A transcript no listener could act on: most of its words wrong, a lost house term and half its words, two lost
    terms and a third of its words, or a number turned into a different number ("fifty" -> "fifteen")."""
    lost = t.get("lost") or []
    wer = t.get("wer", 0)
    if wer > 0.6 or lost and wer > 0.5 or len(lost) >= 2 and wer > 0.33:
        return True
    if any(x.startswith("number:") for x in lost):
        before = {w for w in vocab.norm_words(written) if w.isdigit()}
        return bool({w for w in vocab.norm_words(heard) if w.isdigit()} - before)
    return False


def load_heard(work: str = WORK) -> dict:
    out = {}
    for f in glob.glob(f"{work}/heard/*__*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            out[r["id"]] = r
    return out


def build(a):
    p = json.load(open(f"{a.work}/plan.json"))
    rows = [json.loads(l) for l in open(p["source"])]
    test = leakage.held_out()
    heard = load_heard(a.work)
    kinds = p["kinds"]
    n_h = round(SHARES["homophone"] * len(rows))
    quota = collections.Counter({k: round(v * n_h) for k, v in HOMOPHONE_KINDS.items()})
    stats = collections.Counter()
    houses: dict = {}
    out = []
    candidates = sorted((r for r in rows if kinds.get(r["id"], {"kind": "x"})["kind"] is None),
                        key=lambda r: stable("hom", r["id"]))
    hom = {}
    for r in candidates:
        if sum(quota.values()) <= 0:
            break
        key = vocab.house_key(r)
        if key not in houses:
            house = vocab.house_of(r)
            houses[key] = (vocab.Terms(house), house)
        row = homophone_row(r, test, quota, *houses[key])
        if row:
            hom[r["id"]] = row
    rest = [r for r in candidates if r["id"] not in hom]
    casing = {r["id"] for r in rest if stable("case", r["id"]) < SHARES["casing"]}
    samples = collections.defaultdict(list)
    for r in rows:
        k = kinds.get(r["id"], {"kind": "inherited-overlap"})
        if k["kind"] == "roundtrip":
            hr = heard.get(r["id"])
            if hr is None:
                stats["roundtrip_missing"] += 1
                out.append(r)
                continue
            row = copy.deepcopy(r)
            vt = {t["turn"]: t for t in hr["_voice"]["turns"]}
            hu = [m for m in hr["messages"] if m["role"] == "user"]
            kept = 0
            for n, j in enumerate(user_idx(row)):
                t, text = vt.get(n, {}), hu[n]["content"]
                garbled = unrecoverable(t, row["messages"][j]["content"], text)
                if not text.strip() or garbled or leakage.norm(text) in test:
                    stats["roundtrip_turn_kept_written:" + ("empty" if not text.strip() else "garbled" if garbled else "held-out")] += 1
                    continue
                row["messages"][j]["_written"] = row["messages"][j]["content"]
                row["messages"][j]["content"] = text
                kept += 1
            if not kept:
                stats["roundtrip_all_turns_kept_written"] += 1
                out.append(r)
                continue
            row["_v9"] = {"kind": "roundtrip", "tts": k["tts"], "stt": k["stt"], "condition": k["condition"],
                          "voice": hr["_voice"]["turns"][0].get("voice"), "turns": kept}
            stats[f"roundtrip:{k['tts']}"] += 1
            stats[f"roundtrip:{k['stt']}-{k['condition']}"] += 1
            stats["roundtrip"] += 1
            stats["roundtrip_turns"] += kept
            out.append(row)
            samples["roundtrip"].append(row)
        elif r["id"] in hom:
            row = hom[r["id"]]
            stats["homophone"] += 1
            stats[f"homophone:{row['_v9']['sub']}"] += 1
            out.append(row)
            samples["homophone:" + row["_v9"]["sub"]].append(row)
        elif r["id"] in casing:
            row = copy.deepcopy(r)
            changed = 0
            for j in user_idx(row):
                m = row["messages"][j]
                c = stt_case(m["content"])
                if c != m["content"]:
                    m["_written"], m["content"] = m["content"], c
                    changed += 1
            if changed:
                row["_v9"] = {"kind": "casing", "turns": changed}
                stats["casing"] += 1
                samples["casing"].append(row)
            out.append(row)
        else:
            stats["unchanged" if k["kind"] is None else "unchanged_inherited_overlap"] += 1
            out.append(r)
    if a.layout:
        out = [apply_layout(r, a.layout) for r in out]
    with open(a.out, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    lk = leakage.check(a.out, test)
    rep = {"source": p["source"], "rows": len(out), "eligible": p["eligible"], "prompt_layout": a.layout or "classic",
           "counts": dict(sorted(stats.items())),
           "leakage": {k: v for k, v in lk.items() if k != "inherited_top"}, "inherited_top": lk["inherited_top"]}
    json.dump(rep, open(a.out.replace(".jsonl", ".stats.json"), "w"), indent=2)
    print(json.dumps(rep["counts"], indent=1))
    print("leakage:", json.dumps(rep["leakage"]))
    for kind, rs in sorted(samples.items()):
        for r in sorted(rs, key=lambda r: stable("sample", r["id"]))[:3]:
            m = next(m for m in r["messages"] if m.get("_written"))
            print(f"  [{kind}] {m['_written']!r} -> {m['content']!r}")


def main():
    global SALT
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("plan", "build"))
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--work", default=WORK, help="plan, round-trip groups and transcripts")
    ap.add_argument("--layout", default=None, help="system prompt layout of the output rows (default: as the source)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="v9", help="augmentation shares and engine/condition mix")
    ap.add_argument("--salt", default=SALT, help="seed salt: another value draws an independent augmentation (pass to plan and build)")
    a = ap.parse_args()
    SALT = a.salt
    shares, mix = PRESETS[a.preset]
    SHARES.clear(); SHARES.update(shares)
    MIX.clear(); MIX.update(mix)
    plan(a) if a.step == "plan" else build(a)


if __name__ == "__main__":
    main()
