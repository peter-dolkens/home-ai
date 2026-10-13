"""Targeted mishearing suite: single-turn conversations whose user turn is a realistic speech-to-text error.

Each row is a test-v2-style conversation (system prompt, one user turn, the reference tool calls and answer), so
homeai.eval.e2e scores it unchanged against what the user MEANT. The user turn carries the misheard text; the written
one stays in "_written", and "_confusion" says what was changed:

  {"kind": room | person | number | homophone | observed, "term": "alfresco", "heard_as": "al fresco",
   "origin": "curated" | "observed:<engine>-<condition>", "source": <source example id>,
   "recoverable": whether the house context still pins down the meaning ("al fresco" -> alfresco yes;
                  "fifty percent" heard as "fifteen percent" no: nothing can recover a changed number)}

Sources are the first turns of test-v2, wild, wild2 and the Bundook bigbench rows. A curated room that no source
mentions ("battery shed", "media room") is written in by moving an area-targeted request to that room (text, area_id,
answer and must_mention together) when the room has something the service changes. Observed rows come from the
heard/*.jsonl files roundtrip.py wrote: every first turn where an engine lost a house term.

  python3 -m homeai.voice.confusions --out data/generated/heard/confusions.jsonl
  python3 -m homeai.eval.e2e --model homeai-4b-v2 --data data/generated/heard/confusions.jsonl --out outputs/eval/homeai-4b-v2-confusions
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

from . import vocab

SOURCES = {"test-v2": ("data/generated/test-v2.jsonl", None), "wild": ("data/generated/wild.jsonl", None),
           "wild2": ("data/generated/wild2.jsonl", None), "bigbench-bundook": ("data/generated/bigbench.jsonl", "bundook")}

# this house's rooms, as Whisper-class models actually mangle them (spacing, near words, accent)
ROOMS = {
    "alfresco": ["al fresco", "all fresco", "alfresca", "out fresco"],
    "ensuite": ["en suite", "on suite", "and sweet", "ensuit"],
    "mudroom": ["mud room", "mad room", "much room"],
    "Pete's office": ["Pete office", "Pits office", "Peat's office", "PT's office"],
    "Amber's office": ["Amba's office", "Ember's office", "Amber office", "Hamper's office"],
    "dining hall": ["dining whole", "dying hall", "dining all"],
    "carport": ["car port", "carpet", "car park"],
    "battery shed": ["battery shared", "battery shade", "bettery shed"],
    "lounge": ["lunge", "launch", "lunch"],
    "media room": ["meteor room", "medium room", "media rim"],
}
PEOPLE = {
    "Peter": ["Pita", "Peta"], "Pete": ["Pit", "Peat"], "Amber": ["Amba", "Ember"], "Freya": ["Fraya", "Frayer"],
    "Kai": ["Kye", "Chi"], "Hannah": ["Hanna", "Anna"], "Grace": ["Grays"], "Jack": ["Jake", "Jac"], "Josh": ["Gosh"],
    "Noah": ["Nora"], "Chloe": ["Clowie"], "Owen": ["Owing"], "Mia": ["Maya"], "Jess": ["Jazz"], "Liam": ["Leam"],
    "Kate": ["Kait"], "Tom": ["Tim"], "Priya": ["Prayer"], "Sophie": ["Sofie"], "Nina": ["Nena"], "Ben": ["Bin"],
    "Lucy": ["Loosey"], "Charlie": ["Charley"],
}
HOMOPHONES = {  # home-control words and the near words a recogniser hands back instead
    "lights": ["likes", "lies"], "light": ["like"], "blinds": ["lines", "blends"], "fan": ["van", "fun"],
    "fans": ["vans"], "lock": ["look", "lack"], "locked": ["looked"], "heater": ["eater", "heat her"],
    "on": ["an"], "off": ["of"], "aircon": ["air con", "Eric on"], "garage": ["garbage"], "pause": ["paws"],
    "kitchen": ["kitten"], "TV": ["tea vee"], "downlights": ["down lights"],
}
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
TEEN_TY = {13: 30, 14: 40, 15: 50, 16: 60, 17: 70, 18: 80, 19: 90}
TEEN_TY.update({v: k for k, v in TEEN_TY.items()})


def spoken(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    return "one hundred" if n == 100 else str(n)


def stable(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:10], 16)


def first_turn(ex: dict):
    """(conversation cut after the first user turn's reference segment, index of that user turn)."""
    msgs = ex["messages"]
    u = [j for j, m in enumerate(msgs) if m["role"] == "user"]
    end = u[1] if len(u) > 1 else len(msgs)
    return {**{k: v for k, v in ex.items() if k != "messages"}, "messages": copy.deepcopy(msgs[:end])}, u[0]


def pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z'])" + re.escape(term).replace("\\'", "['’]") + r"(?![A-Za-z])", re.I)


def swap(text: str, term: str, heard: str) -> str:
    def rep(m):
        s = m.group(0)
        return heard[:1].upper() + heard[1:] if s[:1].isupper() and not heard[:1].isupper() else heard
    return pattern(term).sub(rep, text, count=1)


def make_row(src: dict, j: int, heard: str, meta: dict, n: int) -> dict:
    row = copy.deepcopy(src)
    u = row["messages"][j]
    u["_written"] = u.get("_written", u["content"])
    u["content"] = heard
    row["id"] = f"conf-{n:04d}-{src['id']}"
    row["_confusion"] = {**meta, "source": src["id"]}
    row.pop("_voice", None)
    return row


# ---------------------------------------------------------------------- moving a request to another room
def area_calls(conv) -> list[dict]:
    return [tc["function"] for m in conv["messages"] for tc in (m.get("tool_calls") or [])]


def relocate(src: dict, j: int, house, to_area: str):
    """The same request aimed at another room, or None when it cannot be moved cleanly (not purely area-targeted,
    the room is not named in the text, or nothing in the new room would change)."""
    from ..tools import execute_tool
    calls = area_calls(src)
    acts = [c for c in calls if c["name"] == "call_service"]
    if not acts or len(acts) != len(calls):
        return None
    froms = set()
    for c in acts:
        ids = (c["arguments"].get("target") or {}).get("area_id")
        if not ids or (c["arguments"].get("target") or {}).get("entity_id"):
            return None
        froms.update(ids if isinstance(ids, list) else [ids])
    if len(froms) != 1 or to_area in froms:
        return None
    fa = froms.pop()
    text = src["messages"][j]["content"]
    names = [n for n in {house.area_spoken(fa), house.area_name(fa)} if pattern(n).search(text)]
    if not names:
        return None
    old, new = names[0], house.area_spoken(to_area)
    if not re.match(r"[A-Z][a-z]+'s ", new):  # "Pete's office" keeps its capital, "Carport" mid-sentence does not
        new = new.lower()
    conv = copy.deepcopy(src)
    conv["messages"][j]["content"] = pattern(old).sub(new, text, count=1)
    for m in conv["messages"][j + 1:]:
        for tc in m.get("tool_calls") or []:
            t = tc["function"]["arguments"]["target"]
            t["area_id"] = [to_area] if isinstance(t["area_id"], list) else to_area
        if m["role"] == "assistant" and m.get("content"):
            m["content"] = pattern(old).sub(lambda x: new[:1].upper() + new[1:] if x.group(0)[:1].isupper() else new, m["content"])
            if m.get("_eval", {}).get("must_mention"):
                m["_eval"]["must_mention"] = [pattern(old).sub(new.lower(), x) for x in m["_eval"]["must_mention"]]
    before = copy.deepcopy(house)
    after = copy.deepcopy(house)
    for c in area_calls(conv):
        execute_tool(after, "call_service", c["arguments"])
    changed = any((e.state, e.attributes) != (before.entities[k].state, before.entities[k].attributes)
                  for k, e in after.entities.items())
    return conv if changed else None


# ---------------------------------------------------------------------- building the suite
def load_sources(root: str) -> list[tuple[dict, int]]:
    out = []
    for name, (path, where) in SOURCES.items():
        for ex in map(json.loads, open(os.path.join(root, path))):
            if where and where not in (ex.get("big") or {}).get("kind", ""):
                continue
            conv, j = first_turn(ex)
            conv["_set"] = name
            out.append((conv, j))
    return out


def curated(sources, per_pair: int, rooms_min: int) -> list[tuple[dict, int, str, dict]]:
    """[(source conversation, user index, misheard text, _confusion meta)]."""
    out = []
    for kind, table in (("room", ROOMS), ("person", PEOPLE), ("homophone", HOMOPHONES)):
        for term, variants in table.items():
            hits = sorted([(c, j) for c, j in sources if pattern(term).search(c["messages"][j]["content"])],
                          key=lambda cj: stable(term, cj[0]["id"]))
            if kind == "room" and len(hits) < rooms_min:
                hits += moved_to(sources, term, rooms_min - len(hits))
            for vi, heard_as in enumerate(variants):
                for c, j in hits[vi * per_pair:(vi + 1) * per_pair] or hits[:per_pair]:
                    text = c["messages"][j]["content"]
                    out.append((c, j, swap(text, term, heard_as),
                                {"kind": kind, "term": term, "heard_as": heard_as, "origin": c.get("_origin", "curated"),
                                 "recoverable": True}))
    for c, j in sources:  # numbers: said as words (fine), and the teen/ty slip (not recoverable)
        text = c["messages"][j]["content"]
        m = re.search(r"\b(\d{1,2})( ?%| percent| degrees?|°)?", text)
        if not m or not area_calls(c):
            continue
        n, unit = int(m.group(1)), m.group(2) or ""
        unit_sp = " percent" if "%" in unit or "percent" in unit else " degrees" if unit else ""
        variants = [(spoken(n) + unit_sp, True, "said as words")]
        if n in TEEN_TY:
            variants.append((spoken(TEEN_TY[n]) + unit_sp, False, "teen/ty slip"))
        for heard_num, ok, why in variants:
            heard = text[:m.start()] + heard_num + text[m.end():]
            out.append((c, j, heard, {"kind": "number", "term": m.group(0), "heard_as": heard_num, "origin": "curated",
                                      "recoverable": ok, "note": why}))
    return out


_HOUSES: dict = {}


def house_cached(ex):
    k = vocab.house_key(ex)
    if k not in _HOUSES:
        _HOUSES[k] = vocab.house_of(ex)
    return copy.deepcopy(_HOUSES[k])


def moved_to(sources, room: str, need: int) -> list[tuple[dict, int]]:
    """Requests from rooms elsewhere in a house that has `room`, moved there."""
    out = []
    for c, j in sorted(sources, key=lambda cj: stable(room, cj[0]["id"])):
        if len(out) >= need:
            break
        if c.get("profile") != "bundook" and "bundook" not in (c.get("big") or {}).get("kind", ""):
            continue
        house = vocab.house_of(c) if c.get("big") else house_cached(c)  # big rows carry a state patch
        to = next((a.area_id for a in house.areas if room.lower() in {a.name.lower(), (a.spoken or "").lower()}), None)
        moved = to and relocate(c, j, house, to)
        if moved:
            moved["_origin"] = f"curated, moved to {room}"
            out.append((moved, j))
    return out


def observed(heard_dir: str, limit: int, max_wer: float = 0.5) -> list[tuple[dict, int, str, dict]]:
    """First turns where a round-trip engine lost a house term but kept most of the words (turn WER <= max_wer), one
    row per distinct (source, transcript)."""
    out, seen = [], set()
    for path in sorted(glob.glob(os.path.join(heard_dir, "*-*-*.jsonl"))):
        base = os.path.basename(path)[:-6]
        if base.startswith(("confusions", "probe")):
            continue
        for ex in map(json.loads, open(path)):
            if "messages" not in ex or not ex.get("_voice"):
                continue
            t0 = ex["_voice"]["turns"][0]
            if not t0.get("lost") or t0.get("wer", 1) > max_wer:  # near misses, not noise-swamped garbage
                continue
            conv, j = first_turn(ex)
            heard = conv["messages"][j]["content"]
            written = conv["messages"][j]["_written"]
            conv["messages"][j]["content"] = written
            key = (ex["id"], vocab.compact(vocab.norm_words(heard)))
            if key in seen:
                continue
            seen.add(key)
            kinds = sorted({x.split(":", 1)[0] for x in t0["lost"]})
            out.append((conv, j, heard, {"kind": "observed", "term": "; ".join(t0["lost"]), "heard_as": heard,
                                         "origin": f"observed:{ex['_voice']['stt']}-{ex['_voice']['condition']}",
                                         "recoverable": "number" not in kinds, "lost_kinds": kinds}))
    out.sort(key=lambda r: stable(r[0]["id"], r[2]))
    return out[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/generated/heard/confusions.jsonl")
    ap.add_argument("--heard", default="data/generated/heard", help="round-trip outputs to mine for observed errors")
    ap.add_argument("--per-pair", type=int, default=2, help="source conversations per (term, misheard form)")
    ap.add_argument("--rooms-min", type=int, default=6, help="write requests into rooms fewer sources mention")
    ap.add_argument("--observed", type=int, default=400, help="cap on observed rows")
    ap.add_argument("--sample", action="store_true", help="also write <out>-sample.jsonl, a per-kind subset (SAMPLE)")
    ap.add_argument("--max-wer", type=float, default=0.5, help="observed rows: highest turn WER kept")
    a = ap.parse_args()
    root = os.getcwd()
    sources = load_sources(root)
    rows = curated(sources, a.per_pair, a.rooms_min) + observed(a.heard, a.observed, a.max_wer)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    stats = collections.Counter()
    seen = set()
    with open(a.out, "w") as fh:
        n = 0
        for c, j, heard, meta in rows:
            if heard == c["messages"][j]["content"] or (c["id"], heard) in seen:
                continue
            seen.add((c["id"], heard))
            fh.write(json.dumps(make_row(c, j, heard, meta, n)) + "\n")
            stats[meta["kind"]] += 1
            stats["recoverable" if meta["recoverable"] else "unrecoverable"] += 1
            n += 1
    json.dump({"rows": n, **stats}, open(a.out.replace(".jsonl", ".stats.json"), "w"), indent=2)
    print(f"{a.out}: {n} rows", dict(stats))
    if a.sample:
        write_sample(a.out, a.sample)


# per kind, for hosts too slow for all of it; every number row stays (the only unrecoverable ones)
SAMPLE = {"room": 20, "person": 14, "homophone": 16, "number": 28, "observed": 40}


def write_sample(path: str, sizes: dict = SAMPLE):
    rows = [json.loads(l) for l in open(path)]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["_confusion"]["kind"]].append(r)
    keep = {r["id"] for k, rs in by.items() for r in sorted(rs, key=lambda r: stable(r["id"]))[: sizes.get(k, 0)]}
    out = path.replace(".jsonl", "-sample.jsonl")
    with open(out, "w") as fh:
        for r in rows:
            if r["id"] in keep:
                fh.write(json.dumps(r) + "\n")
    print(f"{out}: {len(keep)} rows", {k: min(len(v), sizes.get(k, 0)) for k, v in by.items()})


if __name__ == "__main__":
    main()
