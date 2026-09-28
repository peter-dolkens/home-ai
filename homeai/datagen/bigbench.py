"""Benchmark conversations on big houses, for comparing prompt / tool architectures at scale.

    PYTHONPATH=. python3 -m homeai.datagen.bigbench                       # -> data/generated/bigbench.jsonl
    PYTHONPATH=. python3 -m homeai.datagen.bigbench --check 40            # replay reference tool calls, compare

Same row format as build.py, plus "big": {"kind", "seed", "n", "patch"} so the eval harnesses rebuild the
house with homeai.house.bighouse.house_for(ex) (make_house ignores it). "patch" holds the entity and people
changes the scenario made before the transcript starts (speaker placement, everyone-away setups...), so the
rebuilt house is exactly the one the reference was written against. Seeds start at 9_000_000 so they can
never collide with build.py (seed * 100_000 + i) or the wild packs (900_000+)."""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import random

from ..house.bighouse import _pristine, house_for, state_patch
from ..house.generator import pick_speaker
from ..house.schema import Speaker
from ..tools_b import for_toolset
from .common import Conv
from .scenarios import BY_NAME
from .scenarios.aggregate import anything_on, open_check, top_power

# (kind, n entities, examples). random = make_big_house; bundook_* = the new build from its plan.
SCALES = [("random", 3000, 50), ("random", 10000, 50), ("random", 20000, 50), ("bundook_full", 0, 25), ("bundook_projected", 0, 25)]

# scenario -> weight; "missing" is left out (its list of things a house lacks is fixed and big houses have most of them)
MIX = {
    "control_area": 3, "control_here": 2, "control_brightness": 1, "control_named": 4, "control_compound": 2, "control_climate": 2,
    "clarify": 3, "clarify_in_room": 1, "act_only_one": 1, "confirm_sensitive": 2, "control_lock": 1,
    "status_entity": 3, "status_aggregate": 3, "who_home": 1, "presence_where": 4, "presence_in_room": 1,
    "check_house": 3, "away_motion": 1, "goodnight": 3, "leaving": 1,
    "energy_status": 2, "energy_night": 1, "energy_appliance": 1, "weather": 1,
    "media_play": 2, "media_transport": 1, "media_volume": 1, "media_what": 1, "media_multiroom": 1,
    "appliance_status": 1, "water_status": 1, "irrigation_control": 1, "vehicle_status": 1,
    "anything_on": 3, "open_check": 2, "top_power": 2,
}
BUILDERS = {**BY_NAME, "anything_on": anything_on, "open_check": open_check, "top_power": top_power}


def plan_scenarios(n: int, rng: random.Random) -> list[str]:
    """n scenario names, proportional to MIX, every scenario at least once when n allows."""
    total = sum(MIX.values())
    names = [k for k, w in MIX.items() for _ in range(max(1, round(w * n / total)))]
    rng.shuffle(names)
    while len(names) < n:
        names.append(rng.choices(list(MIX), weights=list(MIX.values()))[0])
    return names[:n]


def build_one(kind: str, n: int, seed: int, scenario: str, rng: random.Random, idx: int = 0):
    """Try the scenario on this house and a few neighbours (the builders are time and state gated)."""
    for attempt in range(8):
        s = seed + attempt * 7_919
        base = _pristine(kind, s, n)
        house = pickle.loads(pickle.dumps(base))
        start = {}

        def on_start(conv):  # the house as the transcript starts, after the builder set the scene
            start["patch"] = state_patch(base, conv.house)

        Conv.on_start = on_start
        try:
            speaker = pick_speaker(house, rng)
            conv = BUILDERS[scenario](house, speaker, rng)
        finally:
            Conv.on_start = None
        if conv is None:
            continue
        ex = conv.build()
        if conv.b_issues:
            ex["b_issues"] = conv.b_issues  # design B: what the reference uses that the B tool results did not show
        ex["id"] = f"big-{kind}-{n or 'plan'}-{idx}"
        ex["scenario"] = scenario
        ex["profile"] = None
        ex["big"] = {"kind": kind, "seed": s, "n": n, "patch": start["patch"]}
        ex["entities"] = len(base.entities)
        ex["primary_entities"] = sum(1 for e in base.entities.values() if not e.attributes.get("entity_category"))
        return ex
    return None


def check(path: str, limit: int):
    """Rebuild each house with house_for, check the system prompt matches, then replay every reference tool call in
    order and compare the tool results with the recorded ones."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    step = max(1, len(rows) // max(1, limit))
    bad = collections.Counter()
    shown = 0
    for ex in rows[::step][:limit]:
        house = house_for(ex)
        sp = ex["speaker"]
        speaker = Speaker(sp["name"], sp["area"], sp["channel"]) if sp else None
        _, execute, prompt = for_toolset(ex.get("toolset"))
        if prompt(house, speaker) != ex["messages"][0]["content"]:
            bad["system prompt differs"] += 1
        results = {}
        for m in ex["messages"]:
            for tc in m.get("tool_calls") or []:
                results[tc["id"]] = execute(house, tc["function"]["name"], tc["function"]["arguments"])
            if m["role"] == "tool" and results.get(m["tool_call_id"]) != m["content"]:
                bad[f"tool result differs ({m['name']})"] += 1
        if shown < 3 and any(m.get("tool_calls") for m in ex["messages"]):
            shown += 1
            print(f"--- {ex['id']} {ex['category']} ({ex['entities']} entities)\nuser: {ex['messages'][1]['content']}")
            for m in ex["messages"][2:]:
                if m.get("tool_calls"):
                    print("call:", json.dumps([tc["function"] for tc in m["tool_calls"]])[:300])
                elif m["role"] == "tool":
                    lines = m["content"].split("\n")
                    print(f"tool ({len(m['content'])} chars, {len(lines)} lines):", " / ".join(lines[:6])[:400])
                elif m["role"] in ("assistant", "user"):
                    print(f"{m['role']}:", m["content"][:300])
    n = min(limit, len(rows[::step]))
    print(f"\nchecked {n} examples: " + (", ".join(f"{k}: {v}" for k, v in bad.items()) if bad else "prompts and all tool results identical"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/generated/bigbench.jsonl")
    ap.add_argument("--seed", type=int, default=9_000_000)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the per-scale example counts (quick runs)")
    ap.add_argument("--check", type=int, default=0, help="replay N examples of --out instead of building")
    ap.add_argument("--toolset", default="a", choices=["a", "b"], help="b: design-B transcripts of the same requests (write to bigbench-b.jsonl)")
    a = ap.parse_args()
    if a.check:
        return check(a.out, a.check)
    Conv.toolset = a.toolset
    rows = []
    for si, (kind, n, count) in enumerate(SCALES):
        rng = random.Random(a.seed + si)
        for i, scenario in enumerate(plan_scenarios(max(1, round(count * a.scale)), rng)):
            ex = build_one(kind, n, a.seed + si * 100_000 + i * 10, scenario, rng, i)
            if ex is None:  # fall back to something that always builds
                ex = build_one(kind, n, a.seed + si * 100_000 + i * 10, "status_aggregate", rng, i)
            rows.append(ex)
        print(f"{kind} {n or ''}: {sum(1 for r in rows if r['big']['kind'] == kind and r['big']['n'] == n)} built", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(rows)} conversations -> {a.out}")
    by = collections.Counter((r["big"]["kind"] + (f" {r['big']['n']}" if r["big"]["n"] else ""), r["category"]) for r in rows)
    scales = list(dict.fromkeys(k for k, _ in by))
    cats = sorted({c for _, c in by}, key=lambda c: -sum(v for (s, cc), v in by.items() if cc == c))
    print(f"\n{'category':20s}" + "".join(f"{s:>18s}" for s in scales) + f"{'total':>7s}")
    for c in cats:
        print(f"{c:20s}" + "".join(f"{by.get((s, c), 0):18d}" for s in scales) + f"{sum(by.get((s, c), 0) for s in scales):7d}")
    ents = collections.defaultdict(list)
    for r in rows:
        tool = [len(m["content"]) for m in r["messages"] if m["role"] == "tool"]
        ents[r["big"]["kind"] + (f" {r['big']['n']}" if r["big"]["n"] else "")].append(
            (r["entities"], r["primary_entities"], len(r["messages"][0]["content"]), sum(tool), max(tool, default=0)))
    print()
    for s, v in ents.items():
        mean = lambda i: sum(x[i] for x in v) / len(v)  # noqa: E731
        print(f"{s:18s} entities {mean(0):7.0f}  primary {mean(1):6.0f}  system prompt {mean(2):7.0f} chars (~{mean(2) / 3.6:.0f} tok)  "
              f"tool results/conv {mean(3):7.0f} chars (~{mean(3) / 3.6:.0f} tok), largest {max(x[4] for x in v)} chars")


if __name__ == "__main__":
    main()
