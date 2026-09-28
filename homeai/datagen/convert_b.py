"""Design-B versions of existing evaluation sets.

    PYTHONPATH=. python3 -m homeai.datagen.convert_b --data data/generated/test-v2.jsonl --out data/generated/test-v2-b.jsonl
    PYTHONPATH=. python3 -m homeai.datagen.convert_b --prompt-only --data data/generated/wild.jsonl --out data/generated/wild-b.jsonl

Template sets (ids "<seed>-<i>" from build.py): each example is rebuilt with generate_one, first with the current scenario
registry and then with the registry as it was before the 26 Sep rule scenarios (clarify_in_room, act_only_one) were
added; when the rebuilt design-A transcript equals the stored one, the design-B twin of the same build replaces it
("b_convert": "regen": same requests, house and answers, B tool turns). Anything else, and every example with
--prompt-only (agent-written wild sets), keeps its messages and only gets the B system prompt ("b_convert": "prompt"):
fine for the outcome eval (homeai/eval/e2e.py judges end state and key facts), not for teacher-forced tool-call scoring."""
from __future__ import annotations

import argparse
import collections
import json

from ..house.bighouse import house_for
from ..house.schema import Speaker
from ..house.snapshot import system_prompt, system_prompt_b
from . import build
from .scenarios import SCENARIOS

REGISTRIES = {"current": list(SCENARIOS),
              "pre-rules": [s for s in SCENARIOS if s[0] not in ("clarify_in_room", "act_only_one")]}


def speaker_of(ex):
    sp = ex.get("speaker")
    return Speaker(sp["name"], sp["area"], sp["channel"]) if sp else None


def prompt_only(ex):
    """Same conversation, B system prompt. Returns None when the stored prompt cannot be rebuilt from house_for(ex)."""
    house, sp = house_for(ex), speaker_of(ex)
    if system_prompt(house, sp) != ex["messages"][0]["content"]:
        return None
    out = json.loads(json.dumps(ex))
    out["messages"][0]["content"] = system_prompt_b(house, sp)
    out["toolset"] = "b"
    out["b_convert"] = "prompt"
    return out


def regen(ex):
    try:
        seed, i = (int(x) for x in ex["id"].split("-"))
    except ValueError:
        return None
    for name, reg in REGISTRIES.items():
        build.SCENARIOS[:] = reg
        try:
            a = build.generate_one(i, seed, None, "bundook", 0.4)
            if a is None or a["messages"] != ex["messages"]:
                continue
            b = build.generate_one(i, seed, None, "bundook", 0.4, toolset="b", strict_b=False)
        finally:
            build.SCENARIOS[:] = REGISTRIES["current"]
        said = lambda e: [(m["role"], m["content"], m.get("_eval")) for m in e["messages"][1:] if m["role"] == "user" or (m["role"] == "assistant" and m.get("content"))]  # noqa: E731
        if b is None or said(a) != said(b):
            return None
        out = {**{k: v for k, v in ex.items() if k != "messages"}, "messages": b["messages"], "toolset": "b", "b_convert": "regen"}
        if b.get("b_issues"):
            out["b_issues"] = b["b_issues"]
        out["registry"] = name
        return out
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-only", action="store_true", help="agent-written sets: swap the system prompt only")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    out, how = [], collections.Counter()
    for ex in rows:
        b = None if a.prompt_only else regen(ex)
        if b is None:
            b = prompt_only(ex)
        if b is None:
            how["failed (prompt not reproducible)"] += 1
            continue
        how[b["b_convert"] + (" (flagged)" if b.get("b_issues") else "")] += 1
        out.append(b)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} of {len(rows)} -> {a.out}: " + ", ".join(f"{k} {v}" for k, v in how.items()))


if __name__ == "__main__":
    main()
