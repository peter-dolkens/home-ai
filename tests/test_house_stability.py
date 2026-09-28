"""Guard: existing small houses, their prompts, get_state rendering and generated conversations must not
change when shared code (schema, snapshot, tools) is extended, e.g. for the big-house generator.

    PYTHONPATH=. python3 tests/test_house_stability.py      # print the digest and compare

EXPECTED was recorded on 2026-09-26 before homeai/house/bighouse.py was added."""
from __future__ import annotations

import hashlib
import json

from homeai.datagen.build import generate_one
from homeai.house.generator import make_house
from homeai.house.snapshot import system_prompt
from homeai.tools import TOOLS, get_state_text

EXPECTED = "2f9553e52265b2a4451392182607aaf56cda5a3a4d1cec02abec9ed800a3c575"

# plain seeds, build.py-style seeds (seed * 100_000 + i) and wild-pack seeds (900_000+)
SEEDS = list(range(100)) + [s * 100_000 + i for s in (1, 7) for i in range(50)] + [900_000 + i for i in range(100)]


def digest(verbose: bool = False) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(TOOLS, sort_keys=True).encode())
    for seed in SEEDS:
        for profile in (None, "bundook"):
            house = make_house(seed, profile)
            rows = [(e.entity_id, e.domain, e.name, e.cls, e.area, e.state, json.dumps(e.attributes, sort_keys=True), e.changed_s)
                    for e in house.entities.values()]
            people = [(p.name, p.home, p.room, p.changed_s) for p in house.people]
            areas = [(a.area_id, a.name, a.kind, a.spoken) for a in house.areas]
            h.update(json.dumps([str(house.now), house.description, house.profile, areas, rows, people]).encode())
            h.update(system_prompt(house, None).encode())
            h.update(get_state_text(house).encode())  # every entity through describe_state
    for i in range(150):  # scenario builders end to end (Conv, tools, sim)
        ex = generate_one(i, 3, None, "bundook", 0.4)
        h.update(json.dumps(ex, sort_keys=True).encode())
    d = h.hexdigest()
    if verbose:
        print(d)
    return d


def test_small_houses_unchanged():
    assert digest() == EXPECTED


if __name__ == "__main__":
    d = digest(verbose=True)
    assert d == EXPECTED, f"small-house output changed: {d} != {EXPECTED}"
    print("small houses, prompts, get_state rendering and generated conversations unchanged")
