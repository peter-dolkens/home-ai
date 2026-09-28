"""Offline checks for the room-leaf schema, executor and data generator (homeai/leaf/).

    PYTHONPATH=. python3 tests/test_leaf.py"""
from __future__ import annotations

import random

from homeai.house.sim import apply_service
from homeai.leaf import data as D
from homeai.leaf.schema import apply_leaf, leaf_execute, outcome, room_model, room_tools
from homeai.leaf.tokens import MAX_LEN, count, example_tokens


def _kitchen():
    house = D.build_house({"kind": "bundook_projected", "seed": 50_000_000})
    return house, room_model(house, "kitchen")


def test_tools_are_needle_shaped():
    house, room = _kitchen()
    tools = room_tools(house, "kitchen", room)
    names = [t["name"] for t in tools]
    assert len(names) == len(set(names)) and "read_kitchen" in names and "set_lights" in names
    for t in tools:
        assert set(t) == {"name", "description", "parameters"} and t["description"]
        for p in t["parameters"]["properties"].values():
            enum = p.get("enum") or (p.get("items") or {}).get("enum")
            assert enum is None or (enum and len(enum) == len(set(enum)))
    lights = next(t for t in tools if t["name"] == "set_lights")["parameters"]["properties"]["lights"]["items"]["enum"]
    assert "island pendant" in lights and not any(n.startswith("light.") for n in lights)


def test_execute_except_and_relative():
    house, room = _kitchen()
    r = leaf_execute(house, "kitchen", {"name": "set_lights", "arguments": {"action": "off", "except": ["island pendant"]}}, room)
    ids = r["calls"][0]["target"]["entity_id"]
    assert r["calls"][0]["service"] == "turn_off" and "light.kitchen_downlights" in ids and "light.kitchen_island_pendant" not in ids
    e = house.entities["light.kitchen_downlights"]
    e.state, e.attributes["brightness_pct"] = "on", 60
    apply_service(house, "light", "turn_on", {"entity_id": [e.entity_id]}, {"brightness_step_pct": -25})
    assert e.attributes["brightness_pct"] == 35
    oc = outcome(house, "kitchen", [{"name": "read_kitchen", "arguments": {"what": "temperature"}}])
    assert oc["reads"] and not oc["changes"]
    bad = apply_leaf(house, "kitchen", [{"name": "set_lights", "arguments": {"lights": ["lounge lamp"], "action": "on"}}])
    assert bad["errors"]


def test_generated_examples_execute_and_fit():
    rows, metas = D.gen_generic("train", 120, seed=5)
    assert len(rows) == 120 and any(not r["answers"] for r in rows) and any(len(r["answers"]) == 2 for r in rows)
    for r, m in zip(rows, metas):
        assert set(r) == {"query", "tools", "answers", "reasoning", "system"} and r["reasoning"]
        assert 1 <= len(r["tools"]) <= D.MAX_TOOLS and example_tokens(r) <= MAX_LEN
        names = {t["name"] for t in r["tools"]}
        assert all(a["name"] in names for a in r["answers"])
        house = D.build_house(m["house"])
        assert not outcome(house, m["area_id"], r["answers"])["errors"], r


def test_token_counter():
    assert count("turn on the kitchen lights") == 6
    assert count("<s><|im_start|>user\n") == 4


if __name__ == "__main__":
    test_tools_are_needle_shaped()
    test_execute_except_and_relative()
    test_generated_examples_execute_and_fit()
    test_token_counter()
    print("leaf schema, executor and generator ok")
