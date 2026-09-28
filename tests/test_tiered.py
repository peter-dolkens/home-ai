"""Offline checks for the tiered pipeline (homeai/tiered/): orchestrator schema, service call -> leaf call translation,
ask_rooms / fan-out execution, the orchestrator data rewrite, and the harness sanity run: an oracle orchestrator
(replaying the rewritten reference) with an oracle leaf (the reference leaf calls) must pass every turn of a sample of
bigbench-b and test-v2-b through the outcome scoring.

    PYTHONPATH=. python3 tests/test_tiered.py"""
from __future__ import annotations

import json
import os

from homeai.house.bighouse import big_spec_house
from homeai.leaf.schema import room_model
from homeai.tiered.data import ha_to_leaf, to_orch, verify
from homeai.tiered.eval import evaluate
from homeai.tiered.pipeline import LeafPicker, OracleLeaf, Pipeline
from homeai.tiered.schema import TOOLS_ORCH, leaf_rooms, phrase_calls
from homeai.tools_b import for_toolset

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "generated")


def _bundook():
    return big_spec_house({"kind": "bundook_projected", "seed": 50_000_000})


def _rows(name, step, acting=False):
    """Every step-th conversation (acting: of those whose reference changes the house)."""
    with open(os.path.join(DATA, name)) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if acting:
        rows = [r for r in rows if any(tc["function"]["name"] == "call_service" for m in r["messages"] for tc in m.get("tool_calls") or [])]
    return rows[::step]


def test_orchestrator_schema():
    names = [t["function"]["name"] for t in TOOLS_ORCH]
    assert names == ["find_devices", "get_state", "call_service", "ask_rooms"]
    tools, execute, prompt = for_toolset("orch")
    assert tools is TOOLS_ORCH and "ask_rooms" in prompt(_bundook(), None)
    assert "error" in execute(_bundook(), "ask_rooms", {"requests": []})


def test_service_calls_translate_to_leaf_calls():
    house = _bundook()
    room = room_model(house, "kitchen")
    lights = [e.entity_id for e in house.in_area("kitchen") if e.domain == "light"]
    leaf, why = ha_to_leaf(room, "light", "turn_off", {}, lights)
    assert leaf == [{"name": "set_lights", "arguments": {"action": "off"}}], (leaf, why)
    assert phrase_calls(room, leaf) == "lights off"
    assert verify(house, room, "kitchen", "light", "turn_off", {}, lights, leaf, "kitchen lights off") in ("exact", "housekeeping")
    one = room.fam("lights").devs[0]
    leaf, _ = ha_to_leaf(room, "light", "turn_on", {"brightness_pct": 40}, one.ids[:1])
    assert leaf == [{"name": "set_lights", "arguments": {"brightness": 40, "lights": [one.name]}}]
    assert phrase_calls(room, leaf) == f"{one.name} to 40%"
    locks = [e.entity_id for e in house.entities.values() if e.domain == "lock"][:1]
    assert ha_to_leaf(room, "lock", "unlock", {}, locks)[0] is None


def test_ask_rooms_and_fanout():
    house = _bundook()
    room = room_model(house, "kitchen")
    gold = OracleLeaf({("kitchen", "lights off"): [{"name": "set_lights", "arguments": {"action": "off"}}]})
    script = [{"content": "", "tool_calls": [{"function": {"name": "ask_rooms", "arguments": {"requests": [
                  {"area_id": "kitchen", "instruction": "lights off"}, {"area_id": "security", "instruction": "lock up"},
                  {"area_id": "nowhere", "instruction": "lights off"}]}}}]},
              {"content": "Kitchen lights are off."}]

    class Scripted:
        def chat(self, messages, tools=None):
            return script.pop(0)

    pipe = Pipeline(Scripted(), LeafPicker(gold), "routed")
    text, calls, st = pipe.run([{"role": "system", "content": "x"}, {"role": "user", "content": "kitchen lights off"}], house, "kitchen lights off")
    assert text == "Kitchen lights are off." and st["orch_calls"] == 2 and st["leaf_calls"] == 1 and st["rooms_asked"] == 3
    assert all(house.entities[i].state == "off" for d in room.fam("lights").devs for i in d.ids)
    # fan-out: every room leaf gets the utterance; only the rooms that acted are listed
    fan = OracleLeaf({(a, "kitchen lights off"): [{"name": "set_lights", "arguments": {"action": "on"}}] for a in ("kitchen",)})
    answer = {"content": "Done."}

    class Once:
        def chat(self, messages, tools=None):
            assert messages[-1]["role"] == "tool" and messages[-1]["content"].startswith("kitchen: done")
            assert f"no action in the other {len(leaf_rooms(house)) - 1} rooms" in messages[-1]["content"]
            assert "ask_rooms" not in [t["function"]["name"] for t in tools]
            return answer

    text, _, st = Pipeline(Once(), LeafPicker(fan), "fanout").run([{"role": "system", "content": "x"}, {"role": "user", "content": "kitchen lights off"}],
                                                                   house, "kitchen lights off")
    assert text == "Done." and st["rooms_asked"] == len(leaf_rooms(house))


def test_rewrite_routes_room_calls():
    for ex in _rows("test-v2-b.jsonl", 10):
        conv = to_orch(ex)
        assert conv["toolset"] == "orch" and [m["content"] for m in conv["messages"] if m["role"] == "user"] == \
            [m["content"] for m in ex["messages"] if m["role"] == "user"]
        for m in conv["messages"]:
            for tc in m.get("tool_calls") or []:
                if tc["function"]["name"] == "ask_rooms":
                    assert all(r["instruction"] for r in tc["function"]["arguments"]["requests"])
        if ex["category"] == "control_area":
            assert any(tc["function"]["name"] == "ask_rooms" for m in conv["messages"] for tc in m.get("tool_calls") or []), ex["id"]


def test_oracle_pipeline_scores_full_marks():
    for name, step, acting in (("test-v2-b.jsonl", 5, False), ("bigbench-b.jsonl", 25, False), ("bigbench-b.jsonl", 7, True)):
        rep = evaluate(_rows(name, step, acting), progress=False)
        assert rep["errors"] == 0 and rep["history_errors"] == 0, rep
        assert rep["pass"] >= 0.99, (name, rep["pass"], rep["by_category"])
        print(f"  oracle {name}{' (acting)' if acting else ''}: {rep['examples']} conversations, {rep['turns']} turns, pass {rep['pass']:.1%}, "
              f"ask_rooms/turn {rep['routing']['ask_rooms_per_turn']}, rooms/turn {rep['routing']['rooms_asked_per_turn']}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("tiered pipeline, rewrite and oracle harness OK")
