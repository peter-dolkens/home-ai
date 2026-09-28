"""Design B (homeai/tools_b.py): bounded lookup tools, rooms-only prompt, and B transcripts that pair with their A twins.

    PYTHONPATH=. python3 tests/test_tools_b.py"""
from __future__ import annotations

import json
import random
import sys

sys.path.insert(0, ".")

from homeai.datagen.bigbench import build_one  # noqa: E402
from homeai.datagen.build import generate_one  # noqa: E402
from homeai.datagen.common import Conv  # noqa: E402
from homeai.house.bighouse import _pristine  # noqa: E402
from homeai.house.generator import make_house  # noqa: E402
from homeai.house.snapshot import system_prompt, system_prompt_b  # noqa: E402
from homeai.tools_b import STATE_CHARS, execute_tool_b, find_devices_text, get_state_b_text, ids_in, name_matches  # noqa: E402


def said(ex):
    return [(m["role"], m["content"], m.get("_eval")) for m in ex["messages"][1:] if m["role"] == "user" or (m["role"] == "assistant" and m.get("content"))]


def test_prompt_b_has_no_device_names():
    h = make_house(5, "bundook")
    a, b = system_prompt(h, None), system_prompt_b(h, None)
    assert "Downlights" in a and "Downlights" not in b
    assert "[kitchen]" in b and "light " in b and "find_devices" in b
    assert len(b) < len(a)


def test_tools_on_big_house():
    h = _pristine("random", 9_100_001, 20000)
    big = get_state_b_text(h, classes=["light"])
    assert len(big) <= STATE_CHARS + 2000 and "narrow with areas" in big and "more in" in big
    top = get_state_b_text(h, classes=["power"], sort="power_desc", limit=5)
    watts = [float(line.split(" | ")[3].split()[0]) for line in top.splitlines() if " | " in line]
    assert len(watts) == 5 and watts == sorted(watts, reverse=True) and "(showing 5 of" in top
    on = get_state_b_text(h, classes=["lock"], where=["unlocked"])
    assert all(h.entities[i].state == "unlocked" for i in ids_in(on))
    f = find_devices_text(h, classes=["light"])
    assert len(ids_in(f)) == 40 and "more; narrow by" in f
    assert "unknown where ignored: sideways" in get_state_b_text(h, classes=["lock"], where=["sideways"])
    e = next(x for x in h.entities.values() if x.name.endswith("Spot 1"))
    assert name_matches(h, e, "spot 1") and not name_matches(h, e, "spot 10")
    assert json.loads(execute_tool_b(h, "nope", {}))["success"] is False


def test_b_transcripts_pair_with_a():
    for i in range(60):
        a = generate_one(i, 5, None, "bundook", 0.4)
        b = generate_one(i, 5, None, "bundook", 0.4, toolset="b", strict_b=False)
        assert b["toolset"] == "b" and "toolset" not in a
        assert said(a) == said(b), (i, a["category"])
        assert all(len(m["content"]) <= STATE_CHARS + 2000 for m in b["messages"] if m["role"] == "tool" and m["name"] != "call_service")


def test_big_b_transcript_pairs_with_a():
    for scen in ("check_house", "anything_on", "clarify"):
        rows = []
        for ts in ("a", "b"):
            Conv.toolset = ts
            try:
                rows.append(build_one("random", 3000, 9_200_000, scen, random.Random(1), 0))
            finally:
                Conv.toolset = "a"
        assert said(rows[0]) == said(rows[1]), scen


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("design B tools and transcripts OK")
