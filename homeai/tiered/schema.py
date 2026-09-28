"""Designs C / D: an orchestrator with the design-B prompt and tools plus ask_rooms, which hands short per-room
instructions to small room leaves (homeai/leaf/schema.py toolsets) and gets back a compact per-room result.

    ASK_ROOMS / TOOLS_ORCH          the orchestrator's tool schemas (B's find_devices, get_state, call_service + ask_rooms)
    system_prompt_orch(house, sp)   the design-B prompt with rules for routing to rooms (design C)
    system_prompt_fanout(house, sp) the same for blind fan-out (design D: every room already saw the utterance)
    leaf_rooms(house)               real areas that have a leaf toolset (the groups stay with the orchestrator)
    phrase_calls(room, calls)       leaf calls -> the short, normalised instruction that asks for them ("lights off")
    room_result(house, room, ...)   what a leaf did / read, as one line of the ask_rooms result

The pipeline (homeai/tiered/pipeline.py) and the orchestrator training data (homeai/tiered/data.py) render ask_rooms
results with the same functions, so what the orchestrator trains on is what it gets at runtime."""
from __future__ import annotations

import copy
import json
from typing import Any, Optional

from ..datagen.common import join, short
from ..house.schema import GROUPS, House, Speaker
from ..house.snapshot import SYSTEM_RULES_B, describe_state, inventory_b, people_line, rel_time, speaker_line
from ..leaf.schema import Family, Room, room_model, room_tools
from ..tools_b import TOOLS_B, execute_tool_b

ASK_ROOMS: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ask_rooms",
        "description": (
            "Hand instructions to the rooms' own assistants, which control and read the devices in their room. One short "
            "instruction per room in plain words, no entity ids: \"lights off\", \"desk lamp on\", \"lights to 40%\", "
            "\"set the aircon to 22\", \"close the blinds\", \"play the artist Missy Higgins\", \"read presence and temperature\". "
            "Ask only the rooms the request is for, all in one call. Returns what each room did or read, or that it did nothing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "requests": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"area_id": {"type": "string"}, "instruction": {"type": "string"}},
                        "required": ["area_id", "instruction"],
                    },
                },
            },
            "required": ["requests"],
        },
    },
}
TOOLS_ORCH: list[dict[str, Any]] = TOOLS_B + [ASK_ROOMS]

_RULES_B = SYSTEM_RULES_B.split("\n")
SYSTEM_RULES_ORCH = "\n".join([
    _RULES_B[0],
    "- The list above has each area's id and how many devices of each class it holds, not their names or live values. Every "
    "room has its own assistant for the devices in it: to change them call ask_rooms with one short instruction per room "
    "(\"lights off\", \"desk lamp on\", \"set the aircon to 22\"), naming devices the way people say them. For a named device "
    "whose room you do not know, find it with find_devices first. Also use find_devices to check whether something exists or "
    "which rooms have it.",
    _RULES_B[2],
    "- Use call_service (by entity_id) only for devices in Security, Energy, Water and Weather, and for what a room assistant "
    "cannot do: locks, grouping players, shuffle and repeat, playing on players not marked [MA]. Never say something is done "
    "unless the room or the call reports it done.",
    *_RULES_B[4:],
])
SYSTEM_RULES_FANOUT = "\n".join([
    _RULES_B[0],
    "- The list above has each area's id and how many devices of each class it holds, not their names or live values. Every "
    "room has its own assistant for the devices in it, and each has already been given the request: their results come "
    "as an ask_rooms result. Answer from those results; rooms that did nothing are not listed.",
    _RULES_B[2],
    "- Use call_service (by entity_id) only for devices in Security, Energy, Water and Weather, and for what a room assistant "
    "cannot do: locks, grouping players, shuffle and repeat, playing on players not marked [MA]. Never say something is done "
    "unless a room or the call reports it done.",
    *_RULES_B[4:],
])


def _prompt(house: House, speaker: Optional[Speaker], rules: str) -> str:
    return "\n".join([
        f"You are the voice assistant for {house.description} running Home Assistant.",
        "",
        f"Now: {house.now.strftime('%A %H:%M')}",
        speaker_line(house, speaker),
        f"People: {people_line(house)}",
        "",
        "Areas [id]: device classes with counts",
        inventory_b(house),
        "",
        rules,
    ])


def system_prompt_orch(house: House, speaker: Optional[Speaker]) -> str:
    return _prompt(house, speaker, SYSTEM_RULES_ORCH)


def system_prompt_fanout(house: House, speaker: Optional[Speaker]) -> str:
    return _prompt(house, speaker, SYSTEM_RULES_FANOUT)


def execute_tool_orch(house: House, name: str, args: dict[str, Any]) -> str:
    """for_toolset("orch")'s executor: the B tools. ask_rooms needs room leaves (homeai/tiered/pipeline.py)."""
    if name == "ask_rooms":
        return json.dumps({"success": False, "error": "ask_rooms runs room leaves: use homeai.tiered.pipeline"})
    return execute_tool_b(house, name, args)


# ---------------------------------------------------------------------- rooms
def leaf_rooms(house: House) -> list[str]:
    """Real areas with at least one leaf tool, in inventory order (groups are house level)."""
    return [a.area_id for a in house.areas if a.area_id not in GROUPS and room_tools(house, a.area_id)]


def room_copy(house: House, area_id: str) -> House:
    """A copy of the house whose own entities are only this room's (deep-copied); everything else is shared."""
    h = copy.copy(house)
    h.entities = dict(house.entities)
    for e in house.in_area(area_id):
        h.entities[e.entity_id] = copy.deepcopy(e)
    return h


def rebind(room: Room, house: House) -> Room:
    r = copy.copy(room)
    r.house = house
    return r


# ---------------------------------------------------------------------- instructions from leaf calls
NOUN = {"lights": "lights", "blinds": "blinds", "windows": "windows", "fans": "fan", "switches": "devices", "appliances": "appliances",
        "garage": "garage door", "media": "players", "volume": "players", "music": "speaker", "vacuum": "robot vacuum"}
ARG_KEY = {"lights": "lights", "blinds": "blinds", "windows": "windows", "fans": "fans", "switches": "devices", "appliances": "appliance",
           "aircon": "unit", "media": "player", "volume": "player", "music": "player", "garage": "door"}


def _family(room: Room, name: str) -> Optional[Family]:
    return next((f for f in room.families.values() if f.tool == name), None)


def _names(v) -> list[str]:
    return [v] if isinstance(v, str) else [str(x) for x in (v or [])]


def _num(x) -> str:
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return str(x)


def phrase_call(room: Room, call: dict) -> str:
    """One leaf call -> a short instruction that states every argument as a span (the leaf data's evidence rule)."""
    name, a = call.get("name", ""), call.get("arguments") or {}
    if name.startswith("read_"):
        return f"read {a.get('what')}"
    f = _family(room, name)
    if f is None:
        return name
    k, act = f.kind, a.get("action")
    named = _names(a.get(ARG_KEY.get(k, "")))
    single = f.devs[0].name if len(f.devs) == 1 else None
    if k == "lights":
        tgt = join(named) if named else "lights"
        if a.get("except"):
            tgt = f"all lights except {join(_names(a['except']))}"
        if a.get("brightness") is not None:
            return f"{tgt} to {int(a['brightness'])}%"
        return f"{tgt} {act}"
    if k in ("blinds", "windows"):
        tgt = join(named) if named else NOUN[k]
        if a.get("position") is not None:
            return f"{tgt} to {int(a['position'])}%"
        return f"{act} the {tgt}"
    if k == "garage":
        return f"{act} the {named[0] if named else NOUN[k]}"
    if k in ("aircon", "floor_heating"):
        noun = named[0] if named else ("floor heating" if k == "floor_heating" else "thermostat" if f.tool == "set_thermostat" else "aircon")
        mode, temp = a.get("mode"), a.get("temperature")
        if temp is not None and mode:
            return f"set the {noun} to {_num(temp)} on {mode}"
        if temp is not None:
            return f"set the {noun} to {_num(temp)}"
        if mode:
            return f"{noun} to {mode} mode"
        return f"{noun} {act}"
    if k == "fans":
        tgt = join(named) if named else NOUN[k]
        if a.get("speed"):
            return f"{tgt} on {a['speed']}"
        return f"{tgt} {act}"
    if k in ("media", "volume", "music"):
        p = named[0] if named else single or "players"
        if k == "volume":
            if a.get("volume") is not None:
                return f"{p} volume to {int(a['volume'])}%"
            return f"make the {p} {a.get('change')}"
        if k == "music":
            t = a.get("type")
            q = a.get("query")
            if not t:
                return f"play {q}" + (f" on the {named[0]}" if named else "")
            return f"play the {'radio station' if t == 'radio' else t} {q}" + (f" on the {named[0]}" if named else "")
        if act in ("next", "previous"):
            return f"{act} track on the {p}"
        return f"{act} the {p}"
    if k == "switches":
        return f"{join(named)} {act}"
    if k == "appliances":
        return f"{act} the {named[0]}" if act in ("start", "stop") else f"{named[0]} {act}"
    if k == "vacuum":
        return f"{act} the robot vacuum"
    return name


def phrase_calls(room: Room, calls: list[dict]) -> str:
    """Several leaf calls for one room -> one instruction; consecutive reads merge ("read presence and temperature")."""
    parts: list[str] = []
    reads: list[str] = []
    for c in calls:
        if c.get("name", "").startswith("read_"):
            reads.append(str((c.get("arguments") or {}).get("what")))
            continue
        if reads:
            parts.append("read " + join(reads))
            reads = []
        parts.append(phrase_call(room, c))
    if reads:
        parts.append("read " + join(reads))
    return ", ".join(parts)


# ---------------------------------------------------------------------- results
def _dev_index(room: Room) -> dict[str, tuple[Family, Any]]:
    out = {}
    for f in room.families.values():
        if f.kind in ("volume", "music"):  # the same players as media
            continue
        for d in f.devs:
            for i in d.ids:
                out.setdefault(i, (f, d))
    return out


def done_text(house: House, room: Room, ha: list[dict]) -> str:
    """The devices the leaf's service calls acted on, in their state after the calls ("all lights off; desk lamp on, 40%")."""
    idx = _dev_index(room)
    seen: list[tuple[Family, Any]] = []
    for hc in ha:
        ids = (hc.get("target") or {}).get("entity_id") or []
        for i in ([ids] if isinstance(ids, str) else ids):
            fd = idx.get(i)
            if fd and all(fd[1] is not s[1] for s in seen):
                seen.append(fd)
    parts: list[str] = []
    by_fam: dict[int, list] = {}
    order: list[Family] = []
    for f, d in seen:
        if id(f) not in by_fam:
            by_fam[id(f)] = []
            order.append(f)
        by_fam[id(f)].append(d)
    for f in order:
        devs = by_fam[id(f)]
        states = [describe_state(house.entities[d.ids[0]]) for d in devs]
        if len(devs) > 1 and len(devs) == len(f.devs) and len(set(states)) == 1:
            parts.append(f"all {len(devs)} {NOUN.get(f.kind, f.kind) if f.kind != 'fans' else 'fans'} {states[0]}")
        else:
            parts += [f"{d.name} {s}" for d, s in zip(devs, states)]
    return "; ".join(parts)


def read_text(house: House, reads: list[dict]) -> str:
    ids: list[str] = []
    for r in reads:
        ids += [i for i in r["entity_ids"] if i not in ids]
    out = []
    for i in ids:
        e = house.entities[i]
        out.append(f"{short(e, house)} {describe_state(e)} ({rel_time(e.changed_s)})")
    return "; ".join(out)


def room_result(house: House, room: Room, calls: list, applied: Optional[dict], error: Optional[str] = None) -> str:
    """One room's line of the ask_rooms result. applied: homeai.leaf.schema.apply_leaf's return (None when nothing ran)."""
    head = f"{room.area_id}: "
    if error:
        return head + f"failed - {error}"
    if not calls or applied is None:
        return head + "no action"
    bits = []
    done = done_text(house, room, applied["ha"])
    if done:
        bits.append("done - " + done)
    if applied["reads"]:
        bits.append("read - " + read_text(house, applied["reads"]))
    if applied["errors"]:
        bits.append("failed - " + "; ".join(dict.fromkeys(applied["errors"])))
    return head + (" | ".join(bits) if bits else "no action")


def unknown_room_line(house: House, area_id: str) -> str:
    if area_id in GROUPS:
        return f"{area_id}: not a room; use call_service with entity_id"
    if area_id in house.area_ids():
        return f"{area_id}: no devices a room assistant can use"
    return f"{area_id}: unknown area"


__all__ = ["ASK_ROOMS", "TOOLS_ORCH", "system_prompt_orch", "system_prompt_fanout", "execute_tool_orch", "leaf_rooms", "room_copy",
           "rebind", "phrase_call", "phrase_calls", "room_result", "unknown_room_line", "room_model", "room_tools"]
