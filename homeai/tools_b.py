"""Design B toolset: a rooms-only system prompt (area ids plus per-class device counts, no device names) and
bounded lookup tools, for houses too big to list every device (thousands to tens of thousands of entities).

    find_devices(areas?, classes?, name?)                           what exists: entity_id, name, class (no state), capped
    get_state(areas?, classes?, name?, entity_ids?, where?, sort?, limit?)   live state, filtered and sorted at the tool
                                                                    layer, capped with a count and a hint to narrow
    call_service(...)                                               unchanged from design A (homeai/tools.py)

Design A (homeai/tools.py: TOOLS, execute_tool, homeai/house/snapshot.py: system_prompt) is untouched; an example
opts into B with "toolset": "b" and everything that runs the model (e2e, run_eval, the bridge) picks the prompt,
schemas and executor with for_toolset()."""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from .house.schema import CLASSES, CONTROLLABLE, Entity, House
from .house.sim import apply_service
from .house.snapshot import CLASS_ORDER, describe_state, format_lines, rel_time, system_prompt, system_prompt_b
from .tools import TOOLS, execute_tool

FIND_CAP = 40     # rows find_devices returns
STATE_ROWS = 50   # rows get_state returns unless limit says otherwise
STATE_MAX = 100   # largest limit honoured
STATE_CHARS = 8000  # hard cap on a get_state result, whatever the limit

WHERE_WORDS = ("on", "off", "open", "closed", "locked", "unlocked", "playing", "paused", "armed", "disarmed", "home", "away",
               "unavailable")
WHERE_ALIASES = {"active": "on", "running": "on", "detected": "on", "occupied": "on", "clear": "off", "shut": "closed",
                 "not_home": "away"}
SORTS = ("power_desc", "value_desc", "value_asc", "recent")

TOOLS_B: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "find_devices",
            "description": (
                "Look up devices and sensors without their state: returns entity_id, name and class, grouped by area "
                f"(at most {FIND_CAP}). Filter by areas (area ids, or the groups security, energy, water, weather), classes "
                "and/or name (words in the device name, e.g. \"desk lamp\"). Use it to get the entity_id of a named device "
                "before acting on it, and to check whether something exists or which rooms have it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "areas": {"type": "array", "items": {"type": "string"}},
                    "classes": {"type": "array", "items": {"type": "string"}},
                    "name": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": (
                "Read the current state of devices and sensors. Every filter given must match: areas (area ids or the groups "
                "security, energy, water, weather), classes (light, switch, fan, cover, garage, lock, alarm, climate, "
                "media_player, vacuum, camera, motion, occupancy, door, window, temperature, humidity, appliance, pump, "
                "irrigation, water, vehicle, battery, solar, load, power, forecast, weather, sun, person, illuminance, "
                "air_quality, meter, measurement, detection, safety), name (words in the name), entity_ids, and where: keep "
                "entities in any of these states: on (switched on, running or playing), off, open, closed, locked, unlocked, "
                "playing, paused, armed, disarmed, home, away, or a comparison like power>100 or value<18. sort: power_desc, "
                "value_desc, value_asc or recent (last changed first); limit: rows to return (default "
                f"{STATE_ROWS}). Long results are cut off with a count: narrow the request rather than repeat it. "
                "Config and diagnostic entities only with class device."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "areas": {"type": "array", "items": {"type": "string"}},
                    "classes": {"type": "array", "items": {"type": "string"}},
                    "name": {"type": "string"},
                    "entity_ids": {"type": "array", "items": {"type": "string"}},
                    "where": {"type": "array", "items": {"type": "string"}},
                    "sort": {"type": "string", "enum": list(SORTS)},
                    "limit": {"type": "integer"},
                },
            },
        },
    },
    TOOLS[1],  # call_service, shared with design A
]


# ---------------------------------------------------------------------- matching
def _as_list(v) -> list:
    if v is None or v == "":
        return []
    if isinstance(v, (str, int, float)):
        return [str(v)]
    return [str(x) for x in v]


STOP = {"the", "a", "an", "my", "our", "all", "any"}


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _tokens(s: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", (s or "").lower())]


def name_matches(house: House, e: Entity, query: str) -> bool:
    """Every word of the query starts a word of the entity's name, object id or class ('lamps' finds 'Desk Lamp');
    numbers must match whole ('spot 1' is not 'Spot 10')."""
    q = [t for t in _tokens(query) if t not in STOP]
    if not q:
        return True
    hay = set(_tokens(e.name) + _tokens(e.entity_id.split(".", 1)[-1]) + _tokens(e.cls.replace("_", " ")) + [e.cls])
    return all(t in hay if t.isdigit() else any(h.startswith(t) for h in hay) for t in q)


def primary(e: Entity) -> bool:
    """What B shows by default: no config / diagnostic entities and none of the device-housekeeping class."""
    return not e.attributes.get("entity_category") and e.cls != "device"


def _select(house: House, areas, classes, name, entity_ids):
    areas, classes, ids = _as_list(areas), _as_list(classes), _as_list(entity_ids)
    notes = []
    bad = [c for c in classes if c not in CLASSES]
    if bad:
        notes.append(f"unknown classes ignored: {', '.join(bad)}")
    known = set(house.all_area_keys())
    bad_areas = [a for a in areas if a not in known]
    if bad_areas:
        notes.append(f"unknown areas ignored: {', '.join(bad_areas)}")
    bad_ids = [i for i in ids if i not in house.entities]
    if bad_ids:
        notes.append(f"unknown entity_ids: {', '.join(bad_ids)}")
    want_device = "device" in classes
    out = []
    if ids:
        pool = [house.entities[i] for i in ids if i in house.entities]
    else:
        pool = house.entities.values()
    for e in pool:
        if areas and e.area not in areas:
            continue
        if classes and e.cls not in classes:
            continue
        if not ids and not want_device and not primary(e):
            continue
        if name and not name_matches(house, e, str(name)):
            continue
        out.append(e)
    return out, notes


# ---------------------------------------------------------------------- where / sort
BINARY = {"motion", "occupancy", "door", "window", "detection", "safety"}
SWITCHED = {"light", "switch", "fan", "pump", "irrigation"}


def _float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def power_reading(e: Entity) -> Optional[float]:
    """Watts the entity reports: a device's own draw (switches, appliances, pumps), or a power/load sensor's value."""
    a = e.attributes
    if e.cls in ("switch", "appliance", "pump", "light", "fan") and a.get("power_w") is not None:
        return float(a["power_w"]) if e.state == "on" else 0.0
    if e.cls in ("power", "load") or (e.cls == "solar" and a.get("kind") == "power") or a.get("unit") == "W" \
            or (e.cls == "battery" and a.get("kind") == "power"):
        return _float(e.state)
    return None


def value_reading(e: Entity) -> Optional[float]:
    if e.cls == "climate":
        return _float(e.attributes.get("current_temperature"))
    return _float(e.state)


def is_on(e: Entity) -> Optional[bool]:
    """On / off where the idea applies (None where it does not, e.g. a lock or a temperature)."""
    c, s, a = e.cls, e.state, e.attributes
    if c in SWITCHED or c in BINARY or (e.domain in ("binary_sensor", "input_boolean") and s in ("on", "off")):
        return s == "on"
    if c == "media_player":
        return s in ("playing", "paused", "on")
    if c == "climate":
        return s not in ("off", "unavailable", "unknown")
    if c == "appliance":
        return a.get("status", s) == "running" if "status" in a else s == "on"
    if c == "vacuum":
        return s == "cleaning"
    if c in ("cover", "garage"):
        return s == "open"
    return None


def is_open(e: Entity) -> Optional[bool]:
    if e.cls in ("door", "window"):
        return e.state == "on"
    if e.cls in ("cover", "garage"):
        return e.state == "open"
    if e.cls == "appliance" and e.attributes.get("status") == "door open":
        return True
    return None


CMP = re.compile(r"^\s*([a-z_]*)\s*(>=|<=|>|<|=)\s*(-?[0-9.]+)\s*$")


def _cond(word: str):
    """A where term -> predicate(entity) -> bool, or None if the term is not understood."""
    w = word.strip().lower().replace(" ", "_")
    w = WHERE_ALIASES.get(w, w)
    m = CMP.match(w)
    if m:
        key, op, val = m.group(1) or "value", m.group(2), float(m.group(3))
        read = power_reading if key == "power" else value_reading
        ops = {">": lambda x: x > val, "<": lambda x: x < val, ">=": lambda x: x >= val, "<=": lambda x: x <= val, "=": lambda x: x == val}
        return lambda e: (lambda x: x is not None and ops[op](x))(read(e))
    if w == "on":
        return lambda e: is_on(e) is True
    if w == "off":
        return lambda e: is_on(e) is False
    if w == "open":
        return lambda e: is_open(e) is True
    if w == "closed":
        return lambda e: is_open(e) is False
    if w in ("locked", "unlocked"):
        return lambda e: e.cls == "lock" and e.state == w
    if w in ("playing", "paused"):
        return lambda e: e.cls == "media_player" and e.state == w
    if w == "armed":
        return lambda e: e.cls == "alarm" and e.state.startswith("armed")
    if w == "disarmed":
        return lambda e: e.cls == "alarm" and e.state == "disarmed"
    if w == "home":
        return lambda e: e.cls == "person" and e.state == "home"
    if w == "away":
        return lambda e: e.cls == "person" and e.state != "home"
    if w == "unavailable":
        return lambda e: e.state in ("unavailable", "unknown")
    return None


# ---------------------------------------------------------------------- rendering
def _class_label(e: Entity) -> str:
    return e.cls + (" [MA]" if e.cls == "media_player" and e.attributes.get("ma") else "")


def _order(house: House):
    order = {k: i for i, k in enumerate(house.all_area_keys())}
    corder = {c: i for i, c in enumerate(CLASS_ORDER)}
    return lambda e: (order.get(e.area, 999), corder.get(e.cls, 99), e.name)


def find_devices_text(house: House, areas=(), classes=(), name: str = "") -> str:
    ents, notes = _select(house, areas, classes, name, ())
    ents.sort(key=_order(house))
    shown = ents[:FIND_CAP]
    if not shown:
        text = "No matching devices."
    else:
        out, cur = [], None
        for e in shown:
            if e.area != cur:
                cur = e.area
                out.append(f"[{house.area_name(cur)}]")
            out.append(f"{e.entity_id} | {e.name} | {_class_label(e)}")
        text = "\n".join(out)
        if len(ents) > len(shown):
            text += f"\n... {len(ents) - len(shown)} more; narrow by areas, classes or name"
    if notes:
        text += "\n(" + "; ".join(notes) + ")"
    return text


def get_state_b_text(house: House, areas=(), classes=(), name: str = "", entity_ids=(), where=(), sort: str = "",
                     limit=None) -> str:
    ents, notes = _select(house, areas, classes, name, entity_ids)
    conds = []
    for w in _as_list(where):
        f = _cond(w)
        if f is None:
            notes.append(f"unknown where ignored: {w}")
        else:
            conds.append(f)
    if conds:
        ents = [e for e in ents if any(f(e) for f in conds)]
    sort = (sort or "").strip().lower()
    if sort and sort not in SORTS:
        notes.append(f"unknown sort ignored: {sort}")
        sort = ""
    try:
        cap = max(1, min(STATE_MAX, int(limit))) if limit not in (None, "") else None
    except (TypeError, ValueError):
        cap = None
    rows = cap or STATE_ROWS
    if sort:
        if sort == "recent":
            ents.sort(key=lambda e: e.changed_s)
        else:
            read = power_reading if sort == "power_desc" else value_reading
            sign = 1 if sort == "value_asc" else -1
            ents.sort(key=lambda e: (read(e) is None, sign * (read(e) or 0.0)))
        shown = ents[:rows]
        lines = [f"{e.entity_id} | {e.name} | {house.area_name(e.area)} | {describe_state(e)} | {rel_time(e.changed_s)}" for e in shown]
        text = "\n".join(lines) if lines else "No matching entities."
        while len(text) > STATE_CHARS and len(shown) > 1:
            shown = shown[:-1]
            text = "\n".join(lines[:len(shown)])
    else:
        ents.sort(key=_order(house))
        shown = ents[:rows]
        text = format_lines(shown, house)
        while len(text) > STATE_CHARS and len(shown) > 1:
            shown = shown[:max(1, int(len(shown) * STATE_CHARS / len(text)) - 1)]
            text = format_lines(shown, house)
    if len(shown) < len(ents):
        if cap and len(shown) == cap:
            text += f"\n(showing {len(shown)} of {len(ents)} matches)"
        else:  # the rest by area, so whole-house questions still see every room involved
            rest: dict[str, int] = {}
            for e in ents[len(shown):]:
                rest[e.area] = rest.get(e.area, 0) + 1
            order = {k: i for i, k in enumerate(house.all_area_keys())}
            where_rest = ", ".join(f"{house.area_name(a)} {n}" for a, n in sorted(rest.items(), key=lambda kv: order.get(kv[0], 999)))
            text += (f"\n(showing {len(shown)} of {len(ents)} matches; {len(ents) - len(shown)} more in {where_rest}; "
                     "narrow with areas, classes, name or where)")
    if notes:
        text += "\n(" + "; ".join(notes) + ")"
    return text


def execute_tool_b(house: House, name: str, args: dict[str, Any]) -> str:
    """Design-B tool call against the simulated house (call_service is design A's)."""
    args = args if isinstance(args, dict) else {}
    if name == "find_devices":
        return find_devices_text(house, args.get("areas"), args.get("classes"), args.get("name") or "")
    if name == "get_state":
        return get_state_b_text(house, args.get("areas"), args.get("classes"), args.get("name") or "", args.get("entity_ids"),
                                args.get("where"), args.get("sort") or "", args.get("limit"))
    if name == "call_service":
        res = apply_service(house, args.get("domain", ""), args.get("service", ""), args.get("target") or {}, args.get("data") or {})
        return json.dumps(res)
    return json.dumps({"success": False, "error": f"unknown tool {name}"})


def for_toolset(toolset: Optional[str]):
    """(tool schemas, executor, system prompt builder) for an example's "toolset" ("a" when absent). "orch": the design-C
    orchestrator (homeai/tiered/schema.py: B's tools plus ask_rooms, which only homeai.tiered.pipeline can execute)."""
    if toolset == "orch":
        from .tiered.schema import TOOLS_ORCH, execute_tool_orch, system_prompt_orch
        return TOOLS_ORCH, execute_tool_orch, system_prompt_orch
    if (toolset or "a") == "b":
        return TOOLS_B, execute_tool_b, system_prompt_b
    return TOOLS, execute_tool, system_prompt


ID_LINE = re.compile(r"^([a-z_]+\.[a-z0-9_]+) \|", re.M)


def ids_in(text: str) -> set[str]:
    """Entity ids a find_devices / get_state result shows."""
    return set(ID_LINE.findall(text or ""))


def truncated(text: str) -> bool:
    """True when a result was cut by the default cap (not by an explicit limit): the caller should have narrowed."""
    return "narrow with areas" in (text or "") or "more; narrow by" in (text or "")


__all__ = ["TOOLS_B", "execute_tool_b", "find_devices_text", "get_state_b_text", "for_toolset", "ids_in", "truncated",
           "name_matches", "power_reading", "is_on", "is_open", "CONTROLLABLE"]
