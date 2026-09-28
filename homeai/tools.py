"""Tool definitions (OpenAI/Ollama function-calling format) and their execution
against a House. Shared by data generation, the eval harness and the mock bridge."""
from __future__ import annotations

import json
from typing import Any

from .house.schema import CLASSES, House
from .house.sim import apply_service
from .house.snapshot import format_lines

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": (
                "Read the current state of devices and sensors. Filter by areas (area ids from the "
                "inventory, or the groups security, energy, water, weather) and/or classes (light, switch, fan, "
                "cover, garage, lock, alarm, climate, media_player, vacuum, camera, motion, occupancy, door, "
                "window, temperature, humidity, appliance, pump, irrigation, water, vehicle, battery, solar, load, power, forecast, weather, sun, person). "
                "An entity is returned when it matches an area filter (if given) and a class filter (if given). "
                "Fetch only what the request needs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "areas": {"type": "array", "items": {"type": "string"}},
                    "classes": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_service",
            "description": (
                "Call a Home Assistant service, e.g. light.turn_on, light.turn_off, switch.turn_off, "
                "fan.turn_on, cover.open_cover, cover.close_cover, lock.lock, lock.unlock, "
                "climate.set_temperature, media_player.media_pause, alarm_control_panel.alarm_arm_home, "
                "alarm_control_panel.alarm_arm_away, alarm_control_panel.alarm_disarm, vacuum.start."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "service": {"type": "string"},
                    "target": {
                        "type": "object",
                        "properties": {
                            "entity_id": {"type": "array", "items": {"type": "string"}},
                            "area_id": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "data": {
                        "type": "object",
                        "description": "Service data, e.g. brightness_pct, temperature, position, volume_level, hvac_mode",
                    },
                },
                "required": ["domain", "service", "target"],
            },
        },
    },
]


def get_state_text(house: House, areas=(), classes=()) -> str:
    areas = list(areas or [])
    classes = list(classes or [])
    bad = [c for c in classes if c not in CLASSES]
    known = set(house.all_area_keys())
    bad_areas = [a for a in areas if a not in known]
    ents = house.select(areas, classes)
    # config / diagnostic entities (HA entity_category, big houses only) stay out unless their class is asked for
    ents = [e for e in ents if not e.attributes.get("entity_category") or e.cls in classes]
    text = format_lines(ents, house)
    notes = []
    if bad:
        notes.append(f"unknown classes ignored: {', '.join(bad)}")
    if bad_areas:
        notes.append(f"unknown areas ignored: {', '.join(bad_areas)}")
    if notes:
        text += "\n(" + "; ".join(notes) + ")"
    return text


def execute_tool(house: House, name: str, args: dict[str, Any]) -> str:
    """Execute a tool call against the simulated house and return the tool message content."""
    if name == "get_state":
        return get_state_text(house, args.get("areas") or [], args.get("classes") or [])
    if name == "call_service":
        res = apply_service(house, args.get("domain", ""), args.get("service", ""),
                            args.get("target") or {}, args.get("data") or {})
        return json.dumps(res)
    return json.dumps({"success": False, "error": f"unknown tool {name}"})
