"""In-memory simulation of Home Assistant service calls. Used to produce tool
results during data generation and to back the mock bridge."""
from __future__ import annotations

from typing import Any

from .schema import House


def resolve_target(house: House, domain: str, target: dict) -> tuple[list[str], list[str]]:
    """Return (entity_ids, errors)."""
    ids: list[str] = []
    errors: list[str] = []
    target = target or {}
    if not isinstance(target, dict):  # malformed call (small models emit target as a bare string): HA rejects it
        return [], [f"target must be an object with entity_id / area_id, got {type(target).__name__}"]
    if domain == "music_assistant":
        domain = "media_player"  # MA services target media_player entities
    for eid in _as_list(target.get("entity_id")):
        if eid in house.entities:
            if domain != "homeassistant" and house.entities[eid].domain != domain:
                errors.append(f"{eid} is not a {domain} entity")
            elif eid not in ids:
                ids.append(eid)
        else:
            errors.append(f"unknown entity_id {eid}")
    for aid in _as_list(target.get("area_id")):
        if aid not in house.area_ids():
            errors.append(f"unknown area_id {aid}")
            continue
        for e in house.in_area(aid):
            if e.attributes.get("entity_category"):
                continue  # like HA, area targets skip config / diagnostic entities
            if (domain == "homeassistant" or e.domain == domain) and e.entity_id not in ids:
                ids.append(e.entity_id)
    return ids, errors


def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return list(v)


def apply_service(house: House, domain: str, service: str, target: dict, data: dict | None = None) -> dict[str, Any]:
    data = data or {}
    ids, errors = resolve_target(house, domain, target)
    if domain == "music_assistant":
        bad = [i for i in ids if not house.entities[i].attributes.get("ma")]
        if bad:
            errors.append(f"not Music Assistant players: {', '.join(bad)}")
            ids = [i for i in ids if i not in bad]
        domain = "media_player"
        if service == "play_media":
            data = {"media_content_id": data.get("media_id"), "media_content_type": ("genre" if data.get("radio_mode") else data.get("media_type", "track"))}
    if errors and not ids:
        return {"success": False, "error": "; ".join(errors)}
    if not ids:
        return {"success": False, "error": f"no {domain} entities matched the target"}
    changed = []
    for eid in ids:
        e = house.entities[eid]
        d = e.domain if domain == "homeassistant" else domain
        ok = _apply_one(e, d, service, data)
        if ok:
            e.changed_s = 0
            changed.append(eid)
        else:
            errors.append(f"{d}.{service} not supported for {eid}")
    out: dict[str, Any] = {"success": bool(changed), "changed": changed}
    if errors:
        out["warnings"] = errors
    return out


def _apply_one(e, domain: str, service: str, data: dict) -> bool:
    a = e.attributes
    if domain in ("light", "switch", "fan"):
        if service == "turn_on" and domain == "light" and "brightness_step_pct" in data and not ({"brightness_pct", "brightness"} & set(data)):
            # relative dimming, as HA does it: a step from the current level; stepping to 0 turns the light off
            cur = a.get("brightness_pct", 100) if e.state == "on" else 0
            new = max(0, min(100, int(cur) + int(data["brightness_step_pct"])))
            if new == 0:
                e.state = "off"
                a.pop("brightness_pct", None)
            else:
                e.state, a["brightness_pct"] = "on", new
            return True
        if service == "turn_on":
            e.state = "on"
            if e.cls == "appliance":
                a["status"] = "running"
            if e.cls == "irrigation":
                a["remaining_min"] = int(data.get("duration", a.get("default_min", 20)))
            if domain == "light":
                if "brightness_pct" in data:
                    a["brightness_pct"] = int(data["brightness_pct"])
                elif "brightness" in data:
                    a["brightness_pct"] = round(int(data["brightness"]) / 255 * 100)
                elif "brightness_pct" not in a:
                    a["brightness_pct"] = 100
            if domain == "fan" and "percentage" in data:
                a["percentage"] = int(data["percentage"])
            return True
        if service == "turn_off":
            e.state = "off"
            if e.cls == "appliance":
                a["status"] = "off"
                a.pop("remaining_min", None)
            if e.cls == "irrigation":
                a.pop("remaining_min", None)
            return True
        if service == "toggle":
            e.state = "off" if e.state == "on" else "on"
            return True
        if domain == "fan" and service == "set_percentage":
            a["percentage"] = int(data.get("percentage", 50))
            e.state = "on" if a["percentage"] > 0 else "off"
            return True
        return False
    if domain == "cover":
        if service == "open_cover":
            e.state, a["position"] = "open", 100
            return True
        if service == "close_cover":
            e.state, a["position"] = "closed", 0
            return True
        if service == "set_cover_position":
            pos = int(data.get("position", 50))
            a["position"] = pos
            e.state = "closed" if pos == 0 else "open"
            return True
        if service == "stop_cover":
            return True
        return False
    if domain == "lock":
        if service in ("lock", "unlock"):
            e.state = "locked" if service == "lock" else "unlocked"
            return True
        return False
    if domain == "climate":
        if service == "set_temperature":
            a["temperature"] = float(data.get("temperature", a.get("temperature", 21)))
            if e.state == "off":
                e.state = data.get("hvac_mode", "heat")
            return True
        if service == "set_hvac_mode":
            e.state = data.get("hvac_mode", "heat")
            return True
        if service == "turn_off":
            e.state = "off"
            return True
        if service == "turn_on":
            e.state = a.get("last_mode", "heat")
            return True
        return False
    if domain == "media_player":
        if service == "play_media":
            e.state = "playing"
            a["media_title"] = data.get("media_content_id", a.get("media_title"))
            a["media_content_type"] = data.get("media_content_type", "music")
            a.pop("media_artist", None)
            return True
        if service in ("media_next_track", "media_previous_track"):
            if e.state in ("playing", "paused"):
                a["media_title"] = ("next track" if service == "media_next_track" else "previous track")
            return True
        if service == "media_seek":
            return True
        if service == "shuffle_set":
            a["shuffle"] = bool(data.get("shuffle", True))
            return True
        if service == "repeat_set":
            a["repeat"] = data.get("repeat", "all")
            return True
        if service == "join":
            a["group_members"] = list(data.get("group_members", []))
            return True
        if service == "unjoin":
            a.pop("group_members", None)
            return True
        if service in ("media_pause",):
            if e.state == "playing":
                e.state = "paused"
            return True
        if service in ("media_play",):
            e.state = "playing"
            return True
        if service == "media_play_pause":
            e.state = "paused" if e.state == "playing" else "playing"
            return True
        if service in ("media_stop", "turn_off"):
            e.state = "off"
            a.pop("media_title", None)
            return True
        if service == "turn_on":
            if e.state == "off":
                e.state = "idle"
            return True
        if service == "volume_set":
            a["volume_pct"] = round(float(data.get("volume_level", 0.3)) * 100)
            return True
        if service == "volume_up":
            a["volume_pct"] = min(100, a.get("volume_pct", 30) + 10)
            return True
        if service == "volume_down":
            a["volume_pct"] = max(0, a.get("volume_pct", 30) - 10)
            return True
        if service == "volume_mute":
            a["muted"] = bool(data.get("is_volume_muted", True))
            return True
        return False
    if domain == "alarm_control_panel":
        m = {"alarm_arm_home": "armed_home", "alarm_arm_away": "armed_away",
             "alarm_arm_night": "armed_night", "alarm_disarm": "disarmed"}
        if service in m:
            e.state = m[service]
            return True
        return False
    if domain == "vacuum":
        if service == "start":
            e.state = "cleaning"
            return True
        if service in ("return_to_base", "stop"):
            e.state = "docked" if service == "return_to_base" else "idle"
            return True
        return False
    return False
