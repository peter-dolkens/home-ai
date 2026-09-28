"""Text rendering of house state: the compact inventory that goes in the system
prompt, and the detailed lines returned by get_state. The live bridge uses the
exact same functions, so what the model sees at runtime matches training."""
from __future__ import annotations

from typing import Iterable, Optional

from .schema import CONTROLLABLE, GROUPS, Entity, House, Speaker

CLASS_ORDER = [
    "light", "switch", "fan", "cover", "garage", "lock", "alarm", "climate",
    "media_player", "vacuum", "appliance", "pump", "irrigation", "camera", "motion", "occupancy", "door", "window",
    "temperature", "humidity", "water", "battery", "solar", "load", "power", "forecast", "vehicle",
    "weather", "sun", "person",
    "illuminance", "air_quality", "meter", "measurement", "detection", "safety", "device",
]


def rel_time(s: int) -> str:
    if s < 60:
        return "just now"
    m = s // 60
    if m < 60:
        return f"{m}m ago"
    h = m // 60
    if h < 24:
        mm = m % 60
        if h < 6 and mm >= 5:
            return f"{h}h {mm}m ago"
        return f"{h}h ago"
    return f"{h // 24}d ago"


def num(x, nd: int = 1) -> str:
    if x is None:
        return "?"
    try:
        s = f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def describe_state(e: Entity) -> str:
    a = e.attributes
    c = e.cls
    if c == "light":
        if e.state == "on":
            b = a.get("brightness_pct")
            return f"on, {b}%" if b is not None else "on"
        return e.state
    if c == "switch":
        if e.state == "on" and a.get("source_w") is not None:
            return f"on, supplying {a['source_w']} W"
        if e.state == "on" and a.get("power_w") is not None:
            return f"on, {a['power_w']} W"
        return e.state
    if c == "fan":
        if e.state == "on" and a.get("percentage") is not None:
            return f"on, {a['percentage']}%"
        return e.state
    if c in ("cover", "garage"):
        pos = a.get("position")
        if e.state == "open" and pos is not None and 0 < pos < 100:
            return f"open, {pos}%"
        return e.state
    if c == "lock":
        return e.state
    if c == "door" or c == "window":
        return "open" if e.state == "on" else "closed"
    if c == "motion":
        return "detected" if e.state == "on" else "clear"
    if c == "occupancy":
        return "occupied" if e.state == "on" else "clear"
    if c == "temperature":
        return f"{num(e.state)}°C"
    if c == "humidity":
        return f"{num(e.state, 0)}%"
    if c == "climate":
        cur = a.get("current_temperature")
        if e.state == "off":
            return f"off, currently {num(cur)}°C"
        return f"{e.state}, set {num(a.get('temperature'))}°C, currently {num(cur)}°C"
    if c == "media_player":
        if e.state in ("playing", "paused"):
            t, ar = a.get("media_title"), a.get("media_artist")
            v = a.get("volume_pct")
            s = e.state + (f' "{t}"' if t else "") + (f" by {ar}" if ar else "")
            if a.get("media_content_type") in ("playlist", "radio", "podcast", "album", "genre"):
                s += f" ({a['media_content_type']})"
            s += f", volume {v}%" if v is not None else ""
            if a.get("shuffle"):
                s += ", shuffle"
            if a.get("group_members"):
                s += f", grouped with {len(a['group_members'])} other"
            return s
        return e.state
    if c == "camera":
        return e.state
    if c == "alarm":
        return e.state
    if c == "vacuum":
        return e.state
    if c == "battery":
        k = a.get("kind")
        if k == "soc":
            return f"{num(e.state, 0)}%, {num(a.get('capacity_kwh'))} kWh capacity"
        if k == "power":
            return f"{num(e.state, 0)} W, {a.get('direction')}"
        if k == "remaining":
            return f"{num(e.state)} kWh remaining"
        if k == "runtime":
            return e.state if e.state == "n/a (charging)" else f"{num(e.state)} h at current load"
        if k == "vehicle":
            return f"{num(e.state, 0)}%, {a.get('location', 'unknown location')}"
        return e.state
    if c == "solar":
        return f"{num(e.state, 0)} W" if a.get("kind") == "power" else f"{num(e.state)} kWh today"
    if c == "load" or c == "power":
        return f"{num(e.state, 0)} W"
    if c == "forecast":
        return f"{num(e.state)} kWh " + ("(rest of today)" if a.get("kind") == "remaining_today" else "(tomorrow)")
    if c == "weather":
        f = a.get("forecast_tomorrow") or {}
        s = f"{e.state}, {num(a.get('temperature'))}°C, humidity {num(a.get('humidity'), 0)}%, wind {num(a.get('wind_kmh'), 0)} km/h"
        if f:
            s += f"; tomorrow {f.get('condition')} {num(f.get('low'))}–{num(f.get('high'))}°C"
        return s
    if c == "sun":
        if e.state == "above_horizon":
            return f"up, sets {a.get('next_setting')}"
        return f"down, rises {a.get('next_rising')}"
    if c == "person":
        return "home" if e.state == "home" else "away"
    if c == "appliance":
        st = a.get("status", e.state)
        if st == "running":
            rem = a.get("remaining_min")
            prog = a.get("program")
            return "running" + (f", {prog}" if prog else "") + (f", {rem} min left" if rem is not None else "")
        if st == "finished":
            return "finished" + (f" {rel_time(e.changed_s)}" if e.changed_s else "")
        if st == "door open":
            return "door open"
        if a.get("power_w") and e.state == "on":
            return f"on, {a['power_w']} W"
        return st
    if c == "pump":
        if e.state == "on":
            return "running" + (f", {a['power_w']} W" if a.get("power_w") else "")
        return "off"
    if c == "irrigation":
        if e.state == "on":
            return "watering" + (f", {a['remaining_min']} min left" if a.get("remaining_min") is not None else "")
        return "off" + (f", last ran {rel_time(e.changed_s)}" if e.changed_s > 3600 else "")
    if c == "water":
        if a.get("kind") == "level":
            return f"{num(e.state, 0)}%" + (f", about {num(a['litres'], 0)} L" if a.get("litres") else "")
        if a.get("kind") == "pressure":
            return f"{num(e.state)} bar"
        if a.get("kind") == "flow":
            return f"{num(e.state)} L/min"
        return e.state
    if c == "vehicle":
        bits = [f"{num(e.state, 0)}%"]
        if a.get("range_km") is not None:
            bits.append(f"{num(a['range_km'], 0)} km range")
        bits.append(a.get("location", "location unknown"))
        if a.get("plugged") is not None:
            bits.append("plugged in" if a["plugged"] else "not plugged in")
        if a.get("locked") is not None:
            bits.append("locked" if a["locked"] else "unlocked")
        return ", ".join(bits)
    if c in EXTRA_CLASSES:
        return _describe_extra(e)
    return e.state


EXTRA_CLASSES = {"illuminance", "air_quality", "meter", "measurement", "detection", "safety", "device"}
BINARY_WORDS = {"leak": ("leak detected", "dry"), "connectivity": ("connected", "disconnected"),
                "problem": ("problem", "ok"), "update": ("update available", "up to date")}


def _describe_extra(e: Entity) -> str:
    """Classes only the big-house generators produce: a value with its unit, or a binary word."""
    a = e.attributes
    if e.domain in ("binary_sensor", "update") and e.state in ("on", "off"):
        on, off = BINARY_WORDS.get(a.get("kind"), ("detected", "clear"))
        return on if e.state == "on" else off
    if e.cls == "illuminance":
        return f"{num(e.state, 0)} lx"
    unit = a.get("unit")
    if unit:
        try:
            float(e.state)
        except (TypeError, ValueError):
            return e.state
        return f"{num(e.state, a.get('nd', 1))} {unit}"
    return e.state


def format_lines(entities: Iterable[Entity], house: House) -> str:
    """Detailed state grouped by area, in inventory order."""
    ents = list(entities)
    if not ents:
        return "No matching entities."
    order = {k: i for i, k in enumerate(house.all_area_keys())}
    corder = {c: i for i, c in enumerate(CLASS_ORDER)}
    ents.sort(key=lambda e: (order.get(e.area, 999), corder.get(e.cls, 99), e.name))
    out = []
    cur = None
    for e in ents:
        if e.area != cur:
            cur = e.area
            out.append(f"[{house.area_name(cur)}]")
        out.append(f"{e.entity_id} | {e.name} | {describe_state(e)} | {rel_time(e.changed_s)}")
    return "\n".join(out)


def short_name(e: Entity, area_name: str, *aliases: Optional[str]) -> str:
    n = e.name
    for prefix in (area_name, *[a for a in aliases if a]):
        if n.lower().startswith(prefix.lower() + " ") and len(n) > len(prefix) + 1:
            n = n[len(prefix) + 1:]
            break
    n = n.lstrip("-:– ").strip()  # HA renders device-less names as "Room - Light"
    return n or e.name


def inventory(house: House) -> str:
    lines = []
    corder = {c: i for i, c in enumerate(CLASS_ORDER)}
    for key in house.all_area_keys():
        ents = house.in_area(key)
        if not ents:
            continue
        title = house.area_name(key)
        ctrl: dict[str, list[str]] = {}
        sens: set[str] = set()
        a0 = house.area(key)
        for e in ents:
            if e.cls in CONTROLLABLE:
                nm = short_name(e, title, a0.spoken if a0 else None) if key not in GROUPS else e.name
                if e.cls == "media_player" and e.attributes.get("ma"):
                    nm += " [MA]"
                ctrl.setdefault(e.cls, []).append(nm)
            else:
                sens.add(e.cls)
        parts = [f"{c}({', '.join(sorted(set(n)))})" for c, n in sorted(ctrl.items(), key=lambda kv: corder.get(kv[0], 99))]
        a = house.area(key)
        head = f"- {title} [{key}]" + (f" (say: {a.spoken})" if a and a.spoken and a.spoken.lower() != title.lower() else "")
        s = f"{head}: " + ", ".join(parts) if parts else f"{head}:"
        if sens:
            s += ("; " if parts else " ") + "sensors: " + ", ".join(sorted(sens, key=lambda c: corder.get(c, 99)))
        lines.append(s)
    return "\n".join(lines)


def people_line(house: House) -> str:
    bits = []
    for p in house.people:
        if p.home:
            bits.append(f"{p.name} (home)")
        else:
            bits.append(f"{p.name} (away, {rel_time(p.changed_s).replace(' ago', '')})")
    return ", ".join(bits) if bits else "none configured"


def speaker_line(house: House, sp: Optional[Speaker]) -> str:
    if sp is None:
        return "Speaker: unknown"
    who = sp.name or "unknown"
    if sp.channel == "satellite" and sp.area:
        an = house.area_spoken(sp.area)
        if "'s " in an:  # "from Amber's office satellite", no article before a possessive
            return f"Speaker: {who}, from {an} satellite"
        return f"Speaker: {who}, from the {an.lower()} satellite"
    p = house.person(sp.name) if sp.name else None
    where = "home" if (p and p.home) else "away"
    return f"Speaker: {who}, via the app ({where})"


SYSTEM_RULES = """How to work:
- You only see the inventory above, not live values. Call get_state before answering anything about state, presence, power or weather, and before acting on a specific named device (you need its entity_id). Fetch only the areas and classes you need.
- To change anything call call_service. Target a whole room with area_id; target specific devices with entity_id. Devices in Security, Energy, Water and Weather must be targeted by entity_id. Never say something is done unless the call succeeded.
- Unlocking, disarming the alarm or opening the garage: ask for confirmation first unless the user has just confirmed.
- Music: players marked [MA] are Music Assistant players; start music on them with music_assistant.play_media (data: media_id, media_type one of artist/album/track/playlist/radio/podcast, radio_mode true for a genre or mood). Other players use media_player.play_media (media_content_type, media_content_id). Pause, skip, volume, shuffle and grouping use media_player services on every player.
- If a request is ambiguous ask one short question, but prefer the speaker's own room when that resolves it.
- To locate a person, reason from motion and occupancy sensors, who is home, and elimination. Say how sure you are.
- Be concise and natural; replies are spoken aloud. No lists, no markdown."""


def system_prompt(house: House, speaker: Optional[Speaker]) -> str:
    return "\n".join([
        f"You are the voice assistant for {house.description} running Home Assistant.",
        "",
        f"Now: {house.now.strftime('%A %H:%M')}",
        speaker_line(house, speaker),
        f"People: {people_line(house)}",
        "",
        "Areas [id]: devices",
        inventory(house),
        "",
        SYSTEM_RULES,
    ])


# ---------------------------------------------------------------------- design B (homeai/tools_b.py)
def inventory_b(house: House) -> str:
    """Areas only: id, spoken alias, and per area how many devices of each controllable class it has plus which kinds of
    sensor, counting primary entities (no config / diagnostic entities, no device-housekeeping class). No names."""
    lines = []
    corder = {c: i for i, c in enumerate(CLASS_ORDER)}
    for key in house.all_area_keys():
        ents = [e for e in house.in_area(key) if not e.attributes.get("entity_category") and e.cls != "device"]
        if not ents:
            continue
        counts: dict[str, int] = {}
        ma = 0
        sens: set[str] = set()
        for e in ents:
            if e.cls in CONTROLLABLE:
                counts[e.cls] = counts.get(e.cls, 0) + 1
                ma += e.cls == "media_player" and bool(e.attributes.get("ma"))
            else:
                sens.add(e.cls)
        parts = []
        for c, n in sorted(counts.items(), key=lambda kv: corder.get(kv[0], 99)):
            s = f"{c} {n}"
            if c == "media_player" and ma:
                s += " [MA]" if ma == n else f" [{ma} MA]"
            parts.append(s)
        a = house.area(key)
        title = house.area_name(key)
        head = f"- {title} [{key}]" + (f" (say: {a.spoken})" if a and a.spoken and a.spoken.lower() != title.lower() else "")
        s = f"{head}: " + ", ".join(parts) if parts else f"{head}:"
        if sens:
            s += ("; " if parts else " ") + "sensors: " + ", ".join(sorted(sens, key=lambda c: corder.get(c, 99)))
        lines.append(s)
    return "\n".join(lines)


SYSTEM_RULES_B = """How to work:
- The list above has each area's id and how many devices of each class it holds, not their names or live values. A room's own devices can be targeted by area_id without looking anything up. For a specific named device you need its entity_id: find it with find_devices (by area, class and/or name), then act. Also use find_devices to check whether something exists or which rooms have it.
- Call get_state before answering anything about state, presence, power or weather. Ask only for what the request needs: filter by areas and classes, use where (on, off, open, unlocked, power>100...) for questions about the whole house, and sort with a limit for "the most" or "the warmest". If a result says there are more, narrow it.
- To change anything call call_service. Target a whole room with area_id; target specific devices with entity_id. Devices in Security, Energy, Water and Weather must be targeted by entity_id. Never say something is done unless the call succeeded.
- Unlocking, disarming the alarm or opening the garage: ask for confirmation first unless the user has just confirmed.
- Music: players marked [MA] are Music Assistant players; start music on them with music_assistant.play_media (data: media_id, media_type one of artist/album/track/playlist/radio/podcast, radio_mode true for a genre or mood). Other players use media_player.play_media (media_content_type, media_content_id). Pause, skip, volume, shuffle and grouping use media_player services on every player.
- If a request is ambiguous ask one short question, but prefer the speaker's own room when that resolves it.
- To locate a person, reason from motion and occupancy sensors, who is home, and elimination. Say how sure you are.
- Be concise and natural; replies are spoken aloud. No lists, no markdown."""


def system_prompt_b(house: House, speaker: Optional[Speaker]) -> str:
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
        SYSTEM_RULES_B,
    ])
