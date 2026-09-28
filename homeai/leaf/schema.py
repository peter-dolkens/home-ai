"""Room leaves for the tiered design: a planner splits a spoken request into per-room instructions and a tiny per-room
model (Cactus Compute's Needle 3) turns one instruction into grammar-constrained calls on ITS room's tools, or [] when
the instruction is not for its room or no tool fits.

    room_tools(house, area_id)          the room's toolset, Needle style (name, description, JSON-schema parameters)
    room_model(house, area_id)          the same room as families of named devices (what data.py phrases requests about)
    leaf_execute(house, area_id, call)  one leaf call -> the Home Assistant service calls it stands for, or a read
    apply_leaf(house, area_id, calls)   execute and apply them to the simulated house (homeai/house/sim.py)
    end_state(house, area_id)           the room's controllable state, for end-state scoring

Schema choices, following cactuscompute.com/blog/designing-tools-for-needle:
  - every argument is a span of the instruction, so devices are enums of the names people say ("island pendant",
    "blind 2"), never entity ids; the description lists each name with up to two aliases and leaf_execute maps names
    to entity ids. Leaving the device list out means every device of that kind in the room ("lights off"), and
    `except` takes the ones to leave alone ("all the lights except the pendants"), so neither needs an invented list.
  - one tool per kind of device, polarity as an enum of the spoken words (on / off, open / close, start / stop),
    relative asks as enum words too (dimmer / brighter, warmer / cooler, louder / quieter); numbers carry their bounds
    in the schema (brightness 1-100, temperature 16-30, position 0-100, volume 0-100).
  - one read tool per room, read_<area_id>(what), what an enum of the readings and devices the room actually has.
  - only PRIMARY entities (no entity_category, not the device-housekeeping class); Hue group members are reached
    through their group; fridges and freezers are readable but not switchable.
Rooms with more than five tools engage Needle's retrieval (top five by description embedding), so each description
names the room and its devices."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..datagen.common import short
from ..house.schema import Entity, House
from ..house.sim import apply_service
from ..tools_b import get_state_b_text, primary

# ---------------------------------------------------------------------- spoken names
# name pattern -> aliases (said instead of the name); first two go in the tool description
ALIASES: list[tuple[str, list[str]]] = [
    (r"^downlights?$", ["ceiling lights", "main lights", "big light"]),
    (r"^ceiling$", ["ceiling light", "main light", "big light"]),
    (r"^island pendants?$", ["pendants", "island lights"]),
    (r"^pendants?$", ["pendant lights", "hanging lights"]),
    (r"^bench (strip|lights)$", ["bench lights", "under cabinet lights", "bench strip"]),
    (r"^toe kick strip$", ["kickboard lights", "plinth lights"]),
    (r"^floor (led )?strip$", ["floor strip", "floor lights"]),
    (r"^bulkhead (led )?strip$", ["bulkhead strip", "bulkhead lights"]),
    (r"^rgbww strip (\d+)$", ["strip {1}", "colour strip {1}"]),
    (r"^cove strip$", ["cove lights", "cove lighting"]),
    (r"^strip light$", ["led strip", "strip"]),
    (r"^track (lights|accent|wash|spot|low fill|high fill)$", ["track lights"]),
    (r"^wall (lights|washers)$", ["wall lights", "wall lamps"]),
    (r"^bedside lamps?$", ["bedside lights", "lamps"]),
    (r"^(desk|floor) lamp$", ["lamp"]),
    (r"^mirror lights?$", ["mirror light", "vanity lights"]),
    (r"^night light$", ["nightlight"]),
    (r"^floodlights?$", ["floods", "security lights"]),
    (r"^batten(s| lights)$", ["battens", "shed lights"]),
    (r"^garden spots$", ["spotlights", "garden lights"]),
    (r"^path lights$", ["path lighting", "garden path lights"]),
    (r"^table feature lights$", ["table lights", "feature lights"]),
    (r"^sitting area feature lights$", ["sitting area lights", "lounge feature lights"]),
    (r"^fan light$", ["light on the fan"]),
    (r"^pantry light$", ["pantry light"]),
    (r"^blinds$", ["shades", "blind"]),
    (r"^highlight windows$", ["high windows", "clerestory windows"]),
    (r"^windows$", ["window", "louvres"]),
    (r"^air conditioning( \d+)?$", ["aircon", "air con", "ac"]),
    (r"^house thermostat$", ["thermostat", "heating"]),
    (r"^floor heating$", ["underfloor heating", "heated floor"]),
    (r"^fan$", ["ceiling fan"]),
    (r"^exhaust fan$", ["exhaust", "extractor fan"]),
    (r"^homepod$", ["speaker", "music"]),
    (r"^speaker$", ["homepod", "music"]),
    (r"^tv$", ["telly", "television"]),
    (r"^dishwasher$", ["dishy"]),
    (r"^washing machine$", ["washer", "the wash"]),
    (r"^dryer$", ["tumble dryer", "clothes dryer"]),
    (r"^rangehood$", ["range hood", "extractor"]),
    (r"^cooktop$", ["stove", "hob"]),
    (r"^towel rail$", ["heated towel rail", "towel warmer"]),
    (r"^robot vacuum$", ["vacuum", "robot"]),
    (r"^garage door$", ["garage"]),
    (r"^electric blanket$", ["blanket"]),
    (r"^coffee machine$", ["coffee maker"]),
]


def aliases(name: str) -> list[str]:
    for pat, al in ALIASES:
        m = re.match(pat, name)
        if m:
            out = [a.format(*([m.group(0)] + list(m.groups()))) for a in al]
            return [a for a in out if a != name]
    return []


def spoken(e: Entity, house: House) -> str:
    n = short(e, house)
    if e.cls == "garage" and "garage" not in n:
        n = "garage " + n if n == "door" else n
    return n


@dataclass
class Dev:
    name: str               # the enum value
    ids: list[str]          # entity ids it acts on (a light group: the group and its members)
    aliases: list[str] = field(default_factory=list)
    kind: str = ""          # climate: "aircon" | "thermostat" | "floor"; media: "ma" | "tv" | "player"


@dataclass
class Family:
    tool: str               # tool name
    kind: str               # lights | blinds | windows | garage | aircon | floor_heating | fans | media | volume | music | switches | appliances | vacuum | read
    devs: list[Dev]


@dataclass
class Room:
    house: House
    area_id: str
    name: str               # display name ("Office (Pete)")
    said: str               # how people say it ("Pete's office", "kitchen")
    families: dict[str, Family]
    reads: dict[str, list[str]]  # read_<room> what -> entity ids

    def fam(self, kind: str) -> Optional[Family]:
        return self.families.get(kind)


def _dedupe(devs: list[Dev]) -> list[Dev]:
    seen: dict[str, int] = {}
    for d in devs:
        k = seen.get(d.name, 0)
        seen[d.name] = k + 1
        if k:
            d.name = f"{d.name} {k + 1}"
    return devs


def room_model(house: House, area_id: str) -> Room:
    ents = [e for e in house.in_area(area_id) if primary(e)]
    a = house.area(area_id)
    said = (a.spoken or a.name).lower() if a else area_id
    said = re.sub(r"^(\w+)'s ", lambda m: m.group(1).capitalize() + "'s ", said)
    fams: dict[str, Family] = {}

    def dev(e: Entity, kind: str = "", ids: Optional[list[str]] = None) -> Dev:
        n = spoken(e, house)
        return Dev(n, ids or [e.entity_id], aliases(n), kind)

    lights = [e for e in ents if e.domain == "light" and e.cls == "light"]
    members = {m for e in lights for m in (e.attributes.get("entity_id") or [])}
    ldevs = [dev(e, ids=[e.entity_id] + list(e.attributes.get("entity_id") or [])) for e in lights if e.entity_id not in members]
    if ldevs:
        fams["lights"] = Family("set_lights", "lights", _dedupe(ldevs))
    covers = [e for e in ents if e.domain == "cover" and e.cls == "cover"]
    blinds = [dev(e) for e in covers if "window" not in spoken(e, house)]
    wins = [dev(e) for e in covers if "window" in spoken(e, house)]
    if blinds:
        fams["blinds"] = Family("set_blinds", "blinds", _dedupe(blinds))
    if wins:
        fams["windows"] = Family("set_windows", "windows", _dedupe(wins))
    garages = [dev(e) for e in ents if e.cls == "garage"]
    if garages:
        fams["garage"] = Family("garage_door", "garage", _dedupe(garages))
    clim = [e for e in ents if e.domain == "climate"]
    ac = [dev(e, "thermostat" if "thermostat" in e.name.lower() else "aircon") for e in clim if not e.attributes.get("floor")]
    floor = [dev(e, "floor") for e in clim if e.attributes.get("floor")]
    if ac:
        fams["aircon"] = Family("set_thermostat" if all(d.kind == "thermostat" for d in ac) else "set_aircon", "aircon", _dedupe(ac))
    if floor:
        fams["floor_heating"] = Family("set_floor_heating", "floor_heating", _dedupe(floor))
    fans = [dev(e) for e in ents if e.domain == "fan" and e.cls == "fan"]
    if fans:
        fams["fans"] = Family("set_fans", "fans", _dedupe(fans))
    players = [dev(e, "ma" if e.attributes.get("ma") else "tv" if "tv" in e.name.lower() else "player") for e in ents if e.domain == "media_player"]
    if players:
        players = _dedupe(players)
        fams["media"] = Family("media_control", "media", players)
        fams["volume"] = Family("set_volume", "volume", players)
        ma = [d for d in players if d.kind == "ma"]
        if ma:
            fams["music"] = Family("play_music", "music", ma)
    sw = [dev(e) for e in ents if e.domain == "switch" and e.cls == "switch"]
    if sw:
        fams["switches"] = Family("switch_devices", "switches", _dedupe(sw))
    apps = [dev(e) for e in ents if e.cls == "appliance" and e.attributes.get("appliance") not in ("fridge", "freezer")]
    if apps:
        fams["appliances"] = Family("run_appliance", "appliances", _dedupe(apps))
    vac = [dev(e) for e in ents if e.cls == "vacuum"]
    if vac:
        fams["vacuum"] = Family("robot_vacuum", "vacuum", vac[:1])

    # readings: what -> entity ids, only what the room has
    reads: dict[str, list[str]] = {}

    def add(what: str, ids: list[str]):
        if ids and what not in reads:
            reads[what] = ids

    by = lambda *cls: [e.entity_id for e in ents if e.cls in cls]  # noqa: E731
    add("presence", by("occupancy", "motion"))
    add("temperature", by("temperature") + [e.entity_id for e in clim])
    add("humidity", by("humidity"))
    add("lights", [e.entity_id for e in lights if e.entity_id not in members])
    add("blinds", [i for d in blinds for i in d.ids])
    add("windows", by("window") + [i for d in wins for i in d.ids])
    add("doors", by("door") + [i for d in garages for i in d.ids])
    add("air quality", by("air_quality"))
    add("light level", by("illuminance"))
    add("power use", by("power"))
    add("smoke and leaks", by("safety"))
    for fam_kind in ("aircon", "floor_heating", "fans"):
        f = fams.get(fam_kind)
        if f:
            add({"aircon": "aircon" if f.tool == "set_aircon" else "thermostat", "floor_heating": "floor heating", "fans": "fans"}[fam_kind],
                [i for d in f.devs for i in d.ids])
    for d in players:
        add(d.name, d.ids)
    for d in sw + apps + [dev(e) for e in ents if e.cls == "appliance" and e.attributes.get("appliance") in ("fridge", "freezer")] + vac[:1]:
        add(d.name, d.ids)
    return Room(house, area_id, a.name if a else area_id, said, fams, reads)


# ---------------------------------------------------------------------- tool schemas
def _names(devs: list[Dev]) -> list[str]:
    return [d.name for d in devs]


def _listing(devs: list[Dev], n_alias: int = 2) -> str:
    out = []
    for d in devs:
        al = [a for a in d.aliases if a not in _names(devs)][:n_alias]
        out.append(f"{d.name} ({', '.join(al)})" if al else d.name)
    return ", ".join(out)


def _arr(devs: list[Dev], desc: str) -> dict:
    return {"type": "array", "items": {"type": "string", "enum": _names(devs)}, "description": desc}


def _enum(values: list[str], desc: str = "") -> dict:
    out: dict[str, Any] = {"type": "string", "enum": values}
    if desc:
        out["description"] = desc
    return out


def _int(lo: int, hi: int, desc: str) -> dict:
    return {"type": "integer", "minimum": lo, "maximum": hi, "description": desc}


def _tool(name: str, desc: str, props: dict, required: list[str] | None = None) -> dict:
    params: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        params["required"] = required
    return {"name": name, "description": desc, "parameters": params}


def family_tool(room: Room, f: Family) -> dict:
    r = room.said
    devs, many = f.devs, len(f.devs) > 1
    if f.kind == "lights":
        props = {
            "action": _enum(["on", "off", "dimmer", "brighter"], "on, off, or a bit dimmer / brighter"),
            "brightness": _int(1, 100, "percent"),
        }
        if many:
            props["lights"] = _arr(devs, "the lights named; leave out for every light here")
            props["except"] = _arr(devs, "lights to leave alone (after 'except' or 'but not')")
        return _tool(f.tool, f"Switch or dim lights in the {r}: {_listing(devs)}.", props)
    if f.kind == "blinds":
        props = {"action": _enum(["open", "close", "stop"]), "position": _int(0, 100, "percent open")}
        if many:
            props["blinds"] = _arr(devs, "the blinds named; leave out for all of them")
        return _tool(f.tool, f"Open, close or set blinds in the {r}: {_listing(devs, 1)}.", props)
    if f.kind == "windows":
        props = {"action": _enum(["open", "close"]), "position": _int(0, 100, "percent open")}
        if many:
            props["windows"] = _arr(devs, "the windows named; leave out for all")
        return _tool(f.tool, f"Open or close the motorised windows in the {r}: {_listing(devs)}.", props)
    if f.kind == "garage":
        props = {"action": _enum(["open", "close"])}
        if many:
            props["door"] = _enum(_names(devs))
        return _tool(f.tool, f"Open or close the {_listing(devs, 0)} in the {r}.", props, ["action"])
    if f.kind == "aircon":
        what = "thermostat" if f.tool == "set_thermostat" else "air conditioning"
        props = {
            "action": _enum(["on", "off", "warmer", "cooler"]),
            "mode": _enum(["heat", "cool", "auto", "dry", "fan only"]),
            "temperature": {"type": "number", "minimum": 16, "maximum": 30, "description": "target in degrees C"},
        }
        if many:
            props["unit"] = _enum(_names(devs))
            called = _listing(devs)
        else:
            called = ", ".join(devs[0].aliases[:2]) or devs[0].name
        return _tool(f.tool, f"Control the {r} {what} ({called}): on, off, mode, target temperature, warmer or cooler.", props)
    if f.kind == "floor_heating":
        props = {"action": _enum(["on", "off", "warmer", "cooler"]),
                 "temperature": {"type": "number", "minimum": 15, "maximum": 30, "description": "target in degrees C"}}
        return _tool(f.tool, f"Control the {r} floor heating (underfloor heating): on, off, target temperature, warmer or cooler.", props)
    if f.kind == "fans":
        props = {"action": _enum(["on", "off"]), "speed": _enum(["low", "medium", "high"])}
        if many:
            props["fans"] = _arr(devs, "the fans named; leave out for all")
        return _tool(f.tool, f"Switch fans in the {r} or set their speed: {_listing(devs)}.", props)
    if f.kind == "media":
        props = {"action": _enum(["play", "pause", "stop", "next", "previous", "turn on", "turn off", "mute", "unmute"])}
        if many:
            props["player"] = _enum(_names(devs))
        return _tool(f.tool, f"Play, pause, skip, mute or switch the {r} media players: {_listing(devs)}.", props, ["action"])
    if f.kind == "volume":
        props = {"volume": _int(0, 100, "percent"), "change": _enum(["louder", "quieter"])}
        if many:
            props["player"] = _enum(_names(devs))
        return _tool(f.tool, f"Set the volume in the {r}, or make it louder or quieter: {_listing(devs, 1)}.", props)
    if f.kind == "music":
        props = {"query": {"type": "string", "description": "the artist, album, song, playlist, station or genre as said"},
                 "type": _enum(["artist", "album", "track", "playlist", "radio", "genre"])}
        if many:
            props["player"] = _enum(_names(devs))
        return _tool(f.tool, f"Play music on the {r} speaker ({_listing(devs, 1)}): an artist, album, song, playlist, radio station or genre.",
                     props, ["query"])
    if f.kind == "switches":
        props = {"action": _enum(["on", "off"]), "devices": _arr(devs, "the devices named")}
        return _tool(f.tool, f"Turn plugs and switched devices in the {r} on or off: {_listing(devs, 1)}.", props, ["action", "devices"])
    if f.kind == "appliances":
        props = {"action": _enum(["start", "stop", "on", "off"]), "appliance": _enum(_names(devs))}
        return _tool(f.tool, f"Start or stop an appliance in the {r}: {_listing(devs, 1)}.", props, ["action", "appliance"])
    if f.kind == "vacuum":
        props = {"action": _enum(["start", "stop", "dock"], "dock sends it home")}
        return _tool(f.tool, "Start, stop or dock the robot vacuum.", props, ["action"])
    raise ValueError(f.kind)


def read_tool(room: Room) -> dict:
    whats = list(room.reads)
    return _tool(f"read_{room.area_id}", f"Read the {room.said}: {', '.join(whats)}.",
                 {"what": _enum(whats, "what to read")}, ["what"])


def room_tools(house: House, area_id: str, room: Optional[Room] = None) -> list[dict]:
    room = room or room_model(house, area_id)
    tools = [family_tool(room, f) for f in room.families.values()]
    if room.reads:
        tools.append(read_tool(room))
    return tools


def system_fact(room: Room) -> str:
    """The session fact the leaf runs with (Needle's --system: facts, not instructions)."""
    if room.said.lower() != room.name.lower():
        return f"room: {room.name}, called {room.said}"
    return f"room: {room.name}"


# ---------------------------------------------------------------------- execution
SPEED = {"low": 33, "medium": 67, "high": 100}
STEP = 25  # brightness_step_pct for dimmer / brighter


def _ids(f: Family, names, exclude=None) -> tuple[list[str], list[str]]:
    by = {d.name: d for d in f.devs}
    errors = []
    names = [names] if isinstance(names, str) else list(names or [])
    chosen = [by[n] for n in names if n in by] if names else list(f.devs)
    errors += [f"unknown {f.kind} {n!r}" for n in names if n not in by]
    ex = [exclude] if isinstance(exclude, str) else list(exclude or [])
    errors += [f"unknown {f.kind} {n!r}" for n in ex if n not in by]
    drop = {i for n in ex if n in by for i in by[n].ids}
    return [i for d in chosen for i in d.ids if i not in drop], errors


def _call(domain: str, service: str, ids: list[str], data: Optional[dict] = None) -> dict:
    c: dict[str, Any] = {"domain": domain, "service": service, "target": {"entity_id": ids}}
    if data:
        c["data"] = data
    return c


def leaf_execute(house: House, area_id: str, call: dict, room: Optional[Room] = None) -> dict:
    """One leaf call -> {"calls": [HA service calls]} or {"read": {"what", "entity_ids", "text"}}, plus "error" when the call
    cannot be carried out (unknown tool, device or argument combination)."""
    room = room or room_model(house, area_id)
    name, args = call.get("name", ""), call.get("arguments") or {}
    if not isinstance(args, dict):
        return {"calls": [], "error": "arguments must be an object"}
    if name == f"read_{area_id}":
        what = args.get("what")
        ids = room.reads.get(what)
        if not ids:
            return {"calls": [], "error": f"nothing to read for {what!r}"}
        return {"calls": [], "read": {"what": what, "entity_ids": ids, "text": get_state_b_text(house, entity_ids=ids)}}
    f = next((f for f in room.families.values() if f.tool == name), None)
    if f is None:
        return {"calls": [], "error": f"unknown tool {name}"}
    act = args.get("action")
    out: list[dict] = []
    err: list[str] = []
    k = f.kind
    if k == "lights":
        ids, err = _ids(f, args.get("lights"), args.get("except"))
        if ids:
            if args.get("brightness") is not None:
                out.append(_call("light", "turn_on", ids, {"brightness_pct": int(args["brightness"])}))
            elif act in ("on", "off"):
                out.append(_call("light", f"turn_{act}", ids))
            elif act in ("dimmer", "brighter"):
                out.append(_call("light", "turn_on", ids, {"brightness_step_pct": -STEP if act == "dimmer" else STEP}))
            else:
                err.append("set_lights needs an action or a brightness")
    elif k in ("blinds", "windows"):
        ids, err = _ids(f, args.get(k))
        if ids:
            if args.get("position") is not None:
                out.append(_call("cover", "set_cover_position", ids, {"position": int(args["position"])}))
            elif act in ("open", "close", "stop"):
                out.append(_call("cover", {"open": "open_cover", "close": "close_cover", "stop": "stop_cover"}[act], ids))
            else:
                err.append(f"{name} needs an action or a position")
    elif k == "garage":
        ids, err = _ids(f, args.get("door"))
        if act in ("open", "close") and ids:
            out.append(_call("cover", f"{act}_cover", ids))
        elif not err:
            err.append("garage_door needs open or close")
    elif k in ("aircon", "floor_heating"):
        ids, err = _ids(f, args.get("unit"))
        mode = args.get("mode")
        temp = args.get("temperature")
        if ids:
            if temp is not None:
                data = {"temperature": float(temp)}
                if mode:
                    data["hvac_mode"] = mode.replace(" ", "_")
                out.append(_call("climate", "set_temperature", ids, data))
            elif mode:
                out.append(_call("climate", "set_hvac_mode", ids, {"hvac_mode": mode.replace(" ", "_")}))
            elif act == "off":
                out.append(_call("climate", "turn_off", ids))
            elif act == "on":
                out.append(_call("climate", "set_hvac_mode", ids, {"hvac_mode": "heat"}) if k == "floor_heating" else _call("climate", "turn_on", ids))
            elif act in ("warmer", "cooler"):
                for i in ids:
                    cur = float(house.entities[i].attributes.get("temperature", 21.0))
                    out.append(_call("climate", "set_temperature", [i], {"temperature": cur + (1.0 if act == "warmer" else -1.0)}))
            else:
                err.append(f"{name} needs an action, mode or temperature")
    elif k == "fans":
        ids, err = _ids(f, args.get("fans"))
        if ids:
            if args.get("speed") in SPEED:
                out.append(_call("fan", "set_percentage", ids, {"percentage": SPEED[args["speed"]]}))
            elif act in ("on", "off"):
                out.append(_call("fan", f"turn_{act}", ids))
            else:
                err.append("set_fans needs an action or a speed")
    elif k == "media":
        ids, err = _ids(f, args.get("player"))
        svc = {"play": ("media_play", None), "pause": ("media_pause", None), "stop": ("media_stop", None), "next": ("media_next_track", None),
               "previous": ("media_previous_track", None), "turn on": ("turn_on", None), "turn off": ("turn_off", None),
               "mute": ("volume_mute", {"is_volume_muted": True}), "unmute": ("volume_mute", {"is_volume_muted": False})}.get(act)
        if ids and svc:
            out.append(_call("media_player", svc[0], ids, svc[1]))
        elif ids:
            err.append(f"unknown media action {act!r}")
    elif k == "volume":
        ids, err = _ids(f, args.get("player"))
        if ids:
            if args.get("volume") is not None:
                out.append(_call("media_player", "volume_set", ids, {"volume_level": round(int(args["volume"]) / 100, 2)}))
            elif args.get("change") in ("louder", "quieter"):
                out.append(_call("media_player", "volume_up" if args["change"] == "louder" else "volume_down", ids))
            else:
                err.append("set_volume needs a volume or a change")
    elif k == "music":
        ids, err = _ids(f, args.get("player"))
        q = str(args.get("query") or "").strip()
        if ids and q:
            t = args.get("type")
            data: dict[str, Any] = {"media_id": q}
            if t == "genre":
                data.update(media_type="playlist", radio_mode=True)
            elif t:
                data["media_type"] = t
            out.append(_call("music_assistant", "play_media", ids, data))
        elif ids:
            err.append("play_music needs a query")
    elif k in ("switches", "appliances"):
        ids, err = _ids(f, args.get("devices" if k == "switches" else "appliance"))
        on = act in ("on", "start")
        if act in ("on", "off", "start", "stop") and ids and (args.get("devices") or args.get("appliance")):
            out.append(_call("switch", "turn_on" if on else "turn_off", ids))
        elif not err:
            err.append(f"{name} needs an action and a device")
    elif k == "vacuum":
        ids, err = _ids(f, None)
        svc = {"start": "start", "stop": "stop", "dock": "return_to_base"}.get(act)
        if svc:
            out.append(_call("vacuum", svc, ids))
        else:
            err.append("robot_vacuum needs start, stop or dock")
    res: dict[str, Any] = {"calls": out}
    if err:
        res["error"] = "; ".join(err)
    return res


def apply_leaf(house: House, area_id: str, calls: list[dict], room: Optional[Room] = None) -> dict:
    """Execute a leaf's calls in order against the house (mutated). Returns {"ha": [...], "reads": [...], "errors": [...]}."""
    room = room or room_model(house, area_id)
    ha, reads, errors = [], [], []
    for c in calls or []:
        if not isinstance(c, dict):
            errors.append("call is not an object")
            continue
        r = leaf_execute(house, area_id, c, room)
        if r.get("error"):
            errors.append(r["error"])
        if r.get("read"):
            reads.append(r["read"])
        for hc in r["calls"]:
            res = apply_service(house, hc["domain"], hc["service"], hc["target"], hc.get("data"))
            ha.append(hc)
            if not res.get("success"):
                errors.append(res.get("error") or "; ".join(res.get("warnings", [])))
    return {"ha": ha, "reads": reads, "errors": errors}


STATE_KEYS = ("brightness_pct", "position", "temperature", "percentage", "volume_pct", "muted", "media_title", "media_content_type", "status")


def end_state(house: House, area_id: str) -> dict[str, list]:
    """The room's controllable entities: entity_id -> [state, {attr: value}] for the attributes service calls change."""
    out = {}
    for e in house.in_area(area_id):
        if e.domain in ("light", "cover", "climate", "fan", "media_player", "switch", "vacuum") and not e.attributes.get("entity_category"):
            out[e.entity_id] = [e.state, {k: e.attributes[k] for k in STATE_KEYS if k in e.attributes}]
    return out


def outcome(house: House, area_id: str, calls: list[dict], room: Optional[Room] = None) -> dict:
    """What a leaf's calls do to a copy of the house: changed entities (after state), reads, HA calls, errors."""
    h = copy.copy(house)  # leaf calls only touch the room's own entities (by id): copy just those
    h.entities = dict(house.entities)
    for e in house.in_area(area_id):
        h.entities[e.entity_id] = copy.deepcopy(e)
    before = end_state(h, area_id)
    res = apply_leaf(h, area_id, calls, room_model(h, area_id) if room is None else _rebind(room, h))
    after = end_state(h, area_id)
    changed = {k: v for k, v in after.items() if before.get(k) != v}
    return {"changes": changed, "reads": sorted({i for r in res["reads"] for i in r["entity_ids"]}), "ha": res["ha"], "errors": res["errors"]}


def _rebind(room: Room, house: House) -> Room:
    r = copy.copy(room)
    r.house = house
    return r
