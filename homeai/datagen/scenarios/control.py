"""Device control scenarios."""
from __future__ import annotations

import random
from typing import Optional

from ...house.schema import House, Speaker
from ..common import Conv, cap, is_are, join, pick, room_plain, room_ref, rooms_ref, short, voiceify

LIGHT_CLASS_WORDS = {"light": "lights", "fan": "fan", "cover": "blinds", "media_player": "tv"}


def _areas_with(house: House, cls: str, indoor_only=False) -> list[str]:
    out = []
    for a in house.areas:
        if indoor_only and a.kind == "outdoor":
            continue
        if house.in_area(a.area_id, cls):
            out.append(a.area_id)
    return out


def control_area(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """Whole-room control by area_id, no fetch needed. Lights mostly, plus fans, blinds and media."""
    cls = rng.choices(["light", "fan", "cover", "media_player"], weights=[0.7, 0.1, 0.12, 0.08])[0]
    areas = _areas_with(house, cls)
    if cls == "media_player":
        areas = [a for a in areas if len(house.in_area(a, cls)) == 1]
    if not areas:
        return None
    if cls == "light":
        k = min(len(areas), rng.choices([1, 2, 3], weights=[0.55, 0.32, 0.13])[0])
    else:
        k = 1
    chosen = rng.sample(areas, k)
    c = Conv(house, sp, "control_area", rng)
    rr = rooms_ref(house, chosen)
    if cls == "light":
        on = rng.random() < 0.55
        if on:
            u = pick(rng, ["Turn on the lights in {r}.", "Lights on in {r}.", "Switch on the lights in {r}.",
                           "Can you turn on the lights in {r}?", "Turn the lights on in {r}.",
                           "Put the lights on in {r}."], r=rr)
            if k == 1:
                u = rng.choice([u, f"Turn on the {room_plain(house, chosen[0])} lights.", f"Turn the {room_plain(house, chosen[0])} lights on.",
                                f"Turn on the {room_plain(house, chosen[0])} light."])
        else:
            u = pick(rng, ["Turn off the lights in {r}.", "Lights off in {r}.", "Switch off the lights in {r}.",
                           "Can you turn the lights off in {r}?", "Kill the lights in {r}.",
                           "Turn the lights off in {r}."], r=rr)
            if k == 1:
                u = rng.choice([u, f"Turn off the {room_plain(house, chosen[0])} lights.", f"Turn the {room_plain(house, chosen[0])} lights off.",
                                f"Turn off the {room_plain(house, chosen[0])} light."])
        c.user(voiceify(u, rng))
        c.act([{"domain": "light", "service": "turn_on" if on else "turn_off", "target": {"area_id": chosen}}])
        names = [room_plain(house, a) for a in chosen]
        if on:
            c.say(pick(rng, ["{R} lights are on.", "Done, lights on in {r}.", "Okay, {r} lights on.", "Lights are on in {r}."],
                       R=cap(join(names)), r=join(names)), must=names)
        else:
            c.say(pick(rng, ["{R} lights are off.", "Done, lights off in {r}.", "Okay, {r} lights off.", "Lights are off in {r}."],
                       R=cap(join(names)), r=join(names)), must=names)
        return c
    a = chosen[0]
    name = room_plain(house, a)
    if cls == "fan":
        on = rng.random() < 0.6
        u = pick(rng, ["Turn {s} the fan in {r}.", "{R} fan {s}.", "Turn the {n} fan {s}.", "Switch {s} the {n} fan."],
                 s="on" if on else "off", r=room_ref(house, a), R=cap(room_ref(house, a)), n=name)
        c.user(voiceify(u, rng))
        c.act([{"domain": "fan", "service": "turn_on" if on else "turn_off", "target": {"area_id": [a]}}])
        c.say(pick(rng, ["{N} fan is {s}.", "Done, {n} fan {s}.", "Okay, fan {s} in {r}."], N=cap(name), n=name, s="on" if on else "off", r=room_ref(house, a)), must=[name])
        return c
    if cls == "cover":
        close = rng.random() < 0.55
        u = pick(rng, ["{V} the blinds in {r}.", "{V} the {n} blinds.", "Can you {v} the blinds in {r}?"],
                 V="Close" if close else "Open", v="close" if close else "open", r=room_ref(house, a), n=name)
        c.user(voiceify(u, rng))
        c.act([{"domain": "cover", "service": "close_cover" if close else "open_cover", "target": {"area_id": [a]}}])
        c.say(pick(rng, ["{N} blinds are {s}.", "Done, {n} blinds {s2}.", "Okay, the blinds in {r} are {s2}."],
                   N=cap(name), n=name, s="closing" if close else "opening", s2="closed" if close else "open", r=room_ref(house, a)), must=[name])
        return c
    # media_player, unique in the area
    mp = house.in_area(a, "media_player")[0]
    dev = short(mp, house)
    action = rng.choice(["pause", "play", "off", "mute"])
    if action == "pause":
        u = pick(rng, ["Pause the {d} in {r}.", "Pause the {n} {d}.", "Can you pause the {d} in {r}?"], d=dev, r=room_ref(house, a), n=name)
        call = {"domain": "media_player", "service": "media_pause", "target": {"area_id": [a]}}
        reply = pick(rng, ["Paused the {n} {d}.", "{N} {d} is paused.", "Okay, paused."], n=name, N=cap(name), d=dev)
    elif action == "play":
        u = pick(rng, ["Resume the {d} in {r}.", "Play the {n} {d}.", "Unpause the {n} {d}."], d=dev, r=room_ref(house, a), n=name)
        call = {"domain": "media_player", "service": "media_play", "target": {"area_id": [a]}}
        reply = pick(rng, ["Playing on the {n} {d}.", "{N} {d} is playing.", "Okay, resumed."], n=name, N=cap(name), d=dev)
    elif action == "off":
        u = pick(rng, ["Turn off the {d} in {r}.", "Turn off the {n} {d}.", "Switch the {n} {d} off."], d=dev, r=room_ref(house, a), n=name)
        call = {"domain": "media_player", "service": "turn_off", "target": {"area_id": [a]}}
        reply = pick(rng, ["{N} {d} is off.", "Done, {n} {d} off.", "Turned off the {n} {d}."], n=name, N=cap(name), d=dev)
    else:
        u = pick(rng, ["Mute the {d} in {r}.", "Mute the {n} {d}."], d=dev, r=room_ref(house, a), n=name)
        call = {"domain": "media_player", "service": "volume_mute", "target": {"area_id": [a]}, "data": {"is_volume_muted": True}}
        reply = pick(rng, ["{N} {d} is muted.", "Muted."], n=name, N=cap(name), d=dev)
    c.user(voiceify(u, rng))
    c.act([call])
    c.say(reply, must=[name] if "{n}" in reply or name in reply else None)
    return c


def control_here(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """Implicit room: the speaker's own satellite location."""
    if sp.channel != "satellite" or not sp.area:
        return None
    a = sp.area
    options = [cls for cls in ("light", "fan", "cover", "media_player") if house.in_area(a, cls)]
    if "media_player" in options and len(house.in_area(a, "media_player")) != 1:
        options.remove("media_player")
    if not options:
        return None
    cls = rng.choices(options, weights=[0.7 if o == "light" else 0.1 for o in options])[0]
    c = Conv(house, sp, "control_here", rng)
    name = room_plain(house, a)
    if cls == "light":
        on = rng.random() < 0.5
        if on:
            u = rng.choice(["Turn on the lights.", "Lights on.", "Turn the lights on.", "It's dark in here.", "Can you turn the light on?", "Turn on the light."])
        else:
            u = rng.choice(["Turn off the lights.", "Lights off.", "Turn the lights off.", "Turn off the light.", "Kill the lights."])
        c.user(voiceify(u, rng))
        c.act([{"domain": "light", "service": "turn_on" if on else "turn_off", "target": {"area_id": [a]}}])
        c.say(pick(rng, ["{N} lights {s}.", "Done, lights {s}.", "Okay, {n} lights {s}."], N=cap(name), n=name, s="on" if on else "off"))
    elif cls == "fan":
        on = rng.random() < 0.6
        c.user(voiceify(rng.choice([f"Turn {'on' if on else 'off'} the fan.", f"Fan {'on' if on else 'off'}.", "I'm hot, put the fan on." if on else "Turn the fan off."]), rng))
        c.act([{"domain": "fan", "service": "turn_on" if on else "turn_off", "target": {"area_id": [a]}}])
        c.say(pick(rng, ["Fan's {s}.", "{N} fan {s}.", "Done."], s="on" if on else "off", N=cap(name)))
    elif cls == "cover":
        close = rng.random() < 0.55
        c.user(voiceify(rng.choice([f"{'Close' if close else 'Open'} the blinds.", f"Blinds {'down' if close else 'up'}.", f"Can you {'close' if close else 'open'} the blinds?"]), rng))
        c.act([{"domain": "cover", "service": "close_cover" if close else "open_cover", "target": {"area_id": [a]}}])
        c.say(pick(rng, ["Blinds {s}.", "{N} blinds {s}.", "Done, {s}."], s="closing" if close else "opening", N=cap(name)))
    else:
        mp = house.in_area(a, "media_player")[0]
        dev = short(mp, house)
        pause = rng.random() < 0.6
        c.user(voiceify(rng.choice([f"Pause the {dev}." if pause else f"Turn off the {dev}.", "Pause." if pause else f"{dev.capitalize()} off.", f"Pause that." if pause else f"Switch the {dev} off."]), rng))
        c.act([{"domain": "media_player", "service": "media_pause" if pause else "turn_off", "target": {"area_id": [a]}}])
        c.say(rng.choice(["Paused." if pause else f"{cap(dev)} is off.", "Done."]))
    return c


def control_brightness(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    areas = _areas_with(house, "light")
    if not areas:
        return None
    here = sp.channel == "satellite" and sp.area in areas and rng.random() < 0.35
    a = sp.area if here else rng.choice(areas)
    p = rng.choice([10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 100])
    c = Conv(house, sp, "control_brightness", rng)
    name = room_plain(house, a)
    if here:
        u = pick(rng, ["Dim the lights to {p} percent.", "Set the lights to {p}%.", "Lights to {p} percent.", "Brightness {p}%."], p=p)
    else:
        u = pick(rng, ["Dim the {n} lights to {p} percent.", "Set the {n} lights to {p}%.", "Set the lights in {r} to {p} percent.",
                       "{N} lights to {p} percent.", "Brighten the {n} lights to {p}%."], n=name, N=cap(name), r=room_ref(house, a), p=p)
    c.user(voiceify(u, rng))
    c.act([{"domain": "light", "service": "turn_on", "target": {"area_id": [a]}, "data": {"brightness_pct": p}}])
    c.say(pick(rng, ["{N} lights at {p} percent.", "Done, {n} lights set to {p}%.", "Okay, {p} percent in {r}."], N=cap(name), n=name, p=p, r=room_ref(house, a)),
          must=[str(p)])
    return c


def control_all_lights(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    areas = _areas_with(house, "light")
    on = rng.random() < 0.15
    c = Conv(house, sp, "control_all_lights", rng)
    if on:
        u = rng.choice(["Turn on all the lights.", "All lights on.", "Turn every light on."])
    else:
        u = rng.choice(["Turn off all the lights.", "All lights off.", "Turn off every light in the house.", "Lights off everywhere.", "Kill all the lights."])
    c.user(voiceify(u, rng))
    c.act([{"domain": "light", "service": "turn_on" if on else "turn_off", "target": {"area_id": areas}}])
    c.say(rng.choice(["All the lights are on." if on else "All the lights are off.", "Done, every light is on." if on else "Done, every light is off.",
                      "Okay, all lights on." if on else "Okay, all lights off."]))
    return c


def _distinct_devices(house: House):
    """Controllable devices worth naming individually: lights in rooms with 2+ lights,
    heaters and appliance switches."""
    out = []
    for a in house.areas:
        lights = house.in_area(a.area_id, "light")
        if len(lights) >= 2:
            out.extend([(e, "light") for e in lights])
        for e in house.in_area(a.area_id, "switch"):
            out.append((e, "switch"))
    for e in house.in_area("energy", "switch"):
        if e.entity_id != "switch.generator":
            out.append((e, "switch"))
    return out


def control_named(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """A specific device: fetch the area's class to learn the entity_id, then act."""
    cands = _distinct_devices(house)
    if not cands:
        return None
    e, cls = rng.choice(cands)
    c = Conv(house, sp, "control_named", rng)
    dev = short(e, house)
    if dev in ("ceiling", "downlights", "light", "lights") and cls == "light":
        dev = dev + " light" if dev == "ceiling" else dev
    on = rng.random() < 0.5 if e.state == "off" else rng.random() < 0.3
    s = "on" if on else "off"
    # design B: the device has to be found first, in its room when the user said it (or it is the speaker's), else by name
    by_name = [{"name": dev}, {"name": dev, "classes": [cls]}]
    if e.area == "energy":
        u = pick(rng, ["Turn {s} the {d}.", "Switch the {d} {s}.", "{D} {s} please."], s=s, d=dev, D=cap(dev))
        c.user(voiceify(u, rng))
        c.lookup(areas=["energy"], classes=["switch"], need=[e], find=by_name)
    else:
        name = room_plain(house, e.area)
        same_short = [x for x in house.entities.values() if x.cls == e.cls and short(x, house) == dev]
        here = sp.channel == "satellite" and sp.area == e.area
        room_known = True
        if here and rng.random() < 0.6:
            u = pick(rng, ["Turn {s} the {d}.", "{D} {s}.", "Switch the {d} {s}.", "Can you turn the {d} {s}?"], s=s, d=dev, D=cap(dev))
        elif len(same_short) == 1 and rng.random() < 0.4:
            u = pick(rng, ["Turn {s} the {d}.", "Switch the {d} {s}.", "Turn the {d} {s}."], s=s, d=dev)
            room_known = False
        else:
            u = pick(rng, ["Turn {s} the {d} in {r}.", "Turn {s} the {n} {d}.", "Switch the {n} {d} {s}.", "Turn the {d} in {r} {s}."],
                     s=s, d=dev, r=room_ref(house, e.area), n=name)
        c.user(voiceify(u, rng))
        find = [{"areas": [e.area], "classes": [cls]}] if room_known else by_name
        if cls == "switch" and len(house.in_area(e.area, "switch")) == 1:
            # unique class in the room: area targeting works, no fetch needed (B: it still has to find the name)
            c.find([e], find)
            target = {"entity_id": [e.entity_id]} if c.toolset == "b" else {"area_id": [e.area]}
            c.act([{"domain": e.domain, "service": "turn_on" if on else "turn_off", "target": target}])
            reply = pick(rng, ["{D} is {s}.", "Done, {d} {s}.", "Okay, {n} {d} {s}."], D=cap(dev), d=dev, s=s, n=name)
            if on and e.attributes.get("power_w") and rng.random() < 0.5:
                reply += f" It draws about {e.attributes['power_w']} watts, so keep an eye on the battery."
            c.say(reply, must=[dev.split()[-1]])
            return c
        c.lookup(areas=[e.area], classes=[cls], need=[e], find=find)
    c.act([{"domain": e.domain, "service": "turn_on" if on else "turn_off", "target": {"entity_id": [e.entity_id]}}])
    reply = pick(rng, ["{D} is {s}.", "Done, {d} {s}.", "Okay, {d} {s}."], D=cap(dev), d=dev, s=s)
    if on and e.attributes.get("power_w") and rng.random() < 0.5:
        reply += f" It draws about {e.attributes['power_w']} watts, so keep an eye on the battery."
    c.say(reply, must=[dev.split()[-1]])
    return c


def control_compound(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """Two independent actions in one utterance -> two tool calls in one turn."""
    la = _areas_with(house, "light")
    if not la:
        return None
    c = Conv(house, sp, "control_compound", rng)
    a1 = rng.choice(la)
    off = rng.random() < 0.6
    parts = [(f"turn {'off' if off else 'on'} the {room_plain(house, a1)} lights",
              {"domain": "light", "service": "turn_off" if off else "turn_on", "target": {"area_id": [a1]}},
              f"{room_plain(house, a1)} lights {'off' if off else 'on'}")]
    second = []
    if house.in_area(a1, "cover"):
        second.append((f"close the blinds", {"domain": "cover", "service": "close_cover", "target": {"area_id": [a1]}}, "blinds closed"))
    for a2 in la:
        if a2 != a1:
            second.append((f"turn {'on' if off else 'off'} the {room_plain(house, a2)} lights",
                           {"domain": "light", "service": "turn_on" if off else "turn_off", "target": {"area_id": [a2]}},
                           f"{room_plain(house, a2)} lights {'on' if off else 'off'}"))
    for a in _areas_with(house, "media_player"):
        if len(house.in_area(a, "media_player")) == 1:
            second.append((f"turn off the {room_plain(house, a)} {short(house.in_area(a, 'media_player')[0], house)}",
                           {"domain": "media_player", "service": "turn_off", "target": {"area_id": [a]}},
                           f"{room_plain(house, a)} {short(house.in_area(a, 'media_player')[0], house)} off"))
    for a in _areas_with(house, "fan"):
        second.append((f"turn off the fan in {room_ref(house, a)}", {"domain": "fan", "service": "turn_off", "target": {"area_id": [a]}},
                       f"{room_plain(house, a)} fan off"))
    if not second:
        return None
    parts.append(rng.choice(second))
    u = cap(parts[0][0]) + rng.choice([" and ", ", and ", " and then "]) + parts[1][0] + "."
    c.user(voiceify(u, rng))
    c.act([parts[0][1], parts[1][1]])
    c.say(pick(rng, ["Done: {a}, {b}.", "Okay, {a} and {b}.", "{A} and {b}."], a=parts[0][2], b=parts[1][2], A=cap(parts[0][2])),
          must=[room_plain(house, a1)])
    return c


CLASS_WORDS = {"fan": ("fan", "fan.turn_on", "fan.turn_off"), "cover": ("blinds", "cover.close_cover", "cover.open_cover"),
               "switch": ("heater", "switch.turn_on", "switch.turn_off"), "light": ("light", "light.turn_on", "light.turn_off")}


def clarify_class(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """'Turn on the fan' when several rooms have one and the speaker's room doesn't -> ask which room, then act by area."""
    opts = []
    for cls, (word, on_svc, off_svc) in CLASS_WORDS.items():
        areas = [a.area_id for a in house.areas if house.in_area(a.area_id, cls)
                 and (cls != "switch" or all(e.attributes.get("heater") for e in house.in_area(a.area_id, "switch")))]
        if len(areas) >= 2 and not (sp.channel == "satellite" and sp.area in areas):
            opts.append((cls, word, on_svc, off_svc, areas))
    if not opts:
        return None
    cls, word, on_svc, off_svc, areas = rng.choice(opts)
    c = Conv(house, sp, "clarify", rng)
    on = rng.random() < 0.55
    if cls == "cover":
        u = pick(rng, ["{V} the blinds.", "Blinds {d}.", "Can you {v} the blinds?"], V="Close" if not on else "Open", v="close" if not on else "open", d="down" if not on else "up")
    else:
        u = pick(rng, ["Turn {s} the {w}.", "{W} {s}.", "Switch the {w} {s}.", "Put the {w} {s}."], s="on" if on else "off", w=word, W=cap(word))
    c.user(voiceify(u, rng))
    if cls == "switch":  # design B: the class counts show fans, blinds and lights by room, but not which switches are heaters
        c.find([e for a in areas for e in house.in_area(a, "switch")], [{"name": word}, {"name": word, "classes": ["switch"]}], exact=True)
    names = [room_plain(house, a) for a in areas]
    if len(areas) == 2:
        q = pick(rng, ["Which {w}, {a} or {b}?", "The one in {ra} or {rb}?", "{A} or {b}?"], w=word + ("" if cls != "cover" else ""), a=room_ref(house, areas[0]), b=room_ref(house, areas[1]),
                 ra=room_ref(house, areas[0]), rb=room_ref(house, areas[1]), A=cap(room_ref(house, areas[0])))
    elif len(areas) <= 4:
        q = pick(rng, ["Which room? There's a {w} in {l}.", "There are {w}s in {l}. Which one?", "In which room? {L} all have one."], w=word, l=rooms_ref(house, areas), L=cap(rooms_ref(house, areas)))
    else:  # too many rooms to read out: just ask
        q = pick(rng, ["Which room?", "Which room's {w}?", "Sure, which room?", "Which {w} do you mean?"], w=word)
        names = []
    c.say(q, must=names)
    pickd = rng.choice(areas)
    pn = room_plain(house, pickd)
    c.user(rng.choice([f"The {pn} one.", f"{cap(pn)}.", f"The one in {room_ref(house, pickd)}.", f"{cap(pn)} please.", f"In {room_ref(house, pickd)}."]))
    svc = on_svc if (on if cls != "cover" else on) else off_svc
    if cls == "cover":
        svc = "cover.open_cover" if on else "cover.close_cover"
    dom, srv = svc.split(".")
    c.act([{"domain": dom, "service": srv, "target": {"area_id": [pickd]}}])
    state = ("on" if on else "off") if cls != "cover" else ("opening" if on else "closing")
    c.say(pick(rng, ["{N} {w} {s}.", "Done, {n} {w} {s}.", "Okay, {w} {s} in {r}."], N=cap(pn), n=pn, w=word, s=state, r=room_ref(house, pickd)))
    return c


def _singular_matches(house: House):
    """Singular device words a user says ("the fan", "the lamp") -> the entities each can mean, house-wide."""
    out = {"fan": house.by_class("fan"),
           "lamp": [e for e in house.by_class("light") if "lamp" in short(e, house)],
           "heater": [e for e in house.by_class("switch") if e.attributes.get("heater") or "heater" in e.name.lower()],
           "tv": [e for e in house.by_class("media_player") if "tv" in e.name.lower().split()]}
    return {w: ents for w, ents in out.items() if ents}


WORD_CLASS = {"fan": "fan", "lamp": "light", "heater": "switch", "tv": "media_player"}


def _find_word(c: Conv, ents, w: str, areas=None):
    """Design B: find_devices for what a singular device word can mean, in the speaker's room or house-wide."""
    where = {"areas": areas} if areas else {}
    c.find(ents, [{**where, "classes": [WORD_CLASS[w]], "name": w}, {**where, "name": w}, {**where, "classes": [WORD_CLASS[w]]}], exact=True)


def _act_on(c: Conv, house: House, e, on: bool):
    """Act on one entity: by area when it is the only one of its class in the room, else fetch for its entity_id."""
    svc = "turn_on" if on else "turn_off"
    if len(house.in_area(e.area, e.cls)) == 1:
        c.act([{"domain": e.domain, "service": svc, "target": {"area_id": [e.area]}}])
    else:
        c.lookup(areas=[e.area], classes=[e.cls], need=[e])
        c.act([{"domain": e.domain, "service": svc, "target": {"entity_id": [e.entity_id]}}])


def clarify_in_room(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """Peter's rule 2 (26 Sep): the speaker's own room has several devices the singular word could mean -> ask which
    (by device name), then act on that one."""
    if sp.channel != "satellite" or not sp.area:
        return None
    opts = []
    for w, ents in _singular_matches(house).items():
        here = [e for e in ents if e.area == sp.area]
        names = [short(e, house) for e in here]
        if len(here) >= 2 and len(set(names)) == len(names) and "" not in names:
            opts.append((w, here, names))
    if not opts:
        return None
    w, here, names = rng.choice(opts)
    c = Conv(house, sp, "clarify_in_room", rng)
    on = rng.random() < 0.6
    c.user(voiceify(pick(rng, ["Turn {s} the {w}.", "{W} {s}.", "Switch the {w} {s}.", "Can you put the {w} {s}?"], s="on" if on else "off", w=w, W=cap(w)), rng))
    _find_word(c, here, w, [sp.area])
    # a device literally called "Lamp" next to a "Bedside Lamp" is "the other one" in speech
    spoken = {n: ("other " + w if n == w else n) for n in names}
    order = sorted(names, key=lambda n: n == w)  # the plainly named one last
    if len(here) == 2:
        a, b = spoken[order[0]], spoken[order[1]]
        b_ = "other one" if order[1] == w else b
        q = pick(rng, ["Which one, the {a} or the {b}?", "The {a} or the {b}?", "There are two in here, the {a} and the {b}. Which one?"], a=a, b=b_)
    else:
        q = pick(rng, ["Which one? There's the {l}.", "You've got a few in here: the {l}. Which one?"], l=join([spoken[n] for n in order]))
    c.say(q, must=[n for n in names if n != w] or names)
    e = rng.choice(here)
    dn = short(e, house)
    said = "other one" if dn == w else dn
    c.user(rng.choice([f"The {said}.", f"{cap(said)}.", f"The {said} please.", f"{cap(said)} thanks."]))
    _act_on(c, house, e, on)
    c.say(pick(rng, ["{D} is {s}.", "Done, {d} {s}.", "Okay, {d} {s}."], D=cap(dn), d=dn, s="on" if on else "off"), must=[w])
    return c


def act_only_one(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """Peter's rule 4 (26 Sep): only one device in the whole house matches the singular word -> act on it, whatever
    room it is in (no question). The reply names the room so the user knows which one moved."""
    opts = [(w, ents[0]) for w, ents in _singular_matches(house).items()
            if len(ents) == 1 and not (sp.channel == "satellite" and ents[0].area == sp.area)]
    if not opts:
        return None
    w, e = rng.choice(opts)
    c = Conv(house, sp, "act_only_one", rng)
    on = rng.random() < 0.6
    s_ = "on" if on else "off"
    c.user(voiceify(pick(rng, ["Turn {s} the {w}.", "{W} {s}.", "Switch the {w} {s}.", "Can you turn the {w} {s}?", "Put the {w} {s} please."], s=s_, w=w, W=cap(w)), rng))
    _find_word(c, [e], w)
    _act_on(c, house, e, on)
    rn = room_plain(house, e.area)
    wd = "TV" if w == "tv" else w
    c.say(pick(rng, ["{N} {w} is {s}.", "Done, the {n} {w} is {s}.", "Okay, {n} {w} {s}."], N=cap(rn), n=rn, w=wd, s=s_), must=[rn])
    return c


def control_lock(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """Locking never needs confirmation. Locks live in the security group, so fetch for the entity id."""
    locks = house.by_class("lock")
    if not locks:
        return None
    c = Conv(house, sp, "control_lock", rng)
    if len(locks) > 1 and rng.random() < 0.4:
        c.user(voiceify(rng.choice(["Lock up.", "Lock all the doors.", "Lock everything.", "Can you lock the doors?"]), rng))
        c.fetch(areas=["security"], classes=["lock"])
        c.act([{"domain": "lock", "service": "lock", "target": {"entity_id": [e.entity_id for e in locks]}}])
        c.say(rng.choice(["All the doors are locked.", "Done, everything's locked.", "Locked up."]), must=["lock"])
        return c
    e = rng.choice(locks)
    dn = e.name.lower()
    c.user(voiceify(pick(rng, ["Lock the {d}.", "Can you lock the {d}?", "{D}, lock please.", "Make sure the {d} is locked."], d=dn, D=cap(dn)), rng))
    c.fetch(areas=["security"], classes=["lock"])
    c.act([{"domain": "lock", "service": "lock", "target": {"entity_id": [e.entity_id]}}])
    if e.state == "locked":
        c.say(pick(rng, ["It was already locked, but it's locked.", "{D} is locked; it already was."], D=cap(dn)), must=["lock"])
    else:
        c.say(pick(rng, ["{D} is locked.", "Done, {d} locked.", "Locked."], D=cap(dn), d=dn), must=["lock"])
    return c


def clarify(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """Ambiguous device reference not resolvable from the speaker's room -> ask, then act."""
    if rng.random() < 0.5:
        alt = clarify_class(house, sp, rng)
        if alt is not None:
            return alt
    groups: dict[str, list] = {}
    for e in house.by_class("light"):
        s = short(e, house)
        if "lamp" in s or s in ("downlights",):
            groups.setdefault("lamp", []).append(e)
    generic = {"ceiling", "lights", "light", "downlights", "fan", "heater", "wall lights", "pendant", "ceiling fan", "ceiling light"}
    for cls in ("switch", "fan", "light"):
        by_name: dict[str, list] = {}
        for e in house.by_class(cls):
            s = short(e, house)
            if s not in generic and "lamp" not in s:
                by_name.setdefault(s, []).append(e)
        for s, ents_ in by_name.items():
            if len({e.area for e in ents_}) >= 2:
                groups.setdefault(s, ents_)
    tvs = [e for e in house.by_class("media_player") if "tv" in e.name.lower()]
    if len(tvs) >= 2:
        groups["tv"] = tvs
    opts = []
    for word, ents in groups.items():
        areas = sorted({e.area for e in ents})
        if len(areas) >= 2 and not (sp.channel == "satellite" and sp.area in areas):
            opts.append((word, ents, areas))
    if not opts:
        return None
    word, ents, areas = rng.choice(opts)
    c = Conv(house, sp, "clarify", rng)
    on = rng.random() < 0.6
    c.user(voiceify(pick(rng, ["Turn {s} the {w}.", "{W} {s}.", "Switch the {w} {s}."], s="on" if on else "off", w=word, W=cap(word)), rng))
    cls = ents[0].cls
    c.find(ents, [{"name": word}, {"name": word, "classes": [cls]}, {"classes": [cls]}], exact=True)  # design B: which rooms have one
    names = [room_plain(house, a) for a in areas]
    if len(areas) == 2:
        q = pick(rng, ["Which {w}, the {a} one or the {b} one?", "The {a} {w} or the {b} {w}?", "Do you mean the {w} in {ra} or {rb}?"],
                 w=word, a=names[0], b=names[1], ra=room_ref(house, areas[0]), rb=room_ref(house, areas[1]))
    else:
        q = pick(rng, ["Which {w}? There's one in {l}.", "There are {ws} in {l}. Which one?"], w=word, ws=word + "s", l=rooms_ref(house, areas))
    c.say(q, must=names)
    pickd = rng.choice(areas)
    pn = room_plain(house, pickd)
    c.user(rng.choice([f"The {pn} one.", f"{cap(pn)}.", f"The one in {room_ref(house, pickd)}.", f"{cap(pn)} please."]))
    if word == "tv":
        if len(house.in_area(pickd, "media_player")) == 1:
            c.act([{"domain": "media_player", "service": "turn_on" if on else "turn_off", "target": {"area_id": [pickd]}}])
        else:
            tv = [e for e in ents if e.area == pickd][0]
            c.lookup(areas=[pickd], classes=["media_player"], need=[tv])
            c.act([{"domain": "media_player", "service": "turn_on" if on else "turn_off", "target": {"entity_id": [tv.entity_id]}}])
        c.say(pick(rng, ["{N} TV is {s}.", "Done, {n} TV {s}."], N=cap(pn), n=pn, s="on" if on else "off"))
    else:
        target = [e for e in ents if e.area == pickd]
        c.lookup(areas=[pickd], classes=[target[0].cls], need=target)
        c.act([{"domain": target[0].domain, "service": "turn_on" if on else "turn_off", "target": {"entity_id": [e.entity_id for e in target]}}])
        c.say(pick(rng, ["{N} {w} is {s}.", "Done, {n} {w} {s}.", "Okay, {n} {w} {s}."], N=cap(pn), n=pn, w=word, s="on" if on else "off"))
    return c


def missing(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """Requests for things the house does not have -> say so, no tool call."""
    things = ["the pool pump", "the sprinklers", "the hot tub", "the EV charger", "the dishwasher", "the fish tank light",
              "the sauna", "the pond pump", "the irrigation", "the driveway gate"]
    if not house.by_class("garage"):
        things += ["the garage door", "the garage"]
    if not house.by_class("climate"):
        things += ["the aircon", "the heating", "the air conditioning"]
    if not house.by_class("vacuum"):
        things += ["the vacuum", "the robot vacuum"]
    for a in house.areas:
        if a.kind == "outdoor":
            continue
        for cls, w in (("cover", "blinds"), ("fan", "fan"), ("media_player", "tv"), ("switch", "heater")):
            if not house.in_area(a.area_id, cls):
                things.append(f"the {w} in {room_ref(house, a.area_id)}")
    n_fixed = 10  # the house-wide things above; the rest are absent classes the inventory already shows
    thing = rng.choice(things)
    c = Conv(house, sp, "missing", rng)
    verb = rng.choice(["Turn on", "Turn off", "Switch on", "Start"]) if "blinds" not in thing and "garage" not in thing else rng.choice(["Close", "Open"])
    c.user(voiceify(f"{verb} {thing}.", rng))
    if thing in things[:n_fixed]:  # design B: without names in the prompt, search before saying it isn't there
        c.find([], [{"name": thing.removeprefix("the ")}])
        if c.toolset == "b" and c.msgs[-1]["content"] != "No matching devices.":
            c.b_issues.append(f"missing: {thing} exists")
    c.say(pick(rng, ["I don't have {t} set up.", "There's no {t} in the system, sorry.", "I can't see {t} anywhere in Home Assistant.",
                     "Nothing called {t} is configured, sorry."], t=thing.replace("the EV", "an EV") if thing.startswith("the EV") else thing))
    return c


def confirm_sensitive(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    opts = []
    for e in house.by_class("lock"):
        opts.append(("unlock", e))
    for e in house.by_class("alarm"):
        opts.append(("disarm", e))
    for e in house.by_class("garage"):
        opts.append(("garage", e))
    if not opts:
        return None
    kind, e = rng.choice(opts)
    c = Conv(house, sp, "confirm_sensitive", rng)
    yes = rng.random() < 0.7
    if kind == "unlock":
        dn = e.name.lower()
        c.user(voiceify(pick(rng, ["Unlock the {d}.", "Can you unlock the {d}?", "Open up the {d}.", "{D} unlock.", "Unlock the {d} for me.",
                                   "The courier's at the {d}, can you unlock it?", "Unlock the {d}, I've left my keys inside.",
                                   "Let the plumber in, unlock the {d}.", "I'm outside, unlock the {d} please.", "Pop the {d} open for me.",
                                   "Can you unlock the {d} quick, my hands are full.", "The electrician's here, let them in the {d}.",
                                   "Unlock the {d}, we're just getting home."], d=dn, D=cap(dn)), rng))
        c.say(pick(rng, ["Unlock the {d}, are you sure?", "Just to confirm, you want the {d} unlocked?", "Do you want me to unlock the {d}?"], d=dn), must=[dn])
        if yes:
            c.user(rng.choice(["Yes.", "Yep.", "Yes please.", "Go ahead.", "Do it.", "Yeah, unlock it."]))
            c.fetch(areas=["security"], classes=["lock"])
            c.act([{"domain": "lock", "service": "unlock", "target": {"entity_id": [e.entity_id]}}])
            c.say(pick(rng, ["{D} is unlocked.", "Done, {d} unlocked.", "Okay, the {d} is unlocked."], D=cap(dn), d=dn), must=[dn])
        else:
            c.user(rng.choice(["No.", "Actually no.", "Never mind.", "Leave it.", "No, leave it locked."]))
            c.say(rng.choice(["Okay, leaving it locked.", "No problem, it stays locked.", "Sure, I'll leave it."]))
    elif kind == "disarm":
        c.user(voiceify(rng.choice(["Disarm the alarm.", "Turn off the alarm.", "Can you disarm the alarm?", "Alarm off.",
                                    "Disarm the alarm, the electrician's coming by.", "Turn the alarm off, I'm about to open the door.",
                                    "Kill the alarm, we're home.", "Switch the alarm off for me.", "Can you turn the alarm off before the cleaner gets here?",
                                    "Disarm it, I'm walking in now."]), rng))
        c.say(rng.choice(["Disarm the alarm, are you sure?", "Just to confirm, disarm the alarm?", "You want the alarm disarmed?"]), must=["alarm"])
        if yes:
            c.user(rng.choice(["Yes.", "Yep.", "Yes please.", "Go ahead.", "Confirmed."]))
            c.fetch(areas=["security"], classes=["alarm"])
            c.act([{"domain": "alarm_control_panel", "service": "alarm_disarm", "target": {"entity_id": [e.entity_id]}}])
            c.say(rng.choice(["Alarm is disarmed.", "Done, alarm disarmed.", "Okay, the alarm's off."]), must=["alarm"])
        else:
            c.user(rng.choice(["No.", "Never mind.", "Leave it on."]))
            c.say(rng.choice(["Okay, leaving the alarm armed.", "No problem, it stays armed."]))
    else:
        c.user(voiceify(rng.choice(["Open the garage.", "Open the garage door.", "Can you open the garage?", "Garage up.",
                                    "Open the garage, I'm pulling in.", "Pop the garage for me.", "Can you open the garage door? I need the ute.",
                                    "Open the garage, the delivery's here."]), rng))
        c.say(rng.choice(["Open the garage door, are you sure?", "Just to confirm, open the garage?", "You want the garage door opened?"]), must=["garage"])
        if yes:
            c.user(rng.choice(["Yes.", "Yep.", "Yes please.", "Go ahead.", "Yeah open it."]))
            if e.area in house.area_ids():
                c.act([{"domain": "cover", "service": "open_cover", "target": {"area_id": [e.area]}}])
            else:
                c.fetch(areas=["security"], classes=["garage"])
                c.act([{"domain": "cover", "service": "open_cover", "target": {"entity_id": [e.entity_id]}}])
            c.say(rng.choice(["Garage is opening.", "Done, garage door opening.", "Okay, opening the garage."]), must=["garage"])
        else:
            c.user(rng.choice(["No.", "Never mind.", "Leave it closed."]))
            c.say(rng.choice(["Okay, leaving it closed.", "No problem, the garage stays shut."]))
    return c


def climate_control(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """Climate is always targeted by area_id (never a guessed entity id). With several units the room
    comes from the utterance or from the speaker's location."""
    clims = house.by_class("climate")
    if not clims:
        return None
    c = Conv(house, sp, "control_climate", rng)
    t = rng.choice([18, 19, 20, 21, 22, 23, 24])
    kind = rng.choices(["set", "off", "on", "here"], weights=[0.5, 0.2, 0.15, 0.15])[0]
    here = [e for e in clims if sp.channel == "satellite" and e.area == sp.area]
    if len(clims) == 1:
        cl, where, explicit = clims[0], "", False
    elif kind == "here" and here:
        cl, where, explicit = here[0], "", False
    elif here and rng.random() < 0.4:
        cl, where, explicit = here[0], " in here", False
    else:
        cl = rng.choice(clims)
        where, explicit = f" in {room_ref(house, cl.area)}", True
    unit = "heating" if cl.attributes.get("last_mode") == "heat" else "aircon"
    rn = room_plain(house, cl.area)
    if kind in ("set", "here"):
        if kind == "here":
            u = rng.choice(["It's cold in here.", "I'm cold.", "It's too warm in here.", "I'm hot."])
            cur = float(cl.attributes.get("temperature", 21))
            t = int(cur + 2) if "cold" in u else int(cur - 2)
        else:
            u = pick(rng, ["Set the temperature{w} to {t}.", "Set the {unit}{w} to {t} degrees.", "Make it {t} degrees{w}.", "{Unit}{w} to {t} please."], w=where, t=t, unit=unit, Unit=cap(unit))
        c.user(voiceify(u, rng))
        c.act([{"domain": "climate", "service": "set_temperature", "target": {"area_id": [cl.area]}, "data": {"temperature": t}}])
        c.say(pick(rng, ["{R} set to {t}.", "Done, {t} degrees{w2}.", "Okay, {t} in {r}."], R=cap(rn) + (" " + unit if explicit or len(clims) > 1 else "thermostat"), t=t, w2=(" in " + room_ref(house, cl.area)) if len(clims) > 1 else "", r=room_ref(house, cl.area)), must=[str(t)])
    elif kind == "off":
        c.user(voiceify(pick(rng, ["Turn off the {unit}{w}.", "Turn the {unit}{w} off.", "Switch off the {unit}{w}."], unit=unit, w=where), rng))
        c.act([{"domain": "climate", "service": "turn_off", "target": {"area_id": [cl.area]}}])
        c.say(pick(rng, ["{Unit}{w2} is off.", "Done, {unit} off{w2}."], Unit=cap(unit), unit=unit, w2=(" in " + room_ref(house, cl.area)) if len(clims) > 1 else ""))
    else:
        mode = cl.attributes.get("last_mode", "heat")
        c.user(voiceify(pick(rng, ["Turn on the {unit}{w}.", "Put the {unit} on{w}.", "Warm up {rr}." if mode == "heat" else "Cool down {rr}."], unit=unit, w=where, rr=room_ref(house, cl.area) if len(clims) > 1 else "the house"), rng))
        c.act([{"domain": "climate", "service": "set_hvac_mode", "target": {"area_id": [cl.area]}, "data": {"hvac_mode": mode}}])
        c.say(pick(rng, ["{M} is on{w2}, set to {t}.", "Done, {m} on at {t} degrees{w2}."], M="Heating" if mode == "heat" else "Cooling", m="heating" if mode == "heat" else "cooling", t=int(cl.attributes.get("temperature", 21)), w2=(" in " + room_ref(house, cl.area)) if len(clims) > 1 else ""))
    return c
