"""Whole-house aggregate questions that get hard at big-house scale: "is anything left on?", "what's open?",
"what's using the most power?". Not in the training registry (SCENARIOS); homeai/datagen/bigbench.py uses them."""
from __future__ import annotations

import random
from typing import Optional

from ...house.schema import House, Speaker
from ...house.snapshot import num
from ..common import Conv, cap, is_are, join, pick, room_plain, room_ref, rooms_ref, short, voiceify


def _rooms(house: House, ents) -> list[str]:
    out = []
    for e in ents:
        if e.area not in out:
            out.append(e.area)
    return out


def anything_on(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """Lights by room, plus media, heaters and other big loads, fans, climate and running appliances."""
    c = Conv(house, sp, "anything_on", rng)
    c.user(voiceify(rng.choice(["Is anything left on?", "What's still on in the house?", "Did we leave anything on?", "Is there anything on that shouldn't be?",
                                "What's running at the moment?", "Anything left on around the house?"]), rng, "q"))
    c.fetch(classes=["light", "switch", "fan", "media_player", "climate", "appliance"],
            b=[{"classes": ["light"], "where": ["on"], "_rooms": True}, {"classes": ["fan", "media_player", "climate", "appliance"], "where": ["on"]},
               {"classes": ["switch"], "where": ["power>0"], "sort": "power_desc"}])
    lights = [e for e in house.by_class("light") if e.state == "on" and house.area(e.area)]
    media = [e for e in house.by_class("media_player") if e.state in ("playing", "paused")]
    loads = [e for e in house.by_class("switch") if e.state == "on" and e.attributes.get("power_w") and not e.attributes.get("source_w")]
    fans = [e for e in house.by_class("fan") if e.state == "on"]
    clim = [e for e in house.by_class("climate") if e.state != "off"]
    apps = [e for e in house.by_class("appliance") if e.attributes.get("status") == "running"]
    bits, must = [], []
    rooms = _rooms(house, lights)
    if rooms:
        if len(rooms) <= 4:
            bits.append(f"lights in {rooms_ref(house, rooms)}")
            must += [room_plain(house, a) for a in rooms]
        else:
            bits.append(f"lights in {len(rooms)} rooms, including {rooms_ref(house, rooms[:3])}")
            must += [room_plain(house, a) for a in rooms[:3]]
    for group, word in ((media, None), (loads, None), (fans, None), (apps, "running")):
        if not group:
            continue
        names = [e.name.lower() for e in group[:4]]
        more = f" and {len(group) - 4} more" if len(group) > 4 else ""
        bits.append(join(names) + more + (f" {word}" if word else ""))
        must += names[:2]
    if clim:
        bits.append(("the aircon in " if len(clim) > 1 else "the aircon in ") + rooms_ref(house, _rooms(house, clim)[:3]))
        must.append(room_plain(house, clim[0].area))
    if not bits:
        c.say(rng.choice(["No, everything's off.", "Nothing's on, the house is all off.", "All off."]), must=["off|nothing"])
        return c
    text = pick(rng, ["Still on: {b}.", "You've got {b} on.", "On right now: {b}."], b=join(bits))
    if lights or media or fans:
        text += rng.choice([" Want me to turn them off?", " Shall I switch those off?", ""])
    c.say(text, must=list(dict.fromkeys(must)))
    return c


def open_check(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """Doors, window contacts, the garage and motorised windows."""
    if not (house.by_class("window") or house.by_class("door")):
        return None
    c = Conv(house, sp, "open_check", rng)
    c.user(voiceify(rng.choice(["What's open?", "Is anything open?", "Are any doors or windows open?", "Which doors and windows are open?",
                                "Is the house shut up?", "Have we left anything open?"]), rng, "q"))
    c.fetch(classes=["door", "window", "garage", "cover"],
            b=[{"classes": ["door", "window", "garage"], "where": ["open"]}, {"classes": ["cover"], "name": "window", "where": ["open"]}])
    doors = [e for e in house.by_class("door") if e.state == "on"]
    windows = [e for e in house.by_class("window") if e.state == "on"]
    garages = [e for e in house.by_class("garage") if e.state == "open"]
    motor = [e for e in house.by_class("cover") if e.state == "open" and "window" in e.name.lower()]
    bits, must = [], []
    if doors:
        names = [e.name.lower() if e.area == "security" else short(e, house) for e in doors]
        bits.append(f"the {join(names[:5])}" + (f" and {len(names) - 5} other doors" if len(names) > 5 else ""))
        must += names[:3]
    wrooms = _rooms(house, windows + motor)
    if wrooms:
        bits.append(f"window{'s' if len(windows) + len(motor) > 1 else ''} in {rooms_ref(house, wrooms[:5])}"
                    + (f" and {len(wrooms) - 5} other rooms" if len(wrooms) > 5 else ""))
        must += [room_plain(house, a) for a in wrooms[:3]]
    if garages:
        bits.append(join([e.name.lower() for e in garages]))
        must.append("garage")
    if not bits:
        c.say(rng.choice(["Nothing's open, all the doors and windows are shut.", "All closed.", "Everything's shut."]),
              must=["closed|shut|nothing open|none open|not open"])
        return c
    c.say(pick(rng, ["Open right now: {b}.", "{B} {v} open.", "I can see {b} open."], b=join(bits), B=cap(join(bits)),
               v=is_are(len(doors) + len(windows) + len(garages) + len(motor))), must=list(dict.fromkeys(must)))
    return c


def top_power(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """Rank the power sensors; generation (inverters) is not use."""
    meters = [e for e in house.by_class("power") if not e.attributes.get("entity_category")
              and not any(w in e.name for w in ("Inverter", "MPPT", "Generator"))]
    if len(meters) < 3:
        return None
    c = Conv(house, sp, "top_power", rng)
    c.user(voiceify(rng.choice(["What's using the most power?", "What's drawing all the power right now?", "Where's all our power going?",
                                "What are the biggest loads at the moment?", "Which things are using the most electricity?"]), rng, "q"))
    top = sorted(meters, key=lambda e: -float(e.state))[:3]
    c.fetch(classes=["power"], b={"classes": ["power"], "sort": "power_desc", "limit": 10}, need=[e for e in top if float(e.state) >= 20])
    if float(top[0].state) < 20:
        c.say(rng.choice(["Nothing much, every load is under 20 watts.", "Nothing big, it's all standby draw."]), must=["nothing|standby|under"])
        return c
    parts = [f"{e.name.lower().removesuffix(' power')} at {num(e.state, 0)} watts" for e in top if float(e.state) >= 20]
    c.say(pick(rng, ["The biggest is {a}{r}.", "Top of the list is {a}{r}."], a=parts[0],
               r=(", then " + join(parts[1:])) if len(parts) > 1 else ""), must=[num(top[0].state, 0)])
    return c
