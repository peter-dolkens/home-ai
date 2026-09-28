"""Appliances, water (tank, pump, irrigation) and the vehicle."""
from __future__ import annotations

import random
from typing import Optional

from ...house.schema import House, Speaker
from ...house.snapshot import num
from ..common import Conv, cap, is_are, join, pick, room_plain, room_ref, short, spoken_ago, voiceify


def appliance_status(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    apps = house.by_class("appliance")
    if not apps:
        return None
    e = rng.choice(apps)
    kind = e.attributes.get("appliance", "appliance")
    st = e.attributes.get("status", e.state)
    c = Conv(house, sp, "appliance_status", rng)
    if kind in ("fridge", "freezer"):
        c.user(voiceify(pick(rng, ["Is the {k} door open?", "Did someone leave the {k} open?", "Is the {k} okay?"], k=kind), rng, "q"))
    else:
        c.user(voiceify(pick(rng, ["Is the {k} finished?", "Is the {k} done?", "How long left on the {k}?", "Is the {k} still going?", "What's the {k} doing?"], k=kind), rng, "q"))
    c.fetch(areas=[e.area], classes=["appliance"], b={"classes": ["appliance"]})  # design B: its room is not in the prompt
    if st == "running":
        rem = e.attributes.get("remaining_min")
        prog = e.attributes.get("program")
        c.say(pick(rng, ["Still going, about {r} minutes left{p}.", "The {k} is running{p}, {r} minutes to go.", "Not yet, {r} minutes left."],
                   r=rem, p=f" on {prog}" if prog else "", k=kind), must=[str(rem)])
    elif st == "finished":
        c.say(pick(rng, ["Yes, the {k} finished {ago}.", "It's done, finished {ago}.", "Finished {ago}."], k=kind, ago=spoken_ago(e.changed_s)), must=["finish|done"])
    elif st == "door open":
        c.say(pick(rng, ["The {k} door is open, has been for {d}.", "Yes, the {k} door's open."], k=kind, d=spoken_ago(e.changed_s).replace(" ago", "")), must=["open"])
    elif kind in ("fridge", "freezer"):
        c.say(pick(rng, ["No, the {k} is closed and running normally.", "The {k} is fine, door's shut."], k=kind), must=["closed|shut|fine"])
    else:
        c.say(pick(rng, ["The {k} is off, it's not running.", "Nothing running, the {k} is off."], k=kind), must=["off|not running"])
    return c


def appliance_control(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    apps = [e for e in house.by_class("appliance") if e.attributes.get("appliance") not in ("fridge", "freezer")]
    if not apps:
        return None
    e = rng.choice(apps)
    kind = e.attributes.get("appliance")
    c = Conv(house, sp, "appliance_control", rng)
    off = e.state == "on" or rng.random() < 0.7
    if off:
        c.user(voiceify(pick(rng, ["Turn off the {k}.", "Switch the {k} off.", "Stop the {k}.", "Kill the {k}."], k=kind), rng))
    else:
        c.user(voiceify(pick(rng, ["Start the {k}.", "Turn on the {k}.", "Run the {k}."], k=kind), rng))
    c.fetch(areas=[e.area], classes=["appliance"], b={"classes": ["appliance"]})
    c.act([{"domain": "switch", "service": "turn_off" if off else "turn_on", "target": {"entity_id": [e.entity_id]}}])
    c.say(pick(rng, ["{K} is {s}.", "Done, {k} {s}.", "Okay, {k} {s}."], K=cap(kind), k=kind, s="off" if off else "on"), must=[kind.split(" ")[-1]])
    return c


def water_status(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    tank = house.entities.get("sensor.tank_level")
    pump = house.first("pump")
    if not tank and not pump:
        return None
    c = Conv(house, sp, "water_status", rng)
    kind = rng.choice(["tank", "tank", "pump"] if tank and pump else ["tank"] if tank else ["pump"])
    if kind == "tank":
        c.user(voiceify(rng.choice(["How full is the tank?", "What's the water level?", "How much water have we got?", "How's the tank looking?", "Are we low on water?"]), rng, "q"))
        c.fetch(areas=["water"], b={"areas": ["water"], "classes": ["water"]}, need=[tank])
        pct = float(tank.state)
        litres = tank.attributes.get("litres")
        text = pick(rng, ["The tank's at {p} percent, about {l} litres.", "{p} percent, roughly {l} litres left."], p=num(pct, 0), l=num(round(litres, -2), 0))
        if pct < 25:
            text += " That's getting low."
        c.say(text, must=[num(pct, 0)])
    else:
        c.user(voiceify(rng.choice(["Is the pump running?", "Is the water pump on?", "Has the pump been running?"]), rng, "q"))
        c.fetch(areas=["water"], classes=["pump"])
        if pump.state == "on":
            c.say(pick(rng, ["Yes, the {n} is running, {w} watts.", "It's on, drawing {w} watts."], n=pump.name.lower(), w=pump.attributes.get("power_w")), must=["running|on"])
        else:
            c.say(pick(rng, ["No, the pump's off. It last ran {ago}.", "It's off at the moment, last ran {ago}."], ago=spoken_ago(pump.changed_s)), must=["off|no"])
    return c


def irrigation_control(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    zones = house.by_class("irrigation")
    if not zones:
        return None
    z = rng.choice(zones)
    zn = z.name.lower().replace(" irrigation", "")
    c = Conv(house, sp, "irrigation_control", rng)
    kind = rng.choices(["on", "off", "status", "all_off"], weights=[0.45, 0.25, 0.15, 0.15])[0]
    if kind == "on":
        mins = rng.choice([None, None, 10, 15, 20, 30])
        u = pick(rng, ["Water the {z}.", "Turn on the {z} irrigation.", "Start watering the {z}.", "Run the {z} sprinklers."], z=zn)
        if mins:
            u = u.rstrip(".") + f" for {mins} minutes."
        c.user(voiceify(u, rng))
        c.fetch(areas=["water"], classes=["irrigation"])
        call = {"domain": "switch", "service": "turn_on", "target": {"entity_id": [z.entity_id]}}
        if mins:
            call["data"] = {"duration": mins}
        c.act([call])
        c.say(pick(rng, ["Watering the {z}{m}.", "{Z} irrigation is on{m}.", "Okay, {z} on{m}."], z=zn, Z=cap(zn), m=f" for {mins} minutes" if mins else ""), must=[zn.split(" ")[0]])
    elif kind == "off":
        c.user(voiceify(pick(rng, ["Stop watering the {z}.", "Turn off the {z} irrigation.", "That's enough water on the {z}."], z=zn), rng))
        c.fetch(areas=["water"], classes=["irrigation"])
        c.act([{"domain": "switch", "service": "turn_off", "target": {"entity_id": [z.entity_id]}}])
        c.say(pick(rng, ["{Z} irrigation is off.", "Stopped watering the {z}."], z=zn, Z=cap(zn)), must=[zn.split(" ")[0]])
    elif kind == "all_off":
        c.user(voiceify(rng.choice(["Turn off all the irrigation.", "Stop all the sprinklers.", "Shut off the watering everywhere."]), rng))
        c.fetch(areas=["water"], classes=["irrigation"])
        on = [e for e in zones if e.state == "on"]
        if on:
            c.act([{"domain": "switch", "service": "turn_off", "target": {"entity_id": [e.entity_id for e in on]}}])
            c.say(pick(rng, ["Irrigation's off: {l}.", "Stopped {l}."], l=join([e.name.lower().replace(" irrigation", "") for e in on])))
        else:
            c.say(rng.choice(["Nothing was running; all the irrigation is already off.", "It's all off already."]), must=["already|off"])
    else:
        c.user(voiceify(rng.choice(["Is the irrigation running?", "Is anything being watered right now?", "Are the sprinklers on?"]), rng, "q"))
        c.fetch(areas=["water"], classes=["irrigation"])
        on = [e for e in zones if e.state == "on"]
        if on:
            c.say(pick(rng, ["Yes, the {l} {v} running.", "{L} {v} on right now."], l=join([e.name.lower().replace(" irrigation", "") for e in on]), L=cap(join([e.name.lower().replace(" irrigation", "") for e in on])), v=is_are(len(on))), must=[on[0].name.lower().split(" ")[0]])
        else:
            last = min(zones, key=lambda e: e.changed_s)
            c.say(pick(rng, ["No, nothing's running. The {z} last ran {ago}.", "All off. Last was the {z}, {ago}."], z=last.name.lower().replace(" irrigation", ""), ago=spoken_ago(last.changed_s)), must=["no|off|nothing"])
    return c


def vehicle_status(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    v = house.first("vehicle")
    if not v:
        return None
    car = v.name if "shark" in v.name.lower() else "the car"
    c = Conv(house, sp, "vehicle_status", rng)
    kind = rng.choice(["charge", "charge", "where", "locked", "plugged"])
    a = v.attributes
    if kind == "charge":
        c.user(voiceify(pick(rng, ["How charged is {c}?", "What's {c} at?", "How much range has {c} got?", "Is {c} charged?"], c=car), rng, "q"))
    elif kind == "where":
        c.user(voiceify(pick(rng, ["Where's {c}?", "Is {c} here?", "Is {c} home?"], c=car), rng, "q"))
    elif kind == "locked":
        c.user(voiceify(pick(rng, ["Is {c} locked?", "Did I lock {c}?"], c=car), rng, "q"))
    else:
        c.user(voiceify(pick(rng, ["Is {c} plugged in?", "Is {c} charging?", "Did we plug {c} in?"], c=car), rng, "q"))
    c.fetch(areas=["energy"], classes=["vehicle"])
    soc = num(v.state, 0)
    if kind == "charge":
        c.say(pick(rng, ["{C} is at {s} percent, about {r} kilometres of range.", "{s} percent, roughly {r} k's."], C=cap(car), s=soc, r=num(a.get("range_km"), 0)), must=[soc])
    elif kind == "where":
        loc = a.get("location", "")
        if loc.startswith("parked here"):
            c.say(pick(rng, ["It's here, parked at the house{p}.", "Yes, {c} is here{p}."], c=car, p=", plugged in" if a.get("plugged") else ""), must=["here"])
        elif loc.startswith("at the old house"):
            c.say(pick(rng, ["It's over at the old house{ch}.", "{C} is at the old house{ch}."], C=cap(car), ch=", charging" if "charging" in loc else ""), must=["old house"])
        else:
            c.say(pick(rng, ["It's not here; {c} is away.", "{C} isn't home."], c=car, C=cap(car)), must=["away|not here|isn't"])
    elif kind == "locked":
        c.say(pick(rng, ["Yes, it's locked.", "{C} is locked."], C=cap(car)) if a.get("locked") else pick(rng, ["No, {c} is unlocked.", "It's unlocked."], c=car), must=["locked" if a.get("locked") else "unlocked"])
    else:
        if a.get("plugged"):
            c.say(pick(rng, ["Yes, it's plugged in{ch}, {s} percent.", "Plugged in{ch} at {s} percent."], ch=" and charging" if "charging" in a.get("location", "") else "", s=soc), must=["plugged"])
        else:
            c.say(pick(rng, ["No, it's not plugged in. It's at {s} percent.", "Not plugged in, sitting at {s} percent."], s=soc), must=["not plugged"])
    return c
