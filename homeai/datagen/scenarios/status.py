"""Questions answered by fetching a narrow slice of state."""
from __future__ import annotations

import random
from typing import Optional

from ...house.schema import House, Speaker
from ...house.snapshot import num, rel_time
from ..common import Conv, cap, is_are, isv, join, pick, room_plain, room_ref, rooms_ref, short, spoken_ago, spoken_dur, voiceify


def status_entity(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    kinds = []
    if house.by_class("lock"):
        kinds.append("lock")
    if [e for e in house.by_class("window")]:
        kinds.append("window")
    if house.by_class("light"):
        kinds.append("light")
    if [e for e in house.by_class("temperature") if e.area != "weather"]:
        kinds.append("temperature")
    tvs = [e for e in house.by_class("media_player") if "tv" in e.name.lower()]
    if len(tvs) == 1:
        kinds.append("tv")
    if house.by_class("garage"):
        kinds.append("garage")
    if house.by_class("cover"):
        kinds.append("cover")
    if house.by_class("alarm"):
        kinds.append("alarm")
    if house.by_class("climate"):
        kinds.append("climate")
    heaters = [e for e in house.by_class("switch") if e.attributes.get("heater")]
    if heaters:
        kinds.append("heater")
    if house.by_class("fan"):
        kinds.append("fan")
    if not kinds:
        return None
    kind = rng.choice(kinds)
    c = Conv(house, sp, "status_entity", rng)

    if kind == "lock":
        e = rng.choice(house.by_class("lock"))
        dn = e.name.lower()
        c.user(voiceify(pick(rng, ["Is the {d} locked?", "Did I lock the {d}?", "Is the {d} locked or not?", "Check the {d} for me."], d=dn), rng, "q"))
        c.fetch(areas=["security"], classes=["lock"])
        if e.state == "locked":
            c.say(pick(rng, ["Yes, the {d} is locked.", "The {d} is locked, has been for {t}.", "Locked."], d=dn, t=spoken_dur(e.changed_s)), must=["lock"])
        else:
            c.say(pick(rng, ["No, the {d} is unlocked. Want me to lock it?", "The {d} is unlocked right now. Shall I lock it?"], d=dn), must=["unlocked"])
            if rng.random() < 0.6:
                c.user(rng.choice(["Yes.", "Yes please.", "Please.", "Yeah lock it."]))
                c.act([{"domain": "lock", "service": "lock", "target": {"entity_id": [e.entity_id]}}])
                c.say(pick(rng, ["{D} is locked.", "Done, {d} locked.", "Locked."], D=cap(dn), d=dn))
            else:
                c.user(rng.choice(["No, leave it.", "No that's fine.", "Not yet."]))
                c.say(rng.choice(["Okay.", "No problem.", "Sure, leaving it."]))
        return c

    if kind == "window":
        e = rng.choice(house.by_class("window"))
        rn = room_plain(house, e.area)
        c.user(voiceify(pick(rng, ["Is the {r} window open?", "Did we leave the {r} window open?", "Is the window open in {rr}?"], r=rn, rr=room_ref(house, e.area)), rng, "q"))
        c.fetch(areas=[e.area], classes=["window"])
        if e.state == "on":
            c.say(pick(rng, ["Yes, the window in {rr} is open.", "It is, the window in {rr} has been open for {t}."], rr=room_ref(house, e.area), t=spoken_dur(e.changed_s)), must=["open"])
        else:
            c.say(pick(rng, ["No, the {r} window is closed.", "It's closed."], r=rn), must=["closed"])
        return c

    if kind == "light":
        a = rng.choice([x for x in house.areas if house.in_area(x.area_id, "light")]).area_id
        rn = room_plain(house, a)
        lights = house.in_area(a, "light")
        c.user(voiceify(pick(rng, ["Are the {r} lights on?", "Is the light on in {rr}?", "Are any lights on in {rr}?", "Did I leave the {r} light on?"], r=rn, rr=room_ref(house, a)), rng, "q"))
        c.fetch(areas=[a], classes=["light"])
        on = [e for e in lights if e.state == "on"]
        if on:
            desc = join([f"{short(e, house)} at {e.attributes.get('brightness_pct', 100)}%" for e in on])
            if len(lights) == 1:
                c.say(pick(rng, ["Yes, the {r} light is on at {b}%.", "It's on, {b}%."], r=rn, b=on[0].attributes.get("brightness_pct", 100)), must=["on"])
            else:
                c.say(pick(rng, ["Yes, the {d} {v} on.", "In {rr} the {d} {v} on.", "{D} {v} on."], d=desc, D=cap(desc), v=is_are(len(on)), rr=room_ref(house, a)), must=[short(e, house) for e in on])
        else:
            c.say(pick(rng, ["No, the {r} lights are all off.", "Nothing's on in {rr}.", "No, all off."], r=rn, rr=room_ref(house, a)), must=["off|nothing's on|nothing on|none on|dark|no lights"])
        return c

    if kind == "temperature":
        e = rng.choice([x for x in house.by_class("temperature") if x.area != "weather"])
        rn = room_plain(house, e.area)
        c.user(voiceify(pick(rng, ["What's the temperature in {rr}?", "How warm is {rr}?", "How cold is it in {rr}?", "What temperature is the {r}?"], rr=room_ref(house, e.area), r=rn), rng, "q"))
        c.fetch(areas=[e.area], classes=["temperature"])
        t = num(e.state)
        c.say(pick(rng, ["It's {t} degrees in {rr}.", "{RR} is at {t} degrees.", "{t} degrees."], t=t, rr=room_ref(house, e.area), RR=cap(room_ref(house, e.area))), must=[t])
        return c

    if kind == "tv":
        e = tvs[0]
        rn = room_plain(house, e.area)
        c.user(voiceify(rng.choice(["Is the TV on?", f"Is the {rn} TV on?", "Is anything playing on the TV?", "What's on the TV?"]), rng, "q"))
        # design B: the TV's room is only known when the user said it
        c.fetch(areas=[e.area], classes=["media_player"], b=None if rn in c.msgs[-1]["content"].lower() else {"classes": ["media_player"], "name": "tv"})
        if e.state == "playing":
            c.say(pick(rng, ["Yes, the {r} TV is playing {m} at {v}% volume.", "It's on, playing {m}."], r=rn, m=e.attributes.get("media_title", "something"), v=e.attributes.get("volume_pct", 30)), must=["playing"])
        elif e.state == "paused":
            c.say(pick(rng, ["It's on but paused, on {m}.", "The {r} TV is paused."], r=rn, m=e.attributes.get("media_title", "something")), must=["paused"])
        else:
            c.say(pick(rng, ["No, the {r} TV is {s}.", "It's {s}."], r=rn, s=e.state), must=[e.state])
        return c

    if kind == "garage":
        e = house.by_class("garage")[0]
        c.user(voiceify(rng.choice(["Is the garage open?", "Did I close the garage?", "Is the garage door shut?", "Is the garage closed?"]), rng, "q"))
        c.fetch(areas=[e.area], classes=["garage"])
        if e.state == "open":
            c.say(rng.choice(["The garage door is open. Want me to close it?", "It's open, has been for " + spoken_dur(e.changed_s) + ". Shall I close it?"]), must=["open"])
            if rng.random() < 0.6:
                c.user(rng.choice(["Yes.", "Yes please.", "Close it."]))
                if e.area in house.area_ids():
                    c.act([{"domain": "cover", "service": "close_cover", "target": {"area_id": [e.area]}}])
                else:
                    c.act([{"domain": "cover", "service": "close_cover", "target": {"entity_id": [e.entity_id]}}])
                c.say(rng.choice(["Closing the garage.", "Garage door is closing.", "Done."]))
            else:
                c.user(rng.choice(["No, leave it.", "Not now."]))
                c.say(rng.choice(["Okay.", "Sure."]))
        else:
            c.say(rng.choice(["The garage door is closed.", "Yes, it's closed.", "Closed."]), must=["closed"])
        return c

    if kind == "cover":
        e = rng.choice(house.by_class("cover"))
        rn = room_plain(house, e.area)
        c.user(voiceify(pick(rng, ["Are the {r} blinds closed?", "Are the blinds open in {rr}?", "Did the {r} blinds close?"], r=rn, rr=room_ref(house, e.area)), rng, "q"))
        c.fetch(areas=[e.area], classes=["cover"])
        c.say(pick(rng, ["The {r} blinds are {s}.", "They're {s}."], r=rn, s=e.state), must=[e.state])
        return c

    if kind == "alarm":
        e = house.by_class("alarm")[0]
        c.user(voiceify(rng.choice(["Is the alarm on?", "Is the alarm armed?", "What's the alarm set to?", "Is the alarm set?"]), rng, "q"))
        c.fetch(areas=["security"], classes=["alarm"])
        words = {"disarmed": "disarmed", "armed_home": "armed in home mode", "armed_away": "armed in away mode", "armed_night": "armed in night mode"}
        c.say(pick(rng, ["The alarm is {s}.", "It's {s}."], s=words.get(e.state, e.state)), must=[e.state.split("_")[0]])
        return c

    if kind == "climate":
        e = house.by_class("climate")[0]
        c.user(voiceify(rng.choice(["What's the thermostat set to?", "What's the heating set to?", "What's the aircon doing?", "Is the heating on?"]), rng, "q"))
        c.fetch(areas=[e.area], classes=["climate"])
        cur = num(e.attributes.get("current_temperature"))
        if e.state == "off":
            c.say(pick(rng, ["The thermostat is off. It's reading {c} degrees.", "It's off at the moment, and the house is at {c}."], c=cur), must=["off"])
        else:
            c.say(pick(rng, ["It's set to {t} in {m} mode, and it's currently {c} degrees.", "{M} to {t}, currently {c}."], t=num(e.attributes.get("temperature")), m=e.state, M=cap(e.state) + "ing" if e.state in ("heat", "cool") else cap(e.state), c=cur), must=[num(e.attributes.get("temperature"))])
        return c

    if kind == "heater":
        e = rng.choice(heaters)
        rn = room_plain(house, e.area)
        c.user(voiceify(pick(rng, ["Is the heater on in {rr}?", "Is the {r} heater on?", "Did I leave the {r} heater on?"], rr=room_ref(house, e.area), r=rn), rng, "q"))
        c.fetch(areas=[e.area], classes=["switch"])
        if e.state == "on":
            c.say(pick(rng, ["Yes, the {r} heater is on, drawing {w} watts. Want it off?", "It's on, {w} watts. Shall I turn it off?"], r=rn, w=e.attributes.get("power_w")), must=["on"])
            if rng.random() < 0.5:
                c.user(rng.choice(["Yes.", "Yes please.", "Turn it off."]))
                c.act([{"domain": "switch", "service": "turn_off", "target": {"entity_id": [e.entity_id]}}])
                c.say(rng.choice(["Heater's off.", "Done, heater off."]))
            else:
                c.user(rng.choice(["No, leave it.", "It's fine."]))
                c.say(rng.choice(["Okay.", "Sure."]))
        else:
            c.say(pick(rng, ["No, the {r} heater is off.", "It's off."], r=rn), must=["off"])
        return c

    if kind == "fan":
        e = rng.choice(house.by_class("fan"))
        rn = room_plain(house, e.area)
        c.user(voiceify(pick(rng, ["Is the fan on in {rr}?", "Is the {r} fan running?"], rr=room_ref(house, e.area), r=rn), rng, "q"))
        c.fetch(areas=[e.area], classes=["fan"])
        c.say(pick(rng, ["The {r} fan is {s}.", "It's {s}."], r=rn, s="on" if e.state == "on" else "off"), must=[e.state])
        return c
    return None


def status_aggregate(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    kind = rng.choice(["windows", "lights", "locked", "temps", "doors"])
    c = Conv(house, sp, "status_aggregate", rng)
    if kind == "windows":
        c.user(voiceify(rng.choice(["Are any windows open?", "Did we leave any windows open?", "Which windows are open?", "Are all the windows shut?"]), rng, "q"))
        c.fetch(classes=["window"], b={"classes": ["window"], "where": ["open"]})
        ws = house.by_class("window")
        if not ws:
            c.say(rng.choice(["I don't have any window sensors, so I can't tell.", "There are no window sensors set up, sorry."]))
            return c
        opn = [e for e in ws if e.state == "on"]
        if opn:
            names = [room_plain(house, e.area) for e in opn]
            refs = join([room_ref(house, e.area) for e in opn])
            c.say(pick(rng, ["The window{s} in {n} {v} open.", "Yes, the window{s} in {n} {v} open."], n=refs, s="s" if len(opn) > 1 else "", v=is_are(len(opn))), must=names)
        else:
            c.say(rng.choice(["No, all the windows are closed.", "Everything's shut.", "All closed."]), must=["closed|shut|no windows open|nothing open|none open|not open"])
        return c
    if kind == "doors":
        c.user(voiceify(rng.choice(["Are any doors open?", "Is a door open somewhere?", "Are the doors shut?"]), rng, "q"))
        c.fetch(classes=["door"], b={"classes": ["door"], "where": ["open"]})
        ds = house.by_class("door")
        if not ds:
            c.say("I don't have any door sensors to check.")
            return c
        opn = [e for e in ds if e.state == "on"]
        if opn:
            names = [e.name.lower() if e.area == "security" else short(e, house) for e in opn]
            c.say(pick(rng, ["The {n} {v} open.", "Yes, the {n}."], n=join(names), v=is_are(len(opn))), must=names)
        else:
            c.say(rng.choice(["No, all the doors are closed.", "All shut."]), must=["closed|shut|no doors open|nothing open|none open|not open"])
        return c
    if kind == "lights":
        c.user(voiceify(rng.choice(["Which lights are on?", "Are any lights on?", "What lights did we leave on?", "Any lights on in the house?"]), rng, "q"))
        c.fetch(classes=["light"], b={"classes": ["light"], "where": ["on"], "_rooms": True})
        on = [e for e in house.by_class("light") if e.state == "on"]
        if on:
            rooms = []
            for e in on:
                r = room_plain(house, e.area)
                if r not in rooms:
                    rooms.append(r)
            if len(on) <= 3:
                c.say(pick(rng, ["The {n} {v} on.", "{N} {v} on."], n=join([e.name.lower() for e in on]), N=cap(join([e.name.lower() for e in on])), v=is_are(len(on))), must=rooms)
            else:
                c.say(pick(rng, ["Lights are on in {r}, {k} in total.", "{k} lights are on, in {r}."], r=join(rooms), k=len(on)), must=rooms)
        else:
            c.say(rng.choice(["No lights are on.", "Everything's off.", "None, all off."]), must=["no lights|off|none"])
        return c
    if kind == "locked":
        c.user(voiceify(rng.choice(["Is the house locked up?", "Is everything locked?", "Are the doors locked?", "Are we all locked up?"]), rng, "q"))
        c.fetch(classes=["lock", "door", "garage"], b=[{"classes": ["lock", "garage"]}, {"classes": ["door"], "where": ["open"]}])
        locks = house.by_class("lock")
        unlocked = [e for e in locks if e.state == "unlocked"]
        garage = house.first("garage")
        bits, must = [], []
        if not unlocked:
            bits.append(pick(rng, ["{A} locked", "both doors are locked" if len(locks) == 2 else "the {l} {v} locked"], A="Both doors are" if len(locks) == 2 else cap(join([e.name.lower() for e in locks])) + (" is" if len(locks) == 1 else " are"), l=join([e.name.lower() for e in locks]), v=is_are(len(locks))))
            must.append("locked")
        else:
            bits.append(f"the {join([e.name.lower() for e in unlocked])} {is_are(len(unlocked))} unlocked")
            must += [e.name.lower() for e in unlocked]
        if garage:
            bits.append(f"the garage is {garage.state}")
            # a closed garage only has to be mentioned when there is nothing else to report; "all locked"
            # or "no doors open" count as saying it
            if garage.state == "open":
                must.append("open")
            elif not unlocked:
                must.append("closed|shut|locked|no doors open|nothing open|none open|not open|locked up|secure")
        opend = [e for e in house.by_class("door") if e.state == "on"]
        if opend:
            bits.append(f"the {join([e.name.lower() for e in opend])} {is_are(len(opend))} open")
        text = cap(join(bits)) + "."
        fixes = []
        for e in unlocked:
            fixes.append({"domain": "lock", "service": "lock", "target": {"entity_id": [e.entity_id]}})
        if garage and garage.state == "open":
            fixes.append({"domain": "cover", "service": "close_cover", "target": {"entity_id": [garage.entity_id]}})
        if fixes:
            text += rng.choice([" Want me to sort that?", " Shall I lock up?", " Want me to fix that?"])
        c.say(text, must=must)
        if fixes:
            if rng.random() < 0.7:
                c.user(rng.choice(["Yes.", "Yes please.", "Please.", "Go ahead."]))
                c.act(fixes)
                c.say(rng.choice(["Done, all locked up.", "Locked up.", "All secure now."]))
            else:
                c.user(rng.choice(["No, leave it.", "Not now."]))
                c.say(rng.choice(["Okay.", "Sure."]))
        return c
    # temps
    c.user(voiceify(rng.choice(["What's the temperature around the house?", "Which room is the warmest?", "How warm is it inside?", "Give me the temperatures.", "Which room is coldest?"]), rng, "q"))
    ts = [e for e in house.by_class("temperature") if e.area != "weather"]
    q = c.msgs[1]["content"].lower()
    warm = max(ts, key=lambda e: float(e.state)) if ts else None
    cold = min(ts, key=lambda e: float(e.state)) if ts else None
    # design B: the extremes by sorting when the house has more sensors than one result shows
    top = {"classes": ["temperature"], "sort": "value_desc", "limit": 3}
    bottom = {"classes": ["temperature"], "sort": "value_asc", "limit": 3}
    if "warmest" in q:
        b, need = top, [warm]
    elif "coldest" in q:
        b, need = bottom, [cold]
    elif len(house.by_class("temperature")) > 40:
        b, need = [top, bottom, {"areas": ["weather"], "classes": ["temperature"]}], [warm, cold]
    else:
        b, need = None, []
    c.fetch(classes=["temperature"], b=b, need=[e for e in need if e])
    if not ts:
        c.say("I don't have any indoor temperature sensors.")
        return c
    if "warmest" in q:
        c.say(pick(rng, ["{R} is the warmest at {t} degrees.", "Warmest is {r} at {t}."], R=cap(room_ref(house, warm.area)), r=room_ref(house, warm.area), t=num(warm.state)), must=[room_plain(house, warm.area)])
    elif "coldest" in q:
        c.say(pick(rng, ["{R} is the coldest at {t} degrees.", "Coldest is {r} at {t}."], R=cap(room_ref(house, cold.area)), r=room_ref(house, cold.area), t=num(cold.state)), must=[room_plain(house, cold.area)])
    else:
        out = house.first("temperature", "weather")
        if len(ts) <= 5:
            parts = [f"{room_plain(house, e.area)} {num(e.state)}" for e in ts]
            text = cap(join(parts)) + (f", and it's {num(out.state)} outside." if out else ".")
        else:
            text = pick(rng, ["Most rooms are between {lo} and {hi}. Warmest is {wr} at {wt}, coldest is {cr} at {ct}",
                              "Indoors ranges from {lo} in {cr} to {hi} in {wr}"],
                        lo=num(cold.state), hi=num(warm.state), wr=room_ref(house, warm.area), wt=num(warm.state), cr=room_ref(house, cold.area), ct=num(cold.state))
            text += (f", and it's {num(out.state)} outside." if out else ".")
        c.say(text, must=[num(warm.state)])
    return c


def who_home(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    c = Conv(house, sp, "who_home", rng)
    home = house.home_people()
    away = house.away_people()
    kind = rng.choice(["anyone", "who", "person", "person"])
    if kind in ("anyone", "who"):
        c.user(voiceify(rng.choice(["Is anyone home?", "Who's home?", "Who is at home right now?", "Is anybody home?", "Who's in the house?"]), rng, "q"))
        others = [p for p in home if p.name != sp.name]
        if sp.channel == "satellite" and sp.name:
            if others:
                text = f"Besides you, {join([p.name for p in others])} {is_are(len(others))} home."
            else:
                text = "Just you, as far as I can tell."
        else:
            if home:
                text = f"{join([p.name for p in home])} {is_are(len(home))} home."
            else:
                text = "No one's home right now."
        if away and rng.random() < 0.6:
            text += " " + join([f"{p.name} left {spoken_ago(p.changed_s)}" for p in away]) + "."
        c.say(text, must=[p.name for p in others] if (sp.channel == "satellite" and sp.name) else [p.name for p in home])
        return c
    p = rng.choice([x for x in house.people if x.name != sp.name] or house.people)
    c.user(voiceify(pick(rng, ["Is {n} home?", "Is {n} here?", "Is {n} in?", "Is {n} at home?"], n=p.name), rng, "q"))
    if p.home:
        c.say(pick(rng, ["Yes, {ns} home.", "{n} is home.", "Yep, {ns} here."], n=p.name, ns=isv(p.name)), must=["home|here|yes"])
    else:
        c.say(pick(rng, ["No, {ns} out. They left {t}.", "{n} isn't home, been gone {t2}.", "No, {n} left {t}."], n=p.name, ns=isv(p.name), t=spoken_ago(p.changed_s), t2=spoken_dur(p.changed_s)), must=["no|out|left|isn't"])
    return c
