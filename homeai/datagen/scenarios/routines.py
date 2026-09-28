"""Multi-step routines: check on the house while away, goodnight, leaving."""
from __future__ import annotations

import random
from typing import Optional

from ...house.generator import regen
from ...house.schema import House, Speaker
from ...house.snapshot import num, rel_time
from ..common import Conv, cap, is_are, join, pick, room_plain, room_ref, rooms_ref, spoken_ago, voiceify
from .energy import EnergyView

CHECK_CLASSES = ["lock", "door", "window", "garage", "alarm", "light", "motion", "occupancy", "media_player", "battery", "sun"]
# design B: only what needs attention (on, open, unlocked), the alarm and the battery outlook, and the loads worth shedding.
# Lights go on their own: at big-house scale they can fill a result, and the answer only needs their rooms
LIGHTS_ON = {"classes": ["light"], "where": ["on"], "_rooms": True}
CHECK_B = [LIGHTS_ON, {"classes": ["lock", "door", "window", "garage", "media_player", "motion", "occupancy"], "where": ["on", "open", "unlocked"]},
           {"classes": ["alarm", "battery", "sun"]},
           {"classes": ["switch"], "where": ["power>0"], "sort": "power_desc"}]


def _everyone_away(house: House, rng: random.Random, keep_home: Optional[str] = None):
    for p in house.people:
        if p.name == keep_home:
            continue
        p.home = False
        p.room = None
        p.changed_s = rng.randint(1200, 6 * 3600)
        e = house.entities.get(p.entity_id)
        if e:
            e.state, e.changed_s = "not_home", p.changed_s
    regen(house).place_people(house)
    for e in house.entities.values():
        if e.cls in ("motion", "occupancy") and e.state == "off":
            e.changed_s = max(e.changed_s, rng.randint(1200, 4 * 3600))


def check_house(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    variant = rng.choices(["clean", "issues", "intruder", "someone_home"], weights=[0.22, 0.4, 0.22, 0.16])[0]
    keep = None
    if variant == "someone_home" and len(house.people) >= 2:
        keep = rng.choice(house.people).name
    _everyone_away(house, rng, keep_home=keep)
    if keep:
        p = house.person(keep)
        p.home = True
        indoor = [a.area_id for a in house.areas if a.kind != "outdoor"]
        p.room = rng.choice(indoor)
        house.entities[p.entity_id].state = "home"
        regen(house).place_people(house)
    if variant == "clean":
        for e in house.entities.values():
            if e.cls == "lock":
                e.state = "locked"
            elif e.cls in ("door", "window"):
                e.state = "off"
            elif e.cls == "garage":
                e.state, e.attributes["position"] = "closed", 0
            elif e.cls == "light":
                e.state = "off"
            elif e.cls == "alarm":
                e.state = "armed_away"
            elif e.cls == "media_player" and e.state in ("playing", "paused"):
                e.state = "off"
    if variant == "intruder":
        sensed = [e for e in house.entities.values() if e.cls in ("motion", "occupancy") and house.area(e.area) and house.area(e.area).kind != "outdoor"]
        if sensed:
            s = rng.choice(sensed)
            s.state, s.changed_s = "on", rng.randint(0, 90)
    away_people = [p for p in house.people if not p.home]
    spk = Speaker(name=rng.choice(away_people).name if away_people else None, area=None, channel="app")
    c = Conv(house, spk, "check_house", rng)
    utts = ["Can you check on the house?", "How's the house looking? We're not home.", "Is everything okay at home?",
            "We've left for the weekend, can you make sure the house is secure?", "Give the house a once-over for me.",
            "We're out, is the house all locked up and okay?", "We're away for the day, can you check on the house for me?"]
    if house.now.hour >= 16:
        utts += ["We're away for the evening, can you check on the house for me?", "We're out for the night, can you check everything's okay at home?"]
    c.user(voiceify(rng.choice(utts), rng, "q"))
    c.fetch(classes=CHECK_CLASSES, b=CHECK_B)

    sentences, must, fixes, fixed_desc = [], [], [], []
    # intruder alert first
    inside_motion = [e for e in house.entities.values() if e.cls in ("motion", "occupancy") and e.state == "on"
                     and house.area(e.area) and house.area(e.area).kind != "outdoor"]
    home = house.home_people()
    if inside_motion and not home:
        r = inside_motion[0].area
        cam = house.by_class("camera")
        sentences.append(pick(rng, ["Heads up, there's motion in {r} right now and everyone's marked away.", "First thing: {r} is showing motion as of {ago}, and nobody should be home."],
                              r=room_ref(house, r), ago=spoken_ago(inside_motion[0].changed_s)))
        if cam:
            sentences.append(f"You might want to check the {cam[0].name.lower()}.")
        must.append(room_plain(house, r))
    elif home:
        sentences.append(pick(rng, ["{n} is still showing as home, by the way.", "Note {n}'s phone is still at the house."], n=join([p.name for p in home])))
        must.append(home[0].name)

    locks = house.by_class("lock")
    unlocked = [e for e in locks if e.state == "unlocked"]
    if unlocked:
        sentences.append(f"The {join([e.name.lower() for e in unlocked])} {is_are(len(unlocked))} unlocked.")
        must += [e.name.lower() for e in unlocked]
        fixes.append({"domain": "lock", "service": "lock", "target": {"entity_id": [e.entity_id for e in unlocked]}})
        fixed_desc.append("lock the " + join([e.name.lower() for e in unlocked]))
    open_doors = [e for e in house.by_class("door") if e.state == "on"]
    open_windows = [e for e in house.by_class("window") if e.state == "on"]
    if open_doors:
        sentences.append(f"The {join([e.name.lower() for e in open_doors])} {is_are(len(open_doors))} open.")
        must += [e.name.lower() for e in open_doors]
    if open_windows:
        wn = [room_plain(house, e.area) for e in open_windows]
        sentences.append(f"The window{'s' if len(wn) > 1 else ''} in {join([room_ref(house, e.area) for e in open_windows])} {is_are(len(wn))} open.")
        must += wn
    garage = house.first("garage")
    if garage and garage.state == "open":
        sentences.append("The garage door is open.")
        must.append("garage")
        fixes.append({"domain": "cover", "service": "close_cover", "target": {"entity_id": [garage.entity_id]}})
        fixed_desc.append("close the garage")
    lights_on = [e for e in house.by_class("light") if e.state == "on"]
    if lights_on:
        rooms = []
        for e in lights_on:
            if e.area not in rooms:
                rooms.append(e.area)
        sentences.append(f"Lights are on in {rooms_ref(house, rooms)}.")
        must += [room_plain(house, r) for r in rooms]
        fixes.append({"domain": "light", "service": "turn_off", "target": {"area_id": rooms}})
        fixed_desc.append("turn the lights off")
    alarm = house.first("alarm")
    if alarm and alarm.state == "disarmed":
        sentences.append("The alarm is disarmed.")
        must.append("alarm")
        if not home:
            fixes.append({"domain": "alarm_control_panel", "service": "alarm_arm_away", "target": {"entity_id": [alarm.entity_id]}})
            fixed_desc.append("arm it")
    playing = [e for e in house.by_class("media_player") if e.state in ("playing", "paused")]
    if playing:
        sentences.append(f"The {join([e.name.lower() for e in playing])} {is_are(len(playing))} still on.")
        fixes.append({"domain": "media_player", "service": "turn_off", "target": {"entity_id": [e.entity_id for e in playing]}})
        fixed_desc.append("turn the " + join([e.name.lower() for e in playing]) + " off")

    if not sentences or (len(sentences) == 1 and home and not fixes):
        parts = []
        if locks:
            parts.append("doors are locked" if len(locks) > 1 else "the front door is locked")
        if garage:
            parts.append("garage is closed")
        if house.by_class("window"):
            parts.append("no windows open")
        parts.append("lights are off")
        if alarm:
            parts.append("alarm is " + ("armed" if alarm.state != "disarmed" else "disarmed"))
        sentences.insert(0, pick(rng, ["Everything looks fine: {p}.", "All good. {P}.", "The house is fine, {p}."], p=join(parts), P=cap(join(parts))))
    elif not fixes and not inside_motion:
        sentences.insert(0, rng.choice(["Mostly fine.", "Nearly all good.", "A couple of things."]))
    elif fixes and not inside_motion:
        sentences.insert(0, rng.choice(["Mostly fine, a few things.", "Not quite locked up.", "A few things to sort."]))

    # battery outlook
    v = EnergyView(house)
    if v.direction == "discharging" and v.runtime is not None:
        if v.runtime >= v.h_to_rise + 1:
            sentences.append(f"Battery's at {num(v.soc, 0)} percent and should last the night.")
        else:
            sentences.append(f"Battery's at {num(v.soc, 0)} percent with about {num(v.runtime)} hours at the current load, so it may not make it to sunrise.")
            if v.sheddable:
                fixes.append({"domain": "switch", "service": "turn_off", "target": {"entity_id": [e.entity_id for e in v.sheddable]}})
                fixed_desc.append("turn off the " + join([e.name.lower() for e in v.sheddable]))
    else:
        sentences.append(f"Battery's at {num(v.soc, 0)} percent and {'charging' if v.direction == 'charging' else 'holding'}.")

    if fixes:
        sentences.append(pick(rng, ["Want me to {f}?", "Shall I {f}?", "I can {f} if you like."], f=join(fixed_desc)))
    c.say(" ".join(sentences), must=list(dict.fromkeys(must)) or None)

    if fixes:
        if rng.random() < 0.8:
            c.user(rng.choice(["Yes please.", "Yes, do all that.", "Please.", "Go ahead.", "Yes."]))
            c.act(fixes)
            done = "Done. " + cap(join([d.replace("turn the lights off", "lights are off").replace("lock the", "locked the").replace("close the garage", "garage is closing")
                                        .replace("arm it", "alarm is armed").replace("turn off the", "turned off the").replace("turn the", "turned the") for d in fixed_desc])) + "."
            cannot = []
            if open_windows:
                cannot.append(f"the window{'s' if len(open_windows) > 1 else ''} in {join([room_ref(house, e.area) for e in open_windows])}")
            if open_doors:
                cannot.append(f"the {join([e.name.lower() for e in open_doors])}")
            if cannot:
                done += f" I can't close {join(cannot)}, so keep that in mind."
            c.say(done)
        else:
            c.user(rng.choice(["Just lock the doors." if unlocked else "No, leave it for now.", "No, that's fine.", "Leave it."]))
            if unlocked and c.msgs[-1]["content"].startswith("Just lock"):
                c.act([fixes[0]])
                c.say(pick(rng, ["Done, {l} locked.", "{L} locked."], l=join([e.name.lower() for e in unlocked]), L=cap(join([e.name.lower() for e in unlocked]))))
            else:
                c.say(rng.choice(["Okay, I'll leave it.", "No problem.", "Sure."]))
    return c


def away_motion(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """Everyone is out; the speaker asks whether anything is moving in the house."""
    if not house.sensed_areas():
        return None
    _everyone_away(house, rng)
    intruder = rng.random() < 0.45
    if intruder:
        sensed = [e for e in house.entities.values() if e.cls in ("motion", "occupancy") and house.area(e.area) and house.area(e.area).kind != "outdoor"]
        if not sensed:
            return None
        s = rng.choice(sensed)
        s.state, s.changed_s = "on", rng.randint(0, 120)
    spk = Speaker(name=rng.choice(house.people).name, area=None, channel="app")
    c = Conv(house, spk, "away_motion", rng)
    c.user(voiceify(rng.choice(["Is anyone in the house right now?", "Is there anyone home? We're all out.", "Any movement in the house?",
                                "Is the house empty?", "Can you see anyone inside?", "Has anything moved in the house since we left?"]), rng, "q"))
    c.fetch(classes=["motion", "occupancy"], b={"classes": ["motion", "occupancy"], "where": ["on"], "_rooms": True})
    on = [e for e in house.entities.values() if e.cls in ("motion", "occupancy") and e.state == "on"]
    inside = [e for e in on if house.area(e.area) and house.area(e.area).kind != "outdoor"]
    cams = house.by_class("camera")
    if inside:
        r = inside[0].area
        text = pick(rng, ["Yes. There's motion in {r} as of {ago}, and everyone is marked away.", "Something is moving in {r}, {ago}. Nobody should be home."],
                    r=room_ref(house, r), ago=spoken_ago(inside[0].changed_s))
        if cams:
            text += f" I'd check the {cams[0].name.lower()}."
        c.say(text, must=[room_plain(house, r)])
    else:
        off = [e for e in house.entities.values() if e.cls in ("motion", "occupancy") and e.state == "off"]
        last = min(off, key=lambda e: e.changed_s) if off else None
        text = rng.choice(["No, the house is quiet.", "Nothing moving inside.", "All clear, no motion anywhere."])
        if last:
            text += f" The last activity was in {room_ref(house, last.area)}, {spoken_ago(last.changed_s)}."
            c.b_step({"classes": ["motion", "occupancy"], "where": ["off"], "sort": "recent", "limit": 3}, need=[last])
        c.say(text, must=["no|nothing|quiet|clear"])
    return c


def goodnight(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    beds = [a.area_id for a in house.areas if a.kind == "bedroom"]
    home = house.home_people()
    if not beds or not home or 5 <= house.now.hour < 20:
        return None
    p = rng.choice(home)
    p.room = rng.choice(beds)
    regen(house).place_people(house)
    spk = Speaker(name=p.name if rng.random() < 0.9 else None, area=p.room, channel="satellite")
    c = Conv(house, spk, "goodnight", rng)
    c.user(voiceify(rng.choice(["Goodnight.", "Good night house.", "I'm going to bed.", "We're off to bed.", "Night night.", "Time for bed, lock up please.", "Heading to bed, can you shut the house down?"]), rng))
    c.fetch(classes=["light", "lock", "garage", "alarm", "motion", "occupancy"],
            b=[LIGHTS_ON, {"classes": ["lock", "garage", "motion", "occupancy"], "where": ["on", "open", "unlocked"]}, {"classes": ["alarm"]}])
    occ = set(house.occupied_areas()) - {p.room}
    fixes, bits, must = [], [], []
    unlocked = [e for e in house.by_class("lock") if e.state == "unlocked"]
    if unlocked:
        fixes.append({"domain": "lock", "service": "lock", "target": {"entity_id": [e.entity_id for e in unlocked]}})
        bits.append("doors locked" if len(unlocked) > 1 else f"{unlocked[0].name.lower()} locked")
    garage = house.first("garage")
    if garage and garage.state == "open":
        fixes.append({"domain": "cover", "service": "close_cover", "target": {"entity_id": [garage.entity_id]}})
        bits.append("garage closed")
    lit = []
    left_on = []
    for e in house.by_class("light"):
        if e.state == "on" and e.area != p.room:
            if e.area in occ:
                if e.area not in left_on:
                    left_on.append(e.area)
            elif e.area not in lit:
                lit.append(e.area)
    if lit:
        fixes.append({"domain": "light", "service": "turn_off", "target": {"area_id": lit}})
        bits.append("lights off" + (" except in here" if any(e.state == "on" for e in house.in_area(p.room, "light")) else ""))
    alarm = house.first("alarm")
    if alarm and alarm.state == "disarmed":
        fixes.append({"domain": "alarm_control_panel", "service": "alarm_arm_home", "target": {"entity_id": [alarm.entity_id]}})
        bits.append("alarm set to home")
    if fixes:
        c.act(fixes)
        text = pick(rng, ["Goodnight. {b}.", "Sleep well. {b}.", "Night. {b}."], b=cap(join(bits)))
    else:
        text = rng.choice(["Everything's already locked up and dark. Goodnight.", "All locked and lights off already. Sleep well."])
    if left_on:
        text += f" I left the light{'s' if len(left_on) > 1 else ''} on in {join([room_ref(house, a) for a in left_on])} since someone's still in there."
        must += [room_plain(house, a) for a in left_on]
    c.say(text, must=must or None)
    return c


def leaving(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    home = house.home_people()
    if not home:
        return None
    p = rng.choice(home)
    if rng.random() < 0.6:
        entry = [a.area_id for a in house.areas if a.kind in ("entry", "kitchen", "living")]
        p.room = rng.choice(entry) if entry else p.room
        regen(house).place_people(house)
        spk = Speaker(name=p.name, area=p.room, channel="satellite")
    else:
        spk = Speaker(name=p.name, area=None, channel="app")
    c = Conv(house, spk, "leaving", rng)
    c.user(voiceify(rng.choice(["We're heading out.", "We're leaving now.", "Bye, off to work.", "Heading off for the day, lock up.", "We're going out for a few hours.", "I'm leaving, can you secure the house?", "Off out, see you later."]), rng))
    c.fetch(classes=["light", "lock", "garage", "alarm", "media_player", "motion", "occupancy"],
            b=[LIGHTS_ON, {"classes": ["lock", "garage", "media_player", "motion", "occupancy"], "where": ["on", "open", "unlocked"]}, {"classes": ["alarm"]}])
    others = [q for q in home if q.name != p.name]
    occ_others = set(house.occupied_areas()) - ({p.room} if spk.channel == "satellite" else set())
    fixes, bits, must = [], [], []
    unlocked = [e for e in house.by_class("lock") if e.state == "unlocked"]
    if unlocked:
        fixes.append({"domain": "lock", "service": "lock", "target": {"entity_id": [e.entity_id for e in unlocked]}})
        bits.append("doors locked" if len(unlocked) > 1 else f"{unlocked[0].name.lower()} locked")
    lit = []
    for e in house.by_class("light"):
        if e.state == "on" and not (others and e.area in occ_others) and e.area not in lit:
            lit.append(e.area)
    if lit:
        fixes.append({"domain": "light", "service": "turn_off", "target": {"area_id": lit}})
        bits.append("lights off")
    playing = [e for e in house.by_class("media_player") if e.state in ("playing", "paused") and not (others and e.area in occ_others)]
    if playing:
        fixes.append({"domain": "media_player", "service": "turn_off", "target": {"entity_id": [e.entity_id for e in playing]}})
        bits.append(join([e.name.lower() for e in playing]) + " off")
    alarm = house.first("alarm")
    note = ""
    if alarm and alarm.state == "disarmed":
        if others:
            note = f" {join([q.name for q in others])} {is_are(len(others))} still home, so I've left the alarm off."
            must += [q.name for q in others]
        else:
            fixes.append({"domain": "alarm_control_panel", "service": "alarm_arm_away", "target": {"entity_id": [alarm.entity_id]}})
            bits.append("alarm set to away")
    elif others and lit != [e.area for e in house.by_class("light") if e.state == "on"]:
        note = f" I left the lights on where {join([q.name for q in others])} {is_are(len(others))}."
    if fixes:
        c.act(fixes)
        text = pick(rng, ["See you later. {b}.", "Have a good one. {b}.", "Bye. {b}."], b=cap(join(bits))) + note
    else:
        text = rng.choice(["Everything's already locked and off. See you later.", "All secure already, have a good one."]) + note
    c.say(text, must=must or None)
    return c
