"""'Where is X' reasoning from motion/occupancy sensors, who is home, and elimination."""
from __future__ import annotations

import random
from typing import Optional

from ...house.generator import NAMES
from ...house.schema import House, Speaker
from ...house.snapshot import rel_time
from ..common import Conv, cap, is_are, isv, join, pick, room_plain, room_ref, spoken_ago, spoken_dur, voiceify


def evidence(house, e):
    """Spoken description of a presence sensor that is currently on."""
    if e.cls == "occupancy":
        return f"{room_ref(house, e.area)} has shown occupied for {spoken_dur(e.changed_s)}"
    return f"there was motion in {room_ref(house, e.area)} {spoken_ago(e.changed_s)}"

ASK = ["Where is {n}?", "Where's {n}?", "Do you know where {n} is?", "Any idea where {n} is?", "Where's {n} right now?", "Can you find {n}?"]


def presence_where(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    others = [p for p in house.people if p.name != sp.name]
    if not others:
        return None
    c = Conv(house, sp, "presence_where", rng)

    # unknown person, small share
    if rng.random() < 0.06:
        nm = rng.choice([n for n in NAMES if not house.person(n)])
        c.user(voiceify(pick(rng, ASK, n=nm), rng, "q"))
        c.say(pick(rng, ["I don't have anyone called {n} set up.", "There's no {n} in the house's people list.", "I don't track anyone named {n}, sorry."], n=nm))
        return c

    t = rng.choice(others)
    c.user(voiceify(pick(rng, ASK, n=t.name), rng, "q"))

    if not t.home:
        c.say(pick(rng, ["{ns} not home. They left {ago}.", "{n} is out, {ago2} now.", "Not here. {n}'s phone left {ago}."],
                   n=t.name, ns=isv(t.name), ago=spoken_ago(t.changed_s), ago2=spoken_dur(t.changed_s)), must=["not home|out|left|not here"])
        return c

    # target is home: fetch presence sensors house-wide
    c.fetch(classes=["motion", "occupancy"], b={"classes": ["motion", "occupancy"], "where": ["on"], "_rooms": True})
    occ = set(house.occupied_areas())
    S = sp.area if sp.channel == "satellite" else None
    others_home = [p for p in house.home_people() if p.name not in (sp.name, t.name)]
    cands = sorted(occ - ({S} if S else set()), key=lambda a: house.area_ids().index(a))
    sensors = [e for e in house.entities.values() if e.cls in ("motion", "occupancy")]
    with_you = ""
    if S and any(e.area == S for e in sensors) and rng.random() < 0.5:
        with_you = rng.choice([f" Unless they're in {room_ref(house, S)} with you, I can't separate two people on one sensor.",
                               f" That's assuming they aren't in {room_ref(house, S)} with you."])

    if not cands:
        if S and S in occ and not others_home:
            c.say(pick(rng, ["The only activity I'm seeing is in {r}, where you are, so {ns} probably right there with you, or somewhere without a sensor.",
                             "I'm not picking up motion anywhere except {r}. Either {ns} with you or in a room I can't sense."],
                       r=room_ref(house, S), ns=isv(t.name)), must=[room_plain(house, S)])
            return c
        off = [e for e in sensors if e.state == "off" and e.area != S]
        if off:
            last = min(off, key=lambda e: e.changed_s)
            c.b_step({"classes": ["motion", "occupancy"], "where": ["off"], "sort": "recent", "limit": 5}, need=[last])
            c.say(pick(rng, ["{n} is home according to their phone, but I'm not seeing motion anywhere{x}. The last activity was in {r}, {ago}.",
                             "Their phone says home, but every sensor{x2} is clear. The last activity was in {r}, {ago}, so they may be sitting still or in a room without a sensor."],
                       n=t.name, x=" other than your room" if S else "", x2=" other than yours" if S else "", r=room_ref(house, last.area), ago=spoken_ago(last.changed_s)),
                  must=[room_plain(house, last.area)])
        else:
            c.say(pick(rng, ["{ns} home, but I don't have any presence sensors to go on beyond that.",
                             "Home, according to their phone. There aren't any motion sensors to narrow it down."], ns=isv(t.name)), must=["home"])
        return c

    if len(cands) == 1:
        r = cands[0]
        rn, rr = room_plain(house, r), room_ref(house, r)
        sensor = next(e for e in sensors if e.area == r and e.state == "on")
        ev = evidence(house, sensor)
        if not others_home:
            c.say(pick(rng, ["{ns} in {rr}, most likely. It's the only room showing anyone{apart}: {ev}.",
                             "Probably {rr}. That's the only place showing activity{apart}, and no one else is home.",
                             "{RR}, I'd say. {Ev}, and nothing anywhere else{apart}."],
                       ns=isv(t.name), rr=rr, RR=cap(rr), ev=ev, Ev=cap(ev), apart=" apart from yours" if S else "") + with_you, must=[rn])
        else:
            c.say(pick(rng, ["{Ev}. That's probably {n}, but {o} {v} home too, so I can't be certain.",
                             "Someone's in {rr}: {ev}. It could be {n}, though {o} {v} also home."],
                       rr=rr, ev=ev, Ev=cap(ev), n=t.name, o=join([p.name for p in others_home]), v=is_are(len(others_home))) + with_you, must=[rn])
        return c

    names = [room_plain(house, a) for a in cands]
    refs = [room_ref(house, a) for a in cands]
    if not others_home:
        c.say(pick(rng, ["{ns} in either {a} or {b}. Both have motion and no one else is home{apart}, so I can't narrow it further.",
                         "Either {a} or {b}, both are showing activity. Nobody else is home{apart}, so it's one of those."],
                   ns=isv(t.name), a=join(refs[:-1], "or") if len(refs) > 2 else refs[0], b=refs[-1], apart=" besides you" if S else ""), must=names)
    else:
        c.say(pick(rng, ["There's motion in {l}. {n} is in one of them, but with {o} also home I can't tell which.",
                         "Activity in {l}. {o} {v} home as well, so I can't pin down which one is {n}."],
                   l=join(refs), n=t.name, o=join([p.name for p in others_home]), v=is_are(len(others_home))), must=names)
    return c


def presence_in_room(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """'Is X in the office?' -> check that room's sensor plus who else is home."""
    others = [p for p in house.people if p.name != sp.name and p.home]
    rooms = house.sensed_areas()
    if not others or not rooms:
        return None
    t = rng.choice(others)
    r = rng.choice(rooms)
    c = Conv(house, sp, "presence_in_room", rng)
    c.user(voiceify(pick(rng, ["Is {n} in {r}?", "Is {n} still in {r}?", "Has {n} gone to {r}?"], n=t.name, r=room_ref(house, r)), rng, "q"))
    c.fetch(areas=[r], classes=["motion", "occupancy"])
    sensor = next(e for e in house.in_area(r) if e.cls in ("motion", "occupancy"))
    others_home = [p for p in house.home_people() if p.name not in (sp.name, t.name)]
    if sensor.state == "on":
        ev = evidence(house, sensor)
        if not others_home and not (sp.channel == "satellite" and sp.area == r):
            c.say(pick(rng, ["Looks like it. {Ev}, and no one else is home.", "Yes. {Ev}, and it can only be {n}."],
                       Ev=cap(ev), n=t.name), must=["yes|looks like"])
        else:
            c.say(pick(rng, ["Someone is. {Ev}, but {o} {v} home too so it might not be {n}.", "There's activity in {r}, though I can't tell if it's {n} or {o}."],
                       Ev=cap(ev), r=room_ref(house, r), o=join([p.name for p in others_home]) or "you", v=is_are(len(others_home) or 1), n=t.name), must=["someone|activity"])
    else:
        c.say(pick(rng, ["Doesn't look like it. {R} has been clear for {d}.", "No, nothing in {r} for {d}."],
                   R=cap(room_ref(house, r)), r=room_ref(house, r), d=spoken_dur(sensor.changed_s)), must=["no|doesn't|clear|nothing"])
    return c
