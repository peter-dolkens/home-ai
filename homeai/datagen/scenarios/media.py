"""Music and media control at HomePod parity: play by artist/album/playlist/genre/radio/podcast in a
room or 'here', transport, volume, shuffle/repeat, what's playing, and multi-room grouping.

Modelled on Home Assistant media_player services so the bridge stays a thin mapping
(Music Assistant / Apple TV / HomePod integrations underneath):
  play_media(media_content_type, media_content_id), media_pause/play/stop, media_next_track,
  media_previous_track, volume_set(volume_level 0-1), volume_up/down, volume_mute, shuffle_set, repeat_set, join.
"""
from __future__ import annotations

import random
from typing import Optional

from ...house.generator import ALBUMS, ARTISTS, GENRES, PLAYLISTS, PODCASTS, RADIO
from ...house.schema import House, Speaker
from ..common import Conv, cap, join, pick, room_plain, room_ref, short, voiceify


def _players(house: House):
    """Speakers (not TVs) by area; only areas with exactly one player are area-targetable."""
    out = {}
    for e in house.by_class("media_player"):
        out.setdefault(e.area, []).append(e)
    return out


def _target_room(house: House, sp: Speaker, rng: random.Random, players: dict):
    """Pick the room a media command applies to and how the user refers to it.
    Returns (area_id, phrase or '' for implicit, player entity)."""
    areas = [a for a, ps in players.items() if len(ps) == 1 and house.area(a)]
    if not areas:
        return None, None, None
    if sp.channel == "satellite" and sp.area in areas and rng.random() < 0.6:
        return sp.area, "", players[sp.area][0]
    a = rng.choice(areas)
    return a, f" in {room_ref(house, a)}", players[a][0]


def media_play(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    players = _players(house)
    a, where, p = _target_room(house, sp, rng, players)
    if not a:
        return None
    c = Conv(house, sp, "media_play", rng)
    kind = rng.choices(["artist", "genre", "playlist", "radio", "podcast", "album", "music"], weights=[0.28, 0.2, 0.15, 0.12, 0.08, 0.1, 0.07])[0]
    if kind == "artist":
        who = rng.choice(ARTISTS)
        u = pick(rng, ["Play {x}{w}.", "Put on some {x}{w}.", "Play some {x}{w}.", "Can you play {x}{w}?"], x=who, w=where)
        data = {"media_content_type": "artist", "media_content_id": who}
        said = who
    elif kind == "genre":
        g = rng.choice(GENRES)
        u = pick(rng, ["Play {x}{w}.", "Put on {x}{w}.", "Play {x} please{w}.", "Can we have {x}{w}?"], x=g, w=where)
        data = {"media_content_type": "genre", "media_content_id": g.replace("some ", "").replace("something ", "")}
        said = g
    elif kind == "playlist":
        pl = rng.choice(PLAYLISTS)
        u = pick(rng, ["Play the {x} playlist{w}.", "Put on my {x} playlist{w}.", "Play {x}{w}.", "Start the {x} playlist{w}."], x=pl, w=where)
        data = {"media_content_type": "playlist", "media_content_id": pl}
        said = f"the {pl} playlist"
    elif kind == "radio":
        st = rng.choice(RADIO)
        u = pick(rng, ["Play {x}{w}.", "Put {x} on{w}.", "Tune in to {x}{w}.", "Turn on {x}{w}."], x=st, w=where)
        data = {"media_content_type": "radio", "media_content_id": st}
        said = st
    elif kind == "podcast":
        pc = rng.choice(PODCASTS)
        u = pick(rng, ["Play the {x} podcast{w}.", "Put on the latest {x}{w}.", "Play {x}{w}."], x=pc, w=where)
        data = {"media_content_type": "podcast", "media_content_id": pc}
        said = f"the {pc} podcast"
    elif kind == "album":
        al, ar = rng.choice(ALBUMS)
        u = pick(rng, ["Play {x} by {y}{w}.", "Put on the album {x}{w}.", "Play the whole of {x}{w}."], x=al, y=ar, w=where)
        data = {"media_content_type": "album", "media_content_id": f"{al} by {ar}"}
        said = al
    else:
        u = pick(rng, ["Play some music{w}.", "Put some music on{w}.", "Play something{w}.", "Music please{w}."], w=where)
        data = {"media_content_type": "playlist", "media_content_id": "Favourites"}
        said = "your favourites"
    c.user(voiceify(u, rng))
    if p.attributes.get("ma"):
        c.find([p], [{"areas": [a], "classes": ["media_player"]}])  # design B: the prompt counts MA players, the entity_id needs a lookup
        mtype = data["media_content_type"]
        ma_data = {"media_id": data["media_content_id"], "media_type": "track" if mtype == "music" else mtype}
        if mtype == "genre":
            ma_data = {"media_id": data["media_content_id"], "media_type": "playlist", "radio_mode": True}
        calls = [{"domain": "music_assistant", "service": "play_media", "target": {"entity_id": [p.entity_id]}, "data": ma_data}]
    else:
        calls = [{"domain": "media_player", "service": "play_media", "target": {"area_id": [a]}, "data": data}]
    if kind == "music" or (kind in ("artist", "genre") and rng.random() < 0.3):
        calls.append({"domain": "media_player", "service": "shuffle_set", "target": {"area_id": [a]}, "data": {"shuffle": True}})
    c.act(calls)
    rn = room_plain(house, a)
    c.say(pick(rng, ["Playing {s} in {r}.", "{S} in {r}.", "Okay, {s}, {r}."], s=said, S=cap(said), r=room_ref(house, a) if where or rng.random() < 0.5 else "here"),
          must=[said.split(" ")[0].lower().strip("'s")] if kind not in ("music",) else None)
    return c


def media_transport(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    players = _players(house)
    playing = [e for ps in players.values() for e in ps if e.state in ("playing", "paused")]
    c = Conv(house, sp, "media_transport", rng)
    action = rng.choices(["pause", "resume", "next", "previous", "stop", "stop_all", "shuffle", "repeat"], weights=[0.25, 0.15, 0.2, 0.05, 0.1, 0.12, 0.08, 0.05])[0]
    if action == "stop_all":
        c.user(voiceify(rng.choice(["Stop the music everywhere.", "Stop all the music.", "Turn off the music in the whole house.", "Pause everything.", "Silence, everywhere."]), rng))
        c.fetch(classes=["media_player"], b={"classes": ["media_player"], "where": ["on"]})
        if playing:
            svc = "media_pause" if "pause" in c.msgs[1]["content"].lower() else "media_stop"
            c.act([{"domain": "media_player", "service": svc, "target": {"entity_id": [e.entity_id for e in playing]}}])
            rooms = [room_plain(house, e.area) for e in playing]
            c.say(pick(rng, ["Done, stopped in {r}.", "Music's off in {r}.", "All quiet."], r=join(rooms)) if svc == "media_stop" else pick(rng, ["Paused in {r}.", "Everything's paused."], r=join(rooms)))
        else:
            c.say(rng.choice(["Nothing's playing anywhere.", "It's already quiet, nothing is playing."]), must=["nothing|already|quiet"])
        return c
    # room-scoped transport: prefer a player that is actually playing so the command makes sense
    cands = [e for e in playing if len(players[e.area]) == 1 and house.area(e.area)]
    if not cands:
        return None
    here = [e for e in cands if sp.channel == "satellite" and e.area == sp.area]
    p = here[0] if here and rng.random() < 0.7 else rng.choice(cands)
    where = "" if (here and p is here[0]) else f" in {room_ref(house, p.area)}"
    rn = room_plain(house, p.area)
    svc, u, reply = {
        "pause": ("media_pause", ["Pause{w}.", "Pause the music{w}.", "Pause that{w}.", "Hold the music{w}."], ["Paused.", "Paused{w2}."]),
        "resume": ("media_play", ["Resume{w}.", "Play{w}.", "Keep playing{w}.", "Unpause{w}."], ["Playing again.", "Resumed{w2}."]),
        "next": ("media_next_track", ["Skip this song{w}.", "Next track{w}.", "Skip{w}.", "Next song please{w}.", "I don't like this one, skip it{w}."], ["Skipped.", "Next track{w2}."]),
        "previous": ("media_previous_track", ["Go back a track{w}.", "Play the last song again{w}.", "Previous track{w}."], ["Going back a track.", "Previous track{w2}."]),
        "stop": ("media_stop", ["Stop the music{w}.", "Stop{w}.", "Turn the music off{w}.", "That's enough music{w}."], ["Stopped.", "Music's off{w2}."]),
        "shuffle": ("shuffle_set", ["Shuffle{w}.", "Turn shuffle on{w}.", "Mix it up{w}."], ["Shuffle's on.", "Shuffling{w2}."]),
        "repeat": ("repeat_set", ["Repeat this song{w}.", "Play this on repeat{w}.", "Loop this track{w}."], ["On repeat.", "Repeating this track{w2}."]),
    }[action]
    c.user(voiceify(pick(rng, u, w=where), rng))
    data = {"shuffle": True} if action == "shuffle" else {"repeat": "one"} if action == "repeat" else None
    call = {"domain": "media_player", "service": svc, "target": {"area_id": [p.area]}}
    if data:
        call["data"] = data
    c.act([call])
    c.say(pick(rng, reply, w2=f" in {room_ref(house, p.area)}" if where else ""))
    return c


def media_volume(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    players = _players(house)
    cands = [ps[0] for a, ps in players.items() if len(ps) == 1 and house.area(a) and ps[0].state in ("playing", "paused")]
    if not cands:
        return None
    here = [e for e in cands if sp.channel == "satellite" and e.area == sp.area]
    p = here[0] if here and rng.random() < 0.7 else rng.choice(cands)
    where = "" if (here and p is here[0]) else f" in {room_ref(house, p.area)}"
    c = Conv(house, sp, "media_volume", rng)
    action = rng.choices(["up", "down", "set", "mute", "unmute"], weights=[0.25, 0.3, 0.3, 0.1, 0.05])[0]
    if action == "set":
        v = rng.choice([10, 15, 20, 25, 30, 40, 50, 60, 70])
        c.user(voiceify(pick(rng, ["Volume {v}{w}.", "Set the volume to {v} percent{w}.", "Volume to {v}{w}.", "Turn it to {v} percent{w}."], v=v, w=where), rng))
        c.act([{"domain": "media_player", "service": "volume_set", "target": {"area_id": [p.area]}, "data": {"volume_level": round(v / 100, 2)}}])
        c.say(pick(rng, ["Volume's at {v} percent.", "{v} percent{w2}."], v=v, w2=f" in {room_ref(house, p.area)}" if where else ""), must=[str(v)])
    elif action in ("up", "down"):
        up = action == "up"
        u = rng.choice(["Turn it up{w}.", "Louder{w}.", "A bit louder{w}.", "Volume up{w}.", "Crank it{w}."] if up else ["Turn it down{w}.", "Quieter{w}.", "A bit quieter{w}.", "Volume down{w}.", "Turn it down a bit{w}.", "Too loud{w}."])
        c.user(voiceify(u.format(w=where), rng))
        c.act([{"domain": "media_player", "service": "volume_up" if up else "volume_down", "target": {"area_id": [p.area]}}])
        newv = p.attributes.get("volume_pct")
        c.say(pick(rng, ["Done, {v} percent.", "{D}.", "Okay, {v} percent{w2}."], v=newv, D="Louder" if up else "Quieter", w2=f" in {room_ref(house, p.area)}" if where else ""))
    else:
        mute = action == "mute"
        c.user(voiceify(rng.choice(["Mute{w}.", "Mute the music{w}.", "Shh{w}."] if mute else ["Unmute{w}.", "Sound back on{w}."]).format(w=where), rng))
        c.act([{"domain": "media_player", "service": "volume_mute", "target": {"area_id": [p.area]}, "data": {"is_volume_muted": mute}}])
        c.say(rng.choice(["Muted.", "Quiet."] if mute else ["Unmuted.", "Sound's back."]))
    return c


def media_what(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    players = _players(house)
    cands = [ps[0] for a, ps in players.items() if len(ps) == 1 and house.area(a)]
    if not cands:
        return None
    here = [e for e in cands if sp.channel == "satellite" and e.area == sp.area]
    p = here[0] if here and rng.random() < 0.7 else rng.choice(cands)
    where = "" if (here and p is here[0]) else f" in {room_ref(house, p.area)}"
    c = Conv(house, sp, "media_what", rng)
    c.user(voiceify(pick(rng, ["What's playing{w}?", "What song is this{w}?", "Who sings this{w}?", "What is this{w}?", "What's this song{w}?"], w=where), rng, "q"))
    c.fetch(areas=[p.area], classes=["media_player"])
    a = p.attributes
    if p.state in ("playing", "paused"):
        t, ar = a.get("media_title"), a.get("media_artist")
        if ar:
            c.say(pick(rng, ["That's {t} by {ar}.", "{t}, by {ar}.", "It's {ar}, {t}."], t=t, ar=ar), must=[ar.lower().split(" ")[-1]])
        elif a.get("media_content_type") == "radio":
            c.say(pick(rng, ["That's {t}.", "You're listening to {t}."], t=t), must=[t.lower().split(" ")[0]])
        else:
            c.say(pick(rng, ["That's {t}.", "It's {t}."], t=t), must=[t.lower().split(" ")[0]])
    else:
        c.say(pick(rng, ["Nothing's playing{w2}.", "It's quiet{w2}, nothing on."], w2=f" in {room_ref(house, p.area)}" if where else " here"), must=["nothing|quiet"])
    return c


def media_multiroom(house: House, sp: Speaker, rng: random.Random) -> Optional[Conv]:
    """'Play it in the office too' / 'play everywhere': group another speaker with the one playing."""
    players = _players(house)
    speakers = {a: ps[0] for a, ps in players.items() if len(ps) == 1 and house.area(a) and "tv" not in ps[0].name.lower()}
    playing = [e for e in speakers.values() if e.state == "playing"]
    if not playing or len(speakers) < 2:
        return None
    src = playing[0]
    others = [e for e in speakers.values() if e is not src]
    c = Conv(house, sp, "media_multiroom", rng)
    if rng.random() < 0.35:
        c.user(voiceify(rng.choice(["Play this everywhere.", "Play the music through the whole house.", "Put this on all the speakers."]), rng))
        c.fetch(classes=["media_player"])
        c.act([{"domain": "media_player", "service": "join", "target": {"entity_id": [src.entity_id]}, "data": {"group_members": [e.entity_id for e in others]}}])
        c.say(rng.choice(["Playing everywhere.", "It's on all the speakers now."]))
        return c
    dst = rng.choice(others)
    dn = room_plain(house, dst.area)
    c.user(voiceify(pick(rng, ["Play it in {r} too.", "Add {r}.", "Play this in {r} as well.", "Put the music on in {r} too."], r=room_ref(house, dst.area)), rng))
    c.fetch(classes=["media_player"])
    c.act([{"domain": "media_player", "service": "join", "target": {"entity_id": [src.entity_id]}, "data": {"group_members": [dst.entity_id]}}])
    c.say(pick(rng, ["Added {r}.", "Playing in {r} too.", "{R} is in."], r=room_ref(house, dst.area), R=cap(room_ref(house, dst.area))), must=[dn])
    return c
