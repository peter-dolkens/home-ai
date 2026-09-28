"""Reduce user utterances to slot skeletons so paraphrasing works on a few hundred patterns
instead of thousands of concrete sentences, then expand paraphrases back.

Slots: {room}, {room2}, {room3} (spoken room names), {person}, {device}, {num}.
"""
from __future__ import annotations

import json
import re
from typing import Iterable

from ..house.generator import ALBUMS, ARTISTS, GENRES, PLAYLISTS, PODCASTS, RADIO, make_house
from ..house.schema import House
from ..datagen.common import room_plain, room_ref


def slot_values(house: House, speaker_area: str | None):
    rooms = sorted({room_plain(house, a.area_id) for a in house.areas} | {room_ref(house, a.area_id) for a in house.areas}, key=len, reverse=True)
    people = sorted({p.name for p in house.people}, key=len, reverse=True)
    devices = set()
    for e in house.entities.values():
        if e.cls in ("light", "switch", "fan", "cover", "media_player", "lock", "garage", "climate"):
            n = e.name.lower()
            for a in house.areas:
                for pref in (a.name.lower() + " ", (a.spoken or "").lower() + " "):
                    if pref.strip() and n.startswith(pref):
                        n = n[len(pref):]
            devices.add(n)
    return rooms, people, sorted(devices, key=len, reverse=True)


def skeletonise(text: str, rooms, people, devices):
    """Replace concrete values with slots; returns (skeleton, mapping) or (None, None) if nothing to slot."""
    mapping = {}
    out = text
    ri = 0
    for r in rooms:
        pattern = re.compile(r"(?<![a-z])" + re.escape(r.replace("the ", "")) + r"(?![a-z])", re.I)
        if pattern.search(out):
            ri += 1
            key = "{room}" if ri == 1 else f"{{room{ri}}}"
            m = pattern.search(out)
            mapping[key] = m.group(0)
            out = pattern.sub(key, out, count=1)
    for p in people:
        pattern = re.compile(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", re.I)
        if pattern.search(out):
            mapping["{person}"] = pattern.search(out).group(0)
            out = pattern.sub("{person}", out, count=1)
            break
    for d in devices:
        if len(d) < 3:
            continue
        pattern = re.compile(r"(?<![a-z])" + re.escape(d) + r"(?![a-z])", re.I)
        if pattern.search(out) and "{device}" not in mapping:
            mapping["{device}"] = pattern.search(out).group(0)
            out = pattern.sub("{device}", out, count=1)
    if "{media}" not in mapping:
        for name in MEDIA_NAMES:
            pattern = re.compile(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])", re.I)
            mm = pattern.search(out)
            if mm:
                mapping["{media}"] = mm.group(0)
                out = pattern.sub("{media}", out, count=1)
                break
    m = re.search(r"\b\d+(\.\d+)?\b", out)
    if m:
        mapping["{num}"] = m.group(0)
        out = out[:m.start()] + "{num}" + out[m.end():]
    return out, mapping


MEDIA_NAMES = sorted({*ARTISTS, *[g.replace("some ", "").replace("something ", "") for g in GENRES], *GENRES, *PLAYLISTS, *RADIO, *PODCASTS,
                      *[f"{al} by {ar}" for al, ar in ALBUMS], *[al for al, _ in ALBUMS]}, key=len, reverse=True)


def fill(skeleton: str, mapping: dict) -> str:
    out = skeleton
    for k, v in mapping.items():
        out = out.replace(k, v)
    # "in the Amber's office" -> "in Amber's office"; "the the office" -> "the office"
    out = re.sub(r"\b[Tt]he ([A-Z][a-z]+'s )", r"\1", out)
    out = re.sub(r"\b(the) the\b", r"\1", out)
    return out


def skeleton_for_example(ex: dict, house: House | None = None):
    """house: the example's house when make_house cannot rebuild it (big houses: homeai.house.bighouse.house_for)."""
    house = house or make_house(ex["house_seed"], ex.get("profile"))
    rooms, people, devices = slot_values(house, (ex.get("speaker") or {}).get("area"))
    first = next(m["content"] for m in ex["messages"] if m["role"] == "user")
    return skeletonise(first, rooms, people, devices)


PREFIXES = ("hey house, ", "hey, ", "quick question, ", "can you ", "could you ", "please ", "ok so ", "ok ", "um, ", "right, ")
SUFFIXES = (" please", " thanks")


def normalise(skeleton: str) -> str:
    """Strip the voice-style noise the generator adds so paraphrases key on the base template."""
    t = skeleton.strip().rstrip("?.!").strip().lower()
    changed = True
    while changed:
        changed = False
        for p in PREFIXES:
            if t.startswith(p):
                t = t[len(p):]
                changed = True
        for sfx in SUFFIXES:
            if t.endswith(sfx):
                t = t[: -len(sfx)]
                changed = True
    return t.strip()
