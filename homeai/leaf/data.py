"""Training and test data for room leaves (Needle 3 fine-tune format, one single-turn example per line):

    {"query", "tools", "answers": [{"name", "arguments"}], "reasoning", "system"}

    PYTHONPATH=. python3 -m homeai.leaf.data generic      data/generated/leaf-generic-{train,val}.jsonl   rooms of many random houses
    PYTHONPATH=. python3 -m homeai.leaf.data bundook      data/generated/leaf-bundook/<area_id>-{train,val}.jsonl   every room of
                                                          bundook_full(projected=True), negatives mostly from the other rooms
    PYTHONPATH=. python3 -m homeai.leaf.data test         data/generated/leaf-test-{generic,bundook}.jsonl
    PYTHONPATH=. python3 -m homeai.leaf.data all          everything, plus token-length stats

Positives cover every device x supported action of the room (lights on / off / brightness / dimmer / brighter / except /
several at once, covers, climate modes and temperatures, relative asks, fans, media, volume, music, plugs, appliances,
the vacuum, and every reading of the read tool), phrased either raw (user style, speech-to-text noise, paraphrase bank)
or as a planner hands them over ("kitchen: bench lights off, rangehood off"); ~1 in 5 positives carries two actions
(two calls, in order). Negatives (~1 in 7, answers []) are instructions for another room (hard: the same device name
there), for a device or reading this room lacks, or off-topic / house-wide asks. Arguments only come from evidence:
every argument must be traceable to a span of the final query (checked after paraphrase and noise), optionals left
unsaid are left out, and the reasoning line quotes each span.

Training examples carry the gold tools plus retrieval-like distractors from the same room, at most five (what Needle
shows the model after retrieval) and within the 1024-token cap. Test examples carry the room's whole toolset (Needle
retrieves) and, for scoring on the end state, the house spec, the HA calls and the changed entities the reference makes.

Seeds (clear of build.py seed*100_000+i, wild 900_000+, B training 6_000_000+ / 30_000_000+, bigbench 9_000_000+):
small houses 41_000_000+ (train) / 42_000_000+ (val) / 43_000_000+ (test), big houses 47_000_000+ / 47_500_000+ /
48_000_000+, Bundook states 50_000_000+ / 50_500_000+ / 51_000_000+."""
from __future__ import annotations

import argparse
import collections
import copy
import functools
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..datagen.augment import polarity_ok
from ..datagen.common import room_plain, room_ref, voiceify
from ..datagen.skeleton import normalise
from ..house.bighouse import make_big_house
from ..house.bundook_full import make_bundook_full
from ..house.generator import make_house
from ..house.schema import House
from . import phrases as PH
from .schema import Dev, Family, Room, family_tool, outcome, read_tool, room_model, room_tools, system_fact
from .tokens import example_tokens

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "generated")
BANKS = os.path.join(os.path.dirname(__file__), "..", "..", "data", "paraphrases")
TOKEN_CAP = 1000  # under Needle's --max-len 1024, with margin for the re-implemented tokenizer
MAX_TOOLS = 5     # what Needle keeps per turn after retrieval

SEEDS = {"generic": {"train": (41_000_000, 47_000_000), "val": (42_000_000, 47_500_000), "test": (43_000_000, 48_000_000)},
         "bundook": {"train": 50_000_000, "val": 50_500_000, "test": 51_000_000}}


# ---------------------------------------------------------------------- houses
def build_house(spec: dict) -> House:
    """{"kind": "small", "seed", "profile"} | {"kind": "big", "seed", "n"} | {"kind": "bundook_projected", "seed"}."""
    k = spec["kind"]
    if k == "small":
        return make_house(spec["seed"], spec.get("profile"))
    if k == "big":
        return make_big_house(spec["seed"], spec["n"])
    return copy.deepcopy(_bundook(spec["seed"]))


@functools.lru_cache(maxsize=64)
def _bundook(seed: int) -> House:
    return make_bundook_full(seed, projected=True)


@functools.lru_cache(maxsize=2)
def load_bank(split: str) -> dict[str, list[str]]:
    path = os.path.join(BANKS, "v1-held.json" if split == "test" else "v1-train.json")
    with open(path) as f:
        return {normalise(k): v for k, v in json.load(f).items()}


# ---------------------------------------------------------------------- evidence
def norm(s: str) -> str:
    s = s.lower().replace("'", "").replace("’", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%. ]+", " ", s)).strip()


def find(text: str, cands: list[str]) -> Optional[str]:
    t = norm(text)
    for c in cands:
        c2 = norm(c)
        if c2 and re.search(r"(?<![a-z0-9])" + re.escape(c2) + r"(?![a-z0-9])", t):
            return c2
    return None


@dataclass
class Ev:
    label: str          # "action 'off'", "lights 'island pendant'", "this room"
    cands: list[str]


@dataclass
class Clause:
    kind: str
    call: Optional[dict]
    text: str
    evs: list[Ev] = field(default_factory=list)
    tool: str = ""
    single_dev: str = ""  # the device noun used, for the paraphrase bank's {device}
    num: str = ""
    q: str = ""
    intent: str = ""


def reasoning(query: str, clauses: list[Clause], neg: Optional[str]) -> Optional[str]:
    parts_by_clause = []
    for c in clauses:
        parts = []
        for ev in c.evs:
            f = find(query, ev.cands)
            if f is None:
                return None
            parts.append((norm(query).find(f), f"'{f}' -> {ev.label}"))
        parts.sort()
        parts_by_clause.append((c.tool, [p for _, p in parts]))
    if neg:
        return "; ".join(p for _, ps in parts_by_clause for p in ps) + ("; " if any(ps for _, ps in parts_by_clause) else "") + neg
    if len(parts_by_clause) == 1:
        return "; ".join(parts_by_clause[0][1]) or f"-> {parts_by_clause[0][0]}"
    return "; ".join(f"{tool}: " + ", ".join(ps) for tool, ps in parts_by_clause)


# ---------------------------------------------------------------------- phrasing context
@dataclass
class Ctx:
    room: Room
    rng: random.Random
    split: str        # train | val | test
    style: str        # raw | planner
    mode: str         # none | embed | prefix (how the room is named)
    source: str = "template"

    @property
    def ti(self) -> int:
        return 1 if self.split == "test" else 0


def choose(ctx: Ctx, table: dict, key: str) -> Optional[str]:
    opts = table.get(key)
    if not opts:
        return None
    return ctx.rng.choice(opts[ctx.ti])


def verb(ctx: Ctx, action: str, slot: str) -> str:
    table = PH.ACT if slot == "act" else PH.POL
    return ctx.rng.choice(table[action][ctx.ti])


def pct_phrase(ctx: Ctx, n: int) -> tuple[str, list[str]]:
    r = ctx.rng.random()
    if r < 0.15:
        s = f"{PH.words(n)} percent"
    elif r < 0.55:
        s = f"{n} percent"
    elif r < 0.85:
        s = f"{n}%"
    else:
        s = str(n)
    return s, [s, f"{n} percent", f"{n}%", str(n), PH.words(n)]


def deg_phrase(ctx: Ctx, t: float) -> tuple[str, list[str]]:
    ts = f"{t:g}"
    r = ctx.rng.random()
    if r < 0.12 and t == int(t):
        s = f"{PH.words(int(t))} degrees"
    elif r < 0.7:
        s = f"{ts} degrees"
    else:
        s = ts
    return s, [s, f"{ts} degrees", ts] + ([PH.words(int(t))] if t == int(t) else [])


def dev_word(ctx: Ctx, d: Dev, fam: Family) -> str:
    """The name or one of its aliases (an alias only when no other device of the family answers to it)."""
    others = {n for x in fam.devs if x is not d for n in [x.name] + x.aliases}
    al = [a for a in d.aliases if a not in others and a != d.name]
    if al and ctx.rng.random() < 0.4:
        return ctx.rng.choice(al)
    return d.name


def plural(noun: str) -> bool:
    return noun.endswith("s") and not noun.endswith("ss")


def art(noun: str) -> str:
    return noun if re.search(r"\d$", noun) else "the " + noun


ALL_NOUN = {"lights": ("lights", "light"), "blinds": ("blinds", "blind"), "windows": ("windows", "window"), "fans": ("fans", "fan"),
            "switches": ("plugs", "plug"), "garage": ("garage doors", "garage door")}


def all_phrase(ctx: Ctx, fam: Family, adj: str) -> str:
    many, one = ALL_NOUN.get(fam.kind, ("devices", "device"))
    if len(fam.devs) == 1 and fam.kind in ("lights", "fans"):
        return ctx.rng.choice([f"the {adj}{one}", f"the {adj}{many}"] if fam.kind == "lights" else [f"the {adj}{one}"])
    opts = [f"the {adj}{many}", f"all the {adj}{many}"] + ([f"every {adj}{one}"] if ctx.ti else [])
    return ctx.rng.choice(opts)


# ---------------------------------------------------------------------- combos: every device x action of a room
def combos(room: Room) -> list[tuple]:
    out: list[tuple] = []
    F = room.families
    if "lights" in F:
        f = F["lights"]
        tg = [None] + [[d] for d in f.devs] if len(f.devs) > 1 else [None]
        for t in tg:
            for a in ("on", "off"):
                out.append(("light_all_onoff" if t is None else "light_onoff", "lights", {"action": a, "devs": t}))
            out += [("light_bright", "lights", {"devs": t}), ("light_dimmer", "lights", {"devs": t}), ("light_brighter", "lights", {"devs": t})]
        out.append(("light_on_bright", "lights", {"devs": None}))
        if len(f.devs) > 1:
            for d in f.devs:
                for a in ("on", "off"):
                    out.append(("light_except", "lights", {"action": a, "except": [d]}))
            for a in ("on", "off"):
                out.append(("light_onoff", "lights", {"action": a, "devs": "pair"}))
    for k in ("blinds", "windows"):
        if k in F:
            f = F[k]
            tg = [None] + ([[d] for d in f.devs] if len(f.devs) > 1 else [])
            for t in tg:
                for a in ("open", "close"):
                    out.append(("cover_move", k, {"action": a, "devs": t}))
                out.append(("cover_pos", k, {"devs": t}))
            out.append(("cover_half", k, {"devs": None}))
            if k == "blinds":
                out.append(("cover_stop", k, {"devs": None}))
    if "garage" in F:
        for a in ("open", "close"):
            out.append(("cover_move", "garage", {"action": a, "devs": None}))
    if "aircon" in F:
        f = F["aircon"]
        tg = [None] if len(f.devs) == 1 else [[d] for d in f.devs]
        modes = ["heat", "cool", "auto"] if f.tool == "set_thermostat" else ["heat", "cool", "auto", "dry", "fan only"]
        for t in tg:
            out += [("clim_onoff", "aircon", {"action": "on", "devs": t}), ("clim_onoff", "aircon", {"action": "off", "devs": t}),
                    ("clim_temp", "aircon", {"devs": t}), ("clim_on_temp", "aircon", {"devs": t}), ("clim_warmer", "aircon", {"devs": t}),
                    ("clim_cooler", "aircon", {"devs": t})]
            out += [("clim_mode", "aircon", {"mode": m, "devs": t}) for m in modes]
            out.append(("clim_mode_temp", "aircon", {"devs": t}))
    if "floor_heating" in F:
        for c in ("clim_temp", "clim_warmer", "clim_cooler"):
            out.append((c, "floor_heating", {"devs": None}))
        out += [("clim_onoff", "floor_heating", {"action": "on", "devs": None}), ("clim_onoff", "floor_heating", {"action": "off", "devs": None})]
    if "fans" in F:
        f = F["fans"]
        tg = [None] + ([[d] for d in f.devs] if len(f.devs) > 1 else [])
        for t in tg:
            for a in ("on", "off"):
                out.append(("fan_onoff", "fans", {"action": a, "devs": t}))
            for s in ("low", "medium", "high"):
                out.append(("fan_speed", "fans", {"speed": s, "devs": t}))
    if "media" in F:
        f = F["media"]
        tg = [None] if len(f.devs) == 1 else [None] + [[d] for d in f.devs]
        for t in tg:
            for a in ("play", "pause", "stop", "next", "previous", "turn on", "turn off", "mute", "unmute"):
                out.append((f"media_{a}", "media", {"action": a, "devs": t}))
            out += [("vol_set", "volume", {"devs": t}), ("vol_louder", "volume", {"devs": t}), ("vol_quieter", "volume", {"devs": t})]
    if "music" in F:
        for k in ("artist", "genre", "playlist", "album", "radio", "track"):
            out.append(({"artist": "music_any", "genre": "music_any"}.get(k, f"music_{k}"), "music", {"mkind": k, "devs": None}))
    if "switches" in F:
        f = F["switches"]
        for d in f.devs:
            for a in ("on", "off"):
                out.append(("switch_onoff", "switches", {"action": a, "devs": [d]}))
        if len(f.devs) > 1:
            out += [("switch_onoff", "switches", {"action": a, "devs": "pair"}) for a in ("on", "off")]
    if "appliances" in F:
        for d in F["appliances"].devs:
            for a in ("start", "stop", "on", "off"):
                out.append(("appliance", "appliances", {"action": a, "devs": [d]}))
    if "vacuum" in F:
        out += [("vac_start", "vacuum", {"action": "start"}), ("vac_stop", "vacuum", {"action": "stop"}), ("vac_dock", "vacuum", {"action": "dock"})]
    for what in room.reads:
        out.append(("read", "read", {"what": what}))
    return out


def state_action(room: Room, fam: Family, devs, rng: random.Random, pair=("on", "off")) -> str:
    """Mostly the action that changes something (off when a target is on)."""
    ids = [i for d in (devs or fam.devs) for i in d.ids]
    ents = [room.house.entities[i] for i in ids if i in room.house.entities]
    active = any(e.state in ("on", "open", "playing", "heat", "cool", "auto", "dry", "fan_only", "cleaning") for e in ents)
    want = pair[1] if active else pair[0]
    return want if rng.random() < 0.8 else rng.choice(pair)


# ---------------------------------------------------------------------- clause builder
def fill(tpl: str, slots: dict) -> str:
    out = tpl
    for k, v in slots.items():
        out = out.replace("{" + k + "}", v)
    return re.sub(r"\s+", " ", out).replace(" ,", ",").strip()


ACTION_OF = {"light_dimmer": "dimmer", "light_brighter": "brighter", "light_on_bright": "on", "clim_on_temp": "on", "clim_warmer": "warmer",
             "clim_cooler": "cooler", "cover_stop": "stop"}
NO_ACTION = {"light_bright", "clim_temp", "clim_mode", "clim_mode_temp", "fan_speed", "vol_set", "vol_louder", "vol_quieter", "read"}
ARG_KEY = {"lights": "lights", "blinds": "blinds", "windows": "windows", "fans": "fans", "switches": "devices", "appliances": "appliance",
           "aircon": "unit", "media": "player", "volume": "player", "music": "player", "garage": "door"}
SINGLE_ARG = {"appliance", "unit", "player", "door"}


def make_clause(ctx: Ctx, combo: tuple, room_named: bool = True, natural: bool = False) -> Optional[Clause]:
    """One instruction clause for a combo, in ctx's style / split, naming ctx.room when ctx.mode == 'embed'."""
    intent, kind, p = combo
    room, rng = ctx.room, ctx.rng
    fam = room.families.get(kind) if kind != "read" else None
    evs: list[Ev] = []
    args: dict[str, Any] = {}
    slots: dict[str, str] = {}
    embed = ctx.mode == "embed" and room_named
    key = intent if intent != "read" else read_key(room, p["what"])
    # the action this clause carries (None: the template states values only)
    action = None if intent in NO_ACTION or intent.startswith("music_") else (p.get("action") or ACTION_OF.get(intent))
    if intent == "cover_half":
        action = rng.choice(["open", "close"])
    # template
    table = PH.P if ctx.style == "planner" else PH.T
    tpl = None
    if natural:
        nk = f"{key}:{action}" if f"{key}:{action}" in PH.NATURAL else key
        if nk in PH.NATURAL:
            tpl = rng.choice(PH.NATURAL[nk])
    if tpl is None:
        tpl = choose(ctx, table, key)
    if tpl is None:
        return None
    if intent == "cover_pos":
        action = "open" if "{act}" in tpl else None
    # room placement
    has_dev = any(s in tpl for s in ("{dev}", "{devn}", "{all}"))
    place = "none"
    if embed:
        if "{room}" in tpl:
            place = "room"
        elif "{inr}" in tpl and (not has_dev or rng.random() < 0.5):
            place = "inr"
        elif has_dev:
            place = "adj"
        else:
            place = "suffix"
    rp, rr = room_plain(room.house, room.area_id), room_ref(room.house, room.area_id)
    adj = rp + " " if place == "adj" else ""
    slots["inr"] = f" in {rr}" if place == "inr" else ""
    slots["room"] = rr if place == "room" else rng.choice(["this room", "the room"])
    if place != "none":
        evs.append(Ev("this room", [rp, rr, room.said]))
    # devices
    devs = p.get("devs")
    if devs == "pair":
        devs = rng.sample(fam.devs, 2)
    many = fam is not None and len(fam.devs) > 1
    if devs is None and many and has_dev and kind in ("media", "volume", "music", "aircon"):
        devs = [rng.choice(fam.devs)]  # a template that names a player / unit names one of them
    single_dev = ""
    if kind == "read":
        what = p["what"]
        args["what"] = what
        dev_obj = next(((d, f) for f in room.families.values() for d in f.devs if d.name == what), None)
        if dev_obj:
            w = dev_word(ctx, *dev_obj)
            slots["dev"], slots["devn"] = art(adj + w), adj + w
            single_dev = w
            cands = [w, dev_obj[0].name] + dev_obj[0].aliases + (["playing", "song"] if key == "read_media" else [])
            evs.append(Ev(f"what '{what}'", cands))
        else:
            if has_dev:
                return None
            evs.append(Ev(f"what '{what}'", PH.READ_EVID.get(what, [what])))
        tool = f"read_{room.area_id}"
    else:
        tool = fam.tool
        if devs:
            words_ = [dev_word(ctx, d, fam) for d in devs]
            single_dev = words_[0] if len(words_) == 1 else ""
            phrase = " and ".join(art((adj if i == 0 else "") + w) for i, w in enumerate(words_))
            slots["are"] = "are" if len(words_) > 1 or plural(words_[0]) else "is"
            argk = ARG_KEY.get(kind)
            if argk and (many or kind in ("switches", "appliances")):
                args[argk] = devs[0].name if argk in SINGLE_ARG else [d.name for d in devs]
                for d, w in zip(devs, words_):
                    evs.append(Ev(f"{argk} '{d.name}'", [w, d.name]))
        elif kind in ALL_NOUN:
            phrase = all_phrase(ctx, fam, adj)
            slots["are"] = "are" if plural(phrase.split()[-1]) else "is"
        else:
            w = dev_word(ctx, fam.devs[0], fam)
            single_dev = w
            phrase = art(adj + w)
            slots["are"] = "are" if plural(w) else "is"
        slots["dev"] = phrase
        slots["devn"] = phrase[4:] if phrase.startswith("the ") else phrase
        if "except" in p:
            d = p["except"][0]
            w = dev_word(ctx, d, fam)
            slots["all"] = all_phrase(ctx, fam, adj)
            slots["dev"], slots["devn"] = art(w), w
            args["except"] = [d.name]
            evs.append(Ev(f"except '{d.name}'", [w, d.name]))
            evs.append(Ev("leave out", PH.EVID["except"]))
        if action:
            if "{act}" in tpl:
                slots["act"] = verb(ctx, action, "act") if action in PH.ACT else action
            if "{pol}" in tpl:
                if action not in PH.POL:
                    return None
                slots["pol"] = verb(ctx, action, "pol")
    if action:
        args["action"] = action
        evs.append(Ev(f"action '{action}'", PH.EVID.get(action, [action])))
    # values
    if intent in ("light_bright", "light_on_bright"):
        n = rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 65, 70, 75, 80, 90, 100])
        slots["pct"], c = pct_phrase(ctx, n)
        args["brightness"] = n
        evs.append(Ev(f"brightness {n}", c))
    if intent == "cover_pos":
        n = rng.choice([10, 20, 25, 30, 40, 60, 70, 75, 80, 90])
        slots["pct"], c = pct_phrase(ctx, n)
        args["position"] = n
        evs.append(Ev(f"position {n}", c))
    if intent == "cover_half":
        args["position"] = 50
        evs.append(Ev("position 50", ["halfway"]))
    if intent in ("clim_temp", "clim_on_temp", "clim_mode_temp"):
        lo, hi = (19, 26) if kind == "aircon" else (18, 28)
        t = float(rng.randint(lo, hi)) + (0.5 if rng.random() < 0.15 else 0.0)
        slots["deg"], c = deg_phrase(ctx, t)
        args["temperature"] = int(t) if t == int(t) else t
        evs.append(Ev(f"temperature {args['temperature']}", c))
    if intent in ("clim_mode", "clim_mode_temp"):
        modes = ["heat", "cool", "auto"] if fam.tool == "set_thermostat" else ["heat", "cool", "auto", "dry", "fan only"]
        m = p.get("mode") or rng.choice(modes[:2])
        slots["mode"] = rng.choice(PH.EVID[m])
        args["mode"] = m
        evs.append(Ev(f"mode '{m}'", PH.EVID[m]))
    if intent == "fan_speed":
        slots["speed"] = p["speed"]
        args["speed"] = p["speed"]
        evs.append(Ev(f"speed '{p['speed']}'", [p["speed"]]))
    if intent == "vol_set":
        n = rng.choice([10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80])
        slots["pct"], c = pct_phrase(ctx, n)
        slots["n"] = str(n)
        args["volume"] = n
        evs.append(Ev(f"volume {n}", c))
    if intent in ("vol_louder", "vol_quieter"):
        ch = "louder" if intent == "vol_louder" else "quieter"
        args["change"] = ch
        evs.append(Ev(f"change '{ch}'", PH.EVID[ch]))
    if kind == "music":
        mk = p["mkind"]
        q = rng.choice(PH.MUSIC[mk][ctx.ti])
        slots["q"] = q
        args["query"] = q
        evs.append(Ev(f"query '{q}'", [q]))
        if mk in ("playlist", "album", "radio", "track"):
            args["type"] = mk
            evs.append(Ev(f"type '{mk}'", PH.EVID[mk]))
    text = fill(tpl, slots)
    if "{" in text:
        return None
    if place == "suffix":
        text += f" in {rr}"
    call = {"name": tool, "arguments": _order(args)}
    num = next((f"{args[k]:g}" if isinstance(args[k], float) else str(args[k]) for k in ("brightness", "position", "temperature", "volume") if k in args), "")
    return Clause(kind, call, text, evs, tool, single_dev, num, slots.get("q", ""), key)


ARG_ORDER = ["action", "lights", "blinds", "windows", "fans", "devices", "appliance", "unit", "player", "door", "except", "brightness", "position",
             "mode", "temperature", "speed", "volume", "change", "query", "type", "what"]


def _order(args: dict) -> dict:
    return {k: args[k] for k in sorted(args, key=lambda k: ARG_ORDER.index(k) if k in ARG_ORDER else 99)}


def read_key(room: Room, what: str) -> str:
    if f"read_{what}" in PH.T:
        return f"read_{what}"
    for kind in ("aircon", "floor_heating", "fans"):
        f = room.families.get(kind)
        if f and what in ("aircon", "thermostat", "floor heating", "fans") and kind == {"aircon": "aircon", "thermostat": "aircon",
                                                                                          "floor heating": "floor_heating", "fans": "fans"}[what]:
            return "read_climate"
    f = room.families.get("media")
    if f and any(d.name == what for d in f.devs):
        return "read_media"
    return "read_device"


# ---------------------------------------------------------------------- paraphrase bank, speech noise
def bank_rewrite(ctx: Ctx, text: str, clause: Clause) -> Optional[str]:
    bank = load_bank(ctx.split)
    rp = room_plain(ctx.room.house, ctx.room.area_id)
    t = text
    mapping = {}
    for slot, val in (("{room}", rp), ("{device}", clause.single_dev), ("{media}", clause.q)):
        if not val:
            continue
        m = re.search(r"(?<![a-z])" + re.escape(val) + r"(?![a-z])", t, re.I)
        if m:
            mapping[slot] = m.group(0)
            t = t[:m.start()] + slot + t[m.end():]
    m = re.search(r"\b\d+(\.\d+)?\b", t)
    if m:
        mapping["{num}"] = m.group(0)
        t = t[:m.start()] + "{num}" + t[m.end():]
    opts = bank.get(normalise(t))
    if not opts:
        return None
    opts = list(opts)
    ctx.rng.shuffle(opts)
    for o in opts:
        if set(re.findall(r"\{[a-z0-9]+\}", o)) != set(mapping):
            continue
        new = o
        for k, v in mapping.items():
            new = new.replace(k, v)
        new = re.sub(r"\b[Tt]he ([A-Z][a-z]+'s )", r"\1", new)
        if polarity_ok(text, new):
            return new
    return None


# bank keys (the original scenario templates) classified by the leaf intent they express, so a bank phrasing can stand in
# for any clause of that intent: train draws the key or its v1-train paraphrases, test only v1-held paraphrases that
# never occur in the train bank
Q_WORDS = ("is ", "are ", "did ", "has ", "have ", "any ", "what", "how ", "was ", "which ", "can i ", "should ", "check ", "make sure")


def classify_key(k: str) -> Optional[tuple[str, Optional[str]]]:
    """(intent, action) for a bank key, or None when it is not a single leaf action / reading we phrase."""
    slots = set(re.findall(r"\{(\w+)\}", k))
    if not slots <= {"device", "room", "num", "media"} or re.search(r"\b(and|lock|unlock|garage|irrigation|sprinklers|water|front lawn|every|all|house|tv)\b", k):
        return None
    q = k.startswith(Q_WORDS)
    has = lambda *w: any(re.search(r"\b" + x + r"\b", k) for x in w)  # noqa: E731
    if q:
        if has("temperature", "warm", "cold", "hot") and "device" not in slots:
            return ("read_temperature", None)
        if has("window", "windows") and "device" not in slots:
            return ("read_windows", None)
        if has("light", "lights") and "device" not in slots:
            return ("read_lights", None)
        if "device" in slots and has("on", "running", "going", "left on", "kicked in") and not has("open", "close", "closed", "shut", "locked"):
            return ("read_device", None)
        return None
    if "media" in slots:
        if has("playlist"):
            return ("music_playlist", None)
        if has("album", "podcast", "whole", "latest", "radio", "tune"):
            return None
        return ("music_any", None) if has("play", "put", "turn on", "have", "start") else None
    if "num" in slots:
        if has("degrees", "aircon", "temperature") and "device" not in slots and not has("heating"):
            return ("clim_temp", None)
        if has("volume"):
            return ("vol_set", None)
        if has("dim", "brighten", "brightness", "lights") or ("device" in slots and has("set")):
            return ("light_bright", None) if "%" in k or has("percent") else None
        return None
    if has("pause", "hold") and not has("play"):
        return ("media_pause", None)
    if has("skip", "next"):
        return ("media_next", None)
    if has("previous", "back a track", "last song"):
        return ("media_previous", None)
    if has("resume", "keep playing", "unpause"):
        return ("media_play", None)
    if has("mute", "shh") and not has("unmute"):
        return ("media_mute", None) if "device" not in slots else None
    if has("louder", "crank", "turn it up"):
        return ("vol_louder", None)
    if has("quieter", "turn it down", "too loud"):
        return ("vol_quieter", None)
    if has("blinds"):
        return ("cover_move", "close") if has("close", "shut") else ("cover_move", "open") if has("open") else None
    if has("aircon"):
        return ("clim_onoff", "off") if has("off") else ("clim_onoff", "on") if has("on") else None
    if has("open", "close", "start", "play", "music", "tv", "heater", "heating", "fan", "cool", "warm", "light"):
        if not (has("light") and not has("tv", "fan", "heater", "heating", "open", "close")):
            return None
    off, on = has("off", "kill"), has("on")
    if off == on:
        return None
    act = "off" if off else "on"
    if "device" in slots:
        return ("dev_onoff", act)  # light_onoff / switch_onoff / fan_onoff on a named device
    return ("light_all_onoff", act) if re.search(r"(^|the |\{room\} )lights?\b", k) else None


@functools.lru_cache(maxsize=1)
def bank_intents() -> dict[tuple, list[tuple[str, list[str], list[str]]]]:
    """(intent, action) -> [(key, train phrasings, test phrasings)]."""
    with open(os.path.join(BANKS, "v1-train.json")) as f:
        train = json.load(f)
    with open(os.path.join(BANKS, "v1-held.json")) as f:
        held = json.load(f)
    seen = {normalise(x) for k, v in train.items() for x in [k] + v}
    out: dict[tuple, list] = collections.defaultdict(list)
    for k, v in train.items():
        c = classify_key(k)
        if c:
            test = [h for h in held.get(k, []) if normalise(h) not in seen and classify_key(h) == c]
            out[c].append((k, [k] + [x for x in v if classify_key(x) == c], test))
    return dict(out)


BANK_DEV_INTENTS = {"light_onoff", "switch_onoff", "fan_onoff"}


def bank_phrase(ctx: Ctx, clause: Clause, intent: str, action: Optional[str], has_room: bool) -> Optional[str]:
    """A bank phrasing of the clause's intent with its room / device / number / music slots filled, or None."""
    key = ("dev_onoff" if intent in BANK_DEV_INTENTS else intent, action if intent in BANK_DEV_INTENTS | {"light_all_onoff", "cover_move", "clim_onoff"} else None)
    entries = bank_intents().get(key)
    if not entries:
        return None
    fills = {"{room}": room_plain(ctx.room.house, ctx.room.area_id) if has_room else None, "{device}": clause.single_dev or None,
             "{num}": clause.num or None, "{media}": clause.q or None}
    need = {k for k, v in fills.items() if v}
    opts = []
    for k, tr, te in entries:
        for ph in (te if ctx.split == "test" else tr):
            if set(re.findall(r"\{[a-z]+\}", ph)) == need and ("aircon" not in ph or clause.tool == "set_aircon"):
                opts.append(ph)
    if not opts:
        return None
    ph = ctx.rng.choice(opts)
    for k, v in fills.items():
        if v:
            ph = ph.replace(k, v)
    return re.sub(r"\b[Tt]he ([A-Z][a-z]+'s )", r"\1", ph).replace("the the ", "the ")


FILLERS = ["um", "uh", "like", "so"]


def stt_noise(text: str, rng: random.Random) -> str:
    """Speech-to-text artefacts: no punctuation or capitals, dropped articles and apostrophes, fillers, a repeated word."""
    t = text.lower()
    t = re.sub(r"[,.?!]", "", t)
    if rng.random() < 0.3:
        t = t.replace("'", "")
    w = t.split()
    if len(w) > 3 and rng.random() < 0.3 and "the" in w[1:]:
        i = w.index("the", 1)
        w.pop(i)
    if rng.random() < 0.2:
        w.insert(rng.randrange(0, max(1, len(w) // 2)), rng.choice(FILLERS))
    if len(w) > 2 and rng.random() < 0.12:
        i = rng.randrange(len(w))
        w.insert(i, w[i])
    return " ".join(w)


# ---------------------------------------------------------------------- negatives
LIGHT_POOL = ["desk lamp", "floor lamp", "bedside lamps", "pendants", "wall lights", "strip light", "mirror lights", "floodlights",
              "fan light", "night light", "track lights", "reading lamp", "cove lights", "garden spots", "table lamp"]
SWITCH_POOL = ["kettle", "coffee machine", "toaster", "heater", "electric blanket", "fish tank", "printer", "monitor", "christmas tree",
               "air purifier", "game console", "towel rail", "dehumidifier", "3d printer", "record player"]
APPL_POOL = ["dishwasher", "oven", "washing machine", "dryer", "microwave", "rangehood", "air fryer", "cooktop"]
PHANTOM = {
    "blinds": lambda rng: Family("set_blinds", "blinds", [Dev("blinds", ["blinds"], ["shades"])]),
    "windows": lambda rng: Family("set_windows", "windows", [Dev("windows", ["windows"], [])]),
    "garage": lambda rng: Family("garage_door", "garage", [Dev("garage door", ["x"], ["garage"])]),
    "aircon": lambda rng: Family("set_aircon", "aircon", [Dev("air conditioning", ["x"], ["aircon", "air con"], "aircon")]),
    "floor_heating": lambda rng: Family("set_floor_heating", "floor_heating", [Dev("floor heating", ["x"], ["underfloor heating"], "floor")]),
    "fans": lambda rng: Family("set_fans", "fans", [Dev(rng.choice(["fan", "ceiling fan", "exhaust fan"]), ["x"], [])]),
    "media": lambda rng: Family("media_control", "media", [Dev(rng.choice(["tv", "speaker"]), ["x"], [])]),
    "volume": lambda rng: Family("set_volume", "volume", [Dev(rng.choice(["tv", "speaker"]), ["x"], [])]),
    "music": lambda rng: Family("play_music", "music", [Dev("speaker", ["x"], [], "ma")]),
    "vacuum": lambda rng: Family("robot_vacuum", "vacuum", [Dev("robot vacuum", ["x"], ["vacuum"])]),
}
LACK_WORDS = {"blinds": ["blinds", "blind"], "windows": ["windows"], "fans": ["fans", "fan"], "garage": ["garage"],
              "music": ["play", "playlist", "radio", "album", "song", "put on"],
              "media": ["pause", "skip", "track", "song", "music", "resume", "playing", "mute", "sound", "unpause"],
              "volume": ["volume", "louder", "quieter", "loud", "turn it"],
              "aircon": ["aircon", "air conditioning", "temperature", "warmer", "cooler", "colder", "cool", "warm", "hot", "cold", "heat"],
              "floor_heating": ["floor heating", "temperature", "warmer", "cooler", "colder", "warm", "cold", "heat"], "vacuum": ["vacuum", "robot"]}
LACK_LABEL = {"blinds": "blinds", "windows": "motorised windows", "garage": "garage door", "aircon": "air conditioning", "floor_heating": "floor heating",
              "fans": "fan", "media": "media player", "volume": "media player", "music": "music speaker", "vacuum": "robot vacuum",
              "lights": "such light", "switches": "such device", "appliances": "such appliance", "read": "such reading"}


def room_words(room: Room) -> set[str]:
    return {norm(n) for f in room.families.values() for d in f.devs for n in [d.name] + d.aliases}


def negative(ctx: Ctx, others: list[Room]) -> Optional[tuple[list[Clause], str, str]]:
    """([clause], reason, type) for an instruction this leaf must refuse."""
    rng, room = ctx.rng, ctx.room
    r = rng.random()
    if r < 0.5 and others:
        # another room, preferably with the same kind of device (and name) this room has
        mine = {k for k in room.families}
        cands = []
        for o in others:
            if norm(o.said) in (norm(room.said), norm(room.name)):
                continue
            shared = [c for c in combos(o) if c[1] in mine or c[1] == "read"]
            if shared:
                cands.append((o, shared))
        if cands:
            o, cs = rng.choice(cands)
            words = room_words(room)
            same = [c for c in cs if any(d is not None and norm(d.name) in words for d in (c[2].get("devs") or []) if not isinstance(d, str))]
            combo = rng.choice(same) if same and rng.random() < 0.6 else rng.choice(cs)
            octx = Ctx(o, rng, ctx.split, ctx.style, rng.choice(["embed", "prefix"]) if ctx.mode == "none" else ctx.mode)
            cl = make_clause(octx, combo)
            if cl is None:
                return None
            if octx.mode == "prefix":
                cl.text = prefix(octx) + cl.text
            said = o.said
            cl.evs = [e for e in cl.evs if e.label == "this room"]
            if not cl.evs:
                cl.evs = [Ev("this room", [said, room_plain(o.house, o.area_id), o.name])]
            cl.evs[0].label = f"another room, not the {room.said}"
            cl.call = None
            return [cl], "no call", "other_room"
    if r < 0.8:
        # something this room lacks: a kind of device, a named device, a reading
        lacking = [k for k in PHANTOM if k not in room.families and not (k == "volume" and "media" in room.families)
                   and not (k in ("aircon", "floor_heating") and ({"aircon", "floor_heating"} & set(room.families)))]
        words = room_words(room)
        choice = rng.random()
        combo = None
        fam = None
        if choice < 0.5 and lacking:
            k = rng.choice(lacking)
            fam = PHANTOM[k](rng)
            cs = [c for c in combos(_with(room, k, fam)) if c[1] == k]
            combo = rng.choice(cs) if cs else None
        elif choice < 0.8:
            k = rng.choice(["lights", "switches", "appliances"])
            pool = {"lights": LIGHT_POOL, "switches": SWITCH_POOL, "appliances": APPL_POOL}[k]
            names = [n for n in pool if norm(n) not in words and not any(norm(n) in w or w in norm(n) for w in words)]
            if names:
                nm = rng.choice(names)
                fam = Family({"lights": "set_lights", "switches": "switch_devices", "appliances": "run_appliance"}[k], k, [Dev(nm, ["x"], [])])
                a = {"lights": ["on", "off"], "switches": ["on", "off"], "appliances": ["start", "stop"]}[k]
                intent = {"lights": "light_onoff", "switches": "switch_onoff", "appliances": "appliance"}[k]
                combo = (intent, k, {"action": rng.choice(a), "devs": [fam.devs[0]]})
        else:
            missing = [w for w in PH.READ_EVID if w not in room.reads and w != "lights"]
            if missing:
                combo = ("read", "read", {"what": rng.choice(missing)})
        if combo is None:
            return None
        k = combo[1]
        proom = _with(room, k, fam) if fam else _with_read(room, combo[2]["what"])
        pctx = Ctx(proom, rng, ctx.split, ctx.style, ctx.mode)
        cl = make_clause(pctx, combo)
        if cl is None:
            return None
        if pctx.mode == "prefix":
            cl.text = prefix(pctx) + cl.text
        keep = [e for e in cl.evs if e.label == "this room"]
        if k == "read":
            what = combo[2]["what"]
            ev = Ev(f"no {what} reading in the {room.said}", PH.READ_EVID.get(what, [what]))
        else:
            d = fam.devs[0]
            ev = Ev(f"no {LACK_LABEL.get(k, k)} in the {room.said}", ([cl.single_dev] if cl.single_dev else []) + [d.name] + d.aliases + LACK_WORDS.get(k, []))
        cl.evs = keep + [ev]
        cl.call = None
        return [cl], "no call", "lacking"
    q = rng.choice(PH.OFF_TOPIC[ctx.ti])
    cl = Clause("off_topic", None, q, [], "")
    return [cl], f"not a {room.said} device or reading; no call", "off_topic"


def _with(room: Room, kind: str, fam: Optional[Family]) -> Room:
    r = copy.copy(room)
    r.families = dict(room.families)
    if fam is not None:
        r.families[kind] = fam
    return r


def _with_read(room: Room, what: str) -> Room:
    r = copy.copy(room)
    r.reads = dict(room.reads)
    r.reads[what] = ["x"]
    return r


def prefix(ctx: Ctx) -> str:
    room, rng = ctx.room, ctx.rng
    if ctx.style == "planner":
        opts = [f"{room.name}: ", f"{room.said}: ", f"{room.said} - "] if ctx.ti == 0 else [f"[{room.said}] ", f"{room.name} -> "]
    else:
        rr = room_ref(room.house, room.area_id)
        opts = [f"in {rr}, ", f"{room.said}, "] if ctx.ti == 0 else [f"over in {rr}, ", f"for {rr}, "]
    return rng.choice(opts)


def compatible(a: dict, b: dict) -> bool:
    """Two calls that make sense in one instruction: different tools, two readings, or one tool on different named devices."""
    if a == b:
        return False
    if a["name"] != b["name"] or a["name"].startswith("read_"):
        return True
    keys = ("lights", "blinds", "windows", "fans", "devices", "appliance", "unit", "player")

    def names(c):
        out = set()
        for k in keys:
            v = c["arguments"].get(k)
            out |= set(v) if isinstance(v, list) else {v} if v else set()
        return out
    na, nb = names(a), names(b)
    return bool(na) and bool(nb) and not (na & nb)


# ---------------------------------------------------------------------- one example
@dataclass
class Gen:
    split: str
    rng: random.Random
    neg_rate: float = 1 / 7
    compound_rate: float = 0.2
    stats: collections.Counter = field(default_factory=collections.Counter)

    def example(self, room: Room, others: list[Room], combo=None, neg: Optional[bool] = None) -> Optional[dict]:
        rng = self.rng
        style = "planner" if rng.random() < 0.35 else "raw"
        mode = rng.choices(["none", "embed", "prefix"], weights=[0.3, 0.1, 0.6] if style == "planner" else [0.35, 0.45, 0.2])[0]
        test = self.split == "test"
        natural = test and style == "raw" and rng.random() < 0.25
        ctx = Ctx(room, rng, self.split, style, mode)
        source = "natural" if natural else "planner" if style == "planner" else "template"
        neg = (rng.random() < self.neg_rate) if neg is None else neg
        cs = combos(room)
        if not cs:
            return None
        negtype = None
        if neg:
            r = negative(ctx, others)
            if r is None:
                return None
            clauses, reason, negtype = r
        else:
            combo = combo or rng.choice(cs)
            combo = self._stateful(room, combo)
            c1 = make_clause(ctx, combo, natural=natural)
            if c1 is None:
                return None
            clauses = [c1]
            if rng.random() < (0.3 if style == "planner" else self.compound_rate) and not natural:
                c2combo = self._stateful(room, rng.choice(cs))
                ctx2 = Ctx(room, rng, self.split, style, "none")
                c2 = make_clause(ctx2, c2combo)
                if c2 and compatible(c1.call, c2.call):
                    clauses.append(c2)
            reason = None
        # assemble
        if len(clauses) == 1:
            text = clauses[0].text
        else:
            j = rng.choice(PH.P_JOIN[ctx.ti] if style == "planner" else PH.JOIN[ctx.ti])
            text = clauses[0].text + j + clauses[1].text
        if mode == "prefix" and not neg:
            text = prefix(ctx) + text
        orig = text
        if style == "raw" and len(clauses) == 1 and not natural and clauses[0].call is not None and rng.random() < 0.4:
            c0 = clauses[0]
            para = bank_phrase(ctx, c0, c0.intent, c0.call["arguments"].get("action"), any(e.label == "this room" for e in c0.evs))
            if para and mode == "prefix":
                para = prefix(ctx) + para
            if para is None and not test:
                para = bank_rewrite(ctx, text, c0)
            if para and polarity_ok(orig, para) and reasoning(para, clauses, reason) is not None:
                text, source = para, "bank_held" if test else "bank"
        if style == "raw":
            kind = "q" if clauses[0].kind == "read" or text.split(" ")[0].lower() in ("is", "are", "what", "what's", "how", "has", "did", "any") else "cmd"
            t2 = voiceify(text, rng, kind)
            if rng.random() < 0.35:
                t2 = stt_noise(t2, rng)
            if reasoning(t2, clauses, reason) is not None:
                text = t2
        if style == "planner" and not text[:1].isupper() and rng.random() < 0.5:
            text = text[:1].upper() + text[1:]
        rs = reasoning(text, clauses, reason)
        if rs is None:
            text = orig
            rs = reasoning(text, clauses, reason)
            if rs is None:
                self.stats["dropped_evidence"] += 1
                return None
        answers = [c.call for c in clauses if c.call is not None]
        ex = {"query": text, "answers": answers, "reasoning": rs, "system": system_fact(room)}
        ex["_meta"] = {"area_id": room.area_id, "source": source, "style": style, "negative": negtype,
                       "intent": "+".join((c.kind if c.call is None else c.call["name"]) for c in clauses), "n_calls": len(answers)}
        return ex

    def _stateful(self, room: Room, combo):
        intent, kind, p = combo
        if "action" in p and p["action"] in ("on", "off", "open", "close") and kind in room.families and (self.split == "test" or self.rng.random() < 0.5):
            fam = room.families[kind]
            devs = p.get("devs") if isinstance(p.get("devs"), list) else None
            pair = ("on", "off") if p["action"] in ("on", "off") else ("open", "close")
            p = dict(p, action=state_action(room, fam, devs, self.rng, pair))
        return (intent, kind, p)


# ---------------------------------------------------------------------- tools per example
WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "in", "on", "off", "to", "and", "or", "of", "for", "it", "is", "please", "can", "you", "turn"}


def pick_tools(room: Room, ex: dict, rng: random.Random, all_tools: list[dict]) -> list[dict]:
    """Gold tools plus distractors from the room, <= MAX_TOOLS, like Needle's retrieval: the closest tools by words."""
    if len(all_tools) <= MAX_TOOLS:
        tools = list(all_tools)
    else:
        gold = {a["name"] for a in ex["answers"]}
        q = set(WORD.findall(ex["query"].lower())) - STOP
        keep = [t for t in all_tools if t["name"] in gold]
        rest = [t for t in all_tools if t["name"] not in gold]
        scored = sorted(rest, key=lambda t: (-len(q & set(WORD.findall((t["name"] + " " + t["description"]).lower().replace("_", " ")))), rng.random()))
        while len(keep) < MAX_TOOLS and scored:
            t = scored.pop(0) if rng.random() < 0.65 else scored.pop(rng.randrange(len(scored)))
            keep.append(t)
        tools = keep
    rng.shuffle(tools)
    while example_tokens(dict(ex, tools=tools)) > TOKEN_CAP:
        gold = {a["name"] for a in ex["answers"]}
        drop = [i for i, t in enumerate(tools) if t["name"] not in gold]
        if not drop:
            return []
        tools.pop(drop[-1])
    return tools


def finish(ex: dict, tools: list[dict]) -> dict:
    out = {"query": ex["query"], "tools": tools, "answers": ex["answers"], "reasoning": ex["reasoning"], "system": ex["system"]}
    return out


# ---------------------------------------------------------------------- datasets
def rooms_of(house: House) -> list[Room]:
    out = []
    for a in house.areas:
        r = room_model(house, a.area_id)
        if r.families or r.reads:
            out.append(r)
    return out


def generic_houses(split: str, rng: random.Random):
    s_small, s_big = SEEDS["generic"][split]
    i = 0
    while True:
        if rng.random() < 0.6:
            spec = {"kind": "small", "seed": s_small + i, "profile": "bundook" if rng.random() < 0.2 else None}
        else:
            spec = {"kind": "big", "seed": s_big + 10 * i, "n": rng.choice([800, 1500, 3000, 6000])}
        i += 1
        yield spec, build_house(spec)


def gen_generic(split: str, n: int, seed: int = 1) -> tuple[list[dict], list[dict]]:
    """(examples, metas). Test examples keep the whole room toolset and their expected outcome."""
    rng = random.Random(seed * 7919 + {"train": 0, "val": 1, "test": 2}[split])
    g = Gen(split, rng)
    out, metas = [], []
    for spec, house in generic_houses(split, rng):
        rooms = rooms_of(house)
        if not rooms:
            continue
        for room in rng.sample(rooms, min(len(rooms), 3)):
            tools_all = room_tools(house, room.area_id, room)
            for _ in range(rng.randint(3, 7) if split != "test" else rng.randint(1, 3)):
                ex = g.example(room, [r for r in rooms if r.area_id != room.area_id])
                if ex is None:
                    continue
                row = _row(ex, room, tools_all, spec, split, rng, g)
                if row is None:
                    continue
                out.append(row[0])
                metas.append(row[1])
                if len(out) >= n:
                    return out, metas
    return out, metas


def _row(ex: dict, room: Room, tools_all: list[dict], spec: dict, split: str, rng: random.Random, g: Gen):
    meta = ex.pop("_meta")
    meta["house"] = spec
    if split == "test":
        oc = outcome(room.house, room.area_id, ex["answers"])
        if oc["errors"]:
            g.stats["dropped_exec_error"] += 1
            return None
        # every action must change the room, or end-state scoring could not tell a right call from no call
        if any(not a["name"].startswith("read_") and not outcome(room.house, room.area_id, [a])["changes"] for a in ex["answers"]):
            g.stats["dropped_no_change"] += 1
            return None
        row = dict(finish(ex, tools_all), **{"area_id": room.area_id, "house": spec, "meta": meta,
                                             "expected": {"ha": oc["ha"], "changes": oc["changes"], "reads": oc["reads"]}})
        return row, meta
    tools = pick_tools(room, ex, rng, tools_all)
    if not tools:
        g.stats["dropped_tokens"] += 1
        return None
    oc = outcome(room.house, room.area_id, ex["answers"])
    if oc["errors"]:
        g.stats["dropped_exec_error"] += 1
        return None
    return finish(ex, tools), meta


def bundook_size(room: Room) -> int:
    return max(600, min(1500, 350 + 10 * len(combos(room))))


def gen_bundook_room(area_id: str, split: str, n: int, seed: int = 1) -> tuple[list[dict], list[dict]]:
    """Examples for one Bundook room: a coverage pass over every combo (train), then random fill; states from many seeds."""
    rng = random.Random(seed * 104_729 + hash(area_id) % 10_000 + {"train": 0, "val": 1, "test": 2}[split])
    base = SEEDS["bundook"][split]
    g = Gen(split, rng)
    out, metas = [], []
    k = 0
    passes = 0
    while len(out) < n and passes < 50:
        spec = {"kind": "bundook_projected", "seed": base + k % 40}
        k += 1
        house = build_house(spec)
        rooms = rooms_of(house)
        room = next(r for r in rooms if r.area_id == area_id)
        others = [r for r in rooms if r.area_id != area_id]
        tools_all = room_tools(house, area_id, room)
        cs = combos(room)
        todo = list(cs) if (split == "train" and passes < 2) else [None] * min(len(cs), 40)
        passes += 1
        rng.shuffle(todo)
        for combo in todo:
            ex = g.example(room, others, combo=combo, neg=False if combo is not None else None)
            if ex is None:
                continue
            row = _row(ex, room, tools_all, spec, split, rng, g)
            if row:
                out.append(row[0])
                metas.append(row[1])
            if combo is not None and rng.random() < g.neg_rate * 1.2:  # keep the negative share during the coverage pass
                ex = g.example(room, others, neg=True)
                row = _row(ex, room, tools_all, spec, split, rng, g) if ex else None
                if row:
                    out.append(row[0])
                    metas.append(row[1])
            if len(out) >= n:
                break
    return out[:n], metas[:n]


def write_jsonl(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stats(rows: list[dict], metas: list[dict], label: str) -> dict:
    toks = sorted(example_tokens(r) for r in rows)
    c = collections.Counter()
    for m in metas:
        c["negative" if m["negative"] else "positive"] += 1
        if m["negative"]:
            c["neg:" + m["negative"]] += 1
        c["src:" + m["source"]] += 1
        c[f"calls:{m['n_calls']}"] += 1
    q = lambda p: toks[min(len(toks) - 1, int(p * len(toks)))] if toks else 0  # noqa: E731
    s = {"file": label, "n": len(rows), "tokens_p50": q(0.5), "tokens_p95": q(0.95), "tokens_max": toks[-1] if toks else 0,
         "over_1024": sum(1 for t in toks if t > 1024), **dict(sorted(c.items()))}
    print(json.dumps(s))
    return s


def dedupe_against(test: list[dict], metas: list[dict], train_queries: set[str]):
    keep = [(r, m) for r, m in zip(test, metas) if norm(r["query"]) not in train_queries]
    return [r for r, _ in keep], [m for _, m in keep]


def run_generic(n_train: int, n_val: int, out: str = OUT):
    res = []
    for split, n in (("train", n_train), ("val", n_val)):
        rows, metas = gen_generic(split, n)
        write_jsonl(os.path.join(out, f"leaf-generic-{split}.jsonl"), rows)
        write_jsonl(os.path.join(out, f"leaf-generic-{split}.meta.jsonl"), metas)
        res.append(stats(rows, metas, f"leaf-generic-{split}"))
    return res


def run_bundook(out: str = OUT, only: Optional[str] = None):
    house = build_house({"kind": "bundook_projected", "seed": SEEDS["bundook"]["train"]})
    res = []
    for room in rooms_of(house):
        if only and room.area_id != only:
            continue
        n = bundook_size(room)
        for split, m in (("train", n), ("val", max(60, n // 10))):
            rows, metas = gen_bundook_room(room.area_id, split, m)
            write_jsonl(os.path.join(out, "leaf-bundook", f"{room.area_id}-{split}.jsonl"), rows)
            write_jsonl(os.path.join(out, "leaf-bundook", f"{room.area_id}-{split}.meta.jsonl"), metas)
            res.append(stats(rows, metas, f"leaf-bundook/{room.area_id}-{split}"))
    return res


def _queries(paths: list[str]) -> set[str]:
    out = set()
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                out |= {norm(json.loads(l)["query"]) for l in f}
    return out


def run_test(n_generic: int = 600, per_room: int = 40, out: str = OUT):
    rows, metas = gen_generic("test", n_generic + 100)
    rows, metas = dedupe_against(rows, metas, _queries([os.path.join(out, "leaf-generic-train.jsonl")]))
    rows, metas = rows[:n_generic], metas[:n_generic]
    for i, r in enumerate(rows):
        r["id"] = f"lg-{i}"
    write_jsonl(os.path.join(out, "leaf-test-generic.jsonl"), rows)
    res = [stats(rows, metas, "leaf-test-generic")]
    house = build_house({"kind": "bundook_projected", "seed": SEEDS["bundook"]["test"]})
    brow, bmeta = [], []
    for room in rooms_of(house):
        r_, m_ = gen_bundook_room(room.area_id, "test", per_room + 15)
        seen = _queries([os.path.join(out, "leaf-bundook", f"{room.area_id}-train.jsonl")])
        r_, m_ = dedupe_against(r_, m_, seen)
        brow += r_[:per_room]
        bmeta += m_[:per_room]
    for i, r in enumerate(brow):
        r["id"] = f"lb-{i}"
    write_jsonl(os.path.join(out, "leaf-test-bundook.jsonl"), brow)
    res.append(stats(brow, bmeta, "leaf-test-bundook"))
    # the full room toolset is what the test prompts carry; Needle keeps only the retrieved five per turn
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["generic", "bundook", "test", "all"])
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--val", type=int, default=800)
    ap.add_argument("--room", default=None, help="bundook: just this area_id")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.what in ("generic", "all"):
        run_generic(a.train, a.val, a.out)
    if a.what in ("bundook", "all"):
        run_bundook(a.out, a.room)
    if a.what in ("test", "all"):
        run_test(out=a.out)


if __name__ == "__main__":
    main()
