"""Orchestrator training data for the tiered design (homeai/tiered/schema.py, "toolset": "orch"): design-B transcripts from
the scenario builders (train_b.py's train-b split / val-b, the B test sets) rewritten so that room-level control goes
through ask_rooms instead of call_service.

For every reference call_service, the target is resolved and split by room. The part inside a real room is translated to
the room leaf's own calls (ha_to_leaf: the leaf tool and arguments that stand for the same service call), and routed when
applying those leaf calls (leaf_execute / apply_leaf) leaves the room exactly as the reference call does ("exact"), or
within the tolerances the outcome eval allows where the user gave no number ("approx": a relative dim of 20% is the leaf's
25% step, a fan at 50% is "medium"), or differing only in non-primary entities the leaf never sees ("housekeeping": an
area call also switches the "device"-class light entity of a Hue dimmer module). The routed calls of one assistant turn become ONE ask_rooms call with one request per
room, the instruction written from the leaf calls by a fixed template (phrase_calls: "lights off", "set the aircon to 22",
"play the artist Missy Higgins"), and its result produced by running the pipeline's own ask_rooms executor with an oracle
leaf that returns those calls. House-level calls (Security, Energy, Water, Weather), and room calls no leaf tool can
express (locks, speaker grouping, shuffle / repeat, play_media on players not marked [MA], light group members targeted on
their own, pumps and irrigation) stay as design-B call_service calls. find_devices / get_state stay as they are
(--route-reads also routes single-room readings that a room's read tool covers, as "read presence and temperature").
Every tool result is recomputed by replaying the new transcript, so later reads see the house the leaves left.

    PYTHONPATH=. python3 -m homeai.tiered.data            # train-b-split / val-b / test-v2-b -> train- / val- / test-orch.jsonl
    PYTHONPATH=. python3 -m homeai.tiered.data --data data/generated/bigbench-b.jsonl --out data/generated/bigbench-orch.jsonl

Each output row keeps the B row's fields plus "toolset": "orch", "leaf_gold" ([{area_id, instruction, calls}]: the leaf
calls behind each instruction, for oracle runs and leaf training on orchestrator-style instructions) and "orch_notes"
(what could not be routed or was approximated)."""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional

from ..eval.e2e import entity_match
from ..house.bighouse import house_for
from ..house.schema import GROUPS, House, Speaker
from ..house.sim import apply_service, resolve_target
from ..leaf.schema import SPEED, Family, Room, apply_leaf, room_model
from ..tools_b import execute_tool_b, ids_in, primary
from .pipeline import LeafPicker, OracleLeaf, ask_rooms_text, run_leaves
from .schema import rebind, phrase_calls, room_copy, system_prompt_orch

OUT = "data/generated"
DEFAULT_JOBS = [("train-b-split.jsonl", "train-orch.jsonl"), ("val-b.jsonl", "val-orch.jsonl"), ("test-v2-b.jsonl", "test-orch.jsonl")]

FAMILIES = {"light": ("lights",), "switch": ("switches", "appliances"), "fan": ("fans",), "cover": ("blinds", "windows", "garage"),
            "climate": ("aircon", "floor_heating"), "media_player": ("media", "volume"), "music_assistant": ("music",), "vacuum": ("vacuum",)}
MEDIA = {"media_play": "play", "media_pause": "pause", "media_stop": "stop", "media_next_track": "next", "media_previous_track": "previous",
         "turn_on": "turn on", "turn_off": "turn off"}
ARRAY_KEY = {"lights": "lights", "blinds": "blinds", "windows": "windows", "fans": "fans"}
SINGLE_KEY = {"aircon": "unit", "media": "player", "volume": "player", "music": "player", "garage": "door"}
MODES = ("heat", "cool", "auto", "dry", "fan only")
CLASS_WHAT = {"occupancy": "presence", "motion": "presence", "temperature": "temperature", "humidity": "humidity", "light": "lights",
              "cover": "blinds", "window": "windows", "door": "doors", "air_quality": "air quality", "illuminance": "light level",
              "power": "power use", "safety": "smoke and leaks", "fan": "fans"}


# ---------------------------------------------------------------------- service call -> leaf calls
def _per_dev(f: Family, sel: list, args: dict) -> list[dict]:
    """Families whose device argument is one name: one call for all of them, or one call per device."""
    key = SINGLE_KEY.get(f.kind)
    if len(f.devs) == 1 or len(sel) == len(f.devs) or key is None:
        return [{"name": f.tool, "arguments": dict(args)}]
    return [{"name": f.tool, "arguments": {**args, key: d.name}} for d in sel]


def _arr(f: Family, sel: list, args: dict) -> list[dict]:
    key = ARRAY_KEY[f.kind]
    if len(f.devs) > 1 and len(sel) < len(f.devs):
        args = {**args, key: [d.name for d in sel]}
    return [{"name": f.tool, "arguments": args}]


def _family_calls(f: Family, sel: list, domain: str, service: str, data: dict) -> Optional[list[dict]]:
    k = f.kind
    if k == "lights":
        if service == "turn_off":
            return _arr(f, sel, {"action": "off"})
        if service != "turn_on":
            return None
        if data.get("brightness_pct") is not None:
            b = int(data["brightness_pct"])
            return _arr(f, sel, {"brightness": b}) if b >= 1 else _arr(f, sel, {"action": "off"})
        if data.get("brightness") is not None:
            return _arr(f, sel, {"brightness": max(1, round(int(data["brightness"]) / 255 * 100))})
        if data.get("brightness_step_pct") is not None:
            return _arr(f, sel, {"action": "dimmer" if float(data["brightness_step_pct"]) < 0 else "brighter"})
        return _arr(f, sel, {"action": "on"})
    if k in ("switches", "appliances"):
        if service not in ("turn_on", "turn_off"):
            return None
        on = service == "turn_on"
        if k == "switches":
            return [{"name": f.tool, "arguments": {"action": "on" if on else "off", "devices": [d.name for d in sel]}}]
        return [{"name": f.tool, "arguments": {"action": "start" if on else "stop", "appliance": d.name}} for d in sel]
    if k == "fans":
        pct = data.get("percentage")
        if service == "turn_off" or (pct is not None and int(pct) == 0):
            return _arr(f, sel, {"action": "off"})
        if pct is not None and service in ("turn_on", "set_percentage"):
            speed = min(SPEED, key=lambda s: abs(SPEED[s] - int(pct)))
            return _arr(f, sel, {"speed": speed})
        return _arr(f, sel, {"action": "on"}) if service == "turn_on" else None
    if k in ("blinds", "windows", "garage"):
        if service in ("open_cover", "close_cover"):
            act = "open" if service == "open_cover" else "close"
            return _per_dev(f, sel, {"action": act}) if k == "garage" else _arr(f, sel, {"action": act})
        if service == "set_cover_position" and k != "garage":
            return _arr(f, sel, {"position": int(data.get("position", 50))})
        if service == "stop_cover" and k == "blinds":
            return _arr(f, sel, {"action": "stop"})
        return None
    if k in ("aircon", "floor_heating"):
        mode = (data.get("hvac_mode") or "").replace("_", " ")
        if service == "set_temperature":
            args: dict[str, Any] = {"temperature": float(data.get("temperature", 21))}
            if mode and k == "aircon":
                if mode not in MODES:
                    return None
                args["mode"] = mode
            elif mode and mode != "heat":
                return None
            return _per_dev(f, sel, args)
        if service == "set_hvac_mode":
            if mode == "off":
                return _per_dev(f, sel, {"action": "off"})
            if k == "floor_heating":
                return _per_dev(f, sel, {"action": "on"}) if mode == "heat" else None
            return _per_dev(f, sel, {"mode": mode}) if mode in MODES else None
        if service in ("turn_on", "turn_off"):
            return _per_dev(f, sel, {"action": "on" if service == "turn_on" else "off"})
        return None
    if k == "media":
        if service == "volume_mute":
            return _per_dev(f, sel, {"action": "mute" if data.get("is_volume_muted", True) else "unmute"})
        return _per_dev(f, sel, {"action": MEDIA[service]}) if service in MEDIA else None
    if k == "volume":
        if service == "volume_set":
            return _per_dev(f, sel, {"volume": int(round(float(data.get("volume_level", 0.3)) * 100))})
        if service in ("volume_up", "volume_down"):
            return _per_dev(f, sel, {"change": "louder" if service == "volume_up" else "quieter"})
        return None
    if k == "music":
        if service != "play_media" or not data.get("media_id"):
            return None
        args = {"query": str(data["media_id"])}
        t = "genre" if data.get("radio_mode") else data.get("media_type")
        if t in ("artist", "album", "track", "playlist", "radio", "genre"):
            args["type"] = t
        return _per_dev(f, sel, args)
    if k == "vacuum":
        svc = {"start": "start", "stop": "stop", "return_to_base": "dock"}.get(service)
        return [{"name": f.tool, "arguments": {"action": svc}}] if svc else None
    return None


def ha_to_leaf(room: Room, domain: str, service: str, data: dict, ids: list[str]) -> tuple[Optional[list[dict]], str]:
    """The room leaf calls that stand for one service call on these (resolved, in-room) entity ids, or (None, why).
    Only primary entities need a leaf device: an area target also hits the housekeeping entities of that domain (the
    "device"-class light of a Hue dimmer module), which leaves never see; verify() decides whether that matters."""
    kinds = FAMILIES.get(domain)
    if not kinds:
        return None, f"no leaf tool for {domain}"
    if domain == "media_player":
        kinds = ("volume",) if service.startswith("volume_") and service != "volume_mute" else ("media",)
    house = room.house
    want = {i for i in ids if i not in house.entities or primary(house.entities[i])}
    covered: set[str] = set()
    calls: list[dict] = []
    for kind in kinds:
        f = room.fam(kind)
        if not f:
            continue
        sel = [d for d in f.devs if want & set(d.ids)]
        if not sel:
            continue
        covered |= {i for d in sel for i in d.ids} & want
        got = _family_calls(f, sel, domain, service, data)
        if got is None:
            return None, f"no leaf tool for {domain}.{service}"
        calls += got
    if want - covered:
        cls = sorted({i.split(".")[0] for i in want - covered})
        return None, f"{domain}.{service} on devices no leaf tool reaches ({', '.join(cls)})"
    return calls, ""


def _room_snap(house: House, area_id: str) -> dict:
    return {e.entity_id: (e.state, e.attributes) for e in house.in_area(area_id)}


def verify(house: House, room: Room, area_id: str, domain: str, service: str, data: dict, ids: list[str], leaf: list[dict],
           user_text: str) -> str:
    """"exact" / "approx" (within the outcome eval's tolerances) / "" when the leaf calls do not do what the call does."""
    a = room_copy(house, area_id)
    apply_service(a, domain, service, {"entity_id": ids}, data)
    b = room_copy(house, area_id)
    res = apply_leaf(b, area_id, leaf, rebind(room, b))
    if res["errors"]:
        return ""
    ra, rb = _room_snap(a, area_id), _room_snap(b, area_id)
    if ra == rb:
        return "exact"
    how = "approx"
    for eid in ra:
        if ra[eid] != rb[eid] and not entity_match(eid, ra[eid], rb[eid], user_text)[0]:
            if primary(house.entities[eid]):
                return ""
            how = "housekeeping"  # only a non-primary entity differs (a dimmer module's own light entity)
    return how


# ---------------------------------------------------------------------- one example
class Scratch:
    """The house as the reference calls of the current turn leave it, copying only the entities they touch."""

    def __init__(self, house: House):
        self.base = house
        self.h = copy.copy(house)
        self.h.entities = dict(house.entities)

    def apply(self, domain, service, target, data):
        ids, _ = resolve_target(self.h, domain, target)
        for i in ids:
            if self.h.entities[i] is self.base.entities[i]:
                self.h.entities[i] = copy.deepcopy(self.base.entities[i])
        apply_service(self.h, domain, service, target, data)


def route_call(s: Scratch, args: dict, user_text: str, rooms: dict[str, Room]):
    """One reference call_service -> (B calls left over, {area_id: leaf calls}, notes). Applies it to the scratch house."""
    domain, service = args.get("domain", ""), args.get("service", "")
    target, data = args.get("target") or {}, args.get("data") or {}
    house = s.h
    ids, errors = resolve_target(house, domain, target)
    notes: list[str] = []
    parts: dict[str, list[dict]] = {}
    if errors or not ids:
        s.apply(domain, service, target, data)
        return [args], parts, [f"kept (target does not resolve cleanly): {domain}.{service}"]
    by_area: dict[str, list[str]] = {}
    for i in ids:
        by_area.setdefault(house.entities[i].area, []).append(i)
    left_areas, left_ids = [], []
    for aid, aids in by_area.items():
        if aid in GROUPS or aid not in house.area_ids():
            left_areas.append(aid)
            left_ids += aids
            continue
        room = rooms.get(aid) or rooms.setdefault(aid, room_model(s.base, aid))
        leaf, why = ha_to_leaf(room, domain, service, data, aids)
        how = verify(house, room, aid, domain, service, data, aids, leaf, user_text) if leaf else ""
        if not how:
            notes.append(f"kept room call: {why or f'{domain}.{service} leaf calls do not match'}")
            left_areas.append(aid)
            left_ids += aids
            continue
        if how == "approx":
            notes.append(f"approx: {domain}.{service} {json.dumps(data, sort_keys=True)} via {json.dumps(leaf)}")
        elif how == "housekeeping":
            notes.append(f"housekeeping: {domain}.{service} also changes non-primary entities the leaf cannot reach")
        parts[aid] = leaf
    s.apply(domain, service, target, data)
    if not parts:
        return [args], parts, notes
    left = []
    if left_ids:
        t = {"area_id": [a for a in left_areas if a in house.area_ids()]} if set(target) == {"area_id"} and \
            all(a in house.area_ids() for a in left_areas) else {"entity_id": left_ids}
        left.append({**args, "target": t})
    return left, parts, notes


def route_read(house: House, args: dict, rooms: dict[str, Room]) -> Optional[tuple[str, list[dict]]]:
    """--route-reads: a get_state on one real room, by classes only, that the room's read tool covers entirely."""
    if set(args) - {"areas", "classes"}:
        return None
    areas = args.get("areas") or []
    areas = [areas] if isinstance(areas, str) else areas
    classes = args.get("classes") or []
    classes = [classes] if isinstance(classes, str) else classes
    if len(areas) != 1 or areas[0] in GROUPS or areas[0] not in house.area_ids() or not classes:
        return None
    aid = areas[0]
    room = rooms.get(aid) or rooms.setdefault(aid, room_model(house, aid))
    whats = []
    for c in classes:
        w = CLASS_WHAT.get(c) or ("aircon" if c == "climate" and "aircon" in room.reads else "thermostat" if c == "climate" else None)
        if not w or w not in room.reads:
            return None
        if w not in whats:
            whats.append(w)
    shown = ids_in(execute_tool_b(house, "get_state", args))
    if not shown <= {i for w in whats for i in room.reads[w]}:
        return None
    return aid, [{"name": f"read_{aid}", "arguments": {"what": w}} for w in whats]


def speaker_of(ex):
    sp = ex.get("speaker")
    return Speaker(sp["name"], sp["area"], sp["channel"]) if sp else None


def _args(fn):
    a = fn.get("arguments") or {}
    return json.loads(a) if isinstance(a, str) else a


def to_orch(ex: dict, route_reads: bool = False, house: Optional[House] = None) -> dict:
    """A design-B (or A-tool) example -> the orchestrator transcript of the same conversation (see the module doc).
    house: a fresh copy of house_for(ex) to replay on (it is changed), when the caller already has one."""
    house = house if house is not None else house_for(ex)
    sp = speaker_of(ex)
    rooms: dict[str, Room] = {}
    oracle = OracleLeaf()
    picker = LeafPicker(oracle)
    gold: list[dict] = []
    notes: list[str] = []
    msgs: list[dict[str, Any]] = [{"role": "system", "content": system_prompt_orch(house, sp)}]
    n = 0
    user_text = ""
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)
    try:
        for m in ex["messages"][1:]:
            if m["role"] == "tool":
                continue  # recomputed below
            if m["role"] == "user":
                user_text = m["content"]
                msgs.append(dict(m))
                continue
            if not m.get("tool_calls"):
                msgs.append(dict(m))
                continue
            s = Scratch(house)
            b_calls: list[tuple[str, dict]] = []
            reqs: dict[str, list[dict]] = {}
            for tc in m["tool_calls"]:
                name, args = tc["function"]["name"], _args(tc["function"])
                if name == "call_service":
                    left, parts, nn = route_call(s, args, user_text, rooms)
                    notes += nn
                    b_calls += [("call_service", a) for a in left]
                    for aid, calls in parts.items():
                        reqs.setdefault(aid, []).extend(calls)
                    continue
                r = route_read(s.h, args, rooms) if route_reads and name == "get_state" else None
                if r:
                    reqs.setdefault(r[0], []).extend(r[1])
                    notes.append("routed read")
                else:
                    b_calls.append((name, args))
            tcs, results = [], []
            for name, args in b_calls:
                n += 1
                tcs.append({"id": f"call_{n}", "type": "function", "function": {"name": name, "arguments": args}})
                results.append((f"call_{n}", name, execute_tool_b(house, name, args)))
            if reqs:
                requests = []
                for aid, calls in reqs.items():
                    ins = phrase_calls(rooms[aid], calls)
                    requests.append({"area_id": aid, "instruction": ins})
                    gold.append({"area_id": aid, "instruction": ins, "calls": calls})
                    oracle.gold[(aid, ins)] = calls
                n += 1
                args = {"requests": requests}
                tcs.append({"id": f"call_{n}", "type": "function", "function": {"name": "ask_rooms", "arguments": args}})
                res, _ = run_leaves(house, requests, picker, pool)
                results.append((f"call_{n}", "ask_rooms", ask_rooms_text(res)))
            msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs})
            for cid, name, content in results:
                msgs.append({"role": "tool", "tool_call_id": cid, "name": name, "content": content})
    finally:
        pool.shutdown()
    out = {k: v for k, v in ex.items() if k != "messages"}
    out.update(messages=msgs, toolset="orch", leaf_gold=gold, orch_notes=notes)
    return out


# ---------------------------------------------------------------------- files
def _one(job):
    line, route_reads = job
    ex = json.loads(line)
    try:
        return to_orch(ex, route_reads)
    except Exception as e:  # noqa: BLE001 - reported, not fatal
        return {"id": ex.get("id"), "category": ex.get("category"), "_error": f"{type(e).__name__}: {e}"}


def _house_key(line: str):
    ex = json.loads(line)
    b = ex.get("big") or {}
    return (b.get("kind", ""), b.get("seed", 0), b.get("n", 0), ex.get("house_seed", 0))


def convert_file(src: str, dst: str, route_reads: bool = False, workers: int = 0, limit: int = 0) -> dict:
    lines = [l for l in open(src) if l.strip()][: limit or None]
    order = sorted(range(len(lines)), key=lambda i: _house_key(lines[i]))  # one process keeps a big house's cache warm
    with ProcessPoolExecutor(workers or os.cpu_count() or 4) as pool:
        got = list(pool.map(_one, [(lines[i], route_reads) for i in order], chunksize=8))
    rows: list[Optional[dict]] = [None] * len(lines)
    for i, r in zip(order, got):
        rows[i] = r
    ok = [r for r in rows if r and "_error" not in r]
    with open(dst, "w") as f:
        for r in ok:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return stats(src, dst, rows)


def stats(src: str, dst: str, rows: list) -> dict:
    cats = collections.defaultdict(lambda: collections.Counter())
    reasons = collections.Counter()
    errors = [r for r in rows if r and "_error" in r]
    for r in rows:
        if not r or "_error" in r:
            continue
        c = cats[r["category"]]
        c["examples"] += 1
        asks = [tc for m in r["messages"] for tc in m.get("tool_calls") or [] if tc["function"]["name"] == "ask_rooms"]
        c["ask_rooms"] += len(asks)
        c["requests"] += sum(len(tc["function"]["arguments"]["requests"]) for tc in asks)
        c["call_service_kept"] += sum(1 for m in r["messages"] for tc in m.get("tool_calls") or [] if tc["function"]["name"] == "call_service")
        c["with_ask_rooms"] += bool(asks)
        for note in r["orch_notes"]:
            key = note.split(":")[0]
            c[key] += 1
            if key != "routed read":
                reasons[note if key != "approx" else "approx: " + " ".join(note.split(" ")[1:2])] += 1
    out = {"src": src, "dst": dst, "examples": sum(v["examples"] for v in cats.values()), "errors": len(errors),
           "error_samples": [e["_error"] for e in errors[:5]], "by_category": {k: dict(v) for k, v in sorted(cats.items())},
           "reasons": dict(reasons.most_common())}
    return out


def print_stats(st: dict):
    print(f"\n{st['src']} -> {st['dst']}: {st['examples']} examples" + (f", {st['errors']} failed: {st['error_samples']}" if st["errors"] else ""))
    print(f"{'category':22s} {'n':>5s} {'w/ask':>6s} {'asks':>5s} {'rooms':>6s} {'B call':>7s} {'kept':>5s} {'approx':>6s}")
    for k, v in st["by_category"].items():
        print(f"{k:22s} {v['examples']:5d} {v['with_ask_rooms']:6d} {v['ask_rooms']:5d} {v['requests']:6d} {v['call_service_kept']:7d} "
              f"{v.get('kept room call', 0):5d} {v.get('approx', 0):6d}")
    print("reasons:")
    for k, v in list(st["reasons"].items())[:25]:
        print(f"  {v:6d}  {k[:150]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="one B file to convert (default: train-b-split, val-b, test-v2-b)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--route-reads", action="store_true", help="also route single-room get_state calls a room read tool covers")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    jobs = [(a.data, a.out or a.data.replace("-b.jsonl", "-orch.jsonl"))] if a.data else \
        [(os.path.join(OUT, s), os.path.join(OUT, d)) for s, d in DEFAULT_JOBS]
    for src, dst in jobs:
        st = convert_file(src, dst, a.route_reads, a.workers, a.limit)
        print_stats(st)
        with open(dst.replace(".jsonl", ".stats.json"), "w") as f:
            json.dump(st, f, indent=1)


if __name__ == "__main__":
    main()
