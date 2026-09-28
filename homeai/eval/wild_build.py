"""Turn agent-authored wild specs into eval-ready transcripts, validating against the simulated house.

    python -m homeai.eval.wild_build --packs /tmp/wild --specs /tmp/wild/specs --out data/generated/wild.jsonl
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random

from ..house.generator import make_house
from ..house.schema import CLASSES, GROUPS, Speaker
from ..house.sim import resolve_target
from ..house.snapshot import system_prompt
from ..tools import execute_tool

KNOWN_SERVICES = {
    "light": {"turn_on", "turn_off", "toggle"}, "switch": {"turn_on", "turn_off", "toggle"}, "fan": {"turn_on", "turn_off", "set_percentage", "toggle"},
    "cover": {"open_cover", "close_cover", "set_cover_position", "stop_cover"}, "lock": {"lock", "unlock"},
    "climate": {"set_temperature", "set_hvac_mode", "turn_off", "turn_on"},
    "media_player": {"media_pause", "media_play", "media_play_pause", "media_stop", "turn_off", "turn_on", "volume_set", "volume_up", "volume_down", "volume_mute",
                     "play_media", "media_next_track", "media_previous_track", "shuffle_set", "repeat_set", "join", "unjoin", "media_seek"},
    "music_assistant": {"play_media"}, "alarm_control_panel": {"alarm_arm_home", "alarm_arm_away", "alarm_arm_night", "alarm_disarm"},
    "vacuum": {"start", "return_to_base", "stop"}, "homeassistant": {"turn_on", "turn_off", "toggle"},
}


def validate_call(house, call):
    d, s = call.get("domain"), call.get("service")
    if d not in KNOWN_SERVICES or s not in KNOWN_SERVICES[d]:
        return f"unknown service {d}.{s}"
    t = call.get("target") or {}
    if not t.get("entity_id") and not t.get("area_id"):
        return "call without target"
    for aid in t.get("area_id") or []:
        if aid not in house.area_ids():
            return f"unknown or group area_id {aid} (groups need entity_id)"
    ids, errs = resolve_target(house, d, t)
    if errs and not ids:
        return "; ".join(errs)
    if not ids:
        return f"target resolves to no {d} entities"
    return None


def speaker_problem(house, pack, sp):
    """Optional per-conversation speaker override: {"name": person, "area": area_id | null, "channel": "satellite" | "app"}.
    A person speaking from a satellite must be home and (per the hidden ground truth) in that room, so the sensors agree."""
    if sp is pack["speaker"]:
        return None
    people = {p["name"]: p for p in pack.get("people", [])}
    if sp.get("name") not in people:
        return f"speaker override: unknown person {sp.get('name')}"
    ch = sp.get("channel", "satellite" if sp.get("area") else "app")
    if ch == "app":
        return None if not sp.get("area") else "speaker override: app speaker has no area"
    if sp.get("area") not in house.area_ids():
        return f"speaker override: unknown area {sp.get('area')}"
    if people[sp["name"]].get("room_truth") != sp["area"]:
        return "speaker override: person is not in that room (room_truth)"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    packs = {json.load(open(p))["house_seed"]: json.load(open(p)) for p in glob.glob(os.path.join(a.packs, "house-*.json"))}
    kept, dropped = [], collections.Counter()
    for sp_path in sorted(glob.glob(os.path.join(a.specs, "*.json"))):
        try:
            spec = json.load(open(sp_path))
        except json.JSONDecodeError as e:
            dropped[f"bad json {os.path.basename(sp_path)}"] += 1
            continue
        items = spec if isinstance(spec, list) else spec.get("items") or spec.get("conversations") or []
        seed = spec.get("house_seed") if isinstance(spec, dict) else None
        if seed is None:
            try:
                seed = int(os.path.basename(sp_path).split("-")[1].split(".")[0])
            except (IndexError, ValueError):
                dropped["no house_seed"] += len(items)
                continue
        pack = packs.get(seed)
        if not pack:
            dropped["unknown house"] += len(items)
            continue
        for it in items:
            house = make_house(seed, pack["profile"])
            sp_d = it.get("speaker") or pack["speaker"]
            why_sp = speaker_problem(house, pack, sp_d)
            if why_sp:
                dropped[why_sp[:60]] += 1
                continue
            sp = Speaker(name=sp_d.get("name"), area=sp_d.get("area"), channel=sp_d.get("channel", "satellite" if sp_d.get("area") else "app"))
            msgs = [{"role": "system", "content": system_prompt(house, sp)}]
            ok, reason, ncalls = True, "", 0
            for turn in it.get("turns", []):
                if not turn.get("user"):
                    ok, reason = False, "turn without user text"; break
                msgs.append({"role": "user", "content": turn["user"]})
                fetch = turn.get("fetch")
                if fetch:
                    bad = [c for c in fetch.get("classes", []) if c not in CLASSES] + [x for x in fetch.get("areas", []) if x not in house.all_area_keys()]
                    if bad:
                        ok, reason = False, f"bad fetch filters {bad}"; break
                    args = {k: v for k, v in (("areas", fetch.get("areas")), ("classes", fetch.get("classes"))) if v}
                    ncalls += 1
                    msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{ncalls}", "type": "function", "function": {"name": "get_state", "arguments": args}}]})
                    msgs.append({"role": "tool", "tool_call_id": f"call_{ncalls}", "name": "get_state", "content": execute_tool(house, "get_state", args)})
                calls = turn.get("calls") or []
                if calls:
                    errs = [validate_call(house, c) for c in calls]
                    if any(errs):
                        ok, reason = False, "; ".join(e for e in errs if e); break
                    tcs, results = [], []
                    for c in calls:
                        ncalls += 1
                        args = {"domain": c["domain"], "service": c["service"], "target": c["target"]}
                        if c.get("data"):
                            args["data"] = c["data"]
                        tcs.append({"id": f"call_{ncalls}", "type": "function", "function": {"name": "call_service", "arguments": args}})
                        results.append((f"call_{ncalls}", execute_tool(house, "call_service", args)))
                    msgs.append({"role": "assistant", "content": "", "tool_calls": tcs})
                    for cid, res in results:
                        msgs.append({"role": "tool", "tool_call_id": cid, "name": "call_service", "content": res})
                ans = turn.get("answer")
                if ans:
                    m = {"role": "assistant", "content": ans}
                    must = [x.lower() for x in (turn.get("answer_must") or []) if x]
                    for x in must:
                        if not any(alt.strip() in ans.lower() for alt in x.split("|")):
                            ok, reason = False, f"answer does not contain its own must phrase {x!r}"; break
                    if not ok:
                        break
                    if must:
                        m["_eval"] = {"must_mention": must}
                    msgs.append(m)
                elif not calls and not fetch:
                    ok, reason = False, "turn with nothing expected"; break
            if not ok:
                dropped[reason[:60]] += 1
                continue
            if msgs[-1]["role"] != "assistant":
                dropped["conversation must end with the assistant"] += 1
                continue
            kept.append({"id": f"wild-{seed}-{it.get('id', len(kept))}", "category": "wild_" + str(it.get("category", "mixed")), "house_seed": seed,
                         "profile": pack["profile"], "speaker": sp.__dict__, "messages": msgs, "wild": True})
    with open(a.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)} wild conversations -> {a.out}")
    if dropped:
        print("dropped:", dict(dropped.most_common()))
    print("by category:", dict(collections.Counter(r["category"] for r in kept)))


if __name__ == "__main__":
    main()
