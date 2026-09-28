"""Outcome eval of the tiered pipeline (designs C / D), scored exactly like homeai/eval/e2e.py so the numbers compare
with the design A / B outcome runs on the same file.

For every user turn of a B benchmark conversation (bigbench-b.jsonl, test-v2-b.jsonl, wild-b.jsonl): the house is
rebuilt and the reference calls before the turn applied, the history is the same conversation rewritten for the
orchestrator (homeai/tiered/data.py to_orch: room control as ask_rooms, recomputed results), and the pipeline runs freely
(homeai/tiered/pipeline.py, up to --max-rounds orchestrator rounds). The turn is judged with e2e's own functions: the end
state against the house after the reference B calls (compare / entity_match tolerances), key facts in the reply for
turns that do not act (answer_ok), half credit when the only fault is an unconfirmed unlock / disarm / garage opening
(sensitive_only_mismatch, homeai.eval.sensitive.HALF).

    # routed (C): orchestrator on URL1, one leaf model for every room on URL2
    PYTHONPATH=. python3 -m homeai.tiered.eval --data data/generated/bigbench-b.jsonl \
        --ollama http://URL1:11434 --model qwen3.5:9b --leaf qwen3.5:4b --leaf-url http://URL2:11434
    # room-kind leaves (+ one room pinned to its own model / server)
    ...  --leaf qwen3.5:4b --leaf-kind bedroom,bath=leaf-bedroom --leaf-kind kitchen=leaf-kitchen@http://URL3:11434 --leaf-area office_pete=leaf-office
    # blind fan-out (D)
    ...  --fanout
    # harness check without any model: oracle orchestrator (replays the reference) + oracle leaf (reference leaf calls)
    PYTHONPATH=. python3 -m homeai.tiered.eval --data data/generated/test-v2-b.jsonl --model oracle --leaf oracle --limit 40

Writes report.json and failures.jsonl into outputs/eval/<name>-<set>-tiered/ (name: --name, else the orchestrator model,
plus -fanout). Besides e2e's pass / by_kind / by_category the report has by_scale (bigbench: random 3000 / 10000 /
20000, bundook_full, bundook_projected), latency (orchestrator and leaf calls per turn, seconds per call, parallel leaf
wall time per ask_rooms, wall time per turn) and routing (rooms asked / acting per turn, orchestrator call_service on
room devices: routing leaks)."""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import re
import statistics
import time
from typing import Optional

from ..eval.e2e import answer_ok, apply_ref, compare, sensitive_only_mismatch
from ..eval.run_eval import to_ollama
from ..eval.sensitive import HALF
from ..host import host_info
from ..house.bighouse import house_for
from ..house.schema import Entity
from .data import speaker_of, to_orch
from .pipeline import LeafPicker, OracleLeaf, Pipeline, add_args, orchestrator_from_args, picker_from_args


class OracleOrchestrator:
    """Replays the reference orchestrator turns (to_orch's assistant messages for this user turn), one per chat() call."""

    def __init__(self, script: list[dict]):
        self.script = list(script)

    def chat(self, messages, tools=None) -> dict:
        if not self.script:
            return {"role": "assistant", "content": ""}
        m = self.script.pop(0)
        out = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("tool_calls"):
            out["tool_calls"] = [{"function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}} for tc in m["tool_calls"]]
        return out


def clone(house):
    """A copy the simulator can change without touching the original: service calls only reassign states and attribute
    keys (never mutate nested values), so per-entity attribute dicts suffice. ~20x faster than deepcopy on 20k entities."""
    h = copy.copy(house)
    h.entities = {k: Entity(e.entity_id, e.domain, e.name, e.cls, e.area, e.state, dict(e.attributes), e.changed_s)
                  for k, e in house.entities.items()}
    h.people = [copy.copy(p) for p in house.people]
    return h


def scale_of(ex: dict) -> str:
    b = ex.get("big")
    if not b:
        return "small"
    return b["kind"] + (f" {b['n']}" if b.get("n") else "")


def _dist(xs: list[float]) -> dict:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": round(statistics.mean(xs), 3), "p50": round(xs[len(xs) // 2], 3), "p95": round(xs[min(len(xs) - 1, int(len(xs) * 0.95))], 3)}


def evaluate(rows: list[dict], orch=None, picker: Optional[LeafPicker] = None, mode: str = "routed", max_rounds: int = 4, workers: int = 8,
             route_reads: bool = False, progress: bool = True, fails_fh=None) -> dict:
    """orch None: the oracle orchestrator; picker None: the oracle leaf (both from to_orch's reference)."""
    oracle_leaf = OracleLeaf() if picker is None else None
    pipe = Pipeline(orch, picker or LeafPicker(oracle_leaf), mode, max_rounds, workers)
    cats = collections.defaultdict(lambda: {"n": 0, "ok": 0})
    kinds = collections.defaultdict(lambda: {"n": 0, "ok": 0})
    scales = collections.defaultdict(lambda: {"n": 0, "ok": 0})
    lat = collections.defaultdict(list)
    route = collections.Counter()
    conv_ok, errors, conv_errors = 0, 0, 0
    t0 = time.time()
    for i, ex in enumerate(rows):
        msgs = ex["messages"]
        sp = speaker_of(ex)
        house0 = house_for(ex)
        try:
            conv = to_orch(ex, route_reads, house=clone(house0))
        except Exception as e:  # noqa: BLE001 - the reference cannot be rewritten: every turn counts as failed
            conv, conv_errors = None, conv_errors + 1
            print(f"{ex.get('id')}: to_orch failed: {e}")
        if oracle_leaf is not None and conv:
            oracle_leaf.gold = {}
            oracle_leaf.add(conv["leaf_gold"])
        user_idx = [j for j, m in enumerate(msgs) if m["role"] == "user"]
        cmsgs = conv["messages"] if conv else []
        cuser_idx = [j for j, m in enumerate(cmsgs) if m["role"] == "user"]
        prompt = pipe.prompt(house0, sp)
        all_ok = True
        for k, j in enumerate(user_idx):
            end = user_idx[k + 1] if k + 1 < len(user_idx) else len(msgs)
            seg = msgs[j + 1:end]
            before = clone(house0)
            apply_ref(before, msgs[:j])
            ref_after = clone(before)
            apply_ref(ref_after, seg)
            got_house = clone(before)
            st: dict = {}
            if conv and len(cuser_idx) == len(user_idx):
                cj = cuser_idx[k]
                cend = cuser_idx[k + 1] if k + 1 < len(cuser_idx) else len(cmsgs)
                ctx = to_ollama(cmsgs[:cj + 1])
                ctx[0] = {"role": "system", "content": prompt}
                pipe.orch = orch if orch is not None else OracleOrchestrator([m for m in cmsgs[cj + 1:cend] if m["role"] == "assistant"])
                try:
                    text, calls, st = pipe.run(ctx, got_house, msgs[j]["content"])
                except Exception as e:  # noqa: BLE001
                    text, calls = f"<error {e}>", []
                    errors += 1
            else:
                text, calls = "<error history>", []
                errors += 1
            ref_ans = next((m for m in reversed(seg) if m["role"] == "assistant" and m.get("content")), {})
            must = (ref_ans.get("_eval") or {}).get("must_mention", [])
            s_ok, s_why = compare(ref_after, got_house, msgs[j]["content"])
            acts_ref = any(tc["function"]["name"] == "call_service" for m in seg for tc in (m.get("tool_calls") or []))
            a_ok, a_why = answer_ok(text, [] if acts_ref else must, ex["category"]) if ref_ans else (True, "")
            ok = s_ok and a_ok
            if not ok and not acts_ref and not s_ok and sensitive_only_mismatch(ref_after, got_house, msgs[j]["content"]):
                ok = HALF
                s_why = "unconfirmed sensitive action (half credit): " + s_why
            kind = "act" if acts_ref else "answer"
            for d, key in ((cats, ex["category"]), (kinds, kind), (scales, scale_of(ex))):
                d[key]["n"] += 1
                d[key]["ok"] += ok
            kinds["state"]["n"] += 1; kinds["state"]["ok"] += s_ok
            kinds["reply"]["n"] += 1; kinds["reply"]["ok"] += a_ok
            all_ok = all_ok and ok == 1
            if st:
                for key in ("turn_s", "orch_calls", "orch_s", "leaf_calls", "leaf_s", "leaf_wall_s", "ask_rooms", "rooms_asked", "rooms_acted"):
                    lat[key].append(st[key])
                lat["leaf_latency"] += st["leaf_latencies"]
                lat["orch_latency"] += st["orch_latencies"]
                lat["ask_wall"] += st["ask_walls"]
                route["room_call_service"] += st["room_call_service"]
                route["turns_with_room_call_service"] += bool(st["room_call_service"])
                route["leaf_errors"] += st["leaf_errors"]
            if ok != 1 and fails_fh is not None:
                fails_fh.write(json.dumps({"id": ex["id"], "category": ex["category"], "scale": scale_of(ex), "turn": k, "user": msgs[j]["content"],
                                           "why": "; ".join(w for w in (s_why, a_why) if w), "got_text": text, "got_calls": calls,
                                           "ref_text": ref_ans.get("content"),
                                           "ref_calls": [tc["function"] for m in seg for tc in (m.get("tool_calls") or [])]}) + "\n")
        conv_ok += all_ok
        if progress and (i + 1) % 25 == 0:
            n_ = kinds["state"]["n"]
            print(f"[{i + 1}/{len(rows)}] turns={n_} pass={(kinds['act']['ok'] + kinds['answer']['ok']) / max(1, n_):.1%} {time.time() - t0:.0f}s", flush=True)
    pipe.close()
    total = kinds["state"]["n"]
    passed = kinds["act"]["ok"] + kinds["answer"]["ok"]
    wall = time.time() - t0
    mean = lambda k: round(statistics.mean(lat[k]), 3) if lat[k] else None  # noqa: E731
    return {
        "examples": len(rows), "turns": total, "pass": passed / max(1, total), "conversation_pass": conv_ok / max(1, len(rows)),
        "errors": errors, "history_errors": conv_errors,
        "seconds_per_turn": wall / max(1, sum(lat["orch_calls"]) or total),  # e2e.py's field: wall time per orchestrator call
        "calls_per_request": (sum(lat["orch_calls"]) / max(1, total)) if lat["orch_calls"] else None,
        "latency": {"turn_s": _dist(lat["turn_s"]), "orch_calls_per_turn": mean("orch_calls"), "orch_s_per_call": _dist(lat["orch_latency"]),
                    "orch_s_per_turn": mean("orch_s"), "leaf_calls_per_turn": mean("leaf_calls"), "leaf_s_per_call": _dist(lat["leaf_latency"]),
                    "leaf_s_per_turn": mean("leaf_s"), "leaf_wall_s_per_ask": _dist(lat["ask_wall"]), "leaf_wall_s_per_turn": mean("leaf_wall_s"),
                    "eval_wall_s": round(wall, 1)},
        "routing": {"ask_rooms_per_turn": mean("ask_rooms"), "rooms_asked_per_turn": mean("rooms_asked"), "rooms_acted_per_turn": mean("rooms_acted"),
                    **route},
        "by_kind": kinds, "by_scale": dict(sorted(scales.items())), "by_category": cats,
    }


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--data", default="data/generated/bigbench-b.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", default=None, help="report name (default: the orchestrator model, plus -fanout)")
    ap.add_argument("--out", default=None, help="default outputs/eval/<name>-<set>-tiered")
    ap.add_argument("--route-reads", action="store_true", help="history with single-room reads routed too (as data.py --route-reads)")
    a = ap.parse_args()
    mode = "fanout" if a.fanout else "routed"
    if a.model == "oracle" and a.fanout:
        raise SystemExit("the oracle orchestrator replays routed transcripts: no --fanout")
    orch = None if a.model == "oracle" else orchestrator_from_args(a)
    picker = None if a.leaf == "oracle" else picker_from_args(a)
    name = a.name or re.sub(r"[^A-Za-z0-9._-]+", "-", a.model) + ("-fanout" if a.fanout else "")
    out = a.out or os.path.join("outputs", "eval", f"{name}-{os.path.basename(a.data).replace('.jsonl', '')}-tiered")
    os.makedirs(out, exist_ok=True)
    rows = [json.loads(l) for l in open(a.data) if l.strip()][: a.limit or None]
    with open(os.path.join(out, "failures.jsonl"), "w") as fh:
        rep = evaluate(rows, orch, picker, mode, a.max_rounds, a.workers, a.route_reads, fails_fh=fh)
    if rep["errors"] and rep["errors"] / max(1, rep["turns"]) > 0.2:
        print(f"{rep['errors']}/{rep['turns']} transport errors; not writing a report")
        return 2
    leaf_calls = rep["latency"]["leaf_s_per_call"].get("n", 0)
    if rep["routing"].get("leaf_errors", 0) > 0.2 * max(1, leaf_calls):
        print(f"{rep['routing']['leaf_errors']}/{leaf_calls} leaf transport errors; not writing a report")
        return 2
    rep = {"model": a.model, "orchestrator_url": a.ollama, **host_info(), "leaves": picker.describe() if picker else {"default": "oracle"},
           "data": a.data, "mode": "e2e-tiered", "pipeline": mode, "max_rounds": a.max_rounds, **rep}
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("by_category",)}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
