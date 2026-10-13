"""Teacher-forced evaluation against the held-out generated set.

    python -m homeai.eval.run_eval --model homeai --data data/generated/test.jsonl
    python -m homeai.eval.run_eval --model qwen2.5:7b-instruct --data data/generated/test.jsonl --limit 60   # baseline

For every assistant turn in every example, the model is given the reference
transcript up to that point and asked for the next turn. Scoring:
  fetch  : get_state selection must cover every entity the reference fetched (recall = 1);
           over-fetch ratio is reported separately.
  act    : call_service calls must match the reference in service and resolved
           entity set (area_id and entity_id targets are resolved on the same house).
  answer : no tool call, must_mention / must_not_mention phrases satisfied.
Failures are written to <out>/failures.jsonl with the model output for inspection.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

from ..bridge.agent import OllamaClient, make_client
from .match import item_hit
from .sensitive import HALF, calls_all_sensitive
from .. import prompt_layout  # requests in the layout the model was trained on (v10 stable_prefix, v11 user_context, v12 clock_in_user)
from ..house.bighouse import house_for  # small houses via generator.make_house, big ones via ex["big"]
from ..house.sim import resolve_target
from ..tools import TOOLS
from ..tools_b import execute_tool_b, for_toolset, ids_in


def to_ollama(messages):
    out = []
    for m in messages:
        if m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"]})
        elif m["role"] == "assistant":
            n = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                n["tool_calls"] = [{"function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}} for tc in m["tool_calls"]]
            out.append(n)
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def norm_calls(calls):
    out = []
    for tc in calls or []:
        fn = tc.get("function", {})
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        out.append((fn.get("name"), args if isinstance(args, dict) else {}))
    return out


def fetch_set(house, args, name="get_state", toolset="a"):
    if toolset == "b":  # design B: the entities the call's (filtered, capped) result actually shows
        return ids_in(execute_tool_b(house, name, args))
    areas, classes = args.get("areas") or [], args.get("classes") or []
    areas, classes = ([areas] if isinstance(areas, str) else areas), ([classes] if isinstance(classes, str) else classes)
    return {e.entity_id for e in house.select(areas, classes)}


def act_key(house, args):
    domain, service = args.get("domain", ""), args.get("service", "")
    target = args.get("target")
    ids, _ = resolve_target(house, domain, target if isinstance(target, dict) else {})  # malformed target scores as wrong
    return (("ma:" if domain == "music_assistant" else "") + service, frozenset(ids))


def data_matches(ref: dict, got: dict, user_text: str = "") -> bool:
    """Service data only has to match where the user actually pinned it down: a number that appears in
    the request (brightness 40, 22 degrees, 30 minutes) or a media/content id. Extras the reference author
    invented (a fan percentage nobody asked for, hvac_mode alongside a temperature) are ignored."""
    u = (user_text or "").lower()
    for k, v in (ref or {}).items():
        g = (got or {}).get(k)
        if k in ("media_id", "media_content_id"):
            # only score the content when the user named it ("play Kasey Chambers"); for open requests
            # ("put something mellow on", "start a podcast") any sensible choice is fine
            if str(v).lower() in u and str(g).lower() != str(v).lower():
                return False
            continue
        if isinstance(v, bool):
            if str(g).lower() != str(v).lower():
                return False
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue  # free-form extras (hvac_mode, media_type) are not scored
        num_in_request = str(int(fv)) in u or str(v) in u or (k == "volume_level" and str(int(round(fv * 100))) in u)
        if not num_in_request:
            continue
        try:
            if float(g) != fv:
                return False
        except (TypeError, ValueError):
            return False
    return True


def next_act(msgs, j):
    """The reference's next call_service turn after index j (before the next user turn), if any."""
    for m in msgs[j + 1:]:
        if m["role"] == "user":
            return None
        if m["role"] == "assistant" and m.get("tool_calls") and any(tc["function"]["name"] == "call_service" for tc in m["tool_calls"]):
            return m
    return None


def score_turn(house, ref_msg, got_msg, user_text: str = "", next_act_msg=None, lenient: bool = False, category: str = "",
               toolset: str = "a"):
    """lenient=True (wild set): agent-written reference answers often volunteer extra facts, so an answer
    passes with at least half of the listed phrases; refusals/small talk pass on having no tool call.
    toolset "b": find_devices counts as a fetch, and a fetch covers what the B results showed."""
    looks = {"get_state", "find_devices"} if toolset == "b" else {"get_state"}
    ref_calls = norm_calls(ref_msg.get("tool_calls"))
    got_calls = norm_calls(got_msg.get("tool_calls"))
    if not ref_calls:
        if got_calls:
            return "answer", False, "unexpected_tool_call", None
        text = (got_msg.get("content") or "").lower()
        ev = ref_msg.get("_eval") or {}
        must = ev.get("must_mention", [])
        if lenient:
            if category in ("wild_missing", "wild_smalltalk"):
                must = []
            hits = sum(1 for item in must if item_hit(text, item))
            if must and hits < max(1, (len(must) + 1) // 2):
                missing = [item for item in must if not item_hit(text, item)]
                return "answer", False, f"missing:{'; '.join(missing)}", None
            must = []
        for item in must:
            if not item_hit(text, item):
                return "answer", False, f"missing:{item}", None
        for item in ev.get("must_not_mention", []):
            if any(alt.strip() in text for alt in item.split("|")):
                return "answer", False, f"forbidden:{item}", None
        if not text:
            return "answer", False, "empty", None
        return "answer", True, "", None
    kind = "fetch" if all(n in looks for n, _ in ref_calls) else "act"
    if not got_calls:
        return kind, False, "missing_tool_call", None
    if kind == "fetch":
        need = set().union(*[fetch_set(house, a, n, toolset) for n, a in ref_calls if n in looks])
        got = set().union(*[fetch_set(house, a, n, toolset) for n, a in got_calls if n in looks]) if any(n in looks for n, _ in got_calls) else set()
        if any(n not in looks for n, _ in got_calls):
            # Skipping a fetch is fine when the action itself is right (e.g. area_id targeting
            # where the reference fetched an entity_id first). Compare with the reference's next act.
            if next_act_msg is not None and not any(n in looks for n, _ in got_calls):
                k2, ok2, why2, _ = score_turn(house, next_act_msg, got_msg, user_text, toolset=toolset)
                if ok2:
                    return kind, True, "", 1.0
                return kind, False, f"acted_instead_of_fetching ({why2})", None
            return kind, False, "acted_instead_of_fetching", None
        if not need:
            return kind, True, "", 1.0
        recall = len(need & got) / len(need)
        over = len(got) / len(need)
        return kind, recall == 1.0, ("" if recall == 1.0 else f"recall={recall:.2f}"), over
    # act
    ref_keys = [(act_key(house, a), a.get("data") or {}) for n, a in ref_calls if n == "call_service"]
    got_keys = [(act_key(house, a), a.get("data") or {}) for n, a in got_calls if n == "call_service"]
    if any(n in looks for n, _ in got_calls) and not got_keys:
        return kind, False, "fetched_instead_of_acting", None
    unmatched = []
    pool = list(got_keys)
    for rk, rd in ref_keys:
        hit = None
        for i, (gk, gd) in enumerate(pool):
            if gk == rk and data_matches(rd, gd, user_text):
                hit = i
                break
        if hit is None:
            unmatched.append(rk[0])
        else:
            pool.pop(hit)
    if unmatched:
        return kind, False, f"missing_calls:{','.join(unmatched)}", None
    if pool:
        return kind, False, f"extra_calls:{','.join(k[0][0] for k in pool)}", None
    return kind, True, "", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/generated/test.jsonl")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="report dir (default outputs/eval/<model>)")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--think", choices=["on", "off", "default"], default="off", help="thinking mode for hybrid models (Ollama 'think' flag)")
    prompt_layout.add_arg(ap)
    a = ap.parse_args()
    out_dir = a.out or os.path.join("outputs", "eval", a.model.replace(":", "_").replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    llm = make_client(a.ollama, a.model, a.num_ctx, temperature=0.0, think={"on": True, "off": False, "default": None}[a.think])
    layout = prompt_layout.for_model(a.model, a.prompt_layout)
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    if a.limit:
        rows = rows[:a.limit]

    stats = collections.defaultdict(lambda: {"n": 0, "ok": 0})
    kinds = collections.defaultdict(lambda: {"n": 0, "ok": 0})
    overs = []
    errors = 0
    fails = open(os.path.join(out_dir, "failures.jsonl"), "w")
    t0 = time.time()
    for i, ex in enumerate(rows):
        house = house_for(ex)
        msgs = ex["messages"]
        tools = for_toolset(ex.get("toolset"))[0]
        for j, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            ctx = prompt_layout.render(to_ollama(msgs[:j]), layout)  # scoring below reads the stored (bare) turns
            try:
                got = llm.chat(ctx, tools)
            except Exception as e:  # noqa: BLE001
                got = {"content": f"<error {e}>"}
                errors += 1
            user_text = next((x.get("_written", x["content"]) for x in reversed(msgs[:j]) if x["role"] == "user"), "")  # heard sets
            kind, ok, why, over = score_turn(house, m, got, user_text, next_act(msgs, j), lenient=bool(ex.get("wild")), category=ex["category"],
                                             toolset=ex.get("toolset") or "a")
            if ex.get("toolset") == "b":  # B fetches filter on state (where=on...): keep the house at this point of the reference
                for tc in m.get("tool_calls") or []:
                    if tc["function"]["name"] == "call_service":
                        execute_tool_b(house, "call_service", tc["function"]["arguments"])
            ok = int(ok)
            if not ok and why == "unexpected_tool_call" and calls_all_sensitive(got.get("tool_calls")):
                ok, why = HALF, "unconfirmed sensitive action (half credit)"  # Peter, 25 Sep
            stats[ex["category"]]["n"] += 1
            stats[ex["category"]]["ok"] += ok
            kinds[kind]["n"] += 1
            kinds[kind]["ok"] += ok
            if over is not None:
                overs.append(over)
            if ok != 1:  # half-credit turns stay in the failure log
                fails.write(json.dumps({"id": ex["id"], "category": ex["category"], "turn": j, "kind": kind, "why": why,
                                        "user": next((x["content"] for x in reversed(msgs[:j]) if x["role"] == "user"), ""),
                                        "expected": {"content": m.get("content"), "tool_calls": m.get("tool_calls")},
                                        "got": {"content": got.get("content"), "tool_calls": got.get("tool_calls")}}) + "\n")
        if (i + 1) % 10 == 0:
            done = sum(v["n"] for v in kinds.values())
            okc = sum(v["ok"] for v in kinds.values())
            print(f"[{i + 1}/{len(rows)}] turns={done} pass={okc / max(1, done):.1%}  {time.time() - t0:.0f}s", flush=True)
    fails.close()

    total_n = sum(v["n"] for v in kinds.values())
    if errors and errors / max(1, total_n) > 0.2:
        print(f"\n{errors}/{total_n} turns hit transport errors (Ollama down or restarting); not writing a report so the run is retried later.")
        os.rename(os.path.join(out_dir, "failures.jsonl"), os.path.join(out_dir, "failures.aborted.jsonl"))
        return 2
    total_ok = sum(v["ok"] for v in kinds.values())
    print(f"\nmodel: {a.model}   examples: {len(rows)}   turns: {total_n}   pass: {total_ok / max(1, total_n):.1%}\n")
    print(f"{'turn kind':12s} {'pass':>7s} {'n':>5s}")
    for k, v in sorted(kinds.items()):
        print(f"{k:12s} {v['ok'] / max(1, v['n']):7.1%} {v['n']:5d}")
    if overs:
        print(f"\nfetch over-fetch ratio (entities returned / entities needed): mean {sum(overs) / len(overs):.2f}")
    print(f"\n{'category':22s} {'pass':>7s} {'n':>5s}")
    for k, v in sorted(stats.items(), key=lambda kv: kv[1]["ok"] / max(1, kv[1]["n"])):
        print(f"{k:22s} {v['ok'] / max(1, v['n']):7.1%} {v['n']:5d}")
    report = {"model": a.model, "data": a.data, "examples": len(rows), "turns": total_n, "pass": total_ok / max(1, total_n),
              "seconds_per_turn": (time.time() - t0) / max(1, total_n), "think": a.think, "prompt_layout": layout,
              "by_kind": kinds, "by_category": stats, "overfetch_mean": (sum(overs) / len(overs)) if overs else None}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport: {out_dir}/report.json   failures: {out_dir}/failures.jsonl")


if __name__ == "__main__":
    sys.exit(main())
