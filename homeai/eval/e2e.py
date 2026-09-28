"""Outcome eval: judge what happened, not which fetch the model chose.

For every user turn of a (wild) conversation: restore the reference history and the reference house
state as of that turn, let the model run the full tool loop freely (get_state / call_service in the
simulator, up to 4 rounds), then compare
  state  : the model's house vs the house after the reference calls (per entity, with tolerances
           where the user gave no number: "turn it down a bit" may be -10 or set to 20%), and
  answer : the model's spoken reply must contain at least half of the reference answer's key facts.
A turn passes if both hold. Different-but-valid fetch strategies (solar only vs solar+load, acting
by area_id without fetching first) pass; wrong rooms, extra actions and skipped confirmations fail.

  python3 -m homeai.eval.e2e --model homeai-4b-v2 --data data/generated/wild.jsonl --out outputs/eval/homeai-4b-v2-wild-e2e
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import re
import time

from ..bridge.agent import OllamaClient, make_client
from ..host import host_info  # latency is per host: record which machine served the model
from ..house.bighouse import house_for  # small houses via generator.make_house, big ones via ex["big"]
from ..tools import TOOLS, execute_tool
from ..tools_b import for_toolset  # an example's "toolset": "b" runs the design-B schemas and executor
from .run_eval import to_ollama
from .match import item_hit
from .sensitive import HALF, sensitive_state

NOT_PLAYING = {"paused", "off", "idle", "standby", "on"}
FILLER = {"the", "a", "my", "some", "playlist", "radio", "station", "album", "by", "songs", "music", "podcast", "mix"}


def snap(house):
    return {eid: (e.state, dict(e.attributes)) for eid, e in house.entities.items()}


def words(s):
    return {w for w in re.findall(r"[a-z0-9']+", str(s or "").lower()) if w not in FILLER}


def num_close(a, b, user_has_number, tight, loose):
    if a is None or b is None:
        return a == b
    try:
        return abs(float(a) - float(b)) <= (tight if user_has_number else loose)
    except (TypeError, ValueError):
        return a == b


def entity_match(eid, ref, got, user_text):
    (rs, ra), (gs, ga) = ref, got
    dom = eid.split(".")[0]
    has_num = bool(re.search(r"\d", user_text))
    if dom == "media_player":
        if (rs == "playing") != (gs == "playing"):
            return False, f"{eid}: {gs} vs {rs}"
        if rs == "playing" and ra.get("media_title") != ga.get("media_title"):
            rt, gt = words(ra.get("media_title")), words(ga.get("media_title"))
            named = rt & words(user_text)
            if named and not (named <= gt or rt <= gt or gt <= rt):
                return False, f"{eid}: plays {ga.get('media_title')!r} not {ra.get('media_title')!r}"
        if not num_close(ra.get("volume_pct"), ga.get("volume_pct"), has_num, 2, 15):
            return False, f"{eid}: volume {ga.get('volume_pct')} vs {ra.get('volume_pct')}"
        for k in ("muted", "shuffle", "repeat"):
            if ra.get(k) != ga.get(k):
                return False, f"{eid}: {k} {ga.get(k)} vs {ra.get(k)}"
        members = lambda v: {json.dumps(x, sort_keys=True) if not isinstance(x, str) else x for x in (v if isinstance(v, list) else [v] if v else [])}  # noqa: E731
        if members(ra.get("group_members")) != members(ga.get("group_members")):  # models sometimes pass malformed members
            return False, f"{eid}: group {ga.get('group_members')} vs {ra.get('group_members')}"
        return True, ""
    if rs != gs:
        return False, f"{eid}: {gs} vs {rs}"
    checks = {"brightness_pct": (2, 25), "position": (2, 25), "percentage": (2, 25), "temperature": (0.5, 2), "remaining_min": (1, 30)}
    for k, (tight, loose) in checks.items():
        if k == "brightness_pct" and rs != "on":
            continue
        if (k in ra or k in ga) and not num_close(ra.get(k), ga.get(k), has_num, tight, loose):
            return False, f"{eid}: {k} {ga.get(k)} vs {ra.get(k)}"
    if ra.get("status") != ga.get("status"):
        return False, f"{eid}: status {ga.get('status')} vs {ra.get('status')}"
    return True, ""


def sensitive_only_mismatch(ref_house, got_house, user_text):
    """True when every wrong entity is a lock/alarm/garage the model opened that the reference left closed."""
    r, g = snap(ref_house), snap(got_house)
    bad = [eid for eid in r if r[eid] != g.get(eid) and not entity_match(eid, r[eid], g[eid], user_text)[0]]
    return bool(bad) and all(sensitive_state(eid, g[eid][0]) for eid in bad)


def compare(ref_house, got_house, user_text):
    r, g = snap(ref_house), snap(got_house)
    for eid in r:
        if r[eid] != g.get(eid):
            ok, why = entity_match(eid, r[eid], g[eid], user_text)
            if not ok:
                return False, why
    return True, ""


def answer_ok(text, must, category):
    text = (text or "").lower()
    if not text:
        return False, "empty answer"
    if category in ("wild_missing", "wild_smalltalk") or not must:
        return True, ""
    hits = sum(1 for item in must if item_hit(text, item))
    if hits < max(1, (len(must) + 1) // 2):
        return False, "missing:" + "; ".join(i for i in must if not item_hit(text, i))
    return True, ""


def apply_ref(house, msgs):
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            fn = tc["function"]
            args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"])
            if fn["name"] == "call_service":
                execute_tool(house, "call_service", args)


def free_run(llm, ctx, house, max_rounds=4, tools=TOOLS, execute=execute_tool):
    msgs = list(ctx)
    calls_made, n = [], 0
    for _ in range(max_rounds):
        msg = llm.chat(msgs, tools)
        n += 1
        calls = msg.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": msg.get("content", "") or "", **({"tool_calls": calls} if calls else {})})
        if not calls:
            return (msg.get("content") or "").strip(), calls_made, n
        for tc in calls:
            fn = tc.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls_made.append({"name": fn.get("name"), "arguments": args})
            msgs.append({"role": "tool", "content": execute(house, fn.get("name", ""), args), "tool_name": fn.get("name", "")})
    return "", calls_made, n


def rescore(out_dir):
    """Re-apply the current (more lenient) reply rule to an existing report using its failures file."""
    rep = json.load(open(os.path.join(out_dir, "report.json")))
    fails = [json.loads(l) for l in open(os.path.join(out_dir, "failures.jsonl"))]
    keep = []
    for f in fails:
        acts = any(c.get("name") == "call_service" for c in f.get("ref_calls") or [])
        if acts and f["why"].startswith("missing:") and (f.get("got_text") or "").strip():
            rep["by_kind"]["act"]["ok"] += 1
            rep["by_kind"]["reply"]["ok"] += 1
            rep["by_category"][f["category"]]["ok"] += 1
        else:
            keep.append(f)
    rep["pass"] = (rep["by_kind"]["act"]["ok"] + rep["by_kind"].get("answer", {"ok": 0})["ok"]) / rep["turns"]
    rep["conversation_pass"] = 1 - len({f["id"] for f in keep}) / rep["examples"]
    rep["rescored"] = "reply rule: action turns need any confirmation"
    json.dump(rep, open(os.path.join(out_dir, "report.json"), "w"), indent=2)
    with open(os.path.join(out_dir, "failures.jsonl"), "w") as fh:
        for f in keep:
            fh.write(json.dumps(f) + "\n")
    print(f"{out_dir}: {len(fails) - len(keep)} turns flipped to pass, pass={rep['pass']:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/generated/wild.jsonl")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-ctx", type=int, default=8192, help="context window (the bridge runs 8192)")
    ap.add_argument("--rescore", action="store_true", help="re-apply the current reply rule to an existing --out report")
    a = ap.parse_args()
    if a.rescore:
        return rescore(a.out)
    os.makedirs(a.out, exist_ok=True)
    llm = make_client(a.ollama, a.model, a.num_ctx, temperature=0.0, think=False)
    rows = [json.loads(l) for l in open(a.data) if l.strip()][: a.limit or None]
    cats = collections.defaultdict(lambda: {"n": 0, "ok": 0})
    kinds = collections.defaultdict(lambda: {"n": 0, "ok": 0})
    conv_ok = 0
    calls_total, errors = 0, 0
    fails = open(os.path.join(a.out, "failures.jsonl"), "w")
    # every turn's resulting changes and timing, so two runs (a fast and a slow model) can be reconciled offline
    # (design E, homeai/eval/reconcile.py)
    turns_log = open(os.path.join(a.out, "turns.jsonl"), "w")
    t0 = time.time()
    for i, ex in enumerate(rows):
        msgs = ex["messages"]
        tools, execute, _ = for_toolset(ex.get("toolset"))
        user_idx = [j for j, m in enumerate(msgs) if m["role"] == "user"]
        all_ok = True
        for k, j in enumerate(user_idx):
            end = user_idx[k + 1] if k + 1 < len(user_idx) else len(msgs)
            seg = msgs[j + 1:end]
            before = house_for(ex)
            apply_ref(before, msgs[:j])
            ref_after = copy.deepcopy(before)
            apply_ref(ref_after, seg)
            got_house = copy.deepcopy(before)
            t_turn = time.time()
            try:
                text, calls, n = free_run(llm, to_ollama(msgs[:j + 1]), got_house, tools=tools, execute=execute)
            except Exception as e:  # noqa: BLE001
                text, calls, n = f"<error {e}>", [], 1
                errors += 1
            calls_total += n
            ref_ans = next((m for m in reversed(seg) if m["role"] == "assistant" and m.get("content")), {})
            must = (ref_ans.get("_eval") or {}).get("must_mention", [])
            # heard sets (homeai/voice/roundtrip.py) keep the written request in "_written": tolerances and named titles
            # follow what the user said, the model only ever sees the transcript
            said = msgs[j].get("_written", msgs[j]["content"])
            s_ok, s_why = compare(ref_after, got_house, said)
            acts_ref = any(tc["function"]["name"] == "call_service" for m in seg for tc in (m.get("tool_calls") or []))
            # Turns that change the house are judged on the end state; the confirmation just has to exist
            # ("the alarm's off" is as good as "disarmed"). Questions still need their key facts.
            a_ok, a_why = answer_ok(text, [] if acts_ref else must, ex["category"]) if ref_ans else (True, "")
            ok = s_ok and a_ok
            if not ok and not acts_ref and not s_ok and sensitive_only_mismatch(ref_after, got_house, said):
                ok = HALF  # did the unlock/disarm without asking first: half a failure (Peter, 25 Sep)
                s_why = "unconfirmed sensitive action (half credit): " + s_why
            acts = any(tc["function"]["name"] == "call_service" for m in seg for tc in (m.get("tool_calls") or []))
            kind = "act" if acts else "answer"
            cats[ex["category"]]["n"] += 1; cats[ex["category"]]["ok"] += ok
            kinds[kind]["n"] += 1; kinds[kind]["ok"] += ok
            kinds["state"]["n"] += 1; kinds["state"]["ok"] += s_ok
            kinds["reply"]["n"] += 1; kinds["reply"]["ok"] += a_ok
            all_ok = all_ok and ok == 1
            sb, sg = snap(before), snap(got_house)
            turns_log.write(json.dumps({"id": ex["id"], "turn": k, "ok": ok, "state_ok": s_ok, "reply_ok": a_ok,
                                        "changes": {e: v for e, v in sg.items() if sb.get(e) != v}, "calls": n,
                                        "seconds": round(time.time() - t_turn, 3), "text": text}, default=str) + "\n")
            if ok != 1:  # half-credit turns stay in the failure log
                fails.write(json.dumps({"id": ex["id"], "category": ex["category"], "turn": k, "user": msgs[j]["content"],
                                        **({"written": said} if said != msgs[j]["content"] else {}),
                                        "why": "; ".join(w for w in (s_why, a_why) if w), "got_text": text, "got_calls": calls,
                                        "ref_text": ref_ans.get("content"),
                                        "ref_calls": [tc["function"] for m in seg for tc in (m.get("tool_calls") or [])]}) + "\n")
        conv_ok += all_ok
        if (i + 1) % 25 == 0:
            n_ = kinds["state"]["n"]
            print(f"[{i + 1}/{len(rows)}] turns={n_} pass={sum(v['ok'] for k2, v in kinds.items() if k2 in ('act', 'answer')) / max(1, n_):.1%} {time.time() - t0:.0f}s", flush=True)
    fails.close()
    turns_log.close()
    total = kinds["state"]["n"]
    if errors and errors / max(1, total) > 0.2:
        print(f"{errors}/{total} transport errors; not writing a report")
        return 2
    passed = kinds["act"]["ok"] + kinds["answer"]["ok"]
    rep = {"model": a.model, "data": a.data, "mode": "e2e", **host_info(), "ollama": a.ollama, "examples": len(rows), "turns": total, "pass": passed / max(1, total),
           "conversation_pass": conv_ok / max(1, len(rows)), "seconds_per_turn": (time.time() - t0) / max(1, calls_total),
           "calls_per_request": calls_total / max(1, total), "by_kind": kinds, "by_category": cats}
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("by_category",)}, indent=2))


if __name__ == "__main__":
    main()
