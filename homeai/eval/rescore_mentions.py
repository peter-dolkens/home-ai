"""Re-judge existing reports' "missing:<fact>" failures with the current phrase matcher (homeai.eval.match).

The matcher only accepts more than the plain substring test it replaced, so this can only turn failures into passes.
Idempotent: each report is marked once rescored.

    python3 -m homeai.eval.rescore_mentions [--eval-dir outputs/eval]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from .e2e import answer_ok
from .run_eval import score_turn

MARK = "mentions_rescored"
_DATA: dict = {}


def examples(path):
    if path not in _DATA:
        _DATA[path] = {ex["id"]: ex for ex in (json.loads(l) for l in open(path) if l.strip())}
    return _DATA[path]


def still_fails(rep, f, ex) -> bool:
    msgs = ex["messages"]
    if rep.get("mode") == "e2e":
        user_idx = [j for j, m in enumerate(msgs) if m["role"] == "user"]
        j = user_idx[f["turn"]]
        end = user_idx[f["turn"] + 1] if f["turn"] + 1 < len(user_idx) else len(msgs)
        seg = msgs[j + 1:end]
        if any(tc["function"]["name"] == "call_service" for m in seg for tc in (m.get("tool_calls") or [])):
            return True  # action turns are judged on state; leave them alone
        ref_ans = next((m for m in reversed(seg) if m["role"] == "assistant" and m.get("content")), {})
        ok, _ = answer_ok(f.get("got_text"), (ref_ans.get("_eval") or {}).get("must_mention", []), f["category"])
        return not ok
    if f.get("kind") != "answer":
        return True
    _, ok, _, _ = score_turn(None, msgs[f["turn"]], f.get("got") or {}, f.get("user", ""), lenient=bool(ex.get("wild")),
                             category=f["category"])
    return not ok


def rescore(d: str) -> int:
    rp, fp = os.path.join(d, "report.json"), os.path.join(d, "failures.jsonl")
    if not (os.path.exists(rp) and os.path.exists(fp)):
        return 0
    rep = json.load(open(rp))
    if rep.get(MARK) or not rep.get("data") or not os.path.exists(rep["data"]):
        return 0
    exs = examples(rep["data"])
    fails = [json.loads(l) for l in open(fp) if l.strip()]
    keep, n = [], 0
    for f in fails:
        ex = exs.get(f["id"])
        if ex is None or not f.get("why", "").startswith("missing:") or still_fails(rep, f, ex):
            keep.append(f)
            continue
        rep["by_kind"]["answer"]["ok"] += 1
        if rep.get("mode") == "e2e":
            rep["by_kind"]["reply"]["ok"] += 1
        rep["by_category"][f["category"]]["ok"] += 1
        n += 1
    if rep.get("mode") == "e2e":
        rep["pass"] = (rep["by_kind"].get("act", {"ok": 0})["ok"] + rep["by_kind"]["answer"]["ok"]) / rep["turns"]
        rep["conversation_pass"] = 1 - len({f["id"] for f in keep}) / rep["examples"]
    else:
        rep["pass"] = sum(v["ok"] for v in rep["by_kind"].values()) / rep["turns"]
    rep[MARK] = True
    rep["mentions_flipped"] = n
    json.dump(rep, open(rp, "w"), indent=2)
    with open(fp, "w") as fh:
        for f in keep:
            fh.write(json.dumps(f) + "\n")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="outputs/eval")
    a = ap.parse_args()
    for d in sorted(glob.glob(os.path.join(a.eval_dir, "*"))):
        rp = os.path.join(d, "report.json")
        before = json.load(open(rp)).get("pass") if os.path.exists(rp) else None
        n = rescore(d)
        if n:
            print(f"{os.path.basename(d):44s} {n:3d} turns flipped  {before:.1%} -> {json.load(open(rp))['pass']:.1%}")


if __name__ == "__main__":
    main()
