"""Apply the half-credit rule for unconfirmed sensitive actions (homeai.eval.sensitive) to existing reports.

A failed turn where the reference asks for confirmation (no call_service) and every call the model made was an
unlock / disarm / garage open gets 0.5 instead of 0. Idempotent: each report is marked once rescored.

    python3 -m homeai.eval.rescore_sensitive [--eval-dir outputs/eval]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from .sensitive import HALF, calls_all_sensitive

MARK = "unconfirmed sensitive action (half credit)"


def rescore(d: str) -> int:
    rp, fp = os.path.join(d, "report.json"), os.path.join(d, "failures.jsonl")
    if not (os.path.exists(rp) and os.path.exists(fp)):
        return 0
    rep = json.load(open(rp))
    if rep.get("half_credit_rescored"):
        return 0
    fails = [json.loads(l) for l in open(fp) if l.strip()]
    n = 0
    for f in fails:
        if MARK in f.get("why", ""):
            continue
        if rep.get("mode") == "e2e":
            ref_acts = [c for c in f.get("ref_calls") or [] if c.get("name") == "call_service"]
            hit = not ref_acts and calls_all_sensitive(f.get("got_calls"))
            kind = "answer"
        else:
            hit = f.get("why") == "unexpected_tool_call" and calls_all_sensitive((f.get("got") or {}).get("tool_calls"))
            kind = f.get("kind", "answer")
        if not hit:
            continue
        rep["by_kind"][kind]["ok"] += HALF
        rep["by_category"][f["category"]]["ok"] += HALF
        f["why"] = MARK + ": " + f.get("why", "")
        n += 1
    if rep.get("mode") == "e2e":
        rep["pass"] = (rep["by_kind"].get("act", {"ok": 0})["ok"] + rep["by_kind"].get("answer", {"ok": 0})["ok"]) / rep["turns"]
    else:
        rep["pass"] = sum(v["ok"] for v in rep["by_kind"].values()) / rep["turns"]
    rep["half_credit_rescored"] = True
    rep["half_credit_turns"] = n
    json.dump(rep, open(rp, "w"), indent=2)
    with open(fp, "w") as fh:
        for f in fails:
            fh.write(json.dumps(f) + "\n")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="outputs/eval")
    a = ap.parse_args()
    for d in sorted(glob.glob(os.path.join(a.eval_dir, "*"))):
        before = (json.load(open(os.path.join(d, "report.json"))).get("pass") if os.path.exists(os.path.join(d, "report.json")) else None)
        n = rescore(d)
        if n:
            after = json.load(open(os.path.join(d, "report.json")))["pass"]
            print(f"{os.path.basename(d):40s} {n:3d} turns half-credited  {before:.1%} -> {after:.1%}")


if __name__ == "__main__":
    main()
