"""Design E (Peter, 27 Sep): a fast model acts at once while a slow model runs independently on the pre-request state;
a deterministic reconciler compares the two runs' resulting entity states and, where they differ, reverts the fast
model's changes and applies the slow model's. Scored offline from two outcome-eval runs of the same set (e2e.py writes
turns.jsonl), so any fast/slow pair can be compared without new model calls.

    python3 -m homeai.eval.reconcile outputs/eval/<fast>-wild-e2e outputs/eval/<slow>-wild-e2e

Per turn: agreement = identical resulting changes; flicker = the fast model changed something the slow model then
reverted or changed differently (a visible wrong action); final correctness = the slow model's (it always wins a
disagreement); sensitive turns (unlock, disarm, garage) never take the fast path. Time to first action is the fast
model's turn time when it acted and agreed, else the slow model's.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

VOLATILE = ("changed", "updated", "since", "last_")
SENSITIVE_PREFIX = ("lock.", "alarm_control_panel.", "cover.garage")


def load(d: str) -> dict:
    return {(t["id"], t["turn"]): t for t in (json.loads(l) for l in open(os.path.join(d, "turns.jsonl")) if l.strip())}


def norm(changes: dict) -> dict:
    out = {}
    for eid, v in (changes or {}).items():
        state, attrs = (v[0], v[1]) if isinstance(v, (list, tuple)) and len(v) == 2 else (v, {})
        attrs = {k: a for k, a in (attrs or {}).items() if not any(x in k for x in VOLATILE)}
        out[eid] = (state, json.dumps(attrs, sort_keys=True, default=str))
    return out


def reconcile(fast: dict, slow: dict) -> dict:
    keys = sorted(set(fast) & set(slow))
    rows = []
    for k in keys:
        f, s = fast[k], slow[k]
        fc, sc = norm(f.get("changes")), norm(s.get("changes"))
        sensitive = any(e.startswith(SENSITIVE_PREFIX) for e in set(fc) | set(sc))
        agree = fc == sc
        fast_path = bool(fc) and not sensitive          # the fast model's actions were executed immediately
        flicker = fast_path and not agree
        acted = bool(fc or sc)
        first = f["seconds"] if fast_path and agree else s["seconds"]
        rows.append({"agree": agree, "flicker": flicker, "acted": acted, "sensitive": sensitive, "fast_path": fast_path,
                     "fast_ok": f["ok"], "slow_ok": s["ok"], "final_ok": s["ok"], "first_s": first,
                     "final_s": max(f["seconds"], s["seconds"]) if not agree else first})
    n = len(rows)
    acting = [r for r in rows if r["acted"]]
    mean = lambda xs: round(statistics.mean(xs), 3) if xs else None  # noqa: E731
    return {"turns": n, "acting_turns": len(acting),
            "agreement": mean([r["agree"] for r in rows]), "agreement_acting": mean([r["agree"] for r in acting]),
            "flicker_rate": mean([r["flicker"] for r in rows]), "flicker_rate_acting": mean([r["flicker"] for r in acting]),
            "fast_pass": mean([float(r["fast_ok"]) for r in rows]), "slow_pass": mean([float(r["slow_ok"]) for r in rows]),
            "reconciled_pass": mean([float(r["final_ok"]) for r in rows]),
            "fast_path_share_acting": mean([r["fast_path"] and r["agree"] for r in acting]),
            "time_to_first_action_s_acting": mean([r["first_s"] for r in acting]),
            "slow_alone_s_acting": mean([slow[k]["seconds"] for k in keys if norm(fast[k].get("changes")) or norm(slow[k].get("changes"))]),
            "time_to_final_state_s_acting": mean([r["final_s"] for r in acting])}


def main():
    fast, slow = load(sys.argv[1]), load(sys.argv[2])
    rep = reconcile(fast, slow)
    print(json.dumps(rep, indent=1))
    if len(sys.argv) > 3:
        json.dump(rep, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
