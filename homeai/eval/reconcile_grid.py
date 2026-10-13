"""Design E grid (Peter, 28 Sep): reconcile every fast x slow pair that has per-turn logs on the same outcome sets, so the
tier champions can be mixed and matched. Only pairs where the fast model really is faster (mean seconds per turn on the same
turns) are formed: a slower model acting first and a faster one overruling it is not the design. Logs: outputs/eval-turns/<model>-<set>/turns.jsonl and
outputs/eval/<model>-<set>-e2e/turns.jsonl (sets: testv2, wild, wild2; --sets b: the design-B sets bigbench-b, testv2-b,
wild-b, wild2-b). Writes outputs/reconcile/<fast>__<slow>.json.

    python3 -m homeai.eval.reconcile_grid
    python3 -m homeai.eval.reconcile_grid --eval-dir outputs/eval-falcon --out outputs/reconcile-falcon   # falcon's own logs
    python3 -m homeai.eval.reconcile_grid --sets b --out outputs/reconcile-b   # design B pairs (lookup tools)

Timings come from each model's own run (the 2B ran on lapwing's 3080, the rest on cassowary's 3090). A host grid
(--eval-dir) reads only that host's reports, baseline rows only (no <model>@<host>-<variant> serving variants), so its
timings never mix with another machine's.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

from .reconcile import load, reconcile

SETS = ("testv2", "wild", "wild2")
SETS_B = ("bigbench-b", "testv2-b", "wild-b", "wild2-b")


def logs(eval_dir: str = "", sets: tuple = SETS) -> dict:
    out: dict = {}
    paths = glob.glob(f"{eval_dir}/*-e2e/turns.jsonl") if eval_dir else \
        glob.glob("outputs/eval-turns/*/turns.jsonl") + glob.glob("outputs/eval/*-e2e/turns.jsonl")
    for p in paths:
        d = os.path.basename(os.path.dirname(p))
        alt = "|".join(sorted(map(re.escape, sets), key=len, reverse=True))  # wild2 before wild
        m = re.match(rf"^(?P<m>.+)-(?P<s>{alt})(-e2e)?$", d)
        if eval_dir and not os.path.exists(os.path.join(os.path.dirname(p), "report.json")):
            continue  # a host grid runs while evals are in flight: finished sets only
        if m and "@" not in m["m"] and os.path.getsize(p):
            out.setdefault(m["m"], {}).setdefault(m["s"], os.path.dirname(p))  # eval-turns first: fresher and complete
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="", help="one host's reports (e.g. outputs/eval-falcon) instead of the reference logs")
    ap.add_argument("--out", default="outputs/reconcile")
    ap.add_argument("--sets", choices=("a", "b"), default="a", help="a: testv2 / wild / wild2; b: the design-B sets")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sets = SETS_B if a.sets == "b" else SETS
    L = logs(a.eval_dir, sets)
    models = sorted(L)
    # mean seconds per turn over the three sets, from each model's own logs: only a faster model can be the fast half
    speed = {}
    for m in models:
        secs = [t["seconds"] for s in sets if s in L[m] for t in load(L[m][s]).values()]
        speed[m] = sum(secs) / len(secs) if secs else None
    for fast in models:
        for slow in models:
            if fast == slow or speed[fast] is None or speed[slow] is None or speed[fast] >= speed[slow]:
                continue
            common = [s for s in sets if s in L[fast] and s in L[slow]]
            if len(common) < len(sets):
                continue
            per = {s: reconcile(load(L[fast][s]), load(L[slow][s])) for s in common}
            n = sum(v["turns"] for v in per.values())
            na = sum(v["acting_turns"] for v in per.values())
            w = lambda k, acting=False: round(sum((v[k] or 0) * (v["acting_turns"] if acting else v["turns"]) for v in per.values()) / max(1, na if acting else n), 3)  # noqa: E731
            agg = {"fast": fast, "slow": slow, "turns": n, "acting_turns": na, "fast_s_per_turn": round(speed[fast], 3), "slow_s_per_turn": round(speed[slow], 3),
                   "fast_pass": w("fast_pass"), "slow_pass": w("slow_pass"), "reconciled_pass": w("reconciled_pass"),
                   "agreement_acting": w("agreement_acting", True), "flicker_rate": w("flicker_rate"), "flicker_rate_acting": w("flicker_rate_acting", True),
                   "time_to_first_action_s_acting": w("time_to_first_action_s_acting", True), "slow_alone_s_acting": w("slow_alone_s_acting", True),
                   "time_to_final_state_s_acting": w("time_to_final_state_s_acting", True), "per_set": per}
            json.dump(agg, open(f"{a.out}/{fast}__{slow}.json", "w"), indent=1)
            print(f"{fast:26} -> {slow:26} final {agg['reconciled_pass']:.1%}  flicker {agg['flicker_rate_acting']:.1%} of acting turns  "
                  f"first action {agg['time_to_first_action_s_acting']:.2f}s (slow alone {agg['slow_alone_s_acting']:.2f}s)")


if __name__ == "__main__":
    main()
