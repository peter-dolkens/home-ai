"""Re-score an existing eval run's failures against the current test set (e.g. after loosening
must_mention rules) and rewrite its report.json. Only 'answer' turns that failed on a phrase can
flip; tool-call scoring is unchanged.

    python -m homeai.eval.rescore --run outputs/eval/homeai-4b --data data/generated/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

from ..house.bighouse import house_for  # small houses via generator.make_house, big ones via ex["big"]
from .run_eval import next_act, score_turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data/generated/test.jsonl")
    a = ap.parse_args()
    rows = {r["id"]: r for r in (json.loads(l) for l in open(a.data) if l.strip())}
    rp = os.path.join(a.run, "report.json")
    report = json.load(open(rp))
    fp = os.path.join(a.run, "failures.jsonl")
    fails = [json.loads(l) for l in open(fp) if l.strip()]
    kept, flipped = [], 0
    for f in fails:
        ex = rows.get(f["id"])
        if not ex:
            kept.append(f)
            continue
        ref = ex["messages"][f["turn"]]
        got = {"content": f["got"].get("content"), "tool_calls": f["got"].get("tool_calls")}
        house = house_for(ex)
        kind, ok, why, _ = score_turn(house, ref, got, f.get("user", ""), next_act(ex["messages"], f["turn"]), lenient=bool(ex.get("wild")), category=ex["category"])
        if ok:
            flipped += 1
            report["by_category"][f["category"]]["ok"] += 1
            report["by_kind"][kind]["ok"] += 1
        else:
            f["why"] = why
            kept.append(f)
    total_ok = sum(v["ok"] for v in report["by_kind"].values())
    report["pass"] = total_ok / max(1, report["turns"])
    report["rescored_flips"] = report.get("rescored_flips", 0) + flipped
    json.dump(report, open(rp, "w"), indent=2)
    with open(fp, "w") as out:
        for f in kept:
            out.write(json.dumps(f) + "\n")
    print(f"{a.run}: {flipped} turns flipped to pass; pass now {report['pass']:.1%}")


if __name__ == "__main__":
    main()
