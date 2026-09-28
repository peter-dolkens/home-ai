"""A champion per architecture (Peter, 28 Sep): the single-model design keeps its three slots (homeai.eval.champions)
plus a large-house slot; design B, the tiered pipeline (C/D), the room leaf and the fast+slow pair (E) each get their
own, scored on the sets that fit the design and reported with their speed.

    python3 -m homeai.eval.design_champions      # writes outputs/DESIGN_CHAMPIONS.{json,md}

Inputs: outputs/eval/<model>-<set>-e2e/report.json (A, B), outputs/eval/tiered-<mode>-<leaf>-<set>/report.json (C/D),
outputs/leaf-results/*.summary.json (leaf tests collected from the machines, named <leaf>__<test>.summary.json),
outputs/reconcile/*.json (E, homeai.eval.reconcile).
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics

EVAL = "outputs/eval"


def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def design_a():
    ch = load("outputs/CHAMPIONS.json") or {}
    big = {}
    for p in glob.glob(f"{EVAL}/*-bigbench32k-e2e/report.json"):
        m = os.path.basename(os.path.dirname(p))[: -len("-bigbench32k-e2e")]
        r = load(p)
        big[m] = (r["pass"], r.get("seconds_per_turn"))
    best_big = max(big.items(), key=lambda kv: kv[1][0]) if big else None
    return {"best": ch.get("best"), "best_fast": ch.get("best_fast"), "best_outcome": ch.get("best_outcome"),
            "large_house": best_big and {"model": best_big[0], "pass": best_big[1][0], "s_per_turn": best_big[1][1]},
            "large_house_all": {m: v[0] for m, v in sorted(big.items(), key=lambda kv: -kv[1][0])}}


B_SETS = ["bigbench-b", "testv2-b", "wild-b", "wild2-b"]


def design_b():
    rows = {}
    for p in glob.glob(f"{EVAL}/*-bigbench-b-e2e/report.json"):
        m = os.path.basename(os.path.dirname(p))[: -len("-bigbench-b-e2e")]
        scores = {s: (load(f"{EVAL}/{m}-{s}-e2e/report.json") or {}).get("pass") for s in B_SETS}
        spt = mean([(load(f"{EVAL}/{m}-{s}-e2e/report.json") or {}).get("seconds_per_turn") for s in B_SETS])
        rows[m] = {"scores": scores, "score": mean(scores.values()), "s_per_turn": spt}
    best = max(rows.items(), key=lambda kv: kv[1]["score"] or 0) if rows else None
    return {"best": best and {"model": best[0], **best[1]}, "all": rows}


TIERED = re.compile(r"^tiered-(?P<cfg>.+)-(?P<set>bigbench|testv2|wild2|wild)$")


def design_tiered():
    cfgs: dict = {}
    for p in glob.glob(f"{EVAL}/tiered-*/report.json"):
        m = TIERED.match(os.path.basename(os.path.dirname(p)))
        if not m:
            continue
        r = load(p)
        lat = (r.get("latency") or {}).get("turn_s") or {}
        c = cfgs.setdefault(m["cfg"], {"scores": {}, "turn_p50": [], "turn_mean": []})
        c["scores"][m["set"]] = r["pass"]
        c["turn_p50"].append(lat.get("p50"))
        c["turn_mean"].append(lat.get("mean"))
    out = {k: {"scores": v["scores"], "score": mean(v["scores"].values()), "turn_p50_s": mean(v["turn_p50"]),
               "turn_mean_s": mean(v["turn_mean"])} for k, v in cfgs.items()}
    def best_of(pred):
        full = {k: v for k, v in out.items() if pred(k) and v["scores"]}
        b = max(full.items(), key=lambda kv: (len(kv[1]["scores"]) >= 2, kv[1]["score"] or 0)) if full else None
        return b and {"config": b[0], **b[1]}
    # C routes each instruction to the rooms it concerns; D sends the request to every room leaf (blind fan-out)
    return {"best": best_of(lambda k: not k.startswith("fanout")), "best_fanout": best_of(lambda k: k.startswith("fanout")), "all": out}


def leaves():
    rows = {}
    for p in glob.glob("outputs/leaf-results/*__bundook.summary.json"):
        leaf = os.path.basename(p).split("__")[0]
        o = (load(p) or {}).get("overall") or {}
        rows[leaf] = {k: o.get(k) for k in ("end_state", "exact", "refusal", "false_refusal", "latency_ms_p50")}
    best = max(rows.items(), key=lambda kv: kv[1]["end_state"] or 0) if rows else None
    return {"best": best and {"leaf": best[0], **best[1]}, "all": rows}


def design_e():
    rows = {os.path.basename(p)[:-5]: load(p) for p in glob.glob("outputs/reconcile/*.json")}
    rows = {k: v for k, v in rows.items() if v}
    # final accuracy is the slow model's by construction; among pairs, prefer the fewest visible wrong actions per point of speed-up
    best = max(rows.items(), key=lambda kv: ((kv[1].get("reconciled_pass") or 0) - (kv[1].get("flicker_rate") or 0))) if rows else None
    return {"best": best and {"pair": best[0], **best[1]}, "all": rows}


ACC_SETS = [("test-v2-e2e", "testv2-e2e"), ("wild-e2e", "wild-e2e"), ("wild2-e2e", "wild2-e2e"), ("rules", "rules"), ("bigbench32k", "bigbench32k-e2e")]


def tiers():
    """Per size tier: the accuracy champion (mean of the three outcome evals) and the speed champion (fastest per request
    within 2 points of it)."""
    reg = load("configs/models.json") or {}
    ch = {r["model"]: r for r in (load("outputs/CHAMPIONS.json") or {}).get("rows", [])}
    by_tier: dict = {}
    for m, meta in reg.items():
        if m.startswith("_") or not meta.get("config") or meta.get("track") or meta.get("retired") or not meta.get("tier"):
            continue
        scores = {k: (load(f"{EVAL}/{m}-{d}/report.json") or {}).get("pass") for k, d in ACC_SETS}
        if any(scores[k] is None for k in ("test-v2-e2e", "wild-e2e", "wild2-e2e")):
            continue
        reps = [load(f"{EVAL}/{m}-{d}/report.json") for d in ("testv2-e2e", "wild-e2e", "wild2-e2e")]
        req = mean([r["seconds_per_turn"] * r["calls_per_request"] for r in reps if r and r.get("seconds_per_turn") and r.get("calls_per_request")])
        bench = (ch.get(m) or {}).get("s_per_request")
        # rank on the three outcome evals every candidate has; rules and large houses are shown alongside, not averaged in,
        # so a model is never compared on a different set of tests than its rivals
        acc3 = mean([scores[k] for k in ("test-v2-e2e", "wild-e2e", "wild2-e2e")])
        by_tier.setdefault(meta["tier"], []).append({"model": m, "scores": scores, "accuracy": acc3,
                                                     "s_per_request": bench or req, "timed_on": "latency bench" if bench else "outcome evals"})
    out = {}
    for t, rows in by_tier.items():
        acc = max(rows, key=lambda r: r["accuracy"])
        close = [r for r in rows if r["accuracy"] >= acc["accuracy"] - 0.02 and r["s_per_request"] is not None]
        fast = min(close, key=lambda r: r["s_per_request"]) if close else acc
        out[t] = {"accuracy_champion": acc, "speed_champion": fast, "all": sorted(rows, key=lambda r: -r["accuracy"])}
    return out


def pct(v):
    return "-" if v is None else f"{v:.1%}"


def sec(v):
    return "-" if v is None else f"{v:.2f} s"


def main():
    out = {"A": design_a(), "B": design_b(), "tiered": design_tiered(), "leaf": leaves(), "E": design_e(), "tiers": tiers()}
    json.dump(out, open("outputs/DESIGN_CHAMPIONS.json", "w"), indent=1)
    a, b, t, l, e = out["A"], out["B"]["best"], out["tiered"]["best"], out["leaf"]["best"], out["E"]["best"]
    lh = a["large_house"] or {}
    lines = ["# Champions per design", "",
             "| Design | Champion | Score | Speed |", "|---|---|---|---|",
             f"| A: single model (quality) | {a['best']} | see CHAMPIONS.md | |",
             f"| A: single model (<=1 s/turn) | {a['best_fast']} | | |",
             f"| A: single model (outcome) | {a['best_outcome']} | | |",
             f"| A: large houses, 32k window | {lh.get('model', '-')} | {pct(lh.get('pass'))} bigbench | {sec(lh.get('s_per_turn'))}/turn |",
             f"| B: lookup tools | {b['model'] if b else '-'} | {pct(b and b['score'])} (mean of {', '.join(B_SETS)}) | {sec(b and b['s_per_turn'])}/turn |",
             f"| C/D: tiered pipeline | {t['config'] if t else '-'} | {pct(t and t['score'])} ({', '.join(t['scores']) if t else ''}) | p50 {sec(t and t['turn_p50_s'])}/turn |",
             f"| Room leaf (component) | {l['leaf'] if l else '-'} | {pct(l and l['end_state'])} end state, Bundook leaf test | {'-' if not l else str(l['latency_ms_p50']) + ' ms'} p50 |",
             f"| E: fast + slow | {e['pair'] if e else '-'} | {pct(e and e.get('reconciled_pass'))} final, {pct(e and e.get('flicker_rate'))} flicker | first action {sec(e and e.get('time_to_first_action_s_acting'))} |"]
    lines += ["", "## Per size tier", "", "| Tier | Accuracy champion | Accuracy | Speed champion | Speed |", "|---|---|---|---|---|"]
    for t, v in sorted(out["tiers"].items()):
        a_, f_ = v["accuracy_champion"], v["speed_champion"]
        lines.append(f"| {t} | {a_['model']} | {pct(a_['accuracy'])} | {f_['model']} | {sec(f_['s_per_request'])}/request |")
    open("outputs/DESIGN_CHAMPIONS.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
