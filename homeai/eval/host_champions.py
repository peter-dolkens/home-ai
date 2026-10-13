"""Champion set of one serving host other than the reference GPU box (Peter, 29 Sep: "a champion set of models that run on
falcon too, a separate stream for low-power options"). Everything here comes from that host's own reports, and its
timings are only ever ranked against each other: never against cassowary's (or any other host's) numbers.

  python3 -m homeai.eval.host_champions --host falcon --eval-dir outputs/eval-falcon   # outputs/FALCON_CHAMPIONS.{json,md}

Accuracy = mean pass over the outcome evals test-v2, wild and wild2 (homeai.eval.e2e); gaps under TIE_PTS are ties.
Speed = the fastest mean seconds per request (cold, homeai.eval.latency) among models within SPEED_WINDOW_PTS of the
accuracy champion. Champions are picked from baseline rows (<model>: served as the reference box serves it); serving
variants (<model>@<host>-<variant>: other settings, quantisations or servers) are compared with their own baseline in the
optimisations section, together with the warm prefix-cache bench (homeai.eval.prefix_bench) and the design E pairs
reconciled from the host's own turn logs (homeai.eval.reconcile_grid --eval-dir ... --out outputs/reconcile-<host>).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

SETS = ("testv2", "wild", "wild2")
TIE_PTS = 1.5
SPEED_WINDOW_PTS = 2.0


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def rows_for(eval_dir: str, host: str) -> list[dict]:
    names = set()
    for d in glob.glob(f"{eval_dir}/*/"):
        m = re.match(r"^(?P<r>.+)-(?:(?:testv2|wild2|wild)-e2e|latency|prefix)$", os.path.basename(d.rstrip("/")))
        if m:
            names.add(m["r"])
    rows = []
    for name in sorted(names):
        reps = {s: load(f"{eval_dir}/{name}-{s}-e2e/report.json") for s in SETS}
        lat = load(f"{eval_dir}/{name}-latency/latency.json")
        pre = load(f"{eval_dir}/{name}-prefix/prefix.json")
        hosts = {r.get("host") for r in [*reps.values(), lat, pre] if r}
        model, _, variant = name.partition("@")
        scores = {s: r and r["pass"] for s, r in reps.items()}
        warm = (pre or {}).get("modes", {}).get("prompt") or {}
        rows.append({"row": name, "model": model, "variant": variant.split("-", 1)[-1] if variant else None,
                     "note": next((r.get("host_note") for r in [lat, *reps.values(), pre] if r and r.get("host_note")), None),
                     "ollama_version": next((r.get("ollama_version") for r in [lat, *reps.values(), pre] if r and r.get("ollama_version")), None),
                     "hosts": sorted(h or "?" for h in hosts), "ok_host": hosts <= {host},
                     "scores": scores,
                     "accuracy": sum(scores.values()) / len(SETS) if all(v is not None for v in scores.values()) else None,
                     "e2e_s_per_turn": (sum(r["seconds_per_turn"] for r in reps.values()) / len(SETS)) if all(reps.values()) else None,
                     "s_per_turn": lat and lat["seconds_per_call"], "s_per_request": lat and lat["cold_mean"],
                     "p90_s_per_request": lat and lat["cold_p90"],
                     "warm_s_per_request": warm.get("warm_mean"), "prefix_cold_s_per_request": warm.get("cold_mean"),
                     "prefix": (pre or {}).get("modes")})
    return rows


def reference_accuracy(model: str):
    """The same model's outcome score on the reference box (accuracy only: checks that the host serves the same weights)."""
    reps = [load(f"outputs/eval/{model}-{s}-e2e/report.json") for s in SETS]
    return sum(r["pass"] for r in reps) / len(SETS) if all(reps) else None


def pick(rows: list[dict]) -> tuple:
    scored = [r for r in rows if r["accuracy"] is not None]
    if not scored:
        return None, [], None
    best = max(scored, key=lambda r: r["accuracy"])
    ties = [r["row"] for r in scored if r is not best and best["accuracy"] - r["accuracy"] < TIE_PTS / 100]
    near = [r for r in scored if r["s_per_request"] is not None and best["accuracy"] - r["accuracy"] <= SPEED_WINDOW_PTS / 100]
    fast = min(near, key=lambda r: r["s_per_request"], default=None)
    return best["row"], ties, fast and fast["row"]


def pairs(reconcile_dir: str) -> list[dict]:
    out = [load(p) for p in sorted(glob.glob(f"{reconcile_dir}/*__*.json"))]
    return sorted((p for p in out if p), key=lambda p: -p["reconciled_pass"])


def optimisations(rows: list[dict]) -> list[dict]:
    by = {r["row"]: r for r in rows}
    out, alone = [], []
    for r in rows:
        b = by.get(r["model"])
        if not r["variant"]:
            continue
        if not b:  # no baseline row on this host (e.g. a model only measured on another server build): listed last, no deltas
            alone.append({"row": r["row"], "baseline": None, "note": r["note"], "accuracy": r["accuracy"], "d_accuracy": None,
                          "s_per_request": r["s_per_request"], "d_s_per_request": None, "warm_s_per_request": r["warm_s_per_request"],
                          "d_warm_s_per_request": None, "s_per_turn": r["s_per_turn"], "d_s_per_turn": None})
            continue
        d = lambda k: None if r[k] is None or b[k] is None else r[k] - b[k]  # noqa: E731
        out.append({"row": r["row"], "baseline": b["row"], "note": r["note"], "accuracy": r["accuracy"], "d_accuracy": d("accuracy"),
                    "s_per_request": r["s_per_request"], "d_s_per_request": d("s_per_request"),
                    "warm_s_per_request": r["warm_s_per_request"], "d_warm_s_per_request": d("warm_s_per_request"),
                    "s_per_turn": r["s_per_turn"], "d_s_per_turn": d("s_per_turn")})
    return out + alone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="falcon")
    ap.add_argument("--eval-dir", default="outputs/eval-falcon")
    ap.add_argument("--reconcile-dir", default=None, help="default outputs/reconcile-<host>")
    ap.add_argument("--out", default=None, help="default outputs/<HOST>_CHAMPIONS")
    a = ap.parse_args()
    out = a.out or f"outputs/{a.host.upper()}_CHAMPIONS"
    rows = [r for r in rows_for(a.eval_dir, a.host)]
    for r in rows:
        r["reference_accuracy"] = reference_accuracy(r["model"]) if not r["variant"] else None
    good = [r for r in rows if r["ok_host"]]
    best, ties, fast = pick([r for r in good if not r["variant"]])
    best_all, ties_all, fast_all = pick(good)
    E = pairs(a.reconcile_dir or f"outputs/reconcile-{a.host}")
    opt = optimisations(good)
    hw = next((h for h in ((load(p) or {}).get("hardware") for p in glob.glob(f"{a.eval_dir}/*/*.json")) if h), None)
    rep = {"host": a.host, "hardware": hw, "eval_dir": a.eval_dir, "timings": f"{a.host} only: never comparable with cassowary or any other host",
           "accuracy": best, "accuracy_ties": ties, "speed": fast, "tie_pts": TIE_PTS, "speed_window_pts": SPEED_WINDOW_PTS,
           "with_optimisations": {"accuracy": best_all, "accuracy_ties": ties_all, "speed": fast_all},
           "rows": rows, "pairs": E, "optimisations": opt,
           "other_host_rows": [r["row"] for r in rows if not r["ok_host"]]}
    json.dump(rep, open(out + ".json", "w"), indent=2)

    f = lambda v: "-" if v is None else f"{v:.1%}"  # noqa: E731
    g = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    sg = lambda v: "-" if v is None else f"{v:+.2f}"  # noqa: E731
    sp = lambda v: "-" if v is None else f"{v * 100:+.1f}"  # noqa: E731
    H = a.host
    L = [f"# {H.capitalize()} champions ({hw or H})\n",
         f"**All timings on this page were measured on {H} and are only comparable with other {H} runs. Never rank them "
         f"against cassowary (or any other host): different hardware, different serving stack.** The {H} stream is the "
         f"low-power option to weigh when choosing the complete architecture.\n",
         f"Accuracy = mean pass over the outcome evals ({', '.join(SETS)}; homeai.eval.e2e, served on {H}). Speed = mean seconds "
         f"per request, cold (homeai.eval.latency --n 60: the first user request of 60 test-v2 conversations through the full "
         f"tool loop), idle machine, one model loaded. Gaps under {TIE_PTS} points are ties; the speed champion is the fastest "
         f"model within {SPEED_WINDOW_PTS:.0f} points of the accuracy champion. Champions come from baseline rows (served as "
         f"cassowary serves them: Ollama 0.21.2, Q4_K_M, same Modelfile and server settings).\n",
         f"- **Accuracy champion:** {best or 'none yet'}" + (f" (tied within {TIE_PTS} pts: {', '.join(ties)})" if ties else ""),
         f"- **Speed champion (within {SPEED_WINDOW_PTS:.0f} pts):** {fast or 'none yet'}\n",
         "| model | test-v2 | wild | wild2 | accuracy | ref. accuracy¹ | s/turn | s/request | p90 s/request | warm s/request² |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted([r for r in good if not r["variant"]], key=lambda r: -(r["accuracy"] or 0)):
        tag = " **(acc)**" if r["row"] == best else ""
        tag += " **(speed)**" if r["row"] == fast else ""
        L.append(f"| {r['row']}{tag} | " + " | ".join(f(r["scores"][s]) for s in SETS) + f" | {f(r['accuracy'])} | {f(r['reference_accuracy'])}"
                 f" | {g(r['s_per_turn'])} | {g(r['s_per_request'])} | {g(r['p90_s_per_request'])} | {g(r['warm_s_per_request'])} |")
    L += ["", "¹ The same weights' outcome score on cassowary: accuracy only, a check that falcon serves the same model. "
          "² Warm prefix cache (homeai.eval.prefix_bench, same house and identical system prompt, later requests of a group).", ""]
    if E:
        L += [f"## Design E pairs ({H} turn logs only)\n", "Fast model acts at once, slow model overrules on disagreement "
              f"(homeai.eval.reconcile); both halves timed on {H}.\n",
              "| fast -> slow | final pass | fast pass | flicker (acting) | first action s | final state s | slow alone s |", "|---|---|---|---|---|---|---|"]
        for p in E[:15]:
            L.append(f"| {p['fast']} -> {p['slow']} | {f(p['reconciled_pass'])} | {f(p['fast_pass'])} | {f(p['flicker_rate_acting'])} | "
                     f"{g(p['time_to_first_action_s_acting'])} | {g(p['time_to_final_state_s_acting'])} | {g(p['slow_alone_s_acting'])} |")
        L.append("")
    if opt or any(r["prefix"] for r in good):
        L += [f"## Optimisations ({H} phase 2)\n", f"Each variant against its own {H} baseline (same weights unless the variant is a "
              "different quantisation or server). Deltas: accuracy in points, times in seconds (negative = faster).\n"]
        if opt:
            L += ["| variant | setup | accuracy | Δ acc (pts) | s/request | Δ | warm s/request | Δ | s/turn | Δ |", "|---|---|---|---|---|---|---|---|---|---|"]
            for o in opt:
                L.append(f"| {o['row']} | {(o['note'] or '').replace('falcon stream: ', '')} | {f(o['accuracy'])} | {sp(o['d_accuracy'])} | "
                         f"{g(o['s_per_request'])} | {sg(o['d_s_per_request'])} | {g(o['warm_s_per_request'])} | {sg(o['d_warm_s_per_request'])} | "
                         f"{g(o['s_per_turn'])} | {sg(o['d_s_per_turn'])} |")
            L.append("")
        pre = [r for r in good if r["prefix"]]
        if pre:
            L += ["### Warm prefix cache\n", "Requests grouped by house; the first of a group is cold (another house was last), the rest warm. "
                  "prompt = identical system prompt; house = same house but the Now / Speaker / People lines differ (today's deployment: "
                  "they sit above the inventory); header-last = those lines moved below the rules (timing only). Tokens = prompt tokens "
                  "the server actually evaluated on a request's first call (a cache hit shows as a small number).\n",
                  "| row | mode | cold s/request | warm s/request | cold tokens evaluated | warm tokens evaluated | prompt tokens |", "|---|---|---|---|---|---|---|"]
            for r in pre:
                for mode, m in r["prefix"].items():
                    L.append(f"| {r['row']} | {mode} | {g(m.get('cold_mean'))} | {g(m.get('warm_mean'))} | {g(m.get('cold_eval_tokens'))} | "
                             f"{g(m.get('warm_eval_tokens'))} | {g(m.get('prompt_tokens'))} |")
            L.append("")
        L += [f"With optimisations allowed: accuracy {best_all or '-'}, speed {fast_all or '-'}.\n"]
        rec = os.path.join(a.eval_dir, "RECOMMENDATION.md")
        if os.path.exists(rec):
            L += [open(rec).read().strip(), ""]
    if rep["other_host_rows"]:
        L += [f"Left out (reports from another host): {', '.join(rep['other_host_rows'])}", ""]
    open(out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
