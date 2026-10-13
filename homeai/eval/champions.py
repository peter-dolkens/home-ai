"""Champion pick + results table. Peter keeps two models: the best overall, and the best whose
average response time per model turn is <= 1 s (his yardstick: the per-turn averages the evals
report, measured on an idle GPU). Old models are only pruned when the disk runs short.

Quality = mean pass rate over the CURRENT test sets (a model needs all of them to qualify):
  test-v2 (templates, all categories), wild (agent-written, unseen houses), para (held-out phrasings),
  wild2 (second agent-written set, 240 conversations, never used for failure analysis), and the
  outcome evals wild-e2e / wild2-e2e (homeai.eval.e2e: free-running tool loop, judged on final house
  state + spoken facts, so any valid fetch strategy passes).
Speed = seconds_per_call from homeai.eval.latency (idle GPU, full agent loop, same model turns).

  python3 -m homeai.eval.champions            # outputs/CHAMPIONS.json, CHAMPIONS.md, results.csv
  python3 -m homeai.eval.champions --missing  # "<model> <set>" lines still to run (sets + latency)
  python3 -m homeai.eval.champions --prunable # non-champion fine-tunes, oldest first
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time

EVAL = "outputs/eval"
SETS = {"test-v2": "data/generated/test-v2.jsonl", "wild": "data/generated/wild.jsonl",
        "para": "data/generated/test-v2-para.jsonl", "wild2": "data/generated/wild2.jsonl",
        "wild-e2e": "data/generated/wild.jsonl", "wild2-e2e": "data/generated/wild2.jsonl",
        "test-v2-e2e": "data/generated/test-v2.jsonl",
        "rules": "data/generated/test-rules.jsonl"}
# third champion (Peter, 24 Sep): best on outcome evals alone, no speed limit
OUTCOME_SETS = ["test-v2-e2e", "wild-e2e", "wild2-e2e"]
LIMIT_S = 1.0
# para is shown but not scored: v2 training data predates the 80/20 paraphrase split and contains 43% of
# para's user utterances verbatim (v3+ data: 21%), so it favours v2-trained models.
QUALITY_EXCLUDE = {"para", "test-v2-e2e", "rules"}  # test-v2-e2e only feeds the outcome champion
# rules (Peter, 26 Sep): 200 template conversations testing when to ask "which one?" (one match in the room or in the
# house -> act; several -> ask). A diagnostic for that behaviour, shown but not scored.


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def eval_dirs(model: str):
    """test set -> report dir, following the naming used by overnight.sh."""
    m = model.replace(":", "_") if model.startswith("qwen") else model
    d = {"test-v1": [f"{model}", f"baseline-{model.replace(':', '-')}-full"],
         "test-v2": [f"{model}-testv2", f"{model}-baseline-testv2", model],
         "wild": [f"{model}-wild"], "para": [f"{model}-para"], "wild2": [f"{model}-wild2"],
         "wild-e2e": [f"{model}-wild-e2e"], "wild2-e2e": [f"{model}-wild2-e2e"], "test-v2-e2e": [f"{model}-testv2-e2e"],
         "rules": [f"{model}-rules"]}
    return d, m


def report(model: str, s: str):
    d, _ = eval_dirs(model)
    for name in d[s]:
        r = load(f"{EVAL}/{name}/report.json")
        if not r:
            continue
        if s == "test-v1" and r.get("turns") != 326:
            continue
        if s == "test-v2" and (r.get("turns") == 326 or r.get("data") not in (None, SETS["test-v2"])):
            continue
        return name, r
    return None, None


def current_sets():
    return [s for s, p in SETS.items() if os.path.exists(p)]


def train_stats(cfg_path):
    if not cfg_path:
        return {}
    cfg = load(cfg_path) or {}
    log = f"outputs/{os.path.basename(cfg_path)[:-5]}-train.log"
    st = {k: cfg.get(k) for k in ("base_model", "lora_r", "lora_alpha", "epochs", "lr", "batch_size", "grad_accum", "max_seq_len")}
    st["load_in_4bit"] = cfg.get("load_in_4bit", True)
    try:
        txt = open(log, errors="replace").read()
        m = re.findall(r"'train_loss': ([0-9.]+)", txt)
        st["train_loss"] = float(m[-1]) if m else None
        m = re.findall(r"'eval_loss': ([0-9.]+)", txt)
        st["eval_loss"] = float(m[-1]) if m else None
        m = re.findall(r"'train_runtime': ([0-9.]+)", txt)
        st["train_hours"] = round(float(m[-1]) / 3600, 2) if m else None
    except OSError:
        pass
    return st


def rows_for(models: dict):
    sets = current_sets()
    rows = []
    for m, meta in models.items():
        if m.startswith("_"):
            continue
        scores = {s: (report(m, s)[1] or {}).get("pass") for s in sets}
        lat = load(f"{EVAL}/{m.replace(':', '_')}-latency/latency.json")
        clean = bool(lat) and not lat.get("training_running")
        rows.append({"model": m, **meta, "scores": scores,
                     "quality": (lambda q: sum(q) / len(q))([v for k, v in scores.items() if k not in QUALITY_EXCLUDE])
                                if all(v is not None for k, v in scores.items() if k != "test-v2-e2e") else None,
                     "outcome": (sum(scores[k] for k in OUTCOME_SETS) / len(OUTCOME_SETS))
                                if all(scores.get(k) is not None for k in OUTCOME_SETS) else None,
                     "s_per_turn": lat["seconds_per_call"] if clean else None,
                     "s_per_request": lat["cold_mean"] if clean else None})
    return sets, rows


def write_csv(models, rows, path="outputs/results.csv"):
    cols = ["model", "method", "data", "train_examples", "base_model", "load_in_4bit", "lora_r", "lora_alpha", "epochs", "lr",
            "batch_size", "grad_accum", "max_seq_len", "train_loss", "eval_loss", "train_hours", "test_set", "eval_dir",
            "examples", "turns", "pass", "fetch", "answer", "act", "clarify", "refuse", "eval_s_per_turn",
            "bench_s_per_turn", "bench_s_per_request", "quality_score", "outcome_score", "champion", "updated"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        champs = load("outputs/CHAMPIONS.json") or {}
        for r in rows:
            meta = models[r["model"]]
            st = train_stats(meta.get("config"))
            for s in ["test-v1", *SETS]:
                name, rep = report(r["model"], s)
                if not rep:
                    continue
                kinds = rep.get("by_kind", {})
                kp = {k: round(v["ok"] / v["n"], 4) for k, v in kinds.items() if v.get("n")}
                w.writerow({"model": r["model"], "method": meta.get("method"), "data": meta.get("data"), "train_examples": meta.get("train_examples"),  # untuned baselines have no count
                            **st, "test_set": s, "eval_dir": name, "examples": rep["examples"], "turns": rep["turns"],
                            "pass": round(rep["pass"], 4), **kp, "eval_s_per_turn": rep.get("seconds_per_turn") and round(rep["seconds_per_turn"], 3),
                            "bench_s_per_turn": r["s_per_turn"] and round(r["s_per_turn"], 3),
                            "bench_s_per_request": r["s_per_request"] and round(r["s_per_request"], 3),
                            "quality_score": r["quality"] and round(r["quality"], 4),
                            "outcome_score": r["outcome"] and round(r["outcome"], 4),
                            "champion": ",".join(k for k in ("best", "best_fast", "best_outcome") if champs.get(k) == r["model"]),
                            "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(f"{EVAL}/{name}/report.json")))})


def design_keep():
    """Models the per-design and per-tier champions (homeai.eval.design_champions) and the fast/slow pairs rely on:
    never prunable (a prune on 28 Sep removed the 8-9B tier champion homeai-9b-v2)."""
    d = load("outputs/DESIGN_CHAMPIONS.json") or {}
    keep = set()
    for t in (d.get("tiers") or {}).values():
        keep |= {t[k]["model"] for k in ("accuracy_champion", "speed_champion") if t.get(k)}
    for k in ("large_house",):
        if (d.get("A") or {}).get(k):
            keep.add(d["A"][k]["model"])
    if (d.get("B") or {}).get("best"):
        keep.add(d["B"]["best"]["model"])
    for pair in (d.get("E") or {}).get("all") or {}:
        keep |= set(pair.split("__"))
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="configs/models.json")
    ap.add_argument("--missing", action="store_true")
    ap.add_argument("--prunable", action="store_true")
    a = ap.parse_args()
    models = json.load(open(a.models))
    sets, rows = rows_for(models)
    if a.missing:
        for r in rows:
            if not r["config"] or r.get("retired") or r.get("track"):
                continue  # untuned baselines, retired runs and architecture-track models (own eval sets): not candidates
            for s in sets:
                if r["scores"][s] is None:
                    print(r["model"], s)
            if r["s_per_turn"] is None:
                print(r["model"], "latency")
        return
    tuned = [r for r in rows if r["config"] and not r.get("retired") and r["quality"] is not None]
    ranked = sorted(tuned, key=lambda r: -r["quality"])
    best = ranked[0] if ranked else None
    fast = next((r for r in ranked if r["s_per_turn"] is not None and r["s_per_turn"] <= LIMIT_S), None)
    outc = max([r for r in rows if r["config"] and not r.get("retired") and r["outcome"] is not None], key=lambda r: r["outcome"], default=None)
    if a.prunable:
        newest = [m for m, v in models.items() if not m.startswith("_") and v.get("config")][-1]
        keep = {x["model"] for x in (best, fast, outc) if x} | {newest} | design_keep()
        for r in rows:  # registry order = oldest first
            if r["config"] and r["model"] not in keep and r["quality"] is not None:
                print(r["model"], r["config"])
        return
    out = {"sets": sets, "scored_sets": [x for x in sets if x not in QUALITY_EXCLUDE], "limit_s": LIMIT_S, "best": best and best["model"], "best_fast": fast and fast["model"], "best_outcome": outc and outc["model"], "outcome_sets": OUTCOME_SETS, "rows": rows}
    json.dump(out, open("outputs/CHAMPIONS.json", "w"), indent=2)
    f = lambda v: "-" if v is None else f"{v:.1%}"  # noqa: E731
    g = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    lines = ["# Champions\n", f"Quality = mean pass over {', '.join(x for x in sets if x not in QUALITY_EXCLUDE)} (para shown, not scored: leaked into v2 training data). Speed = seconds per model turn, idle GPU (limit {LIMIT_S:.0f} s).\n",
             f"- **Best overall:** {out['best']}", f"- **Best at <= {LIMIT_S:.0f} s/turn:** {out['best_fast'] or 'none yet'}",
             f"- **Best on outcomes ({', '.join(OUTCOME_SETS)}):** {out['best_outcome'] or 'none yet'}\n",
             "| model | method | data | " + " | ".join(sets) + " | quality | outcome | s/turn | s/request |", "|" + "---|" * (len(sets) + 7)]
    for r in sorted(rows, key=lambda r: -(r["quality"] or 0)):
        lines.append(f"| {r['model']} | {r['method']} | {r['data']} | " + " | ".join(f(r['scores'][s]) for s in sets)
                     + f" | {f(r['quality'])} | {f(r['outcome'])} | {g(r['s_per_turn'])} | {g(r['s_per_request'])} |")
    open("outputs/CHAMPIONS.md", "w").write("\n".join(lines) + "\n")
    write_csv(models, rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
