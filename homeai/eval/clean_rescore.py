"""Outcome evals rescored on uncontaminated turns only: do the model rankings hold without the templated turns a model
also saw in training? No test set changes, no model runs; this reads the e2e run dirs and the training data.

homeai.datagen.leakage found the synthetic test sets share unaugmented user turns with the training data ("yes please",
"go ahead", "how's the house looking, we're not home"): the generator's shared templates. Per model, a test turn is
  contaminated  its user text, after leakage.norm (homeai.voice.vocab's normalisation), is a user turn of the model's
                own training data (configs/models.json -> config "train", plus the parent's data for an init_adapter)
  strict        contaminated, or any earlier user turn of the same conversation is (the context is partly seen)
and pass rates are reported on all / clean / contaminated / strict-clean turns of test-v2-e2e, wild-e2e and wild2-e2e.

Turn k is the k-th user message of the row (0-based, homeai.eval.e2e). Per-turn outcomes come from turns.jsonl where
it exists; older runs predate it, so a turn passes unless failures.jsonl lists it (half credit when the failure is an
unconfirmed sensitive action). failures.jsonl also wins over turns.jsonl, because e2e.rescore rewrites failures and
report.json but not turns.jsonl. Every rebuilt run is checked against its report.json pass rate.

  python3 -m homeai.eval.clean_rescore       # outputs/CLEAN_RESCORE.json, CLEAN_RESCORE.md
  python3 -m homeai.eval.clean_rescore --root outputs/eval-cassowary

--root is a read-only copy of cassowary's run dirs (<root>/eval), configs (<root>/configs) and, for models the registry
hosts elsewhere, that host's run dirs (<root>/eval-<host>, e.g. eval-lapwing). Training and test data are read from
data/generated (identical on both machines). Normalised training user turns are cached per file in <root>/train-norm.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

from ..datagen.leakage import norm, user_texts

SETS = {"test-v2-e2e": ("testv2", "data/generated/test-v2.jsonl"), "wild-e2e": ("wild", "data/generated/wild.jsonl"),
        "wild2-e2e": ("wild2", "data/generated/wild2.jsonl")}
CHAMPIONS = ["homeai-4b-lora-v5", "homeai-4b-lora-v7", "homeai-4b-lora-v8", "homeai-9b-v2", "homeai-gemma4-e4b-v5",
             "homeai-2b-lora-v5", "homeai-lfm25-1.2b-lora-v5-fix2"]
HALF_WHY = "unconfirmed sensitive action (half credit)"
VIEWS = ("all", "clean", "contaminated", "strict_clean")


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


# ---------------------------------------------------------------------- training side
def train_users(path: str, cache_dir: str) -> set:
    """Normalised user turns of one training file, cached next to the eval copy (keyed on size and mtime)."""
    st = os.stat(path)
    key = f"{os.path.basename(path)[:-6]}.{st.st_size}.{int(st.st_mtime)}.json"
    cached = load(os.path.join(cache_dir, key))
    if cached is not None:
        return set(cached)
    out = set()
    for l in open(path):
        if l.strip():
            out.update(k for k in map(norm, user_texts(json.loads(l))) if k)
    os.makedirs(cache_dir, exist_ok=True)
    json.dump(sorted(out), open(os.path.join(cache_dir, key), "w"))
    return out


def resolve_config(cfg_path: str, root: str):
    """The copied config first (what cassowary trained with), the local one otherwise."""
    for p in (os.path.join(root, cfg_path), cfg_path):
        c = load(p)
        if c is not None:
            return c
    return None


def train_files(cfg_path, root: str, configs: dict) -> list[str]:
    """Training files a model has seen: its own, plus its parent's when it starts from another run's adapter (RFT)."""
    cfg = resolve_config(cfg_path, root) if cfg_path else None
    if not cfg:
        return []
    files = [cfg["train"]]
    parent = cfg.get("init_adapter")
    if parent:
        run = os.path.dirname(parent.rstrip("/")) if parent.rstrip("/").endswith("/lora") else parent.rstrip("/")
        up = next((p for p, c in configs.items() if c and c.get("out") == run), None)
        files += [f for f in train_files(up, root, configs) if f not in files] if up else [f"<unresolved {parent}>"]
    return files


# ---------------------------------------------------------------------- test side
def test_turns(path: str) -> dict:
    """(id, turn) -> normalised user text, turn = index among the row's user messages (as homeai.eval.e2e counts)."""
    out = {}
    for ex in jsonl(path):
        users = [m for m in ex["messages"] if m["role"] == "user"]
        for k, m in enumerate(users):
            out[(ex["id"], k)] = norm(m["content"])
    return out


def run_outcomes(run_dir: str, turns: dict):
    """(id, turn) -> ok (1, 0.5 or 0) for one e2e run, and how it was rebuilt."""
    fails = {(f["id"], f["turn"]): f for f in jsonl(os.path.join(run_dir, "failures.jsonl"))}
    tpath = os.path.join(run_dir, "turns.jsonl")
    logged = {(t["id"], t["turn"]): t["ok"] for t in jsonl(tpath)} if os.path.exists(tpath) else {}
    keys = logged or turns
    out = {}
    for key in keys:
        f = fails.get(key)
        if f is None:
            out[key] = 1
        elif key in logged:
            out[key] = logged[key]
        else:
            out[key] = 0.5 if f.get("why", "").startswith(HALF_WHY) else 0
    return out, "turns.jsonl" if logged else "failures.jsonl"


def contamination(turns: dict, seen: set):
    """(id, turn) -> (contaminated, strict): strict also marks every turn after a contaminated one in the conversation."""
    by_conv = collections.defaultdict(list)
    for (i, k) in turns:
        by_conv[i].append(k)
    out = {}
    for i, ks in by_conv.items():
        earlier = False
        for k in sorted(ks):
            hit = turns[(i, k)] in seen
            out[(i, k)] = (hit, hit or earlier)
            earlier = earlier or hit
    return out


def score(outcomes: dict, flags: dict) -> dict:
    groups = {"all": list(outcomes), "clean": [k for k in outcomes if not flags[k][0]],
              "contaminated": [k for k in outcomes if flags[k][0]], "strict_clean": [k for k in outcomes if not flags[k][1]]}
    return {v: {"n": len(ks), "pass": (sum(outcomes[k] for k in ks) / len(ks)) if ks else None} for v, ks in groups.items()}


# ---------------------------------------------------------------------- report
def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs and all(x is not None for x in xs) else None


def ranks(rows, view):
    ranked = sorted([r for r in rows if r["mean"][view] is not None], key=lambda r: -r["mean"][view])
    return {r["model"]: n + 1 for n, r in enumerate(ranked)}


def build(root: str, registry: str, cache_dir: str) -> dict:
    models = {m: v for m, v in json.load(open(registry)).items() if not m.startswith("_")}
    configs = {v["config"]: resolve_config(v["config"], root) for v in models.values() if v.get("config")}
    turns = {s: test_turns(p) for s, (_, p) in SETS.items()}
    seen_cache, rows, warnings, detail = {}, [], [], {}

    def seen_for(files):
        key = tuple(files)
        if key not in seen_cache:
            seen_cache[key] = set().union(*(train_users(f, cache_dir) for f in files if os.path.exists(f))) if files else set()
            warnings.extend(f"missing training file {f}" for f in files if not os.path.exists(f))
        return seen_cache[key]

    for m, meta in models.items():
        files = train_files(meta.get("config"), root, configs)
        seen = seen_for(files)
        # a model hosted elsewhere (registry "host") is evaluated there: its copy wins over cassowary's
        parents = ([f"eval-{meta['host']}"] if meta.get("host") else []) + ["eval"]
        per_set, detail[m] = {}, {}
        for s, (tag, _) in SETS.items():
            cands = [os.path.join(root, e, f"{m}-{tag}-e2e") for e in parents]
            run = next((c for c in cands if os.path.exists(os.path.join(c, "report.json"))), None)
            if not run:
                continue
            rep = load(os.path.join(run, "report.json"))
            outcomes, how = run_outcomes(run, turns[s])
            missing = [k for k in outcomes if k not in turns[s]]
            if missing:
                warnings.append(f"{run}: {len(missing)} turns not in {SETS[s][1]}")
                outcomes = {k: v for k, v in outcomes.items() if k in turns[s]}
            flags = contamination({k: turns[s][k] for k in outcomes}, seen)
            sc = score(outcomes, flags)
            ok = abs(sc["all"]["pass"] - rep["pass"]) < 1e-9 and sc["all"]["n"] == rep["turns"]
            if not ok:
                warnings.append(f"{run}: rebuilt pass {sc['all']['pass']:.4f} over {sc['all']['n']} turns vs report "
                                f"{rep['pass']:.4f} over {rep['turns']}")
            per_set[s] = {**sc, "dir": os.path.relpath(run, root), "from": how, "matches_report": ok}
            detail[m].update({(s, *k): (v, flags[k][0]) for k, v in outcomes.items()})
        if not per_set:
            continue
        rows.append({"model": m, "method": meta.get("method"), "data": meta.get("data"), "train": files,
                     "host": meta.get("host") or "cassowary", "sets": per_set,
                     "mean": {v: mean(per_set[s][v]["pass"] for s in SETS) if len(per_set) == len(SETS) else None
                              for v in VIEWS}})
    rk = {v: ranks(rows, v) for v in ("all", "clean", "strict_clean")}
    for r in rows:
        r["rank"] = {v: rk[v].get(r["model"]) for v in rk}

    # how much of each set a training file's templates cover
    by_train = {}
    for files in sorted({tuple(r["train"]) for r in rows if r["train"]}):
        seen = seen_for(list(files))
        by_train[" + ".join(os.path.basename(f) for f in files)] = {
            s: {"turns": len(t), "contaminated": sum(1 for x in t.values() if x in seen),
                "strict": sum(1 for f in contamination(t, seen).values() if f[1]),
                "conversations_touched": len({i for (i, _), x in t.items() if x in seen}),
                "conversations": len({i for i, _ in t})} for s, t in turns.items()}
    return {"sets": {s: p for s, (_, p) in SETS.items()}, "root": root, "registry": registry,
            "champions": CHAMPIONS, "rows": rows, "paired": paired(rows, detail), "contamination_by_train": by_train,
            "warnings": warnings}


def paired(rows, detail) -> list[dict]:
    """Each top-5 (all, clean or strict) model and champion against the all-turns leader, turn by turn: how many turns
    one passes where the other does not, on all turns and on the turns clean for both. z = (a - b) / sqrt(a + b), a
    sign test on the discordant turns; |z| < 2 is within noise."""
    full = [r for r in rows if r["rank"]["all"]]
    if not full:
        return []
    lead = min(full, key=lambda r: r["rank"]["all"])["model"]
    pick = [r["model"] for r in sorted(full, key=lambda r: r["rank"]["all"]) if r["model"] != lead
            and (min(v for v in r["rank"].values()) <= 5 or r["model"] in CHAMPIONS)]
    out = []
    for m in pick:
        a, b = detail[m], detail[lead]
        rec = {"model": m, "vs": lead}
        for view, keep in (("all", lambda k: True), ("clean", lambda k: not a[k][1] and not b[k][1])):
            ks = [k for k in a if k in b and keep(k)]
            win = sum(1 for k in ks if a[k][0] > b[k][0])
            loss = sum(1 for k in ks if a[k][0] < b[k][0])
            rec[view] = {"turns": len(ks), "better": win, "worse": loss,
                         "z": round((win - loss) / (win + loss) ** 0.5, 2) if win + loss else 0.0}
        out.append(rec)
    return out


def summary(out: dict, rows: list[dict]) -> list[str]:
    """Bullet points: contamination share, leaders, rank agreement and the size of the clean-turn shift."""
    lines = []
    for s in out["sets"]:
        shares = [x[s]["contaminated"] / x[s]["turns"] for x in out["contamination_by_train"].values()]
        strict = [x[s]["strict"] / x[s]["turns"] for x in out["contamination_by_train"].values()]
        if shares:
            lines.append(f"- {s}: {min(shares):.0%}-{max(shares):.0%} of turns contaminated depending on the training set "
                         f"({min(strict):.0%}-{max(strict):.0%} strict)")
    if not rows:
        return lines
    lead = {v: min(rows, key=lambda r: r["rank"][v])["model"] for v in ("all", "clean", "strict_clean")}
    lines.append(f"- leader (mean of the three sets): all turns {lead['all']}, clean {lead['clean']}, strict {lead['strict_clean']}")
    n = len(rows)
    for v in ("clean", "strict_clean"):
        rho = 1 - 6 * sum((r["rank"]["all"] - r["rank"][v]) ** 2 for r in rows) / (n * (n * n - 1)) if n > 1 else 1.0
        top = lambda view: {r["model"] for r in rows if r["rank"][view] <= 10}  # noqa: E731
        lines.append(f"- {v.replace('_', '-')} vs all: Spearman {rho:.2f} over {n} models; top 10 in: "
                     f"{', '.join(sorted(top(v) - top('all'))) or 'none'}, out: {', '.join(sorted(top('all') - top(v))) or 'none'}")
    shift = sorted((r["mean"]["clean"] - r["mean"]["all"]) * 100 for r in rows if r["rank"]["all"] <= 10)
    lines.append(f"- clean - all, top 10 on all turns: {shift[0]:+.1f} to {shift[-1]:+.1f} pts (median {shift[len(shift) // 2]:+.1f})")
    return lines


def markdown(out: dict) -> str:
    f = lambda v: "-" if v is None else f"{v:.1%}"  # noqa: E731
    d = lambda a, b: "-" if a is None or b is None else f"{(a - b) * 100:+.1f}"  # noqa: E731
    rows = [r for r in out["rows"] if r["mean"]["all"] is not None]
    rows.sort(key=lambda r: r["rank"]["all"])
    champs = set(out["champions"])
    star = lambda m: f"**{m}**" if m in champs else m  # noqa: E731
    mv = lambda a, b: "=" if a == b else (f"up {a - b}" if a > b else f"down {b - a}")  # noqa: E731
    lines = ["# Outcome evals on uncontaminated turns\n",
             "Outcome sets test-v2-e2e, wild-e2e and wild2-e2e, rescored per model on the turns whose normalised user "
             "text is *not* a user turn of that model's own training data (homeai.datagen.leakage normaliser). "
             "**strict**: a turn also counts as contaminated when any earlier turn of its conversation is. "
             "Test sets unchanged, no model re-run; built by `python3 -m homeai.eval.clean_rescore` from the "
             f"read-only copy in `{out['root']}`. Champions in bold.\n",
             "## Summary\n", *summary(out, rows), "",
             "## Ranking: all turns vs clean turns (mean of the three sets)\n",
             "| rank all | model | data | all | clean | strict-clean | clean - all (pts) | rank clean | move | rank strict | move |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m, rk = r["mean"], r["rank"]
        lines.append(f"| {rk['all']} | {star(r['model'])} | {r['data']} | {f(m['all'])} | {f(m['clean'])} | "
                     f"{f(m['strict_clean'])} | {d(m['clean'], m['all'])} | {rk['clean']} | {mv(rk['all'], rk['clean'])} | "
                     f"{rk['strict_clean']} | {mv(rk['all'], rk['strict_clean'])} |")
    if out["paired"]:
        lines += [f"\n## Paired check against the all-turns leader ({out['paired'][0]['vs']})\n",
                  "Turns the model passes where the leader fails (better) and the reverse (worse); clean = turns clean for "
                  "both. z = (better - worse) / sqrt(better + worse): |z| < 2 is within noise.\n",
                  "| model | all: better / worse | z | clean: better / worse | z |", "|---|---|---|---|---|"]
        for p in out["paired"]:
            a, c = p["all"], p["clean"]
            lines.append(f"| {star(p['model'])} | {a['better']} / {a['worse']} | {a['z']:+.2f} | {c['better']} / {c['worse']} "
                         f"| {c['z']:+.2f} |")
    lines += ["\n## Per set: pass on all / clean / contaminated / strict-clean turns (turn counts in brackets)\n",
              "| model | set | all | clean | contaminated | strict-clean | clean - all (pts) | source |", "|---|---|---|---|---|---|---|---|"]
    for r in rows + [r for r in out["rows"] if r["mean"]["all"] is None]:
        for s, x in r["sets"].items():
            c = lambda v: f"{f(x[v]['pass'])} ({x[v]['n']})"  # noqa: E731
            lines.append(f"| {star(r['model'])} | {s} | {c('all')} | {c('clean')} | {c('contaminated')} | {c('strict_clean')} | "
                         f"{d(x['clean']['pass'], x['all']['pass'])} | {x['from']}{'' if x['matches_report'] else ' (differs from report!)'} |")
    lines += ["\n## Contaminated test turns per training set\n",
              "Turns whose user text is a training user turn; strict adds every later turn of those conversations.\n",
              "| training data | " + " | ".join(f"{s} contaminated | {s} strict" for s in out["sets"]) + " |",
              "|---|" + "---|---|" * len(out["sets"])]
    for t, sets in out["contamination_by_train"].items():
        cells = []
        for s in out["sets"]:
            x = sets[s]
            cells += [f"{x['contaminated']}/{x['turns']} ({x['contaminated'] / x['turns']:.1%})",
                      f"{x['strict']}/{x['turns']} ({x['strict'] / x['turns']:.1%})"]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    if out["warnings"]:
        lines += ["\n## Warnings\n"] + [f"- {w}" for w in out["warnings"]]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/eval-cassowary")
    ap.add_argument("--registry", help="default: <root>/configs/models.json")
    ap.add_argument("--cache", help="normalised training turns; default: <root>/train-norm")
    ap.add_argument("--out", default="outputs/CLEAN_RESCORE")
    a = ap.parse_args()
    out = build(a.root, a.registry or os.path.join(a.root, "configs/models.json"),
                a.cache or os.path.join(a.root, "train-norm"))
    json.dump(out, open(a.out + ".json", "w"), indent=2)
    md = markdown(out)
    open(a.out + ".md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
