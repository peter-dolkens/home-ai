"""How much accuracy each model loses to speech-to-text: outcome evals on the heard sets against the written baseline.

    python3 -m homeai.eval.heard_summary            # -> outputs/HEARD.json, outputs/HEARD.md

Reads every outputs/eval/*/report.json whose data is a heard set (data/generated/heard/<set>-<engine>-<condition>.jsonl)
or a written set run as a heard baseline (<model>-<set>-heard-written[-s30]), with its turns.jsonl. Comparisons are
paired: a heard run is only compared with the written run of the same model, served by the same host, on the turns
both ran (a 30-row sample and a full run pair on the sample's turns), so a slow host running samples still gives a
fair drop. Models are reported as "<model> @ <host>".

Per model, set, engine and condition: pass rate (e2e's, half credit included), the written pass rate on the same turns,
the drop, turns lost (passed written, failed heard) and regained, and how often the transcript had changed a house term
(room, device, person, number, command; roundtrip.py's per-turn "lost") on the lost turns versus the rest. For the
confusion suite: pass rate by kind and by recoverable versus not, next to the written source turn's pass where the
source was run. Seconds per turn are listed apart with the host that served the model, never in the headline.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

EVAL = "outputs/eval"
SETS = {"test-v2": "testv2", "wild": "wild", "wild2": "wild2", "bigbench-bundook": "bigbench-bundook"}
HEARD = re.compile(r"data/generated/heard/(?P<set>test-v2|wild2|wild|bigbench-bundook)-(?P<cond>.+)\.jsonl$")
WRITTEN = re.compile(r"data/generated/(?P<set>test-v2|wild2|wild|bigbench-bundook)\.jsonl$")


def load_json(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def turns(d: str) -> dict:
    out = {}
    try:
        for l in open(os.path.join(d, "turns.jsonl")):
            t = json.loads(l)
            out[(t["id"], t["turn"])] = float(t["ok"])
    except OSError:
        pass
    return out


_ROWS: dict = {}


def rows(path: str) -> dict:
    if path not in _ROWS:
        _ROWS[path] = {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}
    return _ROWS[path]


def term_changed(path: str) -> dict:
    """(id, turn) -> True when roundtrip.py found a house term of the written turn missing from the transcript."""
    return {(i, t["turn"]): bool(t.get("lost")) for i, r in rows(path).items() for t in (r.get("_voice") or {}).get("turns", [])}


def discover():
    """{model: {"written": {set: [(dir, report)]}, "heard": {(set, cond): [(dir, report)]}, "confusions": [...]}}"""
    found = collections.defaultdict(lambda: {"written": collections.defaultdict(list), "heard": collections.defaultdict(list),
                                             "confusions": []})
    for rp in sorted(glob.glob(f"{EVAL}/*/report.json")):
        d = os.path.dirname(rp)
        rep = load_json(rp)
        if not rep or rep.get("mode") != "e2e":
            continue
        data = rep.get("data", "")
        m = f"{rep['model']} @ {rep.get('host', '?')}"  # pair heard with written only on the same serving host
        if "/heard/confusions" in data:
            found[m]["confusions"].append((d, rep))
        elif HEARD.search(data):
            g = HEARD.search(data)
            found[m]["heard"][(g["set"], g["cond"])].append((d, rep))
        elif WRITTEN.search(data) and "-heard-written" in os.path.basename(d):
            found[m]["written"][WRITTEN.search(data)["set"]].append((d, rep))
    return found


def merged(runs) -> tuple[dict, list]:
    """Turn outcomes over several runs of one cell (a sample and a full run agree on shared turns; full wins)."""
    out, dirs = {}, []
    for d, rep in sorted(runs, key=lambda dr: dr[1].get("turns", 0)):
        out.update(turns(d))
        dirs.append(d)
    return out, dirs


def cell(heard: dict, written: dict, changed: dict) -> dict:
    keys = sorted(set(heard) & set(written))
    n = len(keys)
    if not n:
        return {"turns": 0}
    lost = [k for k in keys if written[k] == 1 and heard[k] < 1]
    regained = [k for k in keys if written[k] < 1 and heard[k] == 1]
    rest = [k for k in keys if k not in set(lost)]
    ch = lambda ks: sum(changed.get(k, False) for k in ks) / max(1, len(ks))  # noqa: E731
    fail_ch = [k for k in keys if changed.get(k)]
    fail_un = [k for k in keys if not changed.get(k)]
    return {
        "turns": n, "pass": round(sum(heard[k] for k in keys) / n, 4), "written_pass": round(sum(written[k] for k in keys) / n, 4),
        "drop": round(sum(written[k] - heard[k] for k in keys) / n, 4), "lost": len(lost), "regained": len(regained),
        "term_changed": round(ch(keys), 4),
        "lost_term_changed": round(ch(lost), 4) if lost else None, "rest_term_changed": round(ch(rest), 4),
        "pass_when_term_changed": round(sum(heard[k] for k in fail_ch) / len(fail_ch), 4) if fail_ch else None,
        "pass_when_term_kept": round(sum(heard[k] for k in fail_un) / len(fail_un), 4) if fail_un else None,
    }


def confusion_cell(runs, written_by_id: dict) -> dict:
    heard, dirs = merged(runs)
    data = runs[0][1]["data"]
    meta = {i: r.get("_confusion", {}) for i, r in rows(data).items()}
    groups = collections.defaultdict(list)
    for (i, _), ok in heard.items():
        c = meta.get(i, {})
        src_ok = written_by_id.get(c.get("source")) if "moved" not in c.get("origin", "") else None
        for g in (f"kind:{c.get('kind')}", f"recoverable:{c.get('recoverable')}", "all"):
            groups[g].append((ok, src_ok))
    out = {}
    for g, v in sorted(groups.items()):
        paired = [(a, b) for a, b in v if b is not None]
        out[g] = {"n": len(v), "pass": round(sum(a for a, _ in v) / len(v), 4),
                  "paired_n": len(paired),
                  "paired_pass": round(sum(a for a, _ in paired) / len(paired), 4) if paired else None,
                  "source_written_pass": round(sum(b for _, b in paired) / len(paired), 4) if paired else None}
    return {"dirs": dirs, "data": data, "groups": out}


def speed(rep: dict) -> dict:
    return {"seconds_per_turn": round(rep.get("seconds_per_turn", 0) * rep.get("calls_per_request", 1), 2),
            "seconds_per_call": round(rep.get("seconds_per_turn", 0), 2), "host": rep.get("host", "?"),
            "hardware": rep.get("hardware", "?"), "ollama": rep.get("ollama")}


def build() -> dict:
    res = {}
    for model, f in sorted(discover().items()):
        wr = {s: merged(r) for s, r in f["written"].items()}
        cells, speeds = {}, []
        for (s, cond), runs in sorted(f["heard"].items()):
            if s not in wr:
                continue
            h, dirs = merged(runs)
            c = cell(h, wr[s][0], term_changed(runs[0][1]["data"]))
            c["dirs"] = dirs
            cells.setdefault(cond, {})[SETS[s]] = c
            speeds += [{"run": os.path.basename(d), **speed(rep)} for d, rep in runs]
        for cond, by_set in cells.items():  # pooled over the sets present
            keys = [c for c in by_set.values() if c.get("turns")]
            n = sum(c["turns"] for c in keys)
            if n:
                by_set["all"] = {"turns": n, **{k: round(sum(c[k] * c["turns"] for c in keys) / n, 4)
                                               for k in ("pass", "written_pass", "drop", "term_changed")},
                                 "lost": sum(c["lost"] for c in keys), "regained": sum(c["regained"] for c in keys),
                                 "sets": sorted(s for s, c in by_set.items() if c.get("turns"))}
        written_by_id = {}
        for s, (t, _) in wr.items():
            for (i, k), ok in t.items():
                if k == 0:
                    written_by_id[i] = ok
        speeds += [{"run": os.path.basename(d), **speed(rep)} for runs in f["written"].values() for d, rep in runs]
        res[model] = {
            "written": {SETS[s]: {"turns": len(t), "pass": round(sum(t.values()) / max(1, len(t)), 4), "dirs": d}
                        for s, (t, d) in wr.items()},
            "heard": cells,
            "confusions": confusion_cell(f["confusions"], written_by_id) if f["confusions"] else None,
            "speed": speeds,
        }
    return res


def tts_caveat(heard_dir: str = "data/generated/heard", sets=("test-v2", "wild", "wild2", "bigbench-bundook")) -> dict:
    """Piper vs Kokoro (--tts kokoro) on the same turns, clean audio: WER and meaning changed per engine. Piper's espeak
    front end mispronounces some house words, so Piper-only errors are synthesis artefacts, not recogniser errors."""
    out = {}
    for engine in ("parakeet", "small"):
        agg = {}
        for tts, sfx in (("piper", ""), ("kokoro", "-kokoro")):
            reps = [load_json(f"{heard_dir}/{s}-{engine}-clean{sfx}.wer.json") for s in sets]
            reps = [r for r in reps if r]
            if len(reps) != len(sets):
                break
            n, w = sum(r["utterances"] for r in reps), sum(r["words"] for r in reps)
            agg[tts] = {"turns": n, "wer": round(sum(r["errors"] for r in reps) / w, 4),
                        "vocab_wer": round(sum(r["vocab_errors"] for r in reps) / sum(r["vocab_words"] for r in reps), 4),
                        "meaning_changed": round(sum(r["meaning_changed_n"] for r in reps) / n, 4)}
        if len(agg) == 2:
            out[engine] = agg
    return out


def md(res: dict, caveat: dict | None = None) -> str:
    L = ["# Heard-set outcome evals", ""]
    if caveat:
        L += ["**Caveat on every Piper-voiced number below: much of the clean-audio damage is Piper's pronunciation, not "
              "the recogniser.** The same turns voiced by Kokoro-82M (misaki lexicon G2P) through the same engines, clean "
              "audio, all four sets (984 turns):", "",
              "| engine | TTS | WER | house-vocab WER | meaning changed |", "|---|---|---|---|---|"]
        for e, v in caveat.items():
            for tts, r in v.items():
                L.append(f"| {e} | {tts} | {r['wer'] * 100:.1f}% | {r['vocab_wer'] * 100:.1f}% | {r['meaning_changed'] * 100:.1f}% |")
        L += ["", "Of the 39 turns every CPU engine lost with Piper, 32 keep their house terms with Kokoro; what remains is "
              "mostly \"ensuite\" heard as \"en suit\" / \"end suit\". Read clean-audio heard drops as an upper bound for "
              "real speech; the noise and far-field penalties on top of them are real.", ""]
    L += ["Pass rate on what speech-to-text handed the model (homeai/voice/roundtrip.py), scored against the written "
         "reference, next to the same model's written pass rate on the same turns. Cells: **heard pass / drop vs "
         "written** (turns). Lost = passed written, failed heard; 'term changed' = the transcript lost a room, device, "
         "person, number or command word.", ""]
    for model, r in res.items():
        L += [f"## {model}", "", "| condition | " + " | ".join(["testv2", "wild", "wild2", "all"]) +
              " | lost / regained | term changed: lost turns vs rest |", "|---" * 7 + "|"]
        wrow = [f"{r['written'][s]['pass'] * 100:.1f} ({r['written'][s]['turns']})" if s in r["written"] else "-"
                for s in ("testv2", "wild", "wild2")]
        L.append("| written | " + " | ".join(wrow) + " | | | |")
        for cond, by in sorted(r["heard"].items(), key=lambda kv: -kv[1].get("all", {}).get("pass", 0)):
            cells = []
            for s in ("testv2", "wild", "wild2", "all"):
                c = by.get(s)
                cells.append(f"{c['pass'] * 100:.1f} / {c['drop'] * -100:+.1f} ({c['turns']})" if c and c.get("turns") else "-")
            a = by.get("all", {})
            sub = [c for s, c in by.items() if s != "all" and c.get("turns")]
            lost = sum(c["lost"] for c in sub)
            lt = sum((c["lost_term_changed"] or 0) * c["lost"] for c in sub) / max(1, lost)
            rest_n = sum(c["turns"] - c["lost"] for c in sub)
            rt = sum(c["rest_term_changed"] * (c["turns"] - c["lost"]) for c in sub) / max(1, rest_n)
            L.append(f"| {cond} | " + " | ".join(cells) + f" | {a.get('lost', '-')} / {a.get('regained', '-')} | "
                     f"{lt * 100:.0f}% vs {rt * 100:.0f}% |")
        conf = r.get("confusions")
        if conf:
            L += ["", f"Confusion suite ({os.path.basename(conf['data'])}): pass on the misheard turn; 'source' = the same "
                  "request written, where it was run (moved-room rows have no written twin).", "",
                  "| group | n | pass | paired n | paired pass | source written pass |", "|---" * 6 + "|"]
            for g, v in conf["groups"].items():
                f = lambda x: "-" if x is None else f"{x * 100:.1f}"  # noqa: E731
                L.append(f"| {g} | {v['n']} | {f(v['pass'])} | {v['paired_n']} | {f(v['paired_pass'])} | {f(v['source_written_pass'])} |")
        L.append("")
    L += ["## Speed (not comparable across hosts; accuracy runs, not a latency benchmark)", "",
          "| run | host | hardware | s/turn | s/call |", "|---|---|---|---|---|"]
    for model, r in res.items():
        for s in r["speed"]:
            L.append(f"| {s['run']} | {s['host']} | {s['hardware']} | {s['seconds_per_turn']} | {s['seconds_per_call']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/HEARD")
    a = ap.parse_args()
    res = build()
    caveat = tts_caveat()
    json.dump({"tts_caveat": caveat, "models": res}, open(a.out + ".json", "w"), indent=2)
    open(a.out + ".md", "w").write(md(res, caveat))
    print(md(res, caveat))


if __name__ == "__main__":
    main()
