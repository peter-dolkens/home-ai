"""Results page, champions first (Peter, 28 Sep): a champion per design, then one uniform leaderboard per design, the
context-window sweep, condensed findings and downloads. Replaces the long per-set page (homeai.eval.report_html).

    python3 -m homeai.eval.report_v2 --out page.html [--findings findings.json] [--artifacts artifacts.json]

Reads outputs/CHAMPIONS.json, outputs/DESIGN_CHAMPIONS.json, outputs/eval/*/report.json, outputs/leaf-results/*.
Speed chips follow Peter's targets: most requests under 2 s, over 5 s is disqualifying.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import html
import json
import os

EVAL = "outputs/eval"


def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def rep(name):
    return load(f"{EVAL}/{name}/report.json")


def esc(s):
    return html.escape(str(s))


def pct(v, bold=False):
    if v is None:
        return '<td class="n dim">–</td>'
    return f'<td class="n{" best" if bold else ""}">{v * 100:.1f}</td>'


def secs(v):
    return '<td class="n dim">–</td>' if v is None else f'<td class="n">{v:.2f}</td>'


def chip(req_s):
    if req_s is None:
        return '<span class="chip">not timed</span>'
    cls = "ok" if req_s < 2 else ("warn" if req_s <= 5 else "bad")
    return f'<span class="chip {cls}">{req_s:.1f} s / request</span>'


def per_request(r):
    """Outcome-eval report -> seconds per request (seconds per model call x calls per request)."""
    if not r:
        return None
    lat = (r.get("latency") or {}).get("turn_s")
    if lat:
        return lat.get("mean")
    if r.get("seconds_per_turn") and r.get("calls_per_request"):
        return r["seconds_per_turn"] * r["calls_per_request"]
    return None


# ---------------------------------------------------------------------------------------------------- sections
def champions_section(ch, dc):
    rows_by_model = {r["model"]: r for r in ch.get("rows", [])}
    a = dc.get("A", {})
    best = rows_by_model.get(ch.get("best") or "", {})
    lh = a.get("large_house") or {}
    b = (dc.get("B") or {}).get("best") or {}
    t = (dc.get("tiered") or {}).get("best") or {}
    d = (dc.get("tiered") or {}).get("best_fanout") or {}
    lf = (dc.get("leaf") or {}).get("best") or {}
    e = (dc.get("E") or {}).get("best") or {}
    b_req = mean_req([f"{b.get('model')}-{s}-e2e" for s in ("bigbench-b", "testv2-b", "wild-b", "wild2-b")]) if b else None
    lh_req = per_request(rep(f"{lh.get('model')}-bigbench32k-e2e")) if lh else None
    rows = [
        ("A · single model", "everyday houses", ch.get("best"), f"{fmt(best.get('quality'))} quality · {fmt(best.get('outcome'))} outcome",
         "mean of test v2, wild, wild 2 and their outcome evals", lat_now(ch.get("best"), best.get("s_per_request"))),
        ("A · single model", "large houses, 32k window", lh.get("model"), f"{fmt(lh.get('pass'))} large-house outcome",
         "200 conversations, 3k–20k entities + the Bundook plan", lh_req),
        ("B · lookup tools", "rooms-only prompt", b.get("model"), f"{fmt(b.get('score'))} outcome",
         "mean of the B versions of large-house, test v2, wild, wild 2", b_req),
        ("C · tiered, routed", "orchestrator sends each room its instruction", t.get("config"), f"{fmt(t.get('score'))} outcome",
         "mean over " + ", ".join((t.get("scores") or {}).keys()), t.get("turn_mean_s")),
        ("D · tiered, fan-out", "every room leaf gets the request", d.get("config") or "dropped",
         f"{fmt(d.get('score'))} outcome" if d else "47% on large houses at ~11 s (28 Sep)",
         "mean over " + ", ".join((d.get("scores") or {}).keys()) if d else "blind fan-out lost on accuracy and speed; not re-timed", d.get("turn_mean_s")),
        ("Room leaf", "component of C and D", lf.get("leaf"), f"{fmt(lf.get('end_state'))} end state",
         "1,040 held-out Bundook room instructions", (lf.get("latency_ms_p50") or 0) / 1000 or None),
        ("A · voice", "transcribed speech (parakeet), everyday test sets", "homeai-4b-lora-v9",
         f"{fmt(voice_scores('homeai-4b-lora-v9', ('testv2', 'wild', 'wild2'))['parakeet-clean'])} heard · "
         f"{fmt(voice_scores('homeai-4b-lora-v9', ('testv2', 'wild', 'wild2'))['confusions'])} misheard-word suite",
         "v9: v8 plus speech-to-text round trips of its own training turns; ties v5 and v8 on typed text",
         lat_now("homeai-4b-lora-v9", per_request(rep("homeai-4b-lora-v9-wild-e2e")))),
        ("E · fast + slow", "reconciled", (e.get("pair") or "being measured").replace("homeai-", "").replace("__", " → "),
         f"{fmt(e.get('reconciled_pass'))} final · {fmt(e.get('flicker_rate'))} flicker" if e else "–",
         "fast model acts, slow model runs independently, reconciler fixes differences", e.get("time_to_first_action_s_acting")),
    ]
    # the 3090's own house champion (Peter, 14 Oct: the two tiers need not share a model): best house heard mean among models
    # timed on the 3090 within the speed target, whatever the family
    timed = {g["model"]: mean_req([f"{g['model']}-{t}-e2e" for t in B_TAGS]) for g in house_voice_groups()}
    gpu = [g for g in house_voice_groups() if timed.get(g["model"]) and timed[g["model"]] <= GPU_TARGET_S]
    if gpu:
        h = max(gpu, key=lambda g: g["heard"])
        rows.insert(0, ("The house, spoken", f"best heard mean among models under {GPU_TARGET_S:.0f} s / request on the 3090",
                        h["model"].replace("homeai-", ""), f"{fmt(h['heard'])} heard · {fmt(h['typed'])} typed",
                        f"{h['seeds']} seed{'s' if h['seeds'] > 1 else ''}; seed alone moves this by 0.2-1.7 points (see the house table)",
                        timed[h["model"]]))
    out = ['<div class="scroll"><table class="champ"><thead><tr><th>Design</th><th>Champion</th><th>Score</th><th>Speed</th></tr></thead><tbody>']
    for design, sub, model, score, basis, req in rows:
        out.append(f'<tr><td><b>{esc(design)}</b><span class="sub">{esc(sub)}</span></td><td class="mono">{esc(model or "–")}</td>'
                   f'<td><b>{esc(score)}</b><span class="sub">{esc(basis)}</span></td><td>{chip(req)}</td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out)


MAC_TARGET_S = 1.0   # warm reply on the Mac within which a model counts as fast enough (Peter: best and best <= 1 s)
GPU_TARGET_S = 2.0   # per request on the 3090 (Peter's target: most requests under 2 s)


def mac_rows():
    """Every model served on falcon (Mac mini M2 Pro): design B rows score the B sets on the Mac and time warm house
    requests (prefix bench); design A rows come from FALCON_CHAMPIONS (outcome evals on the Mac, warm or cold s / request)."""
    out = []
    for p in glob.glob("outputs/eval-falcon/*@falcon-*-testv2-b-e2e/report.json"):
        row = p.split("/")[-2][:-len("-testv2-b-e2e")]
        b = [rep(f"../eval-falcon/{row}-{t}-e2e") for t in B_TAGS]
        h = (load(f"outputs/eval-falcon/{row}-prefix/prefix.json") or {}).get("modes", {}).get("house", {})
        if all(b) and h.get("warm_mean"):
            out.append({"row": row, "design": "B", "acc": sum(r["pass"] for r in b) / 4, "s": h["warm_mean"], "cold": h.get("cold_mean")})
    for r in (load("outputs/FALCON_CHAMPIONS.json") or {}).get("rows", []):
        s = r.get("warm_s_per_request") or r.get("s_per_request")
        if r.get("accuracy") and s and r.get("ok_host", True):
            out.append({"row": r["row"], "design": "A", "acc": r["accuracy"], "s": s, "cold": r.get("s_per_request")})
    return out


def mac_champions_section():
    rows, win = [], 0.02
    for d, label, basis in (("B", "B · lookup tools", "mean of the B versions of large-house, test v2, wild, wild 2, run on the Mac"),
                            ("A", "A · device list in the prompt", "mean of the test v2, wild, wild 2 outcome evals, run on the Mac")):
        c = [r for r in mac_rows() if r["design"] == d]
        if not c:
            continue
        best = max(c, key=lambda r: r["acc"])
        fast = [r for r in c if r["s"] <= MAC_TARGET_S]
        best_fast = max(fast, key=lambda r: r["acc"]) if fast else None
        quickest = min((r for r in c if r["acc"] >= best["acc"] - win), key=lambda r: r["s"])
        fastest = min(c, key=lambda r: r["s"])
        wins = {}
        for what, r in (("most accurate", best), (f"best within {MAC_TARGET_S:.0f} s warm", best_fast),
                        ("fastest within 2 points of the best", quickest), ("fastest", fastest)):
            if r:
                wins.setdefault(r["row"], (r, []))[1].append(what)
        for row, (r, whats) in wins.items():   # one row per model, listing every title it holds
            tag = " (the pick)" if row.startswith(PICK + "@") and d == "B" else ""
            rows.append((label, " · ".join(whats) + tag, row.replace("homeai-", ""), f"{fmt(r['acc'])}", basis, r["s"]))
    pick = next((g for g in house_voice_groups() if g["model"] == PICK), None)
    if pick:
        rows.insert(0, ("The pick for the house", "B · lookup tools, spoken requests, served on the Mac mini", PICK.replace("homeai-", ""),
                        f"{fmt(pick['heard'])} heard · {fmt(pick['typed'])} typed",
                        f"Bundook house, 8 speech-to-text conditions, mean of {pick['seeds']} seeds; ~1.2 s from end of speech to action with medium Whisper",
                        next((r["s"] for r in mac_rows() if r["row"] == PICK + "@falcon-mlx6"), None)))
    out = ['<div class="scroll"><table class="champ"><thead><tr><th>Design</th><th>Champion</th><th>Score</th><th>Warm reply</th></tr></thead><tbody>']
    for design, sub, model, score, basis, req in rows:
        out.append(f'<tr><td><b>{esc(design)}</b><span class="sub">{esc(sub)}</span></td><td class="mono">{esc(model)}</td>'
                   f'<td><b>{esc(score)}</b><span class="sub">{esc(basis)}</span></td><td>{chip(req)}</td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out) + ('<p class="note">Scored and timed on falcon (Mac mini M2 Pro, 32 GB): MLX at 6 or 8 bits (Ollama Metal for older A '
                             'rows), prompt cache on; warm = a request whose house prompt is cached. Never compared with the 3090 numbers.</p>')


def lat_now(model, fallback):
    """Seconds per request from the idle-3090 latency bench on the current Ollama (0.40.3) when re-run there."""
    r = load(f"{EVAL}/{model}-latency-o040/latency.json") if model else None
    return (r or {}).get("cold_mean") or fallback


def fmt(v):
    return "–" if v is None else f"{v * 100:.1f}%"


def mean_req(names):
    xs = [per_request(rep(f"{n}-o040") or rep(n)) for n in names]   # cassowary's current Ollama (0.40.3) when re-run there
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def table(head, rows, note=""):
    """head: names, or (name, "max"|"min") for columns whose best and worst values get coloured."""
    h = "".join(f'<th data-dir="{x[1]}">{esc(x[0])}</th>' if isinstance(x, tuple) else f"<th>{esc(x)}</th>" for x in head)
    body = "\n".join(rows)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>' + (f'<p class="note">{note}</p>' if note else "")


def board_a(ch):
    sets = ["test-v2", "wild", "wild2", "wild-e2e", "wild2-e2e", "test-v2-e2e", "rules"]
    tuned = [r for r in ch.get("rows", []) if r.get("config") and not r.get("track") and r.get("quality") is not None]
    tuned.sort(key=lambda r: -(r["quality"] or 0))
    top = {s: max((r["scores"].get(s) or 0) for r in tuned) for s in sets}
    top["q"] = max((r["quality"] or 0) for r in tuned) if tuned else 0
    top["o"] = max((r["outcome"] or 0) for r in tuned) if tuned else 0
    rows = []
    for r in tuned:
        big = (rep(f"{r['model']}-bigbench32k-e2e") or {}).get("pass")
        champ = ' <span class="pill">champion</span>' if r["model"] in (ch.get("best"), ch.get("best_fast"), ch.get("best_outcome")) else ""
        cells = "".join(pct(r["scores"].get(s), (r["scores"].get(s) or 0) == top[s] and top[s]) for s in sets)
        rows.append(f'<tr><td class="mono">{esc(r["model"])}{champ}<span class="sub">{esc(r.get("method", ""))} · {esc(r.get("data", ""))} data</span></td>'
                    f'{cells}{pct(big)}{pct(r["quality"], r["quality"] == top["q"])}{pct(r["outcome"], r["outcome"] == top["o"])}{secs(r.get("s_per_turn"))}</tr>')
    return table(["Model", *[(x, "max") for x in ("test v2", "wild", "wild 2", "wild outcome", "wild 2 outcome", "test v2 outcome", "rules",
                  "large houses 32k", "quality", "outcome")], ("s / turn", "min")], rows,
                 "Quality = mean of test v2, wild, wild 2, wild outcome and wild 2 outcome. Outcome = mean of the three outcome evals. "
                 "Rules = Peter's clarification rules (diagnostic). Large houses = the 200-conversation benchmark with a 32k window. Best per column in green, worst in red.")


def board_b(dc):
    rows = []
    for m, v in sorted(((dc.get("B") or {}).get("all") or {}).items(), key=lambda kv: -(kv[1]["score"] or 0)):
        s = v["scores"]
        req = mean_req([f"{m}-{x}-e2e" for x in ("bigbench-b", "testv2-b", "wild-b", "wild2-b")])
        rows.append(f'<tr><td class="mono">{esc(m)}</td>{pct(s.get("bigbench-b"))}{pct(s.get("testv2-b"))}{pct(s.get("wild-b"))}{pct(s.get("wild2-b"))}'
                    f'{pct(v["score"], True)}{secs(v.get("s_per_turn"))}<td>{chip(req)}</td></tr>')
    return table(["Model", *[(x, "max") for x in ("large houses", "test v2", "wild", "wild 2", "mean")], ("s / call", "min"), "speed"], rows,
                 "Outcome evals on the design-B versions of each set: same requests and houses, rooms-only prompt and lookup tools. "
                 "Trained without the agent-written conversations, which is most of its gap on everyday houses.")


def board_tiered(dc):
    rows = []
    for cfg, v in sorted(((dc.get("tiered") or {}).get("all") or {}).items(), key=lambda kv: -(kv[1]["score"] or 0)):
        s = v["scores"]
        rows.append(f'<tr><td class="mono">{esc(cfg)}</td>{pct(s.get("bigbench"))}{pct(s.get("testv2"))}{pct(s.get("wild"))}{pct(s.get("wild2"))}'
                    f'{pct(v["score"])}{secs(v.get("turn_p50_s"))}<td>{chip(v.get("turn_mean_s"))}</td></tr>')
    return table(["Pipeline", *[(x, "max") for x in ("large houses", "test v2", "wild", "wild 2", "mean")], ("p50 s / request", "min"), "mean"], rows,
                 "C, routed: a 4B or 2B orchestrator (design B's tools plus ask_rooms) sends one instruction per room to a Qwen3.5 2B leaf. "
                 "D, fan-out: every room leaf gets the request verbatim and the orchestrator composes the answer. "
                 "All timed on cassowary with orchestrator and leaves served by the same 3090 (earlier split-machine runs are excluded).")


def board_tiers(dc):
    rows = []
    order = ["27B", "8-9B", "8B (Gemma E4B, ~4B active)", "4B", "2B", "1-2B", "0.8B", "<1B"]  # largest first
    for t, v in sorted((dc.get("tiers") or {}).items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)):
        a, f = v["accuracy_champion"], v["speed_champion"]
        s = a["scores"]
        same = a["model"] == f["model"]
        rows.append(f'<tr><td><b>{esc(t)}</b></td><td class="mono">{esc(a["model"])}</td>{pct(s.get("test-v2-e2e"))}{pct(s.get("wild-e2e"))}'
                    f'{pct(s.get("wild2-e2e"))}{pct(s.get("rules"))}{pct(s.get("bigbench32k"))}{pct(a["accuracy"], True)}'
                    f'<td>{chip(a["s_per_request"])}</td><td class="mono">{esc(f["model"])}'
                    f'<span class="sub">{esc("%.2f s / request · %.1f%%" % (f["s_per_request"], f["accuracy"] * 100))}'
                    f'{" · also the accuracy champion" if same else ""}</span></td></tr>')
    return table(["Size tier", "Accuracy champion", *[(x, "max") for x in ("test v2 outcome", "wild outcome", "wild 2 outcome", "rules",
                  "large houses 32k", "accuracy")], "speed", "Speed champion"], rows,
                 "Accuracy = mean of the three outcome evals (every candidate has them); rules and large houses are shown but not averaged in, "
                 "so no model is ranked on a different set of tests. "
                 "Speed champion = the fastest per request within 2 points of the tier's accuracy champion. These are the pieces design E mixes.")


def board_e(dc, top=15):
    rows = []
    pairs = sorted(((dc.get("E") or {}).get("all") or {}).items(),  # most accurate first, then the quickest first action
                   key=lambda kv: (-(kv[1].get("reconciled_pass") or 0), kv[1].get("time_to_first_action_s_acting") or 99))
    for k, v in pairs[:top]:
        rows.append(f'<tr><td class="mono">{esc(v.get("fast", k.split("__")[0]))}</td><td class="mono">{esc(v.get("slow", k.split("__")[-1]))}</td>'
                    f'{pct(v.get("fast_pass"))}{pct(v.get("slow_pass"))}{pct(v.get("reconciled_pass"), True)}{pct(v.get("agreement_acting"))}'
                    f'{pct(v.get("flicker_rate_acting"))}{secs(v.get("time_to_first_action_s_acting"))}{secs(v.get("slow_alone_s_acting"))}</tr>')
    return table(["Fast model", "Slow model", ("fast alone", "max"), ("slow alone", "max"), ("reconciled", "max"), ("agree (acting turns)", "max"),
                  ("corrected (acting turns)", "min"), ("first action s", "min"), ("slow alone s", "min")], rows,
                 "Both models answer every turn of test v2, wild and wild 2 independently; the reconciler keeps the fast model's action when "
                 "the resulting states agree and otherwise reverts it and applies the slow model's. Corrected = a wrong action you would "
                 "briefly see. Timings are each model's own idle-3090 run on cassowary, per acting turn. "
                 f"The best {min(top, len(pairs))} of {len(pairs)} design-A pairings are shown. On the house with spoken requests the "
                 "design-B pairing (LFM acting, Spark checking) overruled 10–20% of actions on noisy speech, so the picks run alone.")


def board_leaf(dc):
    rows = []
    for leaf, v in sorted(((dc.get("leaf") or {}).get("all") or {}).items(), key=lambda kv: -(kv[1]["end_state"] or 0)):
        ms = v.get("latency_ms_p50")
        rows.append(f'<tr><td class="mono">{esc(leaf)}</td>{pct(v.get("end_state"))}{pct(v.get("exact"))}{pct(v.get("refusal"))}{pct(v.get("false_refusal"))}'
                    f'<td class="n">{"–" if ms is None else f"{ms:.0f}"}</td></tr>')
    return table(["Leaf model", ("end state", "max"), ("exact call", "max"), ("declines other rooms", "max"), ("wrongly declines own", "min"),
                  ("p50 ms", "min")], rows,
                 "One room's tools plus one instruction per request, 1,040 held-out Bundook instructions in unseen wording. "
                 "Needle 3 runs on the CPU; the Qwen leaves on lapwing's RTX 3080.")


def board_ctx():
    rows = []
    for tag, ctx in (("bigbench-e2e", 8), ("bigbench12k-e2e", 12), ("bigbench16k-e2e", 16), ("bigbench24k-e2e", 24),
                     ("bigbench32k-e2e", 32), ("bigbench48k-e2e", 48)):
        r = rep(f"homeai-4b-lora-v5-{tag}")
        ps_dir = f"{EVAL}/homeai-4b-lora-v5-{tag}/ollama_ps.txt"
        if ctx == 8:
            ps_dir = f"{EVAL}/homeai-4b-lora-v5-bigbench8kps-e2e/ollama_ps.txt"
        if ctx == 32:
            ps_dir = f"{EVAL}/homeai-4b-lora-v5-bigbench32kps-e2e/ollama_ps.txt"
        gb = "–"
        try:
            line = open(ps_dir).read().splitlines()[1].split()
            gb = f"{line[2]} {line[3]}"
        except (OSError, IndexError):
            pass
        rows.append(f'<tr><td class="n">{ctx}k</td>{pct(r and r["pass"])}{secs(r and r.get("seconds_per_turn"))}<td class="n">{esc(gb)}</td></tr>')
    return table(["Context window", ("large-house outcome", "max"), ("s / call", "min"), ("GPU memory", "min")], rows,
                 "homeai-4b-lora-v5 on the large-house benchmark. The everyday test sets never exceed ~4k tokens, so their scores do not "
                 "change with the window. Design B needs no more than 8k (81.2% at both 8k and 16k).")


VOICE_CONDS = [("written", "typed"), ("medium-biased", "whisper medium + names"), ("small-biased", "whisper small + names"),
               ("parakeet-clean", "parakeet"), ("tiny-clean", "whisper tiny"), ("parakeet-farfield", "parakeet, 3 m away"),
               ("parakeet-noisy10", "parakeet, TV / kitchen noise")]
VOICE_MODELS = [("homeai-4b-lora-v9", ("testv2", "wild", "wild2")), ("homeai-4b-lora-v8", ("testv2", "wild", "wild2")),
                ("homeai-4b-lora-v5", ("testv2", "wild", "wild2")), ("homeai-4b-lora-b", ("test-v2-b", "wild-b", "wild2-b")),
                ("homeai-2b-lora-v5", ("testv2", "wild", "wild2")), ("homeai-lfm25-1.2b-lora-v5-fix2", ("testv2", "wild", "wild2"))]


def voice_scores(model, sets):
    """Turn-weighted pass over the three sets per condition (outputs/eval/<model>-<set>-heard-<condition>), plus the
    confusion suite; None where a condition is incomplete."""
    out = {}
    for c, _ in VOICE_CONDS:
        rs = [rep(f"{model}-{s}-heard-{c}") for s in sets]
        out[c] = sum(r["pass"] * r["turns"] for r in rs) / sum(r["turns"] for r in rs) if all(rs) else None
    conf = rep(f"{model}-confusions-heard")
    out["confusions"] = conf and conf["pass"]
    return out


def voice_section(heard):
    rows, best = [], "homeai-4b-lora-v9"
    for m, sets in VOICE_MODELS:
        v = voice_scores(m, sets)
        if v["written"] is None:
            continue
        cells = "".join(pct(v[c]) for c, _ in VOICE_CONDS[1:])
        drops = [v["written"] - v[c] for c, _ in VOICE_CONDS[1:] if v[c] is not None]
        tag = ' <span class="pill">voice champion</span>' if m == best else ""
        design = "B" if m.endswith("-b") else "A"
        rows.append(f'<tr><td class="mono">{esc(m)}{tag}<span class="sub">design {design}</span></td>{pct(v["written"])}{cells}'
                    f'{pct(v["confusions"])}<td class="n">{"–" if not drops else f"−{min(drops) * 100:.1f} to −{max(drops) * 100:.1f}"}</td></tr>')
    caveat = esc((heard or {}).get("tts_caveat", ""))
    return table(["Model", ("typed", "max"), *[(lbl, "max") for _, lbl in VOICE_CONDS[1:]], ("misheard-word suite", "max"), "drop vs typed"], rows,
                 "The same test conversations spoken by synthetic voices (Piper) and transcribed by each speech-to-text engine, scored against "
                 "the original answers: did the model still do the right thing from what it heard? 927 turns per cell (design B: the same "
                 "requests in its own format). The misheard-word suite is 571 targeted confusions (lights/likes, lock/look, room and people "
                 "names, numbers). v9 was trained on round-tripped copies of its own training turns (no test utterance leaked). "
                 f"Accuracy only (run on lapwing); timings are not compared. {caveat}")


# The house (200-conversation Bundook set, design B), heard: the eight standard conditions, typed baseline included, whose
# mean decided the picks; medium Whisper at 3 m and in TV noise were added later for the medium-biased front end.
HOUSE_SET = "bigbench-b-bundook200"
HOUSE_CONDS = ["written", "medium-biased", "small-biased", "small-clean", "parakeet-clean", "tiny-clean", "parakeet-farfield", "parakeet-noisy10"]
HOUSE_EXTRA = ["medium-farfield-biased", "medium-noisy10-biased"]
PICK, ALT = "homeai-lfm25-1.2b-lora-b-v8r256e4", "homeai-spark-x25-4b-lora-b-v7-r32"


def house_voice_groups():
    """One entry per model with every standard condition on the house; a second-seed run (name + "s2") is averaged into its
    base model. Sorted by the eight-condition heard mean."""
    runs = {}
    for p in glob.glob(f"{EVAL}/*-{HOUSE_SET}-heard-written/report.json"):
        m = os.path.basename(os.path.dirname(p))[:-len(f"-{HOUSE_SET}-heard-written")]
        v = {c: (rep(f"{m}-{HOUSE_SET}-heard-{c}") or {}).get("pass") for c in HOUSE_CONDS + HOUSE_EXTRA}
        if all(v[c] is not None for c in HOUSE_CONDS):
            runs[m] = v
    groups = {}
    for m, v in runs.items():
        base = m[:-2] if m.endswith("s2") and m[:-2] in runs else m
        groups.setdefault(base, []).append(v)
    out = []
    for m, vs in groups.items():
        avg = {c: (sum(v[c] for v in vs if v[c] is not None) / n if (n := sum(v[c] is not None for v in vs)) else None)
               for c in HOUSE_CONDS + HOUSE_EXTRA}
        out.append({"model": m, "seeds": len(vs), "typed": avg["written"], "heard": sum(avg[c] for c in HOUSE_CONDS) / len(HOUSE_CONDS), **avg})
    return sorted(out, key=lambda g: -g["heard"])


def house_voice_section():
    rows = []
    for g in house_voice_groups():
        tag = ' <span class="pill">pick</span>' if g["model"] == PICK else (' <span class="pill">alternate</span>' if g["model"] == ALT else "")
        rows.append(f'<tr><td class="mono">{esc(g["model"].replace("homeai-", ""))}{tag}<span class="sub">{g["seeds"]} seed{"s" if g["seeds"] > 1 else ""}</span></td>'
                    f'{pct(g["typed"])}{pct(g["heard"], g["model"] == PICK)}{pct(g["medium-biased"])}{pct(g["medium-farfield-biased"])}'
                    f'{pct(g["medium-noisy10-biased"])}{pct(g["parakeet-clean"])}{pct(g["parakeet-farfield"])}{pct(g["parakeet-noisy10"])}{pct(g["tiny-clean"])}</tr>')
    return table(["Model", ("typed", "max"), ("heard, 8-condition mean", "max"), ("Whisper medium + names", "max"), ("medium, 3 m", "max"),
                  ("medium, TV noise", "max"), ("Parakeet", "max"), ("Parakeet, 3 m", "max"), ("Parakeet, TV noise", "max"), ("Whisper tiny", "max")], rows,
                 "The 200-conversation Bundook set (228 turns) in design B's format, spoken by synthetic voices and transcribed by each "
                 "engine. Heard = mean of the eight standard conditions (typed, Whisper medium / small with the house's names, small, "
                 "Parakeet, tiny, Parakeet at 3 m and in TV noise); this decided the picks. Where a second seed was trained the two are "
                 "averaged: seed alone moves the heard mean by 0.2–1.7 points, so gaps under ~1.5 points are ties. 46% of these turns "
                 "appear word for word in the Bundook-plan training rows; scored on the other 112 turns only, the pick gets 82.6% typed / "
                 "76.3% heard. Accuracy only; speeds are in the B table.")


B_TAGS = ("bigbench-b", "testv2-b", "wild-b", "wild2-b")
A_TAGS = ("testv2", "wild", "wild2")


def ollama_section():
    """13 Oct: the same models on cassowary's upgraded Ollama (0.40.3, flash attention, q8_0 KV cache; outputs/bench-o040.sh,
    <model>-<set>-e2e-o040) against their earlier runs (0.21.2; Spark on ollama-next 0.34.4). Earlier speeds only when that run
    was on cassowary too."""
    rows = []
    models = sorted({os.path.basename(p)[:-len("-testv2-b-e2e-o040")] for p in glob.glob(f"{EVAL}/*-testv2-b-e2e-o040")} |
                    {os.path.basename(p)[:-len("-testv2-e2e-o040")] for p in glob.glob(f"{EVAL}/*-testv2-e2e-o040")})
    for m in models:
        tags = B_TAGS if rep(f"{m}-testv2-b-e2e-o040") else A_TAGS
        new = [rep(f"{m}-{t}-e2e-o040") for t in tags]
        old = [rep(f"{m}-{t}-e2e") for t in tags]
        if not all(new):
            continue
        acc_new = sum(r["pass"] for r in new) / len(new)
        acc_old = sum(r["pass"] for r in old) / len(old) if all(old) else None
        spt_new = sum(r["seconds_per_turn"] for r in new) / len(new)
        same_host = all(old) and all((r.get("host") or "cassowary") == "cassowary" for r in old)
        spt_old = sum(r["seconds_per_turn"] for r in old) / len(old) if same_host else None
        rows.append((m, "B" if tags is B_TAGS else "A", acc_old, acc_new, spt_old, spt_new))
    rows.sort(key=lambda r: -r[3])
    body = [f'<tr><td class="mono">{esc(m.replace("homeai-", ""))}<span class="sub">design {d}</span></td>{pct(ao)}{pct(an)}'
            f'<td class="n">{"–" if ao is None else f"{(an - ao) * 100:+.1f}"}</td>{secs(so)}{secs(sn)}'
            f'<td class="n">{"–" if not so else f"{so / sn:.1f}×"}</td></tr>' for m, d, ao, an, so, sn in rows]
    return table(["Model", ("accuracy, 0.21.2", "max"), ("accuracy, 0.40.3", "max"), "change", ("s / turn before", "min"),
                  ("s / turn now", "min"), "speed-up"], body,
                 "Cassowary's main Ollama moved from 0.21.2 to 0.40.3 (flash attention, q8_0 KV cache) on 13 Oct; every model here was "
                 "re-run on it, one at a time on an idle 3090. Accuracy = mean of the design's outcome sets (B: large houses, test v2, wild, "
                 "wild 2; A: test v2, wild, wild 2). Spark's earlier runs were on ollama-next 0.34.4; earlier speeds are blank where that run "
                 "was on lapwing. Accuracy holds within noise for every current model; where the earlier run was also on cassowary, turns are 1.2-2.1× faster. The one casualty is the "
                 "retired LFM 350M v5 (12-15 points lower; the newly trained small LFMs are unaffected).")


MAC_RUNS = [("homeai-lfm25-1.2b-lora-b-v8r256e4", "pick"), ("homeai-lfm25-2.6b-lora-b-v8r128e4", ""),
            ("homeai-lfm25-350m-lora-b-v8r256e4", ""), ("homeai-lfm25-350m-lora-b-v8r256e4s2", "seed 2"),
            ("homeai-lfm25-350m-lora-b-v8r256e6", "6 epochs"), ("homeai-lfm25-230m-lora-b-v8r256e4", ""),
            ("homeai-lfm25-230m-lora-b-v8r256e6", "6 epochs")]


def mac_section():
    """13-14 Oct (Peter: focus on the small LFMs; models for the Mac mini): each LFM size on the pick's recipe, scored on the
    house (lapwing, heard) and served on falcon at 6-bit MLX (prefix-cached warm requests on the house, the design-B sets)."""
    rows = []
    for m, tag in MAC_RUNS:
        row = f"{m}@falcon-mlx6"
        b = [rep(f"../eval-falcon/{row}-{t}-e2e") for t in B_TAGS]
        h = (load(f"outputs/eval-falcon/{row}-prefix/prefix.json") or {}).get("modes", {}).get("house", {})
        v = {c: (rep(f"{m}-{HOUSE_SET}-heard-{c}") or {}).get("pass") for c in HOUSE_CONDS + HOUSE_EXTRA}
        heard = sum(v[c] for c in HOUSE_CONDS) / len(HOUSE_CONDS) if all(v[c] is not None for c in HOUSE_CONDS) else None
        size = m.split("lfm25-")[1].split("-")[0].upper()
        pill = ' <span class="pill">pick</span>' if tag == "pick" else (f'<span class="sub">{esc(tag)}</span>' if tag else "")
        rows.append(f'<tr><td class="mono">LFM2.5 {esc(size)}{pill}</td>{pct(v["written"])}{pct(heard)}{pct(v["medium-farfield-biased"])}'
                    f'{pct(v["medium-noisy10-biased"])}{pct(sum(r["pass"] for r in b) / 4 if all(b) else None)}'
                    f'{secs(h.get("warm_mean"))}{secs(h.get("warm_p50"))}{secs(h.get("cold_mean"))}</tr>')
    return table(["Model", ("house typed", "max"), ("house heard, 8-condition", "max"), ("medium Whisper, 3 m", "max"),
                  ("medium Whisper, TV noise", "max"), ("B sets on the Mac", "max"), ("warm s", "min"), ("warm p50 s", "min"), ("cold s", "min")], rows,
                 "Every size trained on the pick's recipe (design B, train-b-v8, LoRA r256, 4 epochs; the 2.6B r128). House scores from "
                 "lapwing (Ollama 0.21.2, as the pick's), one seed per row; seed alone moves the heard mean by 0.2-1.7 points. Timing on "
                 "falcon (Mac mini M2 Pro) at 6-bit MLX, prompt cache on: warm = a request whose house prompt is cached, cold = first "
                 "request. Add ~0.46 s for medium Whisper. 8-bit was no more accurate than 6-bit and slightly slower; 6 epochs did not help.")


def falcon_section(fc):
    if not fc:
        return '<p class="note">No falcon results yet.</p>'
    rows = []
    for r in sorted(fc.get("rows", []), key=lambda r: -(r.get("accuracy") or 0)):
        if not r.get("ok_host", True):
            continue
        s = r.get("scores") or {}
        tag = ""
        if r["row"] == fc.get("accuracy"):
            tag += ' <span class="pill">accuracy</span>'
        if r["row"] == fc.get("speed"):
            tag += ' <span class="pill">speed</span>'
        rows.append(f'<tr><td class="mono">{esc(r["row"])}{tag}<span class="sub">{esc(r.get("note", ""))}</span></td>'
                    f'{pct(s.get("testv2"))}{pct(s.get("wild"))}{pct(s.get("wild2"))}{pct(r.get("accuracy"))}'
                    f'{secs(r.get("s_per_turn"))}{secs(r.get("s_per_request"))}{secs(r.get("warm_s_per_request"))}</tr>')
    return table(["Model on falcon", *[(x, "max") for x in ("test v2 outcome", "wild outcome", "wild 2 outcome", "accuracy")],
                  ("s / turn", "min"), ("s / request", "min"), ("warm s / request", "min")], rows,
                 "A separate low-power stream (Peter, 29 Sep): the same models served by a Mac mini (M2 Pro, 32 GB, Metal) through "
                 "Ollama 0.21.2 with cassowary's settings. <b>These timings compare only with each other, never with the 3090 numbers "
                 "above.</b> Accuracy matches cassowary's (same weights). On the Mac the time goes on reading the prompt; Ollama cannot "
                 "reuse a prompt prefix for Qwen3.5, and the volatile lines (time, speaker, presence) sit above the static house list, "
                 "so the v10 layout moves them last. MLX with the context in the user turn fixed that (prompt cache hits): the pick, "
                 "LFM B-v8 r256 at 6-bit MLX, answers in 0.73 s warm (p50 0.51 s) on the house and scores 84.1% on the design-B sets; "
                 "medium Whisper adds 0.46 s. The falcon queue is finished.")


def status_section(status):
    """What is running now (--status JSON: [{"machine", "what", "next"}]), written by hand at publish time."""
    rows = "\n".join(f'<tr><td><b>{esc(x["machine"])}</b></td><td>{x["what"]}</td><td class="dim">{x.get("next", "")}</td></tr>' for x in status)
    return table(["Machine", "Now", "Next"], [rows])


def findings_section(findings):
    items = "\n".join(f'<li><span class="when">{esc(f["when"])}</span><div><b>{esc(f["title"])}</b> {f["text"]}</div></li>' for f in findings)
    return f'<ol class="findings">{items}</ol>'


def downloads_section(arts):
    rows = "\n".join(f'<tr><td class="mono">{esc(a["name"])}</td><td class="mono small">{esc(a["files"])}</td></tr>' for a in arts)
    return (table(["Model", "Archive run (piper: home-ai-archive/runs/<run>)"], [rows]) +
            '<p class="note">Every model is archived on Piper with its GGUF, LoRA adapter and config: '
            '<code>ssh piper.dolkens.net "tar -C home-ai-archive/runs/&lt;run&gt; -cf - gguf" | tar -xf -</code>, or '
            '<code>scripts/archive.sh restore &lt;run&gt; &lt;model&gt; &lt;base&gt;</code> to put one back into Ollama.</p>')


CSS = """
:root{--bg:#F4F6F3;--surface:#FFFFFF;--ink:#18211D;--muted:#5B6962;--line:#D9E0DB;--accent:#2E6B59;--accent-soft:#E3EEE9;
--ok:#2F7A4C;--ok-bg:#E2F1E7;--warn:#9A6414;--warn-bg:#F7EBD5;--bad:#A13D27;--bad-bg:#F6E0DA}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#111714;--surface:#18201C;--ink:#E3EAE6;--muted:#9AA8A1;
--line:#2C3832;--accent:#7CC0AA;--accent-soft:#1F3029;--ok:#7FCB9A;--ok-bg:#1C3326;--warn:#E3B465;--warn-bg:#3A2E18;--bad:#EE9A86;--bad-bg:#3D221C}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#111714;--surface:#18201C;--ink:#E3EAE6;--muted:#9AA8A1;--line:#2C3832;--accent:#7CC0AA;
--accent-soft:#1F3029;--ok:#7FCB9A;--ok-bg:#1C3326;--warn:#E3B465;--warn-bg:#3A2E18;--bad:#EE9A86;--bad-bg:#3D221C}
body{background:var(--bg);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding-inline:20px;padding-block:28px 64px}
h1,h2{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;text-wrap:balance;margin:0}
h1{font-size:30px;font-weight:600;letter-spacing:-.01em}
h2{font-size:20px;font-weight:600;margin-block:36px 10px;padding-top:14px;border-top:1px solid var(--line)}
.lede{color:var(--muted);max-width:72ch;margin:8px 0 0}
.meta{font:12px/1.4 "IBM Plex Mono",ui-monospace,monospace;color:var(--muted);letter-spacing:.02em;text-transform:uppercase;margin-bottom:6px}
.scroll{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{font:600 11.5px/1.3 "IBM Plex Sans Condensed",system-ui,sans-serif;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
td.n{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
td.best{font-weight:600;color:var(--ok)}
td.worst{color:var(--bad)}
td.dim{color:var(--muted)}
.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:13px}
.small{font-size:12px}
.sub{display:block;color:var(--muted);font-size:12px;font-family:"IBM Plex Sans",system-ui,sans-serif;margin-top:2px}
table.champ td{padding-block:12px}
table.champ td:first-child{width:22%}
.pill{display:inline-block;font:600 10.5px/1 "IBM Plex Sans Condensed",sans-serif;text-transform:uppercase;letter-spacing:.06em;
color:var(--accent);background:var(--accent-soft);border-radius:3px;padding:3px 5px;margin-left:6px;vertical-align:1px}
.chip{display:inline-block;font:500 12px/1 "IBM Plex Mono",monospace;padding:5px 8px;border-radius:4px;white-space:nowrap;
color:var(--muted);background:var(--accent-soft)}
.chip.ok{color:var(--ok);background:var(--ok-bg)}.chip.warn{color:var(--warn);background:var(--warn-bg)}.chip.bad{color:var(--bad);background:var(--bad-bg)}
.note{color:var(--muted);font-size:12.5px;max-width:95ch;margin:8px 2px 0}
.findings{list-style:none;padding:0;margin:0;display:grid;gap:10px}
.findings li{display:grid;grid-template-columns:92px 1fr;gap:14px;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px 14px}
.findings .when{font:12px/1.5 "IBM Plex Mono",monospace;color:var(--muted)}
.findings div{max-width:95ch}
code{font-family:"IBM Plex Mono",monospace;font-size:12.5px;background:var(--accent-soft);padding:1px 4px;border-radius:3px}
.legend{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 0;color:var(--muted);font-size:12.5px;align-items:center}
@media (max-width:640px){.findings li{grid-template-columns:1fr;gap:4px}table.champ td:first-child{width:auto}h1{font-size:25px}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--findings", default=None)
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--status", default=None, help="JSON list of {machine, what, next}: the in-progress section")
    a = ap.parse_args()
    ch = load("outputs/CHAMPIONS.json") or {}
    dc = load("outputs/DESIGN_CHAMPIONS.json") or {}
    heard = load("outputs/HEARD.json") or {}
    fc = load("outputs/FALCON_CHAMPIONS.json") or {}
    findings = load(a.findings) if a.findings else []
    arts = load(a.artifacts) if a.artifacts else []
    status = load(a.status) if a.status else []
    now = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    page = f"""<title>Bundook Voice Lab</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<div class="meta">Local voice assistant for 319 Bundook Rd · updated {esc(now)}</div>
<h1>Bundook Voice Lab</h1>
<p class="lede">Fine-tuned local models for a Home Assistant house of about 11,600 entities, compared across designs A to E, on typed and on transcribed speech. Each design keeps its own champion; every score is a held-out test the model never trained on, and every speed compared here comes from the same idle RTX 3090 (cassowary) unless marked otherwise.</p>
<h2 id="champions-mac">Champions: low-power Mac (Mac mini M2 Pro)</h2>
{mac_champions_section()}
<h2 id="champions">Champions: 3090-equipped (cassowary)</h2>
{champions_section(ch, dc)}
<div class="legend">Speed against the target (most requests under 2 s): {chip(1.5)} under 2 s {chip(3.5)} 2–5 s {chip(6)} over 5 s</div>
{'<h2 id="now">In progress</h2>' + status_section(status) if status else ''}
<h2 id="house">The house, spoken: picks and candidates</h2>
{house_voice_section()}
<h2 id="mac">Model size on the Mac mini: 230M to 2.6B</h2>
{mac_section()}
<h2 id="ollama">Ollama 0.40.3 re-benchmark (cassowary)</h2>
{ollama_section()}
<h2 id="tiers">Size tiers: accuracy and speed champions</h2>
{board_tiers(dc)}
<h2 id="fastslow">E · fast + slow pairings</h2>
{board_e(dc)}
<h2 id="voice">Voice robustness on everyday houses (design A, earlier round)</h2>
{voice_section(heard)}
<h2 id="single">A · single model, device list in the prompt</h2>
{board_a(ch)}
<h2 id="lookup">B · single model with lookup tools</h2>
{board_b(dc)}
<h2 id="tiered">C / D · tiered: orchestrator and room leaves</h2>
{board_tiered(dc)}
<h2 id="leaves">Room leaves (used by C and D)</h2>
{board_leaf(dc)}
<h2 id="context">Context window sweep</h2>
{board_ctx()}
<h2 id="falcon">Low-power stream: models on the Mac mini (falcon)</h2>
{falcon_section(fc)}
<h2 id="findings">What we've learned</h2>
{findings_section(findings)}
<h2 id="downloads">Downloads</h2>
{downloads_section(arts)}
</div>
<script>
// per column with a direction: best value green, worst red (ties share the colour; columns with one value are left alone)
document.querySelectorAll("table").forEach(t => {{
  const ths = [...t.querySelectorAll("thead th")];
  ths.forEach((th, i) => {{
    const dir = th.dataset.dir; if (!dir) return;
    const cells = [...t.querySelectorAll("tbody tr")].map(r => r.children[i]).filter(c => c && c.classList.contains("n"));
    const vals = cells.map(c => parseFloat(c.textContent)).filter(v => !isNaN(v));
    if (new Set(vals).size < 2) return;
    const best = dir === "min" ? Math.min(...vals) : Math.max(...vals), worst = dir === "min" ? Math.max(...vals) : Math.min(...vals);
    cells.forEach(c => {{ const v = parseFloat(c.textContent); c.classList.remove("best", "worst");
      if (v === best) c.classList.add("best"); else if (v === worst) c.classList.add("worst"); }});
  }});
}});
</script>"""
    open(a.out, "w").write(page)
    print("wrote", a.out, f"{len(page) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
