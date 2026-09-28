"""Render outputs/eval/*/report.json (+ failures) into a single self-contained HTML results page.

    python -m homeai.eval.report_html --eval-dir outputs/eval --out outputs/results.html
"""
from __future__ import annotations

import argparse
import collections
import glob
import html
import json
import os

CATEGORY_GROUPS = [
    ("Control", ["control_area", "control_here", "control_named", "control_brightness", "control_all_lights", "control_compound", "control_climate", "clarify", "missing", "confirm_sensitive"]),
    ("Status", ["status_entity", "status_aggregate", "who_home"]),
    ("Presence", ["presence_where", "presence_in_room"]),
    ("Routines", ["check_house", "goodnight", "leaving"]),
    ("Energy & weather", ["energy_status", "energy_night", "energy_appliance", "weather"]),
    ("Other", ["smalltalk"]),
]

LABELS = {
    "homeai-4b-testv2": ("homeai-4b · test v2", "phase-1 4B, never saw media/appliance data"),
    "homeai-9b-testv2": ("homeai-9b · test v2", "phase-1 9B, never saw media/appliance data"),
    "homeai-4b-lora-testv2": ("homeai-4b-lora · test v2", "phase-1 bf16 LoRA 4B, never saw media/appliance data"),
    "qwen3.5:4b-baseline-testv2": ("qwen3.5:4b untuned · test v2", "base model, thinking off"),
    "qwen3.5:9b-baseline-testv2": ("qwen3.5:9b untuned · test v2", "base model, thinking off"),
    "homeai-4b-wild": ("homeai-4b · wild", "template-trained 4B on the agent-written set"),
    "homeai-9b-wild": ("homeai-9b · wild", "template-trained 9B on the agent-written set"),
    "homeai-4b-lora-wild": ("homeai-4b-lora · wild", "bf16 LoRA 4B on the agent-written set"),
    "homeai-4b-v2": ("homeai-4b-v2", "Qwen3.5-4B, QLoRA, v2 data (media/appliances/paraphrases)"),
    "homeai-9b-v2": ("homeai-9b-v2", "Qwen3.5-9B, QLoRA, v2 data"),
    "homeai-4b": ("homeai-4b", "Qwen3.5-4B, QLoRA r16, 2 epochs"),
    "homeai-9b": ("homeai-9b", "Qwen3.5-9B, QLoRA r16, 2 epochs"),
    "homeai-4b-lora": ("homeai-4b-lora", "Qwen3.5-4B, bf16 LoRA r16, 2 epochs"),
    "baseline-qwen3.5-4b-full": ("qwen3.5:4b untuned", "base model, thinking off"),
    "baseline-qwen3.5-9b-full": ("qwen3.5:9b untuned", "base model, thinking off"),
}


MODELS: dict = {}
CHAMPS: dict = {}


def why_bucket(f):
    w = f.get("why", "")
    if "got_text" in f:  # outcome eval
        parts = []
        if w and not w.startswith("missing:") and not w.startswith("empty"):
            parts.append("wrong end state")
        if "missing:" in w or "empty answer" in w:
            parts.append("answer missing facts")
        return " + ".join(parts) or w
    return w.split(":")[0].split(" (")[0]


def load(eval_dir: str, include_small: bool):
    runs = []
    for p in sorted(glob.glob(os.path.join(eval_dir, "*", "report.json"))):
        r = json.load(open(p))
        name = os.path.basename(os.path.dirname(p))
        if not include_small and r["examples"] < 100:
            continue
        r["_name"] = name
        ts = testset_of(name)
        r["_set"], r["_model"] = ts, model_of(name)
        meta = MODELS.get(r["_model"], {})
        sub = f'{meta.get("method", "")} · {meta.get("data", "")} data' if meta.get("config") else ("untuned base model" if meta else r["model"])
        r["_label"], r["_sub"] = LABELS.get(name, (f'{r["_model"]} · {ts}', sub))
        r["_mode"] = r.get("mode", "teacher")
        r["_finetuned"] = name.startswith("homeai")
        if "baseline" in name:
            r["_finetuned"] = False
        fails = os.path.join(eval_dir, name, "failures.jsonl")
        r["_why"] = collections.Counter()
        r["_examples"] = []
        if os.path.exists(fails):
            for line in open(fails):
                f = json.loads(line)
                r["_why"][why_bucket(f)] += 1
                if "got" not in f:  # e2e format
                    f["got"] = {"content": f.get("got_text"), "tool_calls": [{"function": c} for c in f.get("got_calls") or []]}
                    f["expected"] = {"content": f.get("ref_text"), "tool_calls": [{"function": c} for c in f.get("ref_calls") or []]}
                if len(r["_examples"]) < 6 and "<error" not in json.dumps(f["got"]):
                    r["_examples"].append(f)
        runs.append(r)
    runs.sort(key=lambda r: -r["pass"])
    return runs


def pct(x):
    return f"{x * 100:.0f}%"


def kind(r, k):
    v = r["by_kind"].get(k)
    return v["ok"] / max(1, v["n"]) if v else None


def esc(s):
    return html.escape(str(s))


SET_SUFFIX = [*[(f"-bigbench{c}-e2e", f"bigbench{c}-e2e") for c in ("8kps", "12k", "16k", "24k", "32kps", "48k")],
              *[(f"-bigbench-b{c}-e2e", f"bigbench-b{c}-e2e") for c in ("8k", "16k")],
              ("-bigbench-b-e2e", "bigbench-b-e2e"), ("-testv2-b-e2e", "test-v2-b-e2e"), ("-wild2-b-e2e", "wild2-b-e2e"), ("-wild-b-e2e", "wild-b-e2e"), ("-bigbench32k-e2e", "bigbench32k-e2e"), ("-bigbench-e2e", "bigbench-e2e"), ("-rules", "rules"), ("-testv2-e2e", "test-v2-e2e"), ("-wild2-e2e", "wild2-e2e"), ("-wild-e2e", "wild-e2e"), ("-wild2", "wild2"), ("-wild", "wild"), ("-para", "para")]
SET_TITLES = {
    "v1": "Test v1 (template set, 150 conversations)",
    "v2": "Test v2 (template set with media, appliances, water, vehicle)",
    "para": "Phrasing test (test v2 re-worded with held-out paraphrases)",
    "bigbench-e2e": "Large-house benchmark, outcome eval (200 conversations on 3k / 10k / 20k-entity houses and the Bundook plan). Architecture track, not scored",
    **{f"bigbench{c}-e2e": f"Large-house benchmark, outcome eval, context sweep: num_ctx {c.replace('ps', '')}" for c in ("8kps", "12k", "16k", "24k", "32kps", "48k")},
    **{f"bigbench-b{c}-e2e": f"Design B on the large-house benchmark, context sweep: num_ctx {c}" for c in ("8k", "16k")},
    "bigbench32k-e2e": "Large-house benchmark, outcome eval with a 32k context window (does a bigger window alone fix large houses?). Architecture track, not scored",
    "bigbench-b-e2e": "Design B on the large-house benchmark (rooms-only prompt + lookup tools; same requests and houses). Architecture track",
    "test-v2-b-e2e": "Design B, outcome eval on test v2 (B prompt and tools). Architecture track",
    "wild-b-e2e": "Design B, outcome eval on the wild set (B prompt and tools). Architecture track",
    "wild2-b-e2e": "Design B, outcome eval on wild set 2 (B prompt and tools). Architecture track",
    "rules": "Clarification rules (200 template conversations: one match in the room or the house -> act; several -> ask). Diagnostic, not scored",
    "wild": "Wild set (303 agent-written conversations, unseen houses)",
    "wild2": "Wild set 2 (240 agent-written conversations, never used to steer training)",
    "wild-e2e": "Outcome eval on the wild set (free-running, judged on final house state and spoken facts)",
    "wild2-e2e": "Outcome eval on wild set 2",
    "test-v2-e2e": "Outcome eval on test v2 (templates judged on end state, not the exact calls)",
}
SET_ORDER = ["bigbench-e2e", "bigbench32k-e2e", "bigbench-b-e2e", "wild2-b-e2e", "wild-b-e2e", "test-v2-b-e2e", "wild2-e2e", "wild-e2e", "test-v2-e2e", "wild2", "wild", "para", "rules", "v2", "v1"]


def testset_of(name: str) -> str:
    for suf, ts in SET_SUFFIX:
        if name.endswith(suf):
            return ts
    if "testv2" in name or name.endswith(("-v2", "-v3", "-v4", "-v5", "-v6")):
        return "v2"
    return "v1"


def model_of(name: str) -> str:
    for suf in ("-testv2-e2e", "-wild2-e2e", "-wild-e2e", "-wild2", "-wild", "-para", "-testv2", "-baseline-testv2"):
        if name.endswith(suf):
            return name[: -len(suf)]
    if name.startswith("baseline-"):
        return name[len("baseline-"):].replace("-full", "").replace("qwen3.5-", "qwen3.5:")
    return name


def render(runs, artifacts, notes):
    css = """
:root{--bg:#F2F4F3;--panel:#FFFFFF;--ink:#17252A;--muted:#5E6E73;--line:#D9DFDD;--accent:#C4611F;--good:#2E8B57;--warn:#C6912C;--bad:#B5423A;--soft:#E8ECEA}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111819;--panel:#182022;--ink:#E4EBE9;--muted:#93A3A8;--line:#2A3538;--accent:#E07A38;--good:#4FB37A;--warn:#D9A845;--bad:#D65C52;--soft:#1F2A2D}}
:root[data-theme="dark"]{--bg:#111819;--panel:#182022;--ink:#E4EBE9;--muted:#93A3A8;--line:#2A3538;--accent:#E07A38;--good:#4FB37A;--warn:#D9A845;--bad:#D65C52;--soft:#1F2A2D}
body{background:var(--bg);color:var(--ink);font-family:"Source Sans 3",system-ui,sans-serif;font-size:16px;line-height:1.5;margin:0;padding-block:32px 64px;padding-inline:clamp(16px,4vw,48px)}
h1,h2,h3{font-family:"Bricolage Grotesque","Source Sans 3",system-ui,sans-serif;text-wrap:balance;margin:0}
h1{font-size:clamp(28px,4vw,40px);font-weight:600;letter-spacing:-.01em}
h2{font-size:22px;font-weight:600;margin-block:40px 12px}
.lede{max-width:64ch;color:var(--muted);margin-top:8px}
.verdict{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:24px}
.tile{background:var(--panel);border:1px solid var(--line);padding:14px 16px}
.tile .k{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.tile .v{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:28px;font-weight:600;font-variant-numeric:tabular-nums}
.tile .s{font-size:13px;color:var(--muted)}
.bars{display:grid;gap:10px;margin-top:8px}
.bar{display:grid;grid-template-columns:minmax(140px,220px) 1fr 64px;gap:12px;align-items:center}
.bar .name{font-weight:600}.bar .sub{display:block;font-size:12px;color:var(--muted);font-weight:400}
.track{height:18px;background:var(--soft);position:relative}
.fill{height:100%;background:var(--muted)}.fill.ft{background:var(--accent)}
.num{font-family:"JetBrains Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;text-align:right}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
.wrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line)}
th,td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
td.n,th.n{text-align:right;font-family:"JetBrains Mono",ui-monospace,monospace;font-size:14px}
td.group{color:var(--muted);font-size:12px;letter-spacing:.06em;text-transform:uppercase;background:var(--soft)}
.cell{display:inline-block;min-width:44px;text-align:right;padding:2px 6px;border-radius:2px}
.c-good{background:color-mix(in srgb,var(--good) 22%,transparent)}.c-warn{background:color-mix(in srgb,var(--warn) 28%,transparent)}.c-bad{background:color-mix(in srgb,var(--bad) 24%,transparent)}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.card{background:var(--panel);border:1px solid var(--line);padding:16px}
.card h3{font-size:16px;margin-bottom:8px}
.ex{border-top:1px solid var(--line);padding-block:8px;font-size:14px}
.ex .u{color:var(--muted)}.ex code{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:12.5px;white-space:pre-wrap;word-break:break-word}
pre{background:var(--panel);border:1px solid var(--line);padding:12px 14px;overflow-x:auto;font-family:"JetBrains Mono",ui-monospace,monospace;font-size:13px}
ul{padding-left:20px}li{margin-block:4px}
.pill{display:inline-block;font-size:12px;padding:1px 8px;border:1px solid var(--line);border-radius:999px;color:var(--muted);margin-left:6px;vertical-align:middle}
.note{font-size:14px;color:var(--muted);max-width:70ch}
details{margin-block:10px}summary{cursor:pointer;font-weight:600;padding-block:6px}
"""
    champs = CHAMPS or {}
    ft = [r for r in runs if r["_finetuned"]]
    base = [r for r in runs if not r["_finetuned"]]
    out = [f"<title>Bundook Voice Model Results</title>",
           '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">',
           f"<style>{css}</style>",
           "<h1>Bundook voice model results</h1>",
           f'<p class="lede">Every fine-tuned candidate is scored on the same current test sets. Quality is the mean pass rate over {esc(", ".join(champs.get("scored_sets", champs.get("sets", []))))}; the phrasing test (para) is shown but not scored because the v2 training data contains 43% of its utterances verbatim. '
           'Speed is seconds per model turn, measured on an idle RTX 3090. Three models are kept: the best overall, the best at one second per turn or less, and the best on the outcome evals alone (' + esc(", ".join(champs.get("outcome_sets", []))) + '), which judge only the end state and the reply, not the exact calls. Updated continuously while training iterates.</p>']
    rows = champs.get("rows", [])
    by = {r["model"]: r for r in rows}
    tiles = []
    for key, title in (("best", "Best overall"), ("best_fast", "Best at 1 s/turn or less"), ("best_outcome", "Best on outcomes")):
        m = champs.get(key)
        if m and m in by:
            r = by[m]
            spt = f' · {r["s_per_turn"]:.2f} s/turn' if r.get("s_per_turn") else ""
            val = r["outcome"] if key == "best_outcome" else r["quality"]
            tiles.append(f'<div class="tile"><div class="k">{title}</div><div class="v">{pct(val)}</div><div class="s">{esc(m)}{spt}</div></div>')
        else:
            tiles.append(f'<div class="tile"><div class="k">{title}</div><div class="v">–</div><div class="s">not decided yet: evals still running</div></div>')
    if base:
        b = max(base, key=lambda r: r["pass"])
        tiles.append(f'<div class="tile"><div class="k">Best untuned</div><div class="v">{pct(b["pass"])}</div><div class="s">{esc(b["_label"])}</div></div>')
    out.append('<div class="verdict">' + "".join(tiles) + "</div>")
    if rows:
        sets = champs.get("sets", [])
        out.append("<h2>Leaderboard</h2><div class=\"wrap\"><table><thead><tr><th>model</th><th>method</th><th>data</th>"
                   + "".join(f'<th class="n">{esc(x)}</th>' for x in sets) + '<th class="n">quality</th><th class="n">outcome</th><th class="n">s/turn</th><th class="n">s/request</th></tr></thead><tbody>')
        for r in sorted(rows, key=lambda r: (r["quality"] is None, -(r["quality"] or 0), r["model"])):
            if not r.get("config") and all(v is None for k, v in r["scores"].items() if k != "test-v2"):
                pass
            tag = "".join(f'<span class="pill">{t}</span>' for k, t in (("best", "best"), ("best_fast", "best ≤1 s"), ("best_outcome", "best outcome")) if champs.get(k) == r["model"])
            ret = ' <span class="pill">retired</span>' if r.get("retired") else ""
            cells = "".join(f'<td class="n">{pct(r["scores"][x]) if r["scores"].get(x) is not None else "–"}</td>' for x in sets)
            q = f'<b>{pct(r["quality"])}</b>' if r["quality"] is not None else "–"
            q += f'</td><td class="n">{pct(r["outcome"]) if r.get("outcome") is not None else "–"}'
            spt = f'{r["s_per_turn"]:.2f}' if r.get("s_per_turn") else "–"
            spr = f'{r["s_per_request"]:.2f}' if r.get("s_per_request") else "–"
            out.append(f'<tr><td><b>{esc(r["model"])}</b>{tag}{ret}</td><td>{esc(r["method"])}</td><td>{esc(r["data"])}</td>{cells}<td class="n">{q}</td><td class="n">{spt}</td><td class="n">{spr}</td></tr>')
        out.append("</tbody></table></div>")
        out.append('<p class="note">A dash means that eval has not run yet; a model only gets a quality score once it has all of them. Retired runs were dominated on every set they have and get no further evals. s/request is the whole spoken request including tool rounds (about two model turns). The full history with training settings is in <code>outputs/results.csv</code> on cassowary.</p>')
    for ts in SET_ORDER:
        group = [r for r in runs if r["_set"] == ts]
        if not group:
            continue
        out.append(f"<h2>{esc(SET_TITLES[ts])}</h2><div class=\"bars\">")
        for r in group:
            cls = "fill ft" if r["_finetuned"] else "fill"
            out.append(f'<div class="bar"><div class="name">{esc(r["_label"])}<span class="sub">{esc(r["_sub"])}</span></div><div class="track"><div class="{cls}" style="width:{r["pass"] * 100:.1f}%"></div></div><div class="num">{pct(r["pass"])}</div></div>')
        out.append("</div>")
        if ts == "wild":
            out.append('<p class="note">Written by Claude subagents against unseen houses, deliberately not following the training templates. Teacher-forced: the model must choose the same fetch as the reference, so valid alternative strategies can fail here; the outcome eval below does not have that bias.</p>')
        if ts == "wild2":
            out.append('<p class="note">A second agent-written set (two Opus and two Sonnet authors, twelve new houses) with a neutral brief. Its failures are never used to design training data, so it stays an honest check of generalisation.</p>')
        if ts.endswith("e2e"):
            out.append('<p class="note">For each user turn the model runs the full tool loop freely in the simulator. The turn passes if the house ends in the same state as the reference (volume, brightness and position get tolerance when the user gave no number) and the reply contains at least half the key facts. Any valid way of getting there counts; wrong rooms, extra actions and skipped confirmations fail.</p>')
    out.append("<h2>Turn types</h2><div class=\"wrap\"><table><thead><tr><th>run</th><th>set</th><th class=\"n\">examples</th><th class=\"n\">turns</th><th class=\"n\">pass</th><th class=\"n\">act</th><th class=\"n\">fetch</th><th class=\"n\">answer</th><th class=\"n\">over-fetch</th><th class=\"n\">s/turn</th></tr></thead><tbody>")
    for r in runs:
        of = f'{r["overfetch_mean"]:.2f}' if r.get("overfetch_mean") else "–"
        spt = f'{r["seconds_per_turn"]:.1f}' if r.get("seconds_per_turn") else "–"
        cells = "".join(f'<td class="n">{pct(kind(r, k)) if kind(r, k) is not None else "–"}</td>' for k in ("act", "fetch", "answer"))
        out.append(f'<tr><td>{esc(r["_model"])}</td><td>{esc(r["_set"])}</td><td class="n">{r["examples"]}</td><td class="n">{r["turns"]}</td><td class="n"><b>{pct(r["pass"])}</b></td>{cells}<td class="n">{of}</td><td class="n">{spt}</td></tr>')
    out.append("</tbody></table></div>")
    out.append('<p class="note">Over-fetch is entities returned by get_state divided by entities needed: 1.00 means the model asked for exactly the right slice of the house.</p>')
    def cell(v):
        if v is None:
            return '<td class="n">–</td>'
        p = v["ok"] / max(1, v["n"])
        c = "c-good" if p >= 0.9 else "c-warn" if p >= 0.6 else "c-bad"
        return f'<td class="n"><span class="cell {c}">{pct(p)}</span> <span style="color:var(--muted);font-size:12px">/{v["n"]}</span></td>'
    out.append("<h2>Per category</h2>")
    for ts in SET_ORDER:
        grp = [r for r in runs if r["_set"] == ts]
        if not grp:
            continue
        out.append(f'<details{" open" if ts == SET_ORDER[0] or (ts == "wild" and not any(r["_set"] == SET_ORDER[0] for r in runs)) else ""}><summary>{esc(SET_TITLES[ts])}</summary><div class="wrap"><table><thead><tr><th>category</th>' + "".join(f'<th class="n">{esc(r["_model"])}</th>' for r in grp) + "</tr></thead><tbody>")
        seen = set()
        for gname, cats in CATEGORY_GROUPS:
            cats = [c for c in cats if any(c in r["by_category"] for r in grp)]
            if not cats:
                continue
            out.append(f'<tr><td class="group" colspan="{len(grp) + 1}">{gname}</td></tr>')
            for c in cats:
                seen.add(c)
                out.append(f"<tr><td>{esc(c)}</td>" + "".join(cell(r["by_category"].get(c)) for r in grp) + "</tr>")
        for c in sorted({c for r in grp for c in r["by_category"]} - seen):
            out.append(f"<tr><td>{esc(c.replace('wild_', ''))}</td>" + "".join(cell(r["by_category"].get(c)) for r in grp) + "</tr>")
        out.append("</tbody></table></div></details>")
    keep = {champs.get("best"), champs.get("best_fast"), champs.get("best_outcome")} - {None}
    if keep:
        pref = [x for x in ("wild2-e2e", "wild-e2e", "wild2") if any(r["_set"] == x for r in runs)][:2]
        ft = [r for r in runs if r["_model"] in keep and r["_set"] in pref] or ft[:3]
    else:
        ft = ft[:3]
    if ft:
        out.append("<h2>What still goes wrong</h2><div class=\"cols\">")
        for r in ft:
            out.append(f'<div class="card"><h3>{esc(r["_label"])}<span class="pill">{sum(r["_why"].values())} failed turns</span></h3>')
            out.append("<ul>" + "".join(f"<li>{esc(w)}: {n}</li>" for w, n in r["_why"].most_common(5)) + "</ul>")
            for f in r["_examples"][:3]:
                got = f["got"].get("tool_calls")
                got_s = json.dumps([tc["function"] for tc in got]) if got else (f["got"].get("content") or "")
                exp = f["expected"].get("tool_calls")
                exp_s = json.dumps([tc["function"] for tc in exp]) if exp else (f["expected"].get("content") or "")
                out.append(f'<div class="ex"><div class="u">{esc(f["category"])} · {esc(f["why"])}<br>“{esc(f["user"])}”</div><div>expected <code>{esc(exp_s[:220])}</code></div><div>got <code>{esc(got_s[:220])}</code></div></div>')
            out.append("</div>")
        out.append("</div>")
    if notes:
        out.append("<h2>Notes from the run</h2><ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul>")
    out.append("<h2>Models and downloads</h2>")
    out.append("<p class=\"note\">Everything lives on cassowary under <code>/ai/experiments/home-ai/outputs/</code>. The Ollama models are ready to use there; the GGUF files are Ollama's own quantised weights and re-import anywhere with the Modelfile shown.</p>")
    out.append("<div class=\"wrap\"><table><thead><tr><th>Ollama model</th><th>run</th><th>files</th></tr></thead><tbody>")
    for a in artifacts:
        out.append(f'<tr><td><b>{esc(a["name"])}</b></td><td>{esc(a["run"])}</td><td>{esc(a["files"])}</td></tr>')
    out.append("</tbody></table></div>")
    out.append("<pre>" + esc("""# pull a model's files (run dir = the "run" column above)
rsync -avP cassowary.dolkens.net:/ai/experiments/home-ai/outputs/<run>/gguf/ ./<model>/   # Ollama-native q4_K_M GGUF
rsync -avP cassowary.dolkens.net:/ai/experiments/home-ai/outputs/<run>/lora/ ./<model>-lora/  # LoRA adapter (~100 MB)

# the full results history with training settings, and the champion pick
rsync -avP cassowary.dolkens.net:/ai/experiments/home-ai/outputs/{results.csv,CHAMPIONS.md} .

# re-import the GGUF into any Ollama 0.21.x
cat > Modelfile <<'EOF'
FROM ./<model>-q4_K_M.gguf
TEMPLATE {{ .Prompt }}
RENDERER qwen3.5
PARSER qwen3.5
PARAMETER num_ctx 8192
PARAMETER temperature 0.2
EOF
ollama create <model> -f Modelfile

# talk to it against a mock house, on cassowary
cd /ai/experiments/home-ai && python3 -m homeai.bridge.cli chat --mock --model <model>""") + "</pre>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="outputs/eval")
    ap.add_argument("--out", default="outputs/results.html")
    ap.add_argument("--artifacts", default=None, help="json list of {name, run, files}")
    ap.add_argument("--notes", default=None, help="json list of html strings")
    ap.add_argument("--include-small", action="store_true")
    ap.add_argument("--models", default="configs/models.json")
    ap.add_argument("--champions", default=None, help="outputs/CHAMPIONS.json")
    a = ap.parse_args()
    global MODELS, CHAMPS
    MODELS = {k: v for k, v in json.load(open(a.models)).items() if not k.startswith("_")} if os.path.exists(a.models) else {}
    CHAMPS = json.load(open(a.champions)) if a.champions else {}
    runs = load(a.eval_dir, a.include_small)
    artifacts = json.load(open(a.artifacts)) if a.artifacts else []
    notes = json.load(open(a.notes)) if a.notes else []
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(render(runs, artifacts, notes))
    print(f"wrote {a.out} with {len(runs)} runs")


if __name__ == "__main__":
    main()
