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
         "mean of test v2, wild, wild 2 and their outcome evals", best.get("s_per_request")),
        ("A · single model", "large houses, 32k window", lh.get("model"), f"{fmt(lh.get('pass'))} large-house outcome",
         "200 conversations, 3k–20k entities + the Bundook plan", lh_req),
        ("B · lookup tools", "rooms-only prompt", b.get("model"), f"{fmt(b.get('score'))} outcome",
         "mean of the B versions of large-house, test v2, wild, wild 2", b_req),
        ("C · tiered, routed", "orchestrator sends each room its instruction", t.get("config"), f"{fmt(t.get('score'))} outcome",
         "mean over " + ", ".join((t.get("scores") or {}).keys()), t.get("turn_mean_s")),
        ("D · tiered, fan-out", "every room leaf gets the request", d.get("config"), f"{fmt(d.get('score'))} outcome",
         "mean over " + ", ".join((d.get("scores") or {}).keys()), d.get("turn_mean_s")),
        ("Room leaf", "component of C and D", lf.get("leaf"), f"{fmt(lf.get('end_state'))} end state",
         "1,040 held-out Bundook room instructions", (lf.get("latency_ms_p50") or 0) / 1000 or None),
        ("E · fast + slow", "reconciled", (e.get("pair") or "being measured").replace("homeai-", "").replace("__", " → "),
         f"{fmt(e.get('reconciled_pass'))} final · {fmt(e.get('flicker_rate'))} flicker" if e else "–",
         "fast model acts, slow model runs independently, reconciler fixes differences", e.get("time_to_first_action_s_acting")),
    ]
    out = ['<div class="scroll"><table class="champ"><thead><tr><th>Design</th><th>Champion</th><th>Score</th><th>Speed</th></tr></thead><tbody>']
    for design, sub, model, score, basis, req in rows:
        out.append(f'<tr><td><b>{esc(design)}</b><span class="sub">{esc(sub)}</span></td><td class="mono">{esc(model or "–")}</td>'
                   f'<td><b>{esc(score)}</b><span class="sub">{esc(basis)}</span></td><td>{chip(req)}</td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out)


def fmt(v):
    return "–" if v is None else f"{v * 100:.1f}%"


def mean_req(names):
    xs = [per_request(rep(n)) for n in names]
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
                 "C, routed: a 4B orchestrator (design B's tools plus ask_rooms) sends one instruction per room to a Qwen3.5 2B leaf. "
                 "D, fan-out: every room leaf gets the request verbatim and the orchestrator composes the answer. "
                 "Timed with cassowary and lapwing both running other work.")


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
                    f'<span class="sub">{esc(f"{f["s_per_request"]:.2f} s / request · {f["accuracy"] * 100:.1f}%")}'
                    f'{" · also the accuracy champion" if same else ""}</span></td></tr>')
    return table(["Size tier", "Accuracy champion", *[(x, "max") for x in ("test v2 outcome", "wild outcome", "wild 2 outcome", "rules",
                  "large houses 32k", "accuracy")], "speed", "Speed champion"], rows,
                 "Accuracy = mean of the three outcome evals (every candidate has them); rules and large houses are shown but not averaged in, "
                 "so no model is ranked on a different set of tests. "
                 "Speed champion = the fastest per request within 2 points of the tier's accuracy champion. These are the pieces design E mixes.")


def board_e(dc):
    rows = []
    for k, v in sorted(((dc.get("E") or {}).get("all") or {}).items(), key=lambda kv: -(kv[1].get("reconciled_pass") or 0)):
        rows.append(f'<tr><td class="mono">{esc(v.get("fast", k.split("__")[0]))}</td><td class="mono">{esc(v.get("slow", k.split("__")[-1]))}</td>'
                    f'{pct(v.get("fast_pass"))}{pct(v.get("slow_pass"))}{pct(v.get("reconciled_pass"), True)}{pct(v.get("agreement_acting"))}'
                    f'{pct(v.get("flicker_rate_acting"))}{secs(v.get("time_to_first_action_s_acting"))}{secs(v.get("slow_alone_s_acting"))}</tr>')
    return table(["Fast model", "Slow model", ("fast alone", "max"), ("slow alone", "max"), ("reconciled", "max"), ("agree (acting turns)", "max"),
                  ("corrected (acting turns)", "min"), ("first action s", "min"), ("slow alone s", "min")], rows,
                 "Both models answer every turn of test v2, wild and wild 2 independently; the reconciler keeps the fast model's action when "
                 "the resulting states agree and otherwise reverts it and applies the slow model's. Corrected = a wrong action you would "
                 "briefly see. Timings are each model's own run (the 2B on lapwing's 3080, others on the 3090), per acting turn.")


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


def findings_section(findings):
    items = "\n".join(f'<li><span class="when">{esc(f["when"])}</span><div><b>{esc(f["title"])}</b> {f["text"]}</div></li>' for f in findings)
    return f'<ol class="findings">{items}</ol>'


def downloads_section(arts):
    rows = "\n".join(f'<tr><td class="mono">{esc(a["name"])}</td><td class="mono small">{esc(a["files"])}</td></tr>' for a in arts)
    return (table(["Model", "Files (in /ai/experiments/home-ai/outputs/<run>)"], [rows]) +
            '<p class="note">Copy any of them: <code>scp -r cassowary.dolkens.net:/ai/experiments/home-ai/outputs/&lt;run&gt;/gguf ./</code> '
            '(GGUFs are symlinks into /ai/models; scp follows them). Leaf models live on lapwing under <code>~/home-ai/outputs</code>.</p>')


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
    a = ap.parse_args()
    ch = load("outputs/CHAMPIONS.json") or {}
    dc = load("outputs/DESIGN_CHAMPIONS.json") or {}
    findings = load(a.findings) if a.findings else []
    arts = load(a.artifacts) if a.artifacts else []
    now = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    page = f"""<title>Bundook Voice Lab</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<div class="meta">Local voice assistant for 319 Bundook Rd · updated {esc(now)}</div>
<h1>Bundook Voice Lab</h1>
<p class="lede">Fine-tuned local models for a Home Assistant house of about 11,600 entities, compared across five designs (A to E). Each design keeps its own champion; every score is a held-out test the model never trained on.</p>
<h2 id="champions">Champions by design</h2>
{champions_section(ch, dc)}
<div class="legend">Speed against the target (most requests under 2 s): {chip(1.5)} under 2 s {chip(3.5)} 2–5 s {chip(6)} over 5 s</div>
<h2 id="tiers">Size tiers: accuracy and speed champions</h2>
{board_tiers(dc)}
<h2 id="fastslow">E · fast + slow pairings</h2>
{board_e(dc)}
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
