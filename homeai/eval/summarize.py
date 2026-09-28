"""Build outputs/RESULTS.md from every outputs/eval/*/report.json, plus a model artifact index."""
from __future__ import annotations

import glob
import json
import os
import sys


def main():
    reports = []
    for p in sorted(glob.glob("outputs/eval/*/report.json")):
        r = json.load(open(p))
        r["_dir"] = os.path.dirname(p)
        reports.append(r)
    reports.sort(key=lambda r: -r["pass"])
    lines = ["# home-ai results", "", "Teacher-forced eval on `data/generated/test.jsonl` (see README for scoring). Higher is better; over-fetch closest to 1.0 is best.", "",
             "| run | examples | turns | pass | act | answer | fetch | over-fetch |", "|---|---|---|---|---|---|---|---|"]
    for r in reports:
        k = r["by_kind"]
        f = lambda key: f"{k[key]['ok'] / max(1, k[key]['n']):.0%}" if key in k else "-"
        of = f"{r['overfetch_mean']:.2f}" if r.get("overfetch_mean") else "-"
        lines.append(f"| {os.path.basename(r['_dir'])} ({r['model']}) | {r['examples']} | {r['turns']} | {r['pass']:.1%} | {f('act')} | {f('answer')} | {f('fetch')} | {of} |")
    cats = sorted({c for r in reports for c in r["by_category"]})
    lines += ["", "## Per category (pass rate)", "", "| category | " + " | ".join(os.path.basename(r["_dir"]) for r in reports) + " |",
              "|---|" + "---|" * len(reports)]
    for c in cats:
        row = []
        for r in reports:
            v = r["by_category"].get(c)
            row.append(f"{v['ok'] / max(1, v['n']):.0%} ({v['n']})" if v else "-")
        lines.append(f"| {c} | " + " | ".join(row) + " |")
    lines += ["", "## Model artifacts", ""]
    for g in sorted(glob.glob("outputs/*/gguf/*.gguf")):
        run = g.split("/")[1]
        kind = "Ollama-native quantised weights (re-import with `FROM ./file.gguf` + `RENDERER qwen3.5` / `PARSER qwen3.5`)" if "homeai" in os.path.basename(g) else "llama.cpp conversion (for llama-server; NOT loadable by Ollama's Qwen3.5 engine)"
        lines.append(f"- `{g}` ({os.path.getsize(g) / 1e9:.2f} GB), {kind}; run `{run}`, LoRA adapter `outputs/{run}/lora/`, merged 16-bit `outputs/{run}/merged16/`, log `outputs/{run.replace('-r1', '')}-train.log`")
    lines += ["", "Download from the Mac: `rsync -avP cassowary.dolkens.net:/ai/experiments/home-ai/outputs/<run>/gguf/ ./<run>/` (or `.../lora/` for the adapter, `.../merged16/` for full 16-bit weights)",
              "Run on cassowary: `ollama run <name>` or `python -m homeai.bridge.cli chat --mock --model <name>` from /ai/experiments/home-ai.", ""]
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/RESULTS.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
