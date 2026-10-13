"""Offline checks of the per-host champion pick (homeai.eval.host_champions) and the prefix bench's prompt edits."""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, ".")
from homeai.eval.host_champions import pick, rows_for
from homeai.eval.prefix_bench import header_last, tick


def write(d, name, obj):
    os.makedirs(d + "/" + name, exist_ok=True)
    json.dump(obj, open(f"{d}/{name}/{'report' if name.endswith('-e2e') else name.rsplit('-', 1)[1]}.json", "w"))


with tempfile.TemporaryDirectory() as t:
    ev = t + "/eval-h"
    models = {"big": (0.90, 11.0), "mid": (0.885, 6.0), "small": (0.86, 3.0), "big@h-q5": (0.905, 12.0)}
    for m, (acc, s) in models.items():
        for st in ("testv2", "wild", "wild2"):
            write(ev, f"{m}-{st}-e2e", {"pass": acc, "seconds_per_turn": s / 2, "host": "h"})
        write(ev, f"{m}-latency", {"cold_mean": s, "cold_p90": s * 1.3, "seconds_per_call": s / 2, "host": "h"})
    write(ev, "other-latency", {"cold_mean": 1.0, "cold_p90": 1.0, "seconds_per_call": 0.5, "host": "elsewhere"})
    rows = rows_for(ev, "h")
    by = {r["row"]: r for r in rows}
    assert by["big@h-q5"]["model"] == "big" and by["big@h-q5"]["variant"] == "q5", by["big@h-q5"]
    assert not by["other"]["ok_host"], "a report from another host must not be ranked"
    base = [r for r in rows if r["ok_host"] and not r["variant"]]
    best, ties, fast = pick(base)
    assert best == "big" and ties == ["mid"], (best, ties)  # 1.5 points apart: a tie
    assert fast == "mid", fast  # small is 4 points behind: outside the 2-point speed window
    os.makedirs(t + "/outputs", exist_ok=True)
    subprocess.run([sys.executable, "-m", "homeai.eval.host_champions", "--host", "h", "--eval-dir", ev, "--out", t + "/H"],
                   check=True, capture_output=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    md = open(t + "/H.md").read()
    assert "only comparable with other h runs" in md and "never" in md.lower() and "big@h-q5" in md
    rep = json.load(open(t + "/H.json"))
    assert rep["accuracy"] == "big" and rep["speed"] == "mid" and rep["optimisations"][0]["baseline"] == "big"

sp = "You are the voice assistant.\n\nNow: Monday 23:59\nSpeaker: Amber\nPeople: Peter (home)\n\nAreas [id]: devices\n- Kitchen"
assert "Now: Monday 00:01" in tick(sp, 2)
hl = header_last(tick(sp, 1))
assert hl.startswith("You are") and hl.endswith("People: Peter (home)") and hl.index("Areas") < hl.index("Now:")
# falcon stream: a job's prompt layout reaches every eval client (its rows never resolve through configs/models.json)
from homeai.eval import falcon_stream as fs
jobs = {j.row: j for j in fs.JOBS}
for row in ("homeai-4b-lora-v10", "homeai-4b-lora-v10@falcon-mlx4"):
    assert jobs[row].layout == "stable_prefix", row
    assert fs.client(jobs[row])[2]["HOMEAI_PROMPT_LAYOUT"] == "stable_prefix"
assert fs.client(jobs["homeai-4b-lora-v5"])[2]["HOMEAI_PROMPT_LAYOUT"] == "classic"
print("host champions + prefix bench helpers + falcon layouts: ok")
