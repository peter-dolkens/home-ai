"""End-to-end response-time bench: what a person waits for after they stop talking.

Runs the first user request of a fixed sample of conversations through the full agent loop
(model -> get_state/call_service in the simulator -> model ... -> spoken answer) and times the
whole request, not single model calls. Run it on an idle GPU (the overnight driver only runs it
between training jobs); the report records whether a training container was up.

  python3 -m homeai.eval.latency --model homeai-4b-v2 --data data/generated/test-v2.jsonl --n 60

cold = first time this request is seen (system prompt not in Ollama's prefix cache; this is what
production looks like today because the clock line near the top of the prompt changes every minute).
warm = the identical request repeated straight away (upper bound on what prefix caching could give).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time

from ..bridge.agent import OllamaClient, make_client
from ..house.bighouse import house_for  # small houses via generator.make_house, big ones via ex["big"]
from ..tools import TOOLS, execute_tool


def training_running() -> bool:
    try:
        out = subprocess.run(["docker", "ps", "--filter", "name=homeai-train", "-q"], capture_output=True, text=True, timeout=10).stdout
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return False


def run_request(llm: OllamaClient, house, system: str, user: str, max_rounds: int = 4):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    calls_n = 0
    t0 = time.perf_counter()
    for _ in range(max_rounds):
        msg = llm.chat(msgs, TOOLS)
        calls_n += 1
        calls = msg.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": msg.get("content", "") or "", **({"tool_calls": calls} if calls else {})})
        if not calls:
            break
        for tc in calls:
            fn = tc.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            msgs.append({"role": "tool", "content": execute_tool(house, fn.get("name", ""), args), "tool_name": fn.get("name", "")})
    return time.perf_counter() - t0, calls_n


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/generated/test-v2.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_dir = a.out or os.path.join("outputs", "eval", a.model.replace(":", "_") + "-latency")
    os.makedirs(out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    step = max(1, len(rows) // a.n)
    sample = rows[::step][:a.n]  # fixed, category-mixed sample (the files are shuffled at build time)
    llm = make_client(a.ollama, a.model, 8192, temperature=0.0, think=False)
    busy_before = training_running()
    # warm-up: load the model and compile kernels; not counted
    for ex in rows[-3:]:
        run_request(llm, house_for(ex), ex["messages"][0]["content"], ex["messages"][1]["content"])
    cold, warm, ncalls, per = [], [], [], []
    for ex in sample:
        sys_msg, user = ex["messages"][0]["content"], ex["messages"][1]["content"]
        dt, n = run_request(llm, house_for(ex), sys_msg, user)
        cold.append(dt); ncalls.append(n); per.append(dt / n)
        dt2, _ = run_request(llm, house_for(ex), sys_msg, user)
        warm.append(dt2)
    rep = {"model": a.model, "data": a.data, "requests": len(sample),
           "cold_mean": statistics.mean(cold), "cold_p50": pct(cold, .5), "cold_p90": pct(cold, .9),
           "warm_mean": statistics.mean(warm), "warm_p90": pct(warm, .9),
           "calls_per_request": statistics.mean(ncalls), "seconds_per_call": statistics.mean(per),
           "training_running": busy_before or training_running()}
    json.dump(rep, open(os.path.join(out_dir, "latency.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
