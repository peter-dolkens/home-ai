"""End-to-end response-time bench: what a person waits for after they stop talking.

Runs the first user request of a fixed sample of conversations through the full agent loop
(model -> get_state/call_service in the simulator -> model ... -> spoken answer) and times the
whole request, not single model calls. Run it on an idle GPU (the overnight driver only runs it
between training jobs); the report records whether a training container was up.

  python3 -m homeai.eval.latency --model homeai-4b-v2 --data data/generated/test-v2.jsonl --n 60

cold = first time this request is seen (system prompt not in Ollama's prefix cache; this is what
production looks like with the classic layout, whose clock line near the top of the prompt changes every minute; the
stable_prefix layout of v10 moves it to the end, user_context of v11 and clock_in_user of v12 into the user turn;
homeai/house/snapshot.py).
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
from ..host import host_info  # latency is per host: record which machine served the model
from .. import prompt_layout  # requests in the layout the model was trained on (v10 stable_prefix, v11 user_context, v12 clock_in_user)
from ..house.bighouse import house_for  # small houses via generator.make_house, big ones via ex["big"]
from ..tools import TOOLS, execute_tool
from ..tools_b import for_toolset  # an example's "toolset": "b" runs the design-B schemas and executor


def training_running() -> bool:
    try:
        out = subprocess.run(["docker", "ps", "--filter", "name=homeai-train", "-q"], capture_output=True, text=True, timeout=10).stdout
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return False


def run_request(llm: OllamaClient, house, system: str, user: str, max_rounds: int = 4, toolset=None):
    tools, execute = for_toolset(toolset)[:2] if toolset else (TOOLS, execute_tool)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    calls_n = 0
    t0 = time.perf_counter()
    for _ in range(max_rounds):
        msg = llm.chat(msgs, tools)
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
            msgs.append({"role": "tool", "content": execute(house, fn.get("name", ""), args), "tool_name": fn.get("name", "")})
    return time.perf_counter() - t0, calls_n


def first_request(ex: dict, layout) -> tuple[str, str]:
    """(system, user) of a row's first request in the model's prompt layout."""
    m = prompt_layout.render([{"role": x["role"], "content": x["content"]} for x in ex["messages"][:2]], layout)
    return m[0]["content"], m[1]["content"]


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
    ap.add_argument("--num-ctx", type=int, default=8192, help="context window (the bridge runs 8192)")
    prompt_layout.add_arg(ap)
    a = ap.parse_args()
    out_dir = a.out or os.path.join("outputs", "eval", a.model.replace(":", "_") + "-latency")
    os.makedirs(out_dir, exist_ok=True)
    layout = prompt_layout.for_model(a.model, a.prompt_layout)
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    step = max(1, len(rows) // a.n)
    sample = rows[::step][:a.n]  # fixed, category-mixed sample (the files are shuffled at build time)
    llm = make_client(a.ollama, a.model, a.num_ctx, temperature=0.0, think=False)
    busy_before = training_running()
    # warm-up: load the model and compile kernels; not counted
    for ex in rows[-3:]:
        run_request(llm, house_for(ex), *first_request(ex, layout))
    cold, warm, ncalls, per = [], [], [], []
    for ex in sample:
        sys_msg, user = first_request(ex, layout)
        dt, n = run_request(llm, house_for(ex), sys_msg, user)
        cold.append(dt); ncalls.append(n); per.append(dt / n)
        dt2, _ = run_request(llm, house_for(ex), sys_msg, user)
        warm.append(dt2)
    rep = {"model": a.model, "data": a.data, "requests": len(sample),
           "cold_mean": statistics.mean(cold), "cold_p50": pct(cold, .5), "cold_p90": pct(cold, .9),
           "warm_mean": statistics.mean(warm), "warm_p90": pct(warm, .9),
           "calls_per_request": statistics.mean(ncalls), "seconds_per_call": statistics.mean(per),
           "training_running": busy_before or training_running(), "num_ctx": a.num_ctx, "prompt_layout": layout, **host_info(), "ollama": a.ollama}
    json.dump(rep, open(os.path.join(out_dir, "latency.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
