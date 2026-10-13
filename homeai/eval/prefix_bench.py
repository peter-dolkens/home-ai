"""Warm prefix-cache bench (falcon phase 2, Peter 29 Sep): in deployment the house is fixed, so the system prompt (inventory,
rules, tools) repeats on every request and the server could reuse its KV cache. homeai.eval.latency can't show that: every
request there comes from a different house. Here the rows of wild / wild2 (a few houses, many requests each) are grouped
by house and each group's first user requests run through the full tool loop back to back, like latency.run_request:
the first request of a group is cold (another house was served last), the rest are warm.

  prompt       rows whose whole system prompt is identical (same house, clock minute, speaker and presence lines)
  house        the same groups, but the clock line moves on a minute per request: today's deployment; in the classic layout
               the Now / Speaker / People lines sit near the top of the system prompt (homeai/house/snapshot.py), so a
               cached prefix ends at line 3 and the inventory and rules are evaluated again every time
  header-last  as house, in the stable_prefix layout (those lines in a final "Current context:" block; homeai/house/
               snapshot.py relayout): what v10 is trained with; for a model of another layout it is timing only
"prompt" and "house" send the requests in the model's own layout (homeai/prompt_layout.py render: for a stable_prefix
model "house" and "header-last" are the same prompt; for user_context and clock_in_user the clock ticks in the user
turn).

Per request the first model call's prompt counters show whether the prefix was reused: Ollama's prompt_eval_count counts
the tokens it evaluated (a cache hit leaves only the new tail), the OpenAI-style usage of mlx_lm.server gives prompt_tokens
and prompt_tokens_details.cached_tokens.

  python3 -m homeai.eval.prefix_bench --model homeai-4b-lora-v5 --ollama http://127.0.0.1:11610 --out outputs/eval-falcon/homeai-4b-lora-v5-prefix

Design B and big houses (2 Oct): rows with "toolset": "b" run the B tools. --group-by house groups rows by their house
(the system prompt without its Now / Speaker / People lines) instead of the identical prompt, so a set whose requests
come from different speakers and presence states in one house, like the Bundook plan (bigbench-b-bundook: 3 houses), still
forms groups; only the house mode applies there (the prompt mode needs identical prompts).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

from ..bridge.agent import make_client
from ..host import host_info
from ..house.bighouse import house_for
from ..house.snapshot import relayout
from .. import prompt_layout
from .latency import first_request, pct, run_request

MODES = ("prompt", "house", "header-last")


def tick(system: str, minutes: int) -> str:
    """The clock line 'Now: Tuesday 14:07' moved on by `minutes` (what a later request in the same house sees)."""
    def one(l):
        if not l.startswith("Now: "):
            return l
        day, hm = l[5:].rsplit(" ", 1)
        h, m = map(int, hm.split(":"))
        t = (h * 60 + m + minutes) % (24 * 60)
        return f"Now: {day} {t // 60:02d}:{t % 60:02d}"
    return "\n".join(one(l) for l in system.split("\n"))


def header_last(system: str) -> str:
    return relayout(system, "stable_prefix")


CONTEXT_LINES = ("Now:", "Speaker:", "People:")


def house_key(system: str) -> str:
    return "\n".join(l for l in system.split("\n") if not l.startswith(CONTEXT_LINES))


def groups(rows: list[dict], n_groups: int, per_group: int, by_house: bool = False) -> list[list[dict]]:
    """The n_groups biggest sets of rows with an identical system prompt (by_house: the same house, whatever the
    speaker, presence and clock lines say), per_group rows each."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        k = r["messages"][0]["content"]
        by.setdefault(house_key(k) if by_house else k, []).append(r)
    return [rs[:per_group] for _, rs in sorted(by.items(), key=lambda kv: -len(kv[1]))[:n_groups] if len(rs) >= 3]


def first_call(llm) -> dict:
    """Prompt counters of a request's first model call: (evaluated tokens, prompt tokens if known, prompt seconds)."""
    s = llm.last or {}
    if "prompt_eval_count" in s or "prompt_eval_duration" in s:  # Ollama
        return {"eval_tokens": s.get("prompt_eval_count"), "prompt_s": (s.get("prompt_eval_duration") or 0) / 1e9}
    total = s.get("prompt_tokens")
    cached = (s.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return {"eval_tokens": None if total is None else total - cached, "prompt_tokens": total, "cached_tokens": cached}


class Probe:
    """Wraps the client so every call of a request keeps its prompt counters (the first one: the prefix across requests;
    the later ones: the request's own history, an exact extension of the previous call)."""

    def __init__(self, llm):
        self.llm, self.calls = llm, []

    def chat(self, messages, tools):
        msg = self.llm.chat(messages, tools)
        self.calls.append(first_call(self.llm))
        return msg


def bench(llm, rows: list[dict], mode: str, n_groups: int, per_group: int, layout=None, by_house: bool = False) -> dict:
    gs = groups(rows, n_groups, per_group, by_house)
    cold, warm = [], []
    for g in gs:
        for i, ex in enumerate(g):
            first = {**ex, "messages": ex["messages"][:2]}
            if mode != "prompt":  # never the exact prompt the prompt mode cached
                first["messages"] = [{**first["messages"][0], "content": tick(first["messages"][0]["content"], i + 1)},
                                     first["messages"][1]]
            sys_msg, user = first_request(first, "stable_prefix" if mode == "header-last" else layout)
            p = Probe(llm)
            dt, n = run_request(p, house_for(ex), sys_msg, user, toolset=ex.get("toolset"))
            later = [c["eval_tokens"] for c in p.calls[1:] if c.get("eval_tokens") is not None]
            (warm if i else cold).append({"s": dt, "calls": n, **(p.calls[0] if p.calls else {}),
                                          "later_eval_tokens": statistics.mean(later) if later else None})
    mean = lambda xs: round(statistics.mean(xs), 3) if xs else None  # noqa: E731
    col = lambda xs, k: [x[k] for x in xs if x.get(k) is not None]  # noqa: E731
    return {"groups": len(gs), "requests": len(cold) + len(warm),
            "cold_mean": mean(col(cold, "s")), "warm_mean": mean(col(warm, "s")),
            "cold_p50": cold and round(pct(col(cold, "s"), .5), 3), "warm_p50": warm and round(pct(col(warm, "s"), .5), 3),
            "warm_p90": warm and round(pct(col(warm, "s"), .9), 3),
            "cold_eval_tokens": mean(col(cold, "eval_tokens")), "warm_eval_tokens": mean(col(warm, "eval_tokens")),
            "cold_prompt_s": mean(col(cold, "prompt_s")), "warm_prompt_s": mean(col(warm, "prompt_s")),
            "prompt_tokens": mean(col(cold + warm, "prompt_tokens")) or mean(col(cold, "eval_tokens")),
            "later_eval_tokens": mean(col(cold + warm, "later_eval_tokens")),  # 2nd+ calls of a request
            "calls_per_request": mean(col(cold + warm, "calls"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/generated/wild.jsonl,data/generated/wild2.jsonl")
    ap.add_argument("--ollama", default="http://localhost:11434", help="Ollama URL, or an OpenAI-compatible one ending in /v1")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--groups", type=int, default=6)
    ap.add_argument("--per-group", type=int, default=6)
    ap.add_argument("--group-by", choices=("prompt", "house"), default="prompt",
                    help="house: group by house only (speaker / presence lines may differ); runs the house mode")
    ap.add_argument("--out", required=True)
    prompt_layout.add_arg(ap)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    layout = prompt_layout.for_model(a.model, a.prompt_layout)
    rows = [json.loads(l) for d in a.data.split(",") for l in open(d) if l.strip()]
    llm = make_client(a.ollama, a.model, a.num_ctx, temperature=0.0, think=False)
    ex = rows[-1]  # warm-up: load the model and compile kernels; not counted
    run_request(llm, house_for(ex), *first_request(ex, layout), toolset=ex.get("toolset"))
    t0 = time.time()
    by_house = a.group_by == "house"
    names = ["house"] if by_house else a.modes.split(",")
    modes = {m: bench(llm, rows, m, a.groups, a.per_group, layout, by_house) for m in names}
    rep = {"model": a.model, "data": a.data, "num_ctx": a.num_ctx, "prompt_layout": layout, "group_by": a.group_by, "modes": modes, "seconds": round(time.time() - t0, 1),
           **host_info(), "ollama": a.ollama}
    json.dump(rep, open(os.path.join(a.out, "prefix.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
