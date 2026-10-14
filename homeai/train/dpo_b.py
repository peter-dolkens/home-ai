"""Preference pairs for DPO from a design-B model's own failures (14 Oct).

Rejection sampling (homeai/train/rft_b.py) found nothing to learn: most training turns were already solved, and the
model's own successes teach it what it already does. DPO also uses the failures. Here the model replays the B training
conversations turn by turn (the reference history up to each user turn, then k free-running attempts with the B tools
at temperature > 0), each attempt judged like the outcome eval (end state, plus the reply's key facts). A turn with at
least one failed attempt gives pairs: chosen = the best success (fewest model calls, then the shortest reply), or the
reference transcript when no attempt solved it; rejected = a failed attempt. Prompts come from the B training file only.

  python3 -m homeai.train.dpo_b sample --model homeai-lfm25-1.2b-lora-b-v8r256e4 --ollama http://ollama:11434 \
      --out outputs/dpo-b/lfm12 --k 4 --limit 3000
  python3 -m homeai.train.dpo_b pairs --run outputs/dpo-b/lfm12 --out data/generated/dpo-b-lfm12.jsonl

Rows are stored in the classic layout like the other B data; the training config's prompt_layout moves the context
lines at load time (homeai/train/dpo.py).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import os
import random
import time

from ..bridge.agent import OllamaClient
from ..eval.e2e import answer_ok, compare
from ..house.bighouse import house_for
from ..tools_b import for_toolset
from .rft import TEST_FILES, rd
from .rft_b import run_trace

TOOLS_B, EXEC_B = for_toolset("b")[:2]
META = ("category", "house_seed", "profile", "speaker", "big", "toolset")


def apply_ref_b(house, msgs):
    """Replay the reference's service calls with the B executor (the house state the earlier turns left)."""
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            fn = tc["function"]
            if fn["name"] == "call_service":
                args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"])
                EXEC_B(house, "call_service", args)


def strip_eval(msgs):
    return [{k: v for k, v in m.items() if k != "_eval"} for m in msgs]


def pool(src: str, limit: int, seed: int) -> list[dict]:
    test_seeds = {r["house_seed"] for f in TEST_FILES + ["data/generated/wild-b.jsonl", "data/generated/wild2-b.jsonl"]
                  if os.path.exists(f) for r in rd(f)}
    rows = [r for r in rd(src) if r.get("toolset") == "b" and r.get("house_seed") not in test_seeds]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def sample_conv(llm_factory, ex, k, layout):
    msgs = ex["messages"]
    users = [j for j, m in enumerate(msgs) if m["role"] == "user"]
    turns = []
    for t, j in enumerate(users):
        end = users[t + 1] if t + 1 < len(users) else len(msgs)
        seg = msgs[j + 1:end]
        ref_ans = next((m for m in reversed(seg) if m["role"] == "assistant" and m.get("content")), None)
        if ref_ans is None:
            continue
        before = house_for(ex)
        apply_ref_b(before, msgs[:j])
        ref_after = copy.deepcopy(before)
        apply_ref_b(ref_after, seg)
        must = (ref_ans.get("_eval") or {}).get("must_mention", [])
        acts = any(tc["function"]["name"] == "call_service" for m in seg for tc in (m.get("tool_calls") or []))
        ctx = strip_eval(msgs[:j + 1])
        attempts = []
        for _ in range(k):
            house = copy.deepcopy(before)
            try:
                text, n, trace = run_trace(llm_factory(), ctx, house, layout)
            except Exception as e:  # noqa: BLE001
                attempts.append({"ok": None, "why": f"error: {e}"[:200]})
                continue
            s_ok, s_why = compare(ref_after, house, msgs[j]["content"])
            a_ok, a_why = answer_ok(text, [] if acts else must, ex["category"])
            attempts.append({"ok": bool(s_ok and a_ok), "why": s_why or a_why, "calls": n, "text": text, "trace": trace})
        turns.append({"turn": t, "j": j, "ref": strip_eval(seg), "attempts": attempts})
    return {"id": ex["id"], **{m: ex[m] for m in META if m in ex}, "messages": strip_eval(msgs), "turns": turns}


def cmd_sample(a):
    os.makedirs(a.out, exist_ok=True)
    rows = pool(a.src, a.limit, a.seed)
    done_path = os.path.join(a.out, "samples.jsonl")
    done = {json.loads(l)["id"] for l in open(done_path)} if os.path.exists(done_path) else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"pool {len(rows)} B conversations, {len(todo)} to sample, k={a.k}, temperature={a.temperature}", flush=True)
    factory = lambda: OllamaClient(a.ollama, a.model, 8192, temperature=a.temperature, think=False)  # noqa: E731
    t0, n = time.time(), 0
    with open(done_path, "a") as fh, cf.ThreadPoolExecutor(a.parallel) as ex_:
        for res in ex_.map(lambda r: sample_conv(factory, r, a.k, a.layout), todo):
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 50 == 0:
                print(f"[{n}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    print(f"sampled {n} conversations in {time.time() - t0:.0f}s")


def cmd_pairs(a):
    res = rd(os.path.join(a.run, "samples.jsonl"))
    out, stats = [], {"turns": 0, "all_ok": 0, "mixed": 0, "all_fail": 0, "errors": 0}
    rng = random.Random(a.seed)
    for r in res:
        for t in r["turns"]:
            att = [x for x in t["attempts"] if x["ok"] is not None]
            stats["errors"] += len(t["attempts"]) - len(att)
            if not att:
                continue
            stats["turns"] += 1
            good = sorted((x for x in att if x["ok"]), key=lambda x: (x["calls"], len(x["text"])))
            bad = [x for x in att if not x["ok"]]
            if not bad:
                stats["all_ok"] += 1
                continue
            stats["mixed" if good else "all_fail"] += 1
            chosen = good[0]["trace"] if good else t["ref"]
            seen = set()
            for b in rng.sample(bad, len(bad)):
                key = json.dumps(b["trace"], sort_keys=True)
                if key in seen or len(seen) >= a.per_turn:
                    continue
                seen.add(key)
                row = {k: r[k] for k in META if k in r}
                row.update({"id": f"dpo-{r['id']}-t{t['turn']}-{len(seen)}", "prompt": r["messages"][:t["j"] + 1],
                            "chosen": chosen, "rejected": b["trace"], "why": b["why"], "chosen_src": "self" if good else "ref"})
                out.append(row)
    with open(a.out, "w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(res)} conversations, {stats} -> {len(out)} pairs -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--model", required=True)
    s.add_argument("--ollama", default="http://localhost:11434")
    s.add_argument("--layout", default="user_context", help="the prompt layout the sampling model was trained on")
    s.add_argument("--src", default="data/generated/train-b-v8.jsonl")
    s.add_argument("--out", required=True)
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--parallel", type=int, default=2)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--seed", type=int, default=14)
    s.set_defaults(fn=cmd_sample)
    s = sub.add_parser("pairs")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--per-turn", type=int, default=2, help="distinct rejected attempts per turn")
    s.add_argument("--seed", type=int, default=14)
    s.set_defaults(fn=cmd_pairs)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
