"""Design-B transcripts for the agent-written ("wild") training conversations, by rejection sampling (3 Oct).

Design B's training data has no wild-style conversations: their references are design-A transcripts (A tools, A
system prompt), which homeai.datagen.train_b cannot rebuild. B is weakest exactly there (wild-b 77%, wild2-b 78-81%).
Here a B model plays each wild training conversation in the simulator with the B tools and the B system prompt, k
attempts per user turn at temperature > 0, judged like the outcome eval against the A reference (end state, plus the
reply's key facts). The best success (fewest model calls, then the shortest reply) becomes that turn's B transcript and
the history for the next turn, so every target is a verified B attempt in a B context; a conversation stops at the first
turn no attempt solves. Prompts come from train-v9's wild rows only (house seeds of every test set excluded).

  python3 -m homeai.train.rft_b sample --model homeai-4b-lora-b-v2u --layout user_context --ollama http://localhost:11434 \
      --out outputs/rft-b/wild --k 4
  python3 -m homeai.train.rft_b select --run outputs/rft-b/wild --out data/generated/train-b-wild.jsonl

Rows are stored in the classic layout (system_prompt_b default) like the other B data; the training config's
prompt_layout moves the context lines at load time.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import os
import time

from ..bridge.agent import OllamaClient
from ..datagen.convert_b import prompt_only
from ..eval.e2e import answer_ok, apply_ref, compare
from ..eval.run_eval import to_ollama
from ..house.bighouse import house_for
from ..prompt_layout import render
from ..tools_b import for_toolset
from .rft import TEST_FILES, rd

TOOLS_B, EXEC_B = for_toolset("b")[:2]


def pool(src: str) -> list[dict]:
    test_seeds = {r["house_seed"] for f in TEST_FILES + ["data/generated/wild-b.jsonl", "data/generated/wild2-b.jsonl"]
                  if os.path.exists(f) for r in rd(f)}
    out = []
    for r in rd(src):
        if not r.get("wild") or r.get("house_seed") in test_seeds:
            continue
        b = prompt_only(r)  # None when the stored A prompt cannot be rebuilt from the house (then neither can B's)
        if b is not None:
            b["a_messages"] = r["messages"]
            out.append(b)
    return out


def run_trace(llm, msgs, house, layout, max_rounds=4):
    """Free-running B tool loop; (reply, n_model_calls, trace in training-message format)."""
    ctx, trace, n, cid = list(msgs), [], 0, 0
    for _ in range(max_rounds):
        msg = llm.chat(to_ollama(render(ctx, layout)), TOOLS_B)
        n += 1
        calls = msg.get("tool_calls") or []
        content = (msg.get("content") or "").strip()
        if not calls:
            trace.append({"role": "assistant", "content": content})
            return content, n, trace
        tcs, results = [], []
        for tc in calls:
            fn = tc.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            cid += 1
            res = EXEC_B(house, fn.get("name", ""), args)
            tcs.append({"id": f"call_{cid}", "type": "function", "function": {"name": fn.get("name", ""), "arguments": args}})
            results.append({"role": "tool", "tool_call_id": f"call_{cid}", "name": fn.get("name", ""), "content": res})
        step = [{"role": "assistant", "content": "", "tool_calls": tcs}] + results
        trace.extend(step)
        ctx.extend(step)
    return "", n, trace


def sample_conv(llm_factory, ex, k, layout):
    a = ex["a_messages"]
    users = [j for j, m in enumerate(a) if m["role"] == "user"]
    history = [ex["messages"][0]]  # the B system prompt
    turns = []
    for t, j in enumerate(users):
        end = users[t + 1] if t + 1 < len(users) else len(a)
        seg = a[j + 1:end]
        before = house_for(ex)
        apply_ref(before, a[:j])  # the solved earlier turns left the reference state
        ref_after = copy.deepcopy(before)
        apply_ref(ref_after, seg)
        ref_ans = next((m for m in reversed(seg) if m["role"] == "assistant" and m.get("content")), {})
        must = (ref_ans.get("_eval") or {}).get("must_mention", [])
        acts = any(tc["function"]["name"] == "call_service" for m in seg for tc in (m.get("tool_calls") or []))
        user = {"role": "user", "content": a[j]["content"]}
        best = None
        for _ in range(k):
            house = copy.deepcopy(before)
            try:
                text, n, trace = run_trace(llm_factory(), history + [user], house, layout)
            except Exception:  # noqa: BLE001
                continue
            s_ok, _ = compare(ref_after, house, user["content"])
            a_ok, _ = answer_ok(text, [] if acts else must, ex["category"]) if ref_ans else (True, "")
            if s_ok and a_ok and (best is None or (n, len(text)) < (best[0], len(best[1]))):
                best = (n, text, trace)
        turns.append({"turn": t, "solved": best is not None, "calls": best and best[0]})
        if best is None:
            break
        history += [user] + best[2]
    solved = sum(1 for x in turns if x["solved"])
    row = None
    if solved:
        row = {k_: ex[k_] for k_ in ("category", "house_seed", "profile", "speaker") if k_ in ex}
        row.update({"id": f"rftb-{ex['id']}", "toolset": "b", "rft_b": True, "messages": history})
    return {"id": ex["id"], "turns": turns, "user_turns": len(users), "row": row}


def cmd_sample(a):
    os.makedirs(a.out, exist_ok=True)
    rows = pool(a.src)
    done_path = os.path.join(a.out, "samples.jsonl")
    done = {json.loads(l)["id"] for l in open(done_path)} if os.path.exists(done_path) else set()
    todo = [r for r in rows if r["id"] not in done][: a.limit or None]
    print(f"pool {len(rows)} wild conversations, {len(todo)} to sample, k={a.k}, temperature={a.temperature}", flush=True)
    factory = lambda: OllamaClient(a.ollama, a.model, 8192, temperature=a.temperature, think=False)  # noqa: E731
    t0, n = time.time(), 0
    with open(done_path, "a") as fh, cf.ThreadPoolExecutor(a.parallel) as pool_:
        for res in pool_.map(lambda r: sample_conv(factory, r, a.k, a.layout), todo):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            n += 1
            if n % 50 == 0:
                print(f"[{n}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    print(f"sampled {n} conversations in {time.time() - t0:.0f}s")


def cmd_select(a):
    res = rd(os.path.join(a.run, "samples.jsonl"))
    rows = [r["row"] for r in res if r["row"]]
    full = sum(1 for r in res if r["row"] and len(r["turns"]) == r["user_turns"] and all(t["solved"] for t in r["turns"]))
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    turns = sum(1 for r in res for t in r["turns"] if t["solved"])
    print(f"{len(res)} conversations sampled: {len(rows)} kept ({full} complete), {turns} solved turns -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--model", required=True)
    s.add_argument("--ollama", default="http://localhost:11434")
    s.add_argument("--layout", default="user_context", help="the prompt layout the sampling model was trained on")
    s.add_argument("--src", default="data/generated/train-v9.jsonl")
    s.add_argument("--out", required=True)
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--parallel", type=int, default=2)
    s.add_argument("--limit", type=int, default=0)
    s.set_defaults(fn=cmd_sample)
    s = sub.add_parser("select")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_select)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
