"""Rejection-sampling reinforcement (expert iteration) with the outcome reward.

The model attempts each training request K times in the simulator (full tool loop, temperature > 0).
Each attempt is judged like the outcome eval: the house must end in the reference state and the reply
must carry the key facts. Per request the best successful attempt is kept (fewest model calls, then the
shortest reply); a conversation is cut at the first turn the model never solves, so every target is the
model's own verified attempt. Fine-tuning the model on its own
wins rewards any valid way of reaching the outcome instead of one prescribed call sequence, and the
fewest-calls preference rewards speed.

  python3 -m homeai.train.rft pick-base                      # best 4B candidate by outcome score
  python3 -m homeai.train.rft make-config --base homeai-4b-v4 --name homeai-4b-rft1
  python3 -m homeai.train.rft sample --model homeai-4b-v4 --out outputs/rft/homeai-4b-rft1
  python3 -m homeai.train.rft select --run outputs/rft/homeai-4b-rft1 --out data/generated/train-rft1.jsonl

Prompts come from training-side data only (train-v5: agent-written conversations + template corpus);
no test house is ever sampled.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import copy
import json
import os
import random
import time

from ..bridge.agent import OllamaClient
from ..house.generator import make_house
from ..tools import TOOLS, execute_tool
from ..eval.e2e import answer_ok, apply_ref, compare
from ..eval.run_eval import to_ollama

SENSITIVE = {"unlock", "alarm_disarm"}  # + open_cover on a garage entity
TEST_FILES = ["data/generated/test-v1.jsonl", "data/generated/test-v2.jsonl", "data/generated/test-v2-para.jsonl",
              "data/generated/wild.jsonl", "data/generated/wild2.jsonl"]


def rd(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def prompt_pool(n_agent: int, n_template: int, seed: int):
    rows = rd("data/generated/train-v5.jsonl")
    test_seeds = {r["house_seed"] for f in TEST_FILES if os.path.exists(f) for r in rd(f)}
    rows = [r for r in rows if r["house_seed"] not in test_seeds]
    rng = random.Random(seed)
    agent = [r for r in rows if str(r.get("id", "")).startswith("v5a-")]
    templ = [r for r in rows if not r.get("wild")]
    rng.shuffle(agent)
    by_cat = collections.defaultdict(list)
    for r in templ:
        by_cat[r["category"]].append(r)
    picked, cats = [], sorted(by_cat)
    while len(picked) < n_template and any(by_cat.values()):  # round-robin over categories = balanced
        for c in cats:
            if by_cat[c] and len(picked) < n_template:
                picked.append(by_cat[c].pop(rng.randrange(len(by_cat[c]))))
    return agent[:n_agent] + picked


def run_trace(llm, ctx, house, max_rounds=4):
    """Free-running tool loop; returns (reply, n_model_calls, trace in training-message format)."""
    msgs, trace, n, cid = list(ctx), [], 0, 0
    for _ in range(max_rounds):
        msg = llm.chat(msgs, TOOLS)
        n += 1
        calls = msg.get("tool_calls") or []
        content = msg.get("content", "") or ""
        msgs.append({"role": "assistant", "content": content, **({"tool_calls": calls} if calls else {})})
        if not calls:
            trace.append({"role": "assistant", "content": content.strip()})
            return content.strip(), n, trace
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
            res = execute_tool(house, fn.get("name", ""), args)
            tcs.append({"id": f"call_{cid}", "type": "function", "function": {"name": fn.get("name", ""), "arguments": args}})
            results.append({"role": "tool", "tool_call_id": f"call_{cid}", "name": fn.get("name", ""), "content": res})
            msgs.append({"role": "tool", "content": res, "tool_name": fn.get("name", "")})
        trace.append({"role": "assistant", "content": "", "tool_calls": tcs})
        trace.extend(results)
    return "", n, trace


def sample_conv(llm_factory, ex, k):
    """All user turns of one conversation, k attempts each (history = reference, as in the outcome eval)."""
    out = []
    msgs = ex["messages"]
    users = [j for j, m in enumerate(msgs) if m["role"] == "user"]
    for t, j in enumerate(users):
        end = users[t + 1] if t + 1 < len(users) else len(msgs)
        seg = msgs[j + 1:end]
        before = make_house(ex["house_seed"], ex.get("profile"))
        apply_ref(before, msgs[:j])
        ref_after = copy.deepcopy(before)
        apply_ref(ref_after, seg)
        ref_ans = next((m for m in reversed(seg) if m["role"] == "assistant" and m.get("content")), {})
        must = (ref_ans.get("_eval") or {}).get("must_mention", [])
        acts = any(tc["function"]["name"] == "call_service" for m in seg for tc in (m.get("tool_calls") or []))
        ref_calls = sum(1 for m in seg if m["role"] == "assistant")
        attempts = []
        for _ in range(k):
            house = copy.deepcopy(before)
            try:
                text, n, trace = run_trace(llm_factory(), to_ollama(msgs[:j + 1]), house)
            except Exception as e:  # noqa: BLE001
                attempts.append({"ok": False, "why": f"error {e}"[:120]})
                continue
            s_ok, s_why = compare(ref_after, house, msgs[j]["content"])
            a_ok, a_why = answer_ok(text, [] if acts else must, ex["category"]) if ref_ans else (True, "")
            attempts.append({"ok": s_ok and a_ok, "why": "; ".join(w for w in (s_why, a_why) if w)[:160], "calls": n, "trace": trace, "reply": text})
        out.append({"id": ex["id"], "turn": t, "user_index": j, "category": ex["category"], "ref_calls": ref_calls, "attempts": attempts})
    return out


def cmd_sample(a):
    os.makedirs(a.out, exist_ok=True)
    pool = prompt_pool(a.agent, a.template, a.seed)
    json.dump([r["id"] for r in pool], open(os.path.join(a.out, "pool_ids.json"), "w"))
    done_path = os.path.join(a.out, "samples.jsonl")
    done = {json.loads(l)["id"] for l in open(done_path)} if os.path.exists(done_path) else set()
    todo = [r for r in pool if r["id"] not in done]
    print(f"pool {len(pool)} conversations, {len(todo)} to sample, k={a.k}, temperature={a.temperature}", flush=True)
    factory = lambda: OllamaClient(a.ollama, a.model, 8192, temperature=a.temperature, think=False)  # noqa: E731
    t0, n = time.time(), 0
    with open(done_path, "a") as fh, cf.ThreadPoolExecutor(a.parallel) as ex:
        for res in ex.map(lambda r: sample_conv(factory, r, a.k), todo):
            for turn in res:
                fh.write(json.dumps(turn) + "\n")
            fh.flush()
            n += 1
            if n % 50 == 0:
                print(f"[{n}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    print(f"sampled {n} conversations in {time.time() - t0:.0f}s")


def cmd_select(a):
    turns = rd(os.path.join(a.run, "samples.jsonl"))
    rows = {r["id"]: r for r in rd("data/generated/train-v5.jsonl")}
    by_conv = collections.defaultdict(dict)
    for t in turns:
        by_conv[t["id"]][t["user_index"]] = t
    stats = collections.Counter()
    cat_ok = collections.defaultdict(lambda: [0, 0])
    out = []
    for cid, tmap in by_conv.items():
        ex = rows.get(cid)
        if not ex:
            continue
        msgs = ex["messages"]
        users = [j for j, m in enumerate(msgs) if m["role"] == "user"]
        new = [msgs[0]]
        any_self = False
        for t, j in enumerate(users):
            end = users[t + 1] if t + 1 < len(users) else len(msgs)
            new.append(msgs[j])
            turn = tmap.get(j)
            wins = [x for x in (turn or {}).get("attempts", []) if x.get("ok")]
            seg = msgs[j + 1:end]
            ref_ans = next((m for m in reversed(seg) if m["role"] == "assistant" and m.get("content")), {})
            must = [x.lower() for x in (ref_ans.get("_eval") or {}).get("must_mention", [])]
            if a.strict_facts and must:
                # training reward stricter than the eval's half-the-facts rule: a win must state every key fact
                wins = [x for x in wins if all(any(alt.strip() in (x.get("reply") or "").lower() for alt in f.split("|")) for f in must)]
            if a.confirm_sensitive:
                # never reinforce an unconfirmed unlock / disarm / garage open, even when the user pre-confirmed
                asked_before = len(new) >= 2 and new[-2]["role"] == "assistant" and "?" in (new[-2].get("content") or "")
                def sensitive(x):
                    for m in x["trace"]:
                        for tc in m.get("tool_calls") or []:
                            ar = tc["function"].get("arguments") or {}
                            ids = json.dumps(ar.get("target") or {})
                            if ar.get("service") in SENSITIVE or (ar.get("service") == "open_cover" and "garage" in ids):
                                return True
                    return False
                if not asked_before:
                    wins = [x for x in wins if not sensitive(x)]
            ref_len = len(ref_ans.get("content") or "")
            stats["turns"] += 1
            cat_ok[ex["category"]][1] += 1
            if wins:
                if a.tiebreak == "shortest":
                    best = min(wins, key=lambda x: (x["calls"], len(x.get("reply") or "")))
                else:  # fewest calls, then the reply closest in length to the reference (no terseness bias)
                    best = min(wins, key=lambda x: (x["calls"], abs(len(x.get("reply") or "") - ref_len)))
                new.extend(best["trace"])
                any_self = True
                stats["self"] += 1
                cat_ok[ex["category"]][0] += 1
                stats["self_calls"] += best["calls"]
                stats["ref_calls_same_turns"] += turn["ref_calls"]
                stats["fewer_calls_than_ref"] += best["calls"] < turn["ref_calls"]
            else:
                # never solved: stop here, so every assistant target in the example is the model's own verified
                # attempt (pure outcome-reward signal, no new reference data mixed into the comparison)
                stats["unsolved"] += 1
                stats["truncated_after"] += len(users) - t - 1
                new.pop()  # drop the unsolved user turn
                break
        if any_self:
            out.append({**ex, "id": f"rft-{cid}", "messages": new})
    mix = []
    if a.mix and a.mix_n:
        pool = [r for r in rd(a.mix) if not str(r.get("id", "")).startswith("v5a-")]
        mix = random.Random(a.seed).sample(pool, min(a.mix_n, len(pool)))
    data = out + mix
    random.Random(a.seed).shuffle(data)
    with open(a.out, "w") as fh:
        for r in data:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"turns": stats["turns"], "solved_by_model": stats["self"], "unsolved": stats["unsolved"], "later_turns_dropped": stats["truncated_after"],
               "solve_rate": stats["self"] / max(1, stats["turns"]),
               "mean_calls_chosen": stats["self_calls"] / max(1, stats["self"]),
               "mean_calls_reference_same_turns": stats["ref_calls_same_turns"] / max(1, stats["self"]),
               "chosen_fewer_calls_than_reference": stats["fewer_calls_than_ref"],
               "conversations": len(out), "mixed_in_original": len(mix), "examples": len(data),
               "solve_rate_by_category": {c: round(v[0] / v[1], 3) for c, v in sorted(cat_ok.items())}}
    json.dump(summary, open(os.path.join(a.run, "select.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


def cmd_pick_base(a):
    ch = json.load(open("outputs/CHAMPIONS.json"))
    models = json.load(open("configs/models.json"))
    cands = []
    for r in ch["rows"]:
        meta = models.get(r["model"], {})
        cfg = json.load(open(meta["config"])) if meta.get("config") else {}
        if cfg and "4B" in cfg.get("base_model", "") and not meta.get("retired") and r.get("outcome") is not None \
                and not meta.get("method", "").startswith("RFT"):
            cands.append((r["outcome"], r["model"]))
    if not cands:
        raise SystemExit("no 4B candidate with a complete outcome score yet")
    print(max(cands)[1])


def cmd_make_config(a):
    models = json.load(open("configs/models.json"))
    base_meta = models[a.base]
    cfg = json.load(open(base_meta["config"]))
    n = a.name.split("-")[-1]
    cfg.update({"_comment": f"Rejection-sampling reinforcement from {a.base}: fine-tune its adapter on its own outcome-verified attempts "
                            f"(fewest calls wins) mixed with a slice of its original training data against drift.",
                "init_adapter": os.path.join(cfg["out"], "lora"), "train": a.train, "val": cfg.get("val"),
                "out": f"outputs/qwen35-4b-{n}", "epochs": a.epochs, "lr": a.lr})
    path = f"configs/qwen35-4b-{n}.json"
    json.dump(cfg, open(path, "w"), indent=2)
    models[a.name] = {"config": path, "method": f"RFT on {a.base}", "data": f"rft({base_meta['data']})",
                      "train_examples": sum(1 for _ in open(a.train)) if os.path.exists(a.train) else None}
    json.dump(models, open("configs/models.json", "w"), indent=2)
    print(path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--ollama", default="http://localhost:11434")
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--agent", type=int, default=600)
    s.add_argument("--template", type=int, default=400)
    s.add_argument("--parallel", type=int, default=4)
    s.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("select")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--mix", default=None, help="the base model's own training data, mixed in against drift")
    s.add_argument("--mix-n", type=int, default=1500)
    s.add_argument("--strict-facts", action="store_true", help="wins must contain every key fact (not half)")
    s.add_argument("--confirm-sensitive", action="store_true", help="drop wins that unlock/disarm without a confirmation turn")
    s.add_argument("--tiebreak", choices=["shortest", "ref-length"], default="shortest")
    s.add_argument("--seed", type=int, default=7)
    sub.add_parser("pick-base")
    s = sub.add_parser("make-config")
    s.add_argument("--base", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--train", required=True)
    s.add_argument("--epochs", type=int, default=1)
    s.add_argument("--lr", type=float, default=5e-5)
    a = ap.parse_args()
    {"sample": cmd_sample, "select": cmd_select, "pick-base": cmd_pick_base, "make-config": cmd_make_config}[a.cmd](a)


if __name__ == "__main__":
    main()
