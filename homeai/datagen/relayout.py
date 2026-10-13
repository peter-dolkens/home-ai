"""Prompt layout conversion and its checks (homeai/house/snapshot.py: LAYOUTS, relayout_messages; homeai/prompt_layout.py).

v10 (stable_prefix) moves the volatile Now / Speaker / People lines from the top of the system prompt into a final
"Current context:" block; v11 (user_context) moves them out of the system message into a "[context] ..." first line of
every user turn. Either way the tool schemas, intro, inventory and rules form a prefix that is identical for every
request in a house. v12 (clock_in_user) moves only the clock, into a "[now] ..." first line of every user turn: the
system message keeps Speaker and People where classic has them and is identical for every request from one room
while nobody comes or goes.

    python3 -m homeai.datagen.relayout convert data/generated/train-v9.jsonl data/generated/train-v10.jsonl
    python3 -m homeai.datagen.relayout sets                  # *-sp.jsonl copies of the eval sets (+ -b sets)
    python3 -m homeai.datagen.relayout check SRC DST         # DST's system prompts are SRC's, re-laid-out, by row id
    python3 -m homeai.datagen.relayout heard [--layout L]    # every heard/ set converts losslessly (in memory)
    python3 -m homeai.datagen.relayout oracle data/generated/test-v2-sp.jsonl ... [--layout L]   # e2e replay
    scripts/mlx-env.sh python -m homeai.datagen.relayout prefix   # token-level shared prefix, Qwen3.5 tokenizer

Every conversion is checked: the new conversation holds exactly the old system prompt's lines (as a multiset; the
"Current context:" header and blank separators aside, and for user_context / clock_in_user counting the first turn's
context / clock line), converting it back gives the old system prompt byte for byte, every user turn carries the bare
utterance under its context / clock line, and every other message and key of the row is unchanged. The evals render
only the requests they send in the model's layout (homeai/prompt_layout.py render) and score on the stored rows, so
the *-sp copies and the heard sets serve every layout; the copies exist for inspection and for tools that read files
directly. Training renders rows at load time (homeai/train/data.py load_jsonl), so v11 and v12 need no data file of
their own."""
from __future__ import annotations

import argparse
import collections
import copy
import glob
import json
import os
import sys
import time
from typing import Optional

from ..house.snapshot import (CLOCK, CLOCK_TAG, CONTEXT_HEAD, CONTEXT_TAG, relayout, relayout_messages, split_context,
                              split_user)
from ..prompt_layout import apply

SETS = ("test-v2", "wild", "wild2", "bigbench", "test-rules", "test-v2-b", "wild-b", "wild2-b", "bigbench-b")
SUFFIX = {"stable_prefix": "-sp", "user_context": "-uc", "clock_in_user": "-cu"}


def content_lines(msgs: list[dict]) -> collections.Counter:
    """The system prompt's non-blank lines (without the "Current context:" header) plus the first user turn's context
    or clock line (blank separators move with the block; the convert-back check covers them byte for byte)."""
    c = collections.Counter(msgs[0]["content"].split("\n"))
    c.pop(CONTEXT_HEAD, None)
    c.pop("", None)
    first = next((m for m in msgs if m["role"] == "user"), None)
    c.update((split_user(first["content"])[0] or []) if first else [])
    return c


def bare(msgs: list[dict]) -> list[dict]:
    """The turns after the system message with every user turn's context / clock line removed."""
    return [{**m, "content": split_user(m["content"])[1]} if m["role"] == "user" else m for m in msgs[1:]]


def layout_of(msgs: list[dict]) -> str:
    sys_ = msgs[0]["content"]
    tagged = [m["content"] for m in msgs if m["role"] == "user" and split_user(m["content"])[0]]
    if tagged:
        return "clock_in_user" if tagged[0].startswith(CLOCK_TAG) else "user_context"
    return "stable_prefix" if CONTEXT_HEAD in sys_ and relayout(sys_, "stable_prefix") == sys_ else "classic"


def problems(src: dict, dst: dict, whole_row: bool = True) -> list[str]:
    """What is wrong with dst as a re-laid-out src ([] when nothing). whole_row False (an augmented dst): the system
    prompt and the context lines only."""
    a, b = src["messages"], dst["messages"]
    bad = []
    if content_lines(a) != content_lines(b):
        bad.append("line multiset changed")
    if relayout_messages(b, "classic")[0] != relayout_messages(a, "classic")[0]:
        bad.append("does not convert back")
    lb = layout_of(b)
    if lb in ("user_context", "clock_in_user"):
        want = split_context(relayout_messages(a, "classic")[0]["content"])[1]
        want = want if lb == "user_context" else [l for l in want if l.startswith(CLOCK)]
        if any(m["role"] == "user" and split_user(m["content"])[0] != want for m in b):
            bad.append("a user turn lacks the conversation's context / clock line")
        if lb == "clock_in_user" and CLOCK in b[0]["content"]:
            bad.append("the clock is still in the system prompt")
    if whole_row:
        if bare(a) != bare(b):
            bad.append("non-system messages changed")
        if {k: v for k, v in src.items() if k != "messages"} != {k: v for k, v in dst.items() if k != "messages"}:
            bad.append("row keys changed")
    return bad


def convert(src: str, dst: str, layout: str = "stable_prefix") -> dict:
    n = changed = 0
    bad: list = []
    with open(dst, "w") as fh:
        for l in open(src):
            if not l.strip():
                continue
            row = json.loads(l)
            new = apply(row, layout)
            n += 1
            changed += new is not row
            p = problems(row, new)
            if p:
                bad.append((row.get("id"), p))
            fh.write(json.dumps(new) + "\n")
    rep = {"src": src, "dst": dst, "layout": layout, "rows": n, "changed": changed, "problems": len(bad), "examples": bad[:5]}
    print(json.dumps(rep))
    if bad:
        raise SystemExit(f"{dst}: {len(bad)} rows failed the content check")
    return rep


def check(src: str, dst: str) -> dict:
    """DST's system prompts against SRC's by row id (DST may be augmented: only the system message is compared)."""
    by_id = {r["id"]: r for r in map(json.loads, filter(str.strip, open(src)))}
    n, bad, layouts = 0, [], collections.Counter()
    for r in map(json.loads, filter(str.strip, open(dst))):
        s = by_id.get(r["id"])
        if s is None:
            bad.append((r["id"], ["no source row"]))
            continue
        n += 1
        layouts[layout_of(r["messages"])] += 1
        p = problems(s, r, whole_row=False)
        if p:
            bad.append((r["id"], p))
    rep = {"src": src, "dst": dst, "rows": n, "layouts": dict(layouts), "problems": len(bad), "examples": bad[:5]}
    print(json.dumps(rep))
    if bad:
        raise SystemExit(1)
    return rep


def heard(root: str = "data/generated/heard", layout: str = "stable_prefix", write: bool = False) -> dict:
    files = sorted(f for f in glob.glob(f"{root}/*.jsonl") if not f.endswith(tuple(f"{x}.jsonl" for x in SUFFIX.values())))
    rows = bad = sets = 0
    for f in files:
        out = open(f[:-6] + SUFFIX[layout] + ".jsonl", "w") if write else None
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            if "messages" not in r:  # leaf / confusion rows: no system prompt of this kind
                continue
            new = apply(r, layout)
            rows += 1
            bad += bool(problems(r, new))
            if out:
                out.write(json.dumps(new) + "\n")
        sets += 1
    rep = {"heard_sets": sets, "layout": layout, "rows": rows, "problems": bad}
    print(json.dumps(rep))
    if bad:
        raise SystemExit(1)
    return rep


# ---------------------------------------------------------------------- oracle replay through homeai.eval.e2e
class Replay:
    """A perfect model for homeai.eval.e2e.Run: answers each call with the reference's next assistant message of the
    current user turn, and checks that the system prompt it is sent is in the expected layout."""

    def __init__(self, layout: str):
        self.layout, self.ex, self.bad_prompt = layout, None, 0

    def chat(self, messages, tools=None):
        sys_msg, users = messages[0]["content"], [m["content"] for m in messages if m["role"] == "user"]
        if self.layout == "user_context":
            ok = not split_context(sys_msg)[1] and CONTEXT_HEAD not in sys_msg and all(u.startswith(CONTEXT_TAG) for u in users)
        elif self.layout == "clock_in_user":  # Speaker / People where classic has them, no clock, every turn's clock
            ctx = split_context(sys_msg)[1]
            ok = bool(ctx) and not any(l.startswith(CLOCK) for l in ctx) and relayout(sys_msg, "classic") == sys_msg \
                and all(u.startswith(CLOCK_TAG) for u in users)
        else:
            ok = relayout(sys_msg, self.layout) == sys_msg and (self.layout == "stable_prefix") == (CONTEXT_HEAD in sys_msg) \
                and not any(u.startswith((CONTEXT_TAG, CLOCK_TAG)) for u in users)
        self.bad_prompt += not ok
        ref = self.ex["messages"]
        j = max(i for i, m in enumerate(messages) if m["role"] == "user")
        k = sum(1 for m in messages[j + 1:] if m["role"] == "assistant")
        seg = []
        for m in ref[j + 1:]:
            if m["role"] == "user":
                break
            if m["role"] == "assistant":
                seg.append(m)
        if k >= len(seg):
            return {"role": "assistant", "content": ""}
        m = seg[k]
        out = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("tool_calls"):
            out["tool_calls"] = [{"function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                                 for tc in m["tool_calls"]]
        return out


def oracle(paths: list[str], layout: str, out_root: str, limit: int = 0) -> list[dict]:
    from ..eval.e2e import Run, load_rows
    reps = []
    for p in paths:
        t0 = time.time()
        out = os.path.join(out_root, os.path.basename(p)[:-6])
        run, llm = Run(out, p, layout), Replay(layout)
        for ex in load_rows(p, limit, 0):
            llm.ex = ex
            run.example(llm, ex)
        run.fails.close()
        run.turns_log.close()
        rep = {"data": p, "layout": layout, "examples": run.examples, "turns": run.kinds["state"]["n"],
               "pass": round(run.pass_rate(), 4), "conversation_pass": round(run.conv_ok / max(1, run.examples), 4),
               "errors": run.errors, "wrong_layout_calls": llm.bad_prompt, "seconds": round(time.time() - t0, 1)}
        json.dump(rep, open(os.path.join(out, "oracle.json"), "w"), indent=2)
        print(json.dumps(rep), flush=True)
        reps.append(rep)
    return reps


# ---------------------------------------------------------------------- token-level prefix check
LAYOUT_ORDER = ("classic", "stable_prefix", "user_context", "clock_in_user")


def _render(tok, messages: list[dict], tools: list, layout: str) -> list[int]:
    """Token ids of a request (system first, classic as stored, user turns may carry "_context") in `layout`."""
    from ..train.data import clean_messages
    ids = tok.apply_chat_template(clean_messages(relayout_messages(messages, layout)), tools=tools, tokenize=True,
                                  add_generation_prompt=True)
    return list(ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids)


def _shared(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _static_tokens(tok, system: str, tools: list, layout: str) -> int:
    """Tokens of the rendered request before the first context line (tool schemas + intro + inventory + rules, and for
    user_context the chat template's turn markers); clock_in_user: before the clock line, so the whole system message
    with its Speaker and People lines (what a request from the same room with the same people home can reuse)."""
    msgs = relayout_messages([{"role": "system", "content": system}, {"role": "user", "content": "x"}], layout)
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
    cut = text.index("\n" + CONTEXT_HEAD) if layout == "stable_prefix" else \
        text.index(CLOCK_TAG if layout == "clock_in_user" else CONTEXT_TAG)
    return len(tok(text[:cut], add_special_tokens=False)["input_ids"])


def later(house, minutes: int, flip: Optional[int] = 0):
    """The same house `minutes` later with person `flip` having come or gone (a request at another time; flip None:
    nobody did)."""
    import datetime as dt
    h = copy.copy(house)
    h.now = house.now + dt.timedelta(minutes=minutes)
    h.people = [copy.copy(p) for p in house.people]
    if h.people and flip is not None:
        p = h.people[flip % len(h.people)]
        p.home, p.changed_s = not p.home, 60
    return h


# what changes between the two requests of prefix_check (v12, Peter 1 Oct): the clock alone (the same satellite, nobody
# came or went), the clock and the speaker's room, the clock and who is home, and v10/v11's mixed case (clock, a
# different person on the app, presence)
CHANGES = ("time", "time+room", "time+presence", "time+speaker+presence")


def prefix_check(tokenizer: str = "Qwen/Qwen3.5-4B") -> dict:
    """Two requests for the same house, per kind of change between them (CHANGES): tokens they share at the front, per
    layout. Houses: small design-A houses (Bundook profile, random) and the full Bundook plan (design B's rooms-only
    prompt)."""
    from transformers import AutoTokenizer
    from ..house.bighouse import house_for
    from ..house.schema import Speaker
    from ..house.snapshot import system_prompt, system_prompt_b
    from ..tools import TOOLS
    from ..tools_b import TOOLS_B
    tok = AutoTokenizer.from_pretrained(tokenizer)
    tok = getattr(tok, "tokenizer", tok)
    rows = [json.loads(l) for l in open("data/generated/test-v2.jsonl")]
    rows_b = [json.loads(l) for l in open("data/generated/bigbench-b.jsonl")]
    cases = [("A, small house (test-v2, bundook profile)", next(r for r in rows if r.get("profile") == "bundook"), system_prompt, TOOLS),
             ("A, small house (test-v2, random profile)", next(r for r in rows if r.get("profile") != "bundook"), system_prompt, TOOLS),
             ("B, Bundook full plan (bigbench-b)", next(r for r in rows_b if (r.get("big") or {}).get("kind") == "bundook_full"),
              system_prompt_b, TOOLS_B)]
    out = []
    for name, ex, prompt, tools in cases:
        h1 = house_for(ex)
        areas = [a.area_id for a in h1.areas]
        who = [p.name for p in h1.people] or ["Peter", "Amber"]
        sp1 = Speaker(who[0], areas[0], "satellite")
        p1 = prompt(h1, sp1)
        r1 = [{"role": "system", "content": p1}, {"role": "user", "content": "turn the lights off"}]
        second = {"time": (later(h1, 197, None), sp1),
                  "time+room": (later(h1, 197, None), Speaker(who[0], areas[1], "satellite")),
                  "time+presence": (later(h1, 197, len(who) - 1), sp1),
                  "time+speaker+presence": (later(h1, 197), Speaker(who[-1], None, "app"))}
        for change in CHANGES:
            h2, sp2 = second[change]
            r2 = [{"role": "system", "content": prompt(h2, sp2)}, {"role": "user", "content": "is anyone home?"}]
            row = {"case": name, "change": change, "entities": len(h1.entities)}
            for layout in LAYOUT_ORDER:
                a, b = _render(tok, r1, tools, layout), _render(tok, r2, tools, layout)
                n = _shared(a, b)
                row[layout] = {"tokens": len(a), "shared": n, "share": round(n / len(a), 4)}
                if layout != "classic":
                    st = _static_tokens(tok, p1, tools, layout)
                    row[layout] |= {"static_tokens": st, "covers_static": n >= st - 1}  # a BPE merge at the boundary
            out.append(row)
    rep = {"tokenizer": tokenizer, "requests": out, "within_conversation": cross_turn(tok)}
    print(json.dumps(rep, indent=1))
    return rep


def cross_turn(tok, n: int = 40) -> dict:
    """Multi-turn wild conversations: the tokens a server can reuse (i) between rounds of one request (the tool result
    is appended; the prompt is built once per request) and (ii) from the last call of turn k to the first call of turn
    k+1, with the context refreshed for turn k+1 (clock +1 min; earlier user_context / clock_in_user turns keep the line
    they were sent with, as the bridge does), per layout."""
    from ..eval.prefix_bench import tick
    from ..tools import TOOLS
    rows = [json.loads(l) for l in open("data/generated/wild.jsonl")]
    rows = [r for r in rows if sum(m["role"] == "user" for m in r["messages"]) >= 2 and not r.get("toolset")][:n]
    sums = collections.defaultdict(lambda: [0, 0])
    rounds = collections.defaultdict(lambda: [0, 0])
    turns = 0
    for ex in rows:
        msgs = ex["messages"]
        ctx = split_context(msgs[0]["content"])[1]
        ctx2 = tick("\n".join(ctx), 1).split("\n")
        if msgs[2]["role"] == "assistant" and msgs[2].get("tool_calls"):  # (i): round 2 extends round 1's prompt
            for layout in LAYOUT_ORDER:
                a, b = _render(tok, msgs[:2], TOOLS, layout), _render(tok, msgs[:4], TOOLS, layout)
                rounds[layout][0] += _shared(a, b)
                rounds[layout][1] += len(a)
        users = [j for j, m in enumerate(msgs) if m["role"] == "user"]
        for k in range(len(users) - 1):
            j2 = users[k + 1]
            last_k = msgs[:j2]  # turn k's last call saw this, minus its final answer
            while last_k[-1]["role"] == "assistant" and not last_k[-1].get("tool_calls"):
                last_k = last_k[:-1]
            h1 = [{**m, "_context": ctx} if m["role"] == "user" else m for m in last_k]
            h2 = [{**m, "_context": ctx2 if i == j2 else ctx} if m["role"] == "user" else m for i, m in enumerate(msgs[:j2 + 1])]
            turns += 1
            for layout in LAYOUT_ORDER:
                a, b = _render(tok, h1, TOOLS, layout), _render(tok, h2, TOOLS, layout)
                sums[layout][0] += _shared(a, b)
                sums[layout][1] += len(b)
    return {"conversations": len(rows), "turn_changes": turns,
            "rounds": {k: {"round1_tokens": v[1], "reused_by_round2": v[0], "share": round(v[0] / max(1, v[1]), 4)}
                       for k, v in rounds.items()},
            "turns": {k: {"reused_tokens": v[0], "prompt_tokens": v[1], "share": round(v[0] / max(1, v[1]), 4),
                          "recomputed_per_turn": round((v[1] - v[0]) / max(1, turns), 1)} for k, v in sums.items()}}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("src")
    c.add_argument("dst")
    c.add_argument("--layout", default="stable_prefix")
    s = sub.add_parser("sets")
    s.add_argument("--sets", default=",".join(SETS))
    s.add_argument("--layout", default="stable_prefix")
    k = sub.add_parser("check")
    k.add_argument("src")
    k.add_argument("dst")
    h = sub.add_parser("heard")
    h.add_argument("--layout", default="stable_prefix")
    h.add_argument("--write", action="store_true", help="also write <set>-sp.jsonl / -uc.jsonl / -cu.jsonl next to each heard set")
    o = sub.add_parser("oracle")
    o.add_argument("data", nargs="+")
    o.add_argument("--layout", default="stable_prefix")
    o.add_argument("--out", default="outputs/eval/oracle-layout")
    o.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("prefix")
    p.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    p.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "convert":
        convert(a.src, a.dst, a.layout)
    elif a.cmd == "sets":
        for name in a.sets.split(","):
            convert(f"data/generated/{name}.jsonl", f"data/generated/{name}{SUFFIX[a.layout]}.jsonl", a.layout)
    elif a.cmd == "check":
        check(a.src, a.dst)
    elif a.cmd == "heard":
        heard(layout=a.layout, write=a.write)
    elif a.cmd == "oracle":
        reps = oracle(a.data, a.layout, a.out, a.limit)
        sys.exit(0 if all(r["pass"] == 1 and not r["wrong_layout_calls"] and not r["errors"] for r in reps) else 1)
    elif a.cmd == "prefix":
        rep = prefix_check(a.tokenizer)
        if a.json:
            json.dump(rep, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
