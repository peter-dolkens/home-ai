"""Offline oracle check that heard sets drop into the outcome eval unchanged.

For every heard row: everything but the user turns' content must equal the source row (same reference calls, answers,
must_mention), the written text must sit in "_written", and replaying the reference through homeai.eval.e2e's own
apply_ref / compare / answer_ok (a perfect model) must pass every turn. No model, no Ollama.

  python3 -m homeai.voice.check data/generated/heard/wild-small-clean.jsonl [more.jsonl ...]
"""
from __future__ import annotations

import copy
import json
import sys

from ..eval.e2e import answer_ok, apply_ref, compare
from . import vocab
from .roundtrip import SETS, load_set


def strip(ex: dict) -> dict:
    """The row as the evaluator's reference sees it: user contents and round-trip metadata removed."""
    ex = copy.deepcopy(ex)
    for k in ("_voice", "_set", "_written", "query", "_confusion"):
        ex.pop(k, None)
    for m in ex.get("messages", []):
        if m["role"] == "user":
            m.pop("_written", None)
            m["content"] = None
    return ex


def oracle(ex: dict) -> list[str]:
    """Problems found replaying the reference as a perfect model would; [] when every turn passes."""
    msgs, bad = ex["messages"], []
    user_idx = [j for j, m in enumerate(msgs) if m["role"] == "user"]
    for k, j in enumerate(user_idx):
        end = user_idx[k + 1] if k + 1 < len(user_idx) else len(msgs)
        seg = msgs[j + 1:end]
        before = vocab.house_of(ex)
        apply_ref(before, msgs[:j])
        ref_after, got = copy.deepcopy(before), copy.deepcopy(before)
        apply_ref(ref_after, seg)
        apply_ref(got, seg)
        ok, why = compare(ref_after, got, msgs[j].get("_written", msgs[j]["content"]))
        ref_ans = next((m for m in reversed(seg) if m["role"] == "assistant" and m.get("content")), {})
        must = (ref_ans.get("_eval") or {}).get("must_mention", [])
        acts = any(tc["function"]["name"] == "call_service" for m in seg for tc in (m.get("tool_calls") or []))
        a_ok, a_why = answer_ok(ref_ans.get("content", ""), [] if acts else must, ex["category"]) if ref_ans else (True, "")
        if not ok or not a_ok:
            bad.append(f"turn {k}: {why or a_why}")
    return bad


def check(path: str) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    base = path.rsplit("/", 1)[-1]
    stem = max([s for s in list(SETS) + ["probe"] if base.startswith(s + "-")], key=len, default=None)
    src = {r["id"]: r for r in load_set(stem, ".")} if stem else {}
    res = {"rows": len(rows), "turns": 0, "changed_turns": 0, "reference_mismatch": 0, "missing_written": 0,
           "oracle_fail": 0, "examples": []}
    for ex in rows:
        if "messages" not in ex:
            continue
        users = [m for m in ex["messages"] if m["role"] == "user"]
        res["turns"] += len(users)
        res["missing_written"] += sum("_written" not in m for m in users)
        res["changed_turns"] += sum(m.get("_written") != m["content"] for m in users)
        s = src.get(ex["id"])
        if s is not None and strip(ex) != strip(s):
            res["reference_mismatch"] += 1
        if s is not None and [m["content"] for m in s["messages"] if m["role"] == "user"] != [m.get("_written") for m in users]:
            res["reference_mismatch"] += 1
        bad = oracle(ex)
        if bad:
            res["oracle_fail"] += 1
            res["examples"].append({"id": ex["id"], "problems": bad})
    res["examples"] = res["examples"][:10]
    return res


def main():
    ok = True
    for p in sys.argv[1:]:
        r = check(p)
        ok &= not (r["reference_mismatch"] or r["missing_written"] or r["oracle_fail"])
        print(p, json.dumps(r))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
