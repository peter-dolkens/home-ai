"""Room-leaf data (Needle format: query / tools / answers / system) as chat examples for the Qwen training stack
(homeai/train), so a small general model can be fine-tuned as the room leaf and served by Ollama.

Each example keeps its own room's tools; the system turn carries the room fact plus the leaf rule the Ollama leaf eval
uses (homeai.leaf.eval.LEAF_RULE). Negatives (answers []) become a short spoken decline with no tool call.

    PYTHONPATH=. python3 -m homeai.leaf.chat data/generated/leaf-generic-train.jsonl data/generated/leaf-generic-chat-train.jsonl
"""
from __future__ import annotations

import json
import random
import sys

from .eval import LEAF_RULE

DECLINES = ["That's not something I can do in this room.", "That's not in this room.", "Nothing here for that.",
            "I can't do that from this room."]


def to_chat(ex: dict, rng: random.Random) -> dict:
    system = "\n".join(x for x in (ex.get("system"), LEAF_RULE) if x)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": ex["query"]}]
    if ex.get("answers"):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {"name": a["name"], "arguments": a.get("arguments") or {}}}
            for i, a in enumerate(ex["answers"])]})
    else:
        msgs.append({"role": "assistant", "content": rng.choice(DECLINES)})
    return {"messages": msgs, "tools": ex["tools"], "category": "leaf_negative" if not ex.get("answers") else "leaf"}


def main():
    src, dst = sys.argv[1], sys.argv[2]
    rng = random.Random(5)
    rows = [json.loads(l) for l in open(src) if l.strip()]
    with open(dst, "w") as f:
        for ex in rows:
            f.write(json.dumps(to_chat(ex, rng), ensure_ascii=False) + "\n")
    print(f"{dst}: {len(rows)} examples")


if __name__ == "__main__":
    main()
