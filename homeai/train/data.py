"""Turn generated JSONL into chat-template text. Kept free of GPU deps so it can
be dry-run on any machine with `transformers` installed."""
from __future__ import annotations

import json
from typing import Any

from ..tools import TOOLS
from ..tools_b import for_toolset


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip eval-only keys; keep exactly what the chat template expects."""
    out = []
    for m in messages:
        n = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls"):
            n["tool_calls"] = [{"id": tc.get("id", f"call_{i}"), "type": "function",
                                "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                               for i, tc in enumerate(m["tool_calls"])]
        if m["role"] == "tool":
            n["tool_call_id"] = m.get("tool_call_id", "")
            n["name"] = m.get("name", "")
        out.append(n)
    return out


def text_tokenizer(tokenizer):
    """Vision-language checkpoints (Qwen3.5 small models) load as a processor; training is
    text-only, so use the inner tokenizer everywhere."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def render(tokenizer, example: dict[str, Any]) -> str:
    if example.get("tools"):  # room-leaf examples carry their own room's tool schemas (homeai/leaf/chat.py)
        tools = [t if "function" in t else {"type": "function", "function": t} for t in example["tools"]]
    else:
        # design B examples carry "toolset": "b" and train on B's schemas; orchestrator examples "orch" (B's + ask_rooms)
        tools = for_toolset(example.get("toolset"))[0]
    return text_tokenizer(tokenizer).apply_chat_template(clean_messages(example["messages"]), tools=tools, tokenize=False, add_generation_prompt=False)


def render_all(tokenizer, examples, max_len: int):
    tok = text_tokenizer(tokenizer)
    texts, dropped, lengths = [], 0, []
    for ex in examples:
        t = render(tok, ex)
        n = len(tok(text=t, add_special_tokens=False)["input_ids"])
        lengths.append(n)
        if n > max_len:
            dropped += 1
            continue
        texts.append({"text": t, "category": ex["category"]})
    return texts, dropped, lengths
