"""Turn generated JSONL into chat-template text. Kept free of GPU deps so it can
be dry-run on any machine with `transformers` installed."""
from __future__ import annotations

import json
from typing import Any, Optional

from ..prompt_layout import apply as apply_layout
from ..tools import TOOLS
from ..tools_b import for_toolset


def load_jsonl(path: str, layout: Optional[str] = None) -> list[dict[str, Any]]:
    """layout: the config's "prompt_layout" (v10 "stable_prefix", v11 "user_context", v12 "clock_in_user"): every row's
    conversation rendered in it (homeai/house/snapshot.py relayout_messages; user_context adds each user turn's context
    line here, clock_in_user its clock line, so the files keep bare utterances), and the train and the val file train alike whichever layout they were stored in.
    None (every config before v10) leaves the rows as stored."""
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return [apply_layout(r, layout) for r in rows] if layout else rows


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


def check_tool_calls_rendered(tokenizer, examples, sample: int = 20) -> None:
    """Fail fast when the chat template silently drops assistant tool calls (an old LFM2.5 1.2B template did, and
    the model learned to answer with nothing)."""
    with_calls = [e for e in examples if any(m.get("tool_calls") for m in e["messages"])][:sample]
    for ex in with_calls:
        text = render(tokenizer, ex)
        names = {tc["function"]["name"] for m in ex["messages"] for tc in m.get("tool_calls") or []}
        # the tool list in the system prompt names every tool, so look after the first user turn
        user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        tail = text[text.find(user) + len(user):] if user and user in text else text
        missing = [n for n in names if n not in tail]
        if missing:
            raise SystemExit(f"chat template drops tool calls ({missing} not rendered in {ex.get('id')}); "
                             "set chat_template_from in the config")


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
