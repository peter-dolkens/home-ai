"""Token counts for Needle 3 examples, rendered the way the engine puts them on the wire (cactuscompute.com/blog/porting-needle):

    <s><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<tools>{tools, minified JSON}</tools>\n{query}<|im_end|>\n
    <|im_start|>assistant\n<think>{reasoning}</think><tool_call>{answers, minified JSON}</tool_call><|im_end|>

The tokenizer is Needle's own SentencePiece BPE (8192 pieces with merge scores, byte fallback), vendored from
huggingface.co/Cactus-Compute/needle3 (Apache-2.0) as data/needle/tokenizer.vocab. Encoding here is a re-implementation
(whitespace pre-split, highest-score merge first), so counts can differ from the engine's by a token or two per word
boundary; the fine-tune cap (--max-len 1024) is checked with a margin for that."""
from __future__ import annotations

import functools
import json
import os
import re

VOCAB = os.path.join(os.path.dirname(__file__), "..", "..", "data", "needle", "tokenizer.vocab")
SPECIAL = re.compile(r"(<s>|</s>|<\|im_start\|>|<\|im_end\|>|<think>|</think>|<tools>|</tools>|<tool_call>|</tool_call>)")
MAX_LEN = 1024


@functools.lru_cache(maxsize=1)
def _vocab() -> dict[str, float]:
    out = {}
    with open(VOCAB, encoding="utf-8") as f:
        for line in f:
            piece, _, score = line.rstrip("\n").partition("\t")
            out[piece] = float(score or 0)
    return out


@functools.lru_cache(maxsize=200_000)
def _word(w: str) -> int:
    """Tokens for one whitespace-delimited chunk (leading '▁' included)."""
    v = _vocab()
    parts = list(w)
    while len(parts) > 1:
        best, at = None, -1
        for i in range(len(parts) - 1):
            s = v.get(parts[i] + parts[i + 1])
            if s is not None and (best is None or s > best):
                best, at = s, i
        if at < 0:
            break
        parts[at:at + 2] = [parts[at] + parts[at + 1]]
    return sum(1 if p in v else len(p.encode("utf-8")) for p in parts)  # byte fallback for unknown pieces


def count(text: str) -> int:
    n = 0
    for seg in SPECIAL.split(text):
        if not seg:
            continue
        if SPECIAL.fullmatch(seg):
            n += 1
            continue
        for line_i, line in enumerate(seg.split("\n")):
            if line_i:
                n += 1  # <0x0A>
            for w in re.findall(r"▁?[^▁]+|▁", line.replace(" ", "▁")):
                n += _word(w)
    return n


def minified(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def render(ex: dict, tools: list | None = None) -> str:
    """One training example on the wire. tools: override the example's tool list (e.g. the retrieved five)."""
    tools = ex["tools"] if tools is None else tools
    sys_part = f"<|im_start|>system\n{ex['system']}<|im_end|>\n" if ex.get("system") else ""
    return (f"<s>{sys_part}<|im_start|>user\n<tools>{minified(tools)}</tools>\n{ex['query']}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>{ex.get('reasoning', '')}</think><tool_call>{minified(ex['answers'])}</tool_call><|im_end|>")


def example_tokens(ex: dict, tools: list | None = None) -> int:
    return count(render(ex, tools))


def tools_tokens(tools: list) -> int:
    return count(minified(tools))
