"""Offline checks for the MLX trainer's recipe pieces (homeai/train/train_mlx.py), no MLX or transformers needed: the
Unsloth train_on_responses_only label rule (assistant turns up to and including <|im_end|>, tool results and the "\\n"
after <|im_end|> masked), mask_spans with keep_open, the transformers cosine-with-warmup schedule, the seeded step
windows (every example once per epoch).

    PYTHONPATH=. python3 tests/test_train_mlx.py

On frogmouth the labels were also compared with unsloth_zoo 2026.9.7's own function on 1,193 rendered examples of
train-v5, train-orch and leaf-generic-chat-train: identical."""
from __future__ import annotations

import re

from homeai.train.train_mlx import Labeller, epoch_windows, schedule

SPECIAL = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]


class _Added:
    def __init__(self, content):
        self.content, self.special = content, True


class _Tok:
    """ChatML-ish toy tokenizer: special markers are single ids, every other character is its own id."""
    bos_token_id, eos_token_id = None, 1
    all_special_ids = [0, 1, 2]
    added_tokens_decoder = {i: _Added(s) for i, s in enumerate(SPECIAL)}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for part in re.split("(" + "|".join(map(re.escape, SPECIAL)) + ")", text):
            ids += [SPECIAL.index(part)] if part in SPECIAL else [100 + ord(c) for c in part]
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(SPECIAL[i] if i < 3 else chr(i - 100) for i in ids)


CFG = {"instruction_part": "<|im_start|>user\n", "response_part": "<|im_start|>assistant\n"}
TEXT = ("<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nlights?<|im_end|>\n<|im_start|>assistant\ncall<|im_end|>\n"
        "<|im_start|>user\n<tool_response>on</tool_response><|im_end|>\n<|im_start|>assistant\nthey're on<|im_end|>\n")


def _labelled(cfg, text=TEXT):
    tok = _Tok()
    ids = tok(text)["input_ids"]
    labels = Labeller(tok, cfg)(ids)
    spans, cur = [], []
    for i, l in zip(ids, labels):
        if l != -100:
            cur.append(i)
        elif cur:
            spans.append(tok.decode(cur))
            cur = []
    return spans + ([tok.decode(cur)] if cur else [])


def test_response_labels():
    assert _labelled(CFG) == ["call<|im_end|>", "they're on<|im_end|>"]
    # an assistant turn cut off without its EOS is labelled to the end
    assert _labelled(CFG, TEXT[:-len("<|im_end|>\n")]) == ["call<|im_end|>", "they're on"]


def test_mask_spans_keep_open():
    text = "<|im_start|>assistant\n<tool_call>x</tool_call><tool_response>on</tool_response>done<|im_end|>\n"
    cfg = dict(CFG, mask_spans=[["<tool_response>", "</tool_response>"]])
    assert _labelled(cfg, text) == ["<tool_call>x</tool_call>", "done<|im_end|>"]
    cfg = dict(CFG, mask_spans=[["<tool_response>", "</tool_response>", "keep_open"]])
    assert _labelled(cfg, text) == ["<tool_call>x</tool_call><tool_response>", "done<|im_end|>"]


def test_schedule():
    cfg = {"batch_size": 1, "grad_accum": 16, "epochs": 2, "lr": 2e-4}
    total, warmup, lr = schedule(cfg, 7487)
    assert (total, warmup) == (936, 47)  # ceil(7487 / 16) = 468 steps per epoch, ceil(5%) warmup
    assert lr(0) == 0.0 and abs(lr(47) - 2e-4) < 1e-12 and lr(46) < lr(47) and abs(lr(total) - 0.0) < 1e-12


def test_windows():
    cfg = {"batch_size": 2, "grad_accum": 3, "seed": 3407}
    ws = epoch_windows(13, cfg, 0)
    assert [len(w) for w in ws] == [3, 3, 1] and sorted(i for w in ws for mb in w for i in mb) == list(range(13))
    assert ws == epoch_windows(13, cfg, 0) and ws != epoch_windows(13, cfg, 1)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
