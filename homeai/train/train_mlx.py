"""LoRA fine-tune on Apple Silicon with MLX (frogmouth M5, falcon M2 Pro): the Unsloth recipe of homeai/train/train.py,
driven by the same config JSON files. Run everything through scripts/mlx-env.sh (venv and caches under .mlx/).

    scripts/mlx-env.sh python -m homeai.train.train_mlx --config configs/qwen35-2b-lora-v5.json
    scripts/mlx-env.sh python -m homeai.train.train_mlx --config configs/qwen35-2b-lora-v5.json --max-steps 50 --set out=outputs/x
    scripts/mlx-env.sh python -m homeai.train.train_mlx --config configs/qwen35-2b-lora-v5.json --fuse   # -> <out>/merged16
    scripts/mlx-env.sh python -m homeai.train.train_mlx --config configs/qwen35-2b-lora-v5.json --dry-run  # tokens, labels, steps

Same as train.py (Unsloth + TRL SFTTrainer):
  text       render_all() from data.py (the model's chat template, per-example tools or the toolset); examples over
             max_seq_len are dropped, no packing, tokenised as SFTTrainer does (no special tokens unless the template
             lacks the BOS)
  labels     unsloth_zoo train_on_responses_only (2026.9): every assistant turn from after response_part up to and
             including the tokenizer EOS (<|im_end|>), never past the next <|im_start|>; then cfg mask_spans
             (with "keep_open") exactly as train.py applies them
  LoRA       rank, alpha (scale alpha / r), dropout 0, target_modules matched by module name as PEFT does; for Qwen3.5
             that is q/k/v/o in the full-attention layers and gate/up/down in every layer (the Gated DeltaNet projections
             have other names and stay frozen, as on the CUDA boxes)
  optimiser  AdamW (bias-corrected, betas 0.9/0.999, eps 1e-8, weight decay 0.01), grad-norm clip 1.0, lr warmup over
             ceil(5%) of the steps from 0 then cosine to 0 (get_cosine_schedule_with_warmup); one step is
             batch_size x grad_accum examples and its loss the token-weighted mean over all of them (num_items_in_batch;
             run one example per forward pass, which is the same sum without padding)
  val        mean per-example loss over the first val_max_examples val examples <= val_max_tokens, after every epoch
Deviations: AdamW moments in fp32 (bitsandbytes adamw_8bit on CUDA); the data order is a random.Random shuffle of the seed
(torch's RandomSampler there); logits are computed only where a label follows (the loss is the same); and the Gated
DeltaNet layers train through a chunked form of the delta rule below, because mlx-lm has only a forward Metal kernel
and a per-token loop for them.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import shutil
import statistics
import sys
import time

from ..host import host_info
from .data import load_jsonl, render_all, text_tokenizer
from .train import load_config

GB = 1024 ** 3


# ---- tokens and labels (no MLX needed: --dry-run and the label checks run anywhere with transformers) ----

def find(seq, sub, start):
    for i in range(start, len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1


class Labeller:
    """input_ids -> labels, as Unsloth's train_on_responses_only (force_match=True) followed by train.py's mask_spans."""

    def __init__(self, tok, cfg):
        enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
        self.resp, self.instr = enc(cfg["response_part"]), enc(cfg["instruction_part"])
        # the shared special opener of both markers (<|im_start|>) delimits every message, tool and system turns too
        start = []
        for q, a in zip(self.instr, self.resp):
            if q != a:
                break
            start.append(q)
        added = getattr(tok, "added_tokens_decoder", None) or {}
        special = {i for i, t in added.items() if getattr(t, "special", False)} | set(getattr(tok, "all_special_ids", None) or [])
        if not all(i in special for i in start) or not any(getattr(added.get(i), "content", "").strip() for i in start):
            start = []
        self.start, self.bos, self.eos = start, tok.bos_token_id, tok.eos_token_id
        self.spans = [(enc(sp[0]), enc(sp[1]), len(sp) > 2 and sp[2] == "keep_open") for sp in cfg.get("mask_spans") or []]

    def __call__(self, ids: list[int]) -> list[int]:
        n, resp, instr, start = len(ids), self.resp, self.instr, self.start
        at = lambda j, sub: ids[j:j + len(sub)] == sub  # noqa: E731
        labels = [-100] * n
        j = 0
        while j < n:
            if ids[j] == resp[0] and at(j, resp):
                a = j = j + len(resp)
                while j < n:
                    t = ids[j]
                    if t == self.eos:  # keep the assistant's EOS
                        labels[a:j + 1] = ids[a:j + 1]
                        break
                    if t == self.bos or (start and at(j, start)) or at(j, resp):
                        labels[a:j] = ids[a:j]
                        j -= 1  # revisit the boundary so a following assistant turn is found
                        break
                    if j == n - 1 or at(j, instr):
                        end = n if j == n - 1 else j
                        labels[a:end] = ids[a:end]
                        if j != n - 1:
                            j += len(instr)
                        break
                    j += 1
            j += 1
        for o, c, keep_open in self.spans:
            i = find(ids, o, 0)
            while i >= 0:
                k = find(ids, c, i + len(o))
                end = len(ids) if k < 0 else k + len(c)
                s = i + len(o) if keep_open else i
                labels[s:end] = [-100] * (end - s)
                i = find(ids, o, end)
        return labels


def encode(tok, examples, cfg, max_len, labeller=None):
    """render_all() texts -> [{"ids", "labels"}], plus the drop count and token lengths render_all reports."""
    tok = text_tokenizer(tok)
    texts, dropped, lengths = render_all(tok, examples, max_len)
    # SFTTrainer adds special tokens unless the rendered text / template already carries the BOS (unsloth_zoo sft_prepare_dataset)
    bos = tok.bos_token
    add_special = not (bos and (bool(texts) and texts[0]["text"].startswith(bos) or bos in (tok.chat_template or "")))
    labeller = labeller or Labeller(tok, cfg)
    rows = []
    for t in texts:
        ids = tok(t["text"], add_special_tokens=add_special)["input_ids"]
        rows.append({"ids": ids, "labels": labeller(ids), "category": t["category"]})
    return rows, dropped, lengths


def schedule(cfg, n_train):
    """(optimizer steps, warmup steps, lr(step)) as transformers' Trainer computes them for this config."""
    batches = math.ceil(n_train / cfg["batch_size"])
    per_epoch = math.ceil(batches / cfg["grad_accum"])
    total = per_epoch * int(cfg["epochs"])
    warmup = math.ceil(total * cfg.get("warmup_ratio", 0.05))

    def lr(step):  # 0-based optimizer step; the first step runs at lr 0 exactly as get_cosine_schedule_with_warmup
        if step < warmup:
            return cfg["lr"] * step / max(1, warmup)
        return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return total, warmup, lr


def epoch_windows(n, cfg, epoch):
    """Optimizer steps of one epoch: lists of micro-batches (lists of example indices), shuffled from the seed."""
    order = list(range(n))
    random.Random(cfg["seed"] + epoch).shuffle(order)
    bs, ga = cfg["batch_size"], cfg["grad_accum"]
    batches = [order[i:i + bs] for i in range(0, n, bs)]
    return [batches[i:i + ga] for i in range(0, len(batches), ga)]


# ---- MLX model pieces ----

def _inv_unit_lower(L):
    """(I + L)^-1 for strictly lower-triangular L (..., n, n), n a power of two, by block substitution: batched matmuls
    with a gradient, unlike mx.linalg.solve_triangular (CPU only, no vjp)."""
    import mlx.core as mx
    n = L.shape[-1]
    if n == 1:
        return mx.ones_like(L)
    h = n // 2
    both = _inv_unit_lower(mx.stack([L[..., :h, :h], L[..., h:, h:]]))
    ai, di = both[0], both[1]
    c = -(di @ L[..., h:, :h] @ ai)
    return mx.concatenate([mx.concatenate([ai, mx.zeros_like(c)], -1), mx.concatenate([c, di], -1)], -2)


def chunked_gated_delta(q, k, v, log_g, beta, state=None, mask=None, chunk=64):
    """Gated delta rule over a whole sequence in chunks (the WY / UT form of transformers' torch_chunk_gated_delta_rule
    and FLA), differentiable and in fp32. Takes mlx-lm's layout: q, k (B, T, Hk, Dk) already normalised and scaled
    (GatedDeltaNet.__call__), v (B, T, Hv, Dv), log_g / beta (B, T, Hv), state (B, Hv, Dv, Dk). Returns y (B, T, Hv, Dv)
    in q's dtype and the final state. Memory is O(T / chunk) states instead of the per-token loop's O(T)."""
    import mlx.core as mx
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    dtype = q.dtype
    if Hv // Hk > 1:
        q, k = mx.repeat(q, Hv // Hk, -2), mx.repeat(k, Hv // Hk, -2)
    if mask is not None:  # padded steps leave the state alone
        beta, log_g = mx.where(mask[..., None], beta, 0), mx.where(mask[..., None], log_g, 0)
    pad = -T % chunk

    def chunked(x):  # (B, T, H, ...) -> (B, H, n, chunk, ...) in fp32
        x = x.astype(mx.float32).swapaxes(1, 2)
        if pad:
            x = mx.pad(x, [(0, 0), (0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 3))
        return x.reshape(B, Hv, -1, chunk, *x.shape[3:])
    q, k, v, beta, g = (chunked(x) for x in (q, k, v, beta, log_g))
    n = q.shape[2]
    kb, vb = k * beta[..., None], v * beta[..., None]
    cum = mx.cumsum(g, axis=-1)  # log decay from the chunk start
    tril = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
    decay = mx.where(tril, mx.exp(mx.where(tril, cum[..., :, None] - cum[..., None, :], 0)), 0)  # no exp of positives
    kk = kb @ k.swapaxes(-1, -2) * decay
    inv = _inv_unit_lower(mx.where(mx.tril(tril, -1), kk, 0))
    u = inv @ vb  # values with the within-chunk delta corrections
    w = inv @ (kb * mx.exp(cum)[..., None])  # decayed keys reading the state carried in
    attn = q @ k.swapaxes(-1, -2) * decay
    qd = q * mx.exp(cum)[..., None]
    kd = k * mx.exp(cum[..., -1:] - cum)[..., None]
    cd = mx.exp(cum[..., -1])[..., None, None]
    s = mx.zeros((B, Hv, Dk, Dv), mx.float32) if state is None else state.astype(mx.float32).swapaxes(-1, -2)
    y, s = _delta_scan()(u, w, qd, attn, kd, cd, s)
    y = y.reshape(B, Hv, -1, Dv)[:, :, :T]
    return y.swapaxes(1, 2).astype(dtype), s.swapaxes(-1, -2)


def _scan_forward(u, w, qd, attn, kd, cd, s):
    """Sequential part over chunks i (S the state entering chunk i):
    vn = u - w S;  y = qd S + attn vn;  S' = cd S + kd^T vn.  Returns per-chunk y, vn and entering states."""
    import mlx.core as mx
    parts = [[x.squeeze(2) for x in mx.split(t, t.shape[2], axis=2)] for t in (u, w, qd, attn, kd, cd)]
    ys, vns, states = [], [], []
    for ui, wi, qi, ai, ki, ci in zip(*parts):
        states.append(s)
        vn = ui - wi @ s
        ys.append(qi @ s + ai @ vn)
        vns.append(vn)
        s = s * ci + ki.swapaxes(-1, -2) @ vn
    return parts, ys, vns, states, s


_SCAN = None


def _delta_scan():
    """The chunk scan with a hand-written vjp. Autodiff through the loop would take one slice per chunk and each
    slice's vjp scatters into a full-size zero array: quadratic in T (0.5 s per layer per 4.6k tokens on the M5)."""
    global _SCAN
    if _SCAN is not None:
        return _SCAN
    import mlx.core as mx

    @mx.custom_function
    def scan(u, w, qd, attn, kd, cd, s0):
        _, ys, _, _, s = _scan_forward(u, w, qd, attn, kd, cd, s0)
        return mx.stack(ys, axis=2), s

    @scan.vjp
    def scan_vjp(primals, cotangents, outputs):
        (u_, w_, qd_, attn_, kd_, cd_), _, vns, states, _ = _scan_forward(*primals)
        dy, ds = cotangents
        dys = [x.squeeze(2) for x in mx.split(dy, dy.shape[2], axis=2)]
        ds = mx.zeros_like(primals[-1]) if ds is None else ds
        g = {k: [] for k in ("u", "w", "qd", "attn", "kd", "cd")}
        for i in reversed(range(len(states))):
            S, vn, dyi = states[i], vns[i], dys[i]
            dvn = attn_[i].swapaxes(-1, -2) @ dyi + kd_[i] @ ds
            g["attn"].append(dyi @ vn.swapaxes(-1, -2))
            g["qd"].append(dyi @ S.swapaxes(-1, -2))
            g["kd"].append(vn @ ds.swapaxes(-1, -2))
            g["cd"].append((ds * S).sum(axis=(-2, -1), keepdims=True))
            g["u"].append(dvn)
            g["w"].append(-(dvn @ S.swapaxes(-1, -2)))
            ds = cd_[i] * ds + qd_[i].swapaxes(-1, -2) @ dyi - w_[i].swapaxes(-1, -2) @ dvn
        return tuple(mx.stack(g[k][::-1], axis=2) for k in ("u", "w", "qd", "attn", "kd", "cd")) + (ds,)
    _SCAN = scan
    return scan


def patch_gated_delta():
    """Qwen3.5's linear-attention layers call gated_delta_update(use_kernel=not self.training): in training that is a
    per-token Python loop keeping every step's state for the backward pass (6k tokens x 1 MB x 18 layers). Route the
    training path to chunked_gated_delta; inference keeps the Metal kernel."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5
    if getattr(qwen3_5, "_homeai_chunked", False):
        return
    orig = qwen3_5.gated_delta_update

    def update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
        if use_kernel:
            return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
        # compute_g in log space, softplus in fp32 as transformers does
        log_g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a.astype(mx.float32) + dt_bias.astype(mx.float32))
        return chunked_gated_delta(q, k, v, log_g, mx.sigmoid(b), state, mask)
    qwen3_5.gated_delta_update = update
    qwen3_5._homeai_chunked = True


def split_model(model):
    """(backbone, head): backbone(ids) -> final-norm hidden states, head(h) -> logits. mlx-lm keeps the decoder in
    .model (LFM2, Qwen3) or .language_model.model (Qwen3.5 loads as a vision-language checkpoint)."""
    lm = getattr(model, "language_model", model)
    inner = lm.model
    head = lm.lm_head if "lm_head" in lm else inner.embed_tokens.as_linear
    return inner, head


def lora_keys(model, targets):
    """Per-layer module paths whose last name is in targets (PEFT's target_modules name match)."""
    import mlx.nn as nn
    keys = set()
    for layer in model.layers:
        for path, m in layer.named_modules():
            if isinstance(m, (nn.Linear, nn.QuantizedLinear)) and path.split(".")[-1] in targets:
                keys.add(path)
    return sorted(keys)


def base_path(cfg):
    from huggingface_hub import snapshot_download
    return snapshot_download(cfg["base_model"])


def load_lora_model(cfg, train=True):
    """Base model in bf16 (4-bit when load_in_4bit: QLoRA via MLX quantisation), frozen, with the config's LoRA."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import load_model
    from pathlib import Path
    patch_gated_delta()
    model, _ = load_model(Path(base_path(cfg)))
    if cfg.get("load_in_4bit", True):
        nn.quantize(model, group_size=64, bits=4, class_predicate=lambda p, m: isinstance(m, nn.Linear) and ".layers." in p)
    model.freeze()
    targets = cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    keys = lora_keys(model, targets)
    mx.random.seed(cfg["seed"])
    linear_to_lora_layers(model, len(model.layers), {"rank": cfg["lora_r"], "scale": cfg["lora_alpha"] / cfg["lora_r"],
                                                     "dropout": 0.0, "keys": keys})
    model.train(train)
    return model, keys


def checkpoint_layers(layer, above=0):
    """mlx-lm's grad_checkpoint (recompute each decoder layer in the backward pass), but only for sequences longer than
    `above` tokens: short ones keep their activations and skip the recompute (about a third of the step)."""
    import mlx.core as mx
    cls = type(layer)
    fn = getattr(cls, "_homeai_call", cls.__call__)
    cls._homeai_call = fn

    def call(model, x, *args, **kwargs):
        if x.shape[1] <= above:
            return fn(model, x, *args, **kwargs)

        def inner(params, x, *args, **kwargs):
            model.update(params)
            return fn(model, x, *args, **kwargs)
        return mx.checkpoint(inner)(model.trainable_parameters(), x, *args, **kwargs)
    cls.__call__ = call


def batch_arrays(rows):
    """Right-padded ids (B, T) plus the flat positions whose next token is labelled and those targets. Right padding
    needs no mask here: causal attention and the recurrent layers never look ahead, and pads carry no label."""
    import mlx.core as mx
    import numpy as np
    T = max(len(r["ids"]) for r in rows)
    ids = np.zeros((len(rows), T), np.int32)
    pos, tgt = [], []
    for b, r in enumerate(rows):
        ids[b, :len(r["ids"])] = r["ids"]
        lab = np.asarray(r["labels"][1:])
        p = np.nonzero(lab != -100)[0]
        pos.append(b * T + p)
        tgt.append(lab[p])
    return mx.array(ids), mx.array(np.concatenate(pos).astype(np.int32)), mx.array(np.concatenate(tgt).astype(np.int32))


def make_loss(backbone, head, chunk=1024):
    """Summed cross entropy over labelled positions. Logits (vocab 248k) are formed only there and chunk-wise with
    recomputation, so a long assistant span never materialises seq x vocab in fp32."""
    import mlx.core as mx
    import mlx.nn as nn

    @mx.checkpoint
    def ce(h, t):  # checkpoint differentiates every input: the integer targets need a stop_gradient
        return nn.losses.cross_entropy(head(h).astype(mx.float32), mx.stop_gradient(t), reduction="sum")

    def loss(model, ids, pos, tgt):
        h = backbone(ids)
        h = h.reshape(-1, h.shape[-1])[pos]
        return sum(ce(h[i:i + chunk], tgt[i:i + chunk]) for i in range(0, h.shape[0], chunk))
    return loss


# ---- adapters on disk ----

def save_adapters(model, cfg, keys, path):
    """MLX adapter (adapters.safetensors + adapter_config.json, mlx_lm.load(adapter_path=...) reads it) and the same
    weights under PEFT names (adapter_model.safetensors), so the CUDA boxes' merge / PEFT tooling can load it too."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    os.makedirs(path, exist_ok=True)
    w = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(os.path.join(path, "adapters.safetensors"), w)
    json.dump({"fine_tune_type": "lora", "num_layers": len(model.layers), "model": cfg["base_model"],
               "lora_parameters": {"rank": cfg["lora_r"], "scale": cfg["lora_alpha"] / cfg["lora_r"], "dropout": 0.0, "keys": keys}},
              open(os.path.join(path, "adapter_config.json"), "w"), indent=2)
    names = hf_names(cfg, [k.rsplit(".", 1)[0] for k in w])
    peft = {}
    for k, v in w.items():
        mod, which = k.rsplit(".", 1)
        if mod in names:
            peft[f"base_model.model.{names[mod]}.{'lora_A' if which == 'lora_a' else 'lora_B'}.weight"] = v.T.astype(mx.float32)
    mx.save_safetensors(os.path.join(path, "adapter_model.safetensors"), peft, metadata={"format": "pt"})
    targets = sorted({k.split(".")[-1] for k in keys})
    json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": cfg["base_model"], "r": cfg["lora_r"],
               "lora_alpha": cfg["lora_alpha"], "lora_dropout": 0.0, "bias": "none", "target_modules": targets, "use_rslora": False,
               "fan_in_fan_out": False, "init_lora_weights": True}, open(os.path.join(path, "adapter_config.peft.json"), "w"), indent=2)


def hf_names(cfg, mlx_modules):
    """mlx module path -> checkpoint module path, by running the model's own sanitize() on each checkpoint key."""
    import mlx.core as mx
    from mlx_lm.utils import load_config as mlx_config, _get_classes
    src = base_path(cfg)
    conf = mlx_config(__import__("pathlib").Path(src))
    model_cls, args_cls = _get_classes(conf)
    probe = model_cls(args_cls.from_dict(conf))
    idx = os.path.join(src, "model.safetensors.index.json")
    hf_keys = list(json.load(open(idx))["weight_map"]) if os.path.exists(idx) else \
        [k for f in glob.glob(os.path.join(src, "*.safetensors")) for k in mx.load(f)]
    want, out = set(mlx_modules), {}
    for hk in hf_keys:
        if not hk.endswith(".weight"):
            continue
        got = probe.sanitize({hk: mx.zeros((1,))}) if hasattr(probe, "sanitize") else {hk: None}
        for mk in got:
            if mk.endswith(".weight") and mk[:-7] in want:
                out[mk[:-7]] = hk[:-7]
    return out


def fuse(cfg, lora_dir=None, out=None):
    """Adapter + base checkpoint -> <out>/merged16: the base's own safetensors shards (vision tower and MTP included,
    as Unsloth's merged_16bit keeps them) with W + scale * B A on every adapted linear, merged in fp32 and saved in the
    base dtype, plus the base's config / tokenizer / template files. Loads in transformers and in `ollama create`."""
    import mlx.core as mx
    lora_dir = lora_dir or os.path.join(cfg["out"], "lora")
    out = out or os.path.join(cfg["out"], "merged16")
    src = base_path(cfg)
    ad = mx.load(os.path.join(lora_dir, "adapters.safetensors"))
    acfg = json.load(open(os.path.join(lora_dir, "adapter_config.json")))
    scale = acfg["lora_parameters"]["scale"]
    mods = sorted({k.rsplit(".", 1)[0] for k in ad})
    names = hf_names(cfg, mods)
    missing = [m for m in mods if m not in names]
    if missing:
        raise SystemExit(f"fuse: no checkpoint tensor for {missing[:5]}")
    delta_for = {names[m] + ".weight": m for m in mods}
    os.makedirs(out, exist_ok=True)
    done = 0
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        w, meta = mx.load(f, return_metadata=True)
        for hk, m in delta_for.items():
            if hk in w:
                a, b = ad[m + ".lora_a"].astype(mx.float32), ad[m + ".lora_b"].astype(mx.float32)
                w[hk] = (w[hk].astype(mx.float32) + scale * (b.T @ a.T)).astype(w[hk].dtype)
                done += 1
        mx.save_safetensors(os.path.join(out, os.path.basename(f)), w, metadata={**(meta or {}), "format": "pt"})
        del w
    if done != len(delta_for):
        raise SystemExit(f"fuse: merged {done} of {len(delta_for)} adapted weights")
    for f in glob.glob(os.path.join(src, "*")):
        if os.path.isfile(f) and not f.endswith(".safetensors") and os.path.basename(f) not in ("README.md", "LICENSE"):
            shutil.copyfile(f, os.path.join(out, os.path.basename(f)))  # not copy(): HF cache blobs are read-only
    json.dump({"base_model": cfg["base_model"], "adapter": lora_dir, "merged_weights": done, "scale": scale, **host_info()},
              open(os.path.join(out, "homeai-merge.json"), "w"), indent=2)
    print(f"fused {done} weights -> {out}")
    return out


# ---- training ----

def evaluate(model, loss_fn, rows, bs=1):
    """Mean per-example loss (the Trainer's eval_loss with eval batch 1)."""
    import mlx.core as mx
    model.eval()
    losses = []
    for r in rows:
        ids, pos, tgt = batch_arrays([r])
        if tgt.size:
            losses.append((loss_fn(model, ids, pos, tgt) / tgt.size).item())
    model.train()
    return statistics.mean(losses) if losses else None


def train(cfg, max_steps=0, log_every=10):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from transformers import AutoTokenizer

    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    mx.set_memory_limit(int(cfg.get("mlx_memory_limit_gb", 16) * GB))
    mx.set_cache_limit(int(cfg.get("mlx_cache_limit_gb", 4) * GB))
    tok = text_tokenizer(AutoTokenizer.from_pretrained(cfg.get("hf_tokenizer", cfg["base_model"])))
    rows, dropped, lengths = encode(tok, load_jsonl(cfg["train"], cfg.get("prompt_layout")), cfg, cfg["max_seq_len"])
    print(f"train: {len(rows)} examples (dropped {dropped} over {cfg['max_seq_len']} tokens); "
          f"mean {statistics.mean(lengths):.0f} tokens, max {max(lengths)}")
    n_lab = sum(sum(1 for x in r["labels"] if x != -100) for r in rows[:50])
    print(f"label check: {n_lab} trained tokens in the first 50 examples")
    if n_lab == 0:
        raise SystemExit("no assistant tokens found: check response_part / instruction_part")
    val = []
    if cfg.get("val") and os.path.exists(cfg["val"]):
        vmax = cfg.get("val_max_tokens", 2048)
        val, vdrop, _ = encode(tok, load_jsonl(cfg["val"], cfg.get("prompt_layout")), cfg, vmax)
        val = val[: cfg.get("val_max_examples", 200)]
        print(f"val: {len(val)} examples (dropped {vdrop} over {vmax} tokens)")

    model, keys = load_lora_model(cfg)
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"LoRA r={cfg['lora_r']} alpha={cfg['lora_alpha']} on {len(keys)} module kinds {keys}: {n_train / 1e6:.2f}M trainable")
    if cfg.get("gradient_checkpointing", True):
        checkpoint_layers(model.layers[0], cfg.get("checkpoint_above", 0))
    backbone, head = split_model(model)
    loss_fn = make_loss(backbone, head)
    value_grad = nn.value_and_grad(model, loss_fn)
    total, warmup, lr_at = schedule(cfg, len(rows))
    opt = optim.AdamW(learning_rate=0.0, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01, bias_correction=True)
    print(f"steps: {total} ({cfg['epochs']} epochs x {math.ceil(math.ceil(len(rows) / cfg['batch_size']) / cfg['grad_accum'])}), "
          f"warmup {warmup}, effective batch {cfg['batch_size'] * cfg['grad_accum']}")

    ck_dir = os.path.join(out, "checkpoints")
    step = 0
    cks = sorted(glob.glob(os.path.join(ck_dir, "step-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
    if cks and cfg.get("resume", True) and not max_steps:
        model.load_weights(os.path.join(cks[-1], "adapters.safetensors"), strict=False)
        opt.init(model.trainable_parameters())
        opt.state = tree_unflatten(list(mx.load(os.path.join(cks[-1], "optimizer.safetensors")).items()))
        step = json.load(open(os.path.join(cks[-1], "state.json")))["step"]
        print(f"resuming from {cks[-1]} (step {step})")

    log_path = os.path.join(out, "train_log.jsonl")
    log = open(log_path, "a")
    log.write(json.dumps({"event": "start", "config": cfg, **host_info(), "train_examples": len(rows), "steps": total,
                          "warmup": warmup, "trainable_params": n_train, "lora_keys": keys, "resumed_at": step,
                          "mlx": mx.__version__, "time": time.strftime("%F %T")}) + "\n")
    log.flush()
    windows = []  # (epoch, fraction of the epoch done after this step, micro-batches)
    for e in range(int(cfg["epochs"])):
        ws = epoch_windows(len(rows), cfg, e)
        windows += [(e, (i + 1) / len(ws), w) for i, w in enumerate(ws)]
    stop = min(total, max_steps) if max_steps else total
    t_start, tok_seen, lab_seen, first = time.time(), 0, 0, step
    recent = []
    mx.reset_peak_memory()
    while step < stop:
        epoch, frac, window = windows[step]
        t0 = time.time()
        acc, loss_sum, n_tok, n_in = None, 0.0, 0, 0
        # one example per forward pass: the step's loss is a sum over all its tokens, so batch_size only fixes which
        # examples share a step; padded pairs measured slower in MLX than singles
        for i in (i for mb in window for i in mb):
            ids, pos, tgt = batch_arrays([rows[i]])
            loss, grads = value_grad(model, ids, pos, tgt)
            acc = grads if acc is None else tree_map(mx.add, acc, grads)
            mx.eval(loss, acc)
            loss_sum += loss.item()
            n_tok += tgt.size
            n_in += ids.size
        acc = tree_map(lambda g: g / max(1, n_tok), acc)  # token-weighted mean over the whole step
        acc, gnorm = optim.clip_grad_norm(acc, 1.0)
        lr = lr_at(step)
        opt.learning_rate = lr
        opt.update(model, acc)
        mx.eval(model.trainable_parameters(), opt.state, gnorm)
        step += 1
        dt = time.time() - t0
        tok_seen += n_in
        lab_seen += n_tok
        step_loss = loss_sum / max(1, n_tok)
        recent.append(step_loss)
        rec = {"step": step, "epoch": round(epoch + frac, 4),
               "loss": round(step_loss, 5), "lr": lr, "grad_norm": round(gnorm.item(), 4), "tokens": n_in, "trained_tokens": n_tok,
               "step_s": round(dt, 2), "tokens_per_s": round(n_in / dt, 1),
               "peak_gb": round(mx.get_peak_memory() / GB, 2), "active_gb": round(mx.get_active_memory() / GB, 2)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % log_every == 0 or step == 1 or step == stop:
            el = time.time() - t_start
            print(f"step {step}/{total} loss {statistics.mean(recent):.4f} lr {lr:.2e} gnorm {rec['grad_norm']:.3f} "
                  f"{tok_seen / el:.0f} tok/s peak {rec['peak_gb']:.1f} GB eta {(stop - step) * el / (step - first) / 3600:.2f} h",
                  flush=True)
            recent = []
        if step % cfg.get("save_steps", 100) == 0 and step < stop:
            d = os.path.join(ck_dir, f"step-{step}")
            save_adapters(model, cfg, keys, d)
            mx.save_safetensors(os.path.join(d, "optimizer.safetensors"), dict(tree_flatten(opt.state)))
            json.dump({"step": step}, open(os.path.join(d, "state.json"), "w"))
            for old in sorted(glob.glob(os.path.join(ck_dir, "step-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))[:-2]:
                shutil.rmtree(old)
        epoch_end = step == stop or windows[step][0] != epoch
        if epoch_end and val:
            vl = evaluate(model, loss_fn, val)
            print(f"epoch {epoch + 1}: eval_loss {vl:.4f}", flush=True)
            log.write(json.dumps({"event": "eval", "step": step, "epoch": epoch + 1, "eval_loss": vl}) + "\n")
            log.flush()
    el = time.time() - t_start
    by_step = {r["step"]: r for r in map(json.loads, open(log_path)) if "loss" in r}  # a resumed run may repeat steps
    losses = [by_step[k]["loss"] for k in sorted(by_step)]
    summary = {"event": "done", **host_info(), "steps": step, "train_loss": statistics.mean(losses) if losses else None,
               "train_seconds_all_sessions": round(sum(r["step_s"] for r in by_step.values()), 1),
               "train_seconds": round(el, 1), "tokens": tok_seen, "trained_tokens": lab_seen, "tokens_per_s": round(tok_seen / el, 1),
               "peak_gb": round(mx.get_peak_memory() / GB, 2), "time": time.strftime("%F %T")}
    log.write(json.dumps(summary) + "\n")
    log.close()
    lora_dir = os.path.join(out, "lora")
    save_adapters(model, cfg, keys, lora_dir)
    tok.save_pretrained(lora_dir)
    with open(os.path.join(out, "config.used.json"), "w") as f:
        json.dump({**cfg, "backend": "mlx", **host_info()}, f, indent=2)
    print(json.dumps(summary))
    print(f"saved LoRA adapter to {lora_dir}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override config keys, e.g. --set epochs=1 out=outputs/x")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (the schedule stays the full run's)")
    ap.add_argument("--dry-run", action="store_true", help="tokens, labels and the step schedule only (no MLX)")
    ap.add_argument("--fuse", action="store_true", help="merge <out>/lora into the base -> <out>/merged16 (bf16 safetensors)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if a.fuse:
        fuse(cfg)
        return 0
    if a.dry_run:
        from transformers import AutoTokenizer
        tok = text_tokenizer(AutoTokenizer.from_pretrained(cfg.get("hf_tokenizer", cfg["base_model"])))
        rows, dropped, lengths = encode(tok, load_jsonl(cfg["train"], cfg.get("prompt_layout")), cfg, cfg["max_seq_len"])
        total, warmup, _ = schedule(cfg, len(rows))
        n_in, n_lab = sum(len(r["ids"]) for r in rows), sum(sum(1 for x in r["labels"] if x != -100) for r in rows)
        print(f"examples kept {len(rows)}, dropped {dropped}; tokens/epoch {n_in:,}, trained {n_lab:,} ({n_lab / n_in:.1%}); "
              f"steps {total}, warmup {warmup}")
        return 0
    return train(cfg, a.max_steps)


if __name__ == "__main__":
    sys.exit(main())
