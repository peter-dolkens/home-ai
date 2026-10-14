"""DPO on preference pairs from homeai/train/dpo_b.py (14 Oct). Plain transformers + PEFT, run in the Unsloth container.

A fresh LoRA on top of an SFT model's merged weights (the config's base_model is that merged16 directory); the
reference policy is the same network with the adapter switched off, so only one copy of the weights is in memory.
Loss: -log sigmoid(beta * (policy margin - reference margin)) over the assistant tokens of the new turn (tool results
and the shared history never count), plus sft_weight x the chosen side's per-token NLL (as RPO) so the model keeps
producing the transcripts it already gets right. The output layer runs only where a label follows (as run_hf).

    scripts/train-docker.sh dpo configs/lfm25-1.2b-dpo-b-v1.json

Writes <out>/lora, then <out>/merged16 (merged on CPU) for the usual Ollama import and MLX build.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
import sys
import time

from .data import load_jsonl, render, text_tokenizer
from .train import load_config


def build_rows(pairs, layout):
    """Each pair as two training rows (prompt + chosen, prompt + rejected) in the config's prompt layout."""
    from ..prompt_layout import apply as apply_layout
    out = []
    for p in pairs:
        meta = {k: v for k, v in p.items() if k not in ("prompt", "chosen", "rejected")}
        c = apply_layout({**meta, "messages": p["prompt"] + p["chosen"]}, layout)
        r = apply_layout({**meta, "messages": p["prompt"] + p["rejected"]}, layout)
        out.append((p, c, r, sum(1 for m in p["prompt"] if m["role"] == "user")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="tokenise and report, no training")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)

    import torch
    from transformers import AutoTokenizer

    tok = text_tokenizer(AutoTokenizer.from_pretrained(cfg.get("hf_tokenizer", cfg["base_model"]), trust_remote_code=True))
    if cfg.get("chat_template_from"):
        tok.chat_template = AutoTokenizer.from_pretrained(cfg["chat_template_from"]).chat_template
    user = tok("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    resp = tok(cfg["response_part"], add_special_tokens=False)["input_ids"]
    end = tok(cfg["end_part"], add_special_tokens=False)["input_ids"]

    def find(seq, sub, start):
        for i in range(start, len(seq) - len(sub) + 1):
            if seq[i:i + len(sub)] == sub:
                return i
        return -1

    def encode(row, n_user):
        """ids, labels: only assistant spans after the prompt's last user turn are labelled."""
        ids = tok(render(tok, row), add_special_tokens=False)["input_ids"]
        pos = -1
        for _ in range(n_user):
            pos = find(ids, user, pos + 1)
        labels = [-100] * len(ids)
        i = find(ids, resp, pos)
        while i >= 0:
            s = i + len(resp)
            j = find(ids, end, s)
            e = len(ids) if j < 0 else j + len(end)
            labels[s:e] = ids[s:e]
            i = find(ids, resp, e)
        return ids, labels

    pairs = [json.loads(l) for l in open(cfg["pairs"]) if l.strip()]
    rows = build_rows(pairs, cfg.get("prompt_layout"))
    data, dropped, lens = [], 0, []
    for p, c, r, n_user in rows:
        ci, cl = encode(c, n_user)
        ri, rl = encode(r, n_user)
        lens.append(max(len(ci), len(ri)))
        if max(len(ci), len(ri)) > cfg["max_seq_len"] or not any(x != -100 for x in cl) or not any(x != -100 for x in rl):
            dropped += 1
            continue
        data.append((ci, cl, ri, rl))
    n_lab = statistics.mean(sum(1 for x in cl if x != -100) for _, cl, _, _ in data[:200])
    print(f"pairs: {len(data)} kept, {dropped} dropped (over {cfg['max_seq_len']} tokens or nothing labelled); "
          f"mean {statistics.mean(lens):.0f} tokens; ~{n_lab:.0f} labelled chosen tokens per pair", flush=True)
    if a.dry_run:
        ci, cl, _, _ = data[0]
        print(tok.decode([t for t in cl if t != -100]))
        return 0

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

    if cfg.get("cuda_memory_gb"):
        torch.cuda.set_per_process_memory_fraction(min(1.0, cfg["cuda_memory_gb"] * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    torch.manual_seed(cfg["seed"])
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], trust_remote_code=True, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0, bias="none",
                                             target_modules=cfg["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def logp(ids, labels):
        """Summed log-probability of the labelled tokens, and their count."""
        x = torch.tensor([ids], device="cuda")
        nxt = torch.tensor(labels[1:], device="cuda")
        keep = (nxt != -100).nonzero(as_tuple=True)[0]
        logits = model(input_ids=x, logits_to_keep=keep, use_cache=False).logits[0].float()
        lp = torch.log_softmax(logits, -1).gather(1, nxt[keep].unsqueeze(1)).squeeze(1)
        return lp.sum(), len(keep)

    params = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(params, lr=cfg["lr"], weight_decay=0.0)
    except Exception:  # noqa: BLE001
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    accum, epochs, beta, sft_w = cfg["grad_accum"], cfg["epochs"], cfg["beta"], cfg.get("sft_weight", 0.0)
    steps = math.ceil(len(data) * epochs / accum)
    sched = get_cosine_schedule_with_warmup(opt, cfg.get("warmup_steps", 10), steps)
    print(f"{steps} optimizer steps ({epochs} epochs x {len(data)} pairs / {accum}), beta {beta}, sft_weight {sft_w}, lr {cfg['lr']}", flush=True)

    rng = random.Random(cfg["seed"])
    model.train()
    t0, step, win = time.time(), 0, []
    for ep in range(math.ceil(epochs)):
        order = list(range(len(data)))
        rng.shuffle(order)
        if ep + 1 > epochs:
            order = order[: int(len(order) * (epochs - ep))]
        for n, i in enumerate(order, 1):
            ci, cl, ri, rl = data[i]
            with torch.no_grad(), model.disable_adapter():
                rc, _ = logp(ci, cl)
                rr, _ = logp(ri, rl)
            pc, nc = logp(ci, cl)
            pr, _ = logp(ri, rl)
            margin = beta * ((pc - rc) - (pr - rr))
            loss = -torch.nn.functional.logsigmoid(margin) + sft_w * (-pc / nc)
            (loss / accum).backward()
            win.append((loss.item(), margin.item(), float(margin.item() > 0), (pc - rc).item(), (pr - rr).item()))
            if n % accum == 0 or n == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.get("logging_steps", 5) == 0 or step == steps:
                    L, M, A, C, R = (statistics.mean(v) for v in zip(*win))
                    el = time.time() - t0
                    print(f"step {step}/{steps} loss {L:.4f} margin {M:.3f} acc {A:.2f} chosen d {C:.2f} rejected d {R:.2f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {el / step:.1f}s/step eta {el / step * (steps - step) / 60:.0f} min", flush=True)
                    win = []
                if step % cfg.get("save_steps", 100) == 0:
                    model.save_pretrained(os.path.join(cfg["out"], "lora"))

    lora_dir = os.path.join(cfg["out"], "lora")
    model.save_pretrained(lora_dir)
    tok.save_pretrained(lora_dir)
    with open(os.path.join(cfg["out"], "config.used.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"saved LoRA adapter to {lora_dir} in {(time.time() - t0) / 60:.0f} min", flush=True)

    # merge on CPU into a full bf16 copy: the same merged16 layout the SFT runs leave for the Ollama import / MLX build
    del model, opt
    torch.cuda.empty_cache()
    merged = os.path.join(cfg["out"], "merged16")
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cpu"})
    PeftModel.from_pretrained(base, lora_dir).merge_and_unload().save_pretrained(merged, safe_serialization=True)
    tok.save_pretrained(merged)
    for f in ("chat_template.jinja", "generation_config.json"):
        if os.path.exists(os.path.join(cfg["base_model"], f)):
            shutil.copy(os.path.join(cfg["base_model"], f), merged)
    open(os.path.join(cfg["out"], "merge.log"), "w").write(f"merged {lora_dir} into {cfg['base_model']} -> {merged}\n")
    print("merged ->", merged)
    return 0


if __name__ == "__main__":
    sys.exit(main())
