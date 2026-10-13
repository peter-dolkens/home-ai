"""QLoRA fine-tune with Unsloth. Run this on the GPU box (RTX 3080 or similar).

    pip install -e ".[train]"
    python -m homeai.train.train --config configs/qwen25-7b.json
    python -m homeai.train.train --config configs/qwen25-7b.json --dry-run   # token stats only, no GPU needed

Only assistant turns contribute to the loss (train_on_responses_only). The tool
schemas from homeai.tools are rendered into every example via the model's own
chat template, so what the model sees in training matches what Ollama sends.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import sys

from .data import check_tool_calls_rendered, load_jsonl, render_all, text_tokenizer


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    for kv in overrides:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        cfg[k] = v
    return cfg


def make_trainer(model, tokenizer, train_ds, val_ds, cfg):
    """Handle TRL/SFTTrainer signature drift across versions."""
    from trl import SFTConfig, SFTTrainer
    from unsloth import is_bfloat16_supported

    args_kw = dict(
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        warmup_ratio=0.05,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=cfg["seed"],
        output_dir=os.path.join(cfg["out"], "checkpoints"),
        report_to="none",
        save_strategy="steps",   # a crash (e.g. OOM at the epoch-end validation) loses at most save_steps
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=2,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
    )
    sig = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in sig:
        args_kw["max_length"] = cfg["max_seq_len"]
    elif "max_seq_length" in sig:
        args_kw["max_seq_length"] = cfg["max_seq_len"]
    if "dataset_text_field" in sig:
        args_kw["dataset_text_field"] = "text"
    if val_ds is not None:
        args_kw["eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"] = "epoch"
        # validation materialises full-vocab logits (248k x seq) in fp32: keep the batch at 1 and loss-only
        args_kw["per_device_eval_batch_size"] = cfg.get("eval_batch_size", 1)
        args_kw["prediction_loss_only"] = True
        if "eval_accumulation_steps" in sig:
            args_kw["eval_accumulation_steps"] = 1
    if "dataset_num_proc" in sig:
        args_kw["dataset_num_proc"] = 2
    args = SFTConfig(**args_kw)

    tk = dict(model=model, train_dataset=train_ds, eval_dataset=val_ds, args=args)
    tsig = inspect.signature(SFTTrainer.__init__).parameters
    tk["processing_class" if "processing_class" in tsig else "tokenizer"] = tokenizer
    if "dataset_text_field" in tsig:
        tk["dataset_text_field"] = "text"
    if "max_seq_length" in tsig:
        tk["max_seq_length"] = cfg["max_seq_len"]
    return SFTTrainer(**tk)


def patch_rope_validation():
    """Spark X2.5's custom config stores rope parameters per attention type, which transformers 5.17's validator
    misreads (it runs before layer_types is set). The model code reads them itself, so skip what the validator can't parse."""
    import transformers.modeling_rope_utils as mru
    cls = getattr(mru, "RotaryEmbeddingConfigMixin", None)
    if cls is None or getattr(cls, "_homeai_patched", False):
        return
    orig = cls.validate_rope

    def safe(self):
        try:
            orig(self)
        except AttributeError:
            pass
    cls.validate_rope = safe
    cls._homeai_patched = True
    try:  # huggingface_hub dataclass validators may hold the original function object
        from transformers.configuration_utils import PreTrainedConfig
        vals = getattr(PreTrainedConfig, "__class_validators__", None)
        if isinstance(vals, list):
            PreTrainedConfig.__class_validators__ = [safe if getattr(v, "__name__", "") == "validate_rope" else v for v in vals]
    except Exception:  # noqa: BLE001
        pass


def run_hf(cfg, train, val):
    """Plain transformers + PEFT training for architectures Unsloth does not support yet (e.g. Spark X2.5,
    custom code via trust_remote_code). Loss only on assistant turns: from each response_part marker up to and
    including the next end_part marker."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    patch_rope_validation()
    tok = AutoTokenizer.from_pretrained(cfg.get("hf_tokenizer", cfg["base_model"]), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True) if cfg.get("load_in_4bit", True) else None
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], trust_remote_code=True, dtype=torch.bfloat16,
                                                 quantization_config=q, device_map={"": 0})
    if q is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0, bias="none",
                                             target_modules=cfg["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    resp = tok(cfg["response_part"], add_special_tokens=False)["input_ids"]
    end = tok(cfg["end_part"], add_special_tokens=False)["input_ids"]

    def find(seq, sub, start):
        for i in range(start, len(seq) - len(sub) + 1):
            if seq[i:i + len(sub)] == sub:
                return i
        return -1

    def encode(texts):
        rows = []
        for t in texts:
            ids = tok(t["text"], add_special_tokens=False)["input_ids"]
            labels = [-100] * len(ids)
            i = find(ids, resp, 0)
            while i >= 0:
                a = i + len(resp)
                j = find(ids, end, a)
                b = len(ids) if j < 0 else j + len(end)
                labels[a:b] = ids[a:b]
                i = find(ids, resp, b)
            rows.append({"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)})
        return rows

    train_texts, dropped, lengths = render_all(tok, train, cfg["max_seq_len"])
    print(f"train: {len(train_texts)} examples (dropped {dropped} over {cfg['max_seq_len']} tokens); mean {statistics.mean(lengths):.0f} tokens")
    train_ds = Dataset.from_list(encode(train_texts))
    val_ds = None
    if val:
        vt, vdrop, _ = render_all(tok, val, cfg.get("val_max_tokens", 2048))
        val_ds = Dataset.from_list(encode(vt[: cfg.get("val_max_examples", 200)])) if vt else None
    n_lab = sum(sum(1 for x in r if x != -100) for r in train_ds["labels"][:50])
    print(f"label check: {n_lab} trained tokens in the first 50 examples")
    if n_lab == 0:
        raise SystemExit("no assistant tokens found: check response_part / end_part")

    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (m - len(seq))  # noqa: E731
        return {"input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
                "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
                "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch])}

    args = TrainingArguments(output_dir=os.path.join(cfg["out"], "checkpoints"), per_device_train_batch_size=cfg["batch_size"],
                             gradient_accumulation_steps=cfg["grad_accum"], num_train_epochs=cfg["epochs"], learning_rate=cfg["lr"],
                             warmup_steps=cfg.get("warmup_steps", 10), logging_steps=10, lr_scheduler_type="cosine", weight_decay=0.01,
                             optim="paged_adamw_8bit", bf16=True, save_strategy="steps", save_steps=cfg.get("save_steps", 100),
                             save_total_limit=2, eval_strategy="epoch" if val_ds is not None else "no", per_device_eval_batch_size=1,
                             prediction_loss_only=True, report_to="none", seed=cfg["seed"], remove_unused_columns=False,
                             gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    class LabelledLogitsTrainer(Trainer):
        """Run the output layer only where a label follows. Spark's 131k vocabulary makes the full logits for a
        4,608-token example ~2.4 GB in fp32, which OOM'd next to the ~6 GB other services hold; assistant tokens are a
        small share of each conversation. Needs a forward that takes logits_to_keep as an index tensor (Spark does)."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            nxt = labels[:, 1:]
            keep = (nxt != -100).any(dim=0).nonzero(as_tuple=True)[0]  # positions whose next token is trained
            out = model(**inputs, logits_to_keep=keep, use_cache=False)
            logits = out.logits.float()  # (batch, len(keep), vocab)
            target = nxt[:, keep]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                                                     ignore_index=-100, reduction="sum")
            n = num_items_in_batch if num_items_in_batch is not None else (target != -100).sum()
            loss = loss / n
            return (loss, out) if return_outputs else loss

    trainer_cls = LabelledLogitsTrainer if cfg.get("labelled_logits_only") else Trainer
    trainer = trainer_cls(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate)
    import glob as _glob
    cks = sorted(_glob.glob(os.path.join(cfg["out"], "checkpoints", "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
    print(trainer.train(resume_from_checkpoint=cks[-1] if cks and cfg.get("resume", True) else None))
    lora_dir = os.path.join(cfg["out"], "lora")
    model.save_pretrained(lora_dir)
    tok.save_pretrained(lora_dir)
    print(f"saved LoRA adapter to {lora_dir}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override config keys, e.g. --set epochs=1 out=outputs/x")
    ap.add_argument("--dry-run", action="store_true", help="render + count tokens with the HF tokenizer only")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)

    train = load_jsonl(cfg["train"], cfg.get("prompt_layout"))
    val = load_jsonl(cfg["val"], cfg.get("prompt_layout")) if cfg.get("val") and os.path.exists(cfg["val"]) else []

    if a.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.get("hf_tokenizer", cfg["base_model"]))
        texts, dropped, lengths = render_all(tok, train, cfg["max_seq_len"])
        print(f"examples: {len(train)}  kept: {len(texts)}  dropped(>max_seq_len): {dropped}")
        print(f"tokens/example  mean {statistics.mean(lengths):.0f}  p50 {statistics.median(lengths):.0f}  max {max(lengths)}")
        print(f"total tokens/epoch ~ {sum(lengths):,}")
        print("\n--- sample rendered example ---\n" + texts[0]["text"][:3000])
        return 0

    if cfg.get("backend") == "hf":
        return run_hf(cfg, train, val)

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"], max_seq_length=cfg["max_seq_len"], load_in_4bit=cfg.get("load_in_4bit", True), dtype=None,
    )
    processor, tokenizer = tokenizer, text_tokenizer(tokenizer)  # text-only training on a possibly-VL checkpoint
    if cfg.get("chat_template_from"):
        # Unsloth swaps LiquidAI/LFM2.5-1.2B-Instruct for its unsloth/ mirror, whose older template drops assistant
        # tool_calls: every tool-call turn trained as an empty reply (29 Sep). Take the template from a repo that renders them.
        from transformers import AutoTokenizer
        tokenizer.chat_template = AutoTokenizer.from_pretrained(cfg["chat_template_from"]).chat_template
    model = FastLanguageModel.get_peft_model(
        model, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0, bias="none",
        # per-architecture override: LFM2 names its projections out_proj / in_proj (conv blocks) / w1 w2 w3 (feed-forward), so the
        # Qwen-style default matched only q/k/v in 6 of 16 layers and the LFM2.5 fine-tunes barely changed the model
        target_modules=cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=cfg["seed"],
    )
    if cfg.get("init_adapter"):
        # continue from an existing adapter (reinforcement round): same LoRA shape, copy its weights in
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        sd = load_file(os.path.join(cfg["init_adapter"], "adapter_model.safetensors"))
        res = set_peft_model_state_dict(model, sd)
        missing = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
        unexpected = list(getattr(res, "unexpected_keys", []))
        print(f"init adapter {cfg['init_adapter']}: {len(sd)} tensors, {len(missing)} missing lora keys, {len(unexpected)} unexpected")
        if unexpected or len(missing) > 0.1 * len(sd):
            raise SystemExit("init_adapter does not match this LoRA layout")

    check_tool_calls_rendered(tokenizer, train)
    train_texts, dropped, lengths = render_all(tokenizer, train, cfg["max_seq_len"])
    print(f"train: {len(train_texts)} examples (dropped {dropped} over {cfg['max_seq_len']} tokens); "
          f"mean {statistics.mean(lengths):.0f} tokens, max {max(lengths)}")
    train_ds = Dataset.from_list(train_texts)
    val_ds = None
    if val:
        # Validation upcasts full-vocab logits (seq x 248k) to fp32 in one block: a 4.5k-token agent-written
        # conversation needs ~4.2 GB on top of training and OOM'd 4b-v3/4b-v4 at the epoch end. Keep it short.
        vmax = cfg.get("val_max_tokens", 2048)
        vt, vdrop, _ = render_all(tokenizer, val, vmax)
        vt = vt[: cfg.get("val_max_examples", 200)]
        print(f"val: {len(vt)} examples (dropped {vdrop} over {vmax} tokens)")
        val_ds = Dataset.from_list(vt) if vt else None

    trainer = make_trainer(model, tokenizer, train_ds, val_ds, cfg)
    trainer = train_on_responses_only(trainer, instruction_part=cfg["instruction_part"], response_part=cfg["response_part"])
    if cfg.get("mask_spans"):
        # e.g. Gemma 4 puts tool results inside the model turn: never train the model to predict house state
        # A third element "keep_open" leaves the opening marker as a label. Gemma 4 needs it: the model itself must
        # emit <|tool_response> right after <tool_call|> (Ollama stops generation there); masking it taught the
        # first Gemma run to chain call after call in one turn.
        pairs = [(tokenizer(sp[0], add_special_tokens=False)["input_ids"], tokenizer(sp[1], add_special_tokens=False)["input_ids"],
                  len(sp) > 2 and sp[2] == "keep_open") for sp in cfg["mask_spans"]]

        def find(seq, sub, start):
            for i in range(start, len(seq) - len(sub) + 1):
                if seq[i:i + len(sub)] == sub:
                    return i
            return -1

        def mask(ex):
            ids, labels = ex["input_ids"], list(ex["labels"])
            for o, c, keep_open in pairs:
                i = find(ids, o, 0)
                while i >= 0:
                    j = find(ids, c, i + len(o))
                    end = len(ids) if j < 0 else j + len(c)
                    start = i + len(o) if keep_open else i
                    labels[start:end] = [-100] * (end - start)
                    i = find(ids, o, end)
            return {"labels": labels}

        for attr in ("train_dataset", "eval_dataset"):
            ds = getattr(trainer, attr, None)
            if ds is not None and "labels" in ds.column_names:
                setattr(trainer, attr, ds.map(mask, num_proc=4))
            elif ds is not None:
                raise SystemExit(f"mask_spans: {attr} has no labels column to mask")
        print(f"masked spans {cfg['mask_spans']} (token ids {[(o, c) for o, c, _ in pairs]})")
    resume = None
    if cfg.get("resume", True):
        import glob as _glob
        cks = sorted(_glob.glob(os.path.join(cfg["out"], "checkpoints", "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
        if cks:
            resume = cks[-1]
            print(f"resuming from {resume}")
    stats = trainer.train(resume_from_checkpoint=resume)
    print(stats)

    lora_dir = os.path.join(cfg["out"], "lora")
    model.save_pretrained(lora_dir)
    processor.save_pretrained(lora_dir)
    with open(os.path.join(cfg["out"], "config.used.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"saved LoRA adapter to {lora_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
