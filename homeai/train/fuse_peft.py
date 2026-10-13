"""PEFT LoRA adapter (the CUDA boxes' Unsloth runs, archived on Piper) + HF base checkpoint -> 16-bit safetensors, on a Mac
with MLX only (no torch / peft / Unsloth): W + alpha/r * B A on every adapted linear, merged in fp32 and saved in the base
dtype, the base's shards otherwise untouched (vision tower and MTP included, as Unsloth's merged_16bit keeps them), plus
the base's config files with the adapter dir's tokenizer and chat template on top (what training rendered with).
Not bit-identical to the CUDA boxes' Unsloth merge: `ollama create --quantize q4_K_M` of the v5 result differs from the
archived Q4_K_M in every quantised tensor, even untouched ones (token_embd, MTP), with or without a float16 cast, so the
falcon quantisation variants carry a rebuilt-Q4_K_M control.
Used for the falcon phase 2 builds (other quantisations via `ollama create --quantize`, and mlx_lm.convert).

  scripts/mlx-env.sh python -m homeai.train.fuse_peft --lora outputs/qwen35-4b-lora-v5/lora --out outputs/qwen35-4b-lora-v5/merged16
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

from ..host import host_info

PREFIX = "base_model.model."
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json")


def fuse(lora: str, out: str, base: str = "") -> str:
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    cfg = json.load(open(os.path.join(lora, "adapter_config.json")))
    base = base or cfg["base_model_name_or_path"]
    src = base if os.path.isdir(base) else snapshot_download(base)
    scale = cfg["lora_alpha"] / cfg["r"]
    if cfg.get("use_rslora") or cfg.get("use_dora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise SystemExit("fuse_peft: rsLoRA / DoRA / per-module ranks are not handled")
    ad = mx.load(os.path.join(lora, "adapter_model.safetensors"))
    mods = sorted({k[len(PREFIX):].rsplit(".lora_", 1)[0] for k in ad})
    todo = {m + ".weight": m for m in mods}
    os.makedirs(out, exist_ok=True)
    done = 0
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        w, meta = mx.load(f, return_metadata=True)
        for hk, m in todo.items():
            if hk in w:
                a = ad[f"{PREFIX}{m}.lora_A.weight"].astype(mx.float32)  # r x in
                b = ad[f"{PREFIX}{m}.lora_B.weight"].astype(mx.float32)  # out x r
                w[hk] = (w[hk].astype(mx.float32) + scale * (b @ a)).astype(w[hk].dtype)
                done += 1
        mx.save_safetensors(os.path.join(out, os.path.basename(f)), w, metadata={**(meta or {}), "format": "pt"})
        del w
    if done != len(todo):
        raise SystemExit(f"fuse_peft: merged {done} of {len(todo)} adapted weights")
    for f in glob.glob(os.path.join(src, "*")):
        if os.path.isfile(f) and not f.endswith(".safetensors") and os.path.basename(f) not in ("README.md", "LICENSE", ".gitattributes"):
            shutil.copyfile(f, os.path.join(out, os.path.basename(f)))  # not copy(): HF cache blobs are read-only
    for n in TOKENIZER_FILES:
        if os.path.exists(os.path.join(lora, n)):
            shutil.copyfile(os.path.join(lora, n), os.path.join(out, n))
    json.dump({"base_model": base, "adapter": lora, "merged_weights": done, "scale": scale, "method": "fuse_peft", **host_info()},
              open(os.path.join(out, "homeai-merge.json"), "w"), indent=2)
    print(f"fused {done} weights -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True, help="PEFT adapter dir (adapter_config.json + adapter_model.safetensors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="", help="HF repo id or local dir (default: the adapter's base_model_name_or_path)")
    a = ap.parse_args()
    fuse(a.lora, a.out, a.base)


if __name__ == "__main__":
    main()
