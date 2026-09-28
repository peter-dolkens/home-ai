"""Merge the LoRA adapter into the base weights and save 16-bit safetensors for Ollama's own
converter (`ollama create --quantize q4_K_M` from a safetensors directory).
    python scripts/merge16.py configs/qwen35-4b.json
"""
import json, os, sys


def restore_config_keys(out, base_model):
    """save_pretrained can drop config keys the base had (Gemma 4 lost text_config.global_head_dim, so Ollama's
    converter wrote attention.key_length=0 and the model crashed on load). Copy back any key missing from the save."""
    from huggingface_hub import hf_hub_download
    base = json.load(open(hf_hub_download(base_model, "config.json")))
    p = os.path.join(out, "config.json")
    merged = json.load(open(p))
    added = []

    def fill(dst, src, path):
        for k, v in src.items():
            if k not in dst:
                dst[k] = v
                added.append(path + k)
            elif isinstance(v, dict) and isinstance(dst[k], dict):
                fill(dst[k], v, path + k + ".")
    fill(merged, base, "")
    if added:
        json.dump(merged, open(p, "w"), indent=2)
        print("restored config keys from base:", added)


cfg = json.load(open(sys.argv[1]))
out = os.path.join(cfg["out"], "merged16")
if os.path.exists(os.path.join(out, "config.json")):
    print("exists", out); sys.exit(0)
if cfg.get("backend") == "hf":
    # models Unsloth doesn't support (Spark X2.5): plain transformers + PEFT merge, keep the remote code files
    import glob, shutil, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    sys.path.insert(0, os.getcwd())
    from homeai.train.train import patch_rope_validation
    patch_rope_validation()
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cpu"})
    model = PeftModel.from_pretrained(base, os.path.join(cfg["out"], "lora")).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(cfg.get("hf_tokenizer", cfg["base_model"]), trust_remote_code=True)
    tok.save_pretrained(out)
    from huggingface_hub import snapshot_download
    src = snapshot_download(cfg["base_model"], allow_patterns=["*.py", "chat_template.jinja", "generation_config.json"])
    for f in glob.glob(os.path.join(src, "*")):
        if os.path.isfile(f):  # the snapshot can hold folders too (Spark's model card images)
            shutil.copy(f, out)
    restore_config_keys(out, cfg["base_model"])
    print("merged (hf) ->", out, os.listdir(out)); sys.exit(0)
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(model_name=os.path.join(cfg["out"], "lora"), max_seq_length=cfg["max_seq_len"],
                                              load_in_4bit=cfg.get("load_in_4bit", True), dtype=None)
model.save_pretrained_merged(out, tok, save_method="merged_16bit")
restore_config_keys(out, cfg["base_model"])
print("merged ->", out, os.listdir(out))
