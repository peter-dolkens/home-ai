"""Merge the LoRA adapter, quantise to GGUF and register the model with Ollama.

    python -m homeai.train.export --config configs/qwen25-7b.json --name homeai
    python -m homeai.train.export --config configs/qwen25-7b.json --name homeai --modelfile-only  # gguf already built

The Ollama TEMPLATE (with tool-calling support) is copied from the matching
official Ollama model so the prompt format matches what the base model expects.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys


def llama_cpp_convert(merged_dir: str, gguf_dir: str, quant: str):
    """Convert a merged HF checkpoint with llama.cpp (present in the Unsloth image at /opt/unsloth/llama.cpp)."""
    roots = [p for p in ("/opt/unsloth/llama.cpp", "/opt/unsloth-studio/llama.cpp", os.path.expanduser("~/llama.cpp")) if os.path.isdir(p)]
    if not roots:
        raise RuntimeError("llama.cpp not found for fallback conversion")
    root = roots[0]
    f16 = os.path.join(gguf_dir, "model-f16.gguf")
    subprocess.run([sys.executable, os.path.join(root, "convert_hf_to_gguf.py"), merged_dir, "--outfile", f16, "--outtype", "f16"], check=True)
    quantize = next((p for p in glob.glob(os.path.join(root, "**", "llama-quantize"), recursive=True)), None) or shutil.which("llama-quantize")
    if not quantize:
        raise RuntimeError("llama-quantize binary not found")
    out = os.path.join(gguf_dir, f"model-{quant}.gguf")
    subprocess.run([quantize, f16, out, quant.upper()], check=True)
    os.remove(f16)


def collect_gguf(out_dir: str, gguf_dir: str, quant: str):
    """Unsloth writes to <gguf_dir>_gguf/ and leaves merged safetensors behind. Move the text
    GGUF into <gguf_dir>/model-<quant>.gguf and remove the multi-GB leftovers."""
    candidates = []
    for d in (gguf_dir, gguf_dir + "_gguf", out_dir):
        candidates += glob.glob(os.path.join(d, "*.gguf"))
    text = [g for g in candidates if "mmproj" not in os.path.basename(g).lower()]
    if not text:
        return
    quant_match = [g for g in text if quant.lower() in os.path.basename(g).lower()] or text
    src = sorted(quant_match, key=os.path.getmtime)[-1]
    os.makedirs(gguf_dir, exist_ok=True)
    dst = os.path.join(gguf_dir, f"model-{quant}.gguf")
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.move(src, dst)
    for g in candidates:
        if os.path.exists(g) and os.path.abspath(g) != os.path.abspath(dst):
            os.remove(g)  # mmproj / other quants
    for pattern in ("*.safetensors", "*.safetensors.index.json"):
        for f in glob.glob(os.path.join(gguf_dir, pattern)):
            os.remove(f)
    for d in (gguf_dir + "_gguf", os.path.join(out_dir, "merged16")):
        shutil.rmtree(d, ignore_errors=True)
    print(f"collected {dst} ({os.path.getsize(dst) / 1e9:.2f} GB)")


def ollama_template(tag: str) -> tuple[str, list[str]]:
    """Return (TEMPLATE body, stop PARAMETER lines) from `ollama show --modelfile <tag>`."""
    try:
        out = subprocess.run(["ollama", "show", "--modelfile", tag], capture_output=True, text=True, check=True).stdout
    except Exception as e:  # noqa: BLE001
        print(f"warning: could not read template from ollama ({e}); pull it with `ollama pull {tag}` and re-run with --modelfile-only")
        return "", []
    m = re.search(r'TEMPLATE\s+"""(.*?)"""', out, re.S)
    stops = re.findall(r'^PARAMETER stop .*$', out, re.M)
    return (m.group(1) if m else ""), stops


def write_modelfile(gguf: str, out_dir: str, tag: str, num_ctx: int, temperature: float) -> str:
    template, stops = ollama_template(tag)
    lines = [f"FROM ./{os.path.basename(gguf)}"]
    if template:
        lines.append(f'TEMPLATE """{template}"""')
    else:
        lines.append('# TEMPLATE missing: paste the TEMPLATE block from `ollama show --modelfile ' + tag + '` here')
    lines += stops
    lines += [f"PARAMETER num_ctx {num_ctx}", f"PARAMETER temperature {temperature}", "PARAMETER top_p 0.9"]
    path = os.path.join(out_dir, "Modelfile")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default="homeai")
    ap.add_argument("--quant", default="q4_k_m", help="q4_k_m fits a 10GB 3080 with 8k context; q5_k_m if you have 12GB")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--modelfile-only", action="store_true")
    ap.add_argument("--no-create", action="store_true", help="write the Modelfile but skip `ollama create`")
    ap.add_argument("--collect-only", action="store_true", help="only tidy an existing conversion into <out>/gguf/")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    lora_dir = os.path.join(cfg["out"], "lora")
    gguf_dir = os.path.join(cfg["out"], "gguf")
    os.makedirs(gguf_dir, exist_ok=True)

    if a.collect_only:
        collect_gguf(cfg["out"], gguf_dir, a.quant)
        print(sorted(glob.glob(os.path.join(gguf_dir, "*.gguf"))))
        return 0
    if not a.modelfile_only:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=lora_dir, max_seq_length=cfg["max_seq_len"],
                                                             load_in_4bit=cfg.get("load_in_4bit", True), dtype=None)
        try:
            model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=a.quant)
        except Exception as e:  # noqa: BLE001
            print(f"save_pretrained_gguf failed ({e}); falling back to merged 16-bit + llama.cpp convert")
            merged = os.path.join(cfg["out"], "merged16")
            model.save_pretrained_merged(merged, tokenizer, save_method="merged_16bit")
            llama_cpp_convert(merged, gguf_dir, a.quant)

    if not a.modelfile_only:
        collect_gguf(cfg["out"], gguf_dir, a.quant)
    ggufs = sorted(glob.glob(os.path.join(gguf_dir, "*.gguf")), key=os.path.getmtime)
    if not ggufs:
        print(f"no .gguf found in {gguf_dir}", file=sys.stderr)
        return 1
    gguf = ggufs[-1]
    mf = write_modelfile(gguf, gguf_dir, cfg["ollama_template_from"], a.num_ctx, a.temperature)
    print(f"wrote {mf}")
    if a.no_create or not shutil.which("ollama"):
        print(f"run:  cd {gguf_dir} && ollama create {a.name} -f Modelfile")
        return 0
    subprocess.run(["ollama", "create", a.name, "-f", "Modelfile"], cwd=gguf_dir, check=True)
    print(f"created ollama model '{a.name}'. Try:  python -m homeai.bridge.cli chat --mock --model {a.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
