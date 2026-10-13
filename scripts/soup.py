"""Weight-average fine-tunes of the same base ("model soup", 14 Oct): typically the two seeds of a recipe. Averages every
tensor of the runs' merged16 checkpoints (same base, so this averages the fine-tuned deltas) and writes the result, with the
first run's config / tokenizer files, to --out/merged16, plus lora/SOUP and merge.log markers so scripts/train-leaf-qwen.sh
imports it like a trained run.
  .train-venv/bin/python scripts/soup.py --runs outputs/A outputs/B --out outputs/C
"""
import argparse, glob, os, shutil
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
src = [os.path.join(r, "merged16") for r in a.runs]
dst = os.path.join(a.out, "merged16"); os.makedirs(dst, exist_ok=True)
shards = sorted(os.path.basename(p) for p in glob.glob(os.path.join(src[0], "*.safetensors")))
for sh in shards:
    parts = [load_file(os.path.join(s, sh)) for s in src]
    keys = parts[0].keys()
    assert all(p.keys() == keys for p in parts), f"{sh}: tensor names differ between runs"
    out = {k: (sum(p[k].float() for p in parts) / len(parts)).to(parts[0][k].dtype) for k in keys}
    save_file(out, os.path.join(dst, sh), metadata={"format": "pt"})
    print(f"{sh}: averaged {len(keys)} tensors over {len(parts)} runs")
for f in os.listdir(src[0]):
    if not f.endswith(".safetensors"):
        p = os.path.join(src[0], f)
        (shutil.copytree if os.path.isdir(p) else shutil.copy)(p, os.path.join(dst, f))
os.makedirs(os.path.join(a.out, "lora"), exist_ok=True)
open(os.path.join(a.out, "lora", "SOUP"), "w").write("weight average of: " + " ".join(a.runs) + "\n")
open(os.path.join(a.out, "merge.log"), "w").write("model soup: merged16 written by scripts/soup.py\n")
shutil.copy(os.path.join(a.runs[0], "config.used.json"), os.path.join(a.out, "config.used.json"))
