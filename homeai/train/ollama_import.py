"""Host-side: register a fine-tuned model with the Ollama container on cassowary.

    python -m homeai.train.ollama_import --run outputs/qwen35-4b-r1 --name homeai-4b --base qwen3.5:4b

Default route (works for Qwen3.5, whose Ollama engine needs Ollama's own tensor layout): copy the
merged 16-bit safetensors from <run>/merged16 into the container and let `ollama create --quantize`
convert them. The quantised blob is then copied back to <run>/gguf/<name>-q4_K_M.gguf so it can be
downloaded and re-imported anywhere with a one-line Modelfile. `--gguf` uses a llama.cpp GGUF instead.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

OLLAMA = ["docker", "exec", "ollama", "ollama"]


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training output dir containing gguf/")
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", required=True, help="Ollama tag whose TEMPLATE to reuse, e.g. qwen3.5:4b")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--container", default="ollama")
    ap.add_argument("--quantize", default="q4_K_M")
    ap.add_argument("--gguf", action="store_true", help="import a llama.cpp GGUF from <run>/gguf instead of merged safetensors")
    a = ap.parse_args()
    if a.gguf:
        ggufs = sorted(glob.glob(os.path.join(a.run, "gguf", "*.gguf")), key=os.path.getmtime)
        ggufs = [g for g in ggufs if "f16" not in os.path.basename(g).lower() or len(ggufs) == 1]
        ggufs = [g for g in ggufs if os.path.basename(g).startswith(a.name)] or ggufs  # prefer the Ollama-native file
        if not ggufs:
            sys.exit(f"no gguf in {a.run}/gguf")
        src, from_line = ggufs[-1], f"FROM /import/{a.name}/{os.path.basename(ggufs[-1])}"
    else:
        src = os.path.join(a.run, "merged16")
        if not os.path.exists(os.path.join(src, "config.json")):
            sys.exit(f"no merged safetensors in {src} (run scripts/train-docker.sh merge <config>)")
        from_line = f"FROM /import/{a.name}/model"
    show = sh([*OLLAMA, "show", "--modelfile", a.base])
    # Reuse the base model's prompt plumbing verbatim: either a TEMPLATE block (Go template) or
    # Ollama's built-in RENDERER/PARSER pair (newer models such as qwen3.5), plus stop tokens.
    keep: list[str] = []
    m = re.search(r'TEMPLATE\s+"""(.*?)"""', show, re.S)
    if m:
        keep.append(f'TEMPLATE """{m.group(1)}"""')
    else:
        m1 = re.search(r"^TEMPLATE\s+(.+)$", show, re.M)
        if m1:
            keep.append(f"TEMPLATE {m1.group(1).strip()}")
    keep += re.findall(r"^(?:RENDERER|PARSER)\s+\S+$", show, re.M)
    keep += re.findall(r"^PARAMETER stop .*$", show, re.M)
    if not keep:
        sys.exit(f"could not find TEMPLATE/RENDERER/PARSER in `ollama show --modelfile {a.base}`")
    # run dirs are written by the training container (root); keep the Modelfile somewhere we own
    os.makedirs("outputs/modelfiles", exist_ok=True)
    modelfile = os.path.join("outputs", "modelfiles", f"{a.name}.Modelfile")
    with open(modelfile, "w") as f:
        f.write(from_line + "\n")
        for line in keep:
            f.write(line + "\n")
        f.write(f"PARAMETER num_ctx {a.num_ctx}\nPARAMETER temperature {a.temperature}\nPARAMETER top_p 0.9\nPARAMETER top_k 20\n")
    subprocess.run(["docker", "exec", a.container, "rm", "-rf", f"/import/{a.name}"], check=False)
    subprocess.run(["docker", "exec", a.container, "mkdir", "-p", f"/import/{a.name}"], check=True)
    try:
        subprocess.run(["docker", "cp", src, f"{a.container}:/import/{a.name}/" + ("" if a.gguf else "model")], check=True)
        subprocess.run(["docker", "cp", modelfile, f"{a.container}:/import/{a.name}/Modelfile"], check=True)
        if a.gguf:
            subprocess.run(["docker", "exec", a.container, "ollama", "create", a.name, "-f", f"/import/{a.name}/Modelfile"], check=True)
        else:
            # Two steps keep peak memory down on big models: convert to f16 first, then quantise the
            # f16 model (a single `--quantize` from safetensors was OOM-killed for the 9B on 31 GB RAM).
            f16 = f"{a.name}-f16"
            subprocess.run(["docker", "exec", a.container, "ollama", "create", f16, "-f", f"/import/{a.name}/Modelfile"], check=True)
            subprocess.run(["docker", "exec", a.container, "rm", "-rf", f"/import/{a.name}/model"], check=False)
            qmf = os.path.join("outputs", "modelfiles", f"{a.name}.quant.Modelfile")
            with open(qmf, "w") as f:
                f.write(f"FROM {f16}\n")
            subprocess.run(["docker", "cp", qmf, f"{a.container}:/import/{a.name}/Modelfile.quant"], check=True)
            subprocess.run(["docker", "exec", a.container, "ollama", "create", a.name, "-f", f"/import/{a.name}/Modelfile.quant", "--quantize", a.quantize], check=True)
            subprocess.run(["docker", "exec", a.container, "ollama", "rm", f16], check=False)
    finally:
        subprocess.run(["docker", "exec", a.container, "rm", "-rf", f"/import/{a.name}"], check=False)
    # pull the quantised weights back out as a portable GGUF
    mf = sh([*OLLAMA, "show", "--modelfile", a.name])
    blob = re.search(r"^FROM\s+(\S*blobs/\S+)", mf, re.M)
    if blob and not a.gguf:
        os.makedirs(os.path.join(a.run, "gguf"), exist_ok=True)
        dst = os.path.join(a.run, "gguf", f"{a.name}-{a.quantize}.gguf")
        subprocess.run(["docker", "cp", f"{a.container}:{blob.group(1)}", dst], check=True)
        print(f"portable GGUF: {dst} ({os.path.getsize(dst) / 1e9:.2f} GB)")
    print(sh([*OLLAMA, "list"]))
    print(f"created {a.name} from {src}")


if __name__ == "__main__":
    main()
