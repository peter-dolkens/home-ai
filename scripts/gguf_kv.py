"""Rewrite one array-valued metadata key of a GGUF; the tensor infos and data are copied unchanged. llama.cpp's
llama-quantize refuses Ollama's qwen35 GGUFs because Ollama writes qwen35.rope.dimension_sections with 3 entries and
llama.cpp expects 4 (scripts/falcon-phase2-build.sh pads it for llama-quantize, then restores Ollama's 3 on the output).
    python3 scripts/gguf_kv.py in.gguf out.gguf qwen35.rope.dimension_sections 11,11,10,0
"""
import shutil
import struct
import sys

FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def skip(f, t):
    if t == 8:
        f.read(rd(f, "<Q"))
    elif t == 9:
        at, n = rd(f, "<I"), rd(f, "<Q")
        for _ in range(n):
            skip(f, at)
    else:
        f.read(struct.calcsize(FMT[t]))


def main(src, dst, key, vals):
    f = open(src, "rb")
    assert f.read(4) == b"GGUF", "not a GGUF"
    ver, nt, nk = rd(f, "<I"), rd(f, "<Q"), rd(f, "<Q")
    out, align, found = bytearray(b"GGUF" + struct.pack("<IQQ", ver, nt, nk)), 32, False
    for _ in range(nk):
        start = f.tell()
        k = f.read(rd(f, "<Q")).decode()
        t = rd(f, "<I")
        at = rd(f, "<I") if t == 9 else None
        if at is not None:
            f.seek(-4, 1)
        vstart = f.tell()
        skip(f, t)
        end = f.tell()
        if k == "general.alignment":
            f.seek(vstart); align = rd(f, "<I"); f.seek(end)
        if k == key:
            assert t == 9 and at in FMT, "numeric array keys only"
            items = [float(x) if at in (6, 12) else int(x) for x in vals.split(",")]
            out += struct.pack("<Q", len(k.encode())) + k.encode() + struct.pack("<IIQ", 9, at, len(items))
            out += b"".join(struct.pack(FMT[at], x) for x in items)
            found = True
        else:
            f.seek(start); out += f.read(end - start)
    if not found:
        raise SystemExit(f"{key} not in {src}")
    start = f.tell()
    for _ in range(nt):  # tensor infos: name, n_dims, dims, type, offset (offsets are relative to the data section)
        f.read(rd(f, "<Q")); nd = rd(f, "<I"); f.read(8 * nd + 12)
    end = f.tell()
    f.seek(start); out += f.read(end - start)
    out += b"\0" * (-len(out) % align)
    with open(dst, "wb") as o:
        o.write(out)
        f.seek(-(-end // align) * align)
        shutil.copyfileobj(f, o, 1 << 24)
    print(f"{key} = {vals} -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
