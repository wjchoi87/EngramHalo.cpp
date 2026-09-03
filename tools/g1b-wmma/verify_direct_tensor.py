#!/usr/bin/env python3
"""direct-ROCmFP4 텐서를 디퀀트하여 BF16-direct(F32 저장) 원본과 값 비교 — 변환기 독립 검증."""
import struct, sys, math
from gguf_tensor_list import parse_header

ROCMFP4_FILE = "/home/wjchoi/models/flashnext-gguf/Qwen3.8-Flash-Next-ROCmFP4-direct.gguf"
BF16_FILE    = "/home/wjchoi/models/flashnext-gguf/Qwen3.8-Flash-Next-BF16-direct.gguf"
CB = [0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10]

def ue4m3_full(x):
    e = (x >> 3) & 0xF; m = x & 7; s = -1.0 if (x & 0x80) else 1.0
    if e == 0:
        return s * m * (2.0**-9)      # 프로덕션 half=m/1024 ×2
    return s * (1.0 + m/8.0) * (2.0**(e-7))

def dequant_rocmfp4(buf, n_elems):
    out = []
    nblocks = len(buf) // 18
    for b in range(nblocks):
        base = b*18
        for half in range(2):
            sc = ue4m3_full(buf[base + 16 + half])
            for i in range(16):
                byte = buf[base + half*8 + i//2]
                nib = (byte >> 4) if (i & 1) else (byte & 0xF)
                out.append(CB[nib] * sc)
    return out[:n_elems]

def data_start(path, alignment=32):
    import os
    with open(path, "rb") as f:
        struct.unpack("<IIQQ", f.read(24))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_kv):
            read_str(f); (vt,) = struct.unpack("<I", f.read(4)); read_val(f, vt)
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_tensors):
            read_str(f); f.read(4); f.read(8*struct.unpack("<I", f.read(4))[0] if False else 0)
            # tensor info: n_dims(I) 다시 읽기
        # 위 루프는 파서 재구현 대신 parse_header 후 위치 추적이 필요하므로 아래에서 재파싱
    return None

def read_tensor(path, meta, name, base=0):
    nd, ne, tt, off = meta[name]
    n = 1
    for d in ne: n *= d
    with open(path, "rb") as f:
        f.seek(base + off)
        raw = f.read(n*4 if tt == 0 else n//32*18)
    if tt == 0:  # F32
        return list(struct.unpack(f"<{n}f", raw))
    return dequant_rocmfp4(raw, n)

def compare(name):
    ra, base_a = parse_header(ROCMFP4_FILE)
    rb, base_b = parse_header(BF16_FILE)
    a = read_tensor(ROCMFP4_FILE, ra, name, base_a)
    b = read_tensor(BF16_FILE, rb, name, base_b)
    n = len(a)
    maxabs = 0.0; maxrel = 0.0; sumsq = 0.0; refsq = 0.0; big = 0
    for x, y in zip(a, b):
        d = abs(x - y); ry = abs(y)
        maxabs = max(maxabs, d)
        if ry > 1e-6: maxrel = max(maxrel, d/ry)
        sumsq += d*d; refsq += y*y
        if d > max(0.02*ry, 0.05): big += 1
    rel = math.sqrt(sumsq/refsq) if refsq > 0 else 0
    print(f"{name}: n={n} maxabs={maxabs:.4f} maxrel={maxrel:.3f} rms_rel={rel:.4f} outliers={big}/{n}")

if __name__ == "__main__":
    for name in sys.argv[1:]:
        compare(name)
