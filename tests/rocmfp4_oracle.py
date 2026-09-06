#!/usr/bin/env python3
"""Canonical ROCmFP4 oracle — 22c4f74 reference dequant, byte-exact.

Extracts real blocks from the production GGUF, dequantizes with the reference
algorithm, and writes (w_bytes, w_ref, act, gpu_out_ref) binary files for the
standalone harness to consume.
"""
import struct, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')

# --- GGUF header parser (from gguf_tensor_list.py) ---
GGUF_MAGIC = 0x46554747
def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")
def read_val(f, vtype):
    if vtype == 0: return struct.unpack("<B", f.read(1))[0]
    if vtype == 1: return struct.unpack("<b", f.read(1))[0]
    if vtype == 2: return struct.unpack("<H", f.read(2))[0]
    if vtype == 3: return struct.unpack("<h", f.read(2))[0]
    if vtype == 4: return struct.unpack("<I", f.read(4))[0]
    if vtype == 5: return struct.unpack("<i", f.read(4))[0]
    if vtype == 6: return struct.unpack("<f", f.read(4))[0]
    if vtype == 7: return struct.unpack("<B", f.read(1))[0]
    if vtype == 8: return read_str(f)
    if vtype == 9:
        (et, n) = struct.unpack("<IQ", f.read(12))
        return [read_val(f, et) for _ in range(n)]
    if vtype == 10: return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11: return struct.unpack("<q", f.read(8))[0]
    if vtype == 12: return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"bad vtype {vtype}")

def parse_header(path):
    tensors = {}
    with open(path, "rb") as f:
        (magic, version, n_tensors, n_kv) = struct.unpack("<IIQQ", f.read(24))
        assert magic == GGUF_MAGIC, hex(magic)
        alignment = 32
        for _ in range(n_kv):
            key = read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if key == "general.alignment":
                alignment = read_val(f, vtype)
            else:
                read_val(f, vtype)
        for _ in range(n_tensors):
            name = read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            ne = struct.unpack(f"<{'Q'*n_dims}", f.read(8*n_dims))
            (ttype,) = struct.unpack("<I", f.read(4))
            (off,) = struct.unpack("<Q", f.read(8))
            tensors[name] = (n_dims, ne, ttype, off)
        header_end = f.tell()
    data_base = (header_end + alignment - 1) // alignment * alignment
    return tensors, data_base

# --- Reference UE4M3 decode (22c4f74 rocmfp4.c, half-scale convention) ---
CB = np.array([0,1,2,3,4,6,8,10, 0,-1,-2,-3,-4,-6,-8,-10], dtype=np.float32)
def ue4m3_to_fp32_half(v):
    exp = (np.int32(v) >> 3) & 0x0f
    man = np.int32(v) & 0x07
    # CPU reference: e==0x7f or e==0xff → 0.0f (explicit)
    # But HIP _finite version: no such check, just exp==0 → subnormal
    # For canonical oracle: use CPU reference (includes the 0x7f/0xff → 0)
    if v == 0x7f or v == 0xff:
        return np.float32(0.0)
    if exp == 0:
        return np.float32(man) * np.float32(1.0/1024.0)
    bits = (np.uint32(exp + 119) << np.uint32(23)) | (np.uint32(man) << np.uint32(20))
    return bits.view(np.float32) if hasattr(bits, 'view') else np.array([bits],dtype=np.uint32).view(np.float32)[0]

def dequant_block_18(blk_bytes):
    """22c4f74 reference: 18-byte block → 32 f32 values."""
    qs = np.frombuffer(bytes(blk_bytes[:16]), dtype=np.uint8)
    d0 = ue4m3_to_fp32_half(blk_bytes[16])
    d1 = ue4m3_to_fp32_half(blk_bytes[17])
    out = np.zeros(32, dtype=np.float32)
    for j in range(16):
        lo = int(qs[j] & 0x0f)
        hi = int(qs[j] >> 4)
        out[j]      = np.float32(CB[lo]) * d0
        out[j+16]   = np.float32(CB[hi]) * d1
    return out

def dequant_row(raw_18b_rows):
    """raw: (n_blocks, 18) uint8 → (n_blocks*32,) f32"""
    nb = raw_18b_rows.shape[0]
    out = np.zeros(nb*32, dtype=np.float32)
    for b in range(nb):
        out[b*32:(b+1)*32] = dequant_block_18(raw_18b_rows[b])
    return out

if __name__ == "__main__":
    GGUF = "/home/wjchoi/models/flashnext-gguf/Qwen3.8-Flash-Next-ROCmFP4-direct.gguf"
    tensors, db = parse_header(GGUF)
    f = open(GGUF, "rb")

    # ple_key.weight: ne=[2560, 10240], type=ROCMFP4
    # K=2560 (elements per row), N=10240 (rows)
    # Row r occupies blocks [r*K/32, (r+1)*K/32) = r*80 blocks of 18 bytes.
    name = "blk.1.ple_key.weight"
    nd, ne, tt, off = tensors[name]
    K, N = ne[0], ne[1]
    nb_per_row = K // 32
    row_bytes = nb_per_row * 18

    f.seek(db + off)
    # Extract first 8 rows (8×2560 = 20480 elements) for the harness.
    n_rows = 8
    raw = np.frombuffer(f.read(n_rows * row_bytes), dtype=np.uint8).reshape(n_rows, nb_per_row, 18)
    f.close()

    print(f"{name}: K={K} N={N} rows_extracted={n_rows}")

    # Dequantize each row → (n_rows, K) f32 matrix.
    W = np.zeros((n_rows, K), dtype=np.float32)
    for r in range(n_rows):
        W[r] = dequant_row(raw[r])  # dequant_row takes (nb,18) → K floats

    print(f"W shape: {W.shape}, non-finite: {np.count_nonzero(~np.isfinite(W))}, absmax: {np.abs(W).max():.6g}")

    # Deterministic activation: [K, M] — same formula as standalone harness.
    M_max = 128
    act = np.zeros((K, M_max), dtype=np.float32)
    for i in range(K * M_max):
        act.flat[i] = np.float32((i * 2654435761 % 2001) - 1000) / 1000.0

    # CPU reference matmul: ref[n, m] = sum_k W[n,k] * act[k,m]
    REF = (W @ act).astype(np.float32)
    print(f"REF shape: {REF.shape}, non-finite: {np.count_nonzero(~np.isfinite(REF))}, absmax: {np.abs(REF).max():.6g}")

    # Write binary files for the standalone harness.
    np.concatenate([raw.reshape(-1)]).tofile("/tmp/oracle_w.bin")
    act.tofile("/tmp/oracle_act.bin")
    REF.tofile("/tmp/oracle_ref.bin")
    print(f"wrote: /tmp/oracle_w.bin ({raw.nbytes}B), /tmp/oracle_act.bin ({act.nbytes*4}B), /tmp/oracle_ref.bin ({REF.nbytes*4}B)")
