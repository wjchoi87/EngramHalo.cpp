#!/usr/bin/env python3
"""GGUF 헤더 파서: 텐서 (name, n_dims, ne, type, offset) 목록 추출. 330GB 파일도 헤더만 읽음."""
import struct, sys

GGUF_MAGIC = 0x46554747  # 'GGUF'
TENSORTYPES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
    10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",
    18:"Q2_K_S",19:"Q3_K_S",20:"Q3_K_M",21:"Q4_K_S",22:"Q4_K_M",23:"Q3_K_L",24:"Q4_K_L",
    25:"Q5_K_M",26:"Q5_K_L",27:"Q6_K_L",28:"IQ2_S",29:"IQ4_NL",30:"IQ3_S",31:"IQ2_M",
    32:"IQ4_XS",33:"IQ1_S",34:"IQ4_NL4",36:"BF16",37:"Q4_0_4_4",38:"Q8_0_...", 100:"ROCMFP4",101:"ROCMFP4_FAST"}

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
                read_val(f, vtype)  # skip
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

if __name__ == "__main__":
    t, _ = parse_header(sys.argv[1])
    pat = sys.argv[2] if len(sys.argv) > 2 else ""
    for name in sorted(t):
        if pat in name:
            nd, ne, tt, off = t[name]
            print(f"{name:50s} ne={list(ne)} type={TENSORTYPES.get(tt, tt)} off={off}")
