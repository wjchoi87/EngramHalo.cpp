#!/usr/bin/env python3
"""qwen4_exp HF BF16 safetensors → GGUF(BF16) 직접 변환기 (numpy/torch 불요).

unsloth UD-Q4_K_XL GGUF의 KV 스키마·텐서 명명(EngramHalo 로더 검증済)을 1:1 복제하고
데이터는 BF16 원본을 바이트 수준으로 전달한다. 변환은 전부 바이트 범위/shape 메타 수준:
 - indexer.index_qk_proj [640,2560] → q_proj[512,2560] + k_proj[128,2560] (행 분리)
 - experts.gate_up_proj [512,1280,2560] → gate_exps[:, :640,:] + up_exps[:, 640:,:]
 - conv1d [C,1,K]→[C,K], shared_expert_gate [1,H]→[H] (연속 메모리라 shape만 변경)
 - ngram_embedding.shard_0..127 → per_layer_token_embd (102GB, 스트리밍 복사)
 - A_log → -exp(A_log) (f32, 48 floats — 유일한 산술 변환)
KV는 unsloth 샤드1의 67개를 타입 보존으로 그대로 재기록(tokenizer 포함).
"""
import json, os, struct, sys

SRC_DIR = '/home/wjchoi/models/Qwen3.8-Flash-Next-BF16'
REF_DIR = '/home/wjchoi/models/flashnext-gguf'
REF_SHARDS = [f'{REF_DIR}/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{i}-of-00004.gguf' for i in (1, 2, 3, 4)]
OUT = sys.argv[1] if len(sys.argv) > 1 else f'{REF_DIR}/Qwen3.8-Flash-Next-BF16-direct.gguf'
ALIGN = 32
GGUF_MAGIC = b'GGUF'

# ---------- GGUF 읽기 (헤더 전용) ----------
def _rstr(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')

def _rval(f, t, et_out=None):
    fmt = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<B', 10: '<Q', 11: '<q', 12: '<d'}
    if t == 8: return _rstr(f)
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        if et_out is not None: et_out.append(et)
        return [_rval(f, et) for _ in range(n)]
    s = struct.calcsize(fmt[t])
    return struct.unpack(fmt[t], f.read(s))[0]

def read_gguf(path, want_kv=True, want_tensors=True):
    f = open(path, 'rb')
    assert f.read(4) == GGUF_MAGIC
    ver, nt, nk = struct.unpack('<IQQ', f.read(20))
    kv = []
    if want_kv:
        for _ in range(nk):
            k = _rstr(f); t = struct.unpack('<I', f.read(4))[0]
            ets = []
            v = _rval(f, t, ets)
            kv.append((k, t, v, ets[0] if ets else None))
    else:
        f.seek(0)
    tensors = []
    if want_tensors:
        f = open(path, 'rb'); f.read(4)
        ver, nt, nk = struct.unpack('<IQQ', f.read(20))
        for _ in range(nk):
            k = _rstr(f); t = struct.unpack('<I', f.read(4))[0]; _rval(f, t)
        for _ in range(nt):
            name = _rstr(f); nd = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack('<' + 'Q' * nd, f.read(8 * nd))
            dt = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            tensors.append((name, dims, dt, off))
    f.close()
    return kv, tensors

# ---------- GGUF 쓰기 ----------
def _wstr(out, s):
    b = s.encode('utf-8')
    out.write(struct.pack('<Q', len(b))); out.write(b)

def _wval(out, t, v, et_hint=None):
    fmt = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<B', 10: '<Q', 11: '<q', 12: '<d'}
    if t == 8: _wstr(out, v); return
    if t == 9:
        et = et_hint if et_hint is not None else (4 if all(isinstance(x, int) for x in v) else 6)
        out.write(struct.pack('<I', et)); out.write(struct.pack('<Q', len(v)))
        for x in v: _wval(out, et, x)
        return
    out.write(struct.pack(fmt[t], v))

DT_F32, DT_BF16 = 0, 30
SAF_DTYPE_SIZE = {'BF16': 2, 'F16': 2, 'F32': 4, 'I32': 4, 'I64': 8, 'U8': 1, 'I8': 1}
SAF2GGUF = {'BF16': DT_BF16, 'F16': 1, 'F32': DT_F32, 'I32': 4, 'I64': 10, 'U8': 7, 'I8': 1}

# ---------- safetensors 헤더 ----------
def read_safetensors_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n).decode('utf-8'))
    data_start = 8 + n
    out = {}
    for name, m in hdr.items():
        if name == '__metadata__': continue
        s = m['data_offsets']
        out[name] = (m['dtype'], tuple(m['shape']), data_start + s[0], data_start + s[1])
    return out

def main():
    # 1) 참조 KV (shard 1) + 참조 텐서 dims (검증용)
    ref_kv, _ = read_gguf(REF_SHARDS[0], want_kv=True, want_tensors=False)
    print(f'ref KV: {len(ref_kv)} entries')
    # GDN head 순열 (48 v-heads = 16 k-groups × 3): llama 컨벤션 인터리브
    import numpy as np
    PERM48 = [3 * (i % 16) + i // 16 for i in range(48)]
    CONV_PERM = np.load('/tmp/conv_perm.npy').tolist()
    kv_map = {k: (t, v) for k, t, v, _e in ref_kv}
    # unsloth 참조 텐서 dtype (비양자 텐서의 저장 dtype을 그대로 강제 — norm류 f32 등)
    ref_dt = {}
    for i in (2, 3, 4):
        _kv, _ts = read_gguf(REF_SHARDS[i - 1], want_kv=False)
        for nm, _dm, dtt, _off in _ts:
            ref_dt[nm] = dtt

    # 2) safetensors 인덱스
    idx = json.load(open(f'{SRC_DIR}/model.safetensors.index.json'))
    wm = idx['weight_map']
    st_files = sorted(set(wm.values()))
    st = {}  # name -> (path, dtype, shape, off0, off1)
    for fn in st_files:
        p = f'{SRC_DIR}/{fn}'
        for name, (dt, shp, o0, o1) in read_safetensors_header(p).items():
            if name.startswith('model.visual') or name.startswith('mtp'):
                continue
            st[name] = (p, dt, shp, o0, o1)
    print(f'safetensors tensors (visual/mtp 제외): {len(st)}')

    # 3) 출력 텐서 목록 구성: (out_name, ne, gguf_dtype, spec)
    #    spec: ('copy', path, off0, off1) | ('exp', name) | ('splitrows', path, off0, mid, off1)
    out_tensors = []
    def gguf_ne(shape):  # numpy(HF) shape → GGUF ne (역순), 선행 1차원 제거
        if len(shape) == 2 and shape[0] == 1: shape = (shape[1],)
        return tuple(reversed(shape))

    def out_dtype(out_name, src_dt):
        # 참조에 있으면 참조 dtype(양자 dtype은 무시 — 비양자 저장 dtype만 따름), 없으면 소스 dtype
        rd = ref_dt.get(out_name)
        if rd in (DT_F32, DT_BF16, 1):
            return rd
        return SAF2GGUF[src_dt]

    def add_direct(out_name, src_name, expect_shape=None):
        p, dt, shp, o0, o1 = st[src_name]
        if expect_shape is not None:
            assert shp == expect_shape, f'{src_name}: {shp} != {expect_shape}'
        od = out_dtype(out_name, dt)
        if od == SAF2GGUF[dt]:
            out_tensors.append((out_name, gguf_ne(shp), od, ('copy', p, o0, o1)))
        elif SAF2GGUF[dt] == DT_BF16 and od == DT_F32:
            out_tensors.append((out_name, gguf_ne(shp), od, ('bf16f32', p, o0, o1)))
        else:
            raise ValueError(f'{out_name}: dtype 변환 미지원 {dt} -> {od}')

    add_direct('token_embd.weight', 'model.language_model.embed_tokens.weight')
    add_direct('output.weight', 'lm_head.weight')
    add_direct('output_hc_norm.weight', 'model.language_model.hyper_connection_mixer.hc_norm.weight')
    add_direct('output_hc_down.weight', 'model.language_model.hyper_connection_mixer.input_mix_weight_down.weight')
    add_direct('output_hc_up.weight', 'model.language_model.hyper_connection_mixer.input_mix_weight_up.weight')

    PLE_LAYER = 1
    for N in range(48):
        pre = f'model.language_model.layers.{N}.'
        b = f'blk.{N}.'
        _p, _dt, _shp, _o0, _o1 = st[pre + 'attn_hyper_connection.hc_norm.weight']
        out_tensors.append((b + 'hc_attn_norm.weight', (10240,), DT_F32, ('bf16f32plus1', _p, _o0, _o1)))
        add_direct(b + 'hc_attn_down.weight', pre + 'attn_hyper_connection.input_mix_weight_down.weight')
        add_direct(b + 'hc_attn_up.weight', pre + 'attn_hyper_connection.input_mix_weight_up.weight')
        add_direct(b + 'hc_attn_inject.weight', pre + 'attn_hyper_connection.block_inject_weight.weight')
        _p, _dt, _shp, _o0, _o1 = st[pre + 'mlp_hyper_connection.hc_norm.weight']
        out_tensors.append((b + 'hc_ffn_norm.weight', (10240,), DT_F32, ('bf16f32plus1', _p, _o0, _o1)))
        add_direct(b + 'hc_ffn_down.weight', pre + 'mlp_hyper_connection.input_mix_weight_down.weight')
        add_direct(b + 'hc_ffn_up.weight', pre + 'mlp_hyper_connection.input_mix_weight_up.weight')
        add_direct(b + 'hc_ffn_inject.weight', pre + 'mlp_hyper_connection.block_inject_weight.weight')
        if N % 4 == 3:  # full attention layer
            add_direct(b + 'attn_q.weight', pre + 'self_attn.q_proj.weight')
            add_direct(b + 'attn_k.weight', pre + 'self_attn.k_proj.weight')
            add_direct(b + 'attn_v.weight', pre + 'self_attn.v_proj.weight')
            add_direct(b + 'attn_output.weight', pre + 'self_attn.o_proj.weight')
            _p, _dt, _shp, _o0, _o1 = st[pre + 'self_attn.q_norm.weight']
            out_tensors.append((b + 'attn_q_norm.weight', (256,), DT_F32, ('bf16f32plus1', _p, _o0, _o1)))
            _p, _dt, _shp, _o0, _o1 = st[pre + 'self_attn.k_norm.weight']
            out_tensors.append((b + 'attn_k_norm.weight', (256,), DT_F32, ('bf16f32plus1', _p, _o0, _o1)))
            # index_qk_proj [640, 2560] → q[512] + k[128] (행 분리: out dim 기준)
            p, dt, shp, o0, o1 = st[pre + 'self_attn.indexer.index_qk_proj.weight']
            assert shp == (640, 2560), shp
            esz = SAF_DTYPE_SIZE[dt]
            mid = o0 + 512 * 2560 * esz
            out_tensors.append((b + 'indexer.q_proj.weight', (2560, 512), out_dtype(b + 'indexer.q_proj.weight', dt), ('copy', p, o0, mid)))
            out_tensors.append((b + 'indexer.k_proj.weight', (2560, 128), out_dtype(b + 'indexer.k_proj.weight', dt), ('copy', p, mid, o1)))
            add_direct(b + 'indexer.q_norm.weight', pre + 'self_attn.indexer.q_layernorm.weight')
            add_direct(b + 'indexer.k_norm.weight', pre + 'self_attn.indexer.k_layernorm.weight')
        else:  # GDN layer
            p, dt_, shp_, o0_, o1_ = st[pre + 'linear_attn.in_proj_qkv.weight']
            specs_qkv = [(p, o0_, o0_ + 4096 * 2560 * 2)]
            rz = 128 * 2560 * 2
            vbase = o0_ + 4096 * 2560 * 2
            for g in range(48):
                j = 3 * (g % 16) + g // 16
                specs_qkv.append((p, vbase + j * rz, vbase + (j + 1) * rz))
            out_tensors.append((b + 'attn_qkv.weight', (2560, 10240), DT_BF16, ('multi', specs_qkv)))
            # in_proj_z: v-head (16,3) tiled 재순서 — 행 단위 (48×128)
            p, dt_, shp_, o0_, o1_ = st[pre + 'linear_attn.in_proj_z.weight']
            assert dt_ == 'BF16'
            specs_z = []
            rz = 128 * 2560 * 2
            base_z = o0_
            for g in range(48):
                j = 3 * (g % 16) + g // 16
                specs_z.append((p, base_z + j * rz, base_z + (j + 1) * rz))
            out_tensors.append((b + 'attn_gate.weight', (2560, 6144), DT_BF16, ('multi', specs_z)))
            out_tensors.append((b + 'ssm_beta.weight', (2560, 48), DT_F32, ('gdnproj', pre + 'linear_attn.in_proj_b.weight')))
            out_tensors.append((b + 'ssm_alpha.weight', (2560, 48), DT_F32, ('gdnproj', pre + 'linear_attn.in_proj_a.weight')))
            p, dt, shp, o0, o1 = st[pre + 'linear_attn.out_proj.weight']
            assert dt == 'BF16'
            out_tensors.append((b + 'ssm_out.weight', (6144, 2560), DT_BF16, ('ssmout', p, o0, o1)))
            p, dt, shp, o0, o1 = st[pre + 'linear_attn.A_log']
            out_tensors.append((b + 'ssm_a', (shp[0] if len(shp) == 1 else shp[-1],), DT_F32, ('ssma', pre + 'linear_attn.A_log')))
            p, dt_, shp_, o0_, o1_ = st[pre + 'linear_attn.dt_bias']
            out_tensors.append((b + 'ssm_dt.bias', (48,), DT_F32, ('dtperm', pre + 'linear_attn.dt_bias')))
            add_direct(b + 'ssm_norm.weight', pre + 'linear_attn.norm.weight')
            p, dt, shp, o0, o1 = st[pre + 'linear_attn.conv1d.weight']
            assert shp[1] == 1, shp
            out_tensors.append((b + 'ssm_conv1d.weight', (4, 10240), DT_F32, ('convperm', p, o0, o1)))
        add_direct(b + 'ffn_gate_inp.weight', pre + 'mlp.gate.weight')
        add_direct(b + 'ffn_gate_inp_shexp.weight', pre + 'mlp.shared_expert_gate.weight')  # shape [2560] 가정
        add_direct(b + 'ffn_gate_shexp.weight', pre + 'mlp.shared_expert.gate_proj.weight')
        add_direct(b + 'ffn_up_shexp.weight', pre + 'mlp.shared_expert.up_proj.weight')
        add_direct(b + 'ffn_down_shexp.weight', pre + 'mlp.shared_expert.down_proj.weight')
        # experts gate_up split: HF [E=512, 2ff=1280, in=2560] row-major.
        # expert 블록이 연속이므로 gate([e,:640,:]) / up([e,640:,:])는 expert별 2-range 복사로 분리.
        p, dt, shp, o0, o1 = st[pre + 'mlp.experts.gate_up_proj']
        assert shp == (512, 1280, 2560), shp
        esz = SAF_DTYPE_SIZE[dt]
        e_block = 1280 * 2560 * esz
        g_half = 640 * 2560 * esz
        specs_gate, specs_up = [], []
        for e in range(512):
            b0 = o0 + e * e_block
            specs_gate.append((p, b0, b0 + g_half))
            specs_up.append((p, b0 + g_half, b0 + e_block))
        out_tensors.append((b + 'ffn_gate_exps.weight', (2560, 640, 512), out_dtype(b + 'ffn_gate_exps.weight', dt), ('multi', specs_gate)))
        out_tensors.append((b + 'ffn_up_exps.weight', (2560, 640, 512), out_dtype(b + 'ffn_up_exps.weight', dt), ('multi', specs_up)))
        add_direct(b + 'ffn_down_exps.weight', pre + 'mlp.experts.down_proj')
        if N == PLE_LAYER:
            add_direct(b + 'ple_key.weight', pre + 'ple.key_proj.weight')
            add_direct(b + 'ple_value.weight', pre + 'ple.value_proj.weight')
            p, dt, shp, o0, o1 = st[pre + 'ple.conv1d.weight']
            assert shp[1] == 1, shp
            _od = out_dtype(b + 'ple_conv1d.weight', dt)
            _spec = ('copy', p, o0, o1) if _od == SAF2GGUF[dt] else ('bf16f32', p, o0, o1)
            out_tensors.append((b + 'ple_conv1d.weight', gguf_ne((shp[0], shp[2])), _od, _spec))
            _p, _dt, _shp, _o0, _o1 = st[pre + 'ple.norm_key.weight']
            out_tensors.append((b + 'ple_norm_key.weight', (10240,), DT_F32, ('bf16f32plus1', _p, _o0, _o1)))
            _p, _dt, _shp, _o0, _o1 = st[pre + 'ple.norm_conv.weight']
            out_tensors.append((b + 'ple_norm_conv.weight', (10240,), DT_F32, ('bf16f32plus1', _p, _o0, _o1)))
            _p, _dt, _shp, _o0, _o1 = st[pre + 'ple.norm_query.weight']
            out_tensors.append((b + 'ple_norm_query.weight', (10240,), DT_F32, ('bf16f32plus1', _p, _o0, _o1)))

    # PLE table: shard 0..127 concat → per_layer_token_embd
    ple_base = f'model.language_model.layers.{PLE_LAYER}.ple.ple_embedding'
    ple_srcs = []
    total_rows = 0
    for s in range(128):
        p, dt, shp, o0, o1 = st[f'{ple_base}.ngram_embedding.shard_{s}.weight']
        assert dt == 'BF16' and shp[1] == 160, (dt, shp)
        ple_srcs.append((p, o0, o1))
        total_rows += shp[0]
    head_vocab = kv_map['qwen4exp.ple.head_vocab_sizes'][1]
    head_off = kv_map['qwen4exp.ple.head_offsets'][1]
    # 물리 rows는 논리 vocab 합과 다를 수 있다(헤드 패딩) — unsloth 참조 테이블 rows와 일치해야 한다
    _, ref_dims, _dt, _off = None, None, None, None
    for i in (2, 3, 4):
        _kv, _ts = read_gguf(REF_SHARDS[i - 1], want_kv=False)
        for nm, dm, dtt, off in _ts:
            if nm == 'per_layer_token_embd.weight': ref_dims = dm
        if ref_dims: break
    assert ref_dims and ref_dims[1] == total_rows, (ref_dims, total_rows)
    print(f'PLE rows={total_rows} (ref={list(ref_dims)}) 물리 테이블 일치')
    out_tensors.append(('per_layer_token_embd.weight', (160, total_rows), DT_BF16, ('multi', ple_srcs)))
    print(f'출력 텐서: {len(out_tensors)} (PLE rows={total_rows})')

    # 4) 데이터 오프셋 계산 (PLE multi 제외 본체 순서, PLE는 마지막에 이미 배치됨)
    placed = []
    off = 0
    for name, ne, dt, spec in out_tensors:
        n_elems = 1
        for d in ne: n_elems *= d
        esz = {DT_F32: 4, DT_BF16: 2, 1: 2, 4: 4, 10: 8}[dt]
        sz = n_elems * esz
        pad = (-off) % ALIGN
        placed.append((name, ne, dt, spec, off + pad, sz))
        off += pad + sz
    total_data = off

    # 6) 쓰기
    with open(OUT, 'wb') as out:
        out.write(GGUF_MAGIC)
        n_kv_out = sum(1 for k, t, v, et in ref_kv if not k.startswith('split.'))
        out.write(struct.pack('<IQQ', 3, len(placed), n_kv_out))
        for k, t, v, et in ref_kv:
            if k.startswith('split.'):
                continue
            _wstr(out, k); out.write(struct.pack('<I', t)); _wval(out, t, v, et)
        for name, ne, dt, spec, toff, sz in placed:
            _wstr(out, name)
            out.write(struct.pack('<I', len(ne)))
            out.write(struct.pack('<' + 'Q' * len(ne), *ne))
            out.write(struct.pack('<I', dt))
            out.write(struct.pack('<Q', toff))
        hdr_end = out.tell()
        data_base = (hdr_end + ALIGN - 1) // ALIGN * ALIGN
        out.write(b'\x00' * (data_base - hdr_end))
        # 데이터 기록 (파일 순 스트리밍)
        open_files = {}
        def src_f(p):
            if p not in open_files: open_files[p] = open(p, 'rb', buffering=0)
            return open_files[p]
        def copy_range(f, o0, nbytes):
            f.seek(o0)
            remaining = nbytes
            while remaining > 0:
                chunk = f.read(min(1 << 24, remaining))
                if not chunk: raise EOFError('source shorter than expected')
                out.write(chunk)
                remaining -= len(chunk)
        for i, (name, ne, dt, spec, toff, sz) in enumerate(placed):
            cur = out.tell() - data_base
            pad = toff - cur
            if i < 30 or pad < 0:
                print(f'  DBG[{i}] {name} toff={toff} cur={cur} sz={sz} pad={pad}')
            if pad > 0: out.write(b'\x00' * pad)
            assert pad >= 0, f'{name}: negative pad — 순서 역행'
            if spec[0] == 'copy':
                _, p, o0, o1 = spec
                copy_range(src_f(p), o0, o1 - o0)
            elif spec[0] == 'bf16f32':
                _, p, o0, o1 = spec
                import numpy as np
                n = o1 - o0
                f = src_f(p); f.seek(o0)
                raw = np.frombuffer(f.read(n), dtype='<u2')
                out.write((raw.astype('<u4') << 16).view('<f4').tobytes())
            elif spec[0] == 'dtperm':
                _, sname = spec
                p, dt_, shp_, o0_, o1_ = st[sname]
                f = src_f(p); f.seek(o0)
                if dt_ == 'BF16':
                    a = (np.frombuffer(f.read(96), dtype='<u2').astype('<u4') << 16).view('<f4')
                else:
                    a = np.frombuffer(f.read(192), dtype='<f4')
                v = -a * 0 + a  # dt_bias는 순열만 적용 (부호 변환 없음)
                v = v[PERM48]
                out.write(v.astype('<f4').tobytes())
            elif spec[0] == 'bf16f32plus1':
                _, p, o0, o1 = spec
                f = src_f(p); f.seek(o0)
                raw = np.frombuffer(f.read(o1 - o0), dtype='<u2')
                out.write(((raw.astype('<u4') << 16).view('<f4') + 1.0).astype('<f4').tobytes())
            elif spec[0] == 'ssma':
                _, sname = spec
                p, dt_, shp, o0, o1 = st[sname]
                n = 1
                for d in shp: n *= d
                f = src_f(p); f.seek(o0)
                if dt_ == 'BF16':
                    a = (np.frombuffer(f.read(n * 2), dtype='<u2').astype('<u4') << 16).view('<f4')
                else:
                    a = np.frombuffer(f.read(n * 4), dtype='<f4')
                v = -np.exp(a.astype('<f4'))
                v = v[PERM48]
                out.write(v.astype('<f4').tobytes())
            elif spec[0] == 'gdnproj':
                _, sname = spec
                p, dt_, shp_, o0, o1 = st[sname]
                f = src_f(p); f.seek(o0)
                a = (np.frombuffer(f.read(o1 - o0), dtype='<u2').astype('<u4') << 16).view('<f4').reshape(48, 2560)
                out.write(a[PERM48].astype('<f4').tobytes())
            elif spec[0] == 'ssmout':
                _, p, o0, o1 = spec
                f = src_f(p); f.seek(o0)
                a = np.frombuffer(f.read(o1 - o0), dtype='<u2').reshape(2560, 48, 128)
                a = a[:, PERM48, :]
                out.write(a.reshape(-1).astype('<u2').tobytes())
            elif spec[0] == 'convperm':
                _, p, o0, o1 = spec
                f = src_f(p); f.seek(o0)
                raw = np.frombuffer(f.read(o1 - o0), dtype='<u2')
                a = (raw.astype('<u4') << 16).view('<f4').reshape(10240, 4)
                out.write(a[CONV_PERM].astype('<f4').tobytes())
            elif spec[0] == 'exp':
                _, sname = spec
                p, dt_, shp, o0, o1 = st[sname]
                import numpy as np
                n = 1
                for d in shp: n *= d
                f = src_f(p); f.seek(o0)
                if dt_ == 'BF16':
                    raw = np.frombuffer(f.read(n * 2), dtype='<u2')
                    a = (raw.astype('<u4') << 16).view('<f4')
                elif dt_ == 'F16':
                    raw = np.frombuffer(f.read(n * 2), dtype='<f2')
                    a = raw.astype('<f4')
                else:
                    a = np.frombuffer(f.read(n * 4), dtype='<f4')
                out.write((-np.exp(a.astype('<f4'))).astype('<f4').tobytes())
            elif spec[0] == 'multi':
                cur2 = out.tell()
                for p, o0, o1 in spec[1]:
                    copy_range(src_f(p), o0, o1 - o0)
                assert out.tell() - cur2 == sz
            if (i + 1) % 100 == 0:
                print(f'  [{i+1}/{len(placed)}] {name} @{out.tell()/2**30:.1f}GiB')
    print(f'완료: {OUT} {os.path.getsize(OUT)/2**30:.1f}GiB, data={total_data/2**30:.1f}GiB')

if __name__ == '__main__':
    main()
