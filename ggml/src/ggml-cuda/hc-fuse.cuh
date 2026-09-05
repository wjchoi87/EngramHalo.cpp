#pragma once

// Qwen4Exp hyper-connection prefill 체인의 elementwise 퓨전 커널 (Task C).
// 모든 커널은 unfused op 시퀀스와 동일 연산·동일 순서(f32) — bit-exact 목표.
// 게이트: GGML_CUDA_HCFUSION (기본 OFF, 승인 후 기본 ON 전환).

#include "common.cuh"
#include "unary.cuh"

// K1: stream collapse + scale
//   unfused: mixed = cont(view0); mixed += view1; += view2; += view3; mixed *= s
//   g: gated3 [E, C, T] contiguous, out: [E, T]
static __global__ void hc_collapse_f32(const float * __restrict__ g, float * __restrict__ out,
                                       const int E, const int C, const int n, const float s) {
    const int64_t idx = blockIdx.x*(int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int e    = idx % E;
    const int64_t ro = (int64_t)(idx / E) * E * C;
    float acc = g[e + ro];
    for (int c = 1; c < C; ++c) {
        acc = acc + g[e + c*E + ro];   // 순서 보존: 좌→우 연속 덧셈
    }
    out[idx] = acc * s;
}

// K2: comb_w — w = s2 * sigmoid(x * s1)
static __global__ void hc_comb_w_f32(const float * __restrict__ x, float * __restrict__ out,
                                     const int64_t n, const float s1, const float s2) {
    const int64_t idx = blockIdx.x*(int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float t  = __fmul_rn(x[idx], s1);
    const float w0 = __fdiv_rn(1.0f, 1.0f + expf(-t));
    out[idx] = __fmul_rn(w0, s2);
}

// K3: comb_fma — out[e,c,t] = res[e,c,t] + b[e,t] * w[c,t]
//   res [E,C,T], b [E,T], w [C,T] (flat contiguous 각각)
static __global__ void hc_comb_fma_f32(const float * __restrict__ res, const float * __restrict__ b,
                                       const float * __restrict__ w, float * __restrict__ out,
                                       const int E, const int C, const int64_t n) {
    const int64_t idx = blockIdx.x*(int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int e  = idx % E;
    const int64_t ct = idx / E;
    const int c  = ct % C;
    const int64_t t  = ct / C;
    const float m  = __fmul_rn(b[e + t*E], w[c + t*C]);
    out[idx] = __fadd_rn(res[idx], m);
}

// K4: gated — out = xn * sigmoid(mm)
static __global__ void hc_sigm_mul_f32(const float * __restrict__ xn, const float * __restrict__ mm,
                                       float * __restrict__ out, const int64_t n) {
    const int64_t idx = blockIdx.x*(int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g = __fdiv_rn(1.0f, 1.0f + expf(-mm[idx]));
    out[idx] = __fmul_rn(xn[idx], g);
}

// K5: lo = silu(mm * s)
static __global__ void hc_silu_scale_f32(const float * __restrict__ mm, float * __restrict__ out,
                                         const int64_t n, const float s) {
    const int64_t idx = blockIdx.x*(int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float t = __fmul_rn(mm[idx], s);
    out[idx] = __fdiv_rn(t, 1.0f + expf(-t));
}

// 게이트: GGML_CUDA_HCFUSION=1로 opt-in (기본 OFF).
// Task C 판정: cgraph 인접성 문제로 C2만 발동, end-to-end 개선 <10% → 실패 판정(§121-a)에 따라 기본 OFF.
// 재개 조건: cgraph 인접성 확정(빌더 재배선 검증) 후 기본 ON 전환 검토.
inline bool hc_fuse_enabled() {
    static const bool enabled = [] {
        const char * t = getenv("GGML_CUDA_HCFUSION");
        return t && t[0] && !(t[0] == '0' && t[1] == '\0');
    }();
    return enabled;
}
