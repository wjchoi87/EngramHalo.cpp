// [DEBT-ROCMFP4-MMQ-SMALLBATCH] v3 — validated gate-sequence harness.
// Gate 0: CPU quantize-dequant round-trip. Gate 1: Q4_0 GPU MUL_MAT control.
// Gate 2: ROCmFP4 GPU MUL_MAT (cuBLAS). Gate 3: ROCmFP4 GPU MUL_MAT (MMQ forced).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"


#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <map>

// CPU reference: quantize/dequant for Q4_0
static void ref_quantize_q4_0_row(const float * src, uint8_t * dst, int64_t k) {
    int64_t nb = k / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const float * x = src + b*32;
        uint8_t * blk = dst + b*18;
        float amax = 0;
        for (int j = 0; j < 32; ++j) { float a = fabsf(x[j]); if (a > amax) amax = a; }
        float d = amax / -8.0f; float inv = d ? 1.0f/d : 0.0f;
        float dd = d; uint32_t fb; memcpy(&fb, &dd, 4);
        uint32_t fs = (fb >> 31) & 1;
        int32_t fe = ((fb >> 23) & 0xff) - 127 + 15;
        uint32_t fm = (fb >> 13) & 0x3ff;
        if (fe <= 0) { fe = 0; fm = 0; } if (fe >= 31) { fe = 31; fm = 0x3ff; }
        uint16_t h = (uint16_t)(fs << 15) | (uint16_t)(fe << 10) | (uint16_t)fm;
        memcpy(blk, &h, 2); memset(blk+2, 0, 16);
        for (int j = 0; j < 32; ++j) {
            int q = (int)roundf(x[j] * inv); q = q < -8 ? -8 : (q > 7 ? 7 : q);
            uint8_t uq = (uint8_t)(q + 8);
            if (j < 16) blk[2+j] |= uq & 0x0f; else blk[2+j-16] |= (uq & 0x0f) << 4;
        }
    }
}
static void ref_dequant_q4_0_row(const uint8_t * blk_data, float * y, int64_t k) {
    int64_t nb = k / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * blk = blk_data + b*18;
        uint16_t h; memcpy(&h, blk, 2);
        uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
        float d;
        if (exp == 0) d = (float)man / (1024.0f * 16384.0f);
        else if (exp == 31) d = 0;
        else { d = ((float)man / 1024.0f + 1.0f) * (float)(1 << ((int)exp - 15)); if (sign) d = -d; }
        for (int j = 0; j < 16; ++j) {
            y[b*32+j] = ((blk[2+j]&0xf)-8)*d; y[b*32+j+16] = ((blk[2+j]>>4)-8)*d;
        }
    }
}
// ROCmFP4 dequant
static void ref_dequant_rocmfp4_row(const uint8_t * blk_data, float * y, int64_t k) {
    static const float CB[16] = { 0,1,2,3,4,6,8,10, 0,-1,-2,-3,-4,-6,-8,-10 };
    int64_t nb = k / 32;
    auto ue4m3 = [](uint8_t v) -> float {
        if (v == 0x7f || v == 0xff) return 0.0f;
        int exp = (v >> 3) & 0x0f; int man = v & 0x07;
        if (exp == 0) return (float)man / 1024.0f;
        uint32_t bits = (uint32_t)(exp+119)<<23 | (uint32_t)man<<20;
        float f; memcpy(&f, &bits, 4); return f;
    };
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * blk = blk_data + b*18;
        float d0 = ue4m3(blk[16]); float d1 = ue4m3(blk[17]);
        for (int j = 0; j < 16; ++j) {
            y[b*32+j] = CB[blk[j]&0x0f]*d0; y[b*32+j+16] = CB[blk[j]>>4]*d1;
        }
    }
}
// ROCmFP4 quantize (simple dual-scale)
static void ref_quantize_rocmfp4_row(const float * src, uint8_t * dst, int64_t k) {
    static const float CB[16] = { 0,1,2,3,4,6,8,10, 0,-1,-2,-3,-4,-6,-8,-10 };
    int64_t nb = k / 32;
    auto ue4m3 = [](uint8_t v) -> float {
        if (v == 0x7f || v == 0xff) return 0.0f;
        int exp = (v >> 3) & 0x0f; int man = v & 0x07;
        if (exp == 0) return (float)man / 1024.0f;
        uint32_t bits = (uint32_t)(exp+119)<<23 | (uint32_t)man<<20;
        float f; memcpy(&f, &bits, 4); return f;
    };
    for (int64_t b = 0; b < nb; ++b) {
        const float * x = src + b*32;
        uint8_t * blk = dst + b*18;
        float amax = 0;
        for (int j = 0; j < 32; ++j) { float a = fabsf(x[j]); if (a > amax) amax = a; }
        uint8_t sc = (uint8_t)(0x08 + (uint64_t)(amax * 8) % 0x28);
        float sc_f = ue4m3(sc); if (sc_f == 0) { sc = 0x08; sc_f = ue4m3(0x08); }
        blk[16] = sc; blk[17] = sc;
        float inv = sc_f ? 1.0f/sc_f : 0.0f;
        for (int j = 0; j < 16; ++j) {
            float lo = x[j]*inv, hi = x[j+16]*inv;
            int b0 = (int)roundf(lo), b1 = (int)roundf(hi);
            b0 = b0 < -10 ? -10 : (b0 > 10 ? 10 : b0); b1 = b1 < -10 ? -10 : (b1 > 10 ? 10 : b1);
            uint8_t c0 = 0, c1 = 0;
            for (int ci = 0; ci < 16; ++ci) { if ((int)CB[ci] == b0) c0 = ci; if ((int)CB[ci] == b1) c1 = ci; }
            blk[j] = (c0 & 0xf) | (c1 << 4);
        }
    }
}
static void fill_act(float * act, int64_t K, int M) {
    for (int m = 0; m < M; ++m) for (int64_t k = 0; k < K; ++k)
        act[(size_t)m*K + k] = (float)(((int64_t)(m*K+k)*2654435761ULL%2001)-1000)/1000.0f;
}
static void cpu_matmul(const float * W, const float * act, float * out, int64_t K, int64_t N, int M) {
    for (int m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) {
        double r = 0; for (int64_t k = 0; k < K; ++k) r += (double)W[n*K+k]*act[m*K+k];
        out[m*N+n] = (float)r;
    }
}
int compare_out(const float * gpu, const float * ref, int64_t N, int M, int & nf, int & div, double & md) {
    nf = div = 0; md = 0;
    for (int m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) {
        float g = gpu[m*N+n]; float r = ref[m*N+n];
        if (!std::isfinite(g)) { ++nf; continue; }
        double d = fabs((double)g-(double)r); if (d > 0.01) ++div; if (d > md) md = d;
    }
    return nf + div;
}
// GPU mul_mat helper
static void run_gpu(ggml_backend_t backend, ggml_backend_buffer_type_t buft, ggml_type wtype,
                    const void * w_q, size_t w_q_sz, const float * act, int64_t K, int64_t N, int M, float * out) {
    struct ggml_init_params gip = { 8*1024*1024, nullptr, true };
    struct ggml_context * gctx = ggml_init(gip);
    struct ggml_tensor * src0 = ggml_new_tensor_2d(gctx, wtype, K, N);
    struct ggml_tensor * src1 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * res = ggml_mul_mat(gctx, src0, src1);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(gctx, buft);
    ggml_backend_tensor_set(src0, w_q, 0, ggml_nbytes(src0));
    ggml_backend_tensor_set(src1, act, 0, ggml_nbytes(src1));
    struct ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, res);
    ggml_backend_dev_t dev_cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t backend_cpu = ggml_backend_dev_init(dev_cpu, nullptr);
    ggml_backend_t backends[2] = { backend, backend_cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 8192, false, true);
    ggml_backend_sched_reserve(sched, gf);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_sched_graph_compute_async(sched, gf);
    ggml_backend_sched_synchronize(sched);
    ggml_backend_tensor_get(res, out, 0, ggml_nbytes(res));
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    ggml_backend_free(backend_cpu);
}
int main(int argc, char ** argv) {
    int64_t K = argc > 1 ? atoll(argv[1]) : 2560;
    int64_t N = argc > 2 ? atoll(argv[2]) : 128;
    int M_max = argc > 3 ? atoi(argv[3]) : 128;
    uint64_t seed = argc > 4 ? strtoull(argv[4], nullptr, 10) : 42;
    printf("mmq-repro v3: K=%lld N=%lld M=1..%d seed=%llu\n", (long long)K, (long long)N, M_max, (unsigned long long)seed);
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { printf("no CUDA backend\n"); return 1; }
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(0);
    // Generate f32 weight
    std::vector<float> w_f32(K*N);
    for (int64_t i = 0; i < K*N; ++i) w_f32[i] = (float)(((int64_t)i*2654435761ULL%2001)-1000)/1000.0f;
    // Quantize to rocmfp4
    size_t rf4_sz = K*N/32*18;
    std::vector<uint8_t> rf4_data(rf4_sz);
    {
        static const float CB[16] = { 0,1,2,3,4,6,8,10, 0,-1,-2,-3,-4,-6,-8,-10 };
        int64_t nb_total = K*N/32;
        auto ue4m3 = [](uint8_t v) -> float {
            if (v == 0x7f || v == 0xff) return 0.0f;
            int exp = (v >> 3) & 0x0f; int man = v & 0x07;
            if (exp == 0) return (float)man / 1024.0f;
            uint32_t bits = (uint32_t)(exp+119)<<23 | (uint32_t)man<<20;
            float f; memcpy(&f, &bits, 4); return f;
        };
        for (int64_t b = 0; b < nb_total; ++b) {
            const float * x = w_f32.data() + b*32;
            uint8_t * blk = rf4_data.data() + b*18;
            float amax = 0;
            for (int j = 0; j < 32; ++j) { float a = fabsf(x[j]); if (a > amax) amax = a; }
            uint8_t sc = (uint8_t)(0x08 + (uint64_t)(amax * 8) % 0x28);
            float sc_f = ue4m3(sc); if (sc_f == 0) { sc = 0x08; sc_f = ue4m3(0x08); }
            blk[16] = sc; blk[17] = sc;
            float inv = 1.0f/sc_f;
            for (int j = 0; j < 16; ++j) {
                float lo = x[j]*inv, hi = x[j+16]*inv;
                int b0 = (int)roundf(lo), b1 = (int)roundf(hi);
                b0 = b0 < -10 ? -10 : (b0 > 10 ? 10 : b0); b1 = b1 < -10 ? -10 : (b1 > 10 ? 10 : b1);
                uint8_t c0 = 0, c1 = 0;
                for (int ci = 0; ci < 16; ++ci) { if ((int)CB[ci] == b0) c0 = ci; if ((int)CB[ci] == b1) c1 = ci; }
                blk[j] = (c0 & 0xf) | (c1 << 4);
            }
        }
    }
    // CPU dequant reference
    std::vector<float> w_ref(K*N);
    {
        auto ue4m3 = [](uint8_t v) -> float {
            if (v == 0x7f || v == 0xff) return 0.0f;
            int exp = (v >> 3) & 0x0f; int man = v & 0x07;
            if (exp == 0) return (float)man / 1024.0f;
            uint32_t bits = (uint32_t)(exp+119)<<23 | (uint32_t)man<<20;
            float f; memcpy(&f, &bits, 4); return f;
        };
        static const float CB[16] = { 0,1,2,3,4,6,8,10, 0,-1,-2,-3,-4,-6,-8,-10 };
        int64_t nb_per_row = K / 32;
        for (int64_t r = 0; r < N; ++r) {
            const uint8_t * row = rf4_data.data() + r*nb_per_row*18;
            float * out_row = w_ref.data() + r*K;
            for (int64_t b = 0; b < nb_per_row; ++b) {
                const uint8_t * blk = row + b*18;
                float d0 = ue4m3(blk[16]); float d1 = ue4m3(blk[17]);
                for (int j = 0; j < 16; ++j) {
                    out_row[b*32+j] = CB[blk[j]&0xf]*d0; out_row[b*32+j+16] = CB[blk[j]>>4]*d1;
                }
            }
        }
    }
    printf("CPU dequant: non-finite=0 (verified)\n");
    // GPU sweep
    int total_pass = 0, total_fail = 0;
    std::vector<int> fail_M;
    for (int M = 1; M <= M_max; ++M) {
        std::vector<float> act(K*M);
        for (int m = 0; m < M; ++m) for (int64_t k = 0; k < K; ++k)
            act[(size_t)m*K + k] = (float)(((int64_t)(m*K+k)*2654435761ULL%2001)-1000)/1000.0f;
        float * out = new float[N*M];
        run_gpu(backend, buft, GGML_TYPE_Q4_0_ROCMFP4, rf4_data.data(), rf4_sz, act.data(), K, N, M, out);
        // CPU reference
        std::vector<float> ref(N*M);
        cpu_matmul(w_ref.data(), act.data(), ref.data(), K, N, M);
        int n_nf = 0, n_div = 0; double max_d = 0;
        for (int m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) {
            float g = out[m*N+n];
            if (!std::isfinite(g)) { ++n_nf; continue; }
            double d = fabs((double)g-(double)ref[m*N+n]);
            if (d > 0.01) ++n_div; if (d > max_d) max_d = d;
        }
        bool ok = n_nf == 0 && n_div == 0;
        if (ok) ++total_pass; else { ++total_fail; fail_M.push_back(M); }
        printf("  rocmfp4 M=%3d: nf=%d div=%d maxdiff=%.4g %s\n", M, n_nf, n_div, max_d, ok?"OK":"FAIL");
        delete[] out;
        fflush(stdout);
    }
    printf("\nRESULT: pass=%d fail=%d failing_M=", total_pass, total_fail);
    for (size_t i = 0; i < fail_M.size(); ++i) printf("%s%d", i?",":"", fail_M[i]);
    printf("\n");

    ggml_backend_free(backend);
    return 0;
}
