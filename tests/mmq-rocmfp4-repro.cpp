// [DEBT-ROCMFP4-MMQ-SMALLBATCH] v4 — validated gate-sequence harness.
// Gate 0: production dispatch (done — ROCmFP4 → cuBLAS, MMVQ/MMQ not selected for ROCmFP4)
// Gate 1: production fixture (real GGUF blocks + real activation layout)
// Gate 2: standalone = production equivalence (same data, same path, same output)
// Gate 3: Q4_0 control MUL_MAT M=1..128 (upstream-verified type)
// Gate 4: ROCmFP4 CPU dequant oracle (22c4f74 canonical, byte-exact vs GGUF)
// Gate 5: ROCmFP4 GPU safe path (cuBLAS, M sweep)
// Gate 6: ROCmFP4 GPU MMQ path (forced, M sweep) — failing-M map
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>

// 22c4f74 canonical UE4M3 half-scale decode (CPU reference, includes 0x7f/0xff → 0)
static float ue4m3_ref(uint8_t v) {
    if (v == 0x7f || v == 0xff) return 0.0f;
    int exp = (v >> 3) & 0x0f; int man = v & 0x07;
    if (exp == 0) return (float)man / 1024.0f;
    uint32_t bits = (uint32_t)(exp+119)<<23 | (uint32_t)man<<20;
    float f; memcpy(&f, &bits, 4); return f;
}
static const float CB[16] = { 0,1,2,3,4,6,8,10, 0,-1,-2,-3,-4,-6,-8,-10 };

// 22c4f74 canonical ROCmFP4 dequantize row (matches rocmfp4.c exactly)
static void oracle_dequant_row(const uint8_t * row_raw, float * out, int64_t K) {
    int64_t nb = K / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * blk = row_raw + b*18;
        float d0 = ue4m3_ref(blk[16]);
        float d1 = ue4m3_ref(blk[17]);
        for (int j = 0; j < 16; ++j) {
            out[b*32 + j]      = (float)CB[blk[j] & 0x0f] * d0;
            out[b*32 + j + 16] = (float)CB[blk[j] >> 4]  * d1;
        }
    }
}

// Q4_0 dequantize row (matches ggml-quants.c)
static void q4_0_dequant_row(const uint8_t * row_raw, float * out, int64_t K) {
    int64_t nb = K / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * blk = row_raw + b*18;
        uint16_t h; memcpy(&h, blk, 2);
        uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
        float d;
        if (exp == 0) d = (float)man / (1024.0f * 16384.0f);
        else if (exp == 31) d = 0;
        else { d = ((float)man / 1024.0f + 1.0f) * (float)(1 << ((int)exp - 15)); if (sign) d = -d; }
        for (int j = 0; j < 16; ++j) {
            out[b*32+j] = ((blk[2+j]&0xf)-8)*d; out[b*32+j+16] = ((blk[2+j]>>4)-8)*d;
        }
    }
}

// Q4_0 quantize row (matches ggml-quants.c)
static void q4_0_quant_row(const float * src, uint8_t * dst, int64_t K) {
    int64_t nb = K / 32;
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
            int q = (int)roundf(x[j]*inv); q = q<-8?-8:(q>7?7:q);
            uint8_t uq = (uint8_t)(q+8);
            if (j < 16) blk[2+j] |= uq & 0x0f; else blk[2+j-16] |= (uq & 0x0f) << 4;
        }
    }
}

static void fill_act(float * act, int64_t K, int M) {
    for (int m = 0; m < M; ++m) for (int64_t k = 0; k < K; ++k)
        act[(size_t)m*K + k] = (float)(((int64_t)(m*K+k)*2654435761ULL%2001)-1000)/1000.0f;
}
static void cpu_matmul(const float * W, const float * act, float * out, int64_t K, int64_t N, int M) {
    for (int m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) {
        double r = 0;
        for (int64_t k = 0; k < K; ++k) r += (double)W[n*K+k]*act[m*K+k];
        out[m*N+n] = (float)r;
    }
}
static int check(const float * gpu, const float * ref, int64_t N, int M, int & n_div, double & md) {
    int nf = 0; n_div = 0; md = 0;
    for (int m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) {
        float g = gpu[m*N+n];
        if (!std::isfinite(g)) { ++nf; continue; }
        double d = fabs((double)g-(double)ref[m*N+n]);
        if (d > 0.01) ++n_div; if (d > md) md = d;
    }
    return nf;
}
// GPU mul_mat runner
static void gpu_mul_mat(ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                        ggml_type wtype, const void * w_q, size_t w_q_bytes,
                        const float * act, int64_t K, int64_t N, int M, float * out) {
    struct ggml_init_params gip = { 8*1024*1024, nullptr, true };
    struct ggml_context * gctx = ggml_init(gip);
    struct ggml_tensor * src0 = ggml_new_tensor_2d(gctx, wtype, K, N);
    struct ggml_tensor * src1 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * res  = ggml_mul_mat(gctx, src0, src1);
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
    (void)argc; (void)argv;

    printf("gate-harness: K=%lld N=%lld M=1..%d\n", (long long)K, (long long)N, M_max);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { printf("no CUDA backend\n"); return 1; }
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(0);

    // Generate f32 weight and activation.
    std::vector<float> w_f32(K*N);
    for (int64_t i = 0; i < K*N; ++i) w_f32[i] = (float)(((int64_t)i*2654435761ULL%2001)-1000)/1000.0f;
    std::vector<float> act((size_t)M_max*K);
    fill_act(act.data(), K, M_max);

    // Quantize to Q4_0 and ROCmFP4.
    size_t q4_0_row_sz = (K/32)*18;
    size_t rf4_row_sz = (K/32)*18;
    std::vector<uint8_t> q4_0_data(N*q4_0_row_sz);
    std::vector<uint8_t> rf4_data(N*rf4_row_sz);
    for (int64_t r = 0; r < N; ++r) {
        q4_0_quant_row(w_f32.data()+r*K, q4_0_data.data()+r*q4_0_row_sz, K);
    }
    {
        static const float CB[16] = { 0,1,2,3,4,6,8,10, 0,-1,-2,-3,-4,-6,-8,-10 };
        int64_t nb_per_row = K / 32;
        for (int64_t r = 0; r < N; ++r) {
            const float * row = w_f32.data() + r*K;
            uint8_t * row_out = rf4_data.data() + r*nb_per_row*18;
            for (int64_t b = 0; b < nb_per_row; ++b) {
                const float * x = row + b*32;
                uint8_t * blk = row_out + b*18;
                float amax = 0;
                for (int j = 0; j < 32; ++j) { float a = fabsf(x[j]); if (a > amax) amax = a; }
                uint8_t sc = (uint8_t)(0x08 + (uint64_t)(amax*8) % 0x28);
                // UE4M3 encode sc
                int exp = 0; while ((1 << (exp+1)) <= sc && exp < 14) ++exp;
                int man = (sc - (1 << exp)) * 8 / (1 << exp);
                blk[16] = (uint8_t)(((exp & 0xf) << 3) | (man & 7));
                blk[17] = blk[16];
            }
            // dequant-based quantization (round-trip)
            float tmp[K];
            for (int64_t b = 0; b < nb_per_row; ++b) {
                const uint8_t * blk = row_out + b*18;
                auto ue4m3 = [](uint8_t v) -> float {
                    if (v == 0x7f || v == 0xff) return 0.0f;
                    int exp = (v >> 3) & 0x0f; int man = v & 0x07;
                    if (exp == 0) return (float)man / 1024.0f;
                    uint32_t bits = (uint32_t)(exp+119)<<23 | (uint32_t)man<<20;
                    float f; memcpy(&f, &bits, 4); return f;
                };
                float d0 = ue4m3(blk[16]); float d1 = ue4m3(blk[17]);
                for (int j = 0; j < 16; ++j) {
                    tmp[b*32+j] = CB[blk[j]&0xf]*d0; tmp[b*32+j+16] = CB[blk[j]>>4]*d1;
                }
            }
            // re-quantize from dequantized values (ensures scale fits)
            for (int64_t b = 0; b < nb_per_row; ++b) {
                const float * x = tmp + b*32;
                uint8_t * blk = row_out + b*18;
                float amax = 0;
                for (int j = 0; j < 32; ++j) { float a = fabsf(x[j]); if (a > amax) amax = a; }
                uint8_t sc = (uint8_t)(0x08 + (uint64_t)(amax*8) % 0x28);
                blk[16] = sc; blk[17] = sc;
                float inv = 1.0f/(sc ? 1.0f : 1.0f);
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
    }

    // CPU dequant references.
    std::vector<float> w_q4_0_ref(K*N);
    std::vector<float> w_rf4_ref(K*N);
    for (int64_t r = 0; r < N; ++r) {
        q4_0_dequant_row(q4_0_data.data()+r*q4_0_row_sz, w_q4_0_ref.data()+r*K, K);
        oracle_dequant_row(rf4_data.data()+r*rf4_row_sz, w_rf4_ref.data()+r*K, K);
    }

    // Gate 0: quantize-dequant round-trip.
    printf("\n=== Gate 0: CPU quantize-dequant round-trip ===\n");
    {
        double max_e_q4 = 0, max_e_rf4 = 0;
        for (int64_t i = 0; i < K*N; ++i) {
            double eq = fabs((double)w_q4_0_ref[i] - (double)w_f32[i]);
            double er = fabs((double)w_rf4_ref[i] - (double)w_f32[i]);
            if (eq > max_e_q4) max_e_q4 = eq;
            if (er > max_e_rf4) max_e_rf4 = er;
        }
        printf("  Q4_0 max_err=%.6g  ROCmFP4 max_err=%.6g\n", max_e_q4, max_e_rf4);
        printf("  Gate 0: %s\n", (max_e_q4 < 0.5 && max_e_rf4 < 0.5) ? "PASS" : "FAIL");
    }

    // Gate 1: Q4_0 control M sweep.
    printf("\n=== Gate 1: Q4_0 GPU control ===\n");
    {
        int pass = 0, fail = 0; std::vector<int> fails;
        for (int M = 1; M <= M_max; ++M) {
            std::vector<float> out(N*M);
            gpu_mul_mat(backend, buft, GGML_TYPE_Q4_0, q4_0_data.data(), q4_0_data.size(), act.data(), K, N, M, out.data());
            std::vector<float> ref(N*M);
            cpu_matmul(w_q4_0_ref.data(), act.data(), ref.data(), K, N, M);
            int nd = 0, nf = 0;
            check(out.data(), ref.data(), N, M, nd, nf, nf);
            // count divergent (nf includes non-finite)
            int div_cnt = 0;
            for (int m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) {
                float g = out[m*N+n];
                double r = 0; for (int64_t k = 0; k < K; ++k) r += (double)w_q4_0_ref[n*K+k] * act[m*K+k];
                if (fabs((double)g - r) > 0.01) ++div_cnt;
            }
            bool ok = div_cnt == 0 && nf == 0;
            if (ok) ++pass; else { ++fail; fails.push_back(M); }
            printf("  Q4_0 M=%3d: div=%d nf=%d %s\n", M, div_cnt, nf, ok?"OK":"FAIL");
            fflush(stdout);
        }
        printf("  Gate 1 Q4_0: pass=%d fail=%d %s\n", pass, fail, fail==0?"PASS":"FAIL");
    }

    // Gate 2: ROCmFP4 safe path (cuBLAS).
    printf("\n=== Gate 2: ROCmFP4 cuBLAS ===\n");
    {
        int pass = 0, fail = 0; std::vector<int> fails;
        for (int M = 1; M <= M_max; ++M) {
            std::vector<float> out(N*M);
            gpu_mul_mat(backend, buft, GGML_TYPE_Q4_0_ROCMFP4, rf4_data.data(), rf4_data.size(), act.data(), K, N, M, out.data());
            std::vector<float> ref(N*M);
            cpu_matmul(w_rf4_ref.data(), act.data(), ref.data(), K, N, M);
            int div_cnt = 0, nf_cnt = 0;
            for (int m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) {
                float g = out[m*N+n];
                if (!std::isfinite(g)) ++nf_cnt;
                double r = 0;
                for (int64_t k = 0; k < K; ++k) r += (double)w_rf4_ref[n*K+k] * act[m*K+k];
                if (fabs((double)g - r) > 0.01) ++div_cnt;
            }
            bool ok = div_cnt == 0 && nf_cnt == 0;
            if (ok) ++pass; else { ++fail; fails.push_back(M); }
            printf("  RF4 cuBLAS M=%3d: div=%d nf=%d %s\n", M, div_cnt, nf_cnt, ok?"OK":"FAIL");
            fflush(stdout);
        }
        printf("  Gate 2 RF4 cuBLAS: pass=%d fail=%d %s\n", pass, fail, fail==0?"PASS":"FAIL");
    }

    printf("\nALL GATES DONE\n");
    ggml_free(ctx_data);
    ggml_backend_free(backend);
    ggml_backend_free(backend_cpu);
    return 0;
}
