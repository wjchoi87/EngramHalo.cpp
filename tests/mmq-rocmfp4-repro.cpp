// [DEBT-ROCMFP4-MMQ-SMALLBATCH] standalone mul_mat_q partial-M repro harness.
// Sweeps M=1..128 with a q4_0_rocmfp4 weight and a real f32 activation, compares
// against a CPU dequant+matmul reference, and records non-finite output,
// first divergent element, and per-M verdicts to identify the failing-M pattern.
//
// Build: registered in tests/CMakeLists.txt via llama_build_and_test.
// Env:   GGML_CUDA_ROCMFP4_MMQ_FORCE=1 to bypass the production safety gate.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
extern "C" {
#include "../ggml/rocmfp4/rocmfp4.h"
}

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

// CPU dequant of q4_0_rocmfp4: {qs[16], e[2]} per 32 elements.
// e[0] scales qs[0..15] lo-nibble codes, e[1] scales hi-nibble codes,
// each half-scale UE4M3. Matches the HIP device-side decode exactly.
static void rocmfp4_dequant_cpu(const uint8_t * blk_bytes, float * y, int64_t k) {
    static const float cb[16] = {
        0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 10.f,
        0.f,-1.f,-2.f,-3.f,-4.f,-6.f,-8.f,-10.f,
    };
    const int64_t nblk = k / 32;
    for (int64_t b = 0; b < nblk; ++b) {
        const uint8_t * blk = blk_bytes + b*18;
        auto ue4m3 = [](uint8_t v) -> float {
            const int exp = (v >> 3) & 0x0f;
            const int man = v & 0x07;
            if (v == 0 || v == 0x7f || v == 0xff) return 0.0f;
            if (exp == 0) return (float) man * (1.0f/1024.0f);
            const uint32_t bits = ((uint32_t) exp + 119u) << 23 | ((uint32_t) man << 20);
            float r; memcpy(&r, &bits, 4); return r;
        };
        const float d0 = ue4m3(blk[16]);
        const float d1 = ue4m3(blk[17]);
        for (int i = 0; i < 16; ++i) {
            y[b*32 + i]      = d0 * cb[blk[i] & 0x0f];
            y[b*32 + 16 + i] = d1 * cb[blk[i] >> 4];
        }
    }
}

static int64_t rng_state = 0x123456789abcdefULL;
static double prng() {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((rng_state >> 33) & 0x7fffffff) / (double)0x7fffffff;
}

// Build a deterministic q4_0_rocmfp4 tensor payload: finite UE4M3 scales
// (random byte among {0x08..0x30} region — all decode finite) + nibble codes.
static void fill_rocmfp4_data(void * data, int64_t n_elements, uint64_t seed) {
    uint8_t * bytes = (uint8_t *) data;
    const int64_t nblk = n_elements / 32;
    uint64_t s = seed;
    auto nxt = [&s]() { s = s*6364136223846793005ULL + 1442695040888963407ULL; return (s >> 33); };
    for (int64_t b = 0; b < nblk; ++b) {
        uint8_t * blk = bytes + b*18;
        for (int i = 0; i < 16; ++i) blk[i] = (uint8_t)(nxt() & 0xff);
        // scale bytes: choose from a set that decodes to finite, small values.
        //   0x08 -> exp=1,man=0 → 2^-6=0.015625 (UE4M3 min normal)
        //   0x30 -> exp=6,man=0 → 2^-1=0.5
        //   0x28 -> exp=5,man=0 → 1.0
        const uint8_t sc = (uint8_t)(0x08 + (nxt() % 0x29));
        blk[16] = sc;
        blk[17] = sc;
    }
}

int main(int argc, char ** argv) {
    const int64_t K = argc > 1 ? atoll(argv[1]) : 2560;
    const int64_t N = argc > 2 ? atoll(argv[2]) : 10240;
    const int   M0 = argc > 3 ? atoi (argv[3]) : 1;
    const int   M1 = argc > 4 ? atoi (argv[4]) : 128;
    const int64_t seed = argc > 5 ? atoll(argv[5]) : 42;

    printf("mmq-rocmfp4-repro: K=%lld N=%lld M=[%d..%d] seed=%llu\n",
           (long long)K, (long long)N, M0, M1, (unsigned long long)seed);

    ggml_time_init();
    struct ggml_init_params ip = {
        /*.mem_size   =*/ 256*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(ip);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { printf("no CUDA backend\n"); return 1; }
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(0);
    printf("backend: %s\n", ggml_backend_name(backend));
    printf("backend: %s\n", ggml_backend_buft_name(buft));

    // Allocate the q4_0_rocmfp4 weight on the host first, then wrap it in a
    // tensor whose type is registered for the CUDA backend dispatch.
    const int64_t n_w = K * N;
    const size_t  w_sz = n_w/32*18;
    std::vector<uint8_t> w_data(w_sz);
    fill_rocmfp4_data(w_data.data(), n_w, seed);

    // CPU reference weight (dequantized).
    std::vector<float> w_ref(n_w);
    for (int64_t r = 0; r < N; ++r) {
        rocmfp4_dequant_cpu(w_data.data() + r*(K/32*18), w_ref.data() + r*K, K);
    }

    int n_fail = 0, n_pass = 0;
    std::vector<int> fail_ms;

    for (int M = M0; M <= M1; ++M) {
        // Host-side activation: [K, M] f32, deterministic.
        std::vector<float> act(K*M);
        for (int64_t i = 0; i < K*M; ++i) act[i] = (float)((i*2654435761ULL % 2001) - 1000) / 1000.0f;

        // Build the graph.
        struct ggml_init_params gip = { /*mem_size*/ 8*1024*1024, nullptr, /*no_alloc*/ true };
        struct ggml_context * gctx = ggml_init(gip);

        struct ggml_tensor * src0 = ggml_new_tensor_2d(gctx, GGML_TYPE_Q4_0_ROCMFP4, K, N);
        struct ggml_tensor * src1 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor * res = ggml_mul_mat(gctx, src0, src1);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(gctx, buft);

        ggml_backend_tensor_set(src0, w_data.data(), 0, w_sz);
        ggml_backend_tensor_set(src1, act.data(), 0, K*M*sizeof(float));

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

        std::vector<float> out(N*M);
        ggml_backend_tensor_get(res, out.data(), 0, N*M*sizeof(float));

        ggml_backend_sched_free(sched);
        ggml_free(gctx);
        ggml_backend_buffer_free(buf);

        // CPU reference: out_ref[n][m] = sum_k W[n][k] * act[k][m].
        // Row-major W is [N, K]. We compute per-column to keep memory bounded.
        int n_nonfinite = 0;
        int n_diverge = 0;
        int64_t first_bad = -1;
        double max_diff_finite = 0;
        for (int m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                const float gpu = out[m*N + n];
                if (!std::isfinite(gpu)) { ++n_nonfinite; if (first_bad < 0) first_bad = m*N + n; continue; }
                // CPU reference dot product (only when we flag a divergent candidate).
                double ref = 0;
                for (int64_t k = 0; k < K; ++k) ref += (double)w_ref[n*K + k] * act[k*M + m];
                const double diff = std::fabs((double)gpu - ref);
                if (diff > 0.01) { ++n_diverge; if (first_bad < 0) first_bad = m*N + n; }
                if (diff > max_diff_finite) max_diff_finite = diff;
            }
        }
        const bool ok = n_nonfinite == 0 && n_diverge == 0;
        if (ok) ++n_pass; else { ++n_fail; fail_ms.push_back(M); }
        printf("M=%3d: nonfinite=%d diverge=%d maxdiff=%.4f first_bad=%lld %s\n",
               M, n_nonfinite, n_diverge, max_diff_finite, (long long)first_bad, ok ? "OK" : "FAIL");
        fflush(stdout);
    }

    printf("\nSUMMARY: pass=%d fail=%d failing_M=[", n_pass, n_fail);
    for (size_t i = 0; i < fail_ms.size(); ++i) printf("%s%d", i?",":"", fail_ms[i]);
    printf("]\n");

    ggml_backend_free(backend);
    ggml_free(ctx);
    return n_fail > 0 ? 1 : 0;
}
