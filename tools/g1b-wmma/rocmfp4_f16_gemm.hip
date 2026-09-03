// G-1B v3: ROCmFP4 → dequant-f16 → f16-WMMA → f32 register accumulator
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace rocwmma;
#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP ERR %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while(0)

using H = _Float16;

__device__ __forceinline__ float cbv_f(int nib) {
    static const float kv[16] = {0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10};
    return kv[nib & 0xF];
}
__device__ __forceinline__ float ue4m3_h(unsigned char x) {
    if (x == 0 || x == 0x7F || x == 0xFF) return 0.0f;
    unsigned e = (x >> 3) & 0xF, m = x & 7;
    float s = (x & 0x80) ? -1.0f : 1.0f;
    return s * (1.0f + m / 8.0f) * exp2f((float)e - 8.0f);  // halved
}

constexpr int BM = 64, BN = 32, BK = 32;

__global__ void k_f16_gemm(const unsigned char* __restrict__ Wq,
                           const signed char*  __restrict__ As,
                           const float* __restrict__ Ascale,   // [M][K/32]
                           float* __restrict__ D,
                           int K, int N, int apply_scale, float* dbg) {
    const int n0 = blockIdx.x * BN;
    const int m0 = blockIdx.y * BM;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int mh = warp / 2, nh = warp % 2;

    __shared__ H sA[BM][BK];
    __shared__ H sW[BK][BN];

    fragment<accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int f = 0; f < 2; ++f)
        #pragma unroll
        for (int g = 0; g < 2; ++g) fill_fragment(acc[f][g], 0.0f);

    for (int kb = 0; kb < K/BK; ++kb) {
        for (int i = tid; i < BM*BK; i += 128) {
            int m = i / BK, k = i % BK;
            float av = (float)As[(size_t)(m0+m)*K + kb*BK + k];
            if (apply_scale) av *= Ascale[(size_t)(m0+m)*(K/BK) + kb];
            sA[m][k] = (H)av;
        }
        for (int i = tid; i < BN*BK; i += 128) {
            int k = i / BN, n = i % BN;
            const unsigned char* src = &Wq[(size_t)(kb*N + n0 + n)*18];
            unsigned char byte = src[k/2];
            int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
            float wsc = ue4m3_h(src[16 + (k >= 16 ? 1 : 0)]) * 2.0f;
            sW[k][n] = (H)(cbv_f(nib) * wsc);
        }
        __syncthreads();
        if (dbg && blockIdx.x == 0 && blockIdx.y == 0 && (kb == 0 || kb == 1)) {
            int base = (kb == 0) ? 0 : 3000;
            if (tid < 64) dbg[base + tid] = (float)sA[tid/32][tid%32];
            __syncthreads();
            for (int i = tid; i < 1024; i += 128) dbg[base + 64 + i] = (float)sW[i/16][i%16];
            dbg[base + 2048] = 111.0f + kb;
        }

        fragment<matrix_b, 16, 16, 16, H, row_major> fb;
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            #pragma unroll
            for (int g = 0; g < 2; ++g) {
                fragment<matrix_a, 16, 16, 16, H, row_major> fa;
                load_matrix_sync(fa, &sA[mh*32 + f*16][g*16], BK);
                load_matrix_sync(fb, &sW[16*g][nh*16], BN);
                mma_sync(acc[f][g], fa, fb, acc[f][g]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int f = 0; f < 2; ++f) {
        fragment<accumulator, 16, 16, 16, float> sum;
        #pragma unroll
        for (int i = 0; i < sum.num_elements; ++i)
            sum.x[i] = acc[f][0].x[i] + acc[f][1].x[i];
        store_matrix_sync(D + (size_t)(m0 + mh*32 + f*16)*N + (n0 + nh*16), sum, N, mem_row_major);
    }
}

static float ue4m3_h_host(unsigned char x) {
    if (x == 0 || x == 0x7F || x == 0xFF) return 0.0f;
    unsigned e = (x >> 3) & 0xF, m = x & 7;
    float s = (x & 0x80) ? -1.0f : 1.0f;
    return s * (1.0f + m / 8.0f) * exp2f((float)e - 8.0f);
}
static void cpu_ref(int M, int K, int N,
                    const std::vector<unsigned char>& Wq,
                    const std::vector<signed char>& As,
                    const std::vector<float>& Asc,
                    std::vector<float>& D) {
    auto cb = [](int n) -> float {
        static const float kv[16] = {0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10};
        return kv[n & 0xF];
    };
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = 0;
            for (int k = 0; k < K; ++k) {
                const unsigned char* src = &Wq[(size_t)(k/32*N + n)*18];
                unsigned char byte = src[k/2];
                int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
                float w = cb(nib) * ue4m3_h_host(src[16 + (k >= 16 ? 1 : 0)]) * 2.0f;
                s += ((float)As[(size_t)m*K + k] * Asc[(size_t)m*(K/32) + k/32]) * w;
            }
            D[(size_t)m*N + n] = s;
        }
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "perf";
    bool dbgA = (std::string(mode) == "dbgA");
    bool dbg1 = (std::string(mode) == "dbg1");
    bool correct = (std::string(mode) == "correct");
    int M = (dbgA || dbg1) ? 64 : (correct ? 64 : 256);
    int K = (dbgA || dbg1) ? 32 : (correct ? 128 : 4096);
    int N = (dbgA || dbg1) ? 32 : 4096;
    int passes = 20;
    if (argc > 3) K = atoi(argv[3]);

    size_t wq_size = (size_t)(K/32) * N * 18;
    std::vector<unsigned char> hWq(wq_size);
    srand(7);
    for (auto& v : hWq) v = rand() & 0xFF;
    std::vector<signed char> hAs((size_t)M*K);
    memset(hAs.data(), 0, hAs.size());
    if (dbgA) {
        // A[0][k] = k (0..15), 나머지 행 0. W = 1 전부. 기대 D[0][n] = sum_k k = 120.
        for (int i = 0; i < 16; ++i) hAs[(size_t)i*K + i] = 1;   // A = I(16)
        memset((void*)hWq.data(), 0x11, wq_size);            // 모든 nibble = 1 → codebook 1
        for (size_t b = 0; b < wq_size; b += 18) { hWq[b+16] = 0x38; hWq[b+17] = 0x38; }  // scale 1
    } else {
        for (auto& v : hAs) v = rand()%15 - 7;
    }

    std::vector<float> hD((size_t)M*N);
    unsigned char* dWq; signed char* dAs; float *dAsc, *dD;
    HIP_CHECK(hipMalloc(&dWq, wq_size));
    HIP_CHECK(hipMalloc(&dAs, hAs.size()));
    std::vector<float> hAsc((size_t)M*(K/32));
    for (auto& v : hAsc) v = 1.0f;
    if (argc > 2 && std::string(argv[2]) == "s1") {
        for (size_t b = 0; b < wq_size; b += 18) { hWq[b+16] = 0x38; hWq[b+17] = 0x38; }
    }
    HIP_CHECK(hipMalloc(&dAsc, hAsc.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dD, hD.size()*sizeof(float)));
    HIP_CHECK(hipMemcpy(dWq, hWq.data(), wq_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAs, hAs.data(), hAs.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAsc, hAsc.data(), hAsc.size()*sizeof(float), hipMemcpyHostToDevice));
    dim3 grid(N/BN, M/BM);

    if (dbg1 || dbgA || correct) {
        float* dbg = nullptr;
        hipError_t lerr = hipGetLastError();  // 이전 launch 오류 플러시
        (void)lerr;
        if (dbgA) { HIP_CHECK(hipMalloc(&dbg, 8192*sizeof(float))); HIP_CHECK(hipMemset(dbg, 0, 8192*sizeof(float))); }
        hipLaunchKernelGGL(k_f16_gemm, grid, dim3(128), 0, 0, dWq, dAs, dAsc, dD, K, N, dbgA ? 0 : 1, dbg);
        if (dbg1) {
            for (int m = 0; m < 4; ++m) {
                printf("GPU row %2d:", m);
                for (int n = 0; n < 32; ++n) printf(" %5.1f", hD[(size_t)m*N + n]);
                printf("\n");
            }
            for (int n = 0; n < 8; ++n) {
                unsigned char b0 = hWq[(size_t)n*18];
                printf("  CPU cb[%d lo] = %.1f  (byte0=0x%02x)\n", b0 & 0xF,
                       (float)(int)((b0 & 0xF) < 8 ? (b0 & 0xF) : -16 + (b0 & 0xF)), b0);
            }
            return 0;
        }
        if (dbgA) {
            float hd[8192];
            HIP_CHECK(hipMemcpy(hd, dbg, sizeof(hd), hipMemcpyDeviceToHost));
            for (int kb2 = 0; kb2 < 2; ++kb2) {
                int base = kb2 * 3000;
                printf("== kb=%d ==\n", kb2);
                printf("sA[0][0..7]: "); for (int i = 0; i < 8; ++i) printf("%.1f ", hd[base+i]); printf("\n");
                printf("sA[1][0..7]: "); for (int i = 0; i < 8; ++i) printf("%.1f ", hd[base+32+i]); printf("\n");
                printf("sW[0..3][0]: "); for (int i = 0; i < 4; ++i) printf("%.1f ", hd[base+64+i*16]); printf("\n");
                printf("sW[16..19][0]: "); for (int i = 0; i < 4; ++i) printf("%.1f ", hd[base+64+16+i*16]); printf("\n");
            }
            HIP_CHECK(hipFree(dbg));
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(hD.data(), dD, hD.size()*sizeof(float), hipMemcpyDeviceToHost));
        if (dbgA) {
            printf("D[m][0] (m=0..15), 기대 = cb(m): 0 1 2 3 4 6 8 10 0 -1 -2 -3 -4 -6 -8 -10\n");
            for (int m = 0; m < 16; ++m)
                printf("  D[%2d][0] = %7.2f\n", m, hD[(size_t)m*N]);
            return 0;
        }
        std::vector<float> ref((size_t)M*N);
        cpu_ref(M, K, N, hWq, hAs, hAsc, ref);
        int bad = 0; float maxrel = 0;
        for (int i = 0; i < M*N; ++i) {
            float e = hD[i] - ref[i]; if (e < 0) e = -e;
            float rel = e / (ref[i] != 0 ? (ref[i] < 0 ? -ref[i] : ref[i]) : 1.0f);
            if (rel > maxrel) maxrel = rel;
            if (rel > 0.02f) ++bad;
        }
        printf("correctness: rel>2%% %d/%d, maxrel=%.4f\n", bad, M*N, maxrel);
        for (int m = 0; m < 2; ++m) {
            printf("GPU row %d:", m);
            for (int n = 0; n < 8; ++n) printf(" %8.2f", hD[(size_t)m*N + n]);
            printf("\nCPU row %d:", m);
            for (int n = 0; n < 8; ++n) printf(" %8.2f", ref[(size_t)m*N + n]);
            printf("\n");
        }
    } else {
        hipLaunchKernelGGL(k_f16_gemm, grid, dim3(128), 0, 0, dWq, dAs, dAsc, dD, K, N, 1, nullptr);
        HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t t0, t1; HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
        HIP_CHECK(hipEventRecord(t0));
        for (int p = 0; p < passes; ++p)
            hipLaunchKernelGGL(k_f16_gemm, grid, dim3(128), 0, 0, dWq, dAs, dAsc, dD, K, N, 1, nullptr);
        HIP_CHECK(hipEventRecord(t1));
        HIP_CHECK(hipDeviceSynchronize());
        float ms; HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        double mac = (double)M*K*N*passes;
        printf("ROCmFP4→f16-WMMA GEMM: M=%d K=%d N=%d passes=%d\n", M, K, N, passes);
        printf("  %.1f ms → %.2f T-MAC/s\n", ms, mac/(ms*1e-3)/1e12);
        printf("  (G-1A dp4a MMQ: 3.0 / v2 int-wmma: 1.24)\n");
    }
    return 0;
}
