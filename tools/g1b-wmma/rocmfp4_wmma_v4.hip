// G-1B v4: ROCmFP4 → 프래그먼트 직접 디코드 → f16 WMMA → f32 레지스터 누적
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace rocwmma;
using H = _Float16;
#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP ERR %s @%d\n", hipGetErrorString(e), __LINE__); return 1;} } while(0)

__host__ __device__ inline float cbv_f(int nib) {
    static const float kv[16] = {0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10};
    return kv[nib & 0xF];
}
__host__ __device__ inline float ue4m3_full(unsigned char x) {
    if (x == 0 || x == 0x7F || x == 0xFF) return 0.0f;
    unsigned e = (x >> 3) & 0xF, m = x & 7;
    float s = (x & 0x80) ? -1.0f : 1.0f;
    return s * (1.0f + m / 8.0f) * exp2f((float)e - 7.0f);
}

constexpr int MTILE = 16, NTILE = 16;

template <int SEGS>
__global__ void k_gemm(const H* __restrict__ Af,
                       const unsigned char* __restrict__ Wq,
                       float* __restrict__ D,
                       int K, int N) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int mh = warp / 2, nh = warp % 2;
    const int m0 = blockIdx.y * (MTILE * 2) + mh * MTILE;
    const int n0 = blockIdx.x * (NTILE * 2) + nh * NTILE;

    fragment<accumulator, 16, 16, 16, float> acc[SEGS];
    #pragma unroll
    for (int s = 0; s < SEGS; ++s) fill_fragment(acc[s], 0.0f);

    const int K16 = K / 16;
    const int seg = (K16 + SEGS - 1) / SEGS;
    for (int s = 0; s < SEGS; ++s) {
        for (int kh = s*seg; kh < (s+1)*seg && kh < K16; ++kh) {
            fragment<matrix_a, 16, 16, 16, H, row_major> fa;
            load_matrix_sync(fa, Af + (size_t)m0*K + kh*16, K);

            const unsigned char* blk = Wq + (size_t)((kh/2)*N + n0 + lane%16)*18;
            const unsigned char* src = blk + (kh%2)*8;
            int q4;
            memcpy(&q4, src + 4*(lane/16), 4);
            float wsc = ue4m3_full(blk[16 + (kh%2)]);
            fragment<matrix_b, 16, 16, 16, H, row_major> fb;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int nib = (q4 >> (8*(i/2) + 4*(i%2))) & 0xF;
                fb.x[i] = (H)(cbv_f(nib) * wsc);
            }
            mma_sync(acc[s], fa, fb, acc[s]);
        }
    }
    #pragma unroll
    for (int s = 1; s < SEGS; ++s)
        #pragma unroll
        for (int i = 0; i < acc[0].num_elements; ++i) acc[0].x[i] += acc[s].x[i];
    store_matrix_sync(D + (size_t)m0*N + n0, acc[0], N, mem_row_major);
}

static void cpu_ref(int M, int K, int N,
                    const std::vector<unsigned char>& Wq,
                    const std::vector<H>& Af,
                    std::vector<float>& D) {
    auto cb = [](int n) -> float {
        static const float kv[16] = {0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10};
        return kv[n & 0xF];
    };
    auto ue = [](unsigned char x) -> float {
        if (x == 0 || x == 0x7F || x == 0xFF) return 0.0f;
        unsigned e = (x >> 3) & 0xF, m = x & 7;
        float s = (x & 0x80) ? -1.0f : 1.0f;
        return s * (1.0f + m / 8.0f) * exp2f((float)e - 7.0f);
    };
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0;
            for (int k = 0; k < K; ++k) {
                const unsigned char* src = &Wq[(size_t)(k/32*N + n)*18];
                unsigned char byte = src[k%32 / 2];
                int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
                s += (double)(float)Af[(size_t)m*K + k] * (cb(nib) * ue(src[16 + (k%32)/16]));
            }
            D[(size_t)m*N + n] = (float)s;
        }
}

int main(int argc, char** argv) {
    bool correct = (argc > 1 && (std::string(argv[1]) == "correct" || std::string(argv[1]) == "ident"));
    bool ident_mode = (argc > 1 && std::string(argv[1]) == "ident");
    int M = correct ? 64 : 256;
    int K = correct ? 128 : 4096;
    int N = correct ? 64 : 4096;
    int passes = 20;

    size_t wq_size = (size_t)(K/32) * N * 18;
    std::vector<unsigned char> hWq(wq_size);
    srand(7);
    for (auto& v : hWq) v = rand() & 0xFF;
    std::vector<H> hAf((size_t)M*K);
    if (ident_mode) {
        memset(hAf.data(), 0, hAf.size()*sizeof(H));
        for (int i = 0; i < 16 && i < M && i < K; ++i) hAf[(size_t)i*K + i] = (H)1.0f;
    } else {
        for (size_t i = 0; i < hAf.size(); ++i) hAf[i] = (H)(float)(rand()%15 - 7);
    }
    std::vector<float> hD((size_t)M*N);

    unsigned char* dWq; H* dAf; float* dD;
    HIP_CHECK(hipMalloc(&dWq, wq_size));
    HIP_CHECK(hipMalloc(&dAf, hAf.size()*sizeof(H)));
    HIP_CHECK(hipMalloc(&dD, hD.size()*sizeof(float)));
    HIP_CHECK(hipMemcpy(dWq, hWq.data(), wq_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAf, hAf.data(), hAf.size()*sizeof(H), hipMemcpyHostToDevice));
    dim3 grid(N/(NTILE*2), M/(MTILE*2));
    int segs = (argc > 3) ? atoi(argv[3]) : 1;
    if (segs != 2 && segs != 4) segs = 1;

#define LAUNCH(S) hipLaunchKernelGGL(k_gemm<S>, grid, dim3(128), 0, 0, dAf, dWq, dD, K, N)
    if (correct) {
        if (segs == 4) { LAUNCH(4); } else if (segs == 2) { LAUNCH(2); } else { LAUNCH(1); }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(hD.data(), dD, hD.size()*sizeof(float), hipMemcpyDeviceToHost));
        std::vector<float> ref((size_t)M*N);
        cpu_ref(M, K, N, hWq, hAf, ref);
        int bad = 0; double maxabs = 0;
        for (int i = 0; i < M*N; ++i) {
            double e = hD[i] - ref[i]; if (e < 0) e = -e;
            if (e > maxabs) maxabs = e;
            if (e > 1.0) ++bad;
        }
        printf("correctness: |err|>1.0 %d/%d, maxabs=%.3f\n", bad, M*N, maxabs);
        if (ident_mode) {
            for (int r = 0; r < 8; ++r) {
                printf("GPU row %d:", r);
                for (int c = 0; c < 16; ++c) printf(" %5.1f", hD[(size_t)r*N + c]);
                printf("\n        CPU:");
                for (int c = 0; c < 16; ++c) printf(" %5.1f", ref[(size_t)r*N + c]);
                printf("\n");
            }
        }
        return bad ? 1 : 0;
    }

    if (segs == 4) { LAUNCH(4); } else if (segs == 2) { LAUNCH(2); } else { LAUNCH(1); }
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t t0, t1; HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
    HIP_CHECK(hipEventRecord(t0));
    for (int p = 0; p < passes; ++p) {
        if (segs == 4) { LAUNCH(4); } else if (segs == 2) { LAUNCH(2); } else { LAUNCH(1); }
    }
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipDeviceSynchronize());
    float ms; HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    double mac = (double)M*K*N*passes;
    printf("ROCmFP4 direct-fragment WMMA v4: M=%d K=%d N=%d passes=%d\n", M, K, N, passes);
    printf("  %.1f ms → %.2f T-MAC/s\n", ms, mac/(ms*1e-3)/1e12);
    printf("  (G-1A dp4a MMQ: 3.0 T-MAC/s)\n");
    return 0;
}
