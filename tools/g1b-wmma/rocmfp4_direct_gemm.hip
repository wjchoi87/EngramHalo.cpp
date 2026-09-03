// G-1B: 2-pass — (1) dequant ROCmFP4→f16 global (2) wmma GEMM global 직접 로드 (sync-free)
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace rocwmma;
using H = _Float16;
#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP ERR %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while(0)

__device__ __forceinline__ float cbv_f(int nib) {
    static const float kv[16] = {0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10};
    return kv[nib & 0xF];
}
__device__ __forceinline__ float ue4m3_h(unsigned char x) {
    if (x == 0 || x == 0x7F || x == 0xFF) return 0.0f;
    unsigned e = (x >> 3) & 0xF, m = x & 7;
    float s = (x & 0x80) ? -1.0f : 1.0f;
    return s * (1.0f + m / 8.0f) * exp2f((float)e - 8.0f);
}

// dequant: thread당 1 element — 블록 (k/32, n), 요소 e
__global__ void k_dequant(const unsigned char* __restrict__ Wq, H* __restrict__ Wf,
                          int K, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;   // n 전체
    int kb = blockIdx.y;                              // k/32 블록
    if (n >= N) return;
    const unsigned char* src = &Wq[(size_t)(kb * N + n) * 18];
    H* dst = &Wf[(size_t)kb * 32 * N + n];            // [k][N] 레이아웃: k = kb*32+e
    for (int e = 0; e < 32; ++e) {
        unsigned char byte = src[e / 2];
        int nib = (e & 1) ? (byte >> 4) : (byte & 0xF);
        float v = cbv_f(nib) * ue4m3_h(src[16 + (e >= 16 ? 1 : 0)]) * 2.0f;
        dst[(size_t)e * N] = (H)v;
    }
}

// GEMM: 16x16x16 타일, f16 global 직접 로드, sync-free
constexpr int NT = 8;  // warp당 n-tile 수 (독립 acc 체인)
__global__ void k_gemm(const H* __restrict__ Af, const H* __restrict__ Wf,
                       float* __restrict__ D, int M, int K, int N) {
    const int n0 = blockIdx.x * (16*NT);
    const int m0 = blockIdx.y * 16;
    fragment<matrix_a, 16, 16, 16, H, row_major> fa;
    fragment<matrix_b, 16, 16, 16, H, row_major> fb[NT];
    fragment<accumulator, 16, 16, 16, float> fd[NT];
    #pragma unroll
    for (int t = 0; t < NT; ++t) { fill_fragment(fd[t], 0.0f); load_matrix_sync(fb[t], &Wf[(size_t)n0 + t*16], N); }
    for (int k0 = 0; k0 < K; k0 += 16) {
        fragment<matrix_a, 16, 16, 16, H, row_major> fag;
        load_matrix_sync(fag, &Af[(size_t)m0*K + k0], K);
        #pragma unroll
        for (int t = 0; t < NT; ++t) {
            fragment<matrix_b, 16, 16, 16, H, row_major> fbg;
            load_matrix_sync(fbg, &Wf[(size_t)k0*N + n0 + t*16], N);
            mma_sync(fd[t], fag, fbg, fd[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < NT; ++t)
        store_matrix_sync(D + (size_t)m0*N + n0 + t*16, fd[t], N, mem_row_major);
}

int main(int argc, char** argv) {
    int M = 256, K = 4096, N = 4096, passes = 20;
    size_t wq_size = (size_t)(K/32) * N * 18;
    std::vector<unsigned char> hWq(wq_size);
    srand(7);
    for (auto& v : hWq) v = rand() & 0xFF;

    unsigned char* dWq; H *dWf, *dAf; float* dD;
    HIP_CHECK(hipMalloc(&dWq, wq_size));
    HIP_CHECK(hipMalloc(&dWf, (size_t)K*N*sizeof(H)));
    HIP_CHECK(hipMalloc(&dAf, (size_t)M*K*sizeof(H)));
    HIP_CHECK(hipMalloc(&dD, (size_t)M*N*sizeof(float)));
    HIP_CHECK(hipMemcpy(dWq, hWq.data(), wq_size, hipMemcpyHostToDevice));
    // A: int8→f16 간단 채움 (벤치)
    {
        std::vector<signed char> hA((size_t)M*K);
        for (auto& v : hA) v = rand()%15 - 7;
        std::vector<H> hAf((size_t)M*K);
        for (size_t i = 0; i < hA.size(); ++i) hAf[i] = (H)hA[i];
        HIP_CHECK(hipMemcpy(dAf, hAf.data(), hAf.size()*sizeof(H), hipMemcpyHostToDevice));
    }

    // dequant (1회)
    hipLaunchKernelGGL(k_dequant, dim3(N/256), dim3(256), 0, 0, dWq, dWf, K, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    printf("dequant OK\n");

    dim3 grid(N/(16*NT), M/16);
    hipLaunchKernelGGL(k_gemm, grid, dim3(32), 0, 0, dAf, dWf, dD, M, K, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    printf("gemm warmup OK\n");
    hipEvent_t t0, t1; HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
    HIP_CHECK(hipEventRecord(t0));
    for (int p = 0; p < passes; ++p)
        hipLaunchKernelGGL(k_gemm, grid, dim3(32), 0, 0, dAf, dWf, dD, M, K, N);
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipDeviceSynchronize());
    float ms; HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    double mac = (double)M*K*N*passes;
    printf("f16-direct WMMA GEMM: M=%d K=%d N=%d passes=%d\n", M, K, N, passes);
    printf("  GEMM %.1f ms → %.2f T-MAC/s (dequant 제외)\n", ms, mac/(ms*1e-3)/1e12);
    return 0;
}
