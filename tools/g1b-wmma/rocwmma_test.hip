// G-1B: rocWMMA API 기반 int8 16x16x16 WMMA 정합성 + 처리량 (gfx1151)
// 컴파일: hipcc --offload-arch=gfx1151 -O3 -std=c++17 rocwmma_test.hip -o rocwmma_test
#include <rocwmma/rocwmma.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace rocwmma;

using InputT  = int8_t;
using ComputeT = int32_t;

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP ERR %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while(0)

__global__ void k_wmma_correct(const InputT* A, const InputT* B, ComputeT* D, int ldk, int ldn) {
    fragment<matrix_a, 16, 16, 16, InputT, row_major> fragA;
    fragment<matrix_b, 16, 16, 16, InputT, row_major> fragB;
    fragment<accumulator, 16, 16, 16, ComputeT> fragD;
    fill_fragment(fragD, static_cast<ComputeT>(0));
    load_matrix_sync(fragA, A, ldk);
    load_matrix_sync(fragB, B, ldn);
    mma_sync(fragD, fragA, fragB, fragD);
    store_matrix_sync(D, fragD, ldn, mem_row_major);
}

__global__ void k_wmma_tp(const InputT* A, const InputT* B, ComputeT* D, int iters) {
    fragment<matrix_a, 16, 16, 16, InputT, row_major> fa;
    fragment<matrix_b, 16, 16, 16, InputT, row_major> fb;
    load_matrix_sync(fa, A, 16);
    load_matrix_sync(fb, B, 16);
    fragment<accumulator, 16, 16, 16, ComputeT> acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) fill_fragment(acc[j], static_cast<ComputeT>(0));
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) mma_sync(acc[j], fa, fb, acc[j]);
    }
    ComputeT s = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) s += acc[j].x[0] + acc[j].x[7];
    if (s == 123456789) D[threadIdx.x] = acc[0].x[0];
}

__global__ void k_dp4a_tp(const int* A, const int* B, int* D, int iters) {
    int l = threadIdx.x;
    int a0 = A[l*4+0], a1 = A[l*4+1], a2 = A[l*4+2], a3 = A[l*4+3];
    int b0 = B[l*4+0], b1 = B[l*4+1], b2 = B[l*4+2], b3 = B[l*4+3];
    int c[8] = {0,0,0,0,0,0,0,0};
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            c[0] = __builtin_amdgcn_sudot4(true, b0, true, a0, c[0], false);
            c[1] = __builtin_amdgcn_sudot4(true, b1, true, a1, c[1], false);
            c[2] = __builtin_amdgcn_sudot4(true, b2, true, a2, c[2], false);
            c[3] = __builtin_amdgcn_sudot4(true, b3, true, a3, c[3], false);
            c[4] = __builtin_amdgcn_sudot4(true, b0, true, a3, c[4], false);
            c[5] = __builtin_amdgcn_sudot4(true, b1, true, a2, c[5], false);
            c[6] = __builtin_amdgcn_sudot4(true, b2, true, a1, c[6], false);
            c[7] = __builtin_amdgcn_sudot4(true, b3, true, a0, c[7], false);
        }
    }
    int s = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) s += c[j];
    if (s == 123456789) D[threadIdx.x] = c[0];
}

int main() {
    const int N = 16;
    InputT hA[N*N], hB[N*N];
    srand(7);
    for (int i = 0; i < N*N; ++i) { hA[i] = rand()%15-7; hB[i] = rand()%15-7; }

    // CPU 참조
    ComputeT ref[N][N];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            ComputeT s = 0;
            for (int k = 0; k < N; ++k) s += (ComputeT)hA[i*N+k] * hB[k*N+j];
            ref[i][j] = s;
        }

    InputT *dA, *dB; ComputeT* dD;
    HIP_CHECK(hipMalloc(&dA, N*N)); HIP_CHECK(hipMalloc(&dB, N*N)); HIP_CHECK(hipMalloc(&dD, N*N*sizeof(ComputeT)));
    HIP_CHECK(hipMemcpy(dA, hA, N*N, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, N*N, hipMemcpyHostToDevice));

    // 1) correctness
    hipLaunchKernelGGL(k_wmma_correct, dim3(1), dim3(32), 0, 0, dA, dB, dD, N, N);
    HIP_CHECK(hipDeviceSynchronize());
    ComputeT hD[N*N];
    HIP_CHECK(hipMemcpy(hD, dD, N*N*sizeof(ComputeT), hipMemcpyDeviceToHost));
    int bad = 0; ComputeT maxerr = 0;
    for (int i = 0; i < N*N; ++i) {
        ComputeT e = hD[i] - ((ComputeT*)ref)[i];
        if (e < 0) e = -e;
        if (e > maxerr) maxerr = e;
        if (e != 0) ++bad;
    }
    printf("correctness: %d/%d 불일치, maxerr=%d\n", bad, N*N, (int)maxerr);

    // 2) throughput (128 workgroup × 32 lane, 8 independent acc chains)
    int iters = 20000;
    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
    // 웜업
    hipLaunchKernelGGL(k_wmma_tp, dim3(2560), dim3(32), 0, 0, dA, dB, dD, iters);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(t0));
    hipLaunchKernelGGL(k_wmma_tp, dim3(2560), dim3(32), 0, 0, dA, dB, dD, iters);
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipDeviceSynchronize());
    float ms_w; HIP_CHECK(hipEventElapsedTime(&ms_w, t0, t1));
    double wmma_mac = 2560.0*32*8*iters*4096.0;
    printf("wmma: %.1f ms → %.1f GOP/s(MAC)\n", ms_w, wmma_mac/(ms_w*1e6));

    int* di = (int*)dD;
    hipLaunchKernelGGL(k_dp4a_tp, dim3(2560), dim3(32), 0, 0, (const int*)dA, (const int*)dB, di, iters);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(t0));
    hipLaunchKernelGGL(k_dp4a_tp, dim3(2560), dim3(32), 0, 0, (const int*)dA, (const int*)dB, di, iters);
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipDeviceSynchronize());
    float ms_d; HIP_CHECK(hipEventElapsedTime(&ms_d, t0, t1));
    double dp4a_mac = 2560.0*32*8*iters*4.0;
    printf("dp4a: %.1f ms → %.1f GOP/s(MAC)\n", ms_d, dp4a_mac/(ms_d*1e6));
    printf("비율 wmma/dp4a = %.2fx\n", (wmma_mac/ms_w)/(dp4a_mac/ms_d));
    return 0;
}
