// f16 WMMA 최소 동작 확인 (gfx1151): A=I → D = B 여야 함
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstring>
using namespace rocwmma;
using H = _Float16;
typedef int v4i __attribute__((vector_size(16)));
typedef int v8i __attribute__((vector_size(32)));
#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP ERR %s @%d\n", hipGetErrorString(e), __LINE__); return 1;} } while(0)

__global__ void k_once(const H* A, const H* B, float* D) {
    int l = threadIdx.x;
    fragment<matrix_a, 16, 16, 16, H, row_major> fa;
    fragment<matrix_b, 16, 16, 16, H, row_major> fb;
    fragment<accumulator, 16, 16, 16, float> fd;
    fill_fragment(fd, 0.0f);
    load_matrix_sync(fa, A, 16);
    load_matrix_sync(fb, B, 16);
    mma_sync(fd, fa, fb, fd);
    store_matrix_sync(D, fd, 16, mem_row_major);
}
int main() {
    H hA[256], hB[256];
    float hD[256];
    // A = identity, B[r][c] = r*16+c (16..255, 구분 가능)
    memset(hA, 0, sizeof(hA)); memset(hB, 0, sizeof(hB));
    for (int i = 0; i < 16; ++i) {
        hA[i*16+i] = (H)1.0f;
        for (int j = 0; j < 16; ++j) hB[i*16+j] = (H)(float)(i*16+j);
    }
    H *dA, *dB; float* dD;
    HIP_CHECK(hipMalloc(&dA, sizeof(hA))); HIP_CHECK(hipMalloc(&dB, sizeof(hB))); HIP_CHECK(hipMalloc(&dD, sizeof(hD)));
    HIP_CHECK(hipMemcpy(dA, hA, sizeof(hA), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, sizeof(hB), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_once, dim3(1), dim3(32), 0, 0, dA, dB, dD);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(hD, dD, sizeof(hD), hipMemcpyDeviceToHost));
    int nonzero = 0, correct = 0;
    for (int i = 0; i < 256; ++i) {
        if (hD[i] != 0) ++nonzero;
        float want = (float)(i/16*16 + i%16);
        if (hD[i] == want) ++correct;
    }
    printf("f16 wmma: nonzero %d/256, 정확 %d/256\n", nonzero, correct);
    printf("D[0][0..3] = %.0f %.0f %.0f %.0f (기대 0 1 2 3)\n", hD[0], hD[1], hD[2], hD[3]);
    printf("D[1][0..3] = %.0f %.0f %.0f %.0f (기대 16 17 18 19)\n", hD[16], hD[17], hD[18], hD[19]);
    return 0;
}
