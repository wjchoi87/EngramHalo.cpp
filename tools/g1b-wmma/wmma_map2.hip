// A 바이트 단위 단일 원소 프로브: 하드웨어 (lane,byte) → 논리 (row,col) 완전 매핑
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
typedef int v4i __attribute__((vector_size(16)));
typedef int v8i __attribute__((vector_size(32)));
__global__ void k_once(const v4i* A, const v4i* B, int* D) {
    int l = threadIdx.x;
    v4i a = A[l], b = B[l];
    v8i c = {};
    v8i dv = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
    int* dp = (int*)&dv;
    #pragma unroll
    for (int j = 0; j < 8; ++j) D[l*8+j] = dp[j];
}
int main() {
    v4i hB[32];
    memset(hB, 0, sizeof(hB));
    // B[r][c]: lane r 행, byte c → 값 = r*16+c + 1 (1..256, u8 범위: &0xFF, 256→0 주의)
    // 안전하게: 값 v = r + 16*c? 도 유니크. 구분 편의상 v = (r+1) + (c)*16 → c=15,r=15 → 255 ✓ r=0,c=0 → 1 ✓
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) {
            unsigned v = (r + 1) + c * 16;      // 1..256, c=15&r=15 → 256 → 0 (충돌!)
            v = v & 0xFF; if (v == 0) v = 200;  // 256만 예외 처리 (r15c15)
            ((unsigned char*)&hB[r])[c] = (unsigned char)v;
        }
    v4i* dA; v4i* dB; int* dD;
    hipMalloc(&dA, 32*16); hipMalloc(&dB, 32*16); hipMalloc(&dD, 32*8*4);
    hipMemcpy(dB, hB, sizeof(hB), hipMemcpyHostToDevice);
    // B 디코드 테이블: 값 → (r,c)
    int v2r[256], v2c[256];
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) {
            unsigned v = (r + 1) + c * 16; v &= 0xFF; if (v == 0) v = 200;
            v2r[v] = r; v2c[v] = c;
        }
    int unresolved = 0;
    for (int L = 0; L < 32; ++L)
        for (int K = 0; K < 16; ++K) {
            v4i hA[32]; memset(hA, 0, sizeof(hA));
            ((unsigned char*)&hA[L])[K] = 1;
            hipMemcpy(dA, hA, sizeof(hA), hipMemcpyHostToDevice);
            hipLaunchKernelGGL(k_once, dim3(1), dim3(32), 0, 0, dA, dB, dD);
            hipDeviceSynchronize();
            int hD[256];
            hipMemcpy(hD, dD, 256*4, hipMemcpyDeviceToHost);
            int nz = 0, r_out = -1, c_out = -1, out_lane = -1, out_dw = -1;
            for (int l = 0; l < 32 && nz <= 1; ++l)
                for (int j = 0; j < 8; ++j) {
                    unsigned char v = ((unsigned char*)&hD[l*8+j])[0];  // byte0만 (i32 값 = B 요소)
                    if (v) { nz++; r_out = v2r[v]; c_out = v2c[v]; out_lane = l; out_dw = j; }
                }
            if (L < 2) {  // lane 0-1만 상세 덤프
                printf("A(%2d,%2d) →", L, K);
                for (int l = 0; l < 32; ++l)
                    for (int j = 0; j < 8; ++j) {
                        unsigned char v = ((unsigned char*)&hD[l*8+j])[0];
                        if (v) printf(" [lane%2d dw%d] v=%3d(A[%d][%d])", l, j, v, v2r[v], v2c[v]);
                    }
                printf("\n");
            }
        }
    printf("unresolved: %d\n", unresolved);
    return 0;
}
