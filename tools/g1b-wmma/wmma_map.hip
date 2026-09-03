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
    v4i hA[32], hB[32];
    memset(hA, 0, sizeof(hA)); memset(hB, 0, sizeof(hB));
    // A = identity (lane l 행 가정), B[r][c] = r*16+c (고유값, 최대 255)
    for (int l = 0; l < 32; ++l) {
        signed char* a = (signed char*)&hA[l];
        unsigned char* b = (unsigned char*)&hB[l];
        for (int c = 0; c < 16; ++c) {
            a[c] = (l == c) ? 1 : 0;
            b[c] = (unsigned char)((l*16 + c) & 0xFF);
        }
    }
    v4i *dA, *dB; int* dD;
    hipMalloc(&dA, sizeof(hA)); hipMalloc(&dB, sizeof(hB)); hipMalloc(&dD, 32*8*4);
    hipMemcpy(dA, hA, sizeof(hA), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, sizeof(hB), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k_once, dim3(1), dim3(32), 0, 0, dA, dB, dD);
    hipDeviceSynchronize();
    int hD[256];
    hipMemcpy(hD, dD, sizeof(hD), hipMemcpyDeviceToHost);
    // A=I → D[a][b] = B[a][b] = a*16+b → 값 v의 v/16=행, v%16=열
    printf("매핑 표 (lane, dword, byte → 논리 D[row][col]):\n");
    int mapped = 0, dup = 0;
    int seen[16][16]; memset(seen, 0, sizeof(seen));
    for (int l = 0; l < 32; ++l)
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < 4; ++k) {
                unsigned char v = ((unsigned char*)&hD[l*8+j])[k];
                if (v == 0) continue;  // 값 0은 (0,0)과 충돌 — B[0][0]=0만 해당
                int r = v / 16, c = v % 16;
                if (seen[r][c]) { dup++; continue; }
                seen[r][c] = 1; mapped++;
                printf("lane=%2d dw=%d byte=%d → D[%2d][%2d]\n", l, j, k, r, c);
            }
    printf("고유 매핑: %d/255 (dup %d), lane16-31 전부 0이면 wave32 반반 구조\n", mapped, dup);
    return 0;
}
