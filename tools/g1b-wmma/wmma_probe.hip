// G-1B standalone WMMA probe for gfx1151 (RDNA3.5)
//  1) 레이아웃 발견: identity 트릭으로 v_wmma_i32_16x16x16_iu8_w32의 lane/요소 매핑 도출
//  2) correctness: 랜덤 A,B CPU 참조 대비
//  3) throughput: wmma 체인 vs dp4a 체인
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef int v4i __attribute__((vector_size(16)));
typedef int v8i __attribute__((vector_size(32)));

// 16x16 i8 one wave32: 각 lane이 4 dword(16 byte) 보유
__device__ __forceinline__ v4i pack_row_i8(const signed char* row) {
    v4i v;
    memcpy(&v, row, 16);
    return v;
}

// --- 커널 1: identity B로 A 레이아웃 관찰 + identity A로 B 레이아웃 관찰 + 랜덤 정합 ---
__global__ void k_wmma_once(const v4i* A, const v4i* B, int* D) {
    int l = threadIdx.x;  // wave32 가정: 32 lane
    v4i a = A[l];
    v4i b = B[l];
    v8i c = {};
    v8i dv = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
    int* dp = (int*)&dv;
    #pragma unroll
    for (int j = 0; j < 8; ++j) D[l * 8 + j] = dp[j];
}

// --- 커널 2: throughput — 독립 accumulator 8개 체인 ---
__global__ void k_wmma_tp(const v4i* A, const v4i* B, v8i* out, int iters) {
    int l = threadIdx.x;
    v4i a = A[l], b = B[l];
    v8i acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = {};
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, acc[j], false);
    }
    long s = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) s += acc[j][0] + acc[j][7];
    if (s == 123456789) out[l] = acc[0];  // dead-store 방지
}

__global__ void k_dp4a_tp(const v4i* A, const v4i* B, int* out, int iters) {
    int l = threadIdx.x;
    v4i av = A[l], bv = B[l];
    int a4[4], b4[4], bb0[4], bb1[4], bb2[4], bb3[4];
    memcpy(a4, &av, 16); memcpy(b4, &bv, 16);
    for (int q = 0; q < 4; ++q) { bb0[q]=b4[q]; bb1[q]=b4[q]+1; bb2[q]=b4[q]+2; bb3[q]=b4[q]+3; }
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int it = 0; it < iters; ++it) {
        c0 = __builtin_amdgcn_sudot4(true, bb0[0], true, a4[0], c0, false);
        c1 = __builtin_amdgcn_sudot4(true, bb1[1], true, a4[1], c1, false);
        c2 = __builtin_amdgcn_sudot4(true, bb2[2], true, a4[2], c2, false);
        c3 = __builtin_amdgcn_sudot4(true, bb3[3], true, a4[3], c3, false);
    }
    long s = (long)c0 + c1 + c2 + c3;
    if (s == 123456789) out[l] = c0;
}

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP ERR %s @%s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)

int main() {
    // A/B: lane l = 16바이트. 의미 매핑은 미지 → identity로 관찰
    // A = identity 16x16: A[r][c] = (r==c). lane l의 row가 논리 행 r_x(l)이라면
    // B = 값행렬 B[r][c] = r*16+c (u8 해석) → D[a][b] = sum A[a][r] B[r][b] = B[a][b] (A=I)
    v4i hA[32], hB[32];
    memset(hA, 0, sizeof(hA));
    memset(hB, 0, sizeof(hB));
    for (int l = 0; l < 32; ++l) {
        signed char* arow = (signed char*)&hA[l];
        unsigned char* brow = (unsigned char*)&hB[l];
        for (int c = 0; c < 16; ++c) {
            arow[c] = (l == c) ? 1 : 0;   // lane l이 "행 l"을 담는 가정 (검증 대상)
            brow[c] = (unsigned char)((l * 16 + c) & 0xFF);
        }
    }
    v4i *dA, *dB; v8i *dD;
    HIP_CHECK(hipMalloc(&dA, sizeof(hA)));
    HIP_CHECK(hipMalloc(&dB, sizeof(hB)));
    HIP_CHECK(hipMalloc(&dD, 32 * 8 * sizeof(int)));
    HIP_CHECK(hipMemcpy(dA, hA, sizeof(hA), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, sizeof(hB), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_wmma_once, dim3(1), dim3(32), 0, 0, dA, dB, (int*)dD);
    HIP_CHECK(hipDeviceSynchronize());
    int hD[32*8];
    HIP_CHECK(hipMemcpy(hD, dD, sizeof(hD), hipMemcpyDeviceToHost));

    // D 매핑 관찰: D[lane l][dword j][byte k] 값 = B[a][b], a=l(가정), b=j*4+k
    printf("== D 레이아웃 관찰 (A=identity, B[r][c]=r*16+c) ==\n");
    printf("기대: A identity가 맞고 D[a][b]=B[a][b]=a*16+b 라면 D[l][j][k] = l*16+(j*4+k)\n");
    int non_zero_lanes = 0, first_ok = -1;
    for (int l = 0; l < 32; ++l) {
        bool any = false;
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < 4; ++k) {
                int v = ((unsigned char*)&hD[l*8+j])[k];
                if (v != 0) any = true;
            }
        if (any) non_zero_lanes++;
    }
    printf("non-zero lane 수: %d/32\n", non_zero_lanes);
    // 관찰표: lane 0과 1의 처음 16바이트 덤프
    for (int l = 0; l < 2; ++l) {
        printf("lane %2d d0..3 바이트:", l);
        for (int j = 0; j < 4; ++j) {
            unsigned char b4[4];
            memcpy(b4, &hD[l*8+j], 4);
            printf(" %3d %3d %3d %3d", b4[0], b4[1], b4[2], b4[3]);
        }
        printf("\n");
    }

    // === correctness: 랜덤 A,B (lane l이 row l을 담는 가정이 맞다면 D[a][b]=CPU mm) ===
    srand(42);
    static signed char Aref[16][16], Bref[16][16];
    for (int l = 0; l < 32; ++l) {
        signed char* ar = (signed char*)&hA[l];
        unsigned char* br = (unsigned char*)&hB[l];
        for (int c = 0; c < 16; ++c) {
            ar[c] = (signed char)(rand() % 15 - 7);
            br[c] = (unsigned char)(rand() % 15);
        }
    }
    // CPU 참조: A[a][r](i8) * B[r][b](u8), a,b < 16
    static int ref[16][16];
    // lane→행 매핑을 모르므로: 두 가설 검증. 가설 H1: lane l = row l.
    for (int a = 0; a < 16; ++a)
        for (int b = 0; b < 16; ++b) {
            int s = 0;
            for (int r = 0; r < 16; ++r) s += (int)Aref[a][r] * (int)Bref[r][b];
            ref[a][b] = s;
        }
    hipMemcpy(dA, hA, sizeof(hA), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, sizeof(hB), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k_wmma_once, dim3(1), dim3(32), 0, 0, dA, dB, (int*)dD);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(hD, dD, sizeof(hD), hipMemcpyDeviceToHost));
    // 가설 H1: D[l][j*4+k] == ref[l][j*4+k]
    int bad = 0, checked = 0;
    for (int l = 0; l < 16; ++l)
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < 4; ++k) {
                int got = ((unsigned char*)&hD[l*8+j])[k];
                // signed accumulator
                int sv; memcpy(&sv, &got, 4);
                int want = ref[l][j*4+k];
                checked++;
                if (sv != want) bad++;
            }
    printf("== correctness(H1 lane=row): %d/%d 불일치\n", bad, checked);

    // === throughput ===
    int iters = 20000;
    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
    // wmma: 워크그룹 32 lane × 8 acc × 4096 MAC = 1,048,576 MAC/iter/λ... 실측으로
    hipLaunchKernelGGL(k_wmma_tp, dim3(1024), dim3(32), 0, 0, dA, dB, (v8i*)dD, iters);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(t0));
    hipLaunchKernelGGL(k_wmma_tp, dim3(1024), dim3(32), 0, 0, dA, dB, (v8i*)dD, iters);
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipDeviceSynchronize());
    float ms_w; HIP_CHECK(hipEventElapsedTime(&ms_w, t0, t1));
    double wmma_ops = 1024.0 * 32 * 8 * (double)iters * 4096.0;  // MAC
    printf("wmma: %.1f ms → %.1f GOP/s (MAC)\n", ms_w, wmma_ops / (ms_w * 1e6));

    int* dO; HIP_CHECK(hipMalloc(&dO, 32 * sizeof(int)));
    hipLaunchKernelGGL(k_dp4a_tp, dim3(1024), dim3(32), 0, 0, dA, dB, dO, iters);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(t0));
    hipLaunchKernelGGL(k_dp4a_tp, dim3(1024), dim3(32), 0, 0, dA, dB, dO, iters);
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipDeviceSynchronize());
    float ms_d; HIP_CHECK(hipEventElapsedTime(&ms_d, t0, t1));
    double dp4a_ops = 1024.0 * 32 * 4 * (double)iters * 4.0;
    printf("dp4a: %.1f ms → %.1f GOP/s (MAC)\n", ms_d, dp4a_ops / (ms_d * 1e6));
    printf("비율 wmma/dp4a = %.2fx\n", (wmma_ops / ms_w) / (dp4a_ops / ms_d));
    return 0;
}
