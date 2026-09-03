// G-1B standalone: ROCmFP4 데이터 플로우 WMMA GEMM 벤치 v2
//   블록당 32n×64m, warp당 2m-frag×2n-frag×2k-half = 8 wmma/sync-cycle
//   스케일 정산: f32 shared 누적 (벤치에서 스케일 1.0)
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace rocwmma;
#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP ERR %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while(0)

__device__ __forceinline__ int cbv(int nib) {
    static const signed char kv[16] = {0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10};
    return kv[nib & 0xF];
}

constexpr int BM = 64, BN = 32, BK = 32;   // 64m × 32n, k-스텝 32
constexpr int NWARP = 4;

__global__ void k_gemm(const unsigned char* __restrict__ Wq,
                       const signed char*  __restrict__ As,
                       float* __restrict__ D,
                       int K, int N) {
    const int n0 = blockIdx.x * BN;
    const int m0 = blockIdx.y * BM;
    const int tid = threadIdx.x;
    const int warp = tid / 32;                 // 0..3
    const int mh = warp / 2;                   // m-half 0..1 (32행)
    const int nh = warp % 2;                   // n-half 0..1 (16n × 2 = 32n)

    __shared__ signed char sA[BM][BK];         // 2KB
    __shared__ signed char sW[BK][BN];         // 1KB
    __shared__ int   sI[NWARP][2][2][16*16];   // [warp][mh-frag? no: n-frag][k-half] — 아래 참조 32KB
    __shared__ float sD[BM][BN];               // 8KB

    for (int i = tid; i < BM*BN; i += 128) sD[i/BN][i%BN] = 0.0f;
    __syncthreads();

    for (int kb = 0; kb < K/BK; ++kb) {
        // 1) 스테이징: A 전체 64×32, W 디코드 32×32
        for (int i = tid; i < BM*BK; i += 128) {
            int m = i / BK, k = i % BK;
            sA[m][k] = As[(size_t)(m0+m)*K + kb*BK + k];
        }
        for (int i = tid; i < BN*BK; i += 128) {
            int k = i / BN, n = i % BN;
            unsigned char byte = Wq[(size_t)(kb*N + n0 + n)*18 + k/2];
            int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
            sW[k][n] = (signed char)cbv(nib);
        }
        __syncthreads();

        // 2) wmma: warp = (mh, nh) → m-frag 2개(16행) × n-frag 2개(16n... nh는 16n) —
        //    BN=32이므로 n-half nh가 16n 담당, m-half mh가 32m → m-frag 2개
        //    warp당: 2 m-frag × 1 n-frag × 2 k-half = 4 wmma
        fragment<matrix_a, 16, 16, 16, int8_t, row_major> fa_k0, fa_k1;  // k-half별 A 열
        fragment<matrix_b, 16, 16, 16, int8_t, row_major> fb_k0, fb_k1;
        fragment<accumulator, 16, 16, 16, int32_t> ia[2][2];  // [m-frag][k-half]
        #pragma unroll
        for (int f = 0; f < 2; ++f) { fill_fragment(ia[f][0], 0); fill_fragment(ia[f][1], 0); }
        load_matrix_sync(fb_k0, &sW[0][nh*16], BN);
        load_matrix_sync(fb_k1, &sW[16][nh*16], BN);
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            load_matrix_sync(fa_k0, &sA[mh*32 + f*16][0], BK);
            mma_sync(ia[f][0], fa_k0, fb_k0, ia[f][0]);
            load_matrix_sync(fa_k1, &sA[mh*32 + f*16][16], BK);
            mma_sync(ia[f][1], fa_k1, fb_k1, ia[f][1]);
        }

        // 3) int frag → shared
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            store_matrix_sync(&sI[warp][f][0][0], ia[f][0], 16, mem_row_major);
            store_matrix_sync(&sI[warp][f][1][0], ia[f][1], 16, mem_row_major);
        }
        __syncthreads();

        // 4) 정산 누적 (스케일 1.0): 전체 8192 elem / 128 thr = 64 each
        for (int i = tid; i < NWARP*2*2*16*16; i += 128) {
            int w = i / (2*2*16*16);
            int r1 = i % (2*2*16*16);
            int f = r1 / (2*16*16);
            int h = (r1 / (16*16)) % 2;
            int idx = r1 % (16*16);
            int r = idx / 16, c = idx % 16;
            int m = (w/2)*32 + f*16 + r;
            int n = (w%2)*16 + c;
            sD[m][n] += (float)sI[w][f][h][idx];
        }
        __syncthreads();
    }

    for (int i = tid; i < BM*BN; i += 128)
        D[(size_t)(m0 + i/BN)*N + n0 + i%BN] = sD[i/BN][i%BN];
}

static void cpu_ref(int M, int K, int N,
                    const std::vector<unsigned char>& Wq,
                    const std::vector<signed char>& As,
                    std::vector<float>& D) {
    auto cb = [](int n) -> int {
        static const int kv[16] = {0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10};
        return kv[n & 0xF];
    };
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = 0;
            for (int k = 0; k < K; ++k) {
                unsigned char byte = Wq[(size_t)(k/32*N + n)*18 + (k%32)/2];
                int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
                s += (float)As[(size_t)m*K + k] * cb(nib);
            }
            D[(size_t)m*N + n] = s;
        }
}

int main(int argc, char** argv) {
    int mode_correct = (argc > 1 && std::string(argv[1]) == "correct");
    int M = mode_correct ? 64 : 256;
    int K = mode_correct ? 128 : 4096;
    int N = mode_correct ? 32  : 4096;
    int passes = mode_correct ? 1 : 20;

    size_t wq_size = (size_t)(K/32) * N * 18;
    std::vector<unsigned char> hWq(wq_size);
    srand(7);
    for (auto& v : hWq) v = rand() & 0xFF;
    std::vector<signed char> hAs((size_t)M*K);
    for (auto& v : hAs) v = rand()%15 - 7;
    std::vector<float> hD((size_t)M*N);

    unsigned char* dWq; signed char* dAs; float* dD;
    HIP_CHECK(hipMalloc(&dWq, wq_size));
    HIP_CHECK(hipMalloc(&dAs, hAs.size()));
    HIP_CHECK(hipMalloc(&dD, hD.size()*sizeof(float)));
    HIP_CHECK(hipMemcpy(dWq, hWq.data(), wq_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAs, hAs.data(), hAs.size(), hipMemcpyHostToDevice));
    dim3 grid(N/BN, M/BM);

    if (mode_correct) {
        hipLaunchKernelGGL(k_gemm, grid, dim3(128), 0, 0, dWq, dAs, dD, K, N);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(hD.data(), dD, hD.size()*sizeof(float), hipMemcpyDeviceToHost));
        std::vector<float> ref((size_t)M*N);
        cpu_ref(M, K, N, hWq, hAs, ref);
        int bad = 0; float maxe = 0;
        for (int i = 0; i < M*N; ++i) {
            float e = hD[i] - ref[i]; if (e < 0) e = -e;
            if (e > maxe) maxe = e;
            if (e != 0) ++bad;
        }
        printf("correctness: %d/%d 불일치, maxerr=%.1f\n", bad, M*N, maxe);
        return bad ? 1 : 0;
    }

    hipLaunchKernelGGL(k_gemm, grid, dim3(128), 0, 0, dWq, dAs, dD, K, N);
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t t0, t1; HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
    HIP_CHECK(hipEventRecord(t0));
    for (int p = 0; p < passes; ++p)
        hipLaunchKernelGGL(k_gemm, grid, dim3(128), 0, 0, dWq, dAs, dD, K, N);
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipDeviceSynchronize());
    float ms; HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    double mac = (double)M*K*N*passes;
    printf("ROCmFP4+WMMA GEMM v2: M=%d K=%d N=%d passes=%d\n", M, K, N, passes);
    printf("  %.1f ms → %.2f T-MAC/s\n", ms, mac/(ms*1e-3)/1e12);
    printf("  (G-1A dp4a MMQ 실측: 3.0 T-MAC/s)\n");
    return 0;
}
