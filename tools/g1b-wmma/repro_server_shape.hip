// G-1B 통합 커널 독립 재현: ggml row-major 레이아웃 [n][k/32] + 서버 shape
// usage: ./repro K N M [crash_only]
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace rocwmma;
using wmma_h = _Float16;
#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP ERR %s @%d\n", hipGetErrorString(e), __LINE__); return 1;} } while(0)

__device__ __forceinline__ float cbv(int nib) {
    static const float kv[16] = {0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10};
    return kv[nib & 0xF];
}
__device__ __forceinline__ float ue4m3(unsigned char x) {
    if (x == 0 || x == 0x7F || x == 0xFF) return 0.0f;
    const unsigned e = (x >> 3) & 0xF, m = x & 7;
    const float s = (x & 0x80) ? -1.0f : 1.0f;
    return s * (1.0f + m / 8.0f) * exp2f((float)e - 7.0f);
}

template <int SEGS, int NT>
__global__ void k_gemm(const wmma_h * __restrict__ Af,
                       const unsigned char * __restrict__ Wq,
                       float * __restrict__ D,
                       const int K, const int N,
                       const int w_stride_blocks, const int d_ldm,
                       const int kh_chunk, const size_t plane_stride) {
    using namespace rocwmma;
    const int z = blockIdx.z;
    const int kh0 = z*kh_chunk;
    const int kh1 = kh0 + kh_chunk;
    const size_t plane = (size_t)z*plane_stride;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int mh = warp / 2, nh = warp % 2;
    const int m0 = blockIdx.y * 32 + mh * 16;
    const int n0 = blockIdx.x * (32*NT) + nh * (16*NT);

    fragment<accumulator, 16, 16, 16, float> acc[SEGS][NT];
#pragma unroll
    for (int s = 0; s < SEGS; ++s)
#pragma unroll
        for (int t = 0; t < NT; ++t) fill_fragment(acc[s][t], 0.0f);

    const int K16 = kh1 - kh0;
    const int seg = (K16 + SEGS - 1) / SEGS;
    for (int s = 0; s < SEGS; ++s) {
        for (int kh = kh0 + s*seg; kh < kh0 + (s+1)*seg && kh < kh1; ++kh) {
            fragment<matrix_a, 16, 16, 16, float16_t, row_major> fa;
            load_matrix_sync(fa, Af + (size_t)m0*K + kh*16, K);

#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const int nt0 = n0 + t*16;
                const unsigned char * row = Wq + (size_t)(nt0 + lane%16) * w_stride_blocks * 18;
                const unsigned char * src = row + (kh/2)*18 + (kh%2)*8;
                int q4;
                memcpy(&q4, src + 4*(lane/16), 4);
                const float wsc = ue4m3(row[(kh/2)*18 + 16 + (kh%2)]);

                fragment<matrix_b, 16, 16, 16, float16_t, row_major> fb;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int nib = (q4 >> (8*(i/2) + 4*(i%2))) & 0xF;
                    fb.x[i] = (float16_t)(cbv(nib) * wsc);
                }
                mma_sync(acc[s][t], fa, fb, acc[s][t]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int s = 1; s < SEGS; ++s)
#pragma unroll
            for (int i = 0; i < acc[0][0].num_elements; ++i) acc[0][t].x[i] += acc[s][t].x[i];
        store_matrix_sync(D + plane + (n0 + t*16) + (size_t)m0*d_ldm, acc[0][t], d_ldm, mem_row_major);
    }
}

__global__ void k_reduce(const float * __restrict__ p, float * __restrict__ D, const int mn, const int S) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < mn) {
        float s = 0.0f;
        for (int z = 0; z < S; ++z) s += p[(size_t)z*mn + i];
        D[i] = s;
    }
}

__global__ void k_conv(const float * __restrict__ x, wmma_h * __restrict__ y, const int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = (wmma_h)x[i];
}

static void cpu_ref(int M, int K, int N, const std::vector<unsigned char>& Wq,
                    const std::vector<float>& A, std::vector<float>& D, int wsb) {
    auto cb = [](int n) { static const float kv[16] = {0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10}; return kv[n & 0xF]; };
    auto ue = [](unsigned char x) {
        if (x == 0 || x == 0x7F || x == 0xFF) return 0.0f;
        unsigned e = (x >> 3) & 0xF, m = x & 7;
        float s = (x & 0x80) ? -1.0f : 1.0f;
        return s * (1.0f + m / 8.0f) * exp2f((float)e - 7.0f);
    };
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0;
            for (int k = 0; k < K; ++k) {
                const unsigned char* row = &Wq[(size_t)n * wsb * 18];
                unsigned char byte = row[(k/32)*18 + (k%32)/2];
                int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
                s += (double)A[(size_t)m*K + k] * (cb(nib) * ue(row[(k/32)*18 + 16 + (k%32)/16]));
            }
            D[(size_t)m*N + n] = (float)s;   // D[m][n], ldm=N (col_major store와 정합)
        }
}

int main(int argc, char** argv) {
    int K = (argc > 1) ? atoi(argv[1]) : 320;
    int N = (argc > 2) ? atoi(argv[2]) : 10240;
    int M = (argc > 3) ? atoi(argv[3]) : 128;
    printf("repro K=%d N=%d M=%d\n", K, N, M);

    const int wsb = K / 32;
    size_t wq_size = (size_t)wsb * N * 18;
    std::vector<unsigned char> hWq(wq_size);
    srand(7);
    for (auto& v : hWq) v = rand() & 0xFF;
    std::vector<float> hA((size_t)M*K);
    for (auto& v : hA) v = (float)(rand()%15 - 7);
    std::vector<float> hD((size_t)M*N);

    unsigned char* dWq; float* dAf32; wmma_h* dA; float* dD;
    HIP_CHECK(hipMalloc(&dWq, wq_size));
    HIP_CHECK(hipMalloc(&dAf32, hA.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dA, hA.size()*sizeof(wmma_h)));
    HIP_CHECK(hipMalloc(&dD, hD.size()*sizeof(float)));
    HIP_CHECK(hipMemcpy(dWq, hWq.data(), wq_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAf32, hA.data(), hA.size()*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dD, 0, hD.size()*sizeof(float)));

    {
        int n = M*K, blk = 256, grid = (n + blk - 1)/blk;
        k_conv<<<grid, blk>>>(dAf32, dA, n);
        HIP_CHECK(hipDeviceSynchronize());
    }

    dim3 grid(N/32, M/32);
    printf("grid %u x %u\n", grid.x, grid.y);
    const int K16 = K/16;
    int nt = 1;
    if (N % 128 == 0 && K >= 2048) nt = 4;
    else if (N % 64 == 0 && K >= 1024) nt = 2;
    grid.x = N/(32*nt);
    const int64_t tiles = (N/(32*nt))*(M/32);
    int split = 1;
    while (split < 8 && tiles*split < 512 && K16 % (2*split) == 0 && split*M*N < (64 << 20)) split *= 2;
    printf("nt=%d split=%d grid=%u,%u\n", nt, split, grid.x, grid.y);
    float* dpart = nullptr;
    if (split > 1) hipMalloc(&dpart, (size_t)split*M*N*sizeof(float));
    auto launch = [&](float* D, int khc, size_t ps, int z) {
        dim3 g(grid.x, grid.y, z);
        if (nt == 4) k_gemm<2,4><<<g, dim3(128), 0, 0>>>(dA, dWq, D, K, N, wsb, N, khc, ps);
        else if (nt == 2) k_gemm<2,2><<<g, dim3(128), 0, 0>>>(dA, dWq, D, K, N, wsb, N, khc, ps);
        else k_gemm<2,1><<<g, dim3(128), 0, 0>>>(dA, dWq, D, K, N, wsb, N, khc, ps);
    };
    if (split == 1) launch(dD, K16, 0, 1);
    else {
        launch(dpart, K16/split, (size_t)M*N, split);
        int mn = M*N, blk = 256;
        k_reduce<<<(mn+blk-1)/blk, blk>>>(dpart, dD, mn, split);
    }
    hipError_t e = hipDeviceSynchronize();
    printf("sync: %s\n", hipGetErrorString(e));
    if (e != hipSuccess) return 1;
    HIP_CHECK(hipMemcpy(hD.data(), dD, hD.size()*sizeof(float), hipMemcpyDeviceToHost));

    if (argc > 4) { // bench mode: argv[4] = passes
        int passes = atoi(argv[4]);
        hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
        // 웜업
        for (int i = 0; i < 3; ++i) {
            if (split==1) launch(dD, K16, 0, 1);
            else { launch(dpart, K16/split, (size_t)M*N, split); int mn=M*N,blk=256; k_reduce<<<(mn+blk-1)/blk,blk>>>(dpart,dD,mn,split); }
        }
        hipDeviceSynchronize();
        hipEventRecord(t0);
        for (int i = 0; i < passes; ++i) {
            if (split==1) launch(dD, K16, 0, 1);
            else { launch(dpart, K16/split, (size_t)M*N, split); int mn=M*N,blk=256; k_reduce<<<(mn+blk-1)/blk,blk>>>(dpart,dD,mn,split); }
        }
        hipEventRecord(t1); hipDeviceSynchronize();
        float ms; hipEventElapsedTime(&ms, t0, t1);
        double mac = (double)M*K*N*passes;
        printf("BENCH K=%d N=%d M=%d: %.3f ms/pass → %.2f T-MAC/s\n", K, N, M, ms/passes, mac/(ms*1e-3)/1e12);
        return 0;
    }
    std::vector<float> ref((size_t)(size_t)M*N);
    cpu_ref(M, K, N, hWq, hA, ref, wsb);
    int bad = 0; double maxabs = 0, maxref = 0, maxerr_at = 0;
    for (size_t i = 0; i < hD.size(); ++i) {
        double er = hD[i] - ref[i]; if (er < 0) er = -er;
        double ar = ref[i]; if (ar < 0) ar = -ar;
        if (ar > maxref) maxref = ar;
        if (er > maxabs) { maxabs = er; maxerr_at = ar; }
        if (er > 1.0) ++bad;
    }
    printf("correctness: |err|>1.0 %d/%zu, maxabs=%.3f (at |ref|=%.1f), max|ref|=%.1f\n", bad, hD.size(), maxabs, maxerr_at, maxref);
    return bad ? 1 : 0;
}
