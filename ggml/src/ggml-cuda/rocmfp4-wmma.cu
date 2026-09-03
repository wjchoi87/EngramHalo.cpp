// G-1B: ROCmFP4 → 레지스터 직접 디코드 → f16 WMMA mul_mat 경로 (gfx1151)
//
// standalone correctness 게이트 통과 (random A/W/UE4M3, 0/4096 불일치, f32 1-2 ulp) 및
// sustained 6.27 T-MAC/s (dp4a MMQ 3.0 대비 2.1×) — tools/g1b-wmma/rocmfp4_wmma_v4.hip 참조.
//
// 프래그먼트 레이아웃은 자동 매핑 오라클(tools/g1b-wmma/wmma_oracle.hip, bijection 검증) 산출:
//   B: lane l, x[i] → B[8*(l/16)+i][l%16]
// 활성화는 F32 → F16 staging(q8_1 quantize 생략), prefill 크기 배치에서만 dispatch.

#include "common.cuh"

// gfx1151은 host pass에서 __gfx1151__ 등 device 매크로가 정의되지 않으므로
// CMake(ggml-hip/CMakeLists.txt)가 CMAKE_HIP_ARCHITECTURES에 gfx1151이 있을 때
// GGML_ROCMFP4_WMMA_GFX1151을 define한다.
#if defined(GGML_USE_HIP) && defined(GGML_ROCMFP4_WMMA_GFX1151)
#define GGML_ROCMFP4_WMMA_ENABLED 1
#include <rocwmma/rocwmma.hpp>
#endif

// near-miss 계측: ROCmFP4 mul_mat이 eligible에서 거절되는 이유 기록 (GGML_ROCMFP4_WMMA_DEBUG=1)
#if defined(GGML_ROCMFP4_WMMA_ENABLED)
#define WMMA_REJECT_LOG(src0_, src1_, dst_, why_) \
    do { if (getenv("GGML_ROCMFP4_WMMA_DEBUG")) { \
        fprintf(stderr, "%s: reject[%s] K=%d N=%d M=%d ne1_1=%d ne2=%d cont=%d%d%d nb1=%zu\n", \
                __func__, why_, (int)(src0_)->ne[0], (int)(src0_)->ne[1], (int)(src1_)->ne[1], \
                (int)(src1_)->ne[1], (int)(src1_)->ne[2], \
                (int)ggml_is_contiguous(src0_), (int)ggml_is_contiguous(src1_), (int)ggml_is_contiguous(dst_), \
                (size_t)(src0_)->nb[1]); } } while (0)
#else
#define WMMA_REJECT_LOG(src0_, src1_, dst_, why_) ((void) 0)
#endif

__device__ __forceinline__ float wmma_rocmfp4_cbv(int nib) {
    static const float kv[16] = {0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10};
    return kv[nib & 0xF];
}

__device__ __forceinline__ float wmma_rocmfp4_ue4m3(unsigned char x) {
    if (x == 0 || x == 0x7F || x == 0xFF) return 0.0f;
    const unsigned e = (x >> 3) & 0xF, m = x & 7;
    const float s = (x & 0x80) ? -1.0f : 1.0f;
    return s * (1.0f + m / 8.0f) * exp2f((float)e - 7.0f);
}

#if defined(GGML_ROCMFP4_WMMA_ENABLED)

namespace {

using wmma_h = rocwmma::float16_t;

// 블록(4 warp) → 32x32 출력. warp는 16x16 타일.
// Af: src1 F16 staging [M x K] row-major (M = 토큰)
// Wq: src0 Q4_0_ROCMFP4, 32k 블록(18B: qs 16B + UE4M3 2B) 행우선 [K/32 x N]
// D : dst F32 [N x M] (GGML dst[n + m*ldm])
// split-K: kh ∈ [kh0, kh1) 구간만 계산해 Dp + plane에 누적 (plane stride = M*N, dst와 동일 인덱싱)
template <int SEGS, int NT>
__global__ void k_rocmfp4_wmma_gemm(const wmma_h * __restrict__ Af,
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
    for (int s = 0; s < SEGS; ++s) {
#pragma unroll
        for (int t = 0; t < NT; ++t) fill_fragment(acc[s][t], 0.0f);
    }

    const int K16 = kh1 - kh0;
    const int seg = (K16 + SEGS - 1) / SEGS;
    for (int s = 0; s < SEGS; ++s) {
        for (int kh = kh0 + s*seg; kh < kh0 + (s+1)*seg && kh < kh1; ++kh) {
            fragment<matrix_a, 16, 16, 16, float16_t, row_major> fa;
            load_matrix_sync(fa, Af + (size_t)m0*K + kh*16, K);

#pragma unroll
            for (int t = 0; t < NT; ++t) {
                // 오라클 매핑: lane l의 fb.x[i] ↔ W[k = kh*16 + 8*(l/16) + i][n = nt0 + l%16]
                // ggml row-major 블록 레이아웃 [n][k/32]: 행 바이트 n*w_stride_blocks*18, 블록 (kh/2)*18
                const int nt0 = n0 + t*16;
                const unsigned char * row = Wq + (size_t)(nt0 + lane%16) * w_stride_blocks * 18;
                const unsigned char * src = row + (kh/2)*18 + (kh%2)*8;
                int q4;
                memcpy(&q4, src + 4*(lane/16), 4);
                const float wsc = wmma_rocmfp4_ue4m3(row[(kh/2)*18 + 16 + (kh%2)]);

                fragment<matrix_b, 16, 16, 16, float16_t, row_major> fb;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int nib = (q4 >> (8*(i/2) + 4*(i%2))) & 0xF;
                    fb.x[i] = (float16_t)(wmma_rocmfp4_cbv(nib) * wsc);
                }
                mma_sync(acc[s][t], fa, fb, acc[s][t]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int s = 1; s < SEGS; ++s) {
#pragma unroll
            for (int i = 0; i < acc[0][0].num_elements; ++i) {
                acc[0][t].x[i] += acc[s][t].x[i];
            }
        }
        // GGML dst[n + m*ldm] (n fast). acc의 (i=m, j=n)를 row_major store: ptr[i*ldm + j]
        store_matrix_sync(D + plane + (n0 + t*16) + (size_t)m0*d_ldm, acc[0][t], d_ldm, mem_row_major);
    }
}

// split-K 부분합 환원: p[z][i] (i = m*N + n) → D[i]
__global__ void k_rocmfp4_wmma_reduce(const float * __restrict__ p, float * __restrict__ D,
                                      const int mn, const int S) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < mn) {
        float s = 0.0f;
        for (int z = 0; z < S; ++z) {
            s += p[(size_t)z*mn + i];
        }
        D[i] = s;
    }
}

__global__ void k_f32_to_f16_wmma(const float * __restrict__ x, wmma_h * __restrict__ y, const int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = (wmma_h)x[i];
    }
}

} // namespace

#endif // GGML_ROCMFP4_WMMA_ENABLED

bool ggml_cuda_rocmfp4_wmma_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
#if defined(GGML_ROCMFP4_WMMA_ENABLED)
    if (src0->type != GGML_TYPE_Q4_0_ROCMFP4) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        WMMA_REJECT_LOG(src0, src1, dst, "non-contiguous");
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    if (src1->ne[0] != K || dst->ne[0] != N || dst->ne[1] != M) {
        WMMA_REJECT_LOG(src0, src1, dst, "dim-mismatch");
        return false;
    }
    if (src1->ne[2] != 1 || src1->ne[3] != 1) { // 2D 배치만 (MUL_MAT_ID/3D는 기존 MMQ 경로)
        WMMA_REJECT_LOG(src0, src1, dst, "3d-batch");
        return false;
    }
    if (K % 32 != 0 || N % 32 != 0 || M % 32 != 0) {
        WMMA_REJECT_LOG(src0, src1, dst, "mod32");
        return false;
    }
    if (M < 64) { // 작은 배치는 MMVQ/MMQ 유지
        return false;
    }
    if (src0->nb[1] % 18 != 0 || dst->nb[1] % 4 != 0) {
        WMMA_REJECT_LOG(src0, src1, dst, "stride-align");
        return false;
    }
    // 런타임 토글: GGML_ROCMFP4_WMMA=0 → 기존 MMVQ/MMQ 경로 (동일 바이너리 A/B용)
    static const bool env_enabled = []() {
        const char * env = getenv("GGML_ROCMFP4_WMMA");
        return env == nullptr || atoi(env) != 0;
    }();
    if (!env_enabled) {
        return false;
    }
    if (getenv("GGML_ROCMFP4_WMMA_DEBUG")) {
        fprintf(stderr, "%s: wmma dispatch K=%d N=%d M=%d nb0=%zu nb1=%zu dst_nb1=%zu\n",
                __func__, (int)K, (int)N, (int)M, (size_t)src0->nb[0], (size_t)src0->nb[1], (size_t)dst->nb[1]);
    }
    return true;
#else
    GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst);
    return false;
#endif
}

void ggml_cuda_mul_mat_rocmfp4_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#if defined(GGML_ROCMFP4_WMMA_ENABLED)
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<wmma_h> src1_f16(ctx.pool(), K*M);
    {
        const int n_elem = (int)(K*M);
        const int block = 256;
        const int grid = (n_elem + block - 1) / block;
        k_f32_to_f16_wmma<<<grid, block, 0, stream>>>((const float *) src1->data, src1_f16.ptr, n_elem);
    }
    CUDA_CHECK(cudaGetLastError());

    const int w_stride_blocks = (int)(src0->nb[1] / 18);
    const int d_ldm           = (int)(dst->nb[1] / 4);

    // NT(타일당 열 폭): 큰 K에서 A 재판독이 병목 → N이 64/128 정렬이면 폭을 넓힌다
    int nt = 1;
    if (N % 128 == 0 && K >= 2048) nt = 4;
    else if (N % 64 == 0 && K >= 1024) nt = 2;
    dim3 grid((unsigned)(N/(32*nt)), (unsigned)(M/32));

    // split-K: 타일 수가 적어 CU가 채워지지 않는 small-N shape에서 k를 분할 (K16은 짝수 보장)
    const int K16 = (int)(K / 16);
    const int64_t tiles = (N/(32*nt)) * (M/32);
    int split = 1;
    while (split < 8 && tiles*split < 512 && K16 % (2*split) == 0 && split*M*N < (64 << 20)) {
        split *= 2;
    }

#define WMMA_LAUNCH(DP, KH, PS) \
    do { if (nt == 4) k_rocmfp4_wmma_gemm<2,4><<<grid, dim3(128), 0, stream>>>( \
                src1_f16.ptr, (const unsigned char *) src0->data, (DP), (int) K, (int) N, w_stride_blocks, d_ldm, (KH), (PS)); \
        else if (nt == 2) k_rocmfp4_wmma_gemm<2,2><<<grid, dim3(128), 0, stream>>>( \
                src1_f16.ptr, (const unsigned char *) src0->data, (DP), (int) K, (int) N, w_stride_blocks, d_ldm, (KH), (PS)); \
        else k_rocmfp4_wmma_gemm<2,1><<<grid, dim3(128), 0, stream>>>( \
                src1_f16.ptr, (const unsigned char *) src0->data, (DP), (int) K, (int) N, w_stride_blocks, d_ldm, (KH), (PS)); \
    } while (0)

    if (split == 1) {
        WMMA_LAUNCH((float *) dst->data, K16, 0);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    grid.z = (unsigned) split;
    ggml_cuda_pool_alloc<float> d_part(ctx.pool(), split*M*N);
    const int kh_chunk = K16 / split;
    WMMA_LAUNCH(d_part.ptr, kh_chunk, M*N);
    CUDA_CHECK(cudaGetLastError());
    {
        const int mn = (int)(M*N);
        const int block = 256;
        const int rgrid = (mn + block - 1) / block;
        k_rocmfp4_wmma_reduce<<<rgrid, block, 0, stream>>>(d_part.ptr, (float *) dst->data, mn, split);
    }
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED(ctx); GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst);
    GGML_ABORT("rocmfp4 wmma path not compiled");
#endif
}
