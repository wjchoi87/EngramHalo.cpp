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
template <int SEGS>
__global__ void k_rocmfp4_wmma_gemm(const wmma_h * __restrict__ Af,
                                    const unsigned char * __restrict__ Wq,
                                    float * __restrict__ D,
                                    const int K, const int N,
                                    const int w_stride_blocks, const int d_ldm) {
    using namespace rocwmma;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int mh = warp / 2, nh = warp % 2;
    const int m0 = blockIdx.y * 32 + mh * 16;
    const int n0 = blockIdx.x * 32 + nh * 16;

    fragment<accumulator, 16, 16, 16, float> acc[SEGS];
#pragma unroll
    for (int s = 0; s < SEGS; ++s) {
        fill_fragment(acc[s], 0.0f);
    }

    const int K16 = K / 16;
    const int seg = (K16 + SEGS - 1) / SEGS;
    for (int s = 0; s < SEGS; ++s) {
        for (int kh = s*seg; kh < (s+1)*seg && kh < K16; ++kh) {
            fragment<matrix_a, 16, 16, 16, float16_t, row_major> fa;
            load_matrix_sync(fa, Af + (size_t)m0*K + kh*16, K);

            // 오라클 매핑: lane l의 fb.x[i] ↔ W[k = kh*16 + 8*(l/16) + i][n = n0 + l%16]
            // ggml row-major 블록 레이아웃 [n][k/32]: 행 바이트 오프셋 n*w_stride_blocks*18, 블록 (kh/2)*18
            const unsigned char * row = Wq + (size_t)(n0 + lane%16) * w_stride_blocks * 18;
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
            mma_sync(acc[s], fa, fb, acc[s]);
        }
    }
#pragma unroll
    for (int s = 1; s < SEGS; ++s) {
#pragma unroll
        for (int i = 0; i < acc[0].num_elements; ++i) {
            acc[0].x[i] += acc[s].x[i];
        }
    }
    // GGML dst[n + m*ldm] (n fast). acc의 (i=m, j=n)를 row_major store: ptr[i*ldm + j]
    store_matrix_sync(D + n0 + (size_t)m0*d_ldm, acc[0], d_ldm, mem_row_major);
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
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    if (src1->ne[0] != K || dst->ne[0] != N || dst->ne[1] != M) {
        return false;
    }
    if (src1->ne[2] != 1 || src1->ne[3] != 1) { // 2D 배치만 (MUL_MAT_ID/3D는 기존 MMQ 경로)
        return false;
    }
    if (K % 32 != 0 || N % 32 != 0 || M % 32 != 0) {
        return false;
    }
    if (M < 64) { // 작은 배치는 MMVQ/MMQ 유지
        return false;
    }
    if (src0->nb[1] % 18 != 0 || dst->nb[1] % 4 != 0) {
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
    dim3 grid((unsigned)(N/32), (unsigned)(M/32));
    k_rocmfp4_wmma_gemm<2><<<grid, dim3(128), 0, stream>>>(
        src1_f16.ptr, (const unsigned char *) src0->data, (float *) dst->data,
        (int) K, (int) N, w_stride_blocks, d_ldm);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED(ctx); GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst);
    GGML_ABORT("rocmfp4 wmma path not compiled");
#endif
}
