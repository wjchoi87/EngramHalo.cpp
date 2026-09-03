// G-1B 자동 mapping oracle: wmma fragment .x[] ↔ 논리 행렬 좌표 전체 매핑 산출
//  - basis input: 원소별 고유 ID (f16: 1+r*16+c / iu8: 래퍼 없이 부호없는 바이트)
//  - 검증: 완전성(모든 x 슬롯 매핑), bijection(중복/미커버), D redundancy 구조
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace rocwmma;
using H = _Float16;
#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP ERR %s @%d\n", hipGetErrorString(e), __LINE__); return 1;} } while(0)

// ===== f16 oracle =====
__global__ void k_frag_f16(const H* A, const H* B, float* xA, float* xB, float* xD, float* Dmat) {
    int lane = threadIdx.x % 32;
    fragment<matrix_a, 16, 16, 16, H, row_major> fa;
    fragment<matrix_b, 16, 16, 16, H, row_major> fb;
    fragment<accumulator, 16, 16, 16, float> fd;
    load_matrix_sync(fa, A, 16);
    load_matrix_sync(fb, B, 16);
    fill_fragment(fd, 0.0f);
    mma_sync(fd, fa, fb, fd);
    for (int i = 0; i < fa.num_elements; ++i) xA[lane * fa.num_elements + i] = (float)fa.x[i];
    for (int i = 0; i < fb.num_elements; ++i) xB[lane * fb.num_elements + i] = (float)fb.x[i];
    for (int i = 0; i < fd.num_elements; ++i) xD[lane * fd.num_elements + i] = (float)fd.x[i];
    store_matrix_sync(Dmat, fd, 16, mem_row_major);
}

// ===== iu8 oracle =====
__global__ void k_frag_iu8(const signed char* A, const signed char* B, int* xA, int* xB, int* xD, int* Dmat) {
    int lane = threadIdx.x % 32;
    fragment<matrix_a, 16, 16, 16, int8_t, row_major> fa;
    fragment<matrix_b, 16, 16, 16, int8_t, row_major> fb;
    fragment<accumulator, 16, 16, 16, int32_t> fd;
    load_matrix_sync(fa, A, 16);
    load_matrix_sync(fb, B, 16);
    fill_fragment(fd, 0);
    mma_sync(fd, fa, fb, fd);
    for (int i = 0; i < fa.num_elements; ++i) xA[lane * fa.num_elements + i] = (int)fa.x[i];
    for (int i = 0; i < fb.num_elements; ++i) xB[lane * fb.num_elements + i] = (int)fb.x[i];
    for (int i = 0; i < fd.num_elements; ++i) xD[lane * fd.num_elements + i] = (int)fd.x[i];
    store_matrix_sync(Dmat, fd, 16, mem_row_major);
}

static int g_fail = 0;

// 매핑 테이블: v (고유ID) → (row, col). iu8은 고유ID 부여를 위해
// A 원소 값 = ID (signed char 범위 내) — 여기서는 A/B 모두 값 = ID+1 (1..256은 불가 →
// iu8: 값 v ∈ [-128,127], ID = v+128 ∈ [0,255] → row/col 복원: ID = r*16+c
template<int MAXN>
static void dump_map(const char* tag, const std::vector<float>& x, int lanes, int elems,
                     int lanes_used_expected) {
    // xA[lane * elems + i] = ID → (row, col) 기록
    int seen[MAXN][16] = {};
    int dup = 0, mapped = 0;
    for (int l = 0; l < lanes; ++l)
        for (int i = 0; i < elems; ++i) {
            float v = x[l * elems + i];
            if (v == 0.0f) continue;
            int id = (int)v - 1;
            int r = id / 16, c = id % 16;
            if (r < 0 || r >= MAXN || c >= 16 || seen[r][c]) { ++dup; continue; }
            seen[r][c] = 1; ++mapped;
        }
    (void)lanes_used_expected;
    printf("  %s: mapped=%d dup=%d\n", tag, mapped, dup);
}

int main() {
    // ===== f16 oracle =====
    {
        std::vector<H> hA(256), hB(256), hD(256);
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c) {
                int id = 1 + r * 16 + c;          // 1..256, f16 정확 (정수 ≤2048)
                hA[r*16+c] = (H)(float)id;
                hB[r*16+c] = (H)(float)id;
            }
        H *dA, *dB; float *xA, *xB, *xD, *dD;
        HIP_CHECK(hipMalloc(&dA, 512)); HIP_CHECK(hipMalloc(&dB, 512));
        HIP_CHECK(hipMalloc(&xA, 32*64*sizeof(float))); HIP_CHECK(hipMalloc(&xB, 32*64*sizeof(float)));
        HIP_CHECK(hipMalloc(&xD, 32*64*sizeof(float))); HIP_CHECK(hipMalloc(&dD, 256*sizeof(float)));
        HIP_CHECK(hipMemcpy(dA, hA.data(), 512, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dB, hB.data(), 512, hipMemcpyHostToDevice));
        hipLaunchKernelGGL(k_frag_f16, dim3(1), dim3(32), 0, 0, dA, dB, xA, xB, xD, dD);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> hxA(32*64), hxB(32*64), hxD(32*64), hDm(256);
        HIP_CHECK(hipMemcpy(hxA.data(), xA, hxA.size()*4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hxB.data(), xB, hxB.size()*4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hxD.data(), xD, hxD.size()*4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hDm.data(), dD, 256*4, hipMemcpyDeviceToHost));

        printf("===== f16 wmma oracle =====\n");
        // A: A 매트릭스의 값 = 자기 좌표 ID → xA 덤프의 값이 곧 좌표
        // xA[lane*ne+i] 값 → (row, col): ID = 1+r*16+c
        // 출력: lane별 매핑 압축 (x index i ↔ (r,c))
        int cnt[16][16] = {};
        for (int l = 0; l < 32; ++l)
            for (int i = 0; i < 8; ++i) {
                float v = hxA[l*8+i];
                if (v <= 0) continue;
                int id = (int)(v + 0.5f) - 1;
                cnt[id/16][id%16]++;
            }
        // 매핑 요약: 각 (r,c)가 몇 개 lane·index에서 나타나는지
        int once = 0, multi = 0, zero = 0;
        for (int r = 0; r < 16; ++r) for (int c = 0; c < 16; ++c) {
            if (cnt[r][c] == 0) ++zero; else if (cnt[r][c] == 1) ++once; else ++multi;
        }
        printf("A frag: 1x=%d 2x+=%d 0x=%d (of 256)\n", once, multi, zero);
        // lane별 담당 열/행 패턴: lane l의 x[i] 값들의 (r,c)
        for (int l = 0; l < 4; ++l) {
            printf("  A lane %2d:", l);
            for (int i = 0; i < 8; ++i) {
                float v = hxA[l*8+i];
                int id = v > 0 ? (int)(v + 0.5f) - 1 : -1;
                printf(" [%d,%d]", id < 0 ? -1 : id/16, id < 0 ? -1 : id%16);
            }
            printf("\n");
        }
        // D store_matrix_sync 결과 vs xD 매핑 일치 확인:
        // Dmat (store 결과) == Σ A*B 정확 (A=B=ID 행렬이면 D[r][c] = Σ_k ID_A[r][k]*ID_B[k][c] —
        // 정합성은 별도 correctness 테스트에서, 여기서는 xD 덤프의 0/비0 분포만)
        int nzd = 0;
        for (int i = 0; i < 32*8; ++i) if (hxD[i] != 0) ++nzd;
        printf("D frag nonzero x 슬롯: %d/256\n", nzd);
        // B fragment 매핑 (lane l x[i] → B[r][c])
        printf("B frag mapping:\n");
        for (int l = 0; l < 32; ++l) {
            printf("  B lane %2d:", l);
            for (int i = 0; i < 8; ++i) {
                float v = hxB[l*8+i];
                int id = v > 0 ? (int)(v + 0.5f) - 1 : -1;
                printf(" [%d,%d]", id < 0 ? -1 : id/16, id < 0 ? -1 : id%16);
            }
            printf("\n");
        }
        // store_matrix_sync(Dmat) vs xD 검증: xD는 fd.x — store가 읽는 위치와 비교는
        // correctness 테스트에서 수행
    }

    // ===== iu8 oracle =====
    {
        // A 값: ID = r*16+c ∈ 0..255 → signed char로는 음수도 나옴.
        // iu8 = unsigned 해석이면 값 v = ID (0..255) 그대로. signed 해석이면 ± 절반.
        // 프로브: A = I*1 (단위) 대신 basis: 한 번에 한 원소 = 1, 나머지 0 → D = B의 해당 행.
        // 256회 실행은 과중 → 16회 (A row 단위) + B full 랜덤으로 매핑 역산.
        std::vector<signed char> hA(256), hB(256);
        for (int k = 0; k < 256; ++k) {
            int r = k / 16, c = k % 16;
            hB[k] = (signed char)(c + 1);          // B[r][c] = c+1 (행 구분용 아님, 열 ID)
        }
        std::vector<int> xA(32*64), xB(32*64), xD(32*64), Dm(256);
        std::vector<signed char> hA_i8(256, 0);
        signed char* dA; signed char* dB2; int *xA_, *xB_, *xD_, *dD;
        HIP_CHECK(hipMalloc(&dA, 256)); HIP_CHECK(hipMalloc(&dB2, 256));
        HIP_CHECK(hipMalloc(&xA_, 32*64*4)); HIP_CHECK(hipMalloc(&xB_, 32*64*4));
        HIP_CHECK(hipMalloc(&xD_, 32*64*4)); HIP_CHECK(hipMalloc(&dD, 256*4));
        HIP_CHECK(hipMemcpy(dB2, hB.data(), 256, hipMemcpyHostToDevice));

        // A row r = 1 (r행 전체 1): D[r][c] = B[r][c] → D 위치로 A 행 매핑 관찰
        // 여기서는 xA 덤프로 프래그먼트 매핑을 직접: A = I (대각 1) → xA 값 패턴 관찰
        std::vector<signed char> hI(256, 0);
        for (int k = 0; k < 16; ++k) hI[k*16+k] = 1;
        HIP_CHECK(hipMemcpy(dA, hA_i8.data(), 256, hipMemcpyHostToDevice));
        hipLaunchKernelGGL(k_frag_iu8, dim3(1), dim3(32), 0, 0, dA, dB2, xA_, xB_, xD_, dD);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(xA.data(), xA_, xA.size()*4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(xB.data(), xB_, xB.size()*4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(xD.data(), xD_, xD.size()*4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(Dm.data(), dD, 256*4, hipMemcpyDeviceToHost));

        printf("===== iu8 wmma oracle (A=I) =====\n");
        // xA: A=I → nonzero x 슬롯 = logical A[0..15][0..15] 대각. 값 = 1.
        // nonzero x 슬롯 (lane, i)의 분포 → 어떤 lane·슬롯이 논리 대각을 담는지
        int nz = 0;
        std::vector<std::pair<int,int>> pos;
        for (int l = 0; l < 32; ++l)
            for (int i = 0; i < 16; ++i)
                if (xA[l*16+i] != 0) { ++nz; pos.push_back({l, i}); }
        printf("A(I) nonzero x 슬롯: %d (대각 16개의 기재 위치)\n", nz);
        for (auto& p : pos) printf("  lane=%2d x[%2d]\n", p.first, p.second);
        // iu8 x num_elements: 16/lane (32 lanes × 16 = 512 = 2× redundant)
        printf("D store_matrix_sync 결과 (A=I → D=B, B[r][c]=c+1):\n");
        for (int r = 0; r < 4; ++r) {
            printf("  row %d:", r);
            for (int c = 0; c < 8; ++c) printf(" %d", Dm[r*16+c]);
            printf("\n");
        }
    }
    return 0;
}
