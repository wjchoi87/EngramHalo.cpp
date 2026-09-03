# G-1B WMMA 상태 (2026-09-04, correctness 게이트 통과)

## 확정 사실
1. **gfx1151 WMMA 동작**: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (rocWMMA f16 16x16x16) 실동작.
   iu8 변형도 동작. raw 마이크로벤치 ~727 TOPS(MAC), Strix Halo 830 TOPS(INT8) 대비 ~88%.
2. **프래그먼트 레이아웃 완전 오라클** (`wmma_oracle.hip`, 자동 basis 프로브 + bijection 검증):
   - A: lane l, x[i] → A[l%16][8*(l/16)+i]
   - B: lane l, x[i] → B[8*(l/16)+i][l%16]
   - D(f32 acc): lane l, x[i] → D[l%16][2*(l/16)+i] (store_matrix_sync가 동일 좌표계로 검증됨)
   - rocWMMA fragment<matrix_b> .x[]는 raw builtin 레지스터 레이아웃과 1:1 (identity-A 프로브로 확인)
3. **ROCm 10.0**: 유의미한 차이 없음 → 7.2.4 유지 확정.

## correctness 게이트 통과 (v4, `rocmfp4_wmma_v4.hip`)
- 구조: B 프래그먼트를 ROCmFP4(qs 니블 16B + UE4M3 scale 2B/32k)에서 **레지스터 직접 디코드** →
  f16 WMMA mma_sync → f32 acc. A는 f16 그대로 (활성화 staging은 통합 시 F32→F16 변환).
- 오라클 매핑 그대로 fb.x[i] = codebook(nib) × ue4m3_full(scale), i=0..7 lane-local.
- 결과 (M=64, K=128, N=64, random A ±7 / random W / random UE4M3 scale):
  - **0/4096 불일치, maxabs=0.016~0.023** (f32 누적 1-2 ulp, 값 크기 ~4M 대비) — CPU f64 참조 대비
  - SEGS=1/2/4 (독립 누산기 분할) 모두 통과
- 마지막 버그: 홀수 k-half의 scale byte가 qs 오프셋(`(kh%2)*8`)이 포함된 포인터에서 읽혀
  `block_base[16+9*(kh%2)]`를 읽던 것(+8 어긋남). identity-A 프로브가 kh=0만 검증해 마스킹됐었음.
  → blk(블록 베이스) / src(qs 베이스) 분리로 수정.

## sustained throughput 확정 (M=256, K=4096, N=4096, 20 passes, 5회 반복)
| 구성 | T-MAC/s |
|---|---|
| SEGS=1 (단일 acc 체인) | 5.93~6.29 |
| SEGS=2 / SEGS=4 (독립 acc) | **6.05~6.48, 평균 ≈ 6.27** |
| **G-1A dp4a MMQ (production)** | 3.0 |
| → **WMMA/dp4a 비율** | **2.1×** |

- 독립 누산기(S≥2) 추가 이득 미미: 1024 블록 × 4 워프 오버서브스크립션으로 mma 레이턴시 이미 은닉.
  현재 한계는 B 디코드 ALU + 메모리 경향 (0.5625 B/elem, dp4a와 동일 트래픽).

## 다음 단계
1. **full-model 통합** (standalone correctness 통과 후 허용): F32 src1 → f16 staging → wmma GEMM,
   M 충분히 큰(prefill) case에서 dp4a MMQ 대신 dispatch. MMVQ/decode는 기존 dp4a 유지.
2. 통합 후 Flash-Next direct-quant A/B 재측정 (prefill 중심 개선 기대).
