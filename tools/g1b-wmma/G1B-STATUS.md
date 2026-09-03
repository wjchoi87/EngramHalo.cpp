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

## full-model 통합 (2026-09-04, build-g1a)
- `ggml/src/ggml-cuda/rocmfp4-wmma.cu` 신설: F32 src1 → F16 staging → v4 커널(SEGS=2).
  dispatch: `ggml_cuda_mul_mat`에서 eligible — src0=Q4_0_ROCMFP4, 2D, contiguous, K/N/M%32==0, M≥64,
  `GGML_ROCMFP4_WMMA=0`로 런타임 비활성(동일 바이너리 A/B용), `GGML_ROCMFP4_WMMA_DEBUG=1`로 shape 덤프.
  gfx1151 판정은 host pass에 device 매크로가 없으므로 CMake(`ggml-hip/CMakeLists.txt`)가
  `CMAKE_HIP_ARCHITECTURES`에서 `GGML_ROCMFP4_WMMA_GFX1151`를 define.
- standalone은 `[k/32][n]` 자체 레이아웃이었으나 ggml은 row-major `[n][k/32]` — 통합 시 2건의 버그:
  1) W 블록 인덱스 전치 → illegal memory access (작은 N·큰 K에서 OOB)
  2) dst store 전치(col_major 실수) → 버퍼 밖 write. GGML dst[n + m*N]에는 v4의 row-major store 그대로.
  둘 다 `repro_server_shape.hip` (K=320/N=10240/M=128, K=10240/N=320/M=128)로 격리 재현 후 수정.
- correctness: 서버 greedy 사실성 테스트(Paris/Berlin/Rome) G-1A baseline과 토큰까지 동일.
  227토큰 raw 프롬프트 연속 생성은 ON/OFF 모두 degenerate 반복 — baseline 특성, wmma 무관.
- K=10240 재현에서 maxabs=6.0은 |ref|~1.1e6 지점의 f32 누적 반올림(K·|W| ~3.4e8, 1-2 ulp)으로 benign.

## full A/B — llama-bench 교차 반복 (2026-09-04, 동일 바이너리 env 토글, Flash-Next ROCmFP4 110.47GiB)
조건: `-ngl 99 -p 512 -n 128 -ub 128 -r 3`, ON→OFF→ON→OFF 교차 (서멀 드리프트 상쇄), UMA env.
| 지표 | WMMA OFF (dp4a MMQ) | WMMA ON | 개선 |
|---|---|---|---|
| pp512 (prefill) | 201.0 ± 5.3 / 208.7 ± 5.3 | **306.0 ± 26.7 / 305.2 ± 26.7** | **+49%** (라운드 양쪽 일관, 노이즈 밖 분리) |
| tg128 (decode) | 917.5 ± 117.6 / 1123.3 ± 189.8 | 913.6 ± 123.6 / 1139.0 ± 202.5 | 변화 없음 (경로 미개재 확인용) |
- wmma dispatch 커버리지: 128토큰 청크당 2 op (K=320/N=10240, K=10240/N=320 hc 쌍)만이 eligible — 그럼에도 pp +49%.
- MUL_MAT_ID(MoE expert)·FA·Q4_0_ROCMFP4_FAST는 기존 경로 — 후속 확장 대상.
- decode 절대값(1.1-1.2ms/token)은 G-1A 기록(23 t/s)과 다름 — ON/OFF 동일하므로 wmma 무관, 별도 프로토콜 조사 필요.
