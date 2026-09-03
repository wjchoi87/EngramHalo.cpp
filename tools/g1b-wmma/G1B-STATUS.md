# G-1B WMMA 조사 상태 (2026-09-04)

## 확정 사실
1. **gfx1151에서 WMMA 실행 확인**: `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`와
   rocWMMA f16 16x16x16 모두 gfx1151 하드웨어에서 실제 실행·정확 결과.
   - rocWMMA minimal test: A=I → D=A·B 256/256 bit-exact
   - iu8 map2 오라클: A 원소 단위 프로브로 하드웨어 (lane,byte) → 논리 (row,col) 매핑 확인
2. **raw 처리량**: rocWMMA 8독립체인 마이크로벤치 ~727 TOPS(MAC) — Strix Halo 8060S
   마케팅 830 TOPS(INT8)와 정합하는 컴퓨트 바운드 실측. dp4a(sudot4) 체인 대비 약 500x.
3. **ROCm 10.0 A/B**: 유의미한 차이 없음 (rocWMMA 2.2.1 패치업, LLVM 24, gfx1151 WMMA
   lowering 변경 없음) — LLVM #219248: gfx1151 packed-integer 스케줄링 3.2x 격차(오픈).
   → **G-1B 개발 baseline은 ROCm 7.2.4 유지** (승격 불필요).
4. **이전 -D__GFX11__ 실패 원명확화**: ISA 미지원이 아니라 upstream wmma 커널의
   wave/레이아웃 가정 문제. WMMA 자체는 gfx1151 지원.

## 미해결 (통합 커널 상태)
- ROCmFP4 → codebook i8 디코드 → shared tile → wmma_iu8 → f32 정산 누적 구조의
  통합 커널: **correctness 미통과** (identity-A 프로브에서 D[m][0] 전부 1.0 —
  wmma accumulator 프래그먼트의 비표준 레이아웃(원소 이중 기재 + lane/col 교차)을
  load_matrix_sync/store_matrix_sync가 처리하지 못하는 구조 또는 f/g 인덱스 매핑 오류).
- 처리량: sync/정산 병목으로 1.24-2.53 T-MAC/s (dp4a MMQ 3.0 미달).

## 다음 단계 후보
1. wmma accumulator 프래그먼트의 요소 매핑 완전 오라클 (map2 프로브의 byte1-3 커버)
2. rocwmma IOLayout (row/col_major 외) 조사 — wmma API 대신 mma API + IOLayout 제어
3. dequant→f16 경로: UE4M3 랜덤 스케일 정합성 원인 (f16 overflow 아님 확인, 스케일
   바이트↔half 매핑 재검증)
4. 프래그먼트 레이아웃 비의존 구조: 디코드를 shared가 아니라 레인별 직접
   (스트라이드 매핑 사전 계산) 하는 방식
