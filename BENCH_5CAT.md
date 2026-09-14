# 5품종 모드 비교

생성 2026-09-09 21:28 · 교정 세트 `val_good` (LOCO 자체 validation/good — **학습과 다른 촬영 파티션**)

AUROC (양품 = test/good, 이상 = logical + structural):

| 품종 | struct | logic | both | 최고 | both − 최고 |
|---|---:|---:|---:|---|---:|
| juice_bottle | 99.90 | 73.27 | 99.72 | **struct** | -0.18 |
| breakfast_box | 86.52 | 70.30 | 84.97 | **struct** | -1.56 |
| pushpins | 92.66 | 53.23 | 91.57 | **struct** | -1.09 |
| screw_bag | 82.17 | 49.96 | 72.94 | **struct** | -9.23 |
| splicing_connectors | 95.26 | 62.07 | 94.34 | **struct** | -0.91 |

**최고 모드 분포: {'struct': 5}**

## struct 모드의 운용점 (걸리는 장 수)

| 품종 | 양품 | 이상 | σ=1 오검출/미검 | σ=2 | σ=3 |
|---|---:|---:|---|---|---|
| juice_bottle | 94 | 236 | 14/0 | 5/2 | 1/5 |
| breakfast_box | 102 | 173 | 1/69 | 1/95 | 0/119 |
| pushpins | 138 | 172 | 97/0 | 53/13 | 26/24 |
| screw_bag | 122 | 219 | 6/115 | 2/158 | 0/182 |
| splicing_connectors | 119 | 193 | 9/24 | 4/32 | 3/41 |

## 논리부(patch-hist)가 얼마나 약한가

| 품종 | logic 전체 | logic 의 logical | 원본 CSAD 참고 |
|---|---:|---:|---:|
| juice_bottle | 73.27 | 77.59 | 91.20 |
| breakfast_box | 70.30 | 79.80 | — |
| pushpins | 53.23 | 52.11 | 90.39 |
| screw_bag | 49.96 | 46.57 | 99.95 |
| splicing_connectors | 62.07 | 64.56 | 88.40 |

원본 CSAD 는 **라벨된 test 셋으로 AUROC 베스트를 골라** 얻은 값이다(`11_SCHAD/FINDINGS.md`). 우리 논리부는 현장 조건 그대로 **양품만** 보고
고정 epoch 로 학습한다 — 그 차이가 이 표의 간격이다.