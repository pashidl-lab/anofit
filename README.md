# anofit

**양품 사진만으로 학습하는 기계 시각 이상 탐지.** 긁힘·깨짐 같은 **구조적** 불량과, 부품이
빠졌거나 자리가 바뀐 **논리적** 불량을 함께 잡는다. 학습부터 임계값 결정, 검사 시험까지
대시보드 하나로 한다.

*Anomaly detection for machine vision, trained from good images only — structural and
logical defects, with a dashboard. Documentation is in Korean; the CLI prints Korean.*

- 문서·이슈: <https://github.com/pashidl-lab/anofit> — 대시보드 따라하기: [`docs/dashboard-walkthrough.md`](docs/dashboard-walkthrough.md)
- 문의·라이선스: pashidl.lab@gmail.com

---

## 무엇을 하나

| 모드 | 잡는 것 | 바탕 |
|---|---|---|
| `struct` | 긁힘 · 깨짐 · 오염 · 변형 | SALAD (ICCV 2025) — 교사/학생 + 오토인코더 |
| `logic` | 부품 누락 · 개수 오류 · 위치 바뀜 | CSAD (BMVC 2024) — 부품 분할 + 히스토그램 |
| `dpat` | 양품과 다른 국소 패턴 (0.1.1) | DINOv2 패치 특징 + 양품 메모리 (학습 없음) |
| `both` | 둘 다 (기본) | 점수를 z-score 로 합침 |

0.1.1 부터는 **어느 분기를 쓸지 품종마다 정한다** — 티칭 때 받은 불량이 8장 이상이면 그것으로
고르고(후보 넷 중, 기본값보다 2%p 이상 나을 때만 교체), 없으면 구조부 기본값을 쓴다.
5품종 실측: 고정 `both` 90.91 · 고정 구조부 92.03 · **불량으로 고르면 92.69**
(breakfast_box 는 85.71 → 92.66).

MVTec LOCO 5품종 실측 (AUROC, 양품 = test/good, 이상 = logical + structural, 2026-09-09):

| 품종 | struct | logic | both |
|---|---:|---:|---:|
| juice_bottle | 99.90 | 73.27 | 99.72 |
| breakfast_box | 86.52 | 70.30 | 84.97 |
| pushpins | 92.66 | 53.23 | 91.57 |
| screw_bag | 82.17 | 49.96 | 72.94 |
| splicing_connectors | 95.26 | 62.07 | 94.34 |

이 숫자는 **라벨 없이, 양품만 보고** 학습한 값이다. 논문 값은 라벨된 테스트 셋으로 가장 좋은
에폭을 고른 것이라 직접 비교가 안 된다. 자세한 것은 `BENCH_REPORT.md`·`BENCH_5CAT.md`.

## 설치

```bash
# 1) CUDA 에 맞는 torch 를 먼저 (pip 은 CPU 판을 고른다)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) anofit — 학습까지 하려면 [train]
pip install "anofit[train]"

# 검사만 하는 기계(이미 학습한 번들로 점수만)면 이것으로 충분하다
pip install anofit
```

- Python 3.10 – 3.13, Windows / Linux
- 학습은 GPU 가 필요하다 (실측 요구 VRAM 은 `anofit doctor` 가 말해 준다). 검사는 CPU 로도 된다.
- 산출물·캐시·가중치를 어느 디스크에 둘지는 `anofit.yaml` 로 정한다 — 본보기 [`anofit.example.yaml`](anofit.example.yaml). 없으면 사용자 캐시 폴더에 쌓인다.

## 5분 시작

```bash
anofit doctor --mode both      # 이 기계로 되는가: GPU · 디스크 · 의존성 · 가중치
anofit fetch  --mode both      # 파운데이션 가중치 10.4 GB (원 배포처에서, sha256 대조, 이어받기)
anofit serve                   # 대시보드 → http://127.0.0.1:8732
```

대시보드에서: **준비** 탭에 양품 폴더(서버 경로 또는 업로드) → **학습** 탭에서 품종 이름을 정하고
시작 → 끝나면 **임계값** 탭에서 운용점을 고르고 **검사** 탭에 사진을 넣어 본다.

명령줄로 같은 일:

```bash
anofit train --category my_part --good ./good --mode both       # 번들 생성
anofit threshold <번들>                                          # 임계값이 뜻하는 것
anofit score <번들> photo.png --models-dir <번들>                 # 한 장 판정
anofit export <번들> -o my_part.afz                               # 현장으로 옮길 파일 하나
```

## 가중치는 동봉하지 않는다

`anofit fetch` 가 원 배포처에서 받는다. 각 가중치는 배포처의 라이선스를 따른다
(`THIRD_PARTY_NOTICES.md` §6, 매니페스트 `fetch_manifest.json`).

| | 크기 | 출처 |
|---|---:|---|
| GroundingDINO Swin-T | 0.69 GB | IDEA-Research |
| RAM++ Swin-L | 3.01 GB | recognize-anything |
| SAM-HQ ViT-H | 2.57 GB | SysCV |
| SAM ViT-H | 2.56 GB | Meta |
| EfficientAD teacher | 0.03 GB | SALAD 저장소 |
| Imagenette v2 | 1.56 GB | fast.ai |

`--mode struct` 만이면 6.7 GB, `logic` 만이면 6.3 GB. 첫 학습 때 라이브러리가 스스로 받는 것이
1 GB 쯤 더 있다 (bert-base-uncased, timm wide_resnet50_2, dino_vitbase8).

## 라이선스 · 체험

평가용 라이선스다 (`LICENSE`). **처음 실행한 날부터 90일**은 그대로 쓴다. 그 뒤에는
학습·대시보드·fetch 가 멈추고, **이미 학습한 번들로 검사하는 것은 계속 된다** — 돌고 있는
라인이 서지 않는다.

```bash
anofit license                       # 남은 날짜
anofit license --install "<key>"     # 받은 키 설치 (또는 환경 변수 ANOFIT_LICENSE)
```

키는 pashidl.lab@gmail.com 으로 요청한다. 학습한 모델과 사진은 전부 사용자의 것이다.

포함된 오픈소스(GroundingDINO · SAM · RAM++ · CSAD · SALAD)는 각자의 라이선스를 그대로
가지며, 무엇을 고쳤는지는 `THIRD_PARTY_NOTICES.md` 에 파일 단위로 적었다.

## 안 되는 것 · 알아 둘 것

- 안전이 걸린 곳에는 쓰지 않는다 (`LICENSE` §7).
- 정확도는 사진·조명·부품에 달렸다. 위 표는 MVTec LOCO 에서의 측정이지 약속이 아니다.
- 학습 시간: 품종당 최대 2시간 (RTX 4090, both, 양품 300장 기준). 캐시 1~3 GB, 번들 0.1 GB.
- Windows 에서 폴더 가져오기는 심볼릭 링크를 쓰고, 안 되면 복사한다.
