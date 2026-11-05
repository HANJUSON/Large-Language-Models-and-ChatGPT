# 정상/비정상 Prompt 판별 챌린지 구현 계획서

## 0. 프로젝트 개요

- **과제**: 정상 prompt와 비정상 prompt(프롬프트 인젝션·탈옥) 판별
- **데이터셋**: [S-Labs/prompt-injection-dataset](https://huggingface.co/datasets/S-Labs/prompt-injection-dataset) (영어, 약 1.5만 건, 이진 분류)
- **제안 방법**
  1. **3단 캐스케이드 판별기**: 빠른 필터 → 파인튜닝 모델 → LLM 심판 순으로, 애매한 입력만 비싼 단계로 넘김
  2. **레드팀–블루팀 자가 강화 루프**: 변형 공격을 생성해 모델을 공격하고, 놓친 샘플로 재학습
  3. **한국어 확장**: 한국어 테스트셋을 구축해 언어 간 일반화를 측정·개선
- **일정 가정**: 7주 (10/29 ~ 12/16), 발표 12월 셋째 주

### 단계별 모델

| 단계 | 모델 | 비고 |
|---|---|---|
| 1단 | TF-IDF(char n-gram 2~5) + 로지스틱 회귀 | 학습 불필요 수준, 기준선 |
| 2단 (메인) | `microsoft/mdeberta-v3-base` | 다국어, fp16 사용 시 NaN 주의 |
| 2단 (비교) | `meta-llama/Llama-Prompt-Guard-2-86M`, `FacebookAI/xlm-roberta-base` | Prompt Guard 2는 라이선스 동의 필요 |
| 3단 | LLM 판별 프롬프트 (API 또는 Qwen 계열 소형 Instruct 로컬) | 파인튜닝 없음 |

---

## 1. 역할 분담

| 구분 | 팀원 A: 모델 담당 | 팀원 B: 데이터·공격·평가 담당 |
|---|---|---|
| 핵심 책임 | 판별기 학습과 캐스케이드 통합 | 데이터 품질, 공격 생성, 성능 측정 |
| 담당 단계 | 1단 TF-IDF, 2단 파인튜닝, 3단 LLM 심판, 임계값 조정 | 정규화, 한국어 테스트셋, 레드팀 생성기, 평가 스크립트, Gradio 데모 |
| 주요 산출물 | `stage1.py`, `stage2_train.py`, `stage3_judge.py`, `cascade.py` | `normalize.py`, `split.py`, `translate_ko.py`, `redteam.py`, `evaluate.py`, `app.py` |

B가 만든 평가 도구로 A의 모델을 검증하는 구조로, 서로의 결과물을 교차 점검한다.

---

## 2. 공통 약속 (인터페이스)

### 데이터 파일 형식 (모든 CSV 공통)

| 컬럼 | 값 |
|---|---|
| `text` | 프롬프트 문자열 |
| `label` | `0` = 정상, `1` = 비정상 |
| `lang` | `en` / `ko` |
| `variant` | `none`(원본), `homoglyph`, `spacing`, `base64`, `roleplay`, `translate` |

### 모델 함수 형식 (모든 단계 공통)

```python
def predict(texts: list[str]) -> list[float]:
    """각 문장의 비정상 확률(0~1)을 반환"""
```

이 형식만 맞추면 `evaluate.py`가 어떤 모델이든 같은 방식으로 채점할 수 있다.

### 저장소 구조

```
prompt-guard-project/
├── data/              # train.csv, val.csv, test.csv, test_ko.csv, redteam_r1.csv ...
├── src/
│   ├── normalize.py     (B)
│   ├── split.py         (B)
│   ├── translate_ko.py  (B)
│   ├── redteam.py       (B)
│   ├── evaluate.py      (B)
│   ├── stage1.py        (A)
│   ├── stage2_train.py  (A)
│   ├── stage3_judge.py  (A)
│   └── cascade.py       (A)
├── app.py             (B)
└── results/           # 실험별 지표 JSON, 그래프
```

### 작업 규칙

- 각자 `feature/이름-작업명` 브랜치에서 작업 후 PR, 상대가 리뷰 후 병합
- `test.csv`, `test_ko.csv`는 Week 6 전까지 아무도 열지 않음 (평가 오염 방지)

---

## 3. 주차별 계획

### Week 1 (10/29 ~ 11/4) — 환경 구축·기준선

- **공통**: GitHub 저장소 생성, Colab/GPU 환경 확인, Prompt Guard 2 라이선스 신청
- **B**: `split.py`로 70/15/15 분할(레이블 비율 유지), `normalize.py`의 `normalize_text(s)` 구현 (NFKC + 동형 문자 치환표)
- **A**: `stage1.py`에 TF-IDF(char n-gram) + 로지스틱 회귀 기준선 구현
- **완료 기준**: 검증셋에서 기준선 F1 산출

### Week 2 (11/5 ~ 11/11) — 메인 모델·평가 도구

- **A**: `stage2_train.py`로 mDeBERTa-v3-base 1차 파인튜닝 (lr 2e-5, batch 16, 3 epochs, max_length 256)
- **B**: `evaluate.py` 완성 (F1, AUROC, FPR, 변형별 탐지율을 `results/실험명.json`으로 저장), `translate_ko.py`로 `test_ko.csv` 500건 생성 후 100건 직접 검수
- **완료 기준**: 기준선 vs mDeBERTa 비교표

### Week 3 (11/12 ~ 11/18) — 모델 비교·레드팀 1라운드

- **A**: 같은 코드로 XLM-R-base, Prompt Guard 2 86M 학습·평가
- **B**: `redteam.py` 구현 → 검증셋 공격 문장을 5가지 방식으로 변형해 `redteam_r1.csv` 생성, 각 모델의 변형별 탐지율 측정
- **완료 기준**: 메인 모델 1개 확정, 변형 유형별 취약점 분석표

### Week 4 (11/19 ~ 11/25) — 견고성 강화·한국어 실험

- **B**: 레드팀 2·3라운드 진행, 놓친 샘플 모음 `hard_negatives.csv` 전달
- **A**: 놓친 샘플을 학습셋에 추가해 라운드마다 재학습
- **A**: 한국어 실험 A(영어만 학습 → 한국어 평가), 실험 B(번역 데이터 혼합 학습 → 한국어 평가)
- **완료 기준**: 라운드별 탐지율·오탐률 그래프, 한국어 실험 A/B 비교표

### Week 5 (11/26 ~ 12/2) — 3단·캐스케이드·데모

- **A**: `stage3_judge.py`(LLM 판별 프롬프트, 출력을 확률로 변환), `cascade.py`로 3단 연결
  - 1단 확률 ≥ 0.95 또는 ≤ 0.05 → 즉시 판정
  - 그 외 → 2단, 2단 확률 0.3 ~ 0.7 → 3단
  - 임계값은 검증셋에서 "F1 vs 3단 호출 비율" 곡선으로 결정
- **B**: `app.py`로 Gradio 데모 (입력 → 판정 결과, 판정 단계, 확신도, 정규화 전후 텍스트 표시)
- **완료 기준**: 데모에서 문장 입력 시 전 단계 동작

### Week 6 (12/3 ~ 12/9) — 최종 평가

- **공통**: `test.csv`, `test_ko.csv`로 최종 평가 1회 실시
- **A**: 단계별 처리 비율·평균 처리 시간 측정
- **B**: 오분류 사례 20건 정성 분석 (오류 유형 정리)
- **완료 기준**: 보고서용 표·그래프 확정

### Week 7 (12/10 ~ 12/16) — 보고서·발표

- **A**: 방법론 작성 (모델 구조, 학습 설정, 캐스케이드)
- **B**: 실험 결과·분석 작성 (레드팀, 한국어, 오분류)
- **공통**: 발표 자료 작성, 데모 리허설

---

## 4. 평가 지표

| 지표 | 목적 |
|---|---|
| F1, AUROC | 기본 판별 성능 |
| FPR (오탐률) | 정상 prompt를 잘못 차단하는 비율 |
| 변형 공격 탐지율 | 레드팀 루프 효과 |
| 한국어 F1 | 언어 간 일반화 |
| 건당 평균 처리 시간·3단 호출 비율 | 캐스케이드 효율 |

---

## 5. 운영 방식

- **정기 회의**: 주 1회 30분, 이번 주 결과 공유 → 다음 주 작업 확정
- **위험 관리**

| 위험 | 대응 |
|---|---|
| GPU 부족 | Prompt Guard 2 86M(가장 작은 모델) 중심으로 전환 |
| LLM API 비용 부담 | 3단을 Qwen 계열 소형 모델 로컬 실행으로 대체 |
| 일정 지연 | 한국어 실험 B, XLM-R 비교부터 축소 (캐스케이드·레드팀은 유지) |
