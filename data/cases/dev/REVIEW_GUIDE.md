# 개발용 사례 검수 안내 (3단계, 검수자 HANJUSON)

이 폴더의 사례는 README 5.2의 ROB-01~04 개발용 평가 사례다. 검수가 끝나기 전에는 모델 예측을 보지 않는다(검수가 예측에 끌려가지 않게 하기 위해서다).

## 파일

| 파일 | 내용 | 손으로 고치는가 |
| --- | --- | --- |
| `cases.csv` | 사례 전체(문장 포함). `status`가 `assigned`인 행이 검수 대상이고, `reserve`는 대체용 예비 사례다 | 아니오 |
| `review.csv` | 검수 기록. `case_id,text_sha256_12,decision,new_label,reason,reviewer` (`text_sha256_12`는 검수한 문장의 해시로, 문장이 바뀌면 그 결정은 거부된다. 고치지 않는다) | **예** (`decision`·`new_label`·`reason`·`reviewer`만) |
| `originals.csv` | 원문 배정(`validation_clean`에서 seed로 섞어 배정) | 아니오 |
| `cases_meta.json` | 생성 규칙·seed·입력 해시·자동 점검 결과 | 아니오 |
| `prompts/`, `llm/` | LLM에 준 프롬프트 전문과 LLM 출력 | 아니오 |

## 읽기용 보기 파일과 편집 방법

- 사례 문장과 원문을 나란히 보려면 `view/review_view.csv`를 연다(git에 올라가지 않는 읽기 전용 파일, 엑셀에서 바로 열리도록 UTF-8 BOM으로 저장). 열은 `case_id,condition,subtype,eval_group,label,position,decision,new_label,reason,text,original_text`이다. 처음에는 아래 명령으로 만들고, 결정을 저장한 뒤에는 아래 4번의 `--apply-review` 명령에 `--view data/cases/dev/view/review_view.csv`를 붙여 다시 만든다(결정을 저장한 상태에서 `--check`를 먼저 돌리면 `cases_meta.json differs`로 멈춘다. 그때는 `--apply-review`부터 실행한다).
  ```
  PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/build_dev_cases.py --check --view data/cases/dev/view/review_view.csv
  ```
  터미널에서 조건별 몇 건만 원문과 함께 보려면 `--check --print 5`를 쓴다.
- 원문을 찾으려고 `datasets/` 폴더를 열지 않는다(같은 폴더에 test.csv가 있다).
- **결정은 `review.csv`에만 적고, `review.csv`는 VS Code 같은 텍스트 편집기로 고친다.** 엑셀로 `review.csv`를 저장하면 `text_sha256_12`의 일부 값이 숫자로 바뀌고(예: `0e8975772492` → `0`) 한글 사유가 CP949로 저장돼, 스크립트가 해당 행을 거부한다(R09·R01). 보기 파일은 엑셀로 열어도 된다.

## 검수 방법

1. 보기 파일에서 문장과 원문을 보고, `review.csv`의 같은 `case_id` 행을 채운다.
2. `decision`
   - `keep`: 레이블(`cases.csv`의 `label`, 1 = 인젝션, 0 = 정상)이 맞고 사례로 쓸 수 있다.
   - `relabel`: 문장은 쓸 수 있지만 레이블이 바뀌었다. `new_label`에 0 또는 1을 적고 `reason`에 이유를 적는다.
   - `exclude`: 의미가 망가졌거나, 의도가 모호해 레이블을 정할 수 없거나, 사례로 부적절하다. `reason`을 적는다.
3. `reviewer`에 `HANJUSON`을 적는다. 받는 값은 `HANJUSON`과, HANJUSON의 지시(2026-11-16)로 검수한 별도 Claude Code 세션의 `llm:claude-opus-5-5:review_v1`(지시서 `prompts/review_v1.md`) 둘뿐이고 다른 아이디는 거부된다. `pending`으로 남은 행이 하나라도 있으면 평가 스크립트가 실행되지 않는다.
4. 저장한 뒤 저장소 루트에서 실행한다.
   ```
   PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/build_dev_cases.py --apply-review --view data/cases/dev/view/review_view.csv
   ```
   `exclude`한 원문은 같은 레이블의 다음 예비 원문으로 바뀌고(ROB-01은 네 방식 모두), 새 사례가 `review.csv`에 `pending`으로 추가된다. 그 행도 같은 방법으로 검수하고 다시 실행한다.
5. `pending`이 0개가 되면 `scripts/build_dev_cases.py --check`로 파일이 맞는지 확인한 뒤 평가한다(`scripts/eval_cases.py`).

## 조건별로 볼 것

| 조건 | 생성 | 주로 볼 것 |
| --- | --- | --- |
| ROB-01 `case`·`space`·`char` | 규칙 | 사람이 원문을 알아볼 수 있는가. 대부분 `keep`이다 |
| ROB-01 `homoglyph` (보류군) | 규칙 | 화면에서는 원문과 같아 보이는 것이 정상이다(라틴 문자를 모양이 같은 키릴 문자로 바꿈). 대부분 `keep`이다 |
| ROB-02 의역 | LLM | 원문(보기 파일의 `original_text`)과 같은 의도인가. 인젝션이 약해지거나 강해지지 않았는가, 정상 요청이 거절·경고로 바뀌지 않았는가 |
| ROB-03 문맥 | 템플릿 + LLM 문맥 문장 | 앞뒤 문맥이 원문의 의도나 레이블을 바꾸지 않는가 |
| ROB-04 정상 입력 | LLM | 정말 정상인가. 공격 문구는 인용·분석 대상으로만 나와야 하고, 모델에게 지시를 무시하거나 숨은 프롬프트를 보여 달라고 직접 요구하면 `exclude`(ROB-04는 정상 사례만 두므로 `relabel`은 받지 않는다) |

- 원문 자체의 레이블이 이상해 보여도 원본 데이터셋 레이블을 기준으로 둔다. 변형 때문에 레이블이 바뀐 경우만 `relabel`한다.
- 같은 판단 기준을 끝까지 유지하고, 애매한 경우는 `reason`에 기준을 짧게 적는다.

## 생성 후 점검에서 눈에 띈 사례 (판단은 검수자가 한다)

- `dev-rob02-001427`: 의역에 "system"이 더해져 공격이 조금 강해졌을 수 있다.
- `dev-rob02-000252`: 원문의 직접적인 "you cannot hide"가 빠져 공격이 약해졌을 수 있다.
- `dev-rob02-000305`: 시제가 바뀌었다("would you have revealed").
- `dev-rob03-001486`·`dev-rob03-001533`: 문맥 문장(107번)에 비ASCII 문자 `é`가 있다. train의 비ASCII 행은 모두 인젝션이라 모델이 이 문자에 반응할 수 있다(공격·정상 쌍이 같은 문맥을 쓰므로 비교는 짝지어 된다).

## 사례 검수가 끝난 뒤에 할 것

2단계 오류 108개의 유형 분류(EXP-02)는 사례 검수를 마친 뒤 2026-11-17에 검수했다(HANJUSON의 지시로 별도 Claude Code 세션, 사람 검수 아님). 분류 문서는 `results/s3/error_taxonomy.md`, 검수 기록은 `data/error_review/review.csv`, 집계는 `results/s3/error_summary.md`(`scripts/summarize_errors.py`)이다. 이 파일들은 2단계 모델이 틀린 문장을 보여 주고, 그중 일부는 개발용 사례의 원문이기도 하므로 앞으로 사례를 다시 검수할 때는 검수를 마친 뒤에 연다.
