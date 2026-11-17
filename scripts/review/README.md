# 검수 세션 보조 스크립트 (기록용)

3단계에서 별도 Claude Code 세션에 검수를 맡길 때 쓴 스크립트다(README 5.1·5.2). 실행 당시에는 저장소 밖 임시 폴더에 있었고, 2026-11-17에 내용 그대로 옮겼다. 단위 테스트는 없다.

| 파일 | 쓰임 |
| --- | --- |
| `dev_make_batches.py <저장소> <출력 폴더> <접두사>` | `data/cases/dev/view/review_view.csv`의 검수 대기 사례를 조건별 묶음 파일(`.md`, `.ids`)로 쓴다. 실제 실행: 출력 `data/cases/dev/view/llm_review/in`, 접두사 `r1-` |
| `dev_launch_review.ps1 -Round r1` | 검수 세션 실행(허용·금지 도구 포함). 새 창은 `wt.exe -w new -d <저장소> powershell.exe -NoExit -ExecutionPolicy Bypass -File <이 파일> -Round r1`로 열었다 |
| `dev_merge_review.py <저장소> r1 [--write]` | 검수 세션 출력(`out/r1-NN.json`)의 형식을 검사하고 `data/cases/dev/review.csv`에 옮긴다 |
| `exp02_make_batches.py <저장소>` | 2단계 오류 108개의 문장(validation)·초안 범주·분류 규칙을 묶음 파일(`data/error_review/view/in/e1-0N.md`)로 쓴다. 당시 입력은 `results/s3/error_taxonomy_draft.md`였고 지금은 `error_taxonomy.md`로 바뀌어, 그대로는 다시 실행되지 않는다 |
| `exp02_launch_review.ps1 -Round e1` | EXP-02 검수 세션 실행 |
| `exp02_merge_review.py <저장소>` | 검수 세션 출력을 검사하고 `data/error_review/review.csv`를 쓴다 |

묶음 파일과 검수 세션 출력은 문장을 담고 있어 git에서 제외한다(`view/`).

## 4단계 증강 사례 검수 (2026-11-17)

| 파일 | 쓰임 |
| --- | --- |
| `aug_make_batches.py <저장소> a1-` | `data/cases/train/cases.csv`·`review.csv`에서 검수 대기인 배정 사례를 유형별 묶음 파일(`data/cases/train/view/llm_review/in/a1-NN.md`, `.ids`)로 쓴다. train 원문은 `data_lists`로 읽는다 |
| `aug_launch_review.ps1 -Round a1` | 증강 사례 검수 세션 실행(지시서 `data/cases/train/prompts/aug_review_v1.md`, 허용·금지 도구 포함). 새 창은 3단계와 같은 `wt.exe` 명령으로 열었다 |
| `aug_merge_review.py <저장소> a1 [--write]` | 검수 세션 출력(`out/a1-NN.json`)의 형식을 검사하고 `data/cases/train/review.csv`에 옮긴다. 그 뒤 `scripts/build_train_aug.py`를 실행한다 |
