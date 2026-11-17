"""Build EXP-02 review batches (validation text only) for the separate Claude Code review session.

Writes data/error_review/view/in/e1-01.md (FN) and e1-02.md (FP) plus .ids files.
No model names, seed counts or per-model error flags are put in the batch.
"""
import csv
import os
import re
import sys

ROOT = sys.argv[1]
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import data_lists  # noqa: E402

data_lists.install_test_guard()
rows = data_lists.load_rows()["validation"]

draft = list(csv.DictReader(open(os.path.join(ROOT, "results/s3/error_types_draft.csv"), encoding="utf-8", newline="")))
errors = {r["sentence_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "results/s2/errors.csv"), encoding="utf-8", newline=""))}
assert len(draft) == 108 and set(errors) == {r["sentence_id"] for r in draft}

md = open(os.path.join(ROOT, "results/s3/error_taxonomy_draft.md"), encoding="utf-8").read()
alt = {}
for m in re.finditer(r"^\s*\| (validation:\d{6}) \| ([A-Z-]+) \| ([A-Z-]+|—) \| (.+?) \|$", md, re.M):
    alt[m.group(1)] = (m.group(3), m.group(4))
alt.setdefault("validation:000607", ("—", "의미가 불명확함"))
alt.setdefault("validation:001274", ("FN-PROBE", "거절을 \"배운\" 행동으로 전제하는 질문"))

# taxonomy: sections 2-4 only (no per-model counts)
s2 = md.index("## 2. 배정 절차")
s5 = md.index("## 5. 범주별 집계")
taxonomy = md[s2:s5].rstrip() + "\n"

out_dir = os.path.join(ROOT, "data/error_review/view/in")
os.makedirs(out_dir, exist_ok=True)
for n, et in ((1, "FN"), (2, "FP")):
    items = [r for r in draft if r["error_type"] == et]
    for r in items:
        assert errors[r["sentence_id"]]["error_type"] == et
        lab, _ = rows[r["sentence_id"]]
        assert lab == ("1" if et == "FN" else "0")
    name = f"e1-0{n}"
    lines = [f"# {name}: {et} {len(items)}건 (데이터셋 레이블 {'1 = 인젝션' if et == 'FN' else '0 = 정상'})", "",
             "## 분류 체계 (초안 그대로 고정, 범주 코드는 이 중에서만 고른다)", "", taxonomy, "## 사례", ""]
    for i, r in enumerate(items, 1):
        sid = r["sentence_id"]
        text = rows[sid][1]
        assert "~~~~~" not in text
        lines.append(f"### {i}. {sid}")
        lines.append(f"- 초안 범주: `{r['category']}` (confidence {r['confidence']})")
        if sid in alt:
            a, why = alt[sid]
            lines.append(f"- 초안 작성자가 적은 대안: `{a}` — {why}")
        lines += ["- 문장:", "~~~~~", text, "~~~~~", ""]
    with open(os.path.join(out_dir, name + ".md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_dir, name + ".ids"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(r["sentence_id"] for r in items) + "\n")
    print(name, et, len(items), "alts", sum(1 for r in items if r["sentence_id"] in alt))
