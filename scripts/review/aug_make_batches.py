"""Write review batch files (markdown) for the training-augmentation cases (README 5.3).

Usage: python scripts/review/aug_make_batches.py <repo> <prefix>
Reads data/cases/train/cases.csv and review.csv; only assigned cases whose decision is pending are written.
Originals of train-derived cases are read from the train split through data_lists (never test).
Output: data/cases/train/view/llm_review/in/<prefix>NN.md and .ids (git-ignored).
"""
import csv
import io
import sys
from pathlib import Path

repo, prefix = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(repo / "scripts"))
import data_lists  # noqa: E402

data_lists.install_test_guard()
train = data_lists.load_rows()["train"]
d = repo / "data/cases/train"
cases = list(csv.DictReader(io.StringIO((d / "cases.csv").read_text(encoding="utf-8"), newline="")))
review = {r["case_id"]: r for r in csv.DictReader(io.StringIO((d / "review.csv").read_text(encoding="utf-8"), newline=""))}
todo = [c for c in cases if c["status"] == "assigned" and review.get(c["case_id"], {}).get("decision") == "pending"]
FENCE = "~~~~~"
LABEL = {"1": "1 (인젝션)", "0": "0 (정상)"}
SIZE = {"AUG-01": 100, "AUG-03": 50, "AUG-04": 84, "AUG-05": 84}


def block(title, text):
    assert FENCE not in text
    return f"{title}\n{FENCE}\n{text}\n{FENCE}\n"


batches = []
for cond in ("AUG-04", "AUG-05", "AUG-03", "AUG-01"):
    rs = [c for c in todo if c["condition"] == cond]
    for i in range(0, len(rs), SIZE[cond]):
        part, entries = rs[i:i + SIZE[cond]], []
        for c in part:
            head = f"### `{c['case_id']}` — {cond} `{c['subtype']}`, 제안 레이블 {LABEL[c['label']]}"
            if c["position"]:
                head += f", 문맥 위치 `{c['position']}`"
            s = head + "\n\n"
            if c["source_split"] == "train":
                s += block("원문:", train[c["source_id"]][1]) + "\n"
            s += block("사례:", c["text"])
            entries.append(s)
        batches.append((cond, entries, [c["case_id"] for c in part]))

out = d / "view/llm_review/in"
out.mkdir(parents=True, exist_ok=True)
for i, (cond, entries, ids) in enumerate(batches, 1):
    name = f"{prefix}{i:02d}"
    head = (f"# 검수 묶음 {name} — {cond}, 사례 {len(ids)}건\n\n"
            f"결과 파일: `data/cases/train/view/llm_review/out/{name}.json` (이 묶음의 사례 {len(ids)}건을 모두, 한 번씩)\n\n"
            "문장은 `~~~~~` 줄 사이에 있는 그대로다(여러 칸 띄어쓰기·줄바꿈 포함).\n\n")
    (out / f"{name}.md").write_text(head + "\n".join(entries), encoding="utf-8", newline="\n")
    (out / f"{name}.ids").write_text("\n".join(ids) + "\n", encoding="utf-8", newline="\n")
    print(name, cond, len(ids))
print("total", sum(len(b[2]) for b in batches))
