"""Write review batch files (markdown) from data/cases/dev/view/review_view.csv for the LLM reviewer.

Usage: python make_batches.py <repo> <out_dir> <prefix> [--only-pending-ids FILE]
Only rows whose decision is pending are written. ROB-01 is grouped by original (4 subtypes together).
"""
import csv
import sys
from collections import OrderedDict
from pathlib import Path

repo, out_dir, prefix = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
view = repo / "data/cases/dev/view/review_view.csv"
rows = list(csv.DictReader(view.open(encoding="utf-8-sig", newline="")))
pending = [r for r in rows if r["decision"] == "pending"]
FENCE = "~~~~~"
for r in pending:
    for k in ("text", "original_text"):
        assert FENCE not in r[k], r["case_id"]

LABEL = {"1": "1 (인젝션)", "0": "0 (정상)"}


def block(title, text):
    return f"{title}\n{FENCE}\n{text}\n{FENCE}\n"


def chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


batches = []  # (name, header, [entry strings], [case_ids])
by_cond = OrderedDict((c, [r for r in pending if r["condition"] == c]) for c in ("ROB-04", "ROB-02", "ROB-03", "ROB-01"))

for cond, rs in by_cond.items():
    if not rs:
        continue
    if cond == "ROB-01":
        groups = OrderedDict()
        for r in rs:
            src = r["case_id"].rsplit("-", 1)[1]
            groups.setdefault(src, []).append(r)
        units = list(groups.items())
        for part in chunks(units, 25):
            entries, ids = [], []
            for src, grp in part:
                grp.sort(key=lambda r: ("case", "space", "char", "homoglyph").index(r["subtype"]))
                s = f"### 원문 validation:{src} — 레이블 {LABEL[grp[0]['label']]}\n\n" + block("원문:", grp[0]["original_text"])
                for r in grp:
                    eg = " (보류군)" if r["eval_group"] == "held_out" else ""
                    s += "\n" + block(f"- `{r['case_id']}` — 방식 `{r['subtype']}`{eg}:", r["text"])
                    ids.append(r["case_id"])
                entries.append(s)
            batches.append((cond, entries, ids))
    else:
        for part in chunks(rs, 50):
            entries, ids = [], []
            for r in part:
                head = f"### `{r['case_id']}` — {cond} `{r['subtype']}`, 제안 레이블 {LABEL[r['label']]}"
                if r["position"]:
                    head += f", 문맥 위치 `{r['position']}`"
                s = head + "\n\n"
                if r["original_text"]:
                    s += block("원문:", r["original_text"]) + "\n"
                s += block("사례:", r["text"])
                entries.append(s)
                ids.append(r["case_id"])
            batches.append((cond, entries, ids))

out_dir.mkdir(parents=True, exist_ok=True)
for i, (cond, entries, ids) in enumerate(batches, 1):
    name = f"{prefix}{i:02d}"
    head = (f"# 검수 묶음 {name} — {cond}, 사례 {len(ids)}건\n\n"
            f"결과 파일: `data/cases/dev/view/llm_review/out/{name}.json` (이 묶음의 사례 {len(ids)}건을 모두, 한 번씩)\n\n"
            "문장은 `~~~~~` 줄 사이에 있는 그대로다(여러 칸 띄어쓰기·줄바꿈 포함).\n\n")
    (out_dir / f"{name}.md").write_text(head + "\n".join(entries), encoding="utf-8", newline="\n")
    (out_dir / f"{name}.ids").write_text("\n".join(ids) + "\n", encoding="utf-8", newline="\n")
    print(name, cond, len(ids))
print("total", sum(len(b[2]) for b in batches))
