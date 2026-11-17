"""Validate the review session's batch outputs and merge them into data/cases/train/review.csv.

Usage: python scripts/review/aug_merge_review.py <repo> <round> [--write]
Without --write it only validates and prints the summary. Decisions: keep | exclude (README 5.3).
"""
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

repo, rnd = Path(sys.argv[1]), sys.argv[2]
write = "--write" in sys.argv
REVIEWER = "llm:claude-opus-5-5:aug_review_v1"
COLUMNS = ("case_id", "text_sha256_12", "decision", "reason", "reviewer")
d = repo / "data/cases/train"
work = d / "view/llm_review"
cases = {r["case_id"]: r for r in csv.DictReader(io.StringIO((d / "cases.csv").read_text(encoding="utf-8"), newline=""))}
raw = (d / "review.csv").read_text(encoding="utf-8")
rows = list(csv.DictReader(io.StringIO(raw, newline="")))
assert tuple(rows[0].keys()) == COLUMNS, list(rows[0].keys())
by_id = {r["case_id"]: r for r in rows}

errors, decided = [], {}
batches = sorted(p.stem for p in (work / "in").glob(f"{rnd}-*.md"))
for name in batches:
    ids = (work / "in" / f"{name}.ids").read_text(encoding="utf-8").split()
    out = work / "out" / f"{name}.json"
    if not out.exists():
        errors.append(f"{name}: output missing")
        continue
    items = json.loads(out.read_text(encoding="utf-8"))
    got = [it.get("case_id") for it in items]
    if got != ids:
        errors.append(f"{name}: case ids differ from the batch (missing {sorted(set(ids) - set(got))[:5]}, "
                      f"extra {sorted(set(got) - set(ids))[:5]}, order or duplicates otherwise)")
    for it in items:
        cid = it.get("case_id")
        if set(it) != {"case_id", "decision", "reason"}:
            errors.append(f"{cid}: keys {sorted(it)}")
            continue
        dec, reason = it["decision"], it["reason"]
        if cid not in cases or cid not in by_id:
            errors.append(f"{cid}: unknown case")
            continue
        if by_id[cid]["decision"] != "pending":
            errors.append(f"{cid}: review.csv row is not pending ({by_id[cid]['decision']})")
        if dec not in ("keep", "exclude"):
            errors.append(f"{cid}: decision {dec!r}")
        if not isinstance(reason, str) or "\n" in reason or "\r" in reason or len(reason) > 120:
            errors.append(f"{cid}: reason not one line of at most 120 characters")
        elif dec == "exclude" and not reason.strip():
            errors.append(f"{cid}: exclude without reason")
        decided[cid] = (dec, (reason or "").strip())

left = [cid for cid, r in by_id.items() if r["decision"] == "pending" and cid not in decided]
print(f"round {rnd}: batches {len(batches)}, decided {len(decided)}, pending rows not covered {len(left)}")
cnt = Counter((cases[c]["condition"], cases[c]["subtype"], cases[c]["label"], d_[0]) for c, d_ in decided.items())
for k in sorted(cnt):
    print("  ", *k, cnt[k])
if errors:
    print("ERRORS:")
    for e in errors[:40]:
        print("  ", e)
    sys.exit(1)
if write:
    for cid, (dec, reason) in decided.items():
        by_id[cid].update(decision=dec, reason=reason, reviewer=REVIEWER)
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    for r in rows:
        w.writerow([r[c] for c in COLUMNS])
    (d / "review.csv").write_bytes(buf.getvalue().encode("utf-8"))
    print("review.csv written")
