"""Validate the LLM reviewer's batch outputs and merge them into data/cases/dev/review.csv.

Usage: python merge_review.py <repo> <round> [--write]
Without --write it only validates and prints the summary.
"""
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

repo, rnd = Path(sys.argv[1]), sys.argv[2]
write = "--write" in sys.argv
sys.path.insert(0, str(repo / "scripts"))
import build_dev_cases as B  # noqa: E402

REVIEWER = "llm:claude-opus-5-5:review_v1"
work = repo / "data/cases/dev/view/llm_review"
dev = repo / "data/cases/dev"
cases = {r["case_id"]: r for r in csv.DictReader(io.StringIO((dev / "cases.csv").read_text(encoding="utf-8"), newline=""))}
review_rows = list(csv.DictReader(io.StringIO((dev / "review.csv").read_text(encoding="utf-8"), newline="")))
by_id = {r["case_id"]: r for r in review_rows}

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
    if sorted(got) != sorted(ids) or len(got) != len(set(got)):
        errors.append(f"{name}: case ids differ from the batch (missing {sorted(set(ids) - set(got))[:5]}, "
                      f"extra {sorted(set(got) - set(ids))[:5]}, dup {len(got) - len(set(got))})")
    for it in items:
        cid = it.get("case_id")
        if set(it) != {"case_id", "decision", "new_label", "reason"}:
            errors.append(f"{cid}: keys {sorted(it)}")
            continue
        dec, nl, reason = it["decision"], it["new_label"], it["reason"]
        if cid not in cases or cid not in by_id:
            errors.append(f"{cid}: unknown case")
            continue
        if by_id[cid]["decision"] != "pending":
            errors.append(f"{cid}: review.csv row is not pending ({by_id[cid]['decision']})")
        if dec not in ("keep", "relabel", "exclude"):
            errors.append(f"{cid}: decision {dec!r}")
        if dec == "relabel":
            if cid.startswith("dev-rob04-"):
                errors.append(f"{cid}: ROB-04 relabel")
            if nl not in (0, 1) or str(nl) == cases[cid]["label"]:
                errors.append(f"{cid}: new_label {nl!r} (case label {cases[cid]['label']})")
        elif nl is not None:
            errors.append(f"{cid}: new_label {nl!r} without relabel")
        if not isinstance(reason, str) or "\n" in reason or "\r" in reason:
            errors.append(f"{cid}: reason not one line")
        elif dec in ("relabel", "exclude") and not reason.strip():
            errors.append(f"{cid}: {dec} without reason")
        decided[cid] = (dec, "" if nl is None else str(nl), (reason or "").strip())

pending_left = [cid for cid, r in by_id.items() if r["decision"] == "pending" and cid not in decided]
print(f"round {rnd}: batches {len(batches)}, decided {len(decided)}, pending rows not covered {len(pending_left)}")
cnt = Counter((cases[c]["condition"], cases[c]["subtype"], d[0]) for c, d in decided.items())
for k in sorted(cnt):
    print("  ", *k, cnt[k])
for c, (dec, nl, reason) in sorted(decided.items()):
    if dec != "keep" or reason:
        print(f"  {c} [{cases[c]['label']}] {dec}{' -> ' + nl if nl else ''}: {reason}")
if errors:
    print("ERRORS:")
    for e in errors:
        print("  ", e)
    sys.exit(1)
if write:
    for cid, (dec, nl, reason) in decided.items():
        by_id[cid].update(decision=dec, new_label=nl, reason=reason, reviewer=REVIEWER)
    (dev / "review.csv").write_bytes(B.render_csv(B.REVIEW_COLUMNS, review_rows))
    print("review.csv written")
