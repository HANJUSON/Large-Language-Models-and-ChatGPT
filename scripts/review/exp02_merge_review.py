"""Validate the EXP-02 review session output and write data/error_review/review.csv."""
import csv, json, os, sys
ROOT = sys.argv[1]
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import summarize_errors as S
LLM = "llm:claude-opus-5-5:exp02_review_v1"
V = os.path.join(ROOT, "data/error_review/view")
draft = {r["sentence_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, S.DRAFT_REL), encoding="utf-8", newline=""))}
rows, errs = {}, []
for b in ("e1-01", "e1-02"):
    ids = open(f"{V}/in/{b}.ids", encoding="utf-8").read().split()
    out = json.load(open(f"{V}/out/{b}.json", encoding="utf-8"))
    if [o.get("sentence_id") for o in out] != ids:
        errs.append(f"{b}: ids/order differ"); continue
    for o in out:
        sid = o["sentence_id"]
        if set(o) != {"sentence_id", "decision", "final_category", "reason", "label_note"}:
            errs.append(f"{sid}: keys {sorted(o)}")
        d = draft[sid]
        et = d["error_type"]
        dec, fin = o["decision"], o["final_category"]
        if dec == "keep" and fin is not None: errs.append(f"{sid}: keep with final {fin}")
        if dec == "change" and (fin not in S.CODES[et] or fin == d["category"]): errs.append(f"{sid}: bad change {fin}")
        if dec not in ("keep", "change"): errs.append(f"{sid}: decision {dec}")
        for k in ("reason", "label_note"):
            if not isinstance(o[k], str) or "\n" in o[k] or len(o[k]) > 120: errs.append(f"{sid}: {k} format")
        if dec == "change" and not o["reason"].strip(): errs.append(f"{sid}: change without reason")
        rows[sid] = [sid, et, d["category"], dec, fin or "", o["reason"], o["label_note"], LLM]
if set(rows) != set(draft): errs.append(f"coverage {len(rows)} vs {len(draft)}")
if errs:
    print("\n".join(errs)); sys.exit(1)
path = os.path.join(ROOT, S.REVIEW_REL)
with open(path, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(S.REVIEW_COLUMNS)
    for sid in sorted(rows):
        w.writerow(rows[sid])
print("wrote", path, len(rows))
