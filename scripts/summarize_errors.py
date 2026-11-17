"""EXP-02 error-type summary: reviewed categories of the stage-2 errors -> results/s3/ (README 5.1).

Run from Git Bash at the repo root after the review record is complete:
    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/summarize_errors.py
Options: --check rebuilds in memory and compares with the files already written
(exit 1 on any difference, nothing written); --root DIR / --out DIR for tests.

Rules:
  - Inputs (sentence IDs and codes only, no sentence text is read):
      results/s2/errors.csv           stage-2 errors (scripts/summarize_s2.py): error type,
                                      per-model "model error" flags (wrong in >= 2 of 3 seeds)
      results/s3/error_types_draft.csv draft category and confidence per error (2026-11-16)
      data/error_review/review.csv    review record: decision keep|change, final category,
                                      one-line reason and label note, reviewer
  - Refused (exit 1, nothing written; codes E01-E07) when:
    E01 a file is missing, its header differs, an ID repeats or IDs are not ascending;
    E02 the three files do not hold the same sentence IDs;
    E03 error_type differs between files, or label and error_type disagree (FN = label 1);
    E04 a draft category is not a code of its error type, confidence is not high|medium|low,
        or the draft's reviewer_decision/final_category are not "pending"/empty (the draft file
        stays the unchanged draft record; decisions live in the review record);
    E05 a decision is not keep|change, keep has a final category, change has none, the same
        code as the draft or a code of the other error type, or the review's draft category
        differs from the draft file;
    E06 the reviewer is not one of REVIEWERS (case-insensitive; empty = still pending); outputs
        use the REVIEWERS spelling;
    E07 a reason or label note has a line break or more than 120 characters, or change has
        no reason.
  - Outputs (LF, deterministic): results/s3/error_types.csv (ID, error type, draft and final
    category, confidence, decision, reviewer, label-note flag), error_summary.json (counts and
    input SHA-256), error_summary.md (tables). Counts per final category: model errors of each
    model, any model, all models, and (reference) wrong in any run.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_REL = "scripts/summarize_errors.py"
ERRORS_REL = "results/s2/errors.csv"
DRAFT_REL = "results/s3/error_types_draft.csv"
REVIEW_REL = "data/error_review/review.csv"
OUT_REL = "results/s3"
OUT_FILES = ("error_types.csv", "error_summary.json", "error_summary.md")

FN_CODES = ("FN-SECDOC", "FN-EMBED", "FN-CLAIM", "FN-FRAME", "FN-REPACK", "FN-PERSUADE", "FN-PROBE", "FN-OTHER")
FP_CODES = ("FP-INTENT", "FP-SECMETA", "FP-PSYCH", "FP-TECH", "FP-OTHER")
CODES = {"FN": FN_CODES, "FP": FP_CODES}
NAMES = {
    "FN-SECDOC": "보안 문서 위장 추출", "FN-EMBED": "정상 과제에 끼운 명령·코드",
    "FN-CLAIM": "권한·상태 허위 주장", "FN-FRAME": "상상·가정·역할 프레이밍",
    "FN-REPACK": "형식·문서 재포장 추출", "FN-PERSUADE": "설득·감정 압박",
    "FN-PROBE": "프레이밍 없는 자기 설정 탐색", "FN-OTHER": "기타·불명확",
    "FP-INTENT": "의도 판별 질문", "FP-SECMETA": "AI 보안·처리 방식 메타 질문",
    "FP-PSYCH": "심리·대화 기법 질문", "FP-TECH": "소프트웨어·IT 운영 문장", "FP-OTHER": "기타·불명확",
}
CONFIDENCE = ("high", "medium", "low")
DECISIONS = ("keep", "change")
# README 5.1 (EXP-02 result) and 9: HANJUSON, or the separate Claude Code review session HANJUSON ordered (2026-11-17).
REVIEWERS = ("HANJUSON", "llm:claude-opus-5-5:exp02_review_v1")
MAX_NOTE = 120
DRAFT_COLUMNS = ("sentence_id", "error_type", "category", "confidence", "reviewer_decision", "final_category")
REVIEW_COLUMNS = ("sentence_id", "error_type", "draft_category", "decision", "final_category",
                  "reason", "label_note", "reviewer")
OUT_COLUMNS = ("sentence_id", "error_type", "draft_category", "confidence", "decision", "category",
               "reviewer", "label_note")
DISPLAY = {"bert": "BERT", "roberta": "RoBERTa", "deberta": "DeBERTa-v3"}


def sha256_lf(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_csv(path, columns, name, errors, prefix_ok=False):
    """Rows of a CSV whose header equals columns (or starts with them when prefix_ok)."""
    if not Path(path).is_file():
        errors.append(f"E01 {name}: file missing")
        return None, None
    text = Path(path).read_bytes().decode("utf-8")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        errors.append(f"E01 {name}: empty")
        return None, None
    header = tuple(rows[0])
    if (header[:len(columns)] if prefix_ok else header) != tuple(columns):
        errors.append(f"E01 {name}: header differs")
        return None, None
    out, ids = {}, []
    for i, r in enumerate(rows[1:], 2):
        if len(r) != len(header):
            errors.append(f"E01 {name}: line {i} has {len(r)} fields, expected {len(header)}")
            return None, None
        rec = dict(zip(header, r))
        sid = rec["sentence_id"]
        if sid in out:
            errors.append(f"E01 {name}: {sid} repeated")
            return None, None
        if ids and sid <= ids[-1]:
            errors.append(f"E01 {name}: IDs not ascending at {sid}")
            return None, None
        ids.append(sid)
        out[sid] = rec
    return out, header


def model_names(header):
    return [c[:-len("_model_error")] for c in header if c.endswith("_model_error")]


def summarize(root):
    """(errors, result) from the three input files under root."""
    root = Path(root)
    errors = []
    err_cols = ("sentence_id", "label", "error_type")
    E, eh = read_csv(root / ERRORS_REL, err_cols, "errors.csv", errors, prefix_ok=True)
    D, _ = read_csv(root / DRAFT_REL, DRAFT_COLUMNS, "error_types_draft.csv", errors)
    R, _ = read_csv(root / REVIEW_REL, REVIEW_COLUMNS, "review.csv", errors)
    if errors:
        return errors, None
    models = model_names(eh)
    need = {f"{m}_model_error" for m in models} | {f"{m}_seeds_wrong" for m in models} | {"n_models_error"}
    if not models or not need <= set(eh):
        return ["E01 errors.csv: header differs (no per-model columns)"], None
    if not (set(E) == set(D) == set(R)):
        a, b, c = set(E), set(D), set(R)
        return [f"E02 IDs differ: errors.csv {len(a)}, draft {len(b)}, review {len(c)}, "
                f"only in some: {sorted((a | b | c) - (a & b & c))[:5]}"], None
    out = []
    for sid in sorted(E):
        e, d, r = E[sid], D[sid], R[sid]
        et = e["error_type"]
        if et not in CODES or d["error_type"] != et or r["error_type"] != et:
            errors.append(f"E03 {sid}: error_type differs between files")
            continue
        if e["label"] != ("1" if et == "FN" else "0"):
            errors.append(f"E03 {sid}: label {e['label']!r} does not match {et}")
            continue
        if (d["category"] not in CODES[et] or d["confidence"] not in CONFIDENCE
                or d["reviewer_decision"] != "pending" or d["final_category"]):
            errors.append(f"E04 {sid}: draft category {d['category']!r} / confidence {d['confidence']!r}"
                          f" / reviewer_decision {d['reviewer_decision']!r} / final_category {d['final_category']!r}")
            continue
        dec, fin = r["decision"], r["final_category"]
        if r["draft_category"] != d["category"]:
            errors.append(f"E05 {sid}: review draft_category {r['draft_category']!r} differs from draft {d['category']!r}")
            continue
        if dec not in DECISIONS:
            errors.append(f"E05 {sid}: decision {dec!r} is not keep|change")
            continue
        if dec == "keep" and fin:
            errors.append(f"E05 {sid}: keep with final_category {fin!r}")
            continue
        if dec == "change" and (fin not in CODES[et] or fin == d["category"]):
            errors.append(f"E05 {sid}: change needs a different {et} code, got {fin!r}")
            continue
        reviewer = {x.lower(): x for x in REVIEWERS}.get(r["reviewer"].lower())
        if reviewer is None:
            errors.append(f"E06 {sid}: reviewer {r['reviewer']!r} not accepted (expected one of {', '.join(REVIEWERS)})")
            continue
        bad = [k for k in ("reason", "label_note") if "\n" in r[k] or "\r" in r[k] or len(r[k]) > MAX_NOTE]
        if bad or (dec == "change" and not r["reason"].strip()):
            errors.append(f"E07 {sid}: {', '.join(bad) or 'reason'} must be one line of at most {MAX_NOTE} characters"
                          + ("" if bad else " and is required for change"))
            continue
        flags = {m: e[f"{m}_model_error"] == "1" for m in models}
        out.append({
            "sentence_id": sid, "error_type": et, "draft_category": d["category"],
            "confidence": d["confidence"], "decision": dec,
            "category": fin if dec == "change" else d["category"], "reviewer": reviewer,
            "label_note": "1" if r["label_note"].strip() else "0",
            "models": flags, "n_models": sum(flags.values()),
        })
    if errors:
        return errors, None
    inputs = {rel: sha256_lf(root / rel) for rel in (ERRORS_REL, DRAFT_REL, REVIEW_REL)}
    return [], {"rows": out, "models": models, "inputs": inputs, "counts": count(out, models)}


def count(rows, models):
    res = {}
    for et in ("FN", "FP"):
        cats = {}
        for c in CODES[et] + ("total",):
            sel = [r for r in rows if r["error_type"] == et and (c == "total" or r["category"] == c)]
            cats[c] = {
                **{m: sum(r["models"][m] for r in sel) for m in models},
                "any_model": sum(r["n_models"] >= 1 for r in sel),
                "all_models": sum(r["n_models"] == len(models) for r in sel),
                "any_run": len(sel),
            }
        res[et] = cats
    res["decisions"] = {d: sum(r["decision"] == d for r in rows) for d in DECISIONS}
    res["changes"] = [[r["sentence_id"], r["draft_category"], r["category"]] for r in rows if r["decision"] == "change"]
    res["label_notes"] = [r["sentence_id"] for r in rows if r["label_note"] == "1"]
    res["reviewers"] = sorted({r["reviewer"] for r in rows})
    return res


def types_csv(res):
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(OUT_COLUMNS)
    for r in res["rows"]:
        w.writerow([r[c] for c in OUT_COLUMNS])
    return buf.getvalue()


def summary_json(res, script_sha):
    doc = {"script": {"path": SCRIPT_REL, "sha256_lf": script_sha}, "inputs": res["inputs"],
           "models": res["models"], "reviewers": res["counts"]["reviewers"],
           "decisions": res["counts"]["decisions"], "changes": res["counts"]["changes"],
           "label_notes": res["counts"]["label_notes"],
           "counts": {et: res["counts"][et] for et in ("FN", "FP")}}
    return json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=False) + "\n"


def summary_md(res):
    C, models = res["counts"], res["models"]
    heads = [DISPLAY.get(m, m) for m in models]
    n = len(res["rows"])
    lines = [
        "# EXP-02 오류 유형별 집계 (2단계 오류, `scripts/summarize_errors.py`)", "",
        f"- 대상: `{ERRORS_REL}`의 {n}행(`validation_clean`에서 9개 실행 중 한 번 이상 틀린 문장).",
        f"- 범주: 초안(`{DRAFT_REL}`, 정의·규칙은 `results/s3/error_taxonomy.md`)에 검수 기록(`{REVIEW_REL}`)을 적용한 확정값.",
        f"- 검수자: {', '.join(f'`{x}`' for x in C['reviewers'])}. 결정: keep {C['decisions']['keep']}, change {C['decisions']['change']}.",
        "- 모델별 열은 그 모델의 모델 오류(seed 3개 중 2개 이상에서 틀림) 수, \"세 모델 공통\"은 세 모델 모두의 모델 오류,"
        " 마지막 열은 참고용으로 한 실행 이상에서 틀린 문장 수다.", "",
    ]
    for et, title in (("FN", "FN (공격을 정상으로 판단)"), ("FP", "FP (정상 입력을 공격으로 판단)")):
        lines += [f"## {title}", "",
                  "| 범주 | 이름 | " + " | ".join(heads) + " | 한 모델 이상 | 세 모델 공통 | 참고: 한 실행 이상 |",
                  "|---|---|" + "---:|" * (len(models) + 3)]
        for c in CODES[et] + ("total",):
            v = C[et][c]
            cells = [str(v[m]) for m in models] + [str(v["any_model"]), str(v["all_models"]), str(v["any_run"])]
            if c == "total":
                lines.append(f"| **{et} 합계** | | " + " | ".join(f"**{x}**" for x in cells) + " |")
            else:
                lines.append(f"| `{c}` | {NAMES[c]} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## 검수에서 바뀐 범주", ""]
    if C["changes"]:
        lines += ["| sentence_id | 초안 | 확정 |", "|---|---|---|"]
        lines += [f"| {s} | `{a}` | `{b}` |" for s, a, b in C["changes"]]
    else:
        lines.append("없음.")
    lines += ["", "## 레이블 메모가 있는 문장 (레이블은 바꾸지 않음, 메모는 검수 기록에 있음)", "",
              ", ".join(C["label_notes"]) if C["label_notes"] else "없음.", ""]
    return "\n".join(lines)


def render(res, script_sha):
    return {"error_types.csv": types_csv(res), "error_summary.json": summary_json(res, script_sha),
            "error_summary.md": summary_md(res)}


def parse_args(argv):
    p = argparse.ArgumentParser(description="Summarise reviewed EXP-02 error types into results/s3/ (README 5.1).")
    p.add_argument("--root", help="repository root (default: this repository)")
    p.add_argument("--out", help=f"output folder (default: <root>/{OUT_REL})")
    p.add_argument("--check", action="store_true", help="compare with the written files instead of writing")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root) if args.root else ROOT
    out = Path(args.out) if args.out else root / OUT_REL
    errors, res = summarize(root)
    if errors:
        for e in errors:
            print(("FAIL " + e).encode("ascii", "backslashreplace").decode("ascii"))
        print(f"refused: {len(errors)} error(s); nothing written")
        return 1
    files = render(res, sha256_lf(ROOT / SCRIPT_REL))
    if args.check:
        diff = [n for n, t in files.items() if not (out / n).is_file() or (out / n).read_bytes() != t.encode("utf-8")]
        if diff:
            print(f"FAIL C01 differs from a fresh summary: {', '.join(diff)}")
            return 1
        print(f"check OK: {', '.join(files)} match")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for n, t in files.items():
        (out / n).write_bytes(t.encode("utf-8"))
    C = res["counts"]
    print(f"OK {len(res['rows'])} errors; keep {C['decisions']['keep']}, change {C['decisions']['change']}; "
          f"wrote {', '.join(files)}")
    for et in ("FN", "FP"):
        top = sorted(CODES[et], key=lambda c: -C[et][c]["any_model"])[:3]
        print(f"{et} any-model {C[et]['total']['any_model']}: " + ", ".join(f"{c} {C[et][c]['any_model']}" for c in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
