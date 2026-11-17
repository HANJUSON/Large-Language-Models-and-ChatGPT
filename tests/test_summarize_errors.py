"""Tests for scripts/summarize_errors.py on synthetic inputs (no dataset text, no models).

Run from Git Bash at the repo root:
    PYTHONUTF8=1 .venv/Scripts/python.exe -m unittest tests/test_summarize_errors.py -v
"""
import contextlib
import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_errors as S  # noqa: E402

LLM = "llm:claude-opus-5-5:exp02_review_v1"
# sid, label, type, model-error flags (bert, roberta, deberta), draft category, confidence
ROWS = [
    ("validation:000002", "1", "FN", (0, 0, 0), "FN-PROBE", "high"),
    ("validation:000004", "1", "FN", (1, 0, 0), "FN-FRAME", "high"),
    ("validation:000047", "1", "FN", (1, 1, 1), "FN-REPACK", "medium"),
    ("validation:000263", "1", "FN", (1, 1, 1), "FN-SECDOC", "high"),
    ("validation:000285", "0", "FP", (0, 1, 1), "FP-SECMETA", "high"),
    ("validation:000385", "0", "FP", (1, 0, 0), "FP-TECH", "medium"),
]


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


class SummarizeErrorsTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        models = ("bert", "roberta", "deberta")
        header = (["sentence_id", "label", "error_type"] + [f"{m}_seeds_wrong" for m in models]
                  + [f"{m}_model_error" for m in models] + ["n_models_error", "shared_by"])
        write(self.root / S.ERRORS_REL, header,
              [[s, l, t] + [str(2 * f) for f in fl] + [str(f) for f in fl] + [str(sum(fl)), "x"]
               for s, l, t, fl, _, _ in ROWS])
        write(self.root / S.DRAFT_REL, S.DRAFT_COLUMNS,
              [[s, t, c, conf, "pending", ""] for s, _, t, _, c, conf in ROWS])
        self.review = {s: {"sentence_id": s, "error_type": t, "draft_category": c, "decision": "keep",
                           "final_category": "", "reason": "", "label_note": "", "reviewer": LLM}
                       for s, _, t, _, c, _ in ROWS}
        self.write_review()

    def write_review(self):
        write(self.root / S.REVIEW_REL, S.REVIEW_COLUMNS,
              [[r[c] for c in S.REVIEW_COLUMNS] for _, r in sorted(self.review.items())])

    def run_main(self, *extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = S.main(["--root", str(self.root), *extra])
        return code, buf.getvalue()

    def assert_refused(self, code_prefix):
        code, out = self.run_main()
        self.assertEqual(code, 1, out)
        self.assertIn(code_prefix, out)
        for n in S.OUT_FILES:
            self.assertFalse((self.root / S.OUT_REL / n).exists(), n)

    def test_counts_changes_and_notes(self):
        self.review["validation:000047"].update(decision="change", final_category="FN-PROBE", reason="청중 지정만 있음")
        self.review["validation:000285"]["label_note"] = "레이블 경계"
        self.review["validation:000385"]["reviewer"] = "hanjuson"
        self.write_review()
        code, out = self.run_main()
        self.assertEqual(code, 0, out)
        doc = json.loads((self.root / S.OUT_REL / "error_summary.json").read_text(encoding="utf-8"))
        fn, fp = doc["counts"]["FN"], doc["counts"]["FP"]
        self.assertEqual(fn["FN-PROBE"], {"bert": 1, "roberta": 1, "deberta": 1, "any_model": 1, "all_models": 1, "any_run": 2})
        self.assertEqual(fn["FN-REPACK"]["any_run"], 0)
        self.assertEqual(fn["total"], {"bert": 3, "roberta": 2, "deberta": 2, "any_model": 3, "all_models": 2, "any_run": 4})
        self.assertEqual(fp["total"], {"bert": 1, "roberta": 1, "deberta": 1, "any_model": 2, "all_models": 0, "any_run": 2})
        self.assertEqual(doc["decisions"], {"keep": 5, "change": 1})
        self.assertEqual(doc["changes"], [["validation:000047", "FN-REPACK", "FN-PROBE"]])
        self.assertEqual(doc["label_notes"], ["validation:000285"])
        self.assertEqual(doc["reviewers"], ["HANJUSON", LLM])  # canonical spelling
        types = (self.root / S.OUT_REL / "error_types.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(types[0], ",".join(S.OUT_COLUMNS))
        self.assertIn("validation:000047,FN,FN-REPACK,medium,change,FN-PROBE," + LLM + ",0", types)
        self.assertIn("validation:000285,FP,FP-SECMETA,high,keep,FP-SECMETA," + LLM + ",1", types)
        md = (self.root / S.OUT_REL / "error_summary.md").read_text(encoding="utf-8")
        self.assertIn("| validation:000047 | `FN-REPACK` | `FN-PROBE` |", md)
        self.assertNotIn("청중", md + "".join(types))  # reasons stay in the review record

    def test_check_detects_drift(self):
        self.assertEqual(self.run_main()[0], 0)
        self.assertEqual(self.run_main("--check")[0], 0)
        p = self.root / S.OUT_REL / "error_summary.md"
        p.write_bytes(p.read_bytes() + b"x")
        code, out = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertIn("C01", out)

    def test_e01_missing_header_order(self):
        (self.root / S.REVIEW_REL).unlink()
        self.assert_refused("E01")
        self.write_review()
        text = (self.root / S.DRAFT_REL).read_text(encoding="utf-8").replace("confidence", "conf", 1)
        (self.root / S.DRAFT_REL).write_text(text, encoding="utf-8", newline="\n")
        self.assert_refused("E01")

    def test_e01_unsorted_ids(self):
        rows = sorted(self.review.items(), reverse=True)
        write(self.root / S.REVIEW_REL, S.REVIEW_COLUMNS, [[r[c] for c in S.REVIEW_COLUMNS] for _, r in rows])
        self.assert_refused("E01")

    def test_e02_ids_differ(self):
        del self.review["validation:000002"]
        self.write_review()
        self.assert_refused("E02")

    def test_e03_type_and_label(self):
        self.review["validation:000002"]["error_type"] = "FP"
        self.write_review()
        self.assert_refused("E03")

    def test_e04_draft_code(self):
        text = (self.root / S.DRAFT_REL).read_text(encoding="utf-8").replace("FP-TECH", "FN-PROBE")
        (self.root / S.DRAFT_REL).write_text(text, encoding="utf-8", newline="\n")
        self.review["validation:000385"]["draft_category"] = "FN-PROBE"
        self.write_review()
        self.assert_refused("E04")

    def test_e04_draft_record_unchanged(self):
        for old, new in ((",high,pending,\n", ",high,keep,\n"), (",high,pending,\n", ",high,pending,FN-OTHER\n")):
            with self.subTest(new=new):
                self.setUp()
                p = self.root / S.DRAFT_REL
                text = p.read_text(encoding="utf-8")
                self.assertIn(old, text)
                p.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
                self.assert_refused("E04")

    def test_e05_decisions(self):
        cases = [
            {"decision": "pending"},
            {"decision": "keep", "final_category": "FN-OTHER"},
            {"decision": "change", "final_category": "FN-PROBE", "reason": "같은 코드"},
            {"decision": "change", "final_category": "FP-OTHER", "reason": "다른 종류"},
            {"decision": "change", "final_category": "", "reason": "코드 없음"},
            {"draft_category": "FN-OTHER"},
        ]
        base = dict(self.review["validation:000002"])
        for upd in cases:
            with self.subTest(upd=upd):
                self.review["validation:000002"] = {**base, **upd}
                self.write_review()
                self.assert_refused("E05")

    def test_e06_reviewer(self):
        for who in ("", "claude", "Claude", "llm:claude-opus-5-5:exp02_review_v2", "llm:claude-opus-5-5:review_v1"):
            with self.subTest(who=who):
                self.review["validation:000004"]["reviewer"] = who
                self.write_review()
                self.assert_refused("E06")

    def test_e07_notes(self):
        base = dict(self.review["validation:000004"])
        for upd in ({"reason": "두 줄\n사유"}, {"label_note": "가" * 121},
                    {"decision": "change", "final_category": "FN-PROBE", "reason": " "}):
            with self.subTest(upd=upd):
                self.review["validation:000004"] = {**base, **upd}
                self.write_review()
                self.assert_refused("E07")


if __name__ == "__main__":
    unittest.main()
