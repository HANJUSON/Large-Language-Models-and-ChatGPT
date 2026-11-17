"""CPU tests for scripts/select_thresholds.py on synthetic run folders.

No models, no GPU, no dataset text: for three seeds an A run (s2-roberta-A-s<seed>) and a C run
(s4-roberta-C-s<seed>) are written to a temp dir in train.py's formats. Every run has 122
validation rows: 20 label-1 and 100 label-0 rows in validation_clean plus one excluded row of
each label. Scores are float32(sigmoid(x)) for logits x placed between two candidates (x.x25),
so the expected (TP, FP) of every candidate and the chosen k are computed by hand in the
comments. Expected float32 values come from numpy; the script uses torch.

Run from Git Bash at the repo root:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m unittest tests/test_select_thresholds.py -v
"""
import contextlib
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import data_lists  # noqa: E402
import eval_lib  # noqa: E402
import select_thresholds as T  # noqa: E402
import train  # noqa: E402

SEEDS = [42, 123, 2026]
STEPS = [10, 20, 30]
N_POS, N_NEG = 20, 100
LABELS = [1] * (N_POS + 1) + [0] * (N_NEG + 1)
IN_CLEAN = [1] * N_POS + [0] + [1] * N_NEG + [0]  # rows 20 (label 1) and 121 (label 0) are excluded
N_ROWS = len(LABELS)
COL = {"step": 0, "sentence_id": 1, "label": 2, "in_clean": 3, "margin": 4, "score": 5, "prediction": 6,
       "threshold": 7, "condition": 8}
FILES = ("thresholds.csv", "thresholds.json")
KB = 13  # a candidate whose 9-digit score text is below the float32 threshold as a float64 (see TestGrid)


def f32(x):
    return float(np.float32(x))


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def S(x):
    """float32 score of a logit."""
    return f32(sig(x))


def thr(k):
    return f32(sig(k / 20))


ALWAYS, NEVER = S(9.5), S(-9.5)  # above the largest / below the smallest candidate threshold
EXCL_POS, EXCL_NEG = S(-3.025), S(2.225)  # excluded rows: would be an FN / an FP at 0.5 if they were counted
DECOY = 0.75  # every row at the steps that are not selected (all predicted 1)


def clean(pos, neg):
    """Scores and labels of the validation_clean rows: the given ones, the rest ALWAYS (label 1) / NEVER (0)."""
    return (list(pos) + [ALWAYS] * (N_POS - len(pos)) + list(neg) + [NEVER] * (N_NEG - len(neg)),
            [1] * N_POS + [0] * N_NEG)


def layout(pos, neg):
    """Scores of all 122 rows in file order."""
    s, _ = clean(pos, neg)
    return s[:N_POS] + [EXCL_POS] + s[N_POS:] + [EXCL_NEG]


# (special label-1 scores, special label-0 scores, selected step) per run; hand-computed outcomes below.
SCEN = {
    # At 0.5: TP 18, FP 6 -> cap 6/100 + 1/100 = 7/100 (FP <= 7).
    # k <= -13: TP 20 FP 8 | k -12..-11: 19, 8 | k -10..-9: 19, 7 | k -8..-7: 18, 7 | k -6..60: 18, 6 | k >= 61: 18, 0
    # -> highest recall within the cap is TP 19 at k -10..-9, lower middle k = -10.
    (42, "A"): ([S(-0.425), S(-0.625)], [S(3.025)] * 6 + [S(-0.325), S(-0.525)], 20),
    # At 0.5: TP 18, FP 2 (own cap would be 3/100; the A cap 7/100 applies).
    # k <= -51: 20, 8 | k -50..-41: 19, 8 | k -40..-31: 19, 7 | k -30..-21: 18, 7 | k -20..60: 18, 2 | k >= 61: 18, 0
    # -> k -40..-31, index (120 + 129) // 2 = 124 -> k = -36.
    (42, "C"): ([S(-1.525), S(-2.525)], [S(3.025)] * 2 + [S(-1.025)] * 5 + [S(-2.025)], 20),
    # At 0.5: TP 20, FP 1 -> cap 2/100 = 1/50 (FP <= 2).
    # k <= -41: 20, 3 | k -40..-21: 20, 2 | k -20..20: 20, 1 | k 21..40: 20, 0 | k >= 41: 19, 0
    # -> recall 20/20 for k -40..40, then lowest FPR: k 21..40, (21 + 40) // 2 = 30.
    (123, "A"): ([S(2.025)], [S(1.025), S(-1.025), S(-2.025)], 10),
    # At 0.5: TP 20, FP 4. Three label-0 rows are above every candidate: FP >= 3 > cap 1/50 -> cap not met.
    # k <= 10: 20, 4 | k 11..60: 20, 3 | k 61..100: 19, 3 | k >= 101: 18, 3
    # -> lowest FPR 3/100, then highest recall: k 11..60, (11 + 60) // 2 = 35.
    (123, "C"): ([S(3.025), S(5.025)], [ALWAYS] * 3 + [S(0.525)], 30),
    # At 0.5: TP 20, FP 0 -> cap 1/100 (FP <= 1).
    # k <= -21: 20, 2 | k -20..20: 20, 0 | k >= 21: 19, 0 -> k -20..20, middle k = 0, threshold exactly 0.5.
    (2026, "A"): ([S(1.025)], [S(-1.025)] * 2, 30),
    # One label-1 score equals the float32 threshold of k = 13: attack for k <= 13 (score >= threshold).
    # At 0.5: TP 20, FP 2. k <= 8: 20, 2 | k 9..13: 20, 0 | k >= 14: 19, 0 -> k 9..13, middle k = 11.
    (2026, "C"): ([thr(KB)], [S(KB / 20 - 0.225)] * 2, 10),
}
EXPECTED_CSV = """\
seed,condition,run_id,selected_step,n_clean,n_pos,n_neg,cap_exact,cap,cap_met,k,logit,threshold,k_lo,k_hi,\
tp,fp,fn,tn,recall,fpr,f1,base_tp,base_fp,base_fn,base_tn,base_recall,base_fpr,base_f1
42,B,s2-roberta-A-s42,20,120,20,100,7/100,0.070000,1,-10,-0.50,0.3775406777858734,-10,-9,\
19,7,1,93,0.950000,0.070000,0.826087,18,6,2,94,0.900000,0.060000,0.818182
42,D,s4-roberta-C-s42,20,120,20,100,7/100,0.070000,1,-36,-1.80,0.14185106754302979,-40,-31,\
19,7,1,93,0.950000,0.070000,0.826087,18,2,2,98,0.900000,0.020000,0.900000
123,B,s2-roberta-A-s123,10,120,20,100,1/50,0.020000,1,30,1.50,0.8175745010375977,21,40,\
20,0,0,100,1.000000,0.000000,1.000000,20,1,0,99,1.000000,0.010000,0.975610
123,D,s4-roberta-C-s123,30,120,20,100,1/50,0.020000,0,35,1.75,0.8519527912139893,11,60,\
20,3,0,97,1.000000,0.030000,0.930233,20,4,0,96,1.000000,0.040000,0.909091
2026,B,s2-roberta-A-s2026,30,120,20,100,1/100,0.010000,1,0,0.00,0.5,-20,20,\
20,0,0,100,1.000000,0.000000,1.000000,20,0,0,100,1.000000,0.000000,1.000000
2026,D,s4-roberta-C-s2026,10,120,20,100,1/100,0.010000,1,11,0.55,THR11,9,13,\
20,0,0,100,1.000000,0.000000,1.000000,20,2,0,98,1.000000,0.020000,0.952381
""".replace("THR11", repr(thr(11)))


def sid(i):
    return f"validation:{i:06d}"


def run_id(cond, seed):
    return f"{'s2' if cond == 'A' else 's4'}-roberta-{cond}-s{seed}"


def counts_at_half(scores, labels, in_clean):
    """validation_clean confusion counts at 0.5, counted row by row (not with the script)."""
    c = {"n": 0, "n_pos": 0, "n_neg": 0, "tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for s, y, keep in zip(scores, labels, in_clean):
        if keep:
            p = int(s >= 0.5)
            c["n"] += 1
            c["n_pos" if y else "n_neg"] += 1
            c[("tp" if p else "fn") if y else ("fp" if p else "tn")] += 1
    return c


def write_run(root, cond, seed, scores, sel, labels=LABELS, in_clean=IN_CLEAN):
    """One completed run in train.py's formats; the given scores at the selected step, DECOY elsewhere."""
    rid = run_id(cond, seed)
    d = root / "runs" / rid
    d.mkdir(parents=True, exist_ok=True)
    lines, evals = [train.PRED_HEADER], []
    for step in STEPS:
        ss = scores if step == sel else [DECOY] * len(labels)
        for i, s in enumerate(ss):
            lines.append(f"{step},{sid(i)},{labels[i]},{in_clean[i]},{eval_lib.fmt9(math.log(s / (1 - s)))},"
                         f"{eval_lib.fmt9(s)},{int(s >= 0.5)},0.5,{cond}")
        evals.append({"step": step, "validation_clean": counts_at_half(ss, labels, in_clean)})
    (d / "predictions_validation.csv").write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    train.write_json(d / "metrics.json", {"run_id": rid, "threshold": 0.5, "eval_steps": STEPS, "evals": evals,
                                          "selected_step": sel})
    train.write_json(d / "run.json", {
        "run_id": rid, "stage": rid[:2], "model_key": "roberta", "condition": cond, "seed": seed,
        "status": "completed", "budget": {"eval_steps": STEPS}, "selection": {"selected_step": sel},
        "checkpoint": {"kept": True, "step": sel, "sha256": hashlib.sha256(rid.encode("ascii")).hexdigest()},
        "data": {"validation_rows": len(labels), "validation_clean_rows": sum(in_clean)}})
    return rid


def make_root(base):
    root = Path(base) / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "train.json").write_bytes(json.dumps({"seeds": SEEDS, "threshold": 0.5}).encode("ascii"))
    for (seed, cond), (pos, neg, sel) in SCEN.items():
        write_run(root, cond, seed, layout(pos, neg), sel)
    return root


def run_main(root, *argv):
    """(exit code, stdout) of main; the process-wide test.csv guard is never installed by a test."""
    buf = io.StringIO()
    with mock.patch.object(data_lists, "install_test_guard"), contextlib.redirect_stdout(buf):
        code = T.main(["--root", str(root), *argv])
    return code, buf.getvalue()


def edit_json(path, fn):
    doc = json.loads(path.read_text(encoding="ascii"))
    fn(doc)
    train.write_json(path, doc)


def edit_pred(root, rid, fn):
    """fn(lines) edits the list of lines: lines[at(step, i)] is row i of that step, lines[0] the header."""
    p = root / "runs" / rid / "predictions_validation.csv"
    lines = p.read_bytes().decode("ascii").split("\n")
    fn(lines)
    p.write_bytes("\n".join(lines).encode("ascii"))


def at(step, i):
    return 1 + STEPS.index(step) * N_ROWS + i


def set_field(lines, at_step, i, **fields):
    f = lines[at(at_step, i)].split(",")
    for name, value in fields.items():
        f[COL[name]] = value
    lines[at(at_step, i)] = ",".join(f)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = make_root(self.tmp)
        self.out = self.tmp / "out"


# ---------------------------------------------------------------- candidates and the decision rule

class TestGrid(unittest.TestCase):
    def test_candidates_are_the_float32_sigmoid_of_k_over_20(self):
        cand = T.candidates()
        self.assertEqual([k for k, _ in cand], list(range(-160, 161)))
        self.assertEqual(len(cand), 321)
        self.assertEqual([t for _, t in cand], [thr(k) for k in range(-160, 161)])
        by = dict(cand)
        # float32 values of sigmoid(-8), sigmoid(-0.5), sigmoid(1), sigmoid(8), written out from numpy
        self.assertEqual(by[-160], 0.00033535013790242374)
        self.assertEqual(by[-10], 0.3775406777858734)
        self.assertEqual(by[20], 0.7310585975646973)
        self.assertEqual(by[160], 0.9996646642684937)
        self.assertAlmostEqual(by[20], 1 / (1 + 1 / math.e), delta=3e-8)  # logit 1.00
        self.assertTrue(all(b > a for a, b in zip([t for _, t in cand], [t for _, t in cand][1:])))

    def test_logit_zero_is_exactly_one_half(self):
        self.assertEqual(dict(T.candidates())[0], 0.5)
        self.assertEqual(repr(dict(T.candidates())[0]), "0.5")

    def test_every_threshold_is_a_float32_value(self):
        for k, t in T.candidates():
            self.assertEqual(f32(t), t, k)

    def test_boundary_candidate_of_the_fixture(self):
        # the property the float32-boundary tests rely on: the stored text of thr(KB) is a smaller float64
        self.assertLess(float(eval_lib.fmt9(thr(KB))), thr(KB))
        self.assertLess(thr(KB), sig(KB / 20))


class TestDecisionRule(unittest.TestCase):
    def test_score_equal_to_the_threshold_is_an_attack(self):
        # one label-1 row per candidate with score == threshold_k: at candidate index i the rows i..320 are >=
        scores = [thr(k) for k in range(-160, 161)] + [NEVER]
        labels = [1] * 321 + [0]
        self.assertEqual(T.candidate_counts(scores, labels), [(321 - i, 0) for i in range(321)])

    def test_next_float32_below_the_threshold_is_not_an_attack(self):
        below = [float(np.nextafter(np.float32(thr(k)), np.float32(0))) for k in range(-160, 161)]
        self.assertEqual(below[160], 0.4999999701976776)
        counts = T.candidate_counts(below + [ALWAYS], [0] * 321 + [1])
        self.assertEqual(counts, [(1, 320 - i) for i in range(321)])  # FP: only the rows above index i

    def test_score_is_rounded_to_float32_before_the_comparison(self):
        # 0.49999999 < 0.5 as a float64, but float32(0.49999999) is 0.5
        self.assertEqual(T.decide([0.49999999, 0.49999997, 0.5, 0.50000006], 0.5).tolist(), [1, 0, 1, 1])

    def test_same_decisions_as_numpy_float32(self):
        scores = [thr(k) for k in range(-160, 161, 7)] + [S(x / 8) for x in range(-70, 71)] + [0.0, 1.0]
        labels = [i % 2 for i in range(len(scores))]
        s32, y = np.array(scores, dtype=np.float32), np.array(labels)
        want = []
        for k in range(-160, 161):
            p = s32 >= np.float32(thr(k))
            want.append((int((p & (y == 1)).sum()), int((p & (y == 0)).sum())))
        self.assertEqual(T.candidate_counts(scores, labels), want)


# ---------------------------------------------------------------- cap and selection

class TestCap(unittest.TestCase):
    def test_cap_is_the_exact_fraction(self):
        self.assertEqual(T.fpr_cap(10, 1047), Fraction(2047, 104700))  # 10/1047 + 1/100
        self.assertEqual(T.fpr_cap(3, 1047), Fraction(449, 34900))  # (300 + 1047) / 104700, reduced by 3
        self.assertEqual(T.fpr_cap(6, 100), Fraction(7, 100))
        self.assertEqual(T.fpr_cap(0, 100), Fraction(1, 100))

    def test_fp_equal_to_the_cap_is_allowed_where_float_arithmetic_refuses_it(self):
        self.assertGreater(7 / 100, 6 / 100 + 0.01)  # in floats 7/100 would be over the cap
        pos, neg, _ = SCEN[(42, "A")]
        got = T.select_threshold(*clean(pos, neg), T.fpr_cap(6, 100))
        self.assertEqual((got["tp"], got["fp"], got["cap_met"]), (19, 7, True))
        self.assertEqual((got["k"], got["k_lo"], got["k_hi"], got["logit"]), (-10, -10, -9, -0.5))
        self.assertEqual(got["threshold"], 0.3775406777858734)

    def test_one_false_positive_over_the_cap_is_not_allowed(self):
        # cap just below 7/100: FP <= 6 -> TP 18 everywhere; lowest FPR is FP 0 at k 61..160, (61 + 160) // 2 = 110
        pos, neg, _ = SCEN[(42, "A")]
        got = T.select_threshold(*clean(pos, neg), Fraction(699, 10000))
        self.assertEqual((got["tp"], got["fp"], got["cap_met"]), (18, 0, True))
        self.assertEqual((got["k"], got["k_lo"], got["k_hi"]), (110, 61, 160))


class TestSelection(unittest.TestCase):
    def test_highest_recall_before_lowest_fpr(self):
        # (42, "A"): TP 19 with FP 7 is chosen over TP 18 with FP 6 or FP 0
        pos, neg, _ = SCEN[(42, "A")]
        got = T.select_threshold(*clean(pos, neg), Fraction(7, 100))
        self.assertEqual((got["tp"], got["fp"]), (19, 7))

    def test_lowest_fpr_among_the_candidates_with_the_highest_recall(self):
        # (123, "A"): TP 20 for k -40..40 with FP 2, 1, 0 -> the FP 0 run 21..40
        pos, neg, _ = SCEN[(123, "A")]
        got = T.select_threshold(*clean(pos, neg), Fraction(1, 50))
        self.assertEqual((got["tp"], got["fp"], got["cap_met"]), (20, 0, True))
        self.assertEqual((got["k"], got["k_lo"], got["k_hi"], got["logit"]), (30, 21, 40, 1.5))

    def test_middle_of_an_even_run_is_the_lower_one(self):
        pos, neg, _ = SCEN[(42, "A")]  # run -10..-9 (below zero): -10
        self.assertEqual(T.select_threshold(*clean(pos, neg), Fraction(7, 100))["k"], -10)
        pos, neg, _ = SCEN[(123, "A")]  # run 21..40 (above zero): 30, not 31
        self.assertEqual(T.select_threshold(*clean(pos, neg), Fraction(1, 50))["k"], 30)

    def test_middle_of_an_odd_run_and_logit_zero(self):
        pos, neg, _ = SCEN[(2026, "A")]
        got = T.select_threshold(*clean(pos, neg), Fraction(1, 100))
        self.assertEqual((got["k"], got["k_lo"], got["k_hi"], got["logit"]), (0, -20, 20, 0.0))
        self.assertEqual(repr(got["threshold"]), "0.5")
        self.assertEqual((got["tp"], got["fp"], got["cap_met"]), (20, 0, True))

    def test_cap_not_met_takes_lowest_fpr_then_highest_recall(self):
        pos, neg, _ = SCEN[(123, "C")]
        got = T.select_threshold(*clean(pos, neg), Fraction(1, 50))
        self.assertIs(got["cap_met"], False)
        self.assertEqual((got["tp"], got["fp"]), (20, 3))
        self.assertEqual((got["k"], got["k_lo"], got["k_hi"], got["logit"]), (35, 11, 60, 1.75))
        # with a cap of 3/100 the same candidates meet it
        self.assertIs(T.select_threshold(*clean(pos, neg), Fraction(3, 100))["cap_met"], True)

    def test_cap_not_met_prefers_lower_fpr_over_higher_recall(self):
        # three label-0 rows above every candidate, one at logit 4.025 (attack for k <= 80); one label-1 row at
        # logit 3.025 (attack for k <= 60): k <= 60: TP 20 FP 4 | k 61..80: 19, 4 | k >= 81: 19, 3
        # -> cap 1/50 never met; lowest FPR is FP 3 at k 81..160 although recall is lower there; (81 + 160) // 2
        got = T.select_threshold(*clean([S(3.025)], [ALWAYS] * 3 + [S(4.025)]), Fraction(1, 50))
        self.assertIs(got["cap_met"], False)
        self.assertEqual((got["tp"], got["fp"]), (19, 3))
        self.assertEqual((got["k"], got["k_lo"], got["k_hi"], got["logit"]), (120, 81, 160, 6.0))

    def test_float32_boundary_row_extends_the_run(self):
        pos, neg, _ = SCEN[(2026, "C")]
        got = T.select_threshold(*clean(pos, neg), Fraction(1, 100))
        self.assertEqual((got["k"], got["k_lo"], got["k_hi"]), (11, 9, KB))
        self.assertEqual(got["threshold"], thr(11))

    def test_one_label_only_is_refused(self):
        for labels in ([1, 1], [0, 0], []):
            with self.assertRaises(ValueError):
                T.select_threshold([0.25] * len(labels), labels, Fraction(1, 100))

    def test_metrics_at_a_threshold(self):
        pos, neg, _ = SCEN[(42, "A")]
        m = T.metrics_at(*clean(pos, neg), thr(-10))
        self.assertEqual([m[k] for k in ("tp", "fp", "fn", "tn")], [19, 7, 1, 93])
        self.assertEqual((m["recall_exact"], m["fpr_exact"], m["f1_exact"]), ("19/20", "7/100", "19/23"))


# ---------------------------------------------------------------- end to end

class TestEndToEnd(Base):
    def test_csv_hand_computed(self):
        code, out = run_main(self.root, "--out", str(self.out))
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), sorted(FILES))
        self.assertEqual((self.out / "thresholds.csv").read_bytes().decode("ascii"), EXPECTED_CSV)

    def test_default_output_folder(self):
        code, out = run_main(self.root)
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(p.name for p in (self.root / "results" / "s4").iterdir()), sorted(FILES))
        self.assertIn("wrote results/s4/: thresholds.csv, thresholds.json", out)

    def test_json_records(self):
        run_main(self.root, "--out", str(self.out))
        doc = json.loads((self.out / "thresholds.json").read_bytes())
        self.assertEqual((doc["model"], doc["seeds"], doc["default_threshold"]), ("roberta", SEEDS, 0.5))
        self.assertEqual(doc["candidates"], {"k_min": -160, "k_max": 160, "k_per_logit": 20, "count": 321})
        rows = {(r["seed"], r["condition"]): r for r in doc["thresholds"]}
        self.assertEqual(list(rows), [(s, c) for s in SEEDS for c in ("B", "D")])
        b, d = rows[(42, "B")], rows[(42, "D")]
        self.assertEqual((b["run_id"], b["source_condition"], b["selected_step"]), ("s2-roberta-A-s42", "A", 20))
        self.assertEqual((d["run_id"], d["source_condition"], d["selected_step"]), ("s4-roberta-C-s42", "C", 20))
        self.assertEqual(b["checkpoint_sha256"], hashlib.sha256(b"s2-roberta-A-s42").hexdigest())
        self.assertEqual(d["checkpoint_sha256"], hashlib.sha256(b"s4-roberta-C-s42").hexdigest())
        # the cap of D comes from the A run of the same seed (FP 6 of 100), not from the C run (FP 2)
        cap = {"exact": "7/100", "value": 0.07, "margin": "1/100", "a_run_id": "s2-roberta-A-s42", "a_fp": 6,
               "a_n_neg": 100}
        self.assertEqual((b["cap"], d["cap"]), (cap, cap))
        self.assertEqual((b["k"], b["logit"], b["threshold"], b["k_lo"], b["k_hi"]), (-10, -0.5, thr(-10), -10, -9))
        self.assertEqual((d["k"], d["logit"], d["threshold"], d["k_lo"], d["k_hi"]), (-36, -1.8, thr(-36), -40, -31))
        self.assertEqual(b["chosen"], {"tp": 19, "fp": 7, "fn": 1, "tn": 93, "recall": 0.95, "recall_exact": "19/20",
                                       "fpr": 0.07, "fpr_exact": "7/100", "f1": 19 / 23, "f1_exact": "19/23",
                                       "precision": 19 / 26, "precision_exact": "19/26"})
        self.assertEqual(b["base"], {"tp": 18, "fp": 6, "fn": 2, "tn": 94, "recall": 0.9, "recall_exact": "9/10",
                                     "fpr": 0.06, "fpr_exact": "3/50", "f1": 9 / 11, "f1_exact": "9/11",
                                     "precision": 0.75, "precision_exact": "3/4"})
        self.assertEqual({k: d["base"][k] for k in ("tp", "fp", "fn", "tn")}, {"tp": 18, "fp": 2, "fn": 2, "tn": 98})
        self.assertEqual([rows[(s, c)]["cap_met"] for s in SEEDS for c in ("B", "D")],
                         [True, True, True, False, True, True])
        self.assertEqual(rows[(123, "D")]["cap"]["exact"], "1/50")
        self.assertEqual((rows[(2026, "B")]["k"], rows[(2026, "B")]["threshold"]), (0, 0.5))
        for r in doc["thresholds"]:
            self.assertEqual((r["n_clean"], r["n_pos"], r["n_neg"]), (120, 20, 100))
            self.assertEqual(r["logit"], r["k"] / 20)
            self.assertEqual(f32(r["threshold"]), r["threshold"])
        self.assertEqual([(r["run_id"], r["condition"], r["seed"], r["validation_rows"], r["validation_clean_rows"])
                          for r in doc["runs"]],
                         [(run_id(c, s), c, s, 122, 120) for s in SEEDS for c in ("A", "C")])
        self.assertEqual([r["selected_step"] for r in doc["runs"]], [20, 20, 10, 30, 30, 10])

    def test_json_and_csv_hold_the_same_threshold(self):
        run_main(self.root, "--out", str(self.out))
        doc = json.loads((self.out / "thresholds.json").read_bytes())
        lines = (self.out / "thresholds.csv").read_text(encoding="ascii").split("\n")
        col = lines[0].split(",").index("threshold")
        self.assertEqual([float(ln.split(",")[col]) for ln in lines[1:-1]], [r["threshold"] for r in doc["thresholds"]])

    def test_input_and_script_hashes(self):
        run_main(self.root, "--out", str(self.out))
        doc = json.loads((self.out / "thresholds.json").read_bytes())
        want = {"configs/train.json": hashlib.sha256((self.root / "configs" / "train.json").read_bytes()).hexdigest()}
        for s in SEEDS:
            for c in ("A", "C"):
                for n in ("run.json", "metrics.json", "predictions_validation.csv"):
                    rel = f"runs/{run_id(c, s)}/{n}"
                    want[rel] = hashlib.sha256((self.root / rel).read_bytes()).hexdigest()
        self.assertEqual(doc["inputs_sha256"], want)
        script = (ROOT / "scripts" / "select_thresholds.py").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(doc["script_sha256_lf"], hashlib.sha256(script).hexdigest())

    def test_selected_step_and_in_clean_rows_only(self):
        # Counting the excluded rows would give n 122 and one more FP at 0.5; another step would give FP 100.
        run_main(self.root, "--out", str(self.out))
        doc = json.loads((self.out / "thresholds.json").read_bytes())
        self.assertEqual([(r["base"]["tp"], r["base"]["fp"], r["base"]["fn"], r["base"]["tn"])
                          for r in doc["thresholds"]],
                         [(18, 6, 2, 94), (18, 2, 2, 98), (20, 1, 0, 99), (20, 4, 0, 96), (20, 0, 0, 100),
                          (20, 2, 0, 98)])

    def test_deterministic_lf_no_ids_no_paths(self):
        run_main(self.root, "--out", str(self.out))
        other = make_root(self.tmp / "elsewhere")
        code, _ = run_main(other, "--out", str(self.tmp / "out2"))
        self.assertEqual(code, 0)
        for n in FILES:
            raw = (self.out / n).read_bytes()
            self.assertEqual(raw, (self.tmp / "out2" / n).read_bytes(), n)
            self.assertNotIn(b"\r", raw, n)
            self.assertTrue(raw.endswith(b"\n"), n)
            text = raw.decode("ascii")
            for bad in ("validation:", str(self.tmp), self.tmp.as_posix(), self.tmp.name, "\\"):
                self.assertNotIn(bad, text, n)
        self.assertFalse(list(self.out.glob("*.tmp")))

    def test_stdout_is_ascii_and_reports_every_row(self):
        code, out = run_main(self.root, "--out", str(self.out))
        self.assertEqual(code, 0)
        out.encode("ascii")
        self.assertIn("OK 6 runs checked", out)
        self.assertIn("seed 42 B (s2-roberta-A-s42 step 20): logit -0.50 (k -10 of -10..-9)", out)
        self.assertIn("cap 7/100 met; recall 9/10 -> 19/20, FPR 3/50 -> 7/100", out)
        self.assertIn("seed 123 D (s4-roberta-C-s123 step 30): logit 1.75 (k 35 of 11..60)", out)
        self.assertIn("cap 1/50 NOT MET", out)
        self.assertEqual(sum(ln.startswith("seed ") for ln in out.split("\n")), 6)

    def test_b_can_be_selected_from_the_a_run_alone_through_the_functions(self):
        # the dry-run path: load_run + threshold_row need no C run; the CLI does (TestRefusals.only_a_runs)
        for s in SEEDS:
            shutil.rmtree(self.root / "runs" / run_id("C", s))
        rec, errs = T.load_run(self.root, "s2", "A", 123)
        self.assertEqual(errs, [])
        row = T.threshold_row(123, "B", rec, rec)
        self.assertEqual((row["k"], row["k_lo"], row["k_hi"], row["cap"], row["cap_met"]),
                         (30, 21, 40, Fraction(1, 50), True))
        self.assertEqual(T.csv_row(row)["threshold"], "0.8175745010375977")


class TestCheck(Base):
    def setUp(self):
        super().setUp()
        self.assertEqual(run_main(self.root, "--out", str(self.out))[0], 0)
        self.before = {n: (self.out / n).read_bytes() for n in FILES}

    def test_check_passes_and_writes_nothing(self):
        stamp = {n: (self.out / n).stat().st_mtime_ns for n in FILES}
        code, out = run_main(self.root, "--out", str(self.out), "--check")
        self.assertEqual(code, 0, out)
        self.assertIn("OK --check: thresholds.csv, thresholds.json", out)
        self.assertEqual({n: (self.out / n).stat().st_mtime_ns for n in FILES}, stamp)

    def test_check_reports_a_changed_file(self):
        for n in FILES:
            changed = self.before[n].replace(b"-0.5", b"-0.4", 1)
            self.assertNotEqual(changed, self.before[n])
            (self.out / n).write_bytes(changed)
            code, out = run_main(self.root, "--out", str(self.out), "--check")
            self.assertEqual(code, 1)
            self.assertIn(f"FAIL T09 --check: {n} differs from the fresh selection", out)
            self.assertIn("refused: 1 error(s); nothing written", out)
            self.assertEqual((self.out / n).read_bytes(), changed)  # not repaired
            (self.out / n).write_bytes(self.before[n])

    def test_check_reports_a_missing_file_and_does_not_create_it(self):
        (self.out / "thresholds.json").unlink()
        code, out = run_main(self.root, "--out", str(self.out), "--check")
        self.assertEqual(code, 1)
        self.assertIn("FAIL T09 --check: thresholds.json is missing", out)
        self.assertFalse((self.out / "thresholds.json").exists())

    def test_check_sees_a_changed_input(self):
        pos, neg, sel = SCEN[(2026, "A")]
        write_run(self.root, "A", 2026, layout(pos + [S(0.525)], neg), sel)  # run -20..10 instead of -20..20
        code, out = run_main(self.root, "--out", str(self.out), "--check")
        self.assertEqual(code, 1)
        self.assertIn("T09 --check: thresholds.csv differs", out)
        self.assertIn("T09 --check: thresholds.json differs", out)

    def test_check_refuses_bad_input_before_comparing(self):
        shutil.rmtree(self.root / "runs" / run_id("C", 42))
        code, out = run_main(self.root, "--out", str(self.out), "--check")
        self.assertEqual(code, 1)
        self.assertIn("FAIL T01 s4-roberta-C-s42:", out)
        self.assertNotIn("T09", out)


# ---------------------------------------------------------------- refusals

A42, C42, A123, C123 = run_id("A", 42), run_id("C", 42), run_id("A", 123), run_id("C", 123)


def _run_json(rid, fn):
    return lambda root: edit_json(root / "runs" / rid / "run.json", fn)


def _metrics_json(rid, fn):
    return lambda root: edit_json(root / "runs" / rid / "metrics.json", fn)


def _pred(rid, fn):
    return lambda root: edit_pred(root, rid, fn)


def _config(doc):
    return lambda root: (root / "configs" / "train.json").write_bytes(json.dumps(doc).encode("ascii"))


def _only_a_runs(root):
    for s in SEEDS:
        shutil.rmtree(root / "runs" / run_id("C", s))


def _drop_field(lines):
    lines[at(20, 3)] = lines[at(20, 3)].rsplit(",", 1)[0]


def _swap_ids(lines):
    for step in STEPS:
        set_field(lines, step, 4, sentence_id=sid(5))
        set_field(lines, step, 5, sentence_id=sid(4))


def _crlf(root):
    p = root / "runs" / A42 / "predictions_validation.csv"
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))


def _other_rows(root):
    # consistent files, but another row is the excluded label-1 row (same counts)
    pos, neg, sel = SCEN[(42, "C")]
    write_run(root, "C", 42, layout(pos, neg), sel, in_clean=[0] + [1] * N_POS + [1] * N_NEG + [0])


def _no_label_0(root):
    pos, neg, sel = SCEN[(42, "A")]
    write_run(root, "A", 42, layout(pos, neg), sel, in_clean=[1] * N_POS + [0] * (N_NEG + 2))


def _two_bad_runs(root):
    shutil.rmtree(root / "runs" / C123)
    edit_json(root / "runs" / A42 / "run.json", lambda d: d.update(status="failed"))


def _fp(doc):
    doc["evals"][1]["validation_clean"]["fp"] += 1


LINE = at(20, 3) + 1  # 1-based file line of row 3 at step 20
# name: (mutation of a fresh root, texts that must be in the output, number of errors)
REFUSALS = {
    "only_a_runs": (_only_a_runs, ["FAIL T01 s4-roberta-C-s42: runs/s4-roberta-C-s42/ is missing",
                                   "FAIL T01 s4-roberta-C-s123:", "FAIL T01 s4-roberta-C-s2026:"], 3),
    "a_run_missing": (lambda r: shutil.rmtree(r / "runs" / A123),
                      ["FAIL T01 s2-roberta-A-s123: runs/s2-roberta-A-s123/ is missing"], 1),
    "lock": (lambda r: (r / "runs" / (C42 + ".lock")).write_bytes(b""),
             ["FAIL T01 s4-roberta-C-s42: runs/s4-roberta-C-s42.lock exists"], 1),
    "file_missing": (lambda r: (r / "runs" / A42 / "metrics.json").unlink(),
                     ["FAIL T01 s2-roberta-A-s42: missing metrics.json"], 1),
    "predictions_missing": (lambda r: (r / "runs" / C42 / "predictions_validation.csv").unlink(),
                            ["FAIL T01 s4-roberta-C-s42: missing predictions_validation.csv"], 1),
    "run_json_not_json": (lambda r: (r / "runs" / A42 / "run.json").write_bytes(b"{"),
                          ["FAIL T02 s2-roberta-A-s42: run.json or metrics.json is not valid JSON"], 1),
    "run_json_missing_key": (_run_json(A42, lambda d: d.pop("selection")),
                             ["FAIL T02 s2-roberta-A-s42: run.json or metrics.json: missing key or wrong type"], 1),
    "run_json_a_list": (lambda r: (r / "runs" / A42 / "run.json").write_bytes(b"[]"),
                        ["FAIL T02 s2-roberta-A-s42: run.json or metrics.json: missing key or wrong type"], 1),
    "identity_seed": (_run_json(A42, lambda d: d.update(seed=7)),
                      ["FAIL T02 s2-roberta-A-s42: run.json seed do not match the run ID"], 1),
    "identity_condition": (_run_json(C42, lambda d: d.update(condition="A", stage="s2")),
                           ["FAIL T02 s4-roberta-C-s42: run.json stage, condition do not match the run ID"], 1),
    "identity_model": (_run_json(A42, lambda d: d.update(model_key="bert")),
                       ["FAIL T02 s2-roberta-A-s42: run.json model_key do not match the run ID"], 1),
    "status_failed": (_run_json(C42, lambda d: d.update(status="failed")),
                      ["FAIL T03 s4-roberta-C-s42: status is 'failed', not 'completed'"], 1),
    "selected_step_not_an_eval_step": (
        _run_json(A42, lambda d: d["selection"].update(selected_step=15)),
        ["FAIL T04 s2-roberta-A-s42: selected step 15 is not one of the eval steps [10, 20, 30]"], 1),
    "checkpoint_of_another_step": (
        _run_json(A42, lambda d: d["checkpoint"].update(step=10)),
        ["FAIL T04 s2-roberta-A-s42: checkpoint not kept, SHA-256 not recorded, or its step 10 != selected step 20"],
        1),
    "checkpoint_not_kept": (_run_json(C42, lambda d: d["checkpoint"].update(kept=False)),
                            ["FAIL T04 s4-roberta-C-s42: checkpoint not kept"], 1),
    "checkpoint_sha_missing": (_run_json(C42, lambda d: d["checkpoint"].pop("sha256")),
                               ["FAIL T04 s4-roberta-C-s42: checkpoint not kept, SHA-256 not recorded"], 1),
    "crlf": (_crlf, ["FAIL T05 s2-roberta-A-s42: predictions_validation.csv: header or final LF differs"], 1),
    "no_final_lf": (_pred(A42, lambda ln: ln.pop()),
                    ["FAIL T05 s2-roberta-A-s42: predictions_validation.csv: header or final LF differs"], 1),
    "not_ascii": (lambda r: (r / "runs" / A42 / "predictions_validation.csv").write_bytes(b"\xff\n"),
                  ["FAIL T05 s2-roberta-A-s42: predictions_validation.csv: not ASCII"], 1),
    "line_malformed": (_pred(A42, _drop_field),
                       [f"FAIL T05 s2-roberta-A-s42: predictions_validation.csv: line {LINE} is malformed"], 1),
    "condition_column": (_pred(C42, lambda ln: set_field(ln, 20, 3, condition="A")),
                         [f"FAIL T05 s4-roberta-C-s42: predictions_validation.csv: line {LINE}:", "condition not C"],
                         1),
    "step_not_an_eval_step": (_pred(A42, lambda ln: set_field(ln, 20, 3, step="15")),
                              [f"FAIL T05 s2-roberta-A-s42: predictions_validation.csv: line {LINE}:"], 1),
    "label_not_0_1": (_pred(A42, lambda ln: [set_field(ln, s, 3, label="2") for s in STEPS]),
                      [f"FAIL T05 s2-roberta-A-s42: predictions_validation.csv: line {at(10, 3) + 1}:"], 1),
    "in_clean_not_0_1": (_pred(A42, lambda ln: [set_field(ln, s, 3, in_clean="2") for s in STEPS]),
                         [f"FAIL T05 s2-roberta-A-s42: predictions_validation.csv: line {at(10, 3) + 1}:"], 1),
    "prediction_not_0_1": (_pred(A42, lambda ln: set_field(ln, 20, 3, prediction="2")),
                           [f"FAIL T05 s2-roberta-A-s42: predictions_validation.csv: line {LINE}:"], 1),
    "score_over_1": (_pred(A42, lambda ln: set_field(ln, 20, 3, score="1.5")),
                     [f"FAIL T05 s2-roberta-A-s42: predictions_validation.csv: line {LINE}:",
                      "score not in [0, 1]"], 1),
    "score_nan": (_pred(A42, lambda ln: set_field(ln, 20, 3, score="nan")),
                  [f"FAIL T05 s2-roberta-A-s42: predictions_validation.csv: line {LINE}:"], 1),
    # 0.49999999 is below 0.5 as a float64 but float32(0.49999999) is 0.5: not a stored float32 score
    "score_not_a_float32_text": (
        _pred(A42, lambda ln: set_field(ln, 20, 30, score="0.49999999", prediction="0")),
        [f"FAIL T05 s2-roberta-A-s42: predictions_validation.csv: line {at(20, 30) + 1}: score is not the 9-digit "
         "text of a float32 value"], 1),
    "score_with_more_digits": (
        _pred(A42, lambda ln: set_field(ln, 10, 30, score="0.7500000000")),
        [f"FAIL T05 s2-roberta-A-s42: predictions_validation.csv: line {at(10, 30) + 1}: score is not the 9-digit "
         "text of a float32 value"], 1),
    "rows_differ_between_steps": (
        _pred(A42, lambda ln: set_field(ln, 20, 3, in_clean="0")),
        ["FAIL T05 s2-roberta-A-s42: predictions_validation.csv: step 20: sentence IDs, labels or in_clean differ "
         "from step 10"], 1),
    "ids_not_ascending": (
        _pred(A42, _swap_ids),
        ["FAIL T05 s2-roberta-A-s42: predictions_validation.csv: step 10: no rows, or sentence IDs not strictly "
         "ascending"], 1),
    "no_rows": (_pred(A42, lambda ln: ln.__delitem__(slice(1, len(ln) - 1))),
                ["FAIL T05 s2-roberta-A-s42: predictions_validation.csv: step 10: no rows"], 1),
    "row_count_differs_from_run_json": (
        _run_json(A42, lambda d: d["data"].update(validation_clean_rows=119)),
        ["FAIL T05 s2-roberta-A-s42: predictions_validation.csv: row counts differ from run.json data"], 1),
    "threshold_column": (_pred(C42, lambda ln: set_field(ln, 20, 3, threshold="0.4")),
                         [f"FAIL T06 s4-roberta-C-s42: predictions_validation.csv: line {LINE}: threshold is not "
                          "0.5"], 1),
    "stored_prediction_flipped": (
        _pred(C42, lambda ln: set_field(ln, 30, 3, prediction="0")),
        [f"FAIL T06 s4-roberta-C-s42: predictions_validation.csv: line {at(30, 3) + 1}: stored prediction != "
         "(float32(score) >= 0.5)"], 1),
    "metrics_selected_step": (
        _metrics_json(A42, lambda d: d.update(selected_step=30)),
        ["FAIL T07 s2-roberta-A-s42: metrics.json run_id, threshold, eval steps or selected step differ"], 1),
    "metrics_run_id": (_metrics_json(A42, lambda d: d.update(run_id=C42)),
                       ["FAIL T07 s2-roberta-A-s42: metrics.json run_id, threshold"], 1),
    "metrics_threshold": (_metrics_json(A42, lambda d: d.update(threshold=0.4)),
                          ["FAIL T07 s2-roberta-A-s42: metrics.json run_id, threshold"], 1),
    "metrics_eval_steps": (_metrics_json(A42, lambda d: d["evals"].pop()),
                           ["FAIL T07 s2-roberta-A-s42: metrics.json run_id, threshold, eval steps"], 1),
    "metrics_counts": (
        _metrics_json(A42, _fp),
        ["FAIL T07 s2-roberta-A-s42: the stored scores do not reproduce metrics.json validation_clean at step 20 "
         "(fp)"], 1),
    "rows_differ_between_runs": (
        _other_rows,
        ["FAIL T08 validation rows (IDs, labels, in_clean) differ from s2-roberta-A-s42: s4-roberta-C-s42"], 1),
    "no_label_0_row": (_no_label_0,
                       ["FAIL T08 s2-roberta-A-s42: validation_clean has 20 label-1 and 0 label-0 rows"], 1),
    "config_missing": (lambda r: (r / "configs" / "train.json").unlink(),
                       ["FAIL T00 configs/train.json: cannot read seeds and threshold"], 1),
    "config_no_seeds": (_config({"threshold": 0.5}), ["FAIL T00 configs/train.json: cannot read seeds"], 1),
    "config_seeds_repeat": (_config({"seeds": [42, 42, 123], "threshold": 0.5}),
                            ["FAIL T00 configs/train.json: seeds must be a non-empty list of distinct integers"], 1),
    "config_seeds_empty": (_config({"seeds": [], "threshold": 0.5}), ["FAIL T00 configs/train.json: seeds must"], 1),
    "config_seed_not_int": (_config({"seeds": [42, "123"], "threshold": 0.5}),
                            ["FAIL T00 configs/train.json: seeds must"], 1),
    "config_threshold": (_config({"seeds": SEEDS, "threshold": 0.4}),
                         ["FAIL T00 configs/train.json: threshold is 0.4, not 0.5"], 1),
    "all_errors_reported": (_two_bad_runs, ["FAIL T03 s2-roberta-A-s42:", "FAIL T01 s4-roberta-C-s123:"], 2),
}


class TestRefusals(Base):
    def test_refusals(self):
        for name, (mutate, texts, n_errors) in REFUSALS.items():
            with self.subTest(name):
                root = make_root(self.tmp / name)
                mutate(root)
                out_dir = self.tmp / name / "out"
                code, out = run_main(root, "--out", str(out_dir))
                self.assertEqual(code, 1, out)
                for t in texts:
                    self.assertIn(t, out)
                self.assertIn(f"refused: {n_errors} error(s); nothing written", out)
                self.assertEqual(out.count("FAIL "), n_errors, out)
                self.assertFalse(out_dir.exists())
                self.assertFalse((root / "results").exists())
                out.encode("ascii")

    def test_the_unmutated_root_is_accepted(self):
        self.assertEqual(run_main(self.root, "--out", str(self.out))[0], 0)

    def test_every_code_is_documented_and_tested(self):
        codes = [f"T{i:02d}" for i in range(10)]
        tested = " ".join(t for _, texts, _ in REFUSALS.values() for t in texts)
        for c in codes[:9]:
            self.assertIn(f"\n  {c} ", T.__doc__)
            self.assertIn(f"FAIL {c} ", tested)
        self.assertIn("\n  T09 --check", T.__doc__)


if __name__ == "__main__":
    unittest.main()
