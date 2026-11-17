"""CPU tests for scripts/eval_cases.py on synthetic cases and tiny randomly initialised models.

No GPU (torch.cuda.is_available is patched to False), no pretrained weights or real checkpoints, no
dataset text: validation rows, cases, reviews, stage-2 run folders and results/s2 records are written to
temp dirs; models are tiny BERT configs saved with model_specs.save_checkpoint and loaded through the real
eval_lib.load_run_model; the tokenizer is a word-level stand-in. The test.csv guard is patched (it is a
process-wide audit hook).

Run from Git Bash at the repo root:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m unittest tests/test_eval_cases.py -v
"""
import contextlib
import csv
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, BertConfig  # noqa: E402

import build_lists  # noqa: E402
import data_lists  # noqa: E402
import eval_cases as E  # noqa: E402
import eval_lib  # noqa: E402
import model_specs  # noqa: E402
import summarize_s2  # noqa: E402
import train  # noqa: E402

MODELS = ["bert", "roberta", "deberta"]
SEEDS = [42, 123, 2026]
STEPS = [10, 20, 30]
SEL = 20
MARK = "ZQXTEXT"  # in every synthetic text; must never reach an output file
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo"]
OUT_FILES = ("predictions.csv", "metrics.csv", "models.csv", "changes.csv", "bootstrap.csv", "summary.json",
             "summary.md")


def sid(i):
    return f"validation:{i:06d}"


def tiny_config():
    return BertConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                      intermediate_size=64, max_position_embeddings=300, num_labels=2, pad_token_id=0,
                      initializer_range=0.5)


def tiny_model(seed):
    torch.manual_seed(seed)
    return AutoModelForSequenceClassification.from_config(tiny_config(), dtype=torch.float32,
                                                          attn_implementation="eager").eval()


class WordTok:
    """Word-level stand-in for the pinned tokenizers: [2] + word ids + [3], pad 0."""
    pad_token_id = 0

    def __call__(self, texts, add_special_tokens=True, truncation=False, max_length=None):
        out = []
        for t in texts:
            ids = [2] + [4 + sum(map(ord, w)) % 60 for w in t.split()] + [3]
            if truncation and len(ids) > max_length:
                ids = ids[:max_length - 1] + [3]
            out.append(ids)
        return {"input_ids": out}


def words(i, n):
    return " ".join(WORDS[(i * 7 + k * 3) % len(WORDS)] for k in range(n))


N_VAL = 30
VAL = {sid(i): ("1" if i % 2 == 0 else "0", f"{MARK} orig{i} {words(i, 4)}") for i in range(N_VAL)}


# ---------------------------------------------------------------- synthetic cases and reviews

def case_row(cid, cond, sub, text, *, src=None, label=None, position="", status="assigned", val=VAL):
    derived = cond != "ROB-04"
    return {"case_id": cid, "group_id": src if derived else cid, "source_split": "validation" if derived else "new",
            "source_id": src or "", "purpose": "dev", "condition": cond, "subtype": sub,
            "eval_group": "held_out" if sub == "homoglyph" else "main",
            "label": label if label is not None else (val[src][0] if derived else "0"), "position": position,
            "context_ids": "", "generator": "rule", "generator_version": "test-v1", "status": status,
            "auto_flags": "", "text": text}


def review_row(case, decision="keep", **kw):
    return {"case_id": case["case_id"], "text_sha256_12": build_lists.h12(case["text"]), "decision": decision,
            "new_label": "", "reason": "", "reviewer": "HANJUSON" if decision != "pending" else "", **kw}


def make_cases():
    """The state after build_dev_cases.py --apply-review: ROB-01 original 3 was excluded in review (its four
    cases are review_excluded) and replaced by original 17 (same label 0)."""
    rows = []
    for i in (0, 1, 2, 3, 17):  # ROB-01 originals (labels 1, 0, 1, 0, 0)
        o = VAL[sid(i)][1]
        for sub, t in (("case", o.upper()), ("space", o.replace(" ", "  ") + " x y"),
                       ("char", o.replace("a", "4") + " z"), ("homoglyph", o.replace("a", "\u0430") + " w")):
            rows.append(case_row(f"dev-rob01-{sub}-{i:06d}", "ROB-01", sub, t, src=sid(i),
                                 status="review_excluded" if i == 3 else "assigned"))
    for i in range(4, 8):
        rows.append(case_row(f"dev-rob02-{i:06d}", "ROB-02", "paraphrase", f"{MARK} para{i} {words(i + 1, 5)}",
                             src=sid(i)))
    short, long_ = words(1, 6), words(2, 45)
    for k, i in enumerate(range(8, 14)):
        o, pos = VAL[sid(i)][1], ("prefix", "suffix", "both")[k % 3]
        ctx = long_ if k == 0 else short
        text = {"prefix": f"{ctx}\n\n{o}", "suffix": f"{o}\n\n{ctx}", "both": f"{short}\n\n{o}\n\n{short}"}[pos]
        rows.append(case_row(f"dev-rob03-{i:06d}", "ROB-03", "context", text, src=sid(i), position=pos))
    for j in range(1, 4):
        rows.append(case_row(f"dev-rob04-{j:03d}", "ROB-04", "benign_security", f"{MARK} benign{j} {words(j, 6)}"))
    rows.append(case_row("dev-rob02-000014", "ROB-02", "paraphrase", f"{MARK} reserve", src=sid(14), status="reserve"))
    rows.append(case_row("dev-rob01-case-000015", "ROB-01", "case", f"{MARK} auto", src=sid(15),
                         status="auto_excluded"))
    return rows


def make_review(cases):
    rows = []
    for c in cases:
        if c["status"] not in ("assigned", "review_excluded"):
            continue
        r = review_row(c)
        if c["case_id"] == "dev-rob02-000005":
            r.update(decision="relabel", new_label="1", reason="intent is an attack")
        if c["case_id"] == "dev-rob01-char-000003":
            r.update(decision="exclude", reason="meaning changed")
        rows.append(r)
    return rows


def table_bytes(columns, rows):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def write_table(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(table_bytes(columns, rows))


def read_table(path):
    return list(csv.DictReader(io.StringIO(path.read_bytes().decode("utf-8"), newline="")))


# ---------------------------------------------------------------- synthetic stage-2 runs

def pred_lines(model):
    ids, _ = eval_lib.encode(WordTok(), [VAL[s][1] for s in sorted(VAL)], 256)
    m, s = eval_lib.predict(model, ids, 0, 64, "cpu")
    p = eval_lib.predictions(s, 0.5).tolist()
    lines = [train.PRED_HEADER]
    for step in STEPS:
        for k, x in enumerate(sorted(VAL)):
            lines.append(f"{step},{x},{VAL[x][0]},1,{eval_lib.fmt9(m[k].item())},{eval_lib.fmt9(s[k].item())},"
                         f"{p[k]},0.5,A")
    return ("\n".join(lines) + "\n").encode("ascii")


def write_run(root, model_key, seed, init_seed):
    rid = f"s2-{model_key}-A-s{seed}"
    d = root / "runs" / rid
    model = tiny_model(init_seed)
    info = model_specs.save_checkpoint(model, model_key, d / "checkpoint", rid, SEL)
    (d / "predictions_validation.csv").write_bytes(pred_lines(model))
    train.write_json(d / "metrics.json", {"run_id": rid, "selected_step": SEL})
    train.write_json(d / "run.json", {
        "run_id": rid, "model": {"key": model_key, "tokenizer_fingerprint": "fp"},
        "checkpoint": {"kept": True, "path": f"runs/{rid}/checkpoint/model.safetensors", "sha256": info["sha256"],
                       "size": info["size"], "step": SEL},
        "selection": {"selected_step": SEL}, "budget": {"eval_steps": STEPS}})
    return rid, info["sha256"]


def write_s2_records(root, pins):
    cfg_sha = data_lists.sha256_lf(root / E.CONFIG_REL)
    rows, inputs = [], {E.CONFIG_REL: cfg_sha}
    for (m, s), (rid, sha) in pins.items():
        rows.append({**dict.fromkeys(summarize_s2.RUN_COLUMNS), "model": m, "seed": s, "run_id": rid,
                     "selected_step": SEL, "checkpoint_sha256": sha})
        for n in E.RUN_FILES:
            inputs[f"runs/{rid}/{n}"] = hashlib.sha256((root / "runs" / rid / n).read_bytes()).hexdigest()
    (root / "results" / "s2").mkdir(parents=True, exist_ok=True)
    (root / "results" / "s2" / "runs.csv").write_bytes(
        summarize_s2.csv_text(summarize_s2.RUN_COLUMNS, rows).encode("ascii"))
    train.write_json(root / "results" / "s2" / "summary.json", {
        "inputs_sha256": inputs, "records": {"config_sha256": cfg_sha},
        "runs": [{"run_id": r["run_id"], "selected_step": SEL, "checkpoint_sha256": r["checkpoint_sha256"]}
                 for r in rows]})


def make_root(base):
    root = Path(base) / "repo"
    (root / "configs").mkdir(parents=True)
    shutil.copyfile(ROOT / "configs" / "train.json", root / "configs" / "train.json")
    pins = {}
    for mi, m in enumerate(MODELS):
        for s in SEEDS:
            pins[(m, s)] = write_run(root, m, s, 1000 * mi + s)
    write_s2_records(root, pins)
    cases = make_cases()
    write_table(root / "data" / "cases" / "dev" / "cases.csv", E.CASES_COLUMNS, cases)
    write_table(root / "data" / "cases" / "dev" / "review.csv", E.REVIEW_COLUMNS, make_review(cases))
    return root


TEMPLATE = {}


def setUpModule():
    TEMPLATE["dir"] = Path(tempfile.mkdtemp())
    TEMPLATE["root"] = make_root(TEMPLATE["dir"])


def tearDownModule():
    shutil.rmtree(TEMPLATE["dir"], ignore_errors=True)


def fresh_root(test):
    tmp = Path(tempfile.mkdtemp())
    test.addCleanup(shutil.rmtree, tmp, True)
    root = tmp / "repo"
    shutil.copytree(TEMPLATE["root"], root)
    return tmp, root.resolve()


def run_main(root, argv, calls=None, lib_root=None):
    """E.main with the CPU/tiny-model/test-data patches; returns (exit code, stdout)."""
    calls = calls if calls is not None else []
    buf = io.StringIO()
    with contextlib.ExitStack() as st:
        for p in (mock.patch.object(eval_lib, "ROOT", lib_root or root),
                  mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True),
                  mock.patch("transformers.AutoConfig.from_pretrained", side_effect=lambda *a, **k: tiny_config()),
                  mock.patch.object(model_specs, "load_tokenizer", side_effect=lambda key: WordTok()),
                  mock.patch.object(model_specs, "tokenizer_fingerprint", return_value="fp"),
                  mock.patch.object(data_lists, "install_test_guard", side_effect=lambda: calls.append("guard")),
                  mock.patch.object(data_lists, "load_rows",
                                    side_effect=lambda: calls.append("rows") or {"validation": VAL, "train": {}}),
                  mock.patch.object(torch.cuda, "is_available", return_value=False),
                  contextlib.redirect_stdout(buf)):
            st.enter_context(p)
        code = E.main(["--root", str(root), *argv])
    return code, buf.getvalue()


def rehash(root, rid):
    """Record the current run files of rid in results/s2/summary.json (a consistent but different record)."""
    p = root / "results" / "s2" / "summary.json"
    doc = json.loads(p.read_text(encoding="ascii"))
    for n in E.RUN_FILES:
        doc["inputs_sha256"][f"runs/{rid}/{n}"] = hashlib.sha256((root / "runs" / rid / n).read_bytes()).hexdigest()
    train.write_json(p, doc)


# ---------------------------------------------------------------- hand-computed scenarios

H_RUNS = [{"model": "bert", "seed": s, "run_id": f"r{s}"} for s in SEEDS]


def triple(p):
    return (1.0 if p else -1.0, 0.75 if p else 0.25, p)


def select(case_rows, val, review_rows=None):
    cases, errs = E.check_cases(case_rows, val)
    assert not errs, errs
    rows = review_rows if review_rows is not None else [review_row(c) for c in case_rows]
    review, errs = E.check_review(table_bytes(E.REVIEW_COLUMNS, rows), cases)
    assert not errs, errs
    recs, info, errs = E.select_cases(cases, review, val)
    assert not errs, errs
    return recs, info


class TestMetricsHand(unittest.TestCase):
    """ROB-02: attack originals A1-A3 (rows 0-2) and A4 (row 5, its paraphrase relabelled 0), benign B1, B2
    (rows 3, 4); ROB-04: 2 benign cases. Variant labels: A1-A3 1, A4 B1 B2 0. Original labels: A1-A4 1.

    seed   originals A1 A2 A3 B1 B2 A4   variants A1 A2 A3 B1 B2 A4
    42                1  1  1  0  0  1             0  0  0  0  0  0
    123               1  1  1  0  0  1             1  0  0  1  0  0
    2026              1  1  0  0  0  1             1  1  1  0  0  1
    """
    ORIG = [[1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 0, 1], [1, 1, 0, 0, 0, 1]]
    VAR = [[0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0], [1, 1, 1, 0, 0, 1]]

    def setUp(self):
        val = {sid(i): (y, f"orig {i}") for i, y in enumerate(["1", "1", "1", "0", "0", "1"])}
        rows = [case_row(f"dev-rob02-{i:06d}", "ROB-02", "paraphrase", f"p{i}", src=sid(i), val=val)
                for i in range(6)]
        rows += [case_row(f"dev-rob04-{j:03d}", "ROB-04", "benign_security", f"b{j}") for j in (1, 2)]
        review = [review_row(c) for c in rows]
        review[5].update(decision="relabel", new_label="0", reason="not an attack")
        self.recs, _ = select(rows, val, review)
        rob04 = {"dev-rob04-001": [1, 0, 0], "dev-rob04-002": [0, 0, 0]}
        P = {}
        for k, s in enumerate(SEEDS):
            P[f"r{s}"] = {"orig": {sid(i): triple(p) for i, p in enumerate(self.ORIG[k])},
                          "case": {f"dev-rob02-{i:06d}": triple(p) for i, p in enumerate(self.VAR[k])}}
            P[f"r{s}"]["case"].update({c: triple(ps[k]) for c, ps in rob04.items()})
        self.res = E.evaluate(self.recs, H_RUNS, P)

    def row(self, seed, cond, side, label):
        got = [r for r in self.res["metrics"] if (r["seed"], r["condition"], r["side"], r["label"])
               == (seed, cond, side, label)]
        self.assertEqual(len(got), 1, (seed, cond, side, label))
        return got[0]

    def stat(self, cond, side, label, q):
        got = [r for r in self.res["model_rows"] if (r["condition"], r["side"], r["label"], r["quantity"])
               == (cond, side, label, q)]
        self.assertEqual(len(got), 1)
        return got[0]

    def test_per_seed_metrics(self):
        r = self.row(42, "ROB-02", "variant", "all")  # TP 0, FP 0, FN 3, TN 3
        self.assertEqual([r[k] for k in ("n", "tp", "fp", "fn", "tn")], [6, 0, 0, 3, 3])
        self.assertEqual((r["f1"], r["precision"], r["recall"], r["fpr"], r["roc_auc"]), (0.0, None, 0.0, 0.0, 0.5))
        r = self.row(123, "ROB-02", "variant", "all")  # TP 1, FP 1 (B1), FN 2, TN 2 (B2, A4)
        self.assertEqual([r[k] for k in ("tp", "fp", "fn", "tn")], [1, 1, 2, 2])
        for k, want in (("f1", 0.4), ("precision", 0.5), ("recall", 1 / 3), ("fpr", 1 / 3), ("roc_auc", 0.5)):
            self.assertAlmostEqual(r[k], want, places=12, msg=k)
        r = self.row(2026, "ROB-02", "variant", "all")  # A4 (relabelled 0) flagged: FP 1; AUC 7.5/9
        for k, want in (("f1", 6 / 7), ("precision", 0.75), ("recall", 1.0), ("fpr", 1 / 3), ("roc_auc", 5 / 6)):
            self.assertAlmostEqual(r[k], want, places=12, msg=k)
        r = self.row(2026, "ROB-02", "original", "all")  # original labels: A1-A4 attack, A3 missed
        self.assertEqual([r[k] for k in ("n", "tp", "fp", "fn", "tn")], [6, 3, 0, 1, 2])
        self.assertAlmostEqual(r["roc_auc"], 7 / 8, places=12)
        r = self.row(2026, "ROB-02", "original", "1")
        self.assertEqual((r["n"], r["tp"], r["fn"]), (4, 3, 1))  # A4 counted with its ORIGINAL label
        r = self.row(123, "ROB-02", "variant", "1")  # attack-only: recall only
        self.assertEqual((r["n"], r["tp"], r["fn"], r["fp"], r["tn"], r["precision"], r["f1"], r["fpr"],
                          r["roc_auc"]), (3, 1, 2, None, None, None, None, None, None))
        self.assertAlmostEqual(r["recall"], 1 / 3, places=12)
        r = self.row(123, "ROB-02", "variant", "0")
        self.assertEqual((r["n"], r["fp"], r["tn"], r["recall"], r["tp"]), (3, 1, 2, None, None))
        self.assertAlmostEqual(r["fpr"], 1 / 3, places=12)

    def test_benign_only_cell(self):
        rob04 = [r for r in self.res["metrics"] if r["condition"] == "ROB-04"]
        self.assertEqual({(r["side"], r["label"]) for r in rob04}, {("variant", "0")})  # no "all", no recall
        self.assertEqual([r["fpr"] for r in rob04], [0.5, 0.0, 0.0])
        st = self.stat("ROB-04", "variant", "0", "fpr")
        self.assertAlmostEqual(st["mean"], 1 / 6, places=12)
        self.assertFalse(any(r["condition"] == "ROB-04" for r in self.res["changes"]))

    def test_seed_mean_std_and_na(self):
        st = self.stat("ROB-02", "variant", "1", "recall")  # 0, 1/3, 1
        self.assertAlmostEqual(st["mean"], 4 / 9, places=12)
        self.assertAlmostEqual(st["std"], math.sqrt(21) / 9, places=12)
        st = self.stat("ROB-02", "original", "1", "recall")  # 4/4, 4/4, 3/4
        self.assertEqual(st["n"], 4)
        self.assertAlmostEqual(st["mean"], 11 / 12, places=12)
        st = self.stat("ROB-02", "variant", "all", "precision")  # N/A in seed 42 -> no mean / std
        self.assertEqual((st["mean"], st["std"], st["s42"], st["s123"], st["s2026"]), (None, None, None, 0.5, 0.75))
        st = self.stat("ROB-02", "variant", "all", "f1")
        self.assertAlmostEqual(st["mean"], (0.4 + 6 / 7) / 3, places=12)
        st = self.stat("ROB-02", "variant", "0", "fpr")
        self.assertAlmostEqual(st["mean"], 2 / 9, places=12)

    def test_change_rate_sliced_by_original_label(self):
        ch = {(r["seed"], r["label"]): r for r in self.res["changes"]}
        self.assertEqual([(ch[(s, "all")]["changed"], ch[(s, "all")]["to_benign"], ch[(s, "all")]["to_attack"])
                          for s in SEEDS], [(4, 4, 0), (4, 3, 1), (1, 0, 1)])
        self.assertEqual([ch[(s, "all")]["change_rate"] for s in SEEDS], [4 / 6, 4 / 6, 1 / 6])
        self.assertEqual([(ch[(s, "1")]["n_pairs"], ch[(s, "1")]["change_rate"]) for s in SEEDS],
                         [(4, 1.0), (4, 0.75), (4, 0.25)])  # A1-A4: attack originals (A4's case is label 0)
        self.assertEqual([(ch[(s, "0")]["n_pairs"], ch[(s, "0")]["change_rate"]) for s in SEEDS],
                         [(2, 0.0), (2, 0.5), (2, 0.0)])
        st = self.stat("ROB-02", "change", "all", "change_rate")
        self.assertAlmostEqual(st["mean"], 0.5, places=12)
        self.assertAlmostEqual(st["std"], math.sqrt(1 / 12), places=12)

    def test_bootstrap_point_and_strata(self):
        b = {r["metric"]: r for r in self.res["bootstrap"] if r["kind"] == "variant_vs_original"}
        self.assertAlmostEqual(b["recall"]["estimate_a"], 11 / 12, places=12)
        self.assertAlmostEqual(b["recall"]["estimate_b"], 4 / 9, places=12)
        self.assertAlmostEqual(b["recall"]["diff"], 4 / 9 - 11 / 12, places=12)
        # A4's group stays in its original-label stratum (1) although its case is labelled 0; a resample that
        # draws A4 (group index 5) for all 4 attack slots has no attack variant: recall undefined, CI N/A
        idx = E.resample_indices(["1", "1", "1", "0", "0", "1"])
        undefined = int(((idx == 5).sum(axis=1) == 4).sum())
        self.assertGreater(undefined, 0)
        self.assertEqual((b["recall"]["n_groups"], b["recall"]["strata"], b["recall"]["undefined_resamples"],
                          b["recall"]["nk_merges"], b["recall"]["diff_low"]), (6, "0:2;1:4", undefined, 0, None))
        self.assertAlmostEqual(b["fpr"]["diff"], 2 / 9, places=12)
        lv = [r for r in self.res["bootstrap"] if r["kind"] == "level"]
        self.assertEqual([(r["condition"], r["metric"], r["strata"]) for r in lv], [("ROB-04", "fpr", "0:2")])


def rob01_scenario(with_o4=False, relabel_o2_homoglyph=False):
    """(recs, P) of the TestBootstrapHand scenario; O4 (row 3, attack) has no homoglyph case and its main
    variants are all missed; the relabel turns O2's homoglyph case into label 0."""
    val = {sid(0): ("1", "o0"), sid(1): ("1", "o1"), sid(2): ("0", "o2"), sid(3): ("1", "o3")}
    rows = [case_row(f"dev-rob01-{sub}-{i:06d}", "ROB-01", sub, f"{sub}{i}", src=sid(i), val=val)
            for i in range(3) for sub in E.SUBTYPES["ROB-01"]]
    if with_o4:
        rows += [case_row(f"dev-rob01-{sub}-000003", "ROB-01", sub, f"{sub}3", src=sid(3), val=val)
                 for sub in ("case", "space", "char")]
    review = [review_row(c) for c in rows]
    if relabel_o2_homoglyph:
        next(r for r in review if r["case_id"] == "dev-rob01-homoglyph-000001").update(
            decision="relabel", new_label="0", reason="lost its intent")
    recs, _ = select(rows, val, review)
    main = {0: [1, 1, 1], 1: [0, 0, 1], 2: [0, 0, 0], 3: [0, 0, 0]}
    homo = {0: 1, 1: 0, 2: 1}
    P = {}
    for s in SEEDS:
        case = {}
        for i in range(4 if with_o4 else 3):
            for j, sub in enumerate(("case", "space", "char")):
                case[f"dev-rob01-{sub}-{i:06d}"] = triple(1 if (i == 1 and s == 2026) else main[i][j])
            if i < 3:
                case[f"dev-rob01-homoglyph-{i:06d}"] = triple(homo[i])
        P[f"r{s}"] = {"case": case, "orig": {sid(i): triple([1, 1, 0, 1][i]) for i in range(4)}}
    return recs, P


class TestBootstrapHand(unittest.TestCase):
    """ROB-01: attack originals O1 (row 0), O2 (row 1), benign O3 (row 2), four subtypes each.

    Main variants (case, space, char): O1 1,1,1; O2 0,0,1 (seed 2026: 1,1,1); O3 0,0,0. Homoglyph: O1 1, O2 0,
    O3 1. Originals: O1 1, O2 1, O3 0. With k = draws of O2 among the 2 attack groups of a resample: main
    recall is 1 - k/3 in seeds 42/123 and 1 in 2026, so the 3-seed mean is 1 - 2k/9 and variant - original
    is -2k/9; homoglyph recall is 1 - k/2, homoglyph - main = -5k/18. k is 0, 1, 2 with 1/4, 1/2, 1/4.
    """

    def setUp(self):
        self.recs, self.P = rob01_scenario()
        self.idx = E.resample_indices(["1", "1", "0"])
        self.k = (self.idx == 1).sum(axis=1)

    def rows(self, recs, P):
        res = E.evaluate(recs, H_RUNS, P)
        return {(r["kind"], r["subtype_b"], r["metric"]): r for r in res["bootstrap"]}

    def test_resample_indices_are_the_seeded_stratified_draws(self):
        rng = np.random.default_rng(314159)
        first = np.array([2])[rng.integers(0, 1, size=(2000, 1))]  # stratum "0" (sorted first): group 2
        second = np.array([0, 1])[rng.integers(0, 2, size=(2000, 2))]  # stratum "1": groups 0, 1
        self.assertTrue(np.array_equal(self.idx, np.concatenate([first, second], axis=1)))
        self.assertEqual(self.idx.shape, (2000, 3))
        self.assertTrue(((self.idx == 2).sum(axis=1) == 1).all())  # every resample: 1 benign, 2 attack groups
        self.assertTrue(np.array_equal(E.resample_indices(["1", "1", "0"]), self.idx))

    def test_grouped_paired_distribution(self):
        main = [r for r in self.recs if r["eval_group"] == "main"]
        stratum = {sid(i): s for i, s in enumerate(["1", "1", "0"])}
        rids = [f"r{s}" for s in SEEDS]
        res = E.bootstrap_dist(E.side_items(main, "original"), rids, E.side_items(main, "variant"), rids, "recall",
                               stratum, self.P)
        self.assertEqual((res["groups"], res["merged"]), ([sid(0), sid(1), sid(2)], []))
        self.assertTrue(np.allclose(res["a"][1], 1.0))
        self.assertTrue(np.allclose(res["diff"], -2 * self.k / 9))  # cases of a group move together
        self.assertTrue(0.2 < (self.k == 2).mean() < 0.3 and 0.2 < (self.k == 0).mean() < 0.3)

    def test_rows(self):
        b = self.rows(self.recs, self.P)
        r = b[("variant_vs_original", "main", "recall")]
        self.assertAlmostEqual(r["diff"], -2 / 9, places=12)
        self.assertAlmostEqual(r["diff_low"], -4 / 9, places=12)
        self.assertAlmostEqual(r["diff_high"], 0.0, places=12)
        self.assertEqual((r["ci_vs_0"], r["n_groups"], r["strata"]), ("includes_0", 3, "0:1;1:2"))
        r = b[("variant_vs_original", "main", "fpr")]  # one benign group, kept in every resample (stratified)
        self.assertEqual((r["diff"], r["diff_low"], r["diff_high"], r["undefined_resamples"]), (0.0, 0.0, 0.0, 0))
        r = b[("held_out_vs_main", "homoglyph", "recall")]
        self.assertAlmostEqual(r["diff"], -5 / 18, places=12)
        self.assertAlmostEqual(r["diff_low"], -5 / 9, places=12)
        self.assertAlmostEqual(r["diff_high"], 0.0, places=12)
        r = b[("held_out_vs_main", "homoglyph", "fpr")]
        self.assertEqual((r["estimate_a"], r["estimate_b"], r["diff_low"], r["diff_high"], r["ci_vs_0"]),
                         (0.0, 1.0, 1.0, 1.0, ">0"))
        r = b[("variant_vs_original", "homoglyph", "fpr")]
        self.assertEqual((r["diff"], r["ci_vs_0"]), (1.0, ">0"))
        self.assertEqual(self.rows(self.recs, self.P), b)  # deterministic

    def test_held_out_uses_shared_originals_only(self):
        base = self.rows(self.recs, self.P)
        b = self.rows(*rob01_scenario(with_o4=True))
        self.assertEqual(b[("variant_vs_original", "main", "recall")]["n_groups"], 4)  # O4 is in the main cell
        for metric in ("recall", "fpr"):  # but not in homoglyph - main: same rows as without O4
            self.assertEqual(b[("held_out_vs_main", "homoglyph", metric)],
                             base[("held_out_vs_main", "homoglyph", metric)], metric)
        self.assertEqual(b[("held_out_vs_main", "homoglyph", "recall")]["n_groups"], 3)

    def test_relabelled_held_out_case(self):
        """O2's homoglyph case relabelled 0: its group stays in stratum 1. Homoglyph FPR = 1 / (1 + k) (O3 flagged
        once, O2 drawn k times, never flagged), main FPR 0; homoglyph recall needs O1 (undefined when k = 2)."""
        b = self.rows(*rob01_scenario(relabel_o2_homoglyph=True))
        r = b[("held_out_vs_main", "homoglyph", "fpr")]
        self.assertEqual((r["strata"], r["estimate_a"], r["estimate_b"], r["diff"]), ("0:1;1:2", 0.0, 0.5, 0.5))
        self.assertAlmostEqual(r["diff_low"], 1 / 3, places=12)
        self.assertAlmostEqual(r["diff_high"], 1.0, places=12)
        r = b[("held_out_vs_main", "homoglyph", "recall")]
        self.assertAlmostEqual(r["diff"], 1 - 7 / 9, places=12)
        self.assertEqual((r["diff_low"], r["diff_high"], r["ci_vs_0"]), (None, None, ""))
        self.assertEqual(r["undefined_resamples"], int((self.k == 2).sum()))
        r = b[("variant_vs_original", "homoglyph", "recall")]  # original side keeps O2 as attack
        self.assertEqual((r["strata"], r["estimate_a"]), ("0:1;1:2", 1.0))

    def test_undefined_ci(self):
        self.assertEqual(E._ci(np.array([0.1, np.nan])), (None, None))

    def test_model_vs_model(self):
        runs = H_RUNS + [{"model": "roberta", "seed": s, "run_id": f"q{s}"} for s in SEEDS]
        P = dict(self.P)
        for s in SEEDS:  # roberta detects everything and flags every benign variant
            P[f"q{s}"] = {"case": {c: triple(1) for c in self.P[f"r{s}"]["case"]}, "orig": self.P[f"r{s}"]["orig"]}
        res = E.evaluate(self.recs, runs, P)
        r = [x for x in res["bootstrap"] if x["kind"] == "model_vs_model" and x["subtype_a"] == "main"
             and x["metric"] == "recall"]
        self.assertEqual(len(r), 1)
        self.assertEqual((r[0]["model_a"], r[0]["model_b"]), ("bert", "roberta"))
        self.assertAlmostEqual(r[0]["diff"], 2 / 9, places=12)  # 1 - (1 - 2/9)
        self.assertAlmostEqual(r[0]["diff_high"], 4 / 9, places=12)


class TestNormalisedKeyGroups(unittest.TestCase):
    """README 6.1 item 1: ROB-02 paraphrases of A1 (row 0) and A2 (row 1) share a normalised key, so the two
    groups resample as one (M = rows 0 and 1, id = the smaller). Variants: A1 1, A2 0, A3 1; originals all
    detected. With j = draws of M among the 2 attack draws, variant recall = 2 / (2 + j), diff = -j / (2 + j)."""

    def setUp(self):
        val = {sid(0): ("1", "o0"), sid(1): ("1", "o1"), sid(2): ("1", "o2"), sid(3): ("0", "o3")}
        texts = ["Hello   World", "hello world", "other text", "benign text"]
        rows = [case_row(f"dev-rob02-{i:06d}", "ROB-02", "paraphrase", t, src=sid(i), val=val)
                for i, t in enumerate(texts)]
        self.recs, _ = select(rows, val)
        self.P = {f"r{s}": {"case": {f"dev-rob02-{i:06d}": triple(p) for i, p in enumerate([1, 0, 1, 0])},
                            "orig": {sid(i): triple(p) for i, p in enumerate([1, 1, 1, 0])}} for s in SEEDS}

    def test_merge_groups(self):
        items = [("g3", 1, "a"), ("g1", 1, "b"), ("g2", 1, "c"), ("g4", 0, "d"), ("g5", 0, "e")]
        nk = {"a": "x", "b": "y", "c": "x", "d": "y", "e": ""}  # g3~g2 by x, g1~g4 by y; empty keys never merge
        self.assertEqual(E.merge_groups(items, nk), {"g1": "g1", "g2": "g2", "g3": "g2", "g4": "g1", "g5": "g5"})

    def test_merged_resampling(self):
        res = E.evaluate(self.recs, H_RUNS, self.P)
        r = next(x for x in res["bootstrap"] if x["kind"] == "variant_vs_original" and x["metric"] == "recall")
        self.assertEqual((r["n_groups"], r["strata"], r["nk_merges"]), (3, "0:1;1:2", 1))
        self.assertEqual(res["grouping"], {"nk_merged_group_sets": [[sid(0), sid(1)]],
                                           "comparisons_with_nk_merges": 2})  # recall and FPR comparisons
        nk_of = {("case", x["case_id"]): x["nk"] for x in self.recs}
        nk_of.update({("orig", x["source_id"]): x["orig_nk"] for x in self.recs})
        stratum = {x["group_id"]: x["stratum"] for x in self.recs}
        rids = [f"r{s}" for s in SEEDS]
        d = E.bootstrap_dist(E.side_items(self.recs, "original"), rids, E.side_items(self.recs, "variant"), rids,
                             "recall", stratum, self.P, nk_of=nk_of)
        self.assertEqual((d["groups"], d["merged"]), ([sid(0), sid(2), sid(3)], [[sid(0), sid(1)]]))
        j = (E.resample_indices(["1", "1", "0"]) == 0).sum(axis=1)
        self.assertTrue(np.allclose(d["diff"], -j / (2 + j)))
        self.assertAlmostEqual(r["diff"], 2 / 3 - 1, places=12)


class TestRules(unittest.TestCase):
    def test_used_cases_and_unapplied_rob01_exclude(self):
        cases = make_cases()
        review = make_review(cases)
        recs, info = select(cases, VAL, review)
        ids = {r["case_id"] for r in recs}
        self.assertFalse(any(c.endswith("-000003") for c in ids))  # review_excluded: history rows only
        self.assertEqual((info["rob01_groups_dropped"], info["unapplied_excludes"]), ([], []))
        self.assertEqual(info["status_counts"]["review_excluded"], 4)
        next(r for r in review if r["case_id"] == "dev-rob01-space-000017").update(decision="exclude", reason="x")
        recs, info = select(cases, VAL, review)  # before --apply-review (only --allow-incomplete gets here)
        self.assertFalse(any(r["case_id"].endswith("-000017") for r in recs))
        self.assertEqual((info["rob01_groups_dropped"], info["unapplied_excludes"]),
                         ([sid(17)], ["dev-rob01-space-000017"]))
        self.assertEqual(len(info["rob01_cases_dropped"]), 3)  # the other three subtypes of that original
        r5 = next(r for r in recs if r["case_id"] == "dev-rob02-000005")
        self.assertEqual((r5["label"], r5["orig_label"], r5["stratum"]), (1, 0, "0"))
        self.assertNotIn("dev-rob02-000014", ids)  # reserve
        self.assertNotIn("dev-rob01-case-000015", ids)  # auto_excluded

    def test_rob03_attributes(self):
        recs, _ = select(make_cases(), VAL, make_review(make_cases()))
        r3 = {r["case_id"]: r for r in recs if r["condition"] == "ROB-03"}
        r = r3["dev-rob03-000008"]  # prefix, 45-word context
        self.assertEqual((r["position"], r["words_before"], r["words_after"], r["length"]), ("prefix", 45, 0, "long"))
        r = r3["dev-rob03-000010"]  # both, two 6-word contexts
        self.assertEqual((r["words_before"], r["words_after"], r["length"]), (6, 6, "short"))
        self.assertEqual(E.rob03_words("ctx words\n\nO", "O", "suffix"), None)
        self.assertEqual(E.rob03_words("O\n\nctx words", "O", "suffix"), (0, 2))

    def test_round_keeps_margin_check_precision(self):
        self.assertEqual(E._round({"max_abs_margin_diff_9_digits": 1.234567891e-9, "x": 1.234567891e-9,
                                   "l": [0.1234567]}),
                         {"max_abs_margin_diff_9_digits": 1.234567891e-9, "x": 0.0, "l": [0.123457]})


# ---------------------------------------------------------------- end to end

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp, self.root = fresh_root(self)

    def test_outputs(self):
        calls, out = [], self.tmp / "out"
        code, text = run_main(self.root, ["--out", str(out)], calls)
        self.assertEqual(code, 0, text)
        self.assertEqual(calls[:2], ["guard", "rows"])
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(OUT_FILES))  # no .tmp left
        for n in OUT_FILES:
            b = (out / n).read_bytes()
            self.assertNotIn(b"\r", b, n)
            self.assertNotIn(MARK.encode(), b, n)
            for p in (str(self.tmp), self.tmp.as_posix(), self.tmp.name, "orig1 ", "alpha", "\u0430".encode()):
                self.assertNotIn(p.encode("utf-8") if isinstance(p, str) else p, b, n)
        preds = read_table(out / "predictions.csv")
        doc = json.loads((out / "summary.json").read_text(encoding="ascii"))
        n_used = 4 * 4 + 4 + 6 + 3  # ROB-01 originals 0-2 and 17, ROB-02 4, ROB-03 6, ROB-04 3
        self.assertEqual(len(preds), 9 * n_used)
        self.assertEqual(list(preds[0]), list(E.PRED_COLUMNS))
        self.assertFalse(doc["exploratory"])
        self.assertEqual(doc["device"]["type"], "cpu")
        self.assertEqual(doc["settings"]["attn_implementation"], "eager")
        self.assertEqual(len(doc["runs"]), 9)
        for r in doc["runs"]:
            c = r["original_check"]
            self.assertEqual((c["prediction_flips"], c["max_abs_margin_diff_9_digits"],
                              c["margin_rows_differing_9_digits"], c["validation_rows"]), (0, 0.0, 0, N_VAL))
        self.assertEqual(sorted(doc["inputs_sha256"])[:2], ["configs/train.json", "data/cases/dev/cases.csv"])
        self.assertEqual(len(doc["inputs_sha256"]), 1 + 2 + 2 + 27)
        self.assertEqual(doc["grouping"], {"nk_merged_group_sets": [], "comparisons_with_nk_merges": 0})
        # metrics.csv agrees with predictions.csv: ROB-02 variant recall of s2-roberta-A-s123
        pr = {r["case_id"]: int(r["prediction"]) for r in preds if r["run_id"] == "s2-roberta-A-s123"}
        att = ["dev-rob02-000004", "dev-rob02-000005", "dev-rob02-000006"]  # 000005 relabelled 0 -> 1
        tp = sum(pr[c] for c in att)
        met = [r for r in read_table(out / "metrics.csv") if (r["run_id"], r["condition"], r["side"], r["label"])
               == ("s2-roberta-A-s123", "ROB-02", "variant", "1")]
        self.assertEqual((met[0]["n"], met[0]["tp"], met[0]["recall"]), ("3", str(tp), f"{tp / 3:.6f}"))
        md = (out / "summary.md").read_text(encoding="utf-8")
        self.assertIn("# 3단계 개발용 추가 평가 결과", md)
        self.assertNotIn("탐색적 결과", md)
        mrows = read_table(out / "models.csv")
        self.assertEqual([r["model"] for r in mrows], sorted((r["model"] for r in mrows), key=MODELS.index))
        boot = read_table(out / "bootstrap.csv")
        self.assertEqual({r["kind"] for r in boot}, {"variant_vs_original", "held_out_vs_main", "model_vs_model",
                                                     "level"})
        self.assertEqual({r["nk_merges"] for r in boot}, {"0"})
        slices = {r["slice"] for r in read_table(out / "metrics.csv") if r["condition"] == "ROB-03"}
        self.assertEqual(slices, {"all", "position=prefix", "position=suffix", "position=both", "length=short",
                                  "length=long"})

    def test_deterministic(self):
        outs = [self.tmp / "o1", self.tmp / "o2"]
        for o in outs:
            code, text = run_main(self.root, ["--out", str(o)])
            self.assertEqual(code, 0, text)
        for n in OUT_FILES:
            self.assertEqual((outs[0] / n).read_bytes(), (outs[1] / n).read_bytes(), n)

    def test_review_bom_crlf_blank_rows_and_spaces_accepted(self):
        p = self.root / "data" / "cases" / "dev" / "review.csv"
        p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes().replace(b",keep,", b", keep ,").replace(b"\n", b"\r\n")
                      + b",,,,,\r\n")
        code, text = run_main(self.root, ["--out", str(self.tmp / "out")])
        self.assertEqual(code, 0, text)

    def test_pending_refused_and_allow_incomplete(self):
        p = self.root / "data" / "cases" / "dev" / "review.csv"
        rows = read_table(p)
        rows[0].update(decision="pending", reviewer="")
        write_table(p, E.REVIEW_COLUMNS, rows)
        code, text = run_main(self.root, [])
        self.assertEqual(code, 1, text)
        self.assertIn("P01 review.csv: 1 assigned case(s) are still pending", text)
        self.assertIn("nothing written", text)
        self.assertFalse((self.root / "results" / "s3").exists())
        for out in (self.root / "results" / "s3", self.root / "results" / "s3" / "sub"):
            code, text = run_main(self.root, ["--allow-incomplete", "--out", str(out)])
            self.assertEqual(code, 1, text)
            self.assertIn("E02 --allow-incomplete output must not be inside results/s3", text)
        code, text = run_main(self.root, ["--allow-incomplete"])
        self.assertEqual(code, 0, text)
        out = self.root / "results" / "s3_exploratory"
        self.assertFalse((self.root / "results" / "s3").exists())
        for n in OUT_FILES:
            if n.endswith(".csv"):
                lines = (out / n).read_text(encoding="utf-8").split("\n")
                self.assertTrue(lines[0].endswith(",exploratory"), n)
                self.assertTrue(all(ln.endswith(",1") for ln in lines[1:-1]), n)
        self.assertTrue(json.loads((out / "summary.json").read_text(encoding="ascii"))["exploratory"])
        self.assertIn("탐색적 결과", (out / "summary.md").read_text(encoding="utf-8"))
        self.assertNotIn(rows[0]["case_id"], (out / "predictions.csv").read_text(encoding="utf-8"))

    def test_other_root_refused(self):
        code, text = run_main(self.root, ["--out", str(self.tmp / "out")], lib_root=self.tmp)
        self.assertEqual(code, 1, text)
        self.assertIn("E04 --root must be the repository root", text)
        self.assertFalse((self.tmp / "out").exists())


def _review(fn):
    def mutate(root):
        p = root / "data" / "cases" / "dev" / "review.csv"
        rows = read_table(p)
        rows = fn(rows) or rows
        write_table(p, E.REVIEW_COLUMNS, rows)
    return mutate


def _cases(fn):
    def mutate(root):
        p = root / "data" / "cases" / "dev" / "cases.csv"
        rows = read_table(p)
        fn(rows)
        write_table(p, E.CASES_COLUMNS, rows)
    return mutate


def _by(rows, cid):
    return next(r for r in rows if r["case_id"] == cid)


def _s2_pin(root):
    p = root / "results" / "s2" / "runs.csv"
    rows = read_table(p)
    rows[0]["checkpoint_sha256"] = "0" * 64
    write_table(p, summarize_s2.RUN_COLUMNS, rows)


def _consistent_wrong_pin(root):
    """runs.csv and summary.json agree on a checkpoint SHA-256 that run.json and the file do not have."""
    _s2_pin(root)
    q = root / "results" / "s2" / "summary.json"
    doc = json.loads(q.read_text(encoding="ascii"))
    doc["runs"][0]["checkpoint_sha256"] = "0" * 64
    train.write_json(q, doc)


def _tampered_checkpoint(root):
    f = root / "runs" / "s2-bert-A-s42" / "checkpoint" / "model.safetensors"
    b = bytearray(f.read_bytes())
    b[-1] ^= 1
    f.write_bytes(bytes(b))


def _other_model(root):
    """A valid, consistently recorded checkpoint of another model: pins pass, original predictions differ."""
    rid = "s2-bert-A-s42"
    d = root / "runs" / rid
    tmp = root / "other_ckpt"
    info = model_specs.save_checkpoint(tiny_model(999), "bert", tmp, rid, SEL)
    shutil.copyfile(tmp / "model.safetensors", d / "checkpoint" / "model.safetensors")
    doc = json.loads((d / "run.json").read_text(encoding="ascii"))
    doc["checkpoint"].update(sha256=info["sha256"], size=info["size"])
    train.write_json(d / "run.json", doc)
    p = root / "results" / "s2" / "runs.csv"
    rows = read_table(p)
    rows[0]["checkpoint_sha256"] = info["sha256"]
    write_table(p, summarize_s2.RUN_COLUMNS, rows)
    q = root / "results" / "s2" / "summary.json"
    s = json.loads(q.read_text(encoding="ascii"))
    s["runs"][0]["checkpoint_sha256"] = info["sha256"]
    train.write_json(q, s)
    rehash(root, rid)


def _flip_stored_prediction(root):
    rid = "s2-roberta-A-s123"
    p = root / "runs" / rid / "predictions_validation.csv"
    lines = p.read_text(encoding="ascii").split("\n")
    for k, ln in enumerate(lines):
        f = ln.split(",")
        if f[0] == str(SEL) and f[1] == sid(7):
            m = -float(f[4])
            sc = 1 / (1 + math.exp(-m))
            f[4], f[5], f[6] = eval_lib.fmt9(m), eval_lib.fmt9(sc), str(int(sc >= 0.5))
            lines[k] = ",".join(f)
    p.write_bytes("\n".join(lines).encode("ascii"))
    rehash(root, rid)


def _config_changed(root):
    p = root / "configs" / "train.json"
    p.write_bytes(p.read_bytes() + b"\n")  # still valid JSON, other SHA-256


REFUSALS = {  # name: (mutation of a fresh root, expected text); rows[0..3]: dev-rob01-{case,space,char,homoglyph}-0
    "R05 decision": (_review(lambda rows: rows[0].update(decision="maybe")),
                     "R05 dev-rob01-case-000000: decision not one of"),
    "R06 relabel label": (_review(lambda rows: rows[0].update(decision="relabel", new_label="", reason="x")),
                          "R06 dev-rob01-case-000000: relabel needs new_label 0 or 1"),
    "R06 relabel same label": (_review(lambda rows: _by(rows, "dev-rob02-000004").update(
        decision="relabel", new_label="1", reason="x")), "R06 dev-rob02-000004: new_label equals the case label"),
    "R06 new_label on keep": (_review(lambda rows: rows[1].update(new_label="1")),
                              "R06 dev-rob01-space-000000: new_label only with relabel"),
    "R06 ROB-04 relabel": (_review(lambda rows: _by(rows, "dev-rob04-001").update(
        decision="relabel", new_label="1", reason="x")), "R06 dev-rob04-001: ROB-04 is benign-only"),
    "R07 exclude reason": (_review(lambda rows: rows[3].update(decision="exclude")),
                           "R07 dev-rob01-homoglyph-000000: exclude needs a reason"),
    "R08 reviewer claude": (_review(lambda rows: rows[2].update(reviewer="Claude")),
                            "R08 dev-rob01-char-000000: reviewer must be one of HANJUSON"),
    "R08 reviewer other": (_review(lambda rows: rows[2].update(reviewer="someone")), "reviewer must be one of"),
    "R08 pending reviewer": (_review(lambda rows: rows[2].update(decision="pending")),
                             "R08 dev-rob01-char-000000: a pending row has no reviewer"),
    "R09 stale text": (_cases(lambda rows: _by(rows, "dev-rob02-000004").update(text=f"{MARK} changed")),
                       "R09 dev-rob02-000004: text_sha256_12 differs from the case text"),
    "R09 no hash": (_review(lambda rows: rows[1].update(text_sha256_12="")),
                    "R09 dev-rob01-space-000000: a decided row needs the text_sha256_12"),
    "E10 pending excluded": (_review(lambda rows: _by(rows, "dev-rob01-case-000003").update(
        decision="pending", reviewer="")), "E10 review.csv dev-rob01-case-000003: pending row of a review_excluded"),
    "E10 missing row": (_review(lambda rows: rows[:-1]), "E10 review.csv: 1 assigned case(s) have no row"),
    "E10 reserve case": (_review(lambda rows: rows + [{**rows[0], "case_id": "dev-rob02-000014"}]),
                         "E10 review.csv dev-rob02-000014: not an assigned or review_excluded case"),
    "R04 duplicate": (_review(lambda rows: rows + [dict(rows[0])]), "duplicate case_id dev-rob01-case-000000"),
    "R02 header": (lambda r: (r / "data/cases/dev/review.csv").write_bytes(b"case_id,decision\n"),
                   "R02 review.csv: header is not"),
    "P02 unapplied exclude": (_review(lambda rows: _by(rows, "dev-rob02-000006").update(
        decision="exclude", reason="x")), "P02 review.csv: 1 assigned case(s) have decision exclude"),
    "C01 header": (lambda r: (r / "data/cases/dev/cases.csv").write_bytes(b"case_id\n"), "C01"),
    "C02 label vs original": (_cases(lambda rows: _by(rows, "dev-rob02-000004").update(label="0")),
                              "label differs from the original's validation label"),
    "C02 held-out group": (_cases(lambda rows: _by(rows, "dev-rob01-case-000000").update(eval_group="held_out")),
                           "invalid eval_group"),
    "C02 duplicate original": (_cases(lambda rows: _by(rows, "dev-rob02-000005").update(
        source_id=sid(4), group_id=sid(4), label="1")), "same original as dev-rob02-000004"),
    "C03 ROB-03 position": (_cases(lambda rows: _by(rows, "dev-rob03-000008").update(position="suffix")),
                            "C03 cases.csv dev-rob03-000008: the original is not at position suffix"),
    "E03 config pin": (_config_changed, "E03 configs/train.json: SHA-256 (LF)"),
    "M01 runs.csv vs summary.json": (_s2_pin, "M01 s2-bert-A-s42: runs.csv model/seed, or selected step"),
    "M02 run file changed": (lambda r: (r / "runs/s2-deberta-A-s2026/metrics.json").write_bytes(b"{}\n"),
                             "M02 s2-deberta-A-s2026: metrics.json SHA-256 differs"),
    "M03 pinned checkpoint": (_consistent_wrong_pin, "M03 s2-bert-A-s42: run.json selected step / checkpoint differ"),
    "M04 checkpoint bytes": (_tampered_checkpoint, "M04 s2-bert-A-s42: checkpoint refused"),
    "M05 other model": (_other_model, "M05 s2-bert-A-s42:"),
    "M05 stored prediction": (_flip_stored_prediction,
                              "M05 s2-roberta-A-s123: 1 of 30 original (validation) predictions differ"),
}


class TestRefusals(unittest.TestCase):
    def test_refusals(self):
        for name, (mutate, want) in REFUSALS.items():
            with self.subTest(name):
                tmp, root = fresh_root(self)
                mutate(root)
                out = tmp / "out"
                code, text = run_main(root, ["--out", str(out)])
                self.assertEqual(code, 1, text)
                self.assertIn(want, text)
                self.assertIn("nothing written", text)
                self.assertFalse(out.exists())
                self.assertNotIn(MARK, text)

    def test_pins_and_attn_passed_to_load_run_model(self):
        tmp, root = fresh_root(self)
        seen = []
        real = eval_lib.load_run_model

        def spy(run_dir, *args, **kw):
            seen.append((Path(run_dir).name, args, kw))
            return real(run_dir, *args, **kw)

        with mock.patch.object(eval_lib, "load_run_model", side_effect=spy):
            code, text = run_main(root, ["--out", str(tmp / "out")])
        self.assertEqual(code, 0, text)
        pins = {r["run_id"]: r["checkpoint_sha256"] for r in read_table(root / "results" / "s2" / "runs.csv")}
        self.assertEqual([n for n, _, _ in seen], [f"s2-{m}-A-s{s}" for m in MODELS for s in SEEDS])
        attn = json.loads((root / "configs" / "train.json").read_bytes())["attn_implementation"]
        for n, args, kw in seen:
            self.assertEqual(args, (attn,))
            self.assertEqual(kw["expected_sha256"], pins[n])
            size = json.loads((root / "runs" / n / "run.json").read_text(encoding="ascii"))["checkpoint"]["size"]
            self.assertEqual(kw["expected_size"], size)


if __name__ == "__main__":
    unittest.main()
