"""CPU tests for scripts/summarize_s2.py on synthetic stage-2 run folders.

No models, no GPU, no dataset text: nine run folders with 11 validation rows
(row 5 outside validation_clean) are written to a temp dir in train.py's formats.

Run from Git Bash at the repo root:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m unittest tests/test_summarize_s2.py -v
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
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

import data_lists  # noqa: E402
import eval_lib  # noqa: E402
import model_specs  # noqa: E402
import summarize_s2 as S  # noqa: E402
import train  # noqa: E402

MODELS = ["bert", "roberta", "deberta"]
SEEDS = [42, 123, 2026]
STEPS = [10, 20, 30]
EXCLUDED = 5  # in_clean 0
LABELS = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
BASE = [3.0, 2.0, 1.0, 0.5, 1.0, 4.0, -3.0, -2.0, -1.5, -0.5, -0.2]  # correct-sign margins
# Rows each run gets wrong at the selected step 20 (the margin's sign is flipped).
WRONG = {
    ("bert", 42): {4, 10}, ("bert", 123): {4}, ("bert", 2026): {3, 10},
    ("roberta", 42): {4, 9}, ("roberta", 123): {4, 9, 10}, ("roberta", 2026): {4, 10},
    ("deberta", 42): {4}, ("deberta", 123): {2, 5, 8}, ("deberta", 2026): {2, 4},
}
FILES = ("runs.csv", "models.csv", "errors.csv", "summary.json", "summary.md")


def sid(i):
    return f"validation:{i:06d}"


def margin(i, wrong):
    return -BASE[i] if i in wrong else BASE[i]


def pred_line(step, i, m):
    sc = 1 / (1 + math.exp(-m)) if math.isfinite(m) else 0.0
    return (f"{step},{sid(i)},{LABELS[i]},{int(i != EXCLUDED)},{eval_lib.fmt9(m)},{eval_lib.fmt9(sc)},"
            f"{int(sc >= 0.5)},0.5,A")


def config_bytes(seeds=SEEDS):
    return json.dumps({"seeds": seeds, "expected_updates": 30}).encode("ascii")


def write_run(root, model, seed, cfg_sha, rid=None):
    """One completed run in train.py's formats. Step 10 has one more FN and step 30 one more FP than the
    selected step 20, so selected != first and != last."""
    rid = rid or f"s2-{model}-A-s{seed}"
    d = root / "runs" / rid
    (d / "checkpoint").mkdir(parents=True)
    ckpt = f"checkpoint {rid}".encode("ascii")
    (d / "checkpoint" / "model.safetensors").write_bytes(ckpt)
    wrong = WRONG[(model, seed)]
    in_clean = [i != EXCLUDED for i in range(len(LABELS))]
    lines, evals = [train.PRED_HEADER], []
    for step, w in ((10, wrong | {0}), (20, wrong), (30, wrong | {6})):
        m = [margin(i, w) for i in range(len(LABELS))]
        lines += [pred_line(step, i, m[i]) for i in range(len(LABELS))]
        p = torch.tensor([int(x >= 0) for x in m])
        rec = train.eval_record(step, LABELS, in_clean, torch.tensor(m, dtype=torch.float64), p)
        rec["improved"] = step != 30
        evals.append(rec)
    (d / "predictions_validation.csv").write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    sel = eval_lib.select(evals)["step"]
    assert sel == 20
    train.write_json(d / "metrics.json", {"run_id": rid, "threshold": 0.5, "eval_steps": STEPS, "evals": evals,
                                          "selected_step": sel, "selection_rule": "test"})
    mi, si = MODELS.index(model), SEEDS.index(seed)
    repo, rev = model_specs.MODELS[model]
    train.write_json(d / "run.json", {
        "schema": 1, "run_id": rid, "stage": "s2", "model_key": model, "condition": "A", "seed": seed,
        "attempt": 1 + (model == "roberta" and seed == 123), "status": "completed",
        "code": {"git_commit": "c0ffee", "files_sha256_lf": {"scripts/train.py": "aa"}, "git_dirty": [],
                 "git_dirty_other_count": 2},
        "config": {"path": "configs/train.json", "sha256": cfg_sha},
        "lists": {"mode": "official", "git_tracked_checked": True, "files_sha256": {"validation_clean.csv": "bb"},
                  "lists_meta_sha256": "cc", "review_sha256": "dd"},
        "env": {"requirements_sha256": "ee", "torch": "2.14.1+cu130", "cuda": "13.0", "gpu": {"name": "GPU X"}},
        "budget": {"N": 30, "eval_steps": STEPS, "updates_run": 30},
        "model": {"key": model, "repo": repo, "revision": rev, "parameters": 1000 + mi,
                  "weights": {"file": model_specs.WEIGHTS[model][0], "sha256": model_specs.WEIGHTS[model][1]},
                  "tokenizer_fingerprint": model_specs.TOKENIZER_SHA256[model]},
        "data": {"validation_rows": len(LABELS), "validation_clean_rows": sum(in_clean)},
        "selection": {"selected_step": sel, "saved_steps": [10, 20]},
        "checkpoint": {"kept": True, "path": f"runs/{rid}/checkpoint/model.safetensors",
                       "sha256": hashlib.sha256(ckpt).hexdigest(), "size": len(ckpt), "step": sel,
                       "reload_check": {"passed": True}},
        "resources": {"update_seconds_total": 100.0 + si + 10 * mi, "update_seconds_median": 0.25,
                      "eval_s": [1.0 + mi, 1.5 + mi, 2.0 + mi],
                      "peak_memory_allocated_bytes": 2_000_000_000 + mi, "peak_memory_reserved_bytes": 2_500_000_000},
    })


def make_root(base):
    root = Path(base) / "repo"
    (root / "configs").mkdir(parents=True)
    cfg = config_bytes()
    (root / "configs" / "train.json").write_bytes(cfg)
    for m in MODELS:
        for s in SEEDS:
            write_run(root, m, s, hashlib.sha256(cfg).hexdigest())
    return root


def edit_json(path, fn):
    doc = json.loads(path.read_text(encoding="ascii"))
    fn(doc)
    train.write_json(path, doc)


def edit_bytes(path, old, new, count=1):
    b = path.read_bytes()
    assert b.count(old) >= 1, old
    path.write_bytes(b.replace(old, new, count))


def quiet_main(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = S.main(argv)
    return code, buf.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = make_root(self.tmp)


class TestNumbers(Base):
    def test_metrics_hand_computed(self):
        errs, res = S.summarize(self.root)
        self.assertEqual(errs, [])
        self.assertEqual([(r["model"], r["seed"]) for r in res["rows"]], [(m, s) for m in MODELS for s in SEEDS])
        r = res["rows"][0]  # bert s42: clean pos margins 3, 2, 1, 0.5, -1; clean neg -3, -2, -1.5, -0.5, 0.2
        self.assertEqual((r["run_id"], r["selected_step"]), ("s2-bert-A-s42", 20))
        self.assertEqual([r[f"validation_clean_{k}"] for k in ("n", "tp", "fp", "fn", "tn")], [10, 4, 1, 1, 4])
        for k, want in (("f1", 0.8), ("precision", 0.8), ("recall", 0.8), ("fpr", 0.2), ("roc_auc", 23 / 25)):
            self.assertAlmostEqual(r[f"validation_clean_{k}"], want, places=12, msg=k)
        # validation adds row 5 (label 1, margin 4, correct): TP 5, FN 1, FP 1, TN 4; AUC 28/30
        self.assertEqual([r[f"validation_{k}"] for k in ("n", "tp", "fp", "fn", "tn")], [11, 5, 1, 1, 4])
        for k, want in (("f1", 10 / 12), ("precision", 5 / 6), ("recall", 5 / 6), ("fpr", 0.2),
                        ("roc_auc", 28 / 30)):
            self.assertAlmostEqual(r[f"validation_{k}"], want, places=12, msg=k)
        self.assertEqual((r["update_seconds_total"], r["eval_seconds_mean"], r["peak_memory_allocated_bytes"],
                          r["attempt"]), (100.0, 1.5, 2_000_000_000, 1))

    def test_seed_mean_and_sample_std(self):
        _, res = S.summarize(self.root)
        st = res["stats"]["bert"]
        self.assertEqual(st["validation_clean_fp"]["values"], [1, 0, 1])
        self.assertAlmostEqual(st["validation_clean_fp"]["mean"], 2 / 3, places=12)
        self.assertAlmostEqual(st["validation_clean_fp"]["std"], math.sqrt(1 / 3), places=12)  # n - 1
        self.assertEqual((st["selected_step"]["mean"], st["selected_step"]["std"]), (20.0, 0.0))
        self.assertEqual((st["update_seconds_total"]["mean"], st["update_seconds_total"]["std"]), (101.0, 1.0))
        self.assertEqual(res["stats"]["roberta"]["update_seconds_total"]["values"], [110.0, 111.0, 112.0])
        self.assertEqual(res["stats"]["deberta"]["eval_seconds_mean"]["mean"], 3.5)

    def test_na_gives_no_mean(self):
        rows = [dict.fromkeys(S.QUANTITIES, 1.0) for _ in SEEDS]
        rows[1]["validation_clean_roc_auc"] = None
        st = S.model_stats(rows)
        self.assertEqual((st["validation_clean_roc_auc"]["mean"], st["validation_clean_roc_auc"]["std"]), (None, None))
        self.assertEqual(st["validation_clean_f1"]["std"], 0.0)


class TestErrors(Base):
    def test_two_of_three_rule_and_sets(self):
        _, res = S.summarize(self.root)
        rows = {r["sentence_id"]: r for r in res["error_rows"]}
        # row 5 is outside validation_clean; rows 0 (step 10) and 6 (step 30) are wrong only at unselected steps
        self.assertEqual(sorted(rows), [sid(i) for i in (2, 3, 4, 8, 9, 10)])
        want = {  # i: (error type, seeds wrong bert/roberta/deberta, shared_by)
            2: ("FN", (0, 0, 2), "deberta"), 3: ("FN", (1, 0, 0), "none"), 4: ("FN", (2, 3, 2), "bert;roberta;deberta"),
            8: ("FP", (0, 0, 1), "none"), 9: ("FP", (0, 2, 0), "roberta"), 10: ("FP", (2, 2, 0), "bert;roberta"),
        }
        for i, (et, counts, shared) in want.items():
            r = rows[sid(i)]
            self.assertEqual(r["error_type"], et)
            self.assertEqual(tuple(r[f"{m}_seeds_wrong"] for m in MODELS), counts)
            self.assertEqual(tuple(r[f"{m}_model_error"] for m in MODELS), tuple(int(c >= 2) for c in counts))
            self.assertEqual((r["shared_by"], r["n_models_error"]), (shared, sum(c >= 2 for c in counts)))
        E = res["errors"]
        self.assertEqual(E["per_model"]["bert"]["model_error"], {"total": 2, "fp": 1, "fn": 1})
        self.assertEqual(E["per_model"]["roberta"]["model_error"], {"total": 3, "fp": 2, "fn": 1})
        self.assertEqual(E["per_model"]["deberta"]["model_error"], {"total": 2, "fp": 0, "fn": 2})
        self.assertEqual({m: E["per_model"][m]["only_this_model"]["total"] for m in MODELS},
                         {"bert": 0, "roberta": 1, "deberta": 1})
        self.assertEqual(E["per_model"]["roberta"]["by_seeds_wrong"],
                         {"1": {"total": 0, "fp": 0, "fn": 0}, "2": {"total": 2, "fp": 2, "fn": 0},
                          "3": {"total": 1, "fp": 0, "fn": 1}})
        self.assertEqual(E["all_models"], {"total": 1, "fp": 0, "fn": 1})
        self.assertEqual(E["exactly_two"], {"bert;roberta": {"total": 1, "fp": 1, "fn": 0},
                                            "bert;deberta": {"total": 0, "fp": 0, "fn": 0},
                                            "roberta;deberta": {"total": 0, "fp": 0, "fn": 0}})
        self.assertEqual(E["any_model"], {"total": 4, "fp": 2, "fn": 2})
        self.assertEqual(E["any_run"], {"total": 6, "fp": 3, "fn": 3})


class TestOutputs(Base):
    def test_deterministic_lf_no_paths(self):
        outs = [self.tmp / "out1", self.tmp / "out2"]
        for o in outs:
            code, text = quiet_main(["--root", str(self.root), "--out", str(o)])
            self.assertEqual(code, 0, text)
        self.assertEqual(sorted(p.name for p in outs[0].iterdir()), sorted(FILES))
        for n in FILES:
            a, b = ((o / n).read_bytes() for o in outs)
            self.assertEqual(a, b, n)
            self.assertNotIn(b"\r", a, n)
            for p in (str(self.tmp), self.tmp.as_posix(), self.tmp.name):
                self.assertNotIn(p.encode("utf-8"), a, n)
        runs = (outs[0] / "runs.csv").read_text(encoding="ascii").split("\n")
        self.assertEqual((runs[0].split(","), len(runs)), (list(S.RUN_COLUMNS), 1 + 9 + 1))
        first = dict(zip(S.RUN_COLUMNS, runs[1].split(",")))
        self.assertEqual((first["validation_clean_f1"], first["validation_clean_roc_auc"],
                          first["update_seconds_total"], first["eval_seconds_mean"],
                          first["peak_memory_allocated_bytes"]),
                         ("0.800000", "0.920000", "100.000000", "1.500000", "2000000000"))
        errors = (outs[0] / "errors.csv").read_text(encoding="ascii").split("\n")
        self.assertEqual(errors[0], "sentence_id,label,error_type,bert_seeds_wrong,roberta_seeds_wrong,"
                                    "deberta_seeds_wrong,bert_model_error,roberta_model_error,deberta_model_error,"
                                    "n_models_error,shared_by")
        self.assertEqual(errors[3], "validation:000004,1,FN,2,3,2,1,1,1,3,bert;roberta;deberta")
        models = (outs[0] / "models.csv").read_text(encoding="ascii").split("\n")
        self.assertEqual(models[0], "model,quantity,mean,std,s42,s123,s2026")
        self.assertIn("bert,validation_clean_fp,0.666667,0.577350,1,0,1", models)
        doc = json.loads((outs[0] / "summary.json").read_text(encoding="ascii"))
        self.assertEqual(len(doc["inputs_sha256"]), 1 + 9 * 3)
        self.assertTrue(all(k.startswith(("runs/s2-", "configs/")) for k in doc["inputs_sha256"]))
        self.assertEqual(doc["runs"][0]["exact"]["validation_clean"]["f1"], "4/5")
        md = (outs[0] / "summary.md").read_text(encoding="utf-8")
        # BERT validation_clean F1 4/5, 8/9, 4/5: mean 0.82963, sample std 0.05132
        self.assertIn("| BERT | `validation_clean` | 0.8296 ± 0.0513 | ", md)
        self.assertIn("| `s2-bert-A-s42` | 20 | 0.8000 | 0.8000 | 0.8000 | 0.2000 | 0.9200 | 1 | 1 | 0.8333 | 100.0 |",
                      md)
        self.assertIn("N = 30 update, 평가 시점은 10 / 20 / 30", md)


def _run(r, rid="s2-bert-A-s42"):
    return r / "runs" / rid


def _pred(r, rid="s2-bert-A-s42"):
    return _run(r, rid) / "predictions_validation.csv"


def _line(step, i, m):
    return pred_line(step, i, m).encode("ascii")


def _swap_first_rows(r):
    p = _pred(r)
    lines = p.read_bytes().split(b"\n")
    lines[1], lines[2] = lines[2], lines[1]
    p.write_bytes(b"\n".join(lines))


def _extra_seed(r):
    write_run(r, "bert", 42, hashlib.sha256(config_bytes()).hexdigest(), rid="s2-bert-A-s7")


def _undefined_key(r):
    def fn(d):
        d["evals"][0]["validation_clean"].update(tp=0, fp=0, fn=0)
    edit_json(_run(r) / "metrics.json", fn)


def _select_last(r):
    edit_json(_run(r) / "metrics.json", lambda d: d.update(selected_step=30))
    edit_json(_run(r) / "run.json", lambda d: (d["selection"].update(selected_step=30),
                                               d["checkpoint"].update(step=30)))


def _rj(fn, rid="s2-bert-A-s42"):
    return lambda r: edit_json(_run(r, rid) / "run.json", fn)


REFUSALS = {  # name: (mutation of a fresh root, expected text in the output)
    "S00 config seeds": (lambda r: (r / "configs" / "train.json").write_bytes(config_bytes([42, 123])),
                         "S00 configs/train.json: seeds must be 3 distinct integers"),
    "S01 missing run": (lambda r: shutil.rmtree(_run(r, "s2-roberta-A-s123")), "S01 s2-roberta-A-s123"),
    "S01 lock file": (lambda r: (r / "runs" / "s2-bert-A-s42.lock").write_bytes(b""), ".lock exists"),
    "S01 missing file": (lambda r: (_run(r) / "metrics.json").unlink(), "missing metrics.json"),
    "S01 extra seed": (_extra_seed, "S01 unexpected stage-2 run folders (model or seed not in model_specs / config): "
                                    "s2-bert-A-s7"),
    "S02 invalid json": (lambda r: (_run(r) / "metrics.json").write_bytes(b"{"), "is not valid JSON"),
    "S02 missing key": (_rj(lambda d: d.pop("resources")), "missing key or wrong type (KeyError: 'resources')"),
    "S02 identity": (_rj(lambda d: d.update(seed=7)), "S02 s2-bert-A-s42: run.json seed do not match the run ID"),
    "S03 not completed": (_rj(lambda d: d.update(status="running")), "S03 s2-bert-A-s42: status is 'running'"),
    "S04 reload failed": (_rj(lambda d: d["checkpoint"]["reload_check"].update(passed=False)),
                          "reload check did not pass"),
    "S04 checkpoint missing": (lambda r: (_run(r) / "checkpoint" / "model.safetensors").unlink(),
                               "is missing or its size differs"),
    "S04 checkpoint path": (_rj(lambda d: d["checkpoint"].update(path="runs/x/checkpoint/model.safetensors")),
                            "path not runs/s2-bert-A-s42/checkpoint/model.safetensors"),
    "S04 checkpoint sha": (lambda r: (_run(r) / "checkpoint" / "model.safetensors").write_bytes(
        b"X" + b"checkpoint s2-bert-A-s42"[1:]), "model.safetensors SHA-256 differs from run.json"),
    "S04 checkpoint step": (_rj(lambda d: d["checkpoint"].update(step=10)), "checkpoint step 10 != selected step 20"),
    "S05 selected step": (lambda r: edit_json(_run(r) / "metrics.json", lambda d: d.update(selected_step=10)),
                          "metrics.json selected_step 10 != run.json 20"),
    "S05 selection rule": (_select_last, "does not follow the selection rule"),
    "S05 selection key": (_undefined_key, "selection key undefined in metrics.json"),
    "S06 header": (lambda r: edit_bytes(_pred(r), b"step,sentence_id", b"step,sid"), "header or final LF differs"),
    "S06 final LF": (lambda r: _pred(r).write_bytes(_pred(r).read_bytes().rstrip(b"\n")), "header or final LF"),
    "S06 malformed": (lambda r: edit_bytes(_pred(r), _line(10, 3, 0.5), _line(10, 3, 0.5) + b",x"),
                      "line 5 is malformed"),
    "S06 threshold": (lambda r: edit_bytes(_pred(r), _line(20, 1, 2.0), _line(20, 1, 2.0)[:-5] + b"0.4,A"),
                      "threshold not 0.5"),
    "S06 condition": (lambda r: edit_bytes(_pred(r), _line(20, 1, 2.0), _line(20, 1, 2.0)[:-1] + b"B"),
                      "condition not A"),
    "S06 prediction vs score": (lambda r: edit_bytes(_pred(r), _line(20, 0, 3.0),
                                                     _line(20, 0, 3.0).replace(b",1,0.5,A", b",0,0.5,A")),
                                "prediction != (score >= threshold)"),
    "S06 IDs order": (_swap_first_rows, "step 10: no rows, or sentence IDs not strictly ascending"),
    "S06 step rows": (lambda r: edit_bytes(_pred(r), _line(30, 3, 0.5) + b"\n", b""),
                      "step 30: sentence IDs, labels or in_clean differ from step 10"),
    "S06 row counts": (_rj(lambda d: d["data"].update(validation_rows=12)), "prediction row counts differ"),
    "S06 non-finite": (lambda r: edit_bytes(_pred(r), _line(20, 1, 2.0), _line(20, 1, 2.0).replace(b",2,", b",nan,")),
                       "S06 s2-bert-A-s42: step 20: metrics: 1 of 11 margins are not finite"),
    "S06 predictions vs metrics": (lambda r: edit_bytes(_pred(r), _line(20, 0, 3.0), _line(20, 0, -3.0)),
                                   "step 20 validation_clean: predictions do not reproduce metrics.json"),
    "S06 ROC-AUC only": (lambda r: edit_json(_run(r) / "metrics.json",
                                             lambda d: d["evals"][1]["validation_clean"].update(roc_auc=0.5)),
                         "step 20 validation_clean: predictions do not reproduce metrics.json (roc_auc)"),
    "S07 resources": (_rj(lambda d: d["resources"].update(update_seconds_total=None)),
                      "S07 s2-bert-A-s42: resources"),
    "S07 eval times": (_rj(lambda d: d["resources"].update(eval_s=[1.0])), "S07 s2-bert-A-s42: resources"),
    "S08 code sha": (_rj(lambda d: d["code"]["files_sha256_lf"].update({"scripts/train.py": "x"})),
                     "S08 code files SHA-256 differs"),
    "S08 list sha": (_rj(lambda d: d["lists"]["files_sha256"].update(x="y"), "s2-deberta-A-s2026"),
                     "S08 list files SHA-256 differs"),
    "S08 config sha": (lambda r: (r / "configs" / "train.json").write_bytes(config_bytes() + b" "),
                       "S08 config SHA-256 differs from the current configs/train.json"),
    "S08 model between seeds": (_rj(lambda d: d["model"].update(parameters=999)),
                                "S08 model record of bert differs between its seeds"),
    "S08 validation rows": (lambda r: edit_bytes(_pred(r, "s2-bert-A-s123"), b"validation:000010,",
                                                 b"validation:000011,", 3),
                            "S08 validation rows (IDs, labels, in_clean) differ from s2-bert-A-s42: s2-bert-A-s123"),
    "S09 lists": (_rj(lambda d: d["lists"].update(mode="trial")), "S09 s2-bert-A-s42: lists are not official"),
    "S09 tracked check": (_rj(lambda d: d["lists"].update(git_tracked_checked=False)), "lists are not official"),
    "S09 git dirty": (_rj(lambda d: d["code"].update(git_dirty=[{"status": " M", "path": "scripts/train.py"}])),
                      "code.git_dirty is not empty"),
    "S09 updates run": (_rj(lambda d: d["budget"].update(updates_run=29)), "S09 s2-bert-A-s42: budget"),
    "S09 model pin": (_rj(lambda d: d["model"].update(revision="0" * 40)), "model record differs from model_specs"),
}


class TestRefusals(unittest.TestCase):
    def test_refusals(self):
        for name, (mutate, want) in REFUSALS.items():
            with self.subTest(name):
                tmp = Path(tempfile.mkdtemp())
                self.addCleanup(shutil.rmtree, tmp, True)
                root = make_root(tmp)
                mutate(root)
                out = tmp / "out"
                code, text = quiet_main(["--root", str(root), "--out", str(out)])
                self.assertEqual(code, 1, text)
                self.assertIn(want, text)
                self.assertIn("nothing written", text)
                self.assertFalse(out.exists())


class TestShowText(Base):
    def test_terminal_only_after_guard(self):
        calls = []
        rows = {sid(i): (str(y), f"TEXT-{i} \x1b[31m\n") for i, y in enumerate(LABELS)}
        out = self.tmp / "out"
        with mock.patch.object(data_lists, "install_test_guard", side_effect=lambda: calls.append("guard")), \
                mock.patch.object(data_lists, "load_rows",
                                  side_effect=lambda: calls.append("rows") or {"validation": rows, "train": {}}):
            code, text = quiet_main(["--root", str(self.root), "--out", str(out), "--show-text", "1"])
        self.assertEqual(code, 0, text)
        self.assertEqual(calls, ["guard", "rows"])
        # first model error per model in ID order: bert 4, roberta 4, deberta 2; at most 1 each
        self.assertIn("validation:000004 FN (seeds wrong: bert 2, roberta 3, deberta 2): TEXT-4 \\x1b[31m\\n", text)
        self.assertIn("TEXT-2", text)
        self.assertNotIn("TEXT-10", text)
        self.assertNotIn("\x1b", text)
        for n in FILES:
            self.assertNotIn(b"TEXT-", (out / n).read_bytes(), n)


if __name__ == "__main__":
    unittest.main()
