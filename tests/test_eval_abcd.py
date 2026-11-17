"""CPU tests for scripts/eval_abcd.py on synthetic runs, thresholds and cases.

No GPU (torch.cuda.is_available is patched to False), no pretrained weights or real checkpoints, no dataset
text: validation rows, cases, reviews, the nine stage-2 run folders with results/s2 records, three C run
folders and results/s4/thresholds.json are written to temp dirs. The six representative runs hold tiny BERT
configs saved with model_specs.save_checkpoint and loaded through the real eval_lib.load_run_model; the
tokenizer is a word-level stand-in. The test.csv guard is patched (it is a process-wide audit hook).
The fixture also holds what the real stage-4 inputs may hold: validation rows that share a normalised key,
C runs whose lists_meta / build_lists / train.py SHA-256 differ from the A runs, one C run recorded as a
second attempt and one threshold recorded as chosen without a candidate under the FPR cap.

Run from Git Bash at the repo root:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m unittest tests/test_eval_abcd.py -v
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
import eval_abcd as X  # noqa: E402
import eval_cases as E  # noqa: E402
import eval_lib  # noqa: E402
import model_specs  # noqa: E402
import summarize_s2  # noqa: E402
import train  # noqa: E402

MODELS = ["bert", "roberta", "deberta"]
REP = "roberta"
SEEDS = [42, 123, 2026]
N_UPDATES = 2073  # expected_updates of configs/train.json
STEPS = [691, 1382, 2073]
SEL = {("A", 42): 2073, ("A", 123): 691, ("A", 2026): 691, ("C", 42): 1382, ("C", 123): 2073, ("C", 2026): 691}
LOGITS = {("B", 42): -8.0, ("B", 123): 8.0, ("B", 2026): 0.0, ("D", 42): 1.35, ("D", 123): -8.0, ("D", 2026): -0.6}
MARK = "ZQXTEXT"  # in every synthetic text; must never reach an output file
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo"]
OUT_FILES = ("runs.csv", "metrics.csv", "models.csv", "bootstrap.csv", "predictions.csv", "summary.json",
             "summary.md")
FP = model_specs.TOKENIZER_SHA256[REP]
PAIRS = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "D"), ("C", "D"), ("B", "C")]


def sid(i):
    return f"validation:{i:06d}"


def rid_of(trained, seed):
    return f"{'s2' if trained == 'A' else 's4'}-{REP}-{trained}-s{seed}"


def thr_value(t):
    """float32(sigmoid(t)) as a Python float (design D5)."""
    return float(np.float32(1 / (1 + math.exp(-t))))


T_LO, T_HI = thr_value(-1.0), thr_value(1.0)  # 0.2689414322376251, 0.7310585975646973


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
NOT_CLEAN = 20  # the one validation row outside validation_clean (no case uses it)
SIDS = [sid(i) for i in range(N_VAL)]
VAL = {sid(i): ("1" if i % 2 == 0 else "0", f"{MARK} orig{i} {words(i, 4)}") for i in range(N_VAL)}
# validation_clean rows that no case uses and that share a normalised key (README 6.1 item 1: one group):
# 21, 23 and 27 are benign (other letter case, other spacing); 22 is an attack and 25 benign (mixed labels).
VAL[sid(23)] = ("0", VAL[sid(21)][1].upper())
VAL[sid(27)] = ("0", VAL[sid(21)][1].replace(" ", "  "))
VAL[sid(25)] = ("0", " " + VAL[sid(22)][1].replace(" ", "\t"))
NK_SETS = [[sid(21), sid(23), sid(27)], [sid(22), sid(25)]]
# 29 validation_clean rows (14 attack, 15 benign) in 26 groups: 13 attack (without 22), 12 benign (15 without
# 23, 25 and 27) and the mixed group of 22 and 25; 2 + 1 rows are merged into another row's group
VAL_GROUPS, VAL_STRATA, VAL_NK_MERGES = 26, "0:12;1:13;mixed:1", 3
LABELS = [int(VAL[s][0]) for s in SIDS]
IN_CLEAN = [int(i != NOT_CLEAN) for i in range(N_VAL)]


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
    rows = []
    for i in (0, 1, 2, 3):  # ROB-01 originals (labels 1, 0, 1, 0)
        o = VAL[sid(i)][1]
        for sub, t in (("case", o.upper()), ("space", o.replace(" ", "  ") + " x y"),
                       ("char", o.replace("a", "4") + " z"), ("homoglyph", o.replace("a", "а") + " w")):
            rows.append(case_row(f"dev-rob01-{sub}-{i:06d}", "ROB-01", sub, t, src=sid(i)))
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
    return rows


N_USED = 4 * 4 + 4 + 6 + 3


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


# ---------------------------------------------------------------- synthetic run folders

def step_values(m0, s0, k):
    """(margins, scores, predictions) of one eval step: the model's float32 values with the first k correct
    validation_clean rows turned wrong (margin negated, score mirrored), so the step is strictly worse."""
    m, s = m0.clone(), s0.clone()
    p = eval_lib.predictions(s, 0.5).tolist()
    todo = [i for i in range(N_VAL) if IN_CLEAN[i] and p[i] == LABELS[i]][:k]
    assert len(todo) == k
    for i in todo:
        m[i], s[i] = -m[i], 1 - s[i]
    q = eval_lib.predictions(s, 0.5).tolist()
    assert all(q[i] != p[i] for i in todo)
    return m, s, q


def run_files(model, cond, sel):
    """(predictions_validation.csv bytes, metrics.json document without the run ID): the model's own
    predictions at step sel, worse ones at the other two steps."""
    ids, _ = eval_lib.encode(WordTok(), [VAL[s][1] for s in SIDS], 256)
    m0, s0 = eval_lib.predict(model, ids, 0, 64, "cpu")
    lines, evals = [train.PRED_HEADER], []
    for step in STEPS:
        m, s, p = step_values(m0, s0, 0 if step == sel else 1 + STEPS.index(step))
        lines += [f"{step},{SIDS[i]},{LABELS[i]},{IN_CLEAN[i]},{eval_lib.fmt9(m[i].item())},"
                  f"{eval_lib.fmt9(s[i].item())},{p[i]},0.5,{cond}" for i in range(N_VAL)]
        evals.append(train.eval_record(step, LABELS, IN_CLEAN, m, torch.tensor(p)))
    assert eval_lib.select(evals)["step"] == sel
    return ("\n".join(lines) + "\n").encode("ascii"), {"threshold": 0.5, "eval_steps": STEPS, "evals": evals,
                                                         "selected_step": sel}


ATTEMPT = {("C", 2026): 2}  # the C run of seed 2026 is a second attempt (a rerun); every other run is attempt 1
ENV = {"python": "3.14.3", "platform": "Windows-11-10.0.26200-SP0", "torch": "2.14.1+cu130", "cuda": "13.0",
       "cudnn": 92400, "transformers": "5.22.2", "tokenizers": "0.23.3", "safetensors": "0.8.0",
       "huggingface_hub": "2.4.0", "scikit_learn": "1.9.1",
       "gpu": {"name": "GPU X", "capability": [12, 0], "total_memory_bytes": 8546484224, "driver": "610.88"},
       "requirements_sha256": "ee"}  # the entries train.env_info records


def run_record(rid, trained, seed, sel, ckpt, cfg_sha):
    """A run.json in train.py's layout with the records eval_abcd reads. The lists_meta, build_lists and
    train.py SHA-256 of the C runs differ from the A runs (both scripts change before the C training)."""
    repo, rev = model_specs.MODELS[REP]
    doc = {
        "schema": 1, "run_id": rid, "stage": "s2" if trained == "A" else "s4", "model_key": REP,
        "condition": trained, "seed": seed, "attempt": ATTEMPT.get((trained, seed), 1), "status": "completed",
        "code": {"git_commit": "c0ffee" if trained == "A" else "decade", "git_dirty": [], "git_dirty_other_count": 0,
                 "files_sha256_lf": {"scripts/train.py": "aa" if trained == "A" else "ab",
                                     "scripts/eval_lib.py": "ee"}},
        "config": {"path": "configs/train.json", "sha256": cfg_sha},
        "lists": {"mode": "official", "git_tracked_checked": True, "dataset_revision": "rev1",
                  "files_sha256": {"train_clean.csv": "t1", "validation_clean.csv": "v1"},
                  "lists_meta_sha256": "cc" if trained == "A" else "cd",
                  "build_lists_sha256_lf": "b0" if trained == "A" else "b2", "review_sha256": "dd",
                  "train_clean_rows": 11050},
        "env": json.loads(json.dumps(ENV)),
        "determinism": {"use_deterministic_algorithms": True, "warn_only": False, "nondeterministic_warnings": []},
        "budget": {"N": N_UPDATES, "epochs": 3, "eval_steps": STEPS, "grad_accum": 4, "micro_batch": 4,
                   "update_size": 16, "updates_run": N_UPDATES},
        "model": {"key": REP, "repo": repo, "revision": rev, "parameters": 1000, "precision": "bf16",
                  "attn_implementation": "eager", "tokenizer_fingerprint": FP,
                  "weights": {"file": model_specs.WEIGHTS[REP][0], "sha256": model_specs.WEIGHTS[REP][1],
                              "size": model_specs.WEIGHTS[REP][2]}},
        "data": {"train_rows": 11050, "validation_rows": N_VAL, "validation_clean_rows": sum(IN_CLEAN)},
        "selection": {"selected_step": sel, "saved_steps": [sel]},
        "checkpoint": {"kept": True, "path": f"runs/{rid}/checkpoint/model.safetensors", "sha256": ckpt["sha256"],
                       "size": ckpt["size"], "step": sel, "reload_check": {"passed": True}},
    }
    if trained == "C":
        doc["data"].update(train_clean_rows=11050, augmentation_rows=1700, train_rows=12750, augmentation={
            "dir": "data/cases/train", "files_sha256": {"train_aug.csv": "a1", "cases.csv": "a2"},
            "label_counts": {"0": 850, "1": 850}, "build_train_aug_sha256_lf": "b1"})
    return doc


def write_run(root, trained, seed, init_seed, cfg_sha):
    """A completed representative run with a real tiny checkpoint. Returns (run ID, checkpoint SHA-256)."""
    rid, sel = rid_of(trained, seed), SEL[(trained, seed)]
    d = root / "runs" / rid
    model = tiny_model(init_seed)
    ckpt = model_specs.save_checkpoint(model, REP, d / "checkpoint", rid, sel)
    preds, met = run_files(model, trained, sel)
    (d / "predictions_validation.csv").write_bytes(preds)
    train.write_json(d / "metrics.json", {"run_id": rid, **met})
    train.write_json(d / "run.json", run_record(rid, trained, seed, sel, ckpt, cfg_sha))
    return rid, ckpt["sha256"]


def write_light_run(root, model, seed):
    """A stage-2 run of another model: only what eval_cases.load_stage2 reads (its checkpoint is never loaded)."""
    rid = f"s2-{model}-A-s{seed}"
    d = root / "runs" / rid
    d.mkdir(parents=True)
    lines = [train.PRED_HEADER]
    for step in STEPS:
        for i in range(N_VAL):
            m = 1.5 if LABELS[i] else -1.5
            sc = 1 / (1 + math.exp(-m))
            lines.append(f"{step},{SIDS[i]},{LABELS[i]},{IN_CLEAN[i]},{eval_lib.fmt9(m)},{eval_lib.fmt9(sc)},"
                         f"{int(sc >= 0.5)},0.5,A")
    (d / "predictions_validation.csv").write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    train.write_json(d / "metrics.json", {"run_id": rid})
    sha = hashlib.sha256(rid.encode("ascii")).hexdigest()
    train.write_json(d / "run.json", {"run_id": rid, "checkpoint": {"sha256": sha, "size": 1, "step": STEPS[0]},
                                      "selection": {"selected_step": STEPS[0]}, "budget": {"eval_steps": STEPS}})
    return rid, sha


def file_sha(root, rel):
    return hashlib.sha256((root / rel).read_bytes()).hexdigest()


def write_s2_records(root, pins):
    """results/s2/runs.csv and summary.json holding the pins of the nine stage-2 runs."""
    cfg_sha = data_lists.sha256_lf(root / X.CONFIG_REL)
    rows, inputs = [], {X.CONFIG_REL: cfg_sha}
    for (m, s), (rid, sha) in pins.items():
        step = SEL[("A", s)] if m == REP else STEPS[0]
        rows.append({**dict.fromkeys(summarize_s2.RUN_COLUMNS), "model": m, "seed": s, "run_id": rid,
                     "selected_step": step, "checkpoint_sha256": sha})
        for n in X.RUN_FILES:
            inputs[f"runs/{rid}/{n}"] = file_sha(root, f"runs/{rid}/{n}")
    (root / "results" / "s2").mkdir(parents=True, exist_ok=True)
    (root / "results" / "s2" / "runs.csv").write_bytes(
        summarize_s2.csv_text(summarize_s2.RUN_COLUMNS, rows).encode("ascii"))
    train.write_json(root / "results" / "s2" / "summary.json", {
        "inputs_sha256": inputs, "records": {"config_sha256": cfg_sha},
        "runs": [{"run_id": r["run_id"], "selected_step": r["selected_step"],
                  "checkpoint_sha256": r["checkpoint_sha256"]} for r in rows]})


CAP_NOT_MET = ("D", 123)  # the one threshold the fixture records as chosen without a candidate under the FPR cap


def thresholds_doc(root, logits=LOGITS):
    """A thresholds.json in the document shape of design D5 (scripts/select_thresholds.py): one row per seed
    and condition under "thresholds", the source runs under "runs" and the SHA-256 of the input files."""
    def ckpt(trained, s):
        return json.loads((root / "runs" / rid_of(trained, s) / "run.json").read_text(encoding="ascii"))[
            "checkpoint"]["sha256"]

    rows = []
    for cond, trained in (("B", "A"), ("D", "C")):
        for s in SEEDS:
            t = logits[(cond, s)]
            stored = stored_rows(root, rid_of(trained, s), SEL[(trained, s)])
            chosen, base = (dict(zip(("tp", "fp", "fn", "tn"), counts_at(stored, x))) for x in (thr_value(t), 0.5))
            rows.append({"seed": s, "condition": cond, "source_condition": trained, "run_id": rid_of(trained, s),
                         "selected_step": SEL[(trained, s)], "checkpoint_sha256": ckpt(trained, s),
                         "cap": {"exact": "1/50", "value": 0.02, "a_run_id": rid_of("A", s)},
                         "cap_met": (cond, s) != CAP_NOT_MET,
                         "k": round(t * 20), "logit": t, "threshold": thr_value(t), "k_lo": -160, "k_hi": 160,
                         "chosen": {**chosen, "fpr": 0.25}, "base": {**base, "fpr": 0.25}})
    inputs = {"configs/train.json": "0" * 64}  # recorded by the selection, not compared here
    inputs.update({f"runs/{rid_of(t, s)}/{n}": file_sha(root, f"runs/{rid_of(t, s)}/{n}")
                   for t in "AC" for s in SEEDS for n in ("run.json", "predictions_validation.csv")})
    return {"schema": 1, "model": REP, "seeds": SEEDS, "default_threshold": 0.5, "inputs_sha256": inputs,
            "runs": [{"run_id": rid_of(t, s), "condition": t, "seed": s, "selected_step": SEL[(t, s)],
                      "checkpoint_sha256": ckpt(t, s)} for t in "AC" for s in SEEDS],
            "thresholds": rows}


def write_thresholds(root, doc=None):
    p = root / "results" / "s4" / "thresholds.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    train.write_json(p, doc if doc is not None else thresholds_doc(root))


def make_root(base):
    root = Path(base) / "repo"
    (root / "configs").mkdir(parents=True)
    shutil.copyfile(ROOT / "configs" / "train.json", root / "configs" / "train.json")
    cfg_sha = hashlib.sha256((root / "configs" / "train.json").read_bytes()).hexdigest()
    pins = {}
    for m in MODELS:
        for k, s in enumerate(SEEDS):
            pins[(m, s)] = write_run(root, "A", s, 100 + k, cfg_sha) if m == REP else write_light_run(root, m, s)
    write_s2_records(root, pins)
    for k, s in enumerate(SEEDS):
        write_run(root, "C", s, 200 + k, cfg_sha)
    cases = make_cases()
    write_table(root / "data" / "cases" / "dev" / "cases.csv", E.CASES_COLUMNS, cases)
    write_table(root / "data" / "cases" / "dev" / "review.csv", E.REVIEW_COLUMNS,
                [review_row(c) for c in cases if c["status"] == "assigned"])
    write_thresholds(root)
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
    """X.main with the CPU/tiny-model/test-data patches; returns (exit code, stdout)."""
    calls = calls if calls is not None else []
    buf = io.StringIO()
    with contextlib.ExitStack() as st:
        for p in (mock.patch.object(eval_lib, "ROOT", lib_root or root),
                  mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True),
                  mock.patch("transformers.AutoConfig.from_pretrained", side_effect=lambda *a, **k: tiny_config()),
                  mock.patch.object(model_specs, "load_tokenizer", side_effect=lambda key: WordTok()),
                  mock.patch.object(model_specs, "tokenizer_fingerprint", return_value=FP),
                  mock.patch.object(data_lists, "install_test_guard", side_effect=lambda: calls.append("guard")),
                  mock.patch.object(data_lists, "load_rows",
                                    side_effect=lambda: calls.append("rows") or {"validation": VAL, "train": {}}),
                  mock.patch.object(torch.cuda, "is_available", return_value=False),
                  contextlib.redirect_stdout(buf)):
            st.enter_context(p)
        code = X.main(["--root", str(root), *argv])
    return code, buf.getvalue()


def edit_json(path, fn):
    doc = json.loads(path.read_text(encoding="ascii"))
    fn(doc)
    train.write_json(path, doc)


def stored_rows(root, rid, step):
    """The fields of the predictions_validation.csv rows of one step."""
    lines = (root / "runs" / rid / "predictions_validation.csv").read_text(encoding="ascii").split("\n")[1:-1]
    return [f for f in (ln.split(",") for ln in lines) if int(f[0]) == step]


def counts_at(rows, thr):
    """(tp, fp, fn, tn) of the in_clean rows, computed here with numpy: float32(score) >= float32(threshold)."""
    tp = fp = fn = tn = 0
    for f in rows:
        if f[3] != "1":
            continue
        y, p = f[2] == "1", bool(np.float32(float(f[5])) >= np.float32(thr))
        tp, fp, fn, tn = tp + (y and p), fp + (not y and p), fn + (y and not p), tn + (not y and not p)
    return tp, fp, fn, tn


# ---------------------------------------------------------------- decision rule and stored predictions

class TestDecisionRule(unittest.TestCase):
    def test_score_at_or_above_threshold_is_an_attack(self):
        self.assertEqual(X.decide([0.2, 0.5, 0.7], 0.5), [0, 1, 1])  # a score equal to the threshold is 1
        self.assertEqual(X.decide([], 0.5), [])

    def test_comparison_is_made_in_float32(self):
        self.assertEqual(T_HI, 0.7310585975646973)  # float32(sigmoid(1)); the float64 sigmoid is 0.7310585786...
        below = float(np.nextafter(np.float32(T_HI), np.float32(0)))  # the float32 value just below
        self.assertEqual(X.decide([below, T_HI], T_HI), [0, 1])
        # 0.73105859 is below the threshold as a float64 but rounds to the threshold as a float32: attack
        self.assertLess(0.73105859, T_HI)
        self.assertEqual(X.decide([0.73105859], T_HI), [1])
        # a stored 9-digit score text gives back the float32 score
        self.assertEqual(X.decide([float(eval_lib.fmt9(T_HI))], T_HI), [1])
        self.assertEqual(X.decide([float(eval_lib.fmt9(below))], T_HI), [0])


def pred_bytes(rows, cond="A", steps=(10, 20), thr="0.5"):
    """predictions_validation.csv bytes; rows: [(sentence id, label, in_clean, margin text, score text, pred)]."""
    lines = [train.PRED_HEADER]
    for step in steps:
        lines += [f"{step},{s},{y},{c},{m},{sc},{p},{thr},{cond}" for s, y, c, m, sc, p in rows]
    return ("\n".join(lines) + "\n").encode("ascii")


ROWS = [(sid(0), 1, 1, "2.5", "0.924141824", 1), (sid(1), 0, 1, "-1", "0.268941432", 0),
        (sid(2), 1, 0, "0", "0.5", 1)]


class TestReadScores(unittest.TestCase):
    def test_values(self):
        P, errs = X.read_scores(pred_bytes(ROWS, "C"), [10, 20], "C")
        self.assertEqual(errs, [])
        self.assertEqual((P["ids"], P["labels"], P["in_clean"]), ([sid(0), sid(1), sid(2)], [1, 0, 1], [1, 1, 0]))
        self.assertEqual((P["pred"][20], P["margin"][20]), ([1, 0, 1], [2.5, -1.0, 0.0]))
        # the score is kept as the float32 value the text stands for
        self.assertEqual(P["score"][10], [float(np.float32(0.924141824)), float(np.float32(0.268941432)), 0.5])
        self.assertEqual(sorted(P["score"]), [10, 20])

    def test_same_rows_as_summarize_s2_for_an_a_file(self):
        raw = pred_bytes(ROWS, "A")
        P, errs = X.read_scores(raw, [10, 20], "A")
        S, serrs = summarize_s2.read_predictions(raw, [10, 20])
        self.assertEqual((errs, serrs), ([], []))
        for k in ("ids", "labels", "in_clean", "pred", "margin"):
            self.assertEqual(P[k], S[k], k)

    def test_refusals(self):
        def bad(raw, want, cond="A", steps=(10, 20)):
            P, errs = X.read_scores(raw, list(steps), cond)
            self.assertIsNone(P)
            self.assertEqual(len(errs), 1)
            self.assertIn(want, errs[0])

        ok = pred_bytes(ROWS)
        bad(ok, "condition not C", cond="C")
        bad(pred_bytes(ROWS, thr="0.25"), "threshold not 0.5")
        bad(ok, "step not an eval step", steps=(10,))
        bad(ok.replace(b"\n", b"\r\n"), "header or final LF differs")
        bad(ok[:-1], "header or final LF differs")
        bad(ok.replace(b",0.5,A\n", b",0.5\n", 1), "line 2 is malformed")
        bad(pred_bytes([ROWS[1], ROWS[0]]), "sentence IDs not strictly ascending")
        bad(pred_bytes(ROWS) + b"20," + sid(3).encode() + b",1,1,1,0.75,1,0.5,A\n", "differ from step 10")
        bad(pred_bytes([(sid(0), 1, 1, "nan", "0.75", 1)]), "margin not finite")
        bad(pred_bytes([(sid(0), 1, 1, "1", "1.5", 1)]), "score outside [0, 1]")
        bad(pred_bytes([(sid(0), 1, 1, "1", "0.75", 2)]), "prediction not 0/1")
        bad(pred_bytes([(sid(0), 1, 1, "1", "0.75", 0)]), "a stored prediction is not float32(score) >= 0.5")
        bad(pred_bytes([(sid(0), 1, 1, "-1", "0.25", 1)]), "a stored prediction is not float32(score) >= 0.5")
        # 0.30000000000000004 is a float64 text; float32 prints as 0.300000012
        bad(pred_bytes([(sid(0), 1, 1, "-1", "0.30000000000000004", 0)]), "not a 9-digit float32 value")
        bad(ok, "not a non-empty list of integers", steps=())
        self.assertEqual(X.read_scores(ok, None, "A")[1], ["the eval steps of run.json are not a non-empty list of "
                                                          "integers"])
        self.assertEqual(X.read_scores(b"\xff", [10], "A")[1], ["not ASCII"])


# ---------------------------------------------------------------- thresholds.json reader

COUNTS = {"tp": 3, "fp": 1, "fn": 0, "tn": 2}  # any four integers: the reader keeps them, it does not compare them


def entry(cond, seed, t, **kw):
    return {"run_id": rid_of("A" if cond == "B" else "C", seed), "logit": t, "threshold": thr_value(t),
            "cap_met": True, "chosen": dict(COUNTS), **kw}


def six(logits=LOGITS):
    return {k: entry(k[0], k[1], t) for k, t in logits.items()}


def dumps(doc):
    return json.dumps(doc).encode("ascii")


class TestThresholdsReader(unittest.TestCase):
    def read(self, doc, **kw):
        return X.read_thresholds(doc if isinstance(doc, bytes) else dumps(doc), SEEDS, **kw)

    def test_hand_values(self):
        got, info, errs = self.read({"rows": list(six().values())})
        self.assertEqual(errs, [])
        self.assertEqual(sorted(got), sorted((s, c) for c in "BD" for s in SEEDS))
        # sigmoid(-8) = 0.000335350130466..., as float32 0.00033535013790242374; sigmoid(0) is exactly 0.5
        self.assertEqual(got[(42, "B")], {"threshold": 0.00033535013790242374, "logit": -8.0,
                                          "run_id": "s2-roberta-A-s42", "cap_met": True, "selection_counts": COUNTS})
        self.assertEqual(got[(2026, "B")]["threshold"], 0.5)
        self.assertEqual(got[(42, "D")], {"threshold": 0.7941296100616455, "logit": 1.35,
                                          "run_id": "s4-roberta-C-s42", "cap_met": True, "selection_counts": COUNTS})
        # of "chosen" the four counts are kept; cap_met is kept as recorded
        e = six()
        e[("B", 42)]["chosen"] = {"tp": 5, "fp": 0, "fn": 7, "tn": 2, "recall": 1.0, "fpr_exact": "0/2"}
        e[("D", 123)]["cap_met"] = False
        got, info, errs = self.read({"rows": list(e.values())})
        self.assertEqual(errs, [])
        self.assertEqual(got[(42, "B")]["selection_counts"], {"tp": 5, "fp": 0, "fn": 7, "tn": 2})
        self.assertEqual([got[(s, c)]["cap_met"] for c in "BD" for s in SEEDS], [True] * 4 + [False, True])
        self.assertEqual(info, {"entries": 6, "run_file_sha256_checked": 0})

    def test_counts_and_cap_met_are_required(self):
        """Design D5 has select_thresholds.py record them; an entry without them is refused, not skipped."""
        def bad(change, want, drop=()):
            e = six()
            e[("B", 123)] = {k: v for k, v in {**e[("B", 123)], **change}.items() if k not in drop}
            got, _, errs = self.read({"rows": list(e.values())})
            self.assertEqual(errs, ["T03 thresholds.json B s123: " + want], (change, drop))
            return got[(123, "B")]

        counts = "chosen must hold the integer counts tp, fp, fn, tn at the chosen threshold (design D5)"
        self.assertIsNone(bad({}, counts, drop=("chosen",))["selection_counts"])
        for chosen in ({"tp": 3, "fp": 1, "fn": 0}, {"tp": 3, "fp": 1, "fn": 0, "tn": 2.0},
                       {"tp": 3, "fp": 1, "fn": 0, "tn": "2"}, {"tp": True, "fp": 1, "fn": 0, "tn": 2},
                       {"tp": 3, "fp": 1, "fn": 0, "tn": None}, {}, [3, 1, 0, 2], "3,1,0,2", None):
            self.assertIsNone(bad({"chosen": chosen}, counts)["selection_counts"], chosen)
        cap = "cap_met must be true or false (design D5)"
        self.assertIsNone(bad({}, cap, drop=("cap_met",))["cap_met"])
        for v in (1, 0, "true", None):
            self.assertIsNone(bad({"cap_met": v}, cap)["cap_met"], v)
        e = six()  # both at once, in two entries: every one is reported
        del e[("B", 42)]["chosen"], e[("D", 2026)]["cap_met"], e[("D", 2026)]["chosen"]
        _, _, errs = self.read({"rows": list(e.values())})
        self.assertEqual(errs, ["T03 thresholds.json B s42: " + counts, "T03 thresholds.json D s2026: " + counts,
                                "T03 thresholds.json D s2026: " + cap])

    def test_entries_are_found_at_any_depth_by_their_three_keys(self):
        e = six()
        flat, _, errs1 = self.read({"thresholds": list(e.values())})
        nested, _, errs2 = self.read({"seeds": {str(s): {c: {**e[(c, s)], "metrics": {"threshold": 0.5, "fp": 3}}
                                                         for c in "BD"} for s in SEEDS},
                                      "inputs": [{"run_id": rid_of("A", 42), "sha256": "x"}]})
        self.assertEqual((errs1, errs2), ([], []))
        self.assertEqual(flat, nested)

    def test_missing_duplicate_and_unexpected_entries(self):
        e = six()
        _, _, errs = self.read({"rows": [v for k, v in e.items() if k != ("D", 123)]})
        self.assertEqual(len(errs), 1)
        self.assertIn("T02 thresholds.json: no entry", errs[0])
        self.assertIn("s4-roberta-C-s123", errs[0])
        _, _, errs = self.read({"rows": list(e.values()) + [e[("B", 42)]]})
        self.assertEqual(errs, ["T02 thresholds.json B s42: more than one entry for s2-roberta-A-s42"])
        for rid in ("s2-bert-A-s42", "s2-roberta-A-s7", "s4-roberta-C-s42 ", 42):
            _, _, errs = self.read({"rows": list(e.values()) + [{**e[("B", 42)], "run_id": rid}]})
            self.assertEqual(len(errs), 1, rid)
            self.assertIn("T02 thresholds.json: entry for an unexpected run", errs[0])
        # a B entry cannot stand in for D: the run decides the condition
        _, _, errs = self.read({"rows": [{**v, "run_id": rid_of("A", k[1])} for k, v in e.items()]})
        self.assertTrue(any("more than one entry" in x for x in errs))
        self.assertTrue(any("no entry" in x and "s4-roberta-C-s42" in x for x in errs))

    def test_invalid_values(self):
        def bad(change, want):
            e = six()
            e[("B", 123)] = {**e[("B", 123)], **change}
            _, _, errs = self.read({"rows": list(e.values())})
            self.assertEqual(len(errs), 1, change)
            self.assertIn("T03 thresholds.json B s123: " + want, errs[0])

        f64 = 1 / (1 + math.exp(-8.0))  # the float64 sigmoid is not a float32 value
        self.assertNotEqual(f64, thr_value(8.0))
        bad({"threshold": f64}, "threshold is not a float32 value in (0, 1)")
        for v in (1, True, "0.5", None, 0.0, 1.0, 1.5, float("nan")):
            bad({"threshold": v}, "threshold is not a float32 value in (0, 1)")
        bad({"logit": 0.33, "threshold": thr_value(0.33)}, "logit is not a candidate k/20 within -8..8")
        bad({"logit": 8.05, "threshold": thr_value(8.05)}, "logit is not a candidate k/20 within -8..8")
        bad({"logit": "8.0"}, "logit is not a candidate")
        bad({"logit": 7.95}, "threshold is not sigmoid(logit)")  # the threshold of the neighbouring candidate
        bad({"threshold": 0.5}, "threshold is not sigmoid(logit)")
        bad({"seed": 42}, "seed '42' disagrees with the run s2-roberta-A-s123 used here ('123')")
        bad({"seed": "123"}, "seed '123' disagrees")
        bad({"condition": "D"}, "condition 'D' disagrees")

    def test_optional_members_and_recorded_file_hashes(self):
        e = six()
        rows = [{**v, "seed": k[1], "condition": k[0], "selected_step": SEL[("A" if k[0] == "B" else "C", k[1])],
                 "checkpoint_sha256": "c" + v["run_id"]} for k, v in e.items()]

        def runs():
            return {rid_of(t, s): {"selected_step": SEL[(t, s)], "checkpoint_sha256": "c" + rid_of(t, s)}
                    for t in "AC" for s in SEEDS}

        shas = {"runs/s2-roberta-A-s42/run.json": "aa", "runs/s4-roberta-C-s42/predictions_validation.csv": "bb"}
        doc = {"rows": rows, "inputs_sha256": dict(shas, **{"configs/train.json": "zz"})}
        got, info, errs = self.read(doc, runs=runs(), file_shas=shas)
        self.assertEqual(errs, [])
        self.assertEqual(info, {"entries": 6, "run_file_sha256_checked": 2})
        used = runs()
        used["s4-roberta-C-s42"]["selected_step"] = 691  # selected at step 1382, the run now selects 691
        _, _, errs = self.read(doc, runs=used, file_shas=shas)
        self.assertEqual(errs, ["T03 thresholds.json D s42: selected_step '1382' disagrees with the run "
                                "s4-roberta-C-s42 used here ('691')"])
        used = runs()
        used["s2-roberta-A-s2026"]["checkpoint_sha256"] = "another checkpoint"
        _, _, errs = self.read(doc, runs=used, file_shas=shas)
        self.assertEqual(errs, ["T03 thresholds.json B s2026: checkpoint_sha256 'cs2-roberta-A-s2026' disagrees with "
                                "the run s2-roberta-A-s2026 used here ('another checkpoint')"])
        _, _, errs = self.read(doc, file_shas={**shas, "runs/s2-roberta-A-s42/run.json": "ab"})
        self.assertEqual(len(errs), 1)
        self.assertIn("T03 thresholds.json: the recorded SHA-256 of runs/s2-roberta-A-s42/run.json differs", errs[0])

    def test_the_hashes_of_the_run_files_must_be_recorded(self):
        """run.json and predictions_validation.csv of every run used here; metrics.json when it is recorded."""
        rows = list(six().values())
        a, c = "runs/s2-roberta-A-s42/", "runs/s4-roberta-C-s42/"
        used = {a + "run.json": "aa", a + "predictions_validation.csv": "ab", a + "metrics.json": "ac",
                c + "run.json": "ca", c + "predictions_validation.csv": "cb", c + "metrics.json": "cc"}
        need = "; the document must record run.json and predictions_validation.csv of every source run (design D5)"

        def read(recorded):
            _, info, errs = self.read({"rows": rows, "inputs_sha256": recorded}, file_shas=used)
            return errs, info["run_file_sha256_checked"]

        pinned = {k: v for k, v in used.items() if not k.endswith("metrics.json")}
        self.assertEqual(read(used), ([], 6))
        self.assertEqual(read(pinned), ([], 4))  # metrics.json need not be recorded
        self.assertEqual(read({k: v for k, v in pinned.items() if k != c + "predictions_validation.csv"}), ([
            "T03 thresholds.json: no SHA-256 is recorded for 1 run file(s) used here "
            "(runs/s4-roberta-C-s42/predictions_validation.csv)" + need], 3))
        self.assertEqual(read({a + "metrics.json": "ac", c + "metrics.json": "cc"}), ([
            "T03 thresholds.json: no SHA-256 is recorded for 4 run file(s) used here (" + ", ".join(sorted(pinned))
            + ")" + need], 2))
        self.assertEqual(read({**pinned, a + "run.json": 7})[0], [  # not a string: not a recorded SHA-256
            "T03 thresholds.json: no SHA-256 is recorded for 1 run file(s) used here (runs/s2-roberta-A-s42/run.json)"
            + need])
        errs, checked = read({**pinned, c + "metrics.json": "00"})  # recorded, so compared
        self.assertEqual(checked, 5)
        self.assertEqual(len(errs), 1)
        self.assertIn("the recorded SHA-256 of runs/s4-roberta-C-s42/metrics.json differs", errs[0])
        _, _, errs = self.read({"rows": rows})  # no run files given (format checks only): nothing to compare
        self.assertEqual(errs, [])

    def test_unreadable_documents(self):
        for raw in (b"{", b"\xff\xfe", b""):
            self.assertEqual(self.read(raw), (None, None, ["T01 thresholds.json: not valid JSON"]))
        self.assertEqual(self.read(dumps(list(six().values())))[2],
                         ["T01 thresholds.json: the document is not a JSON object"])


# ---------------------------------------------------------------- hand-computed re-thresholding

H_IDS = [sid(i) for i in range(7)]
H_LABELS = [1, 1, 1, 0, 0, 0, 0]
H_CLEAN = [1, 1, 1, 1, 1, 1, 0]  # row 6 (benign, score 0.95) is outside validation_clean
H_SCORES = {"A": [0.9, 0.4, 0.2, 0.3, 0.1, 0.6, 0.95], "C": [0.9, 0.8, 0.45, 0.3, 0.1, 0.2, 0.95]}
H_THR = {("B", 42): -1.0, ("B", 123): 1.0, ("B", 2026): 0.0, ("D", 42): -1.0, ("D", 123): -1.0, ("D", 2026): 1.0}
H_VALROWS = {s: (str(y), f"text {s}") for s, y in zip(H_IDS, H_LABELS)}


def f32(values):
    return torch.tensor(values, dtype=torch.float32).tolist()


def hand_sources():
    """Six source runs with the stored scores of H_SCORES at their selected step (margin = score - 0.5)."""
    out = []
    for t in "AC":
        for s in SEEDS:
            sc = f32(H_SCORES[t])
            out.append({"model": REP, "seed": s, "run_id": rid_of(t, s), "trained": t, "step": 20, "sha256": "x",
                        "size": 1, "ids": H_IDS, "labels": H_LABELS, "in_clean": H_CLEAN,
                        "pred": [int(v >= 0.5) for v in sc], "margin": [v - 0.5 for v in sc], "score": sc})
    return out


H_COUNTS = {  # tp, fp, fn, tn of the validation_clean rows at the entry's threshold (TestRethresholdHand)
    ("B", 42): (2, 2, 1, 1), ("B", 123): (1, 0, 2, 3), ("B", 2026): (1, 1, 2, 2),
    ("D", 42): (3, 1, 0, 2), ("D", 123): (3, 1, 0, 2), ("D", 2026): (2, 0, 1, 3)}


def hand_thresholds():
    """The six entries as read_thresholds returns them; only D seed 123 is recorded as cap not met."""
    return {(s, c): {"threshold": thr_value(t), "logit": t, "run_id": rid_of("A" if c == "B" else "C", s),
                     "cap_met": (c, s) != ("D", 123),
                     "selection_counts": dict(zip(("tp", "fp", "fn", "tn"), H_COUNTS[(c, s)]))}
            for (c, s), t in H_THR.items()}


def hand_p6(orig_scores=None):
    """Re-predicted values of the six source runs: ROB-02 cases on rows 0 (attack) and 3 (benign) with the
    scores 0.6 and 0.28 (A) or 0.9 and 0.1 (C); originals as stored unless orig_scores says otherwise."""
    P6 = {}
    for t in "AC":
        c0, c3 = f32([0.6, 0.28] if t == "A" else [0.9, 0.1])
        o = f32((orig_scores or {}).get(t, [H_SCORES[t][0], H_SCORES[t][3]]))
        for s in SEEDS:
            P6[rid_of(t, s)] = {"case": {"dev-rob02-000000": (c0 - 0.5, c0, int(c0 >= 0.5)),
                                         "dev-rob02-000003": (c3 - 0.5, c3, int(c3 >= 0.5))},
                                "orig": {sid(0): (o[0] - 0.5, o[0], int(o[0] >= 0.5)),
                                         sid(3): (o[1] - 0.5, o[1], int(o[1] >= 0.5))}}
    return P6


def hand_recs():
    rows = [case_row(f"dev-rob02-{i:06d}", "ROB-02", "paraphrase", f"p{i}", src=sid(i), val=H_VALROWS) for i in (0, 3)]
    return select(rows, H_VALROWS)


def select(case_rows, val, relabel=None):
    """The records eval_cases selects; relabel: {case_id: new label} for review decisions "relabel"."""
    relabel = relabel or {}
    cases, errs = E.check_cases(case_rows, val)
    assert not errs, errs
    rows = [review_row(c, "relabel", new_label=relabel[c["case_id"]], reason="r") if c["case_id"] in relabel
            else review_row(c) for c in case_rows]
    review, errs = E.check_review(table_bytes(E.REVIEW_COLUMNS, rows), cases)
    assert not errs, errs
    recs, info, errs = E.select_cases(cases, review, val)
    assert not errs, errs
    assert info["relabeled"] == sorted(relabel), info["relabeled"]
    return recs


class TestRethresholdHand(unittest.TestCase):
    """validation_clean rows 0-5 (attack, attack, attack, benign, benign, benign); thresholds float32
    sigmoid(-1) = 0.2689..., sigmoid(1) = 0.7310..., 0.5.

    A scores 0.9 0.4 0.2 | 0.3 0.1 0.6:  at 0.5    1 0 0 | 0 0 1  TP 1 FN 2 FP 1 TN 2
                                         at 0.2689 1 1 0 | 1 0 1  TP 2 FN 1 FP 2 TN 1   (B seed 42)
                                         at 0.7311 1 0 0 | 0 0 0  TP 1 FN 2 FP 0 TN 3   (B seed 123)
    C scores 0.9 0.8 0.45 | 0.3 0.1 0.2: at 0.5    1 1 0 | 0 0 0  TP 2 FN 1 FP 0 TN 3
                                         at 0.2689 1 1 1 | 1 0 0  TP 3 FN 0 FP 1 TN 2   (D seeds 42, 123)
                                         at 0.7311 1 1 0 | 0 0 0  TP 2 FN 1 FP 0 TN 3   (D seed 2026)
    """

    def setUp(self):
        self.cruns = X.condition_runs(SEEDS, hand_thresholds())
        self.P, self.checks, self.errs = X.rethreshold(self.cruns, hand_sources(), hand_p6())
        val_items = [(s, y, ("val", s)) for s, y, c in zip(H_IDS, H_LABELS, H_CLEAN) if c]
        self.res = X.evaluate(hand_recs(), self.cruns, self.P, val_items, {})

    def test_condition_runs(self):
        self.assertEqual([(r["model"], r["seed"]) for r in self.cruns], [(c, s) for c in "ABCD" for s in SEEDS])
        self.assertEqual([r["source"] for r in self.cruns],
                         [rid_of("A", s) for s in SEEDS] * 2 + [rid_of("C", s) for s in SEEDS] * 2)
        self.assertEqual([r["threshold"] for r in self.cruns],
                         [0.5] * 3 + [T_LO, T_HI, 0.5] + [0.5] * 3 + [T_LO, T_LO, T_HI])
        self.assertEqual([r["logit"] for r in self.cruns],
                         [None] * 3 + [-1.0, 1.0, 0.0] + [None] * 3 + [-1.0, -1.0, 1.0])
        # the recorded cap_met of the tuned thresholds; A and C have no threshold selection and no cap
        self.assertEqual([r["cap_met"] for r in self.cruns], [None] * 3 + [True] * 3 + [None] * 3 + [True, False, True])
        self.assertEqual(len({r["run_id"] for r in self.cruns}), 12)

    def val_preds(self, key):
        return [self.P[key]["val"][s][2] for s in H_IDS[:6]]

    def test_validation_predictions(self):
        self.assertEqual(self.errs, [])
        self.assertEqual(self.val_preds("A-s42"), [1, 0, 0, 0, 0, 1])
        self.assertEqual(self.val_preds("B-s42"), [1, 1, 0, 1, 0, 1])
        self.assertEqual(self.val_preds("B-s123"), [1, 0, 0, 0, 0, 0])
        self.assertEqual(self.val_preds("B-s2026"), [1, 0, 0, 0, 0, 1])
        self.assertEqual(self.val_preds("C-s123"), [1, 1, 0, 0, 0, 0])
        self.assertEqual(self.val_preds("D-s42"), [1, 1, 1, 1, 0, 0])
        self.assertEqual(self.val_preds("D-s2026"), [1, 1, 0, 0, 0, 0])
        for key in self.P:  # the row outside validation_clean is not an evaluation row
            self.assertEqual(sorted(self.P[key]["val"]), H_IDS[:6], key)
        self.assertEqual(self.P["B-s42"]["val"][sid(1)][:2], (f32([0.4])[0] - 0.5, f32([0.4])[0]))  # stored values

    def test_case_predictions_use_the_condition_threshold(self):
        case = {k: [self.P[k]["case"][c][2] for c in ("dev-rob02-000000", "dev-rob02-000003")] for k in self.P}
        self.assertEqual(case["A-s42"], [1, 0])  # scores 0.6, 0.28 at 0.5
        self.assertEqual(case["B-s42"], [1, 1])  # at 0.2689
        self.assertEqual(case["B-s123"], [0, 0])  # at 0.7311
        self.assertEqual(case["B-s2026"], [1, 0])
        self.assertEqual(case["D-s42"], [1, 0])  # scores 0.9, 0.1
        self.assertEqual(self.P["B-s123"]["case"]["dev-rob02-000000"][:2], self.P["A-s123"]["case"][
            "dev-rob02-000000"][:2])  # same checkpoint: same margin and score
        self.assertEqual(self.checks["B-s42"], {"originals": 2, "stored_vs_repredicted_flips": 0})

    def row(self, cond, seed, es="validation_clean", side="original", label="all"):
        got = [r for r in self.res["metrics"] if (r["model"], r["seed"], r["condition"], r["side"], r["label"])
               == (cond, seed, es, side, label)]
        self.assertEqual(len(got), 1, (cond, seed, es, side, label))
        return got[0]

    def stat(self, cond, q, es="validation_clean", side="original", label="all"):
        got = [r for r in self.res["model_rows"] if (r["model"], r["condition"], r["side"], r["label"], r["quantity"])
               == (cond, es, side, label, q)]
        self.assertEqual(len(got), 1, (cond, q))
        return got[0]

    def test_validation_clean_metrics(self):
        want = {("A", 42): (1, 1, 2, 2), ("B", 42): (2, 2, 1, 1), ("B", 123): (1, 0, 2, 3), ("B", 2026): (1, 1, 2, 2),
                ("C", 42): (2, 0, 1, 3), ("D", 42): (3, 1, 0, 2), ("D", 123): (3, 1, 0, 2), ("D", 2026): (2, 0, 1, 3)}
        for (cond, seed), counts in want.items():
            r = self.row(cond, seed)
            self.assertEqual((r["n"], r["tp"], r["fp"], r["fn"], r["tn"]), (6,) + counts, (cond, seed))
        r = self.row("B", 42)
        for k, v in (("recall", 2 / 3), ("fpr", 2 / 3), ("precision", 0.5), ("f1", 4 / 7)):
            self.assertAlmostEqual(r[k], v, places=12, msg=k)
        r = self.row("D", 42)
        for k, v in (("recall", 1.0), ("fpr", 1 / 3), ("precision", 0.75), ("f1", 6 / 7)):
            self.assertAlmostEqual(r[k], v, places=12, msg=k)
        # ROC-AUC is on the margin, so it does not depend on the threshold: A and B 6/9 (0.9 is above three
        # benign rows, 0.4 above two, 0.2 above one), C and D 9/9
        for cond, auc in (("A", 2 / 3), ("B", 2 / 3), ("C", 1.0), ("D", 1.0)):
            for s in SEEDS:
                self.assertAlmostEqual(self.row(cond, s)["roc_auc"], auc, places=12, msg=(cond, s))

    def test_seed_mean_and_std(self):
        st = self.stat("B", "recall")  # 2/3, 1/3, 1/3
        self.assertAlmostEqual(st["mean"], 4 / 9, places=12)
        self.assertAlmostEqual(st["std"], math.sqrt(1 / 27), places=12)  # sample std (n - 1)
        self.assertEqual((st["s42"], st["s123"], st["n"]), (2 / 3, 1 / 3, 6))
        self.assertAlmostEqual(self.stat("B", "fpr")["mean"], 1 / 3, places=12)  # 2/3, 0, 1/3
        self.assertAlmostEqual(self.stat("B", "f1")["mean"], (4 / 7 + 1 / 2 + 2 / 5) / 3, places=12)
        self.assertAlmostEqual(self.stat("D", "f1")["mean"], (6 / 7 + 6 / 7 + 4 / 5) / 3, places=12)
        self.assertEqual((self.stat("A", "fp")["mean"], self.stat("A", "fp")["std"]), (1.0, 0.0))

    def test_case_metrics_and_change_rate(self):
        self.assertEqual([self.row("B", s, "ROB-02", "variant", "1")["recall"] for s in SEEDS], [1.0, 0.0, 1.0])
        self.assertEqual([self.row("B", s, "ROB-02", "variant", "0")["fpr"] for s in SEEDS], [1.0, 0.0, 0.0])
        # originals (rows 0 and 3, stored = re-predicted scores 0.9 / 0.3 in A): B seed 42 flags row 3
        self.assertEqual([self.row("B", s, "ROB-02", "original", "0")["fpr"] for s in SEEDS], [1.0, 0.0, 0.0])
        ch = {(r["model"], r["seed"]): r for r in self.res["changes"] if r["label"] == "all"}
        # B seed 123: original row 0 detected (0.9), its case missed (0.6 < 0.7311): one changed pair of two
        self.assertEqual((ch[("B", 123)]["changed"], ch[("B", 123)]["to_benign"], ch[("B", 123)]["change_rate"]),
                         (1, 1, 0.5))
        self.assertEqual(ch[("B", 42)]["changed"], 0)

    def test_validation_bootstrap_rows(self):
        b = {(r["condition_a"], r["condition_b"], r["eval_set"], r["metric"]): r for r in self.res["bootstrap"]}
        r = b[("A", "B", "validation_clean", "recall")]
        self.assertEqual((r["kind"], r["side"], r["subtype"], r["n_groups"], r["strata"], r["nk_merges"]),
                         ("primary", "original", "all", 6, "0:3;1:3", 0))
        self.assertAlmostEqual(r["estimate_a"], 1 / 3, places=12)
        self.assertAlmostEqual(r["estimate_b"], 4 / 9, places=12)
        self.assertAlmostEqual(r["diff"], 1 / 9, places=12)  # b - a
        r = b[("A", "D", "validation_clean", "f1")]
        self.assertAlmostEqual(r["estimate_a"], 0.4, places=12)
        self.assertAlmostEqual(r["estimate_b"], (6 / 7 + 6 / 7 + 4 / 5) / 3, places=12)
        self.assertLessEqual(r["diff_low"], r["diff"])
        self.assertGreaterEqual(r["diff_high"], r["diff"])
        self.assertEqual(b[("B", "C", "validation_clean", "fpr")]["kind"], "secondary")
        self.assertAlmostEqual(b[("B", "C", "validation_clean", "fpr")]["diff"], 0 - 1 / 3, places=12)

    def test_counts_recorded_by_the_selection_must_be_the_counts_here(self):
        thr = hand_thresholds()  # the counts of the class docstring: every one of the six entries is compared
        self.assertEqual(thr[(42, "B")]["selection_counts"], {"tp": 2, "fp": 2, "fn": 1, "tn": 1})  # A at 0.2689
        self.assertEqual(thr[(2026, "D")]["selection_counts"], {"tp": 2, "fp": 0, "fn": 1, "tn": 3})  # C at 0.7311
        self.assertEqual(X.selection_count_errors(thr, hand_sources()), ([], 6))
        # with the row outside validation_clean (benign, 0.95) the counts would be fp 3 / fp 1
        thr[(123, "B")]["selection_counts"] = {"tp": 1, "fp": 1, "fn": 2, "tn": 3}
        thr[(42, "D")]["selection_counts"] = {"tp": 2, "fp": 0, "fn": 1, "tn": 3}  # the counts at 0.5, not 0.2689
        errs, checked = X.selection_count_errors(thr, hand_sources())
        self.assertEqual(checked, 6)
        self.assertEqual(errs, [
            "T03 thresholds.json B s123: the recorded counts at the chosen threshold (tp 1, fp 1, fn 2, tn 3) differ "
            "from the validation_clean counts of s2-roberta-A-s123 here (tp 1, fp 0, fn 2, tn 3)",
            "T03 thresholds.json D s42: the recorded counts at the chosen threshold (tp 2, fp 0, fn 1, tn 3) differ "
            "from the validation_clean counts of s4-roberta-C-s42 here (tp 3, fp 1, fn 0, tn 2)"])
        self.assertEqual(X.selection_count_errors(thr, hand_sources()[:3])[1], 3)  # the C runs did not load

    def test_stored_and_repredicted_scores_must_agree_at_the_threshold(self):
        # the original row 3 is stored as 0.3 but re-predicted as 0.26: same decision at 0.5 and at 0.7311,
        # another one at 0.2689 (B seed 42 only)
        P, checks, errs = X.rethreshold(self.cruns, hand_sources(), hand_p6({"A": [0.9, 0.26]}))
        self.assertEqual(len(errs), 1)
        self.assertIn("M08 s2-roberta-A-s42 (condition B, threshold 0.2689414322376251): 1 of 2 case originals",
                      errs[0])
        self.assertEqual(checks["B-s42"]["stored_vs_repredicted_flips"], 1)
        self.assertEqual(checks["B-s123"]["stored_vs_repredicted_flips"], 0)


# ---------------------------------------------------------------- hand-computed evaluation scenarios

E_VAL = {sid(0): ("1", "o0"), sid(1): ("1", "o1"), sid(2): ("0", "o2"), sid(3): ("0", "o3")}
E_ITEMS = [(s, int(y), ("val", s)) for s, (y, _) in E_VAL.items()]
ATTACK = (0, 1)  # originals with label 1


def triple(p):
    return (1.0 if p else -1.0, 0.75 if p else 0.25, p)


def e_cases():
    """ROB-01 (four subtypes) on originals 0, 1 (attack) and 2 (benign); ROB-02 on original 1; two ROB-04."""
    rows = [case_row(f"dev-rob01-{sub}-{i:06d}", "ROB-01", sub, f"{sub}{i}", src=sid(i), val=E_VAL)
            for i in range(3) for sub in E.SUBTYPES["ROB-01"]]
    rows.append(case_row("dev-rob02-000001", "ROB-02", "paraphrase", "p1", src=sid(1), val=E_VAL))
    rows += [case_row(f"dev-rob04-{j:03d}", "ROB-04", "benign_security", f"b{j}") for j in (1, 2)]
    return rows


def e_eval(case_pred, val_pred=None, val_nk=None, cases=None, relabel=None):
    """X.evaluate on hand-written predictions. case_pred(cond, seed, rec) and val_pred(cond, seed,
    sentence_id) give 0/1; originals and (by default) validation rows are predicted correctly."""
    recs = select(cases or e_cases(), E_VAL, relabel)
    cruns = X.condition_runs(SEEDS, {(s, c): {"threshold": T_LO, "logit": -1.0} for s in SEEDS for c in "BD"})
    P = {}
    for cr in cruns:
        c, s = cr["model"], cr["seed"]
        P[cr["run_id"]] = {
            "case": {r["case_id"]: triple(case_pred(c, s, r)) for r in recs},
            "orig": {x: triple(int(E_VAL[x][0])) for x in E_VAL},
            "val": {x: triple(val_pred(c, s, x) if val_pred else int(E_VAL[x][0])) for x in E_VAL}}
    return recs, cruns, P, X.evaluate(recs, cruns, P, E_ITEMS, val_nk or {})


def k_pred(cond, seed, r):
    """A and B: detect only original 0's attack variants, flag the benign homoglyph case and ROB-04 case 1.
    C and D: detect every attack variant, flag the benign main variants, nothing else."""
    base = cond in "AB"
    if r["condition"] == "ROB-04":
        return int(base and r["case_id"] == "dev-rob04-001")
    if r["label"] == 1:
        return int(not base or r["source_id"] == sid(0))
    return int(base == (r["subtype"] == "homoglyph"))


class TestEvaluateHand(unittest.TestCase):
    """Scenario k_pred. Groups: originals 0 and 1 (attack), 2 (benign). With k = draws of original 1 among
    the two attack draws of a resample (0, 1, 2 with 1/4, 1/2, 1/4):
      ROB-01 main recall: A (2 - k) * 3 / 6 = 1 - k/2, C 1, so C - A = k/2 (0.5 on the full sample).
      Pooled ROB-01~03 main: original 0 has 3 attack cases, original 1 has 3 + the ROB-02 case = 4. A detects
      3 (2 - k) of 6 + k, so C - A = 1 - (6 - 3k) / (6 + k) = 4k / (6 + k): 0, 4/7, 1 (4/7 on the full sample).
    """

    @classmethod
    def setUpClass(cls):
        cls.recs, cls.cruns, cls.P, cls.res = e_eval(k_pred)
        cls.boot = {(r["condition_a"], r["condition_b"], r["eval_set"], r["subtype"], r["metric"]): r
                    for r in cls.res["bootstrap"]}

    def test_cells_and_pooled_metrics(self):
        self.assertEqual(self.res["cells"], [
            ("validation_clean", "all", "all"), ("ROB-01", "case", "all"), ("ROB-01", "space", "all"),
            ("ROB-01", "char", "all"), ("ROB-01", "main", "all"), ("ROB-01", "homoglyph", "all"),
            ("ROB-02", "paraphrase", "all"), ("ROB-04", "benign_security", "all"), ("ROB-01~03", "main", "all")])
        rows = {(r["model"], r["side"], r["label"]): r for r in self.res["metrics"]
                if r["condition"] == "ROB-01~03" and r["seed"] == 42}
        r = rows[("A", "variant", "1")]  # 3 + 3 main ROB-01 attack cases + 1 ROB-02; homoglyph left out
        self.assertEqual((r["n"], r["tp"], r["fn"]), (7, 3, 4))
        self.assertAlmostEqual(r["recall"], 3 / 7, places=12)
        self.assertEqual((rows[("C", "variant", "1")]["n"], rows[("C", "variant", "1")]["recall"]), (7, 1.0))
        r = rows[("C", "variant", "0")]  # the three main variants of the benign original
        self.assertEqual((r["n"], r["fp"], r["fpr"]), (3, 3, 1.0))
        self.assertEqual(rows[("A", "original", "1")]["n"], 2)  # each original once

    def test_rows_of_the_eval_cases_cells_equal_eval_cases(self):
        own = [r for r in self.res["metrics"] if r["condition"] not in ("validation_clean", "ROB-01~03")]
        self.assertEqual(own, E.metric_rows(self.recs, self.cruns, self.P))
        self.assertEqual(self.res["changes"], E.change_rows(self.recs, self.cruns, self.P))

    def test_comparisons(self):
        pairs = list(dict.fromkeys((r["condition_a"], r["condition_b"]) for r in self.res["bootstrap"]
                                   if r["kind"] != "level"))
        self.assertEqual(pairs, PAIRS)
        kind = {(r["condition_a"], r["condition_b"]): r["kind"] for r in self.res["bootstrap"] if r["kind"] != "level"}
        self.assertEqual(kind, {**{p: "primary" for p in PAIRS[:3]}, **{p: "secondary" for p in PAIRS[3:]}})
        per_pair = [(r["eval_set"], r["subtype"], r["side"], r["metric"]) for r in self.res["bootstrap"]
                    if (r["condition_a"], r["condition_b"]) == ("A", "C")]
        self.assertEqual(per_pair, [
            ("validation_clean", "all", "original", "recall"), ("validation_clean", "all", "original", "fpr"),
            ("validation_clean", "all", "original", "f1"),
            ("ROB-01", "case", "variant", "recall"), ("ROB-01", "case", "variant", "fpr"),
            ("ROB-01", "space", "variant", "recall"), ("ROB-01", "space", "variant", "fpr"),
            ("ROB-01", "char", "variant", "recall"), ("ROB-01", "char", "variant", "fpr"),
            ("ROB-01", "main", "variant", "recall"), ("ROB-01", "main", "variant", "fpr"),
            ("ROB-01", "homoglyph", "variant", "recall"), ("ROB-01", "homoglyph", "variant", "fpr"),
            ("ROB-02", "paraphrase", "variant", "recall"), ("ROB-04", "benign_security", "variant", "fpr"),
            ("ROB-01~03", "main", "variant", "recall"), ("ROB-01~03", "main", "variant", "fpr")])

    def test_paired_difference_closed_form(self):
        r = self.boot[("A", "C", "ROB-01", "main", "recall")]
        self.assertEqual((r["estimate_a"], r["estimate_b"], r["diff"], r["n_groups"], r["strata"]),
                         (0.5, 1.0, 0.5, 3, "0:1;1:2"))
        self.assertEqual((r["diff_low"], r["diff_high"], r["ci_vs_0"]), (0.0, 1.0, "includes_0"))  # k = 0, k = 2
        r = self.boot[("A", "C", "ROB-01~03", "main", "recall")]
        self.assertAlmostEqual(r["estimate_a"], 3 / 7, places=12)
        self.assertAlmostEqual(r["diff"], 4 / 7, places=12)
        self.assertEqual((r["diff_low"], r["diff_high"], r["n_groups"], r["strata"]), (0.0, 1.0, 3, "0:1;1:2"))
        # the whole distribution: both conditions are evaluated on the same resampled groups
        recs = X.pooled_main(self.recs)
        items = E.side_items(recs, "variant")
        stratum = {sid(0): "1", sid(1): "1", sid(2): "0"}
        d = E.bootstrap_dist(items, ["A-s42", "A-s123", "A-s2026"], items, ["C-s42", "C-s123", "C-s2026"],
                             "recall", stratum, self.P)
        k = (E.resample_indices(["1", "1", "0"]) == 1).sum(axis=1)
        self.assertTrue(np.allclose(d["diff"], 4 * k / (6 + k)))
        # direction: b - a, and the secondary pair B -> C has the numbers of A -> C here (B = A)
        r = self.boot[("B", "C", "ROB-01~03", "main", "recall")]
        self.assertAlmostEqual(r["diff"], 4 / 7, places=12)
        self.assertEqual(self.boot[("A", "B", "ROB-01~03", "main", "recall")]["diff"], 0.0)
        self.assertEqual(self.boot[("C", "D", "ROB-01", "main", "fpr")]["ci_vs_0"], "includes_0")

    def test_benign_fpr_rows_and_rob04_level(self):
        r = self.boot[("A", "C", "ROB-01", "main", "fpr")]  # one benign group, always drawn: 0 -> 1
        self.assertEqual((r["diff"], r["diff_low"], r["diff_high"], r["ci_vs_0"]), (1.0, 1.0, 1.0, ">0"))
        r = self.boot[("A", "C", "ROB-01", "homoglyph", "fpr")]
        self.assertEqual((r["diff"], r["diff_low"], r["diff_high"], r["ci_vs_0"]), (-1.0, -1.0, -1.0, "<0"))
        r = self.boot[("A", "C", "ROB-04", "benign_security", "fpr")]  # j draws of case 1: A j/2, C 0
        self.assertEqual((r["estimate_a"], r["estimate_b"], r["diff"], r["diff_low"], r["diff_high"], r["ci_vs_0"]),
                         (0.5, 0.0, -0.5, -1.0, 0.0, "includes_0"))
        level = [r for r in self.res["bootstrap"] if r["kind"] == "level"]
        self.assertEqual([(r["condition_a"], r["condition_b"], r["eval_set"], r["metric"]) for r in level],
                         [(c, "", "ROB-04", "fpr") for c in "ABCD"])
        self.assertEqual([(r["estimate_a"], r["a_low"], r["a_high"], r["diff"], r["ci_vs_0"]) for r in level[:3]],
                         [(0.5, 0.0, 1.0, None, ""), (0.5, 0.0, 1.0, None, ""), (0.0, 0.0, 0.0, None, "")])

    def test_verdicts_of_this_scenario(self):
        v = {x["comparison"]: x for x in self.res["verdicts"]["rq04_development"]}
        self.assertEqual(list(v), ["A->B", "A->C", "A->D"])
        # A -> C: +4/7 on the pooled attack variants, but the CI lower bound is 0, so (1) is not met
        self.assertAlmostEqual(v["A->C"]["recall_diff"], 4 / 7, places=12)
        self.assertEqual((v["A->C"]["recall_diff_low"], v["A->C"]["c1_recall"], v["A->C"]["c2_fpr"],
                          v["A->C"]["c3_f1"], v["A->C"]["code"], v["A->C"]["text"]),
                         (0.0, False, True, True, "not_confirmed", "개선 확인 못함"))
        self.assertEqual(v["A->B"]["code"], "not_confirmed")
        f = {(x["comparison"], x["eval_set"], x["subtype"]): (x["kind"], x["code"], x["text"])
             for x in self.res["verdicts"]["benign_fpr_change"]}
        self.assertEqual(len(f), 6 * 3)
        self.assertEqual(f[("A->C", "ROB-01", "main")], ("primary", "fpr_increase", "오탐 증가 확인"))
        self.assertEqual(f[("A->C", "ROB-01", "homoglyph")], ("primary", "fpr_decrease", "오탐 감소 확인"))
        self.assertEqual(f[("A->C", "ROB-04", "benign_security")], ("primary", "no_change", "변화 확인 못함"))
        self.assertEqual(f[("B->D", "ROB-01", "homoglyph")], ("secondary", "fpr_decrease", "오탐 감소 확인"))
        self.assertEqual(f[("A->B", "ROB-01", "main")][1], "no_change")

    def test_validation_rows_with_one_normalised_key_are_one_group(self):
        nk = {("val", sid(0)): "same key", ("val", sid(1)): "same key", ("val", sid(2)): "other", ("val", sid(3)): ""}
        _, _, _, res = e_eval(k_pred, val_nk=nk)
        r = next(x for x in res["bootstrap"] if x["eval_set"] == "validation_clean" and x["metric"] == "f1")
        self.assertEqual((r["n_groups"], r["strata"], r["nk_merges"]), (3, "0:2;1:1", 1))
        self.assertEqual(res["grouping"]["nk_merged_group_sets"], [[sid(0), sid(1)]])
        self.assertEqual(res["grouping"]["comparisons_with_nk_merges"], 6 * 3)  # recall, FPR, F1 of every pair
        self.assertEqual(self.res["grouping"], {"nk_merged_group_sets": [], "comparisons_with_nk_merges": 0})

    def test_cases_with_one_normalised_key_are_one_group(self):
        """The paraphrases of the attack originals 0 and 1 share a normalised key, so the two originals are
        one group. A detects the first only, C both. Every resample then holds the merged attack group and
        the benign group once: C - A is 1 - 1/2 in all of them. Unmerged, the CI would be [0, 1]."""
        texts = ["Hello   World", "hello world", "other text"]
        rows = [case_row(f"dev-rob02-{i:06d}", "ROB-02", "paraphrase", t, src=sid(i), val=E_VAL)
                for i, t in enumerate(texts)]
        _, _, _, res = e_eval(lambda c, s, r: int(r["label"] == 1 and (c in "CD" or r["source_id"] == sid(0))),
                              cases=rows)
        for es, sub in (("ROB-02", "paraphrase"), ("ROB-01~03", "main")):
            r = next(x for x in res["bootstrap"] if (x["condition_a"], x["condition_b"], x["eval_set"], x["subtype"],
                                                     x["metric"]) == ("A", "C", es, sub, "recall"))
            self.assertEqual((r["n_groups"], r["strata"], r["nk_merges"]), (2, "0:1;1:1", 1), es)
            self.assertEqual((r["estimate_a"], r["diff"], r["diff_low"], r["diff_high"], r["ci_vs_0"]),
                             (0.5, 0.5, 0.5, 0.5, ">0"), es)
        self.assertEqual(res["grouping"]["nk_merged_group_sets"], [[sid(0), sid(1)]])

    def test_a_relabelled_case_stays_in_the_stratum_of_its_original(self):
        """README 6.1 item 2. ROB-02 cases of the attack originals 0 and 1 and of the benign original 2; the
        review relabels the case of original 1 as benign. Its group stays in the attack stratum (two
        groups, two draws) and the case counts as a benign case. With k = draws of original 1 (0, 1, 2 with
        1/4, 1/2, 1/4) a resample holds k relabelled cases and the benign case of original 2. A and B flag
        the relabelled case only: FPR k / (k + 1); C and D flag no benign case: C - A = -k / (k + 1), that
        is 0, -1/2 or -2/3 (-1/2 on the full sample). The only attack case is the one of original 0, which
        every condition detects; it is absent when k = 2, where the recall is undefined.
        Stratified by the case label, original 1 would be drawn with original 2 instead (FPR j / 2)."""
        rows = [case_row(f"dev-rob02-{i:06d}", "ROB-02", "paraphrase", f"p{i}", src=sid(i), val=E_VAL)
                for i in range(3)]
        recs, _, _, res = e_eval(
            lambda c, s, r: int(r["source_id"] == sid(0) or (c in "AB" and r["source_id"] == sid(1))),
            cases=rows, relabel={"dev-rob02-000001": "0"})
        self.assertEqual([(r["group_id"], r["stratum"], r["label"]) for r in recs],
                         [(sid(0), "1", 1), (sid(1), "1", 0), (sid(2), "0", 0)])
        boot = {(r["condition_a"], r["condition_b"], r["eval_set"], r["subtype"], r["metric"]): r
                for r in res["bootstrap"]}
        # the draws of README 6.1 made here with numpy: the strata in sorted order, "0" (one group) first
        rng = np.random.default_rng(314159)
        rng.integers(0, 1, size=(2000, 1))
        k = rng.integers(0, 2, size=(2000, 2)).sum(axis=1)
        low, high = np.percentile(-k / (k + 1), [2.5, 97.5])
        self.assertEqual((float(low), float(high)), (-2 / 3, 0.0))
        for es, sub in (("ROB-02", "paraphrase"), ("ROB-01~03", "main")):
            r = boot[("A", "C", es, sub, "fpr")]
            self.assertEqual((r["n_groups"], r["strata"], r["nk_merges"], r["undefined_resamples"]),
                             (3, "0:1;1:2", 0, 0), es)
            self.assertEqual((r["estimate_a"], r["estimate_b"], r["diff"]), (0.5, 0.0, -0.5), es)
            self.assertAlmostEqual(r["diff_low"], -2 / 3, places=12, msg=es)
            self.assertEqual((r["diff_high"], r["ci_vs_0"]), (0.0, "includes_0"), es)
            r = boot[("A", "C", es, sub, "recall")]
            self.assertEqual((r["n_groups"], r["strata"], r["estimate_a"], r["estimate_b"], r["diff"]),
                             (3, "0:1;1:2", 1.0, 1.0, 0.0), es)
            self.assertEqual(r["undefined_resamples"], int((k == 2).sum()), es)
            self.assertGreater(r["undefined_resamples"], 400, es)  # about a quarter of the 2,000 resamples
            self.assertEqual((r["diff_low"], r["diff_high"], r["ci_vs_0"]), (None, None, ""), es)
        # the metrics use the case label: the relabelled case is one of the two benign cases
        m = {(r["model"], r["label"]): r for r in res["metrics"]
             if (r["condition"], r["side"], r["seed"]) == ("ROB-02", "variant", 42)}
        self.assertEqual((m[("A", "0")]["n"], m[("A", "0")]["fp"], m[("A", "0")]["fpr"]), (2, 1, 0.5))
        self.assertEqual((m[("A", "1")]["n"], m[("A", "1")]["tp"], m[("A", "1")]["recall"]), (1, 1, 1.0))
        # several cases of one original, one of them relabelled: the group keeps the stratum of its original
        # (by the case labels it would be a group of mixed labels)
        _, _, _, res = e_eval(k_pred, relabel={"dev-rob01-char-000001": "0"})
        rows = [r for r in res["bootstrap"] if r["eval_set"] in ("ROB-01", "ROB-01~03")]
        self.assertEqual(len(rows), 6 * (5 * 2 + 2))  # five ROB-01 cells and the pooled cell, recall and FPR
        self.assertEqual({(r["n_groups"], r["strata"]) for r in rows}, {(3, "0:1;1:2")})


class TestVerdictScenario(unittest.TestCase):
    """A and B miss every attack variant, C and D detect all (every resample: recall 0 against 1). On the
    four validation_clean rows A, B and C are right; D also flags both benign rows: FPR 0 -> 1 and F1 1 ->
    2 * 2 / (2 * 2 + 2 + 0) = 2/3 in every resample (two attack and two benign groups are always drawn)."""

    def test_three_verdicts(self):
        _, _, _, res = e_eval(lambda c, s, r: int(c in "CD" and r["label"] == 1),
                              lambda c, s, x: int(E_VAL[x][0] == "1" or c == "D"))
        v = {x["comparison"]: x for x in res["verdicts"]["rq04_development"]}
        self.assertEqual({k: (x["c1_recall"], x["c2_fpr"], x["c3_f1"], x["code"], x["text"]) for k, x in v.items()}, {
            "A->B": (False, True, True, "not_confirmed", "개선 확인 못함"),
            "A->C": (True, True, True, "confirmed", "개선 확인"),
            "A->D": (True, False, False, "tradeoff", "상충 있는 개선")})
        self.assertEqual((v["A->C"]["recall_diff"], v["A->C"]["recall_diff_low"], v["A->C"]["recall_diff_high"]),
                         (1.0, 1.0, 1.0))
        self.assertEqual((v["A->D"]["fpr_increase"], v["A->D"]["fpr_increase_low"], v["A->D"]["fpr_increase_high"]),
                         (1.0, 1.0, 1.0))
        for k in ("f1_decrease", "f1_decrease_low", "f1_decrease_high"):  # decrease = A - D = 1 - 2/3
            self.assertAlmostEqual(v["A->D"][k], 1 / 3, places=12, msg=k)
        self.assertEqual((v["A->C"]["f1_decrease"], v["A->C"]["fpr_increase"]), (0.0, 0.0))


def brow(diff, low, high):
    return {"diff": diff, "diff_low": low, "diff_high": high}


class TestVerdictRules(unittest.TestCase):
    OK_FPR, OK_F1 = brow(0.0, -0.01, 0.01), brow(0.0, -0.005, 0.005)

    def code(self, recall, fpr=OK_FPR, f1=OK_F1):
        v = X.rq04_verdict(recall, fpr, f1)
        return v["code"], (v["c1_recall"], v["c2_fpr"], v["c3_f1"])

    def test_rule_1_recall_difference_and_lower_bound(self):
        self.assertEqual(self.code(brow(0.12, 0.02, 0.2)), ("confirmed", (True, True, True)))
        self.assertEqual(self.code(brow(0.09, 0.02, 0.2)), ("not_confirmed", (False, True, True)))  # below 10%p
        self.assertEqual(self.code(brow(0.30, 0.0, 0.5)), ("not_confirmed", (False, True, True)))  # lower bound 0
        self.assertEqual(self.code(brow(0.30, -0.01, 0.5))[0], "not_confirmed")
        self.assertEqual(self.code(brow(0.9 - 0.8, 0.02, 0.2))[0], "confirmed")  # 0.09999999999999998 is 10%p
        self.assertLess(0.9 - 0.8, 0.10)
        self.assertEqual(self.code(brow(0.0999, 0.02, 0.2))[0], "not_confirmed")
        self.assertEqual(self.code(brow(0.30, 1e-12, 0.5))[0], "not_confirmed")  # rounding noise is not > 0
        self.assertEqual(self.code(brow(0.30, None, None))[0], "not_confirmed")  # undefined CI
        self.assertEqual(self.code(None), ("not_confirmed", (False, True, True)))
        self.assertEqual(self.code(brow(-0.2, -0.3, -0.1))[0], "not_confirmed")

    def test_rule_2_fpr_increase_upper_bound(self):
        good = brow(0.12, 0.02, 0.2)
        self.assertEqual(self.code(good, brow(0.01, 0.0, 0.02)), ("confirmed", (True, True, True)))  # exactly 2%p
        self.assertEqual(self.code(good, brow(0.01, 0.0, 0.0201)), ("tradeoff", (True, False, True)))
        self.assertEqual(self.code(good, brow(-0.05, -0.08, -0.02))[0], "confirmed")  # FPR went down
        self.assertEqual(self.code(good, brow(0.01, None, None)), ("tradeoff", (True, False, True)))
        self.assertEqual(self.code(good, None), ("tradeoff", (True, False, True)))
        self.assertEqual(self.code(brow(0.05, 0.02, 0.2), brow(0.05, 0.03, 0.08)),
                         ("not_confirmed", (False, False, True)))  # without (1) it is never a tradeoff

    def test_rule_3_f1_decrease_upper_bound(self):
        good = brow(0.12, 0.02, 0.2)
        # F1 difference b - a in [-0.01, 0.004]: the decrease a - b is in [-0.004, 0.01]
        self.assertEqual(self.code(good, f1=brow(-0.003, -0.01, 0.004)), ("confirmed", (True, True, True)))
        self.assertEqual(self.code(good, f1=brow(-0.003, -0.0101, 0.004)), ("tradeoff", (True, True, False)))
        self.assertEqual(self.code(good, f1=brow(0.05, 0.02, 0.08))[0], "confirmed")  # F1 went up
        self.assertEqual(self.code(good, f1=brow(-0.05, -0.08, 0.02))[0], "tradeoff")  # upper bound of b - a unused
        self.assertEqual(self.code(good, f1=brow(0.0, None, None))[0], "tradeoff")
        self.assertEqual(self.code(good, brow(0.05, 0.03, 0.08), brow(-0.05, -0.08, -0.02)),
                         ("tradeoff", (True, False, False)))
        v = X.rq04_verdict(good, brow(0.01, -0.01, 0.015), brow(-0.003, -0.01, 0.004))
        self.assertEqual((v["f1_decrease"], v["f1_decrease_low"], v["f1_decrease_high"]), (0.003, -0.004, 0.01))
        self.assertEqual((v["fpr_increase"], v["fpr_increase_low"], v["fpr_increase_high"]), (0.01, -0.01, 0.015))
        self.assertEqual((v["recall_diff"], v["recall_diff_low"], v["recall_diff_high"], v["text"]),
                         (0.12, 0.02, 0.2, "개선 확인"))
        self.assertEqual(str(X.rq04_verdict(good, brow(0.0, 0.0, 0.0), brow(0.0, 0.0, 0.0))["f1_decrease"]), "0.0")

    def test_limits_are_the_readme_values(self):
        self.assertEqual((X.DELTA_R, X.DELTA_F, X.DELTA_O, X.TOL), (0.10, 0.02, 0.01, 1e-9))
        self.assertEqual((X.PRIMARY, X.SECONDARY), (tuple(PAIRS[:3]), tuple(PAIRS[3:])))
        self.assertEqual((E.N_RESAMPLES, E.BOOT_SEED, E.PERCENTILES), (2000, 314159, (2.5, 97.5)))

    def test_benign_fpr_change(self):
        def code(low, high):
            return X.fpr_change_verdict(brow(0.0, low, high))["code"]

        self.assertEqual(code(-0.3, -0.1), "fpr_decrease")
        self.assertEqual(code(0.1, 0.3), "fpr_increase")
        self.assertEqual(code(-0.1, 0.1), "no_change")
        self.assertEqual(code(-0.1, 0.0), "no_change")  # an upper bound of 0 is not < 0
        self.assertEqual(code(0.0, 0.1), "no_change")
        self.assertEqual(code(1e-12, 0.1), "no_change")
        self.assertEqual(code(None, None), "no_change")
        self.assertEqual(X.fpr_change_verdict(brow(0.0, -0.3, -0.1))["text"], "오탐 감소 확인")
        self.assertEqual(X.fpr_change_verdict(brow(0.0, 0.1, 0.3))["text"], "오탐 증가 확인")
        self.assertEqual(X.fpr_change_verdict(brow(0.0, -0.1, 0.1))["text"], "변화 확인 못함")
        self.assertEqual([X.ci_vs_0(*x) for x in ((None, None), (-2.0, -1.0), (1.0, 2.0), (-1.0, 1.0))],
                         ["", "<0", ">0", "includes_0"])

    def test_rounding_noise_around_zero_has_no_sign(self):
        self.assertEqual([str(X._snap(v)) for v in (-2.7e-17, 2.7e-17, -0.0, 0.0)], ["0.0"] * 4)
        self.assertEqual([X._snap(v) for v in (-0.25, 1e-6, -1e-6, None, 3, "x")], [-0.25, 1e-6, -1e-6, None, 3, "x"])

    def test_verdicts_take_the_pooled_recall_and_validation_clean_rows(self):
        def row(a, b, es, sub, metric, diff, low, high, kind="primary"):
            return {"kind": kind, "condition_a": a, "condition_b": b, "eval_set": es, "subtype": sub, "metric": metric,
                    "estimate_a": 0.5, "estimate_b": None if diff is None else 0.5 + diff, "diff": diff,
                    "diff_low": low, "diff_high": high}

        boot = [row("A", "C", "ROB-01~03", "main", "recall", 0.2, 0.1, 0.3),
                row("A", "C", "ROB-01", "main", "recall", 0.0, 0.0, 0.0),  # not the pooled row
                row("A", "C", "validation_clean", "all", "fpr", 0.01, 0.0, 0.015),
                row("A", "C", "validation_clean", "all", "f1", -0.002, -0.006, 0.001),
                row("A", "C", "ROB-04", "benign_security", "fpr", 0.3, 0.1, 0.5),  # not the rule-2 row
                row("A", "D", "ROB-01~03", "main", "recall", 0.2, 0.1, 0.3),
                row("A", "D", "validation_clean", "all", "fpr", 0.03, 0.02, 0.04),
                row("A", "D", "validation_clean", "all", "f1", -0.002, -0.006, 0.001),
                row("B", "D", "ROB-01~03", "main", "recall", 0.5, 0.4, 0.6, "secondary"),
                row("A", "", "ROB-04", "benign_security", "fpr", None, None, None, "level")]
        v = X.verdicts(boot)
        self.assertEqual([(x["comparison"], x["code"]) for x in v["rq04_development"]],
                         [("A->B", "not_confirmed"), ("A->C", "confirmed"), ("A->D", "tradeoff")])
        self.assertEqual([(x["comparison"], x["eval_set"], x["code"]) for x in v["benign_fpr_change"]],
                         [("A->C", "ROB-04", "fpr_increase")])


# ---------------------------------------------------------------- F1 bootstrap

class TestF1Bootstrap(unittest.TestCase):
    """Groups G0, G1 (attack) and G2 (benign), one row each. Runs a: seeds 42 and 123 predict 1, 0, 1 and
    seed 2026 predicts 1, 1, 1; runs b predict 1, 1, 0 (all right). With k = draws of G1 among the two
    attack draws: a (seeds 42, 123) has TP 2 - k, FN k, FP 1, so F1 = (4 - 2k) / (5 - k); a (seed 2026) has
    TP 2, FP 1, F1 = 4/5; b has F1 = 1. 3-seed mean of a = (2 (4 - 2k) / (5 - k) + 4/5) / 3."""

    def setUp(self):
        self.items = [(sid(i), y, ("val", sid(i))) for i, y in enumerate([1, 1, 0])]
        self.stratum = {sid(0): "1", sid(1): "1", sid(2): "0"}
        a = {42: [1, 0, 1], 123: [1, 0, 1], 2026: [1, 1, 1]}
        self.P = {f"a{s}": {"val": {sid(i): triple(p) for i, p in enumerate(a[s])}} for s in SEEDS}
        self.P.update({f"b{s}": {"val": {sid(i): triple(p) for i, p in enumerate([1, 1, 0])}} for s in SEEDS})
        self.ra, self.rb = [f"a{s}" for s in SEEDS], [f"b{s}" for s in SEEDS]
        self.k = (E.resample_indices(["1", "1", "0"]) == 1).sum(axis=1)

    def test_closed_form(self):
        res = X.f1_dist(self.items, self.ra, self.rb, self.stratum, self.P)
        k = self.k
        mean_a = (2 * (4 - 2 * k) / (5 - k) + 0.8) / 3
        self.assertTrue(np.allclose(res["a"][1], mean_a))
        self.assertTrue(np.allclose(res["b"][1], 1.0))
        self.assertTrue(np.allclose(res["diff"], 1 - mean_a))  # b - a on the same resampled groups
        self.assertAlmostEqual(res["a"][0], (0.5 + 0.5 + 0.8) / 3, places=12)  # full sample: 2/4, 2/4, 4/5
        self.assertEqual(res["b"][0], 1.0)
        self.assertEqual(sorted(set(k.tolist())), [0, 1, 2])
        row = E.bootstrap_row("primary", "validation_clean", "all", "all", "original", "original", "A", "C", "f1", res)
        self.assertAlmostEqual(row["diff"], 0.4, places=12)
        self.assertAlmostEqual(row["diff_low"], 0.2, places=12)  # k = 0: 1 - (1.6 + 0.8) / 3
        self.assertAlmostEqual(row["diff_high"], 11 / 15, places=12)  # k = 2: 1 - 0.8 / 3
        self.assertEqual((row["n_groups"], row["strata"], row["undefined_resamples"]), (3, "0:1;1:2", 0))

    def test_groups_strata_and_merging_are_those_of_bootstrap_dist(self):
        nk = {("val", sid(0)): "one key", ("val", sid(1)): "one key", ("val", sid(2)): "other"}
        res = X.f1_dist(self.items, self.ra, self.rb, self.stratum, self.P, nk)
        ref = E.bootstrap_dist(self.items, self.ra, self.items, self.rb, "recall", self.stratum, self.P, nk_of=nk)
        for key in ("groups", "strata", "merged"):
            self.assertEqual(res[key], ref[key], key)
        self.assertEqual((res["groups"], res["merged"]), ([sid(0), sid(2)], [[sid(0), sid(1)]]))
        # the merged attack group and the benign group are drawn once in every resample: the full-sample F1
        self.assertTrue(np.allclose(res["a"][1], 0.6))
        self.assertTrue(np.allclose(res["diff"], 0.4))
        plain = X.f1_dist(self.items, self.ra, self.rb, self.stratum, self.P)
        ref = E.bootstrap_dist(self.items, self.ra, self.items, self.rb, "recall", self.stratum, self.P)
        self.assertEqual((plain["groups"], plain["strata"], plain["merged"]),
                         (ref["groups"], ref["strata"], ref["merged"]))

    def test_undefined_f1_and_level(self):
        items = [(sid(i), 0, ("val", sid(i))) for i in range(2)]  # no attack row and no attack prediction
        P = {"r": {"val": {sid(0): triple(0), sid(1): triple(0)}}}
        res = X.f1_dist(items, ["r"], None, {sid(0): "0", sid(1): "0"}, P)
        self.assertEqual((res["a"][0], res["b"], res["diff"]), (None, None, None))
        self.assertTrue(np.isnan(res["a"][1]).all())
        self.assertEqual(E._ci(res["a"][1]), (None, None))
        P["r"]["val"][sid(0)] = triple(1)  # one false positive: F1 = 0 when it is drawn, undefined otherwise
        res = X.f1_dist(items, ["r"], None, {sid(0): "0", sid(1): "0"}, P)
        self.assertEqual(res["a"][0], 0.0)
        drawn = (E.resample_indices(["0", "0"]) == 0).any(axis=1)
        self.assertTrue(np.array_equal(np.isnan(res["a"][1]), ~drawn))
        self.assertTrue((res["a"][1][drawn] == 0.0).all())


# ---------------------------------------------------------------- A against C records

def record_sources():
    out = []
    for t in "AC":
        for s in SEEDS:
            rid = rid_of(t, s)
            out.append({"run_id": rid, "trained": t, "seed": s, "ids": SIDS, "labels": LABELS, "in_clean": IN_CLEAN,
                        "doc": run_record(rid, t, s, 691, {"sha256": "x", "size": 1}, "cfg")})
    return out


def set_path(doc, path, value):
    for k in path[:-1]:
        doc = doc[k]
    doc[path[-1]] = value


SAME_AC = {  # the records README 5.3 IMP-02/03 and design D6 (with its CHANGE) require to be equal in A and C
    ("budget", "N"): "budget N", ("budget", "eval_steps"): "budget eval_steps",
    ("budget", "updates_run"): "budget updates_run", ("budget", "epochs"): "budget epochs",
    ("budget", "update_size"): "budget update_size", ("budget", "micro_batch"): "budget micro_batch",
    ("budget", "grad_accum"): "budget grad_accum", ("config", "sha256"): "config SHA-256",
    ("lists", "files_sha256"): "list files SHA-256", ("lists", "dataset_revision"): "dataset revision",
    ("model",): "model record", ("determinism",): "determinism", ("env",): "environment"}
MAY_DIFFER = {  # design D6 CHANGE: may differ between A and C, both values reported
    ("lists", "lists_meta_sha256"): "lists_meta SHA-256", ("lists", "build_lists_sha256_lf"): "build_lists SHA-256",
    ("code", "files_sha256_lf"): "code files SHA-256"}
ENV_OTHER = {  # another value for every entry of the env record train.env_info writes
    "python": "3.14.4", "platform": "Windows-11-10.0.26300-SP0", "torch": "2.14.2+cu130", "cuda": "13.1",
    "cudnn": 92500, "transformers": "5.23.0", "tokenizers": "0.23.4", "safetensors": "0.8.1",
    "huggingface_hub": "2.4.1", "scikit_learn": "1.9.2", "requirements_sha256": "ef", "gpu.name": "GPU Y",
    "gpu.capability": [12, 1], "gpu.total_memory_bytes": 8546484225, "gpu.driver": "611.00"}
V08_42 = "V08 s4-roberta-C-s42 against s2-roberta-A-s42: differs or is not recorded: "


class TestRecords(unittest.TestCase):
    def test_equal_records_and_reported_differences(self):
        errs, info = X.compare_records(record_sources())
        self.assertEqual(errs, [])
        self.assertEqual(info["budget"]["N"], N_UPDATES)
        self.assertEqual(info["code"]["files_differing"], ["scripts/train.py"])  # allowed, reported
        self.assertEqual((info["code"]["A"]["git_commits"], info["code"]["C"]["git_commits"]),
                         (["c0ffee"], ["decade"]))
        self.assertEqual(info["code"]["C"]["files_sha256_lf"]["scripts/train.py"], "ab")
        self.assertEqual(info["augmentation"], {
            "rows": 1700, "train_clean_rows": 11050, "train_rows": 12750, "label_counts": {"0": 850, "1": 850},
            "files_sha256": {"train_aug.csv": "a1", "cases.csv": "a2"}, "build_train_aug_sha256_lf": "b1"})
        self.assertEqual(info["env"], ENV)  # the whole record, equal in A and C
        self.assertNotIn("env_other_differences", info)  # no env entry may differ any more
        self.assertNotIn("data/cases/train", json.dumps(info))  # no folder names of the run record

    def test_records_that_may_differ_between_a_and_c_are_reported_for_both(self):
        """Design D6 CHANGE: build_lists.py and train.py change before the C training."""
        self.assertEqual({path for _, path in X.MAY_DIFFER}, set(MAY_DIFFER))
        errs, info = X.compare_records(record_sources())  # the fixture's A and C runs differ in all three
        self.assertEqual(errs, [])
        self.assertEqual(info["lists_meta_sha256"], {"A": "cc", "C": "cd"})
        self.assertEqual(info["build_lists_sha256_lf"], {"A": "b0", "C": "b2"})
        self.assertEqual((info["code"]["A"]["files_sha256_lf"]["scripts/train.py"],
                          info["code"]["C"]["files_sha256_lf"]["scripts/train.py"]), ("aa", "ab"))
        names = ["lists_meta SHA-256", "build_lists SHA-256", "code files SHA-256"]
        self.assertEqual((info["may_differ_between_A_and_C"], info["differing_between_A_and_C"]), (names, names))
        self.assertEqual((info["equal_in_A_runs"], info["equal_in_C_runs"]),
                         (names, names + ["augmentation record", "augmentation rows", "train_clean rows"]))
        self.assertEqual(set(info["equal_in_A_and_C"]) & set(names), set())
        src = record_sources()  # the lists were not rebuilt: only the code differs
        for r in src[3:]:
            r["doc"]["lists"].update(lists_meta_sha256="cc", build_lists_sha256_lf="b0")
        errs, info = X.compare_records(src)
        self.assertEqual(errs, [])
        self.assertEqual(info["lists_meta_sha256"], {"A": "cc", "C": "cc"})
        self.assertEqual((info["may_differ_between_A_and_C"], info["differing_between_A_and_C"]), (names, names[2:]))

    def test_runs_of_one_condition_must_agree_on_the_records_reported_for_both(self):
        for path, name in MAY_DIFFER.items():
            with self.subTest(name):
                src = record_sources()
                set_path(src[1]["doc"], path, {"x": "y"})  # the A run of seed 123
                self.assertEqual(X.compare_records(src)[0], [
                    f"V08 {name} differs between the A runs or is not recorded: "
                    "s2-roberta-A-s42, s2-roberta-A-s2026 | s2-roberta-A-s123"])
                src = record_sources()
                for r in src[:3]:
                    set_path(r["doc"], path, None)
                self.assertEqual(X.compare_records(src)[0], [
                    f"V08 {name} differs between the A runs or is not recorded: "
                    "s2-roberta-A-s42, s2-roberta-A-s123, s2-roberta-A-s2026"])

    def test_every_required_record(self):
        self.assertEqual({path for _, path in X.SAME_AC}, set(SAME_AC))
        for path, name in SAME_AC.items():
            for change in ("other", None):
                with self.subTest(path=path, change=change):
                    src = record_sources()
                    set_path(src[4]["doc"], path, change)  # the C run of seed 123
                    errs, _ = X.compare_records(src)
                    self.assertEqual(len(errs), 1, errs)
                    self.assertIn("V08 s4-roberta-C-s123 against s2-roberta-A-s123: differs or is not recorded",
                                  errs[0])
                    self.assertIn(name, errs[0])
        src = record_sources()  # a record missing in both runs is not "equal"
        for i in (1, 4):
            del src[i]["doc"]["determinism"]
        self.assertIn("determinism", X.compare_records(src)[0][0])
        src = record_sources()  # one sub-key of a compared object
        src[3]["doc"]["model"]["precision"] = "fp32"
        self.assertIn("model record", X.compare_records(src)[0][0])
        src = record_sources()
        src[5]["doc"]["determinism"]["nondeterministic_warnings"] = ["w"]
        self.assertIn("V08 s4-roberta-C-s2026 against s2-roberta-A-s2026", X.compare_records(src)[0][0])

    def test_every_env_entry_must_be_equal(self):
        """Design D6: the env records of A and C must be equal. No entry is exempt (not the python, platform,
        cuDNN or library versions, not the GPU driver); the refusal names the entries that differ."""
        def flat(d, prefix=""):
            return {k2: v2 for k, v in d.items()
                    for k2, v2 in (flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v}).items()}

        self.assertEqual(sorted(ENV_OTHER), sorted(flat(ENV)))  # every entry of the record is tried
        self.assertEqual(len(ENV_OTHER), 15)
        for key, value in ENV_OTHER.items():
            for i in (0, 3):  # changed in the A run, or in the C run, of seed 42
                with self.subTest(key=key, run=i):
                    self.assertNotEqual(flat(ENV)[key], value)
                    src = record_sources()
                    set_path(src[i]["doc"]["env"], key.split("."), value)
                    self.assertEqual(X.compare_records(src)[0], [V08_42 + f"environment [{key}]"])
        src = record_sources()  # an entry missing in one record, one only there, and a changed one
        del src[3]["doc"]["env"]["cudnn"]
        src[3]["doc"]["env"].update(extra=1)
        src[3]["doc"]["env"]["gpu"]["driver"] = "611.00"
        self.assertEqual(X.compare_records(src)[0], [V08_42 + "environment [cudnn, extra, gpu.driver]"])
        src = record_sources()  # the same change in all three C runs: one refusal per seed
        for r in src[3:]:
            r["doc"]["env"]["transformers"] = "5.23.0"
        self.assertEqual(X.compare_records(src)[0], [
            f"V08 s4-roberta-C-s{s} against s2-roberta-A-s{s}: differs or is not recorded: environment [transformers]"
            for s in SEEDS])
        src = record_sources()  # no env record in either run is not "equal"
        del src[0]["doc"]["env"], src[3]["doc"]["env"]
        self.assertEqual(X.compare_records(src)[0], [V08_42 + "environment"])

    def test_differing_entries_of_other_records_are_named(self):
        src = record_sources()
        src[3]["doc"]["lists"]["files_sha256"]["train_clean.csv"] = "t2"
        src[3]["doc"]["model"]["precision"] = "fp32"
        src[3]["doc"]["budget"]["N"] = 2000
        self.assertEqual(X.compare_records(src)[0], [
            V08_42 + "budget N, list files SHA-256 [train_clean.csv], model record (weights, precision, tokenizer) "
            "[precision]"])

    def test_c_runs_must_agree_among_themselves(self):
        for path, name in ((("lists", "lists_meta_sha256"), "lists_meta SHA-256"),
                           (("lists", "build_lists_sha256_lf"), "build_lists SHA-256"),
                           (("code", "files_sha256_lf"), "code files SHA-256"),
                           (("data", "augmentation"), "augmentation record"),
                           (("data", "augmentation_rows"), "augmentation rows"),
                           (("data", "train_clean_rows"), "train_clean rows")):
            with self.subTest(name):
                src = record_sources()
                set_path(src[5]["doc"], path, {"x": "y"})
                errs, _ = X.compare_records(src)
                self.assertEqual(errs, [f"V08 {name} differs between the C runs or is not recorded: "
                                        "s4-roberta-C-s42, s4-roberta-C-s123 | s4-roberta-C-s2026"])
        src = record_sources()
        for r in src[3:]:
            del r["doc"]["code"]["files_sha256_lf"]
        self.assertIn("V08 code files SHA-256 differs between the C runs or is not recorded",
                      X.compare_records(src)[0][0])

    def test_validation_rows_must_be_the_same(self):
        for key, value in (("in_clean", [1] * N_VAL), ("labels", [0] * N_VAL), ("ids", SIDS[:-1])):
            with self.subTest(key):
                src = record_sources()
                src[4][key] = value
                self.assertEqual(X.compare_records(src)[0], [
                    "V09 validation rows (IDs, labels, in_clean) differ from s2-roberta-A-s42: s4-roberta-C-s123"])


# ---------------------------------------------------------------- end to end

class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.root = (cls.tmp / "repo")
        shutil.copytree(TEMPLATE["root"], cls.root)
        cls.root = cls.root.resolve()
        cls.calls, cls.out = [], cls.tmp / "out"
        cls.code, cls.text = run_main(cls.root, ["--out", str(cls.out)], cls.calls)
        cls.thr = {(c, s): 0.5 if c in "AC" else thr_value(LOGITS[(c, s)]) for c in "ABCD" for s in SEEDS}
        cls.src = {(c, s): rid_of("A" if c in "AB" else "C", s) for c in "ABCD" for s in SEEDS}

    def test_exit_and_files(self):
        self.assertEqual(self.code, 0, self.text)
        self.assertEqual(self.calls[:2], ["guard", "rows"])  # the test.csv guard before the rows are loaded
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), sorted(OUT_FILES))  # no .tmp left
        self.assertTrue(self.text.isascii())
        self.assertIn(f"OK conditions A, B, C, D x 3 seeds: {N_VAL - 1} validation_clean rows and {N_USED} cases on "
                      "cpu", self.text)
        self.assertEqual(self.text.count("development RQ-04"), 3)

    def test_no_text_no_paths_lf(self):
        for n in OUT_FILES:
            b = (self.out / n).read_bytes()
            self.assertNotIn(b"\r", b, n)
            self.assertTrue(b.endswith(b"\n"), n)
            for p in (MARK, str(self.tmp), self.tmp.as_posix(), self.tmp.name, "orig1 ", "alpha", "а",
                      "data/cases/train"):
                self.assertNotIn(p.encode("utf-8"), b, (n, p))
            if n != "summary.md":
                self.assertTrue(b.isascii(), n)
        # rounding noise around 0 never shows as a negative zero
        self.assertNotIn(b"-0.000000", (self.out / "bootstrap.csv").read_bytes())
        self.assertNotIn(b"-0.0000,", (self.out / "summary.md").read_bytes())
        self.assertNotIn(b"-0.0000]", (self.out / "summary.md").read_bytes())

    def test_runs_csv_against_the_stored_scores(self):
        rows = read_table(self.out / "runs.csv")
        self.assertEqual(list(rows[0]), list(X.RUN_COLUMNS))
        self.assertEqual([(r["condition"], int(r["seed"])) for r in rows], [(c, s) for c in "ABCD" for s in SEEDS])
        self.assertEqual(len({r["run_id"] for r in rows}), 6)
        differs = 0
        for r in rows:
            c, s = r["condition"], int(r["seed"])
            t = "A" if c in "AB" else "C"
            self.assertEqual((r["run_id"], r["trained_condition"], int(r["selected_step"])),
                             (self.src[(c, s)], t, SEL[(t, s)]), (c, s))
            self.assertEqual(r["threshold"], repr(self.thr[(c, s)]))
            self.assertEqual(r["logit"], "" if c in "AC" else f"{LOGITS[(c, s)]:.2f}")
            doc = json.loads((self.root / "runs" / r["run_id"] / "run.json").read_text(encoding="ascii"))
            self.assertEqual(r["checkpoint_sha256"], doc["checkpoint"]["sha256"])
            # the selected step's validation_clean rows, re-decided here from the stored score text
            stored = stored_rows(self.root, r["run_id"], SEL[(t, s)])
            tp, fp, fn, tn = counts_at(stored, self.thr[(c, s)])
            self.assertEqual((r["validation_clean_n"], r["validation_clean_tp"], r["validation_clean_fp"],
                              r["validation_clean_fn"], r["validation_clean_tn"]),
                             (str(N_VAL - 1), str(tp), str(fp), str(fn), str(tn)), (c, s))
            self.assertEqual(r["validation_clean_recall"], f"{tp / (tp + fn):.6f}")
            self.assertEqual(r["validation_clean_fpr"], f"{fp / (fp + tn):.6f}")
            self.assertEqual(r["validation_clean_f1"], f"{2 * tp / (2 * tp + fp + fn):.6f}")
            if c in "AC":  # the same as the predictions train.py stored at 0.5
                self.assertEqual(tp + fp, sum(f[6] == "1" and f[3] == "1" for f in stored))
                other = [x for x in STEPS if x != SEL[(t, s)]]
                self.assertTrue(all(counts_at(stored_rows(self.root, r["run_id"], x), 0.5) != (tp, fp, fn, tn)
                                    for x in other))  # another step would give other counts
            else:
                differs += (tp, fp) != counts_at(stored, 0.5)[:2]
        self.assertGreater(differs, 0)  # the fixture's thresholds change decisions
        by = {(r["condition"], r["seed"]): r for r in rows}
        for s in SEEDS:  # ROC-AUC does not depend on the threshold
            self.assertEqual(by[("A", str(s))]["validation_clean_roc_auc"],
                             by[("B", str(s))]["validation_clean_roc_auc"])

    def test_predictions_csv(self):
        preds = read_table(self.out / "predictions.csv")
        self.assertEqual(list(preds[0]), list(X.PRED_COLUMNS))
        self.assertEqual(len(preds), 12 * N_USED)
        ids = sorted({r["case_id"] for r in preds})
        self.assertEqual([(r["condition"], r["run_id"], r["case_id"]) for r in preds],
                         [(c, self.src[(c, s)], i) for c in "ABCD" for s in SEEDS for i in ids])
        self.assertNotIn("dev-rob02-000014", ids)  # reserve
        by = {(r["condition"], r["run_id"], r["case_id"]): r for r in preds}
        changed = 0
        for r in preds:
            seed = int(r["run_id"].rsplit("-s", 1)[1])
            self.assertEqual(r["threshold"], repr(self.thr[(r["condition"], seed)]))
            want = int(np.float32(float(r["score"])) >= np.float32(float(r["threshold"])))
            self.assertEqual(int(r["prediction"]), want, r)
            if r["condition"] in "BD":  # the same checkpoint as A / C: same margin and score
                base = by[("A" if r["condition"] == "B" else "C", r["run_id"], r["case_id"])]
                self.assertEqual((r["margin"], r["score"]), (base["margin"], base["score"]))
                changed += r["prediction"] != base["prediction"]
        self.assertGreater(changed, 0)

    def test_metrics_models_and_bootstrap_csv(self):
        met = read_table(self.out / "metrics.csv")
        self.assertEqual(list(met[0]), list(X.METRIC_COLUMNS))
        runs = {(r["condition"], r["seed"]): r for r in read_table(self.out / "runs.csv")}
        val = [r for r in met if r["eval_set"] == "validation_clean" and r["label"] == "all"]
        self.assertEqual(len(val), 12)
        for r in val:
            run = runs[(r["condition"], r["seed"])]
            self.assertEqual((r["run_id"], r["threshold"], r["side"], r["subtype"], r["slice"]),
                             (run["run_id"], run["threshold"], "original", "all", "all"))
            for k in ("n", "tp", "fp", "fn", "tn", "f1", "precision", "recall", "fpr", "roc_auc"):
                self.assertEqual(r[k], run[f"validation_clean_{k}"], k)
        # metrics.csv agrees with predictions.csv: pooled attack variants of condition D, seed 123
        preds = {r["case_id"]: int(r["prediction"]) for r in read_table(self.out / "predictions.csv")
                 if (r["condition"], r["run_id"]) == ("D", "s4-roberta-C-s123")}
        att = [c for c in preds if c.startswith(("dev-rob01-case", "dev-rob01-space", "dev-rob01-char", "dev-rob02",
                                                 "dev-rob03")) and int(c[-6:]) % 2 == 0]
        self.assertEqual(len(att), 2 * 3 + 2 + 3)
        row = [r for r in met if (r["condition"], r["seed"], r["eval_set"], r["subtype"], r["side"], r["label"])
               == ("D", "123", "ROB-01~03", "main", "variant", "1")]
        tp = sum(preds[c] for c in att)
        self.assertEqual([(r["n"], r["tp"], r["recall"]) for r in row], [(str(len(att)), str(tp),
                                                                         f"{tp / len(att):.6f}")])
        self.assertEqual({r["slice"] for r in met if r["eval_set"] == "ROB-03"},
                         {"all", "position=prefix", "position=suffix", "position=both", "length=short", "length=long"})
        mod = read_table(self.out / "models.csv")
        self.assertEqual(list(mod[0])[:10], ["condition", "eval_set", "subtype", "slice", "side", "label", "n",
                                             "quantity", "mean", "std"])
        self.assertEqual(list(mod[0])[10:], ["s42", "s123", "s2026"])
        self.assertEqual([r["condition"] for r in mod], sorted((r["condition"] for r in mod)))
        self.assertIn("change", {r["side"] for r in mod})
        r = next(x for x in mod if (x["condition"], x["eval_set"], x["label"], x["quantity"])
                 == ("B", "validation_clean", "all", "fp"))
        fps = [int(runs[("B", str(s))]["validation_clean_fp"]) for s in SEEDS]
        self.assertEqual((r["s42"], r["s123"], r["s2026"]), tuple(str(v) for v in fps))
        self.assertEqual(r["mean"], f"{sum(fps) / 3:.6f}")
        boot = read_table(self.out / "bootstrap.csv")
        self.assertEqual(list(boot[0]), list(X.BOOT_COLUMNS))
        self.assertEqual({r["kind"] for r in boot}, {"primary", "secondary", "level"})
        self.assertEqual(list(dict.fromkeys((r["condition_a"], r["condition_b"]) for r in boot
                                            if r["kind"] != "level")), PAIRS)
        self.assertEqual({r["metric"] for r in boot if r["eval_set"] == "validation_clean"}, {"recall", "fpr", "f1"})
        r = next(x for x in boot if (x["condition_a"], x["condition_b"], x["eval_set"], x["metric"])
                 == ("A", "B", "validation_clean", "fpr"))
        a = sum(int(runs[("A", str(s))]["validation_clean_fp"]) / (N_VAL - 1 - 14) for s in SEEDS) / 3
        self.assertEqual(r["estimate_a"], f"{a:.6f}")  # 14 attack and 15 benign rows in validation_clean

    def test_validation_rows_with_one_normalised_key_are_one_bootstrap_group(self):
        """README 6.1 item 1, on the texts the script reads itself: the fixture's validation rows 21, 23 and
        27 (benign) share a normalised key, and so do 22 (attack) and 25 (benign)."""
        for group in NK_SETS:  # different texts with one normalised key; all of them are validation_clean rows
            self.assertEqual(len({VAL[s][1] for s in group}), len(group))
            self.assertEqual(len({build_lists.nk(VAL[s][1]) for s in group}), 1)
            self.assertTrue(all(IN_CLEAN[SIDS.index(s)] for s in group))
        self.assertEqual([[LABELS[SIDS.index(s)] for s in g] for g in NK_SETS], [[0, 0, 0], [1, 0]])
        boot = [r for r in read_table(self.out / "bootstrap.csv") if r["eval_set"] == "validation_clean"]
        self.assertEqual(len(boot), 6 * 3)  # recall, FPR and F1 of the six comparisons
        self.assertEqual({(r["n_groups"], r["strata"], r["nk_merges"]) for r in boot},
                         {(str(VAL_GROUPS), VAL_STRATA, str(VAL_NK_MERGES))})
        doc = json.loads((self.out / "summary.json").read_text(encoding="ascii"))
        self.assertEqual(doc["grouping"], {"nk_merged_group_sets": NK_SETS, "comparisons_with_nk_merges": 6 * 3})
        self.assertEqual({r["nk_merges"] for r in doc["bootstrap"] if r["eval_set"] != "validation_clean"}, {0})
        # the metrics are still per row: all 29 validation_clean rows
        self.assertEqual({r["validation_clean_n"] for r in read_table(self.out / "runs.csv")}, {str(N_VAL - 1)})
        self.assertIn(f"합친 그룹이 있는 비교: {6 * 3}개", (self.out / "summary.md").read_text(encoding="utf-8"))

    def test_summary_json(self):
        doc = json.loads((self.out / "summary.json").read_text(encoding="ascii"))
        self.assertTrue(doc["development"])
        self.assertEqual(doc["verdicts"]["stage"], "development")
        self.assertEqual((doc["representative"], doc["conditions"], doc["seeds"]), (REP, list("ABCD"), SEEDS))
        self.assertEqual(doc["device"]["type"], "cpu")
        self.assertEqual(doc["script_sha256_lf"], data_lists.sha256_lf(ROOT / "scripts" / "eval_abcd.py"))
        want = (["configs/train.json", "data/cases/dev/cases.csv", "data/cases/dev/review.csv",
                 "results/s2/runs.csv", "results/s2/summary.json", "results/s4/thresholds.json"]
                + [f"runs/{rid_of(t, s)}/{n}" for t in "AC" for s in SEEDS for n in X.RUN_FILES])
        self.assertEqual(sorted(doc["inputs_sha256"]), sorted(want))
        for rel, sha in doc["inputs_sha256"].items():
            self.assertEqual(sha, file_sha(self.root, rel), rel)
        self.assertEqual([(r["run_id"], r["trained_condition"], r["selected_step"]) for r in doc["source_runs"]],
                         [(rid_of(t, s), t, SEL[(t, s)]) for t in "AC" for s in SEEDS])
        for r in doc["source_runs"]:
            c = r["original_check"]
            self.assertEqual((c["prediction_flips"], c["margin_rows_differing_9_digits"], c["validation_rows"]),
                             (0, 0, N_VAL))
        self.assertEqual(doc["thresholds"]["file"], "results/s4/thresholds.json")
        self.assertEqual((doc["thresholds"]["entries"], doc["thresholds"]["run_file_sha256_checked"],
                          doc["thresholds"]["selection_counts_checked"]), (6, 12, 6))
        self.assertEqual([(v["condition"], v["seed"], v["threshold"], v["logit"], v["run_id"])
                          for v in doc["thresholds"]["values"]],
                         [(c, s, thr_value(LOGITS[(c, s)]), LOGITS[(c, s)], self.src[(c, s)])
                          for c in "BD" for s in SEEDS])  # full precision, not 6 decimals
        self.assertEqual([r["threshold"] for r in doc["runs"]], [self.thr[(c, s)] for c in "ABCD" for s in SEEDS])
        # README 5.3: a threshold chosen without a candidate under the FPR cap is recorded as such, and a
        # retrained run as a rerun (the fixture: D seed 123, and the C run of seed 2026 as attempt 2)
        self.assertEqual([v["cap_met"] for v in doc["thresholds"]["values"]], [True] * 4 + [False, True])
        self.assertEqual([r["cap_met"] for r in doc["runs"]],
                         [None] * 3 + [True] * 3 + [None] * 3 + [True, False, True])
        self.assertEqual([r["attempt"] for r in doc["source_runs"]], [1, 1, 1, 1, 1, 2])
        self.assertEqual([r["attempt"] for r in doc["runs"]], [1, 1, 1] * 2 + [1, 1, 2] * 2)
        for v in doc["thresholds"]["values"]:  # the counts of the selection that were compared
            stored = stored_rows(self.root, v["run_id"], SEL[("A" if v["condition"] == "B" else "C", v["seed"])])
            self.assertEqual(tuple(v["selection_counts"][k] for k in ("tp", "fp", "fn", "tn")),
                             counts_at(stored, v["threshold"]), v)
        rec = doc["records"]
        self.assertEqual(rec["env"], ENV)
        self.assertNotIn("env_other_differences", rec)
        self.assertEqual((rec["lists_meta_sha256"], rec["build_lists_sha256_lf"]),
                         ({"A": "cc", "C": "cd"}, {"A": "b0", "C": "b2"}))
        self.assertEqual(rec["differing_between_A_and_C"],
                         ["lists_meta SHA-256", "build_lists SHA-256", "code files SHA-256"])
        self.assertIn("environment", rec["equal_in_A_and_C"])
        self.assertNotIn("lists_meta SHA-256", rec["equal_in_A_and_C"])
        r = doc["runs"][3]  # B seed 42: the exact fraction agrees with the counts
        self.assertEqual(r["exact"]["fpr"], "{0.numerator}/{0.denominator}".format(
            __import__("fractions").Fraction(r["validation_clean_fp"], r["validation_clean_fp"]
                                             + r["validation_clean_tn"])))
        self.assertEqual(r["thresholded_original_check"]["stored_vs_repredicted_flips"], 0)
        self.assertEqual(doc["records"]["code"]["files_differing"], ["scripts/train.py"])
        self.assertEqual(doc["records"]["augmentation"]["rows"], 1700)
        self.assertEqual(doc["records"]["budget"]["eval_steps"], STEPS)
        self.assertEqual(len(doc["verdicts"]["rq04_development"]), 3)
        self.assertEqual(len(doc["runs"]), 12)
        self.assertEqual(doc["settings"]["threshold"], 0.5)
        self.assertEqual(doc["cases"]["status_counts"]["assigned"], N_USED)

    def test_summary_md(self):
        md = (self.out / "summary.md").read_text(encoding="utf-8")
        self.assertTrue(md.startswith("# 4단계 A~D 비교 결과 (개발용)\n"))
        for part in ("**개발 결과다.**", "## RQ-04 규칙의 개발 단계 적용 (최종 판정 아님)", "## 주요 비교: A→B, A→C, A→D",
                     "## 보조 비교: B→D, C→D, B→C", "## ROB-04 FPR 수준", "`s4-roberta-C-s2026`",
                     "증강 1,700행", "| ROB-01~03 | main | 11/11 |"):
            self.assertIn(part, md)
        doc = json.loads((self.out / "summary.json").read_text(encoding="ascii"))
        for v in doc["verdicts"]["rq04_development"]:
            self.assertIn(f"| {v['condition_a']}→{v['condition_b']} | ", md)
            self.assertIn(v["text"], md)

    def test_summary_md_runs_table_and_record_notes(self):
        md = (self.out / "summary.md").read_text(encoding="utf-8")
        table = md.split("## 조건별 실행과 임계값\n\n", 1)[1].split("\n\n", 1)[0].split("\n")
        self.assertEqual(table[0], "| 조건 | seed | 실행 | 학습 시도 | 선택 step | 임계값 | logit | FPR 상한 |")
        self.assertEqual(len(table), 2 + 12)
        cells = [[c.strip() for c in ln.strip("|").split("|")] for ln in table[2:]]
        self.assertEqual([(c[0], c[1]) for c in cells], [(c, str(s)) for c in "ABCD" for s in SEEDS])
        # the attempt of the source run (the C run of seed 2026 is a rerun) and the recorded cap_met
        self.assertEqual([c[3] for c in cells], ["1", "1", "1"] * 2 + ["1", "1", "2"] * 2)
        self.assertEqual([c[7] for c in cells], ["-"] * 3 + ["충족"] * 3 + ["-"] * 3 + ["충족", "미충족", "충족"])
        self.assertEqual(cells[4][2:7], ["`s2-roberta-A-s123`", "1", str(SEL[("A", 123)]),
                                         f"{thr_value(LOGITS[('B', 123)]):.6f}", "+8.00"])
        for part in ("2 이상인 실행: `s4-roberta-C-s2026`.", "미충족인 조건: D seed 123.",
                     "환경 기록은 `run.json`의 `env` 전체", "하나라도 다르면 이 스크립트가 거부한다",
                     "A와 C 사이에 달라도 되는 기록은 `lists_meta.json` SHA-256, `build_lists.py` SHA-256, 학습 코드 "
                     "SHA-256이다",
                     "실제로 다른 것: `lists_meta.json` SHA-256, `build_lists.py` SHA-256, 학습 코드 SHA-256. 학습 코드 "
                     "SHA-256이 A와 다른 파일: 1개(`scripts/train.py`)",
                     "기록된 실행 파일 SHA-256 12개가 같고", "TP·FP·FN·TN 기록 6건이 여기서 계산한 값과 같다"
                     "(여섯 실행의 `run.json`·`predictions_validation.csv` SHA-256과 임계값마다의 TP·FP·FN·TN 기록이 "
                     "없거나 다르면 이 스크립트가 거부한다)"):
            self.assertIn(part, md)
        for gone in ("필수 항목 밖에서", "env_other_differences", "기록이 있는 항목만 대조"):
            self.assertNotIn(gone, md)

    def test_deterministic_and_default_output_folder(self):
        code, text = run_main(self.root, [])
        self.assertEqual(code, 0, text)
        s4 = self.root / "results" / "s4"
        self.assertEqual(sorted(p.name for p in s4.iterdir()), sorted(OUT_FILES + ("thresholds.json",)))
        for n in OUT_FILES:
            self.assertEqual((s4 / n).read_bytes(), (self.out / n).read_bytes(), n)
        self.assertIn("wrote results/s4/: runs.csv", text)

    def test_checkpoints_are_loaded_with_their_pins(self):
        seen = []
        real = eval_lib.load_run_model

        def spy(run_dir, *args, **kw):
            seen.append((Path(run_dir).name, args, kw))
            return real(run_dir, *args, **kw)

        with mock.patch.object(eval_lib, "load_run_model", side_effect=spy):
            code, text = run_main(self.root, ["--out", str(self.tmp / "out_spy")])
        self.assertEqual(code, 0, text)
        self.assertEqual([n for n, _, _ in seen], [rid_of(t, s) for t in "AC" for s in SEEDS])  # six, not twelve
        for n, args, kw in seen:
            ck = json.loads((self.root / "runs" / n / "run.json").read_text(encoding="ascii"))["checkpoint"]
            self.assertEqual((args, kw["expected_sha256"], kw["expected_size"]), (("eager",), ck["sha256"], ck["size"]))


# ---------------------------------------------------------------- refusals

def _c(seed, fn, name="run.json"):
    return lambda root: edit_json(root / "runs" / rid_of("C", seed) / name, fn)


def _thr(fn):
    def mutate(root):
        doc = thresholds_doc(root)
        doc = fn(doc) or doc
        write_thresholds(root, doc)
    return mutate


def _thr_drop(*path):
    """Remove one member of the thresholds.json document (path of keys and list indexes)."""
    def fn(doc):
        node = doc
        for k in path[:-1]:
            node = node[k]
        del node[path[-1]]
    return _thr(fn)


def _review(fn):
    def mutate(root):
        p = root / "data" / "cases" / "dev" / "review.csv"
        rows = read_table(p)
        fn(rows)
        write_table(p, E.REVIEW_COLUMNS, rows)
    return mutate


def _no_assigned_case(root):
    """Every case is a reserve and no case is reviewed: nothing pending, nothing excluded, nothing to evaluate."""
    d = root / "data" / "cases" / "dev"
    rows = read_table(d / "cases.csv")
    for r in rows:
        r["status"] = "reserve"
    write_table(d / "cases.csv", E.CASES_COLUMNS, rows)
    write_table(d / "review.csv", E.REVIEW_COLUMNS, [])


def _pred(seed, old, new, trained="C"):
    """Replace text in the selected-step rows of a predictions_validation.csv (first match)."""
    def mutate(root):
        p = root / "runs" / rid_of(trained, seed) / "predictions_validation.csv"
        lines = p.read_text(encoding="ascii").split("\n")
        k = next(i for i, ln in enumerate(lines) if ln.startswith(f"{SEL[(trained, seed)]},") and old in ln)
        lines[k] = lines[k].replace(old, new)
        p.write_bytes("\n".join(lines).encode("ascii"))
    return mutate


def _flip_stored_prediction(root):
    p = root / "runs" / rid_of("C", 42) / "predictions_validation.csv"
    lines = p.read_text(encoding="ascii").split("\n")
    k = next(i for i, ln in enumerate(lines) if ln.startswith(f"{SEL[('C', 42)]},{sid(7)},"))
    f = lines[k].split(",")
    f[6] = str(1 - int(f[6]))
    lines[k] = ",".join(f)
    p.write_bytes("\n".join(lines).encode("ascii"))


def _tamper(trained, seed):
    def mutate(root):
        f = root / "runs" / rid_of(trained, seed) / "checkpoint" / "model.safetensors"
        b = bytearray(f.read_bytes())
        b[-1] ^= 1
        f.write_bytes(bytes(b))
    return mutate


def _other_selected_step(root):
    """run.json and metrics.json agree on a step that the selection rule does not choose."""
    d = root / "runs" / rid_of("C", 42)
    other = STEPS[0]
    assert other != SEL[("C", 42)]
    edit_json(d / "run.json", lambda r: (r["selection"].update(selected_step=other),
                                         r["checkpoint"].update(step=other)))
    edit_json(d / "metrics.json", lambda m: m.update(selected_step=other))


def _other_model(root):
    """A valid, consistently recorded checkpoint of another model in a C run: the file checks pass, the
    re-predicted validation predictions differ from the stored ones."""
    rid = rid_of("C", 123)
    d = root / "runs" / rid
    tmp = root / "other_ckpt"
    info = model_specs.save_checkpoint(tiny_model(999), REP, tmp, rid, SEL[("C", 123)])
    shutil.copyfile(tmp / "model.safetensors", d / "checkpoint" / "model.safetensors")
    shutil.rmtree(tmp)
    edit_json(d / "run.json", lambda r: r["checkpoint"].update(sha256=info["sha256"], size=info["size"]))
    write_thresholds(root)  # as if the thresholds had been selected on the changed run files


def _stale_stored_score(root):
    """The stored score of a case original of s4-roberta-C-s42 is moved across the D threshold (0.7941...)
    without crossing 0.5, so the file still agrees with its predictions and with metrics.json."""
    thr = thr_value(LOGITS[("D", 42)])
    assert thr > 0.5
    p = root / "runs" / rid_of("C", 42) / "predictions_validation.csv"
    lines = p.read_text(encoding="ascii").split("\n")
    used = {sid(i) for i in range(14)}
    for k, ln in enumerate(lines):
        f = ln.split(",")
        if f[0] == str(SEL[("C", 42)]) and f[1] in used and float(f[5]) >= 0.5:
            new = (0.5 + thr) / 2 if float(f[5]) >= thr else (1 + thr) / 2
            f[5] = eval_lib.fmt9(float(np.float32(new)))
            lines[k] = ",".join(f)
            p.write_bytes("\n".join(lines).encode("ascii"))
            write_thresholds(root)  # as if the thresholds had been selected on the changed file
            return
    raise AssertionError("no case original with a score >= 0.5 in the fixture")


def _stale_thresholds(root):
    """thresholds.json records the SHA-256 of a predictions file that has been replaced since."""
    p = root / "runs" / rid_of("C", 2026) / "predictions_validation.csv"
    doc = thresholds_doc(root)
    doc["inputs_sha256"][f"runs/{rid_of('C', 2026)}/predictions_validation.csv"] = "0" * 64
    write_thresholds(root, doc)
    assert p.is_file()


def _config_changed(root):
    p = root / "configs" / "train.json"
    p.write_bytes(p.read_bytes() + b"\n")  # still valid JSON, other SHA-256


def _config(fn):
    """Another configs/train.json that the stage-2 record pins consistently (so the E03 check passes)."""
    def mutate(root):
        p = root / "configs" / "train.json"
        cfg = json.loads(p.read_bytes())
        fn(cfg)
        p.write_bytes(json.dumps(cfg, indent=2).encode("ascii") + b"\n")
        sha = file_sha(root, "configs/train.json")
        edit_json(root / "results" / "s2" / "summary.json",
                  lambda d: (d["inputs_sha256"].update({"configs/train.json": sha}),
                             d["records"].update(config_sha256=sha)))
    return mutate


C42, C123 = "s4-roberta-C-s42", "s4-roberta-C-s123"
REFUSALS = {  # name: (mutation of a fresh root, expected text)
    "V01 folder": (lambda r: shutil.rmtree(r / "runs" / C42), f"V01 {C42}: runs/{C42}/ is missing"),
    "V01 lock": (lambda r: (r / "runs" / f"{C42}.lock").write_bytes(b""), f"V01 {C42}: runs/{C42}.lock exists"),
    "V01 file": (lambda r: (r / "runs" / C42 / "metrics.json").unlink(), f"V01 {C42}: missing metrics.json"),
    "V02 json": (lambda r: (r / "runs" / C42 / "run.json").write_bytes(b"{"), f"V02 {C42}: run.json or metrics.json "
                 "is not valid JSON"),
    "V02 stage": (_c(42, lambda r: r.update(stage="s2")), f"V02 {C42}: run.json stage do not match the run ID"),
    "V02 condition": (_c(42, lambda r: r.update(condition="A")), f"V02 {C42}: run.json condition do not match"),
    "V02 model": (_c(42, lambda r: r.update(model_key="bert")), f"V02 {C42}: run.json model_key do not match"),
    "V02 seed": (_c(42, lambda r: r.update(seed=123)), f"V02 {C42}: run.json seed do not match"),
    "V02 key": (_c(42, lambda r: r.pop("budget")), f"V02 {C42}: run.json or metrics.json: missing key or wrong type"),
    "V03 status": (_c(123, lambda r: r.update(status="failed")), f"V03 {C123}: status is 'failed', not 'completed'"),
    "V04 reload": (_c(42, lambda r: r["checkpoint"]["reload_check"].update(passed=False)),
                   f"V04 {C42}: the checkpoint reload check did not pass"),
    "V04 kept": (_c(42, lambda r: r["checkpoint"].update(kept=False)), f"V04 {C42}: checkpoint not kept"),
    "V04 path": (_c(42, lambda r: r["checkpoint"].update(path="runs/x/checkpoint/model.safetensors")),
                 f"V04 {C42}: checkpoint not kept, path not runs/{C42}/checkpoint/model.safetensors"),
    "V04 size": (_c(42, lambda r: r["checkpoint"].update(size=r["checkpoint"]["size"] + 1)),
                 f"V04 {C42}: runs/{C42}/checkpoint/model.safetensors is missing or its size differs"),
    "V04 bytes": (_tamper("C", 42), f"V04 {C42}: runs/{C42}/checkpoint/model.safetensors SHA-256 differs from "
                  "run.json"),
    "V04 step": (_c(42, lambda r: r["checkpoint"].update(step=STEPS[2])), f"V04 {C42}: checkpoint step 2073 != "
                 "selected step 1382"),
    "V05 threshold": (_c(42, lambda m: m.update(threshold=0.4), "metrics.json"),
                      f"V05 {C42}: metrics.json run_id or threshold (0.4) differs"),
    "V05 run id": (_c(42, lambda m: m.update(run_id=C123), "metrics.json"),
                   f"V05 {C42}: metrics.json run_id or threshold (0.5) differs"),
    "V05 eval steps": (_c(42, lambda m: m.update(eval_steps=[1, 2, 3]), "metrics.json"),
                       f"V05 {C42}: metrics.json eval steps differ from run.json budget"),
    "V05 selected step": (_c(42, lambda m: m.update(selected_step=STEPS[0]), "metrics.json"),
                          f"V05 {C42}: metrics.json selected_step 691 != run.json 1382"),
    "V05 selection rule": (_other_selected_step, f"V05 {C42}: selected step 691 does not follow the selection rule"),
    "V06 condition column": (_pred(42, ",0.5,C", ",0.5,A"), f"V06 {C42}: predictions_validation.csv: line "),
    "V06 threshold column": (_pred(42, ",0.5,C", ",0.4,C"), "threshold not 0.5"),
    "V06 stored prediction": (_flip_stored_prediction, f"V06 {C42}: predictions_validation.csv: step 1382: a stored "
                              "prediction is not float32(score) >= 0.5"),
    "V06 row counts": (_c(42, lambda r: r["data"].update(validation_clean_rows=N_VAL)),
                       f"V06 {C42}: prediction row counts differ from run.json data"),
    "V06 validation rows": (_c(42, lambda r: r["data"].update(validation_rows=N_VAL + 1)),
                            f"V06 {C42}: prediction row counts differ from run.json data"),
    "V06 metrics.json": (_c(42, lambda m: m["evals"][2]["validation_clean"].update(
        tn=m["evals"][2]["validation_clean"]["tn"] - 1), "metrics.json"),
        f"V06 {C42}: step 2073 validation_clean: predictions do not reproduce metrics.json (tn)"),
    "V07 lists": (_c(42, lambda r: r["lists"].update(mode="trial")), f"V07 {C42}: lists are not official"),
    "V07 git state": (_c(42, lambda r: r["code"].update(git_dirty=[{"status": " M", "path": "scripts/train.py"}])),
                      f"V07 {C42}: code.git_dirty is not empty"),
    "V07 budget": (_c(42, lambda r: r["budget"].update(updates_run=2000)), f"V07 {C42}: budget: updates_run 2000"),
    "V07 model pins": (_c(42, lambda r: r["model"]["weights"].update(sha256="0" * 64)),
                       f"V07 {C42}: model record differs from model_specs"),
    "V07 no augmentation rows": (_c(42, lambda r: r["data"].update(augmentation_rows=0, train_rows=11050)),
                                 f"V07 {C42}: the augmentation is not recorded"),
    "V07 train rows": (_c(42, lambda r: r["data"].update(train_rows=11050)), f"V07 {C42}: the augmentation is not "
                       "recorded: data.train_clean_rows 11050 (lists 11050), augmentation_rows 1700 (> 0), train_rows "
                       "11050 (their sum)"),
    "V07 augmentation record": (_c(42, lambda r: r["data"].pop("augmentation")), "V07 s4-roberta-C-s42: the "
                                "augmentation is not recorded"),
    "V07 train_clean rows": (_c(42, lambda r: r["data"].update(train_clean_rows=11000, train_rows=12700)),
                             f"V07 {C42}: the augmentation is not recorded: data.train_clean_rows 11000 (lists "
                             "11050), augmentation_rows 1700 (> 0), train_rows 12700 (their sum)"),
    "V08 lists differ": (_c(123, lambda r: r["lists"]["files_sha256"].update({"train_clean.csv": "t2"})),
                         f"V08 {C123} against s2-roberta-A-s123: differs or is not recorded: list files SHA-256"),
    "V08 env library version": (_c(123, lambda r: r["env"].update(transformers="5.23.0")),
                                f"V08 {C123} against s2-roberta-A-s123: differs or is not recorded: environment "
                                "[transformers]"),
    "V08 env python and cudnn": (_c(123, lambda r: r["env"].update(python="3.15.0", cudnn=99999)),
                                 f"V08 {C123} against s2-roberta-A-s123: differs or is not recorded: environment "
                                 "[cudnn, python]"),
    "V08 env gpu driver": (_c(123, lambda r: r["env"]["gpu"].update(driver="999.99")),
                           f"V08 {C123} against s2-roberta-A-s123: differs or is not recorded: environment "
                           "[gpu.driver]"),
    "V08 env platform": (_c(123, lambda r: r["env"].update(platform="Windows-11-10.0.26300-SP0")),
                         f"V08 {C123} against s2-roberta-A-s123: differs or is not recorded: environment [platform]"),
    "V08 C lists_meta": (_c(123, lambda r: r["lists"].update(lists_meta_sha256="ce")),
                         "V08 lists_meta SHA-256 differs between the C runs or is not recorded: s4-roberta-C-s42, "
                         "s4-roberta-C-s2026 | s4-roberta-C-s123"),
    "V08 precision": (_c(123, lambda r: r["model"].update(precision="fp32")), "model record (weights, precision, "
                      "tokenizer)"),
    "V08 C code": (_c(123, lambda r: r["code"]["files_sha256_lf"].update({"scripts/train.py": "ac"})),
                   "V08 code files SHA-256 differs between the C runs"),
    "T01 missing": (lambda r: (r / "results" / "s4" / "thresholds.json").unlink(),
                    "T01 results/s4/thresholds.json: cannot read (FileNotFoundError); run "
                    "scripts/select_thresholds.py"),
    "T01 json": (lambda r: (r / "results" / "s4" / "thresholds.json").write_bytes(b"[1]\n"),
                 "T01 thresholds.json: the document is not a JSON object"),
    "T02 entry missing": (_thr(lambda d: d["thresholds"].pop()), "T02 thresholds.json: no entry"),
    "T03 float64 threshold": (_thr(lambda d: d["thresholds"][0].update(threshold=1 / (1 + math.exp(8.0)))),
                              "T03 thresholds.json B s42: threshold is not a float32 value in (0, 1)"),
    "T03 other step": (_thr(lambda d: d["thresholds"][3].update(selected_step=STEPS[2])),
                       "T03 thresholds.json D s42: selected_step '2073' disagrees with the run s4-roberta-C-s42"),
    "T03 selection counts": (_thr(lambda d: d["thresholds"][4]["chosen"].update(
        tn=d["thresholds"][4]["chosen"]["tn"] + 1)), "T03 thresholds.json D s123: the recorded counts at the chosen "
        "threshold"),
    "T03 other checkpoint": (_thr(lambda d: d["thresholds"][1].update(checkpoint_sha256="0" * 64)),
                             "T03 thresholds.json B s123: checkpoint_sha256 '00000000000000000000' disagrees with "
                             "the run s2-roberta-A-s123 used here"),
    "T03 stale": (_stale_thresholds, "T03 thresholds.json: the recorded SHA-256 of runs/s4-roberta-C-s2026/"
                  "predictions_validation.csv differs"),
    "T03 count not an integer": (_thr(lambda d: d["thresholds"][0]["chosen"].update(
        tp=float(d["thresholds"][0]["chosen"]["tp"]))), "T03 thresholds.json B s42: chosen must hold the integer "
        "counts tp, fp, fn, tn at the chosen threshold"),
    "T03 count missing": (_thr_drop("thresholds", 0, "chosen", "tn"), "T03 thresholds.json B s42: chosen must hold "
                          "the integer counts tp, fp, fn, tn at the chosen threshold"),
    "T03 no counts": (_thr_drop("thresholds", 5, "chosen"), "T03 thresholds.json D s2026: chosen must hold the "
                      "integer counts tp, fp, fn, tn at the chosen threshold"),
    "T03 no cap_met": (_thr_drop("thresholds", 2, "cap_met"), "T03 thresholds.json B s2026: cap_met must be true or "
                       "false"),
    "T03 run file not recorded": (_thr_drop("inputs_sha256", f"runs/{C123}/predictions_validation.csv"),
                                  "T03 thresholds.json: no SHA-256 is recorded for 1 run file(s) used here "
                                  f"(runs/{C123}/predictions_validation.csv)"),
    "T03 no run file recorded": (_thr_drop("inputs_sha256"), "T03 thresholds.json: no SHA-256 is recorded for 12 run "
                                 "file(s) used here (runs/s2-roberta-A-s123/predictions_validation.csv, "),
    "M02 A run file": (lambda r: (r / "runs" / "s2-roberta-A-s42" / "metrics.json").write_bytes(b"{}\n"),
                       "M02 s2-roberta-A-s42: metrics.json SHA-256 differs"),
    "M02 other model's run": (lambda r: (r / "runs" / "s2-bert-A-s42" / "run.json").write_bytes(b"{}\n"),
                              "M02 s2-bert-A-s42: run.json SHA-256 differs"),
    "M04 A checkpoint": (_tamper("A", 42), "M04 s2-roberta-A-s42: checkpoint refused"),
    "M05 C checkpoint of another model": (_other_model, f"M05 {C123}:"),
    "M08 stored score": (_stale_stored_score, f"M08 {C42} (condition D, threshold 0.7941296100616455): 1 of "),
    "E03 config": (_config_changed, "E03 configs/train.json: SHA-256 (LF)"),
    "E00 threshold": (_config(lambda c: c.update(threshold=0.4)),
                      "E00 configs/train.json: eval precision must be fp32 and threshold 0.5"),
    "E00 eval precision": (_config(lambda c: c["eval"].update(precision="bf16")),
                           "E00 configs/train.json: eval precision must be fp32 and threshold 0.5"),
    "E00 two seeds": (_config(lambda c: c.update(seeds=[42, 123])),
                      "E00 configs/train.json: seeds must be 3 distinct integers"),
    "E00 repeated seed": (_config(lambda c: c.update(seeds=[42, 123, 42])),
                          "E00 configs/train.json: seeds must be 3 distinct integers"),
    "E00 missing key": (_config(lambda c: c.pop("max_length")),
                        "E00 configs/train.json: unreadable or missing keys (KeyError)"),
    "P01 pending": (_review(lambda rows: rows[0].update(decision="pending", reviewer="")),
                    "P01 review.csv: 1 assigned case(s) are still pending"),
    "P02 exclude": (_review(lambda rows: rows[0].update(decision="exclude", reason="x")),
                    "P02 review.csv: 1 assigned case(s) have decision exclude"),
    "C01 cases": (lambda r: (r / "data/cases/dev/cases.csv").write_bytes(b"case_id\n"), "C01"),
    "N01 no case used": (_no_assigned_case, "N01 no case to evaluate (no keep or relabel decision)"),
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
                self.assertTrue(text.rstrip("\n").split("\n")[-1].startswith("refused: "))
                self.assertFalse(out.exists())
                self.assertNotIn(MARK, text)
                self.assertTrue(text.isascii())

    def test_errors_of_several_inputs_are_reported_together(self):
        tmp, root = fresh_root(self)
        shutil.rmtree(root / "runs" / C42)
        (root / "results" / "s4" / "thresholds.json").unlink()
        _review(lambda rows: rows[0].update(decision="pending", reviewer=""))(root)
        code, text = run_main(root, ["--out", str(tmp / "out")])
        self.assertEqual(code, 1)
        for want in ("FAIL P01 ", "FAIL V01 ", "FAIL T01 ", "refused: 3 error(s); nothing written"):
            self.assertIn(want, text)

    def test_a_run_file_changed_after_the_pin_check(self):
        """The A run files are read a second time for the scores: the bytes must still be the pinned ones."""
        tmp, root = fresh_root(self)
        real = E.load_stage2

        def then_changed(*args, **kw):
            out = real(*args, **kw)
            p = root / "runs" / "s2-roberta-A-s123" / "predictions_validation.csv"
            p.write_bytes(p.read_bytes().replace(b",1,0.5,A\n", b",0,0.5,A\n", 1))
            return out

        with mock.patch.object(E, "load_stage2", side_effect=then_changed):
            code, text = run_main(root, ["--out", str(tmp / "out")])
        self.assertEqual(code, 1, text)
        self.assertIn("M02 s2-roberta-A-s123: predictions_validation.csv SHA-256 differs from results/s2/summary.json",
                      text)
        self.assertFalse((tmp / "out").exists())

    def test_protected_output_folders_and_other_root(self):
        tmp, root = fresh_root(self)
        for out in (root / "results" / "s2", root / "results" / "s3", root / "results" / "s3" / "sub"):
            before = sorted(p.name for p in out.iterdir()) if out.exists() else None
            code, text = run_main(root, ["--out", str(out)])
            self.assertEqual(code, 1, text)
            self.assertIn("O01 the output folder must not be results/s2 or results/s3 or inside them", text)
            self.assertEqual(sorted(p.name for p in out.iterdir()) if out.exists() else None, before)
        code, text = run_main(root, ["--out", str(tmp / "out")], lib_root=tmp)
        self.assertEqual(code, 1, text)
        self.assertIn("E04 --root must be the repository root", text)
        self.assertFalse((tmp / "out").exists())

    def test_output_path_that_is_a_file_is_refused_before_the_evaluation(self):
        tmp, root = fresh_root(self)
        f = tmp / "results.txt"
        f.write_bytes(b"not a folder\n")
        early = AssertionError("the evaluation started before the output path was checked")
        with mock.patch.object(E, "load_stage2", side_effect=early), \
                mock.patch.object(E, "predict_runs", side_effect=early):
            for out, shown in ((f, "results.txt"), (f / "sub", "sub"), (f / "a" / "b", "b")):
                code, text = run_main(root, ["--out", str(out)])
                self.assertEqual(code, 1, text)
                self.assertEqual(text, f"FAIL O01 the output folder {shown} is an existing file or lies below one "
                                       "(the outputs are written into a folder)\n"
                                       "refused: 1 error(s); nothing written\n")
                self.assertEqual(f.read_bytes(), b"not a folder\n")
        self.assertEqual(sorted(p.name for p in tmp.iterdir()), ["repo", "results.txt"])
        new = tmp / "new" / "deep"  # a folder that does not exist yet is created, with its parents
        code, text = run_main(root, ["--out", str(new)])
        self.assertEqual(code, 0, text)
        self.assertEqual(sorted(p.name for p in new.iterdir()), sorted(OUT_FILES))

    def test_outputs_are_moved_into_place_only_after_all_are_written(self):
        """A write that fails must leave the earlier outputs as they were (no mixed old and new set) and no
        .tmp file, also not one left by a killed run."""
        tmp, root = fresh_root(self)
        out = tmp / "out"
        out.mkdir()
        for n in OUT_FILES:
            (out / n).write_bytes(b"earlier output\n")
        (out / "summary.json.tmp").write_bytes(b"left by a killed run\n")
        real, written = Path.write_bytes, []

        def fail_on_the_last_file(path, data):
            if path.name == OUT_FILES[-1] + ".tmp":
                raise OSError("disk full")
            written.append(path.name)
            return real(path, data)

        with mock.patch.object(Path, "write_bytes", autospec=True, side_effect=fail_on_the_last_file):
            with self.assertRaises(OSError):
                run_main(root, ["--out", str(out)])
        self.assertEqual(written, [n + ".tmp" for n in OUT_FILES[:-1]])  # six were written, none moved
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(OUT_FILES))
        for n in OUT_FILES:
            self.assertEqual((out / n).read_bytes(), b"earlier output\n", n)
        code, text = run_main(root, ["--out", str(out)])
        self.assertEqual(code, 0, text)
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(OUT_FILES))
        for n in OUT_FILES:
            self.assertNotEqual((out / n).read_bytes(), b"earlier output\n", n)

    def test_thresholds_file_option(self):
        tmp, root = fresh_root(self)
        other = tmp / "thr.json"
        shutil.move(root / "results" / "s4" / "thresholds.json", other)
        code, text = run_main(root, ["--out", str(tmp / "o1")])
        self.assertEqual(code, 1, text)
        code, text = run_main(root, ["--out", str(tmp / "o1"), "--thresholds", str(other)])
        self.assertEqual(code, 0, text)
        doc = json.loads((tmp / "o1" / "summary.json").read_text(encoding="ascii"))
        self.assertEqual(doc["thresholds"]["file"], "thr.json")  # outside the root: the bare name, no path
        self.assertIn("thr.json", doc["inputs_sha256"])


if __name__ == "__main__":
    unittest.main()
