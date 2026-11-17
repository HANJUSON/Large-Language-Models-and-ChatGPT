"""Stage-4 comparison of conditions A-D of the representative model (README 5.3, 6.1, 1.1 RQ-04; design D6).

Run from Git Bash at the repo root after the three C runs and scripts/select_thresholds.py have finished:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/eval_abcd.py
Options: --dir DIR cases folder (default data/cases/dev); --thresholds FILE (default results/s4/thresholds.json);
--out DIR output folder (default results/s4; never results/s2 or results/s3, which hold files of the same
names); --root DIR must be the repository root that eval_lib uses (tests only).

Everything here is a DEVELOPMENT result: validation_clean and the development cases were used to choose the
thresholds and to design the augmentation, so the verdicts below are a development-stage application of the
README 1.1 rules, not the final verdict (that one is made on test_clean and the final cases).

Rules:
  - Conditions (README 5.3, IMP-01), model REPRESENTATIVE (roberta, README 5.1 EXP-04), seeds of
    configs/train.json: A = runs/s2-roberta-A-s<seed> at threshold 0.5; B = the same checkpoints at the B
    thresholds; C = runs/s4-roberta-C-s<seed> at 0.5; D = the same C checkpoints at the D thresholds.
  - One decision rule everywhere (decide): float32(score) >= threshold, computed by eval_lib.predictions on a
    float32 tensor, for stored validation scores and for freshly predicted case scores alike.
  - A runs: pinned by the reviewed record results/s2/ through eval_cases.load_stage2 (selected step,
    checkpoint SHA-256, SHA-256 of run.json / metrics.json / predictions_validation.csv); configs/train.json
    must equal that record (eval_cases.check_config_pin).
  - C runs: verified here like summarize_s2 verifies stage-2 runs: identity (stage s4, condition C, model,
    seed), status completed, official git-tracked lists, clean code state, N = expected_updates and the
    eval steps ceil(N/3), ceil(2N/3), N, model record equal to model_specs, the augmentation recorded
    (data.train_clean_rows = lists.train_clean_rows, data.augmentation_rows > 0, data.train_rows = their sum,
    data.augmentation), reload check passed, checkpoint kept and its size and SHA-256 equal to the file,
    selected step equal to metrics.json and to eval_lib.select, predictions reproducing metrics.json at
    every eval step, condition column C, threshold 0.5.
  - predictions_validation.csv (A and C, read_scores): train.py's header and row format; every stored
    prediction equals float32(score) >= 0.5 and every score is a 9-digit float32 value (so re-thresholding
    the stored text reproduces the float32 score exactly).
  - A against C per seed (SAME_AC; design D6 with its CHANGE on the order of work): budget (N, eval steps,
    updates run, epochs, update size, micro batch, grad accumulation), config SHA-256, list files SHA-256
    (the six list CSVs), dataset revision, model record (weights, precision, tokenizer fingerprint),
    determinism record and the whole environment record (run.json env: python, platform, torch, CUDA,
    cuDNN, the library versions, GPU name, capability, memory and driver, requirements SHA-256) must be
    recorded and equal; no env entry is exempt, and a refusal names the entries that differ. The
    lists_meta SHA-256, the build_lists.py SHA-256 and the code SHA-256 (MAY_DIFFER) may differ between A
    and C, because build_lists.py and train.py change before the C training; both values are reported,
    so each must be recorded and equal among the three A runs and among the three C runs. The
    augmentation record, augmentation rows and train_clean rows must be equal among the three C runs.
    All six runs must hold the same validation rows (IDs, labels, in_clean). The attempt number of every
    source run is reported (README 4.2, 5.3: a retrained run is recorded as a rerun).
  - Thresholds (read_thresholds, the only reader of thresholds.json): a threshold entry is any JSON object
    in the document that has all three keys "threshold" (float), "logit" (number) and "run_id" (string).
    The run_id decides what the entry is: s2-roberta-A-s<seed> is condition B of that seed,
    s4-roberta-C-s<seed> condition D. Exactly one entry per seed and condition is required. threshold must
    be a float32 value in (0, 1); logit a candidate k/20 with -8 <= logit <= 8; |threshold -
    sigmoid(logit)| <= 1e-6. What design D5 has select_thresholds.py record is required, not optional:
    every entry holds "cap_met" (true or false; reported, README 5.3 item 1) and "chosen" with the
    integers tp, fp, fn, tn, which must equal the validation_clean counts this script computes at that
    threshold from the stored scores (the selection and the evaluation use one decision rule); and the
    document records, in a member named runs/<id>/<file>, the SHA-256 of run.json and
    predictions_validation.csv (PINNED_RUN_FILES) of each source run, equal to the file used here (stale
    thresholds). A recorded SHA-256 of another run file used here (metrics.json) is compared when
    present, and so are the optional keys "seed", "condition", "selected_step" and "checkpoint_sha256",
    which must agree with the entry's run as it is used here.
  - Evaluation data: (1) validation_clean = the in_clean rows of predictions_validation.csv at each run's
    selected step, stored scores and margins; (2) the reviewed development cases (eval_cases: cases.csv and
    review.csv checks, keep and relabel used, pending or unapplied exclude refused), predicted with
    eval_cases.predict_runs for the A and C checkpoints (CUDA if available), B and D by re-thresholding
    the float32 scores. For the originals of the cases the decision from the stored score must equal the
    decision from the re-predicted score at every condition's threshold (M08).
  - Metrics (README 6.1, eval_lib.metrics) per condition x seed, then seed mean and sample std (n - 1):
    validation_clean (F1, precision, recall, FPR, ROC-AUC on the margin, confusion counts) and the cells of
    eval_cases (condition, subtype, ROB-03 slices; sides original and variant; labels all, 1, 0) plus the
    pooled cell ROB-01~03 main = the main-group cases of ROB-01 (case, space, char), ROB-02 and ROB-03
    (homoglyph is the held-out group and stays separate); change rates as in eval_cases.
  - Paired bootstrap (README 6.1, eval_cases.bootstrap_dist): groups = original group (group_id; a
    validation row is its own group), merged when texts share a normalised key, stratified by the original
    label, 2,000 resamples, numpy default_rng(314159) restarted per comparison, the same resample indices
    for both conditions, per-seed metric then 3-seed mean then b - a, percentiles 2.5 / 97.5. Comparisons
    A->B, A->C, A->D (primary) and B->D, C->D, B->C (secondary) on: validation_clean recall, FPR and F1;
    variant recall (attack cases) and variant FPR (benign cases) of every cell with slice all and of the
    pooled cell. F1 uses f1_dist: per group and seed 2TP and 2TP + FP + FN are summed over the resampled
    groups (README 6.1: 2TP / (2TP + FP + FN), undefined when the denominator is 0). ROB-04 FPR also gets a
    level row per condition.
  - Development application of README 1.1 RQ-04 to A->B, A->C, A->D: (1) pooled ROB-01~03 main variant
    attack recall difference >= 0.10 and its CI lower bound > 0; (2) CI upper bound of the validation_clean
    FPR increase <= 0.02; (3) CI upper bound of the validation_clean F1 decrease <= 0.01. All three:
    confirmed; (1) without (2) or (3): tradeoff; not (1): not_confirmed. An undefined CI never satisfies a
    condition. Secondary pre-registered result: FPR change on the benign variants of ROB-01 main, ROB-01
    homoglyph and ROB-04: CI upper < 0 fpr_decrease, CI lower > 0 fpr_increase, else no_change.
    Comparisons with 0 and with the limits allow TOL = 1e-9 for float rounding (0.8 -> 0.9 is 0.1).
  - Outputs: runs.csv, metrics.csv, models.csv, bootstrap.csv, predictions.csv, summary.json, summary.md.
    All seven are written as <name>.tmp first and only then moved into place with os.replace; when
    writing a .tmp file fails, the .tmp files are removed and the earlier outputs stay as they were.
    IDs, codes and numbers only (never text or absolute paths), LF, non-integers with 6 decimals
    (thresholds and the margin check keep full precision), no timestamps: the same inputs and device give
    the same bytes.

Refused (FAIL <code> ... lines, "refused: N error(s); nothing written", exit 1):
  E00 config unreadable, eval precision not fp32, threshold not 0.5, or not 3 distinct integer seeds;
  E01 list errors of data_lists.load_rows; E03 config differs from the stage-2 record; E04 --root is not
  eval_lib.ROOT; O01 output folder is results/s2 or results/s3 or inside one of them, or the output path
  is an existing file or lies below one (checked before anything is evaluated);
  C01-C03, R01-R09, E10, P01, P02, N01: cases and review, as in eval_cases;
  M01-M03 stage-2 record and A run files (eval_cases.load_stage2; M02/M03 also for the second read here);
  M04 checkpoint refused by load_run_model; M05 re-predicted validation predictions differ from the stored
  ones; M06 non-finite margins; M08 stored-score and re-predicted decisions differ for case originals;
  V01 C run folder or file missing, or runs/<id>.lock exists; V02 run.json / metrics.json not JSON, a key
  missing, or the identity differs from the run ID; V03 status not completed; V04 checkpoint (reload
  check, kept, path, size, SHA-256, step); V05 metrics.json against run.json and the selection rule;
  V06 predictions_validation.csv malformed, inconsistent or not reproducing metrics.json; V07 a
  precondition of a C run is not recorded; V08 A and C records differ, or the runs of one trained
  condition differ among themselves; V09 validation rows differ between runs;
  T01 thresholds.json unreadable or not a JSON object; T02 entries missing, duplicated or for an unexpected
  run; T03 an entry's values are invalid, cap_met or the chosen counts are missing or malformed, the
  SHA-256 of a pinned run file is not recorded, or a recorded value disagrees with the runs used here.
"""
import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import data_lists

ROOT = data_lists.ROOT
CONFIG_REL = "configs/train.json"

if __name__ == "__main__":
    data_lists.install_test_guard()  # before torch, numpy, sklearn and the rest are imported (as in train.py)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = json.loads(
        (ROOT / CONFIG_REL).read_bytes())["determinism"]["cublas_workspace_config"]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import build_lists  # noqa: E402
import eval_cases as E  # noqa: E402
import eval_lib  # noqa: E402
import model_specs  # noqa: E402
import summarize_s2  # noqa: E402
import train  # noqa: E402

SCRIPT_REL = "scripts/eval_abcd.py"
DIR_REL = E.DIR_REL
OUT_REL = "results/s4"
THRESHOLDS_REL = "results/s4/thresholds.json"
PROTECTED_OUT = ("results/s2", "results/s3")  # reviewed records whose file names this script also writes
REPRESENTATIVE = "roberta"  # README 5.1 EXP-04
STAGE_A, STAGE_C = "s2", "s4"
CONDITIONS = ("A", "B", "C", "D")
TRAINED = {"A": "A", "B": "A", "C": "C", "D": "C"}  # the trained condition whose checkpoints are used (IMP-01)
TUNED = {"B": "A", "D": "C"}  # conditions with a threshold from thresholds.json, and their source condition
PRIMARY = (("A", "B"), ("A", "C"), ("A", "D"))  # README 6.1 item 4
SECONDARY = (("B", "D"), ("C", "D"), ("B", "C"))
THRESHOLD = summarize_s2.THRESHOLD  # 0.5, conditions A and C
THRESHOLD_KEYS = ("threshold", "logit", "run_id")  # the keys of a thresholds.json entry this script relies on
LOGIT_MAX, LOGIT_STEPS = 8.0, 20  # README 5.3 candidates: logit k / 20, k = -160..160
SIGMOID_TOL = 1e-6  # |threshold - sigmoid(logit)|; neighbouring candidates differ by at least 1.6e-5
DELTA_R, DELTA_F, DELTA_O = 0.10, 0.02, 0.01  # README 1.1
TOL = 1e-9  # float rounding allowance in the verdict comparisons
VAL_SET = "validation_clean"
VAL_METRICS = ("recall", "fpr", "f1")
POOLED_SET, POOLED_CONDITIONS = "ROB-01~03", ("ROB-01", "ROB-02", "ROB-03")
FPR_TARGETS = (("ROB-01", "main"), ("ROB-01", "homoglyph"), ("ROB-04", "benign_security"))
RUN_FILES = summarize_s2.INPUT_FILES
PINNED_RUN_FILES = ("run.json", "predictions_validation.csv")  # thresholds.json must record their SHA-256
BUDGET_KEYS = ("N", "eval_steps", "updates_run", "epochs", "update_size", "micro_batch", "grad_accum")
# Records that must be recorded and equal in the A and C run of a seed (README 5.3 IMP-02, IMP-03; design D6
# and its CHANGE). "environment" is the whole env record of run.json: no entry is exempt.
SAME_AC = (tuple((f"budget {k}", ("budget", k)) for k in BUDGET_KEYS)
           + (("config SHA-256", ("config", "sha256")), ("list files SHA-256", ("lists", "files_sha256")),
              ("dataset revision", ("lists", "dataset_revision")),
              ("model record (weights, precision, tokenizer)", ("model",)), ("determinism", ("determinism",)),
              ("environment", ("env",))))
# Records that may differ between A and C (build_lists.py and train.py change before the C training; design
# D6 CHANGE): both values are reported. They must be recorded and equal in the three C runs.
MAY_DIFFER = (("lists_meta SHA-256", ("lists", "lists_meta_sha256")),
              ("build_lists SHA-256", ("lists", "build_lists_sha256_lf")),
              ("code files SHA-256", ("code", "files_sha256_lf")))
# Records that must be recorded and equal in the three C runs.
SAME_C = MAY_DIFFER + (("augmentation record", ("data", "augmentation")),
                       ("augmentation rows", ("data", "augmentation_rows")),
                       ("train_clean rows", ("data", "train_clean_rows")))
AUGMENTATION_KEYS = ("files_sha256", "label_counts", "build_train_aug_sha256_lf")  # reported (no paths)
VAL_COLUMNS = tuple(f"{VAL_SET}_{k}" for k in ("n",) + summarize_s2.METRICS)
RUN_COLUMNS = (("condition", "seed", "run_id", "trained_condition", "selected_step", "checkpoint_sha256",
                "threshold", "logit") + VAL_COLUMNS)
CELL_COLUMNS = ("eval_set", "subtype", "slice")
METRIC_COLUMNS = (("condition", "seed", "run_id", "threshold") + CELL_COLUMNS + ("side", "label", "n")
                  + E.COUNTS + E.METRICS)
CHANGE_COLUMNS = (("condition", "seed", "run_id", "threshold") + CELL_COLUMNS
                  + ("label", "n_pairs", "changed", "to_attack", "to_benign", "change_rate"))
BOOT_COLUMNS = ("kind", "condition_a", "condition_b", "eval_set", "subtype", "side", "metric", "n_groups", "strata",
                "estimate_a", "estimate_b", "diff", "diff_low", "diff_high", "ci_vs_0", "a_low", "a_high", "b_low",
                "b_high", "undefined_resamples", "nk_merges")
PRED_COLUMNS = ("case_id", "run_id", "condition", "margin", "score", "prediction", "threshold")
FULL_PRECISION = E.FULL_PRECISION + ("threshold",)  # summary.json keys not rounded to 6 decimals
RQ04_TEXT = {"confirmed": "개선 확인", "tradeoff": "상충 있는 개선", "not_confirmed": "개선 확인 못함"}
FPR_TEXT = {"fpr_decrease": "오탐 감소 확인", "fpr_increase": "오탐 증가 확인", "no_change": "변화 확인 못함"}
MAY_DIFFER_TEXT = {"lists_meta SHA-256": "`lists_meta.json` SHA-256", "build_lists SHA-256": "`build_lists.py` SHA-256",
                   "code files SHA-256": "학습 코드 SHA-256"}  # summary.md names of the MAY_DIFFER records


def source_run_id(trained, seed):
    return f"{STAGE_A if trained == 'A' else STAGE_C}-{REPRESENTATIVE}-{trained}-s{seed}"


# ---------------------------------------------------------------- decision rule and stored predictions

def decide(scores, threshold):
    """Predictions under the one decision rule used everywhere: float32(score) >= threshold, computed by
    eval_lib.predictions on a float32 tensor (the call train.py and eval_cases.predict_runs make)."""
    return eval_lib.predictions(torch.tensor(list(scores), dtype=torch.float32), threshold).tolist()


def read_scores(raw, steps, condition):
    """({"ids", "labels", "in_clean", "pred", "margin", "score": {step: []}}, errors) from the bytes of a
    predictions_validation.csv of the trained condition A or C.

    summarize_s2.read_predictions with the condition as a parameter and the score kept (as the float32
    value). On top: predictions are 0/1, margins finite, scores in [0, 1]; the stored prediction must
    equal decide(score, 0.5) and each score must be a 9-digit float32 value (eval_lib.fmt9)."""
    try:
        lines = raw.decode("ascii").split("\n")
    except UnicodeDecodeError:
        return None, ["not ASCII"]
    if lines[0] != train.PRED_HEADER or lines[-1] != "":
        return None, ["header or final LF differs"]
    if not (isinstance(steps, list) and steps and all(type(s) is int for s in steps)):
        return None, ["the eval steps of run.json are not a non-empty list of integers"]
    by = {s: [] for s in steps}
    for n, ln in enumerate(lines[1:-1], start=2):
        try:
            step, sid, y, c, m_text, s_text, p, thr, cond = ln.split(",")
            step, y, c, p, m, sc, thr = int(step), int(y), int(c), int(p), float(m_text), float(s_text), float(thr)
        except ValueError:
            return None, [f"line {n} is malformed"]
        if (step not in by or y not in (0, 1) or c not in (0, 1) or p not in (0, 1) or thr != THRESHOLD
                or cond != condition or not math.isfinite(m) or not 0.0 <= sc <= 1.0):
            return None, [f"line {n}: step not an eval step, label/in_clean/prediction not 0/1, threshold not "
                          f"{THRESHOLD}, condition not {condition}, margin not finite, or score outside [0, 1]"]
        by[step].append((sid, y, c, p, m, sc, s_text))
    first = [r[:3] for r in by[steps[0]]]
    ids = [r[0] for r in first]
    if not ids or any(not b > a for a, b in zip(ids, ids[1:])):
        return None, [f"step {steps[0]}: no rows, or sentence IDs not strictly ascending"]
    score = {}
    for s in steps:
        if [r[:3] for r in by[s]] != first:
            return None, [f"step {s}: sentence IDs, labels or in_clean differ from step {steps[0]}"]
        score[s] = torch.tensor([r[5] for r in by[s]], dtype=torch.float32).tolist()
        if [eval_lib.fmt9(v) for v in score[s]] != [r[6] for r in by[s]]:
            return None, [f"step {s}: a score is not a 9-digit float32 value (eval_lib.fmt9)"]
        if decide(score[s], THRESHOLD) != [r[3] for r in by[s]]:
            return None, [f"step {s}: a stored prediction is not float32(score) >= {THRESHOLD}"]
    return {"ids": ids, "labels": [r[1] for r in first], "in_clean": [r[2] for r in first],
            "pred": {s: [r[3] for r in by[s]] for s in steps}, "margin": {s: [r[4] for r in by[s]] for s in steps},
            "score": score}, []


# ---------------------------------------------------------------- source runs (A from results/s2, C verified)

def a_source(root, run, shas):
    """(source-run record, errors): an A run of eval_cases.load_stage2 plus its run.json document, in_clean
    and stored scores. The two files are read again; their SHA-256 must be the pinned one."""
    rid, raw = run["run_id"], {}
    for n in ("run.json", "predictions_validation.csv"):
        rel = f"runs/{rid}/{n}"
        try:
            raw[n] = (root / rel).read_bytes()
        except OSError:
            return None, [f"M02 {rid}: {n} missing"]
        if data_lists.sha256_bytes(raw[n]) != shas[rel]:
            return None, [f"M02 {rid}: {n} SHA-256 differs from {E.S2_REL}/summary.json"]
    try:
        doc = json.loads(raw["run.json"])
        steps = doc["budget"]["eval_steps"]
    except (ValueError, KeyError, TypeError) as exc:
        return None, [f"M03 {rid}: run.json unreadable or missing keys ({type(exc).__name__})"]
    P, perrs = read_scores(raw["predictions_validation.csv"], steps, "A")
    if perrs or run["step"] not in P["pred"]:
        return None, [f"M03 {rid}: predictions_validation.csv: {'; '.join(perrs) or 'no selected step'}"]
    step = run["step"]
    if (P["ids"], P["labels"], P["pred"][step], P["margin"][step]) != (run["ids"], run["labels"], run["pred"],
                                                                      run["margin"]):
        return None, [f"M03 {rid}: predictions_validation.csv read differently by eval_cases.load_stage2"]
    return {**run, "trained": "A", "attempt": doc.get("attempt"), "in_clean": P["in_clean"],
            "score": P["score"][step], "doc": doc,
            "files": {f"runs/{rid}/{n}": shas[f"runs/{rid}/{n}"] for n in RUN_FILES}}, []


def check_c_run(d, run_id, seed, run, met, raw, cfg, errs):
    """Checks of one parsed C run (appends to errs); returns the source-run record or None. May raise
    KeyError/TypeError. The checks of summarize_s2.check_run that apply to a stage-4 run."""
    tag = f"{run_id}:"
    ident = {"run_id": run_id, "stage": STAGE_C, "model_key": REPRESENTATIVE, "condition": "C", "seed": seed}
    bad = [k for k, v in ident.items() if run.get(k) != v]
    if bad:
        errs.append(f"V02 {tag} run.json {', '.join(bad)} do not match the run ID")
    if run.get("status") != "completed":
        errs.append(f"V03 {tag} status is {run.get('status')!r}, not 'completed'")
        return None
    budget, lists, mrec, data = run["budget"], run["lists"], run["model"], run["data"]
    sel, steps, n_total = run["selection"]["selected_step"], budget["eval_steps"], budget["N"]
    if lists.get("mode") != "official" or lists.get("git_tracked_checked") is not True:
        errs.append(f"V07 {tag} lists are not official with the git-tracked check (mode {lists.get('mode')!r}, "
                    f"git_tracked_checked {lists.get('git_tracked_checked')!r})")
    if run["code"].get("git_dirty") != []:
        errs.append(f"V07 {tag} code.git_dirty is not empty (code, config or list files not clean at the start)")
    if budget.get("updates_run") != n_total or n_total != cfg["expected_updates"] or steps != train.eval_steps(n_total):
        errs.append(f"V07 {tag} budget: updates_run {budget.get('updates_run')}, N {n_total} (config expected_updates "
                    f"{cfg['expected_updates']}), eval steps {steps} (want ceil(N/3), ceil(2N/3), N)")
    pins = (REPRESENTATIVE, *model_specs.MODELS[REPRESENTATIVE], model_specs.WEIGHTS[REPRESENTATIVE][1],
            model_specs.TOKENIZER_SHA256[REPRESENTATIVE])
    if (mrec.get("key"), mrec.get("repo"), mrec.get("revision"), (mrec.get("weights") or {}).get("sha256"),
            mrec.get("tokenizer_fingerprint")) != pins:
        errs.append(f"V07 {tag} model record differs from model_specs (key, repo, revision, weights sha256, "
                    "tokenizer fingerprint)")
    clean_rows, aug_rows = data.get("train_clean_rows"), data.get("augmentation_rows")
    if not (type(clean_rows) is int and type(aug_rows) is int and aug_rows > 0
            and clean_rows == lists.get("train_clean_rows") and data.get("train_rows") == clean_rows + aug_rows
            and isinstance(data.get("augmentation"), dict)):
        errs.append(f"V07 {tag} the augmentation is not recorded: data.train_clean_rows {clean_rows!r} (lists "
                    f"{lists.get('train_clean_rows')!r}), augmentation_rows {aug_rows!r} (> 0), train_rows "
                    f"{data.get('train_rows')!r} (their sum), data.augmentation (an object)")
    ck = run["checkpoint"]
    rel = f"runs/{run_id}/checkpoint/{model_specs.CHECKPOINT_FILE}"
    if (ck.get("reload_check") or {}).get("passed") is not True:
        errs.append(f"V04 {tag} the checkpoint reload check did not pass")
    if (ck.get("kept") is not True or ck.get("path") != rel or not isinstance(ck.get("sha256"), str)
            or not isinstance(ck.get("size"), int)):
        errs.append(f"V04 {tag} checkpoint not kept, path not {rel}, or sha256/size not recorded")
    else:
        f = d / "checkpoint" / model_specs.CHECKPOINT_FILE
        if not f.is_file() or f.stat().st_size != ck["size"]:
            errs.append(f"V04 {tag} {rel} is missing or its size differs from run.json")
        elif model_specs.sha256_file(f) != ck["sha256"]:
            errs.append(f"V04 {tag} {rel} SHA-256 differs from run.json")
    if ck.get("step") != sel:
        errs.append(f"V04 {tag} checkpoint step {ck.get('step')} != selected step {sel}")
    if met.get("run_id") != run_id or met.get("threshold") != THRESHOLD:
        errs.append(f"V05 {tag} metrics.json run_id or threshold ({met.get('threshold')}) differs")
    if met.get("eval_steps") != steps or [e["step"] for e in met["evals"]] != steps:
        errs.append(f"V05 {tag} metrics.json eval steps differ from run.json budget {steps}")
        return None
    if met.get("selected_step") != sel:
        errs.append(f"V05 {tag} metrics.json selected_step {met.get('selected_step')} != run.json {sel}")
    else:
        try:
            if eval_lib.select(met["evals"])["step"] != sel:
                errs.append(f"V05 {tag} selected step {sel} does not follow the selection rule (eval_lib.select)")
        except ValueError as exc:
            errs.append(f"V05 {tag} selection key undefined in metrics.json ({exc})")
    P, perrs = read_scores(raw["predictions_validation.csv"], steps, "C")
    if perrs:
        errs += [f"V06 {tag} predictions_validation.csv: {e}" for e in perrs]
        return None
    if len(P["ids"]) != data["validation_rows"] or sum(P["in_clean"]) != data["validation_clean_rows"]:
        errs.append(f"V06 {tag} prediction row counts differ from run.json data")
    for e in met["evals"]:
        try:
            c = summarize_s2.metrics_at(P, e["step"])
        except ValueError as exc:  # sklearn input validation
            errs.append(f"V06 {tag} step {e['step']}: metrics failed ({type(exc).__name__}: {exc})")
            return None
        for s in summarize_s2.SUBSETS:
            diff = [k for k in summarize_s2.EXACT_KEYS if c[s][k] != e[s].get(k)]
            a, b = c[s]["roc_auc"], e[s].get("roc_auc")
            if (a is None) != (b is None) or (a is not None and abs(a - b) > summarize_s2.AUC_TOL):
                diff.append("roc_auc")
            if diff:
                errs.append(f"V06 {tag} step {e['step']} {s}: predictions do not reproduce metrics.json "
                            f"({', '.join(diff)})")
    if errs:
        return None
    return {"model": REPRESENTATIVE, "seed": seed, "run_id": run_id, "trained": "C", "attempt": run.get("attempt"),
            "step": sel, "sha256": ck["sha256"], "size": ck["size"], "ids": P["ids"], "labels": P["labels"],
            "in_clean": P["in_clean"], "pred": P["pred"][sel], "margin": P["margin"][sel], "score": P["score"][sel],
            "doc": run, "files": {f"runs/{run_id}/{n}": data_lists.sha256_bytes(raw[n]) for n in RUN_FILES}}


def load_c_run(root, seed, cfg):
    """(source-run record, errors) of the C run of one seed; the record is None when any check fails."""
    run_id = source_run_id("C", seed)
    d, tag = root / "runs" / run_id, f"{run_id}:"
    if train.lock_path(d).exists():
        return None, [f"V01 {tag} runs/{run_id}.lock exists (the run is in progress, or was killed)"]
    if not d.is_dir():
        return None, [f"V01 {tag} runs/{run_id}/ is missing"]
    missing = [n for n in RUN_FILES if not (d / n).is_file()]
    if missing:
        return None, [f"V01 {tag} missing {', '.join(missing)}"]
    raw = {n: (d / n).read_bytes() for n in RUN_FILES}
    try:
        run, met = json.loads(raw["run.json"]), json.loads(raw["metrics.json"])
    except ValueError:
        return None, [f"V02 {tag} run.json or metrics.json is not valid JSON"]
    errs = []
    try:
        rec = check_c_run(d, run_id, seed, run, met, raw, cfg, errs)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        return None, errs + [f"V02 {tag} run.json or metrics.json: missing key or wrong type "
                             f"({type(exc).__name__}: {str(exc)[:80]})"]
    return (None, errs) if errs else (rec, [])


def _leaves(d, prefix=""):
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else ()):
        if isinstance(v, dict):
            out.update(_leaves(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _differing(name, va, vc):
    """The record name, followed in brackets by the entries of two objects that differ (when there are any)."""
    la, lc = _leaves(va), _leaves(vc)
    keys = [k for k in sorted(set(la) | set(lc)) if la.get(k) != lc.get(k)]
    return f"{name} [{', '.join(keys)}]" if keys else name


def compare_records(sources):
    """(errors, record summary) of the six source runs (three A, three C, each in seed order)."""
    a = {r["seed"]: r for r in sources if r["trained"] == "A"}
    c = {r["seed"]: r for r in sources if r["trained"] == "C"}
    errs = []
    for seed in a:
        da, dc = a[seed]["doc"], c[seed]["doc"]
        differ = [_differing(name, train._get(da, path), train._get(dc, path)) for name, path in SAME_AC
                  if train._get(da, path) is None or train._get(da, path) != train._get(dc, path)]
        if differ:
            errs.append(f"V08 {c[seed]['run_id']} against {a[seed]['run_id']}: differs or is not recorded: "
                        f"{', '.join(differ)}")
    # one value per trained condition: what may differ between A and C is reported as "the A value" and "the C
    # value", so it must be the same in the three runs of a condition
    for t, runs, same in (("A", a, MAY_DIFFER), ("C", c, SAME_C)):
        for name, path in same:
            groups = {}
            for r in runs.values():
                groups.setdefault(json.dumps(train._get(r["doc"], path), sort_keys=True), []).append(r["run_id"])
            if len(groups) > 1 or "null" in groups:
                errs.append(f"V08 {name} differs between the {t} runs or is not recorded: "
                            + " | ".join(", ".join(ids) for ids in groups.values()))
    rows = [(r["ids"], r["labels"], r["in_clean"]) for r in sources]
    other = [r["run_id"] for r, x in zip(sources, rows) if x != rows[0]]
    if other:
        errs.append(f"V09 validation rows (IDs, labels, in_clean) differ from {sources[0]['run_id']}: "
                    f"{', '.join(other)}")
    code = {}
    for t, runs in (("A", a), ("C", c)):
        docs = [r["doc"] for r in runs.values()]
        code[t] = {"files_sha256_lf": train._get(docs[0], ("code", "files_sha256_lf")),
                   "git_commits": sorted({str(train._get(d, ("code", "git_commit"))) for d in docs})}
    fa, fc = (code[t]["files_sha256_lf"] if isinstance(code[t]["files_sha256_lf"], dict) else {} for t in "AC")
    da, dc = next(iter(a.values()))["doc"], next(iter(c.values()))["doc"]
    aug = train._get(dc, ("data", "augmentation"))
    info = {"equal_in_A_and_C": [name for name, _ in SAME_AC], "equal_in_A_runs": [name for name, _ in MAY_DIFFER],
            "equal_in_C_runs": [name for name, _ in SAME_C],
            "may_differ_between_A_and_C": [name for name, _ in MAY_DIFFER],
            "differing_between_A_and_C": [name for name, path in MAY_DIFFER
                                          if train._get(da, path) != train._get(dc, path)],
            "budget": {k: train._get(da, ("budget", k)) for k in BUDGET_KEYS},
            "config_sha256": train._get(da, ("config", "sha256")),
            "lists_files_sha256": train._get(da, ("lists", "files_sha256")),
            "dataset_revision": train._get(da, ("lists", "dataset_revision")),
            "lists_meta_sha256": {t: train._get(d, ("lists", "lists_meta_sha256")) for t, d in (("A", da), ("C", dc))},
            "build_lists_sha256_lf": {t: train._get(d, ("lists", "build_lists_sha256_lf"))
                                      for t, d in (("A", da), ("C", dc))},
            "code": {**code, "files_differing": sorted(k for k in set(fa) | set(fc) if fa.get(k) != fc.get(k))},
            "augmentation": {"rows": train._get(dc, ("data", "augmentation_rows")),
                             "train_clean_rows": train._get(dc, ("data", "train_clean_rows")),
                             "train_rows": train._get(dc, ("data", "train_rows")),
                             **{k: aug.get(k) if isinstance(aug, dict) else None for k in AUGMENTATION_KEYS}},
            "env": train._get(da, ("env",))}
    return errs, info


# ---------------------------------------------------------------- thresholds

def _objects(node):
    """Every JSON object of a parsed document, depth first in document order."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _objects(v)
    elif isinstance(node, list):
        for v in node:
            yield from _objects(v)


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def read_thresholds(raw, seeds, runs=None, file_shas=None):
    """({(seed, "B"|"D"): {"threshold", "logit", "run_id", "cap_met", "selection_counts"}}, info, errors)
    from the thresholds.json bytes.

    The only reader of that file. A threshold entry is any JSON object (at any depth) holding all of
    THRESHOLD_KEYS: "threshold" (float32 value in (0, 1) stored as a float), "logit" (candidate k / 20
    within -8..8) and "run_id" (the source run: s2-<model>-A-s<seed> is condition B, s4-<model>-C-s<seed>
    condition D). Exactly one entry per seed and condition. Every entry must hold "cap_met" (a boolean)
    and "chosen" with the integers tp, fp, fn, tn; selection_counts is those four counts, for
    selection_count_errors (both None where the entry is refused for them). Optional members are checked
    when present: "seed", "condition", and against the source runs used here (runs: {run_id:
    {"selected_step", "checkpoint_sha256"}}) "selected_step" and "checkpoint_sha256".

    file_shas ({repo-relative path: sha256} of the run files used here): a string member of that name
    anywhere in the document must equal the SHA-256, and for the PINNED_RUN_FILES among them at least one
    such member must exist (the thresholds must say which run files they were selected on)."""
    name = "thresholds.json"
    try:
        doc = json.loads(raw)
    except ValueError:
        return None, None, [f"T01 {name}: not valid JSON"]
    if not isinstance(doc, dict):
        return None, None, [f"T01 {name}: the document is not a JSON object"]
    expected = {source_run_id(TUNED[c], s): (s, c) for s in seeds for c in TUNED}
    out, errs = {}, []
    for o in _objects(doc):
        if not all(k in o for k in THRESHOLD_KEYS):
            continue
        rid, thr, t = o["run_id"], o["threshold"], o["logit"]
        if not isinstance(rid, str) or rid not in expected:
            errs.append(f"T02 {name}: entry for an unexpected run {str(rid)[:60]!r} (want "
                        f"{', '.join(sorted(expected))})")
            continue
        seed, cond = expected[rid]
        tag = f"{name} {cond} s{seed}:"
        if (seed, cond) in out:
            errs.append(f"T02 {tag} more than one entry for {rid}")
            continue
        chosen, cap_met = o.get("chosen"), o.get("cap_met")
        counts = {k: chosen.get(k) for k in E.COUNTS} if isinstance(chosen, dict) else {}
        if not (counts and all(type(v) is int for v in counts.values())):
            counts = None
            errs.append(f"T03 {tag} chosen must hold the integer counts {', '.join(E.COUNTS)} at the chosen "
                        "threshold (design D5)")
        if not isinstance(cap_met, bool):
            cap_met = None
            errs.append(f"T03 {tag} cap_met must be true or false (design D5)")
        out[(seed, cond)] = {"threshold": thr, "logit": t, "run_id": rid, "cap_met": cap_met,
                             "selection_counts": counts}
        if not (isinstance(thr, float) and 0.0 < thr < 1.0 and float(np.float32(thr)) == thr):
            errs.append(f"T03 {tag} threshold is not a float32 value in (0, 1) stored as a float")
        elif not (_number(t) and abs(t) <= LOGIT_MAX + SIGMOID_TOL
                  and abs(t * LOGIT_STEPS - round(t * LOGIT_STEPS)) <= SIGMOID_TOL):
            errs.append(f"T03 {tag} logit is not a candidate k/{LOGIT_STEPS} within -{LOGIT_MAX:g}..{LOGIT_MAX:g}")
        elif abs(thr - 1.0 / (1.0 + math.exp(-t))) > SIGMOID_TOL:
            errs.append(f"T03 {tag} threshold is not sigmoid(logit) (float32)")
        used = (runs or {}).get(rid, {})
        for key, want in (("seed", seed), ("condition", cond), ("selected_step", used.get("selected_step")),
                          ("checkpoint_sha256", used.get("checkpoint_sha256"))):
            if key in o and want is not None and (o[key] != want or isinstance(o[key], bool)):
                errs.append(f"T03 {tag} {key} {str(o[key])[:20]!r} disagrees with the run {rid} used here "
                            f"({str(want)[:20]!r})")
    missing = sorted(rid for rid, k in expected.items() if k not in out)
    if missing:
        errs.append(f"T02 {name}: no entry (an object with the keys {', '.join(THRESHOLD_KEYS)}) for "
                    f"{', '.join(missing)}")
    pins, recorded = 0, set()
    for o in _objects(doc):
        for k, v in o.items():
            if k in (file_shas or {}) and isinstance(v, str):
                pins += 1
                recorded.add(k)
                if v != file_shas[k]:
                    errs.append(f"T03 {name}: the recorded SHA-256 of {k} differs from the file used here "
                                "(the thresholds were selected on other run files)")
    unpinned = sorted(k for k in (file_shas or {}) if k.rsplit("/", 1)[-1] in PINNED_RUN_FILES and k not in recorded)
    if unpinned:
        errs.append(f"T03 {name}: no SHA-256 is recorded for {len(unpinned)} run file(s) used here ("
                    f"{', '.join(unpinned)}); the document must record {' and '.join(PINNED_RUN_FILES)} of every "
                    "source run (design D5)")
    return out, {"entries": len(out), "run_file_sha256_checked": pins}, errs


def selection_count_errors(thresholds, sources):
    """(errors, entries checked): the validation_clean counts an entry records for its threshold (member
    "chosen": tp, fp, fn, tn; read_thresholds refuses an entry without them) must equal the counts this
    script gets from the stored scores of the entry's run, so selection and evaluation decide the same
    rows the same way. Every entry whose source run was loaded is checked."""
    src = {r["run_id"]: r for r in sources}
    errs, checked = [], 0
    for (seed, cond), t in sorted(thresholds.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        s, want = src.get(t["run_id"]), t["selection_counts"]
        if s is None:
            continue
        keep = [i for i, c in enumerate(s["in_clean"]) if c]
        m = eval_lib.metrics([s["labels"][i] for i in keep], decide([s["score"][i] for i in keep], t["threshold"]))
        got = {k: m[k] for k in E.COUNTS}
        checked += 1
        if got != want:
            errs.append(f"T03 thresholds.json {cond} s{seed}: the recorded counts at the chosen threshold "
                        f"({', '.join(f'{k} {want[k]}' for k in E.COUNTS)}) differ from the {VAL_SET} counts of "
                        f"{t['run_id']} here ({', '.join(f'{k} {got[k]}' for k in E.COUNTS)})")
    return errs, checked


# ---------------------------------------------------------------- conditions and their predictions

def condition_runs(seeds, thresholds):
    """The evaluated runs (condition x seed, condition order then seed order) in the shape eval_cases
    aggregates: "model" is the condition A-D and "run_id" the key of the prediction table; "source" is
    the run folder whose checkpoint and stored scores are used. "cap_met" is the recorded answer to "did
    a candidate meet the FPR cap" of a tuned threshold (None for A and C, which have no cap)."""
    out = []
    for cond in CONDITIONS:
        for seed in seeds:
            t = thresholds[(seed, cond)] if cond in TUNED else {"threshold": THRESHOLD, "logit": None}
            out.append({"model": cond, "seed": seed, "run_id": f"{cond}-s{seed}",
                        "source": source_run_id(TRAINED[cond], seed), "threshold": t["threshold"],
                        "logit": t["logit"], "cap_met": t.get("cap_met")})
    return out


def rethreshold(cruns, sources, P6):
    """({key: {"case", "orig", "val": {id: (margin, score, prediction)}}}, checks, errors).

    case / orig: the re-predicted float32 scores of eval_cases.predict_runs, decided at the condition's
    threshold; val: the validation_clean rows with the stored margin and score of the selected step. For
    the originals the stored score must give the same decision as the re-predicted one (M08)."""
    src = {r["run_id"]: r for r in sources}
    P, checks, errs = {}, {}, []
    for cr in cruns:
        s, thr = src[cr["source"]], cr["threshold"]
        out = {}
        for kind in ("case", "orig"):
            got = P6[s["run_id"]][kind]
            ids = list(got)
            out[kind] = {i: (got[i][0], got[i][1], p)
                         for i, p in zip(ids, decide([got[i][1] for i in ids], thr))}
        stored = decide(s["score"], thr)
        out["val"] = {sid: (s["margin"][i], s["score"][i], stored[i])
                      for i, sid in enumerate(s["ids"]) if s["in_clean"][i]}
        row = {sid: i for i, sid in enumerate(s["ids"])}
        flips = sum(v[2] != stored[row[sid]] for sid, v in out["orig"].items())
        if flips:
            errs.append(f"M08 {s['run_id']} (condition {cr['model']}, threshold {thr!r}): {flips} of "
                        f"{len(out['orig'])} case originals get another decision from the stored score than "
                        "from the re-predicted score")
        checks[cr["run_id"]] = {"originals": len(out["orig"]), "stored_vs_repredicted_flips": flips}
        P[cr["run_id"]] = out
    return P, checks, errs


# ---------------------------------------------------------------- aggregation

def pooled_main(recs):
    return [r for r in recs if r["condition"] in POOLED_CONDITIONS and r["eval_group"] == "main"]


def eval_cells(recs, val_items):
    """[(eval set, subtype, slice, {side: items})] in report order: validation_clean, the eval_cases cells,
    then the pooled ROB-01~03 main cell."""
    cells = [(VAL_SET, "all", "all", {"original": val_items})]
    for cond, sub, sl, rs in E.cell_defs(recs):
        cells.append((cond, sub, sl, {side: E.side_items(rs, side) for side in E.sides(cond)}))
    pooled = pooled_main(recs)
    if pooled:
        cells.append((POOLED_SET, "main", "all", {side: E.side_items(pooled, side)
                                                 for side in ("original", "variant")}))
    return cells


def metric_rows(cells, cruns, P):
    """eval_cases.metric_rows over the given cells (rows in the eval_cases shape: "model" = condition A-D,
    "condition" = eval set)."""
    rows = []
    for run in cruns:
        pr = P[run["run_id"]]
        for cond, sub, sl, by_side in cells:
            for side, items in by_side.items():
                for lab in E.label_sets(items):
                    sel = E._pick(items, lab)
                    vals = [pr[k][i] for _, _, (k, i) in sel]
                    m = eval_lib.metrics([y for _, y, _ in sel], [v[2] for v in vals], [v[0] for v in vals])
                    rows.append({"model": run["model"], "seed": run["seed"], "run_id": run["run_id"],
                                 "condition": cond, "subtype": sub, "slice": sl, "side": side, "label": lab,
                                 "n": m["n"],
                                 **{q: m[q] if q in E.REPORTED[lab] else None for q in E.COUNTS + E.METRICS}})
    return rows


def f1_dist(items, runs_a, runs_b, stratum, P, nk_of=None):
    """F1 counterpart of eval_cases.bootstrap_dist for one item list under two run sets (same result keys).

    Groups, normalised-key merging, strata and resample indices are those of bootstrap_dist. Per group and
    run, 2TP and 2TP + FP + FN are counted; per resample F1 = sum(2TP) / sum(2TP + FP + FN) per run (NaN
    when the denominator is 0), then the mean over the runs, then b - a. runs_b None gives a level."""
    merged = E.merge_groups(items, nk_of or {})
    members = {}
    for g, m in merged.items():
        members.setdefault(m, []).append(g)
    groups = sorted(members)
    strata = []
    for m in groups:
        s = {stratum[g] for g in members[m]}
        strata.append(next(iter(s)) if len(s) == 1 else "mixed")
    idx = E.resample_indices(strata)
    gi = {g: i for i, g in enumerate(groups)}
    out = {"groups": groups, "strata": strata, "b": None, "diff": None,
           "merged": [sorted(members[m]) for m in groups if len(members[m]) > 1]}
    for name, rids in (("a", runs_a), ("b", runs_b)):
        if rids is None:
            continue
        num = np.zeros((len(rids), len(groups)), dtype=np.int64)  # 2 TP
        den = np.zeros((len(rids), len(groups)), dtype=np.int64)  # 2 TP + FP + FN
        for g, y, (kind, i) in items:
            col = gi[merged[g]]
            for j, rid in enumerate(rids):
                p = P[rid][kind][i][2]
                num[j, col] += 2 * (y == 1 and p == 1)
                den[j, col] += (y == 1) + (p == 1)
        full_num, full_den = num.sum(axis=1), den.sum(axis=1)
        est = (statistics.fmean(int(n) / int(d) for n, d in zip(full_num, full_den))
               if (full_den > 0).all() else None)
        rate = np.empty((len(rids), idx.shape[0]), dtype=np.float64)
        for j in range(len(rids)):
            rnum, rden = num[j][idx].sum(axis=1), den[j][idx].sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                rate[j] = np.where(rden > 0, rnum / np.where(rden > 0, rden, 1), np.nan)
        out[name] = (est, rate.mean(axis=0))
    if out["b"] is not None:
        out["diff"] = out["b"][1] - out["a"][1]
    return out


def _snap(v):
    """Float rounding noise around 0 (|v| < TOL) as exactly 0.0, so its sign never reaches an output."""
    return 0.0 if isinstance(v, float) and abs(v) < TOL else v


def ci_vs_0(low, high):
    """"<0", ">0", "includes_0" or "" (undefined CI); a bound within TOL of 0 counts as 0."""
    if low is None or high is None:
        return ""
    return "<0" if high < -TOL else ">0" if low > TOL else "includes_0"


def bootstrap_rows(recs, cruns, P, val_items, val_nk, merged_sets=None):
    """Bootstrap rows: primary and secondary condition pairs (b - a) on every target, then the ROB-04 FPR
    level of each condition. merged_sets (a set) collects the nk-merged group sets of every comparison."""
    rids = {c: [r["run_id"] for r in cruns if r["model"] == c] for c in CONDITIONS}
    case_nk = {("case", r["case_id"]): r["nk"] for r in recs}
    labels = {}
    for r in recs:
        labels.setdefault(r["group_id"], set()).add(r["stratum"])
    case_stratum = {g: next(iter(s)) if len(s) == 1 else "mixed" for g, s in labels.items()}
    val_stratum = {g: str(y) for g, y, _ in val_items}
    targets = [(VAL_SET, "all", "original", val_items, VAL_METRICS, val_stratum, val_nk)]
    for cond, sub, sl, rs in E.cell_defs(recs):
        if sl == "all":
            targets.append((cond, sub, "variant", E.side_items(rs, "variant"), E.BOOT_METRICS, case_stratum,
                            case_nk))
    pooled = pooled_main(recs)
    if pooled:
        targets.append((POOLED_SET, "main", "variant", E.side_items(pooled, "variant"), E.BOOT_METRICS,
                        case_stratum, case_nk))
    merged_sets = set() if merged_sets is None else merged_sets
    rows = []

    def add(kind, a, b, target, metric):
        es, sub, side, items, _, stratum, nk_of = target
        if not any(y == (0 if metric == "fpr" else 1) for _, y, _ in items):
            return
        ra, rb = rids[a], rids[b] if b else None
        if metric == "f1":
            res = f1_dist(items, ra, rb, stratum, P, nk_of)
        else:
            res = E.bootstrap_dist(items, ra, items if b else None, rb, metric, stratum, P, nk_of=nk_of)
        merged_sets.update(tuple(g) for g in res["merged"])
        r = E.bootstrap_row(kind, es, sub, sub, side, side, a, b, metric, res)
        rows.append({"kind": kind, "condition_a": a, "condition_b": b, "eval_set": es, "subtype": sub, "side": side,
                     **{k: _snap(r[k]) for k in BOOT_COLUMNS[6:]},
                     "ci_vs_0": ci_vs_0(r["diff_low"], r["diff_high"])})

    for kind, pairs in (("primary", PRIMARY), ("secondary", SECONDARY)):
        for a, b in pairs:
            for target in targets:
                for metric in target[4]:
                    add(kind, a, b, target, metric)
    for c in CONDITIONS:
        for target in targets:
            if target[0] == "ROB-04":
                add("level", c, "", target, "fpr")
    return rows


def rq04_verdict(recall, fpr, f1):
    """Development application of README 1.1 RQ-04 to one comparison a -> b. Each argument is the bootstrap
    row (diff = b - a with diff_low / diff_high) of: pooled ROB-01~03 main variant attack recall,
    validation_clean FPR, validation_clean F1; None when that comparison does not exist."""
    def get(row, key):
        return None if row is None else row[key]

    def neg(v):
        return None if v is None else 0.0 - v

    rd, rl, rh = get(recall, "diff"), get(recall, "diff_low"), get(recall, "diff_high")
    fd, fl, fh = get(fpr, "diff"), get(fpr, "diff_low"), get(fpr, "diff_high")
    od, ol, oh = neg(get(f1, "diff")), neg(get(f1, "diff_high")), neg(get(f1, "diff_low"))  # decrease = a - b
    c1 = rd is not None and rl is not None and rd >= DELTA_R - TOL and rl > TOL
    c2 = fh is not None and fh <= DELTA_F + TOL
    c3 = oh is not None and oh <= DELTA_O + TOL
    code = "confirmed" if c1 and c2 and c3 else "tradeoff" if c1 else "not_confirmed"
    return {"recall_diff": rd, "recall_diff_low": rl, "recall_diff_high": rh, "c1_recall": c1,
            "fpr_increase": fd, "fpr_increase_low": fl, "fpr_increase_high": fh, "c2_fpr": c2,
            "f1_decrease": od, "f1_decrease_low": ol, "f1_decrease_high": oh, "c3_f1": c3,
            "code": code, "text": RQ04_TEXT[code]}


def fpr_change_verdict(row):
    """Secondary pre-registered result for one FPR difference row (b - a)."""
    code = {"<0": "fpr_decrease", ">0": "fpr_increase"}.get(ci_vs_0(row["diff_low"], row["diff_high"]), "no_change")
    return {"code": code, "text": FPR_TEXT[code]}


def verdicts(boot):
    by = {(r["condition_a"], r["condition_b"], r["eval_set"], r["subtype"], r["metric"]): r
          for r in boot if r["kind"] != "level"}
    rq04 = [{"comparison": f"{a}->{b}", "condition_a": a, "condition_b": b,
             **rq04_verdict(by.get((a, b, POOLED_SET, "main", "recall")), by.get((a, b, VAL_SET, "all", "fpr")),
                            by.get((a, b, VAL_SET, "all", "f1")))} for a, b in PRIMARY]
    fpr = []
    for kind, pairs in (("primary", PRIMARY), ("secondary", SECONDARY)):
        for a, b in pairs:
            for es, sub in FPR_TARGETS:
                row = by.get((a, b, es, sub, "fpr"))
                if row is not None:
                    fpr.append({"kind": kind, "comparison": f"{a}->{b}", "condition_a": a, "condition_b": b,
                                "eval_set": es, "subtype": sub, "estimate_a": row["estimate_a"],
                                "estimate_b": row["estimate_b"], "diff": row["diff"], "diff_low": row["diff_low"],
                                "diff_high": row["diff_high"], **fpr_change_verdict(row)})
    return {"rq04_development": rq04, "benign_fpr_change": fpr}


def evaluate(recs, cruns, P, val_items, val_nk):
    """Everything computed from the predictions: metric rows, change rows, seed statistics, bootstrap rows,
    grouping and the development verdicts. val_items: [(sentence_id, label, ("val", sentence_id))] of
    validation_clean; val_nk: {("val", sentence_id): normalised key}."""
    seeds = list(dict.fromkeys(r["seed"] for r in cruns))
    cells = eval_cells(recs, val_items)
    mrows, crows = metric_rows(cells, cruns, P), E.change_rows(recs, cruns, P)
    merged = set()
    boot = bootstrap_rows(recs, cruns, P, val_items, val_nk, merged)
    return {"conditions": list(CONDITIONS), "seeds": seeds, "cells": [c[:3] for c in cells], "metrics": mrows,
            "changes": crows, "model_rows": E.model_rows(mrows, crows, list(CONDITIONS), seeds), "bootstrap": boot,
            "grouping": {"nk_merged_group_sets": [list(g) for g in sorted(merged)],
                         "comparisons_with_nk_merges": sum(r["nk_merges"] > 0 for r in boot)},
            "verdicts": verdicts(boot)}


# ---------------------------------------------------------------- output

def _round(obj, key=None):
    if isinstance(obj, float):
        return obj if key in FULL_PRECISION else round(obj, summarize_s2.DP)
    if isinstance(obj, dict):
        return {k: _round(v, k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, key) for v in obj]
    return obj


def _text_threshold(rows):
    """CSV rows: the threshold as repr (full precision), the logit with 2 decimals."""
    return [{**r, "threshold": repr(r["threshold"]),
             **({"logit": None if r["logit"] is None else f"{r['logit'] + 0.0:.2f}"} if "logit" in r else {})}
            for r in rows]


def run_rows(res):
    """One row per condition x seed: the source run and the validation_clean metrics at its threshold.
    runs.csv holds RUN_COLUMNS; attempt and cap_met go to summary.json and summary.md only."""
    src = {r["run_id"]: r for r in res["sources"]}
    val = {(r["model"], r["seed"]): r for r in res["eval"]["metrics"]
           if r["condition"] == VAL_SET and r["label"] == "all"}
    rows = []
    for cr in res["cruns"]:
        s, m = src[cr["source"]], val[(cr["model"], cr["seed"])]
        rows.append({"condition": cr["model"], "seed": cr["seed"], "run_id": cr["source"],
                     "trained_condition": s["trained"], "attempt": s["attempt"], "selected_step": s["step"],
                     "checkpoint_sha256": s["sha256"], "threshold": cr["threshold"], "logit": cr["logit"],
                     "cap_met": cr["cap_met"], **{f"{VAL_SET}_{k}": m[k] for k in ("n",) + summarize_s2.METRICS}})
    return rows


def _by_key(res):
    return {cr["run_id"]: cr for cr in res["cruns"]}


def out_metric_rows(res):
    cr = _by_key(res)
    return [{"condition": r["model"], "seed": r["seed"], "run_id": cr[r["run_id"]]["source"],
             "threshold": cr[r["run_id"]]["threshold"], "eval_set": r["condition"],
             **{k: r[k] for k in METRIC_COLUMNS[5:]}} for r in res["eval"]["metrics"]]


def out_change_rows(res):
    cr = _by_key(res)
    return [{"condition": r["model"], "seed": r["seed"], "run_id": cr[r["run_id"]]["source"],
             "threshold": cr[r["run_id"]]["threshold"], "eval_set": r["condition"],
             **{k: r[k] for k in CHANGE_COLUMNS[5:]}} for r in res["eval"]["changes"]]


def out_model_rows(res):
    return [{"condition": r["model"], "eval_set": r["condition"], **{k: v for k, v in r.items()
                                                                      if k not in ("model", "condition")}}
            for r in res["eval"]["model_rows"]]


def prediction_rows(res):
    P = res["P"]
    return [{"case_id": r["case_id"], "run_id": cr["source"], "condition": cr["model"],
             "margin": eval_lib.fmt9(P[cr["run_id"]]["case"][r["case_id"]][0]),
             "score": eval_lib.fmt9(P[cr["run_id"]]["case"][r["case_id"]][1]),
             "prediction": P[cr["run_id"]]["case"][r["case_id"]][2], "threshold": cr["threshold"]}
            for cr in res["cruns"] for r in res["recs"]]


def summary_md(res):
    ev, seeds = res["eval"], res["eval"]["seeds"]
    st = {(r["model"], r["condition"], r["subtype"], r["slice"], r["side"], r["label"], r["quantity"]): r
          for r in ev["model_rows"]}
    bs = {(r["condition_a"], r["condition_b"], r["eval_set"], r["subtype"], r["metric"]): r
          for r in ev["bootstrap"]}

    def pm(c, es, sub, side, lab, q, dp=4, sl="all"):
        r = st.get((c, es, sub, sl, side, lab, q))
        if r is None:
            return "-"
        return summarize_s2._pm({"mean": r["mean"], "std": r["std"], "values": [r[f"s{s}"] for s in seeds]}, dp)

    def n(es, sub, side, lab):
        r = st.get(("A", es, sub, "all", side, lab, "recall" if lab == "1" else "fpr"))
        return r["n"] if r else 0

    def ci(r):
        return "-" if r is None else E._ci_text(r["diff"], r["diff_low"], r["diff_high"])

    def ok(flag):
        return "충족" if flag else "미충족"

    name = summarize_s2.NAMES.get(REPRESENTATIVE, REPRESENTATIVE)
    rec, runs = res["records"], run_rows(res)
    n_val = runs[0][f"{VAL_SET}_n"]
    cells = [c for c in ev["cells"] if c[0] != VAL_SET and c[2] == "all"]
    out = ["# 4단계 A~D 비교 결과 (개발용)", "",
           f"`{SCRIPT_REL}`가 대표 모델 {name}의 A~D 조건(README 5.3)을 `{VAL_SET}`({n_val:,}행)과 검수한 개발용 추가 평가 "
           f"사례(keep·relabel {len(res['recs'])}개)에서 비교한 결과다. A·B는 2단계 실행 "
           f"`{STAGE_A}-{REPRESENTATIVE}-A-s<seed>`, C·D는 증강 학습 실행 `{STAGE_C}-{REPRESENTATIVE}-C-s<seed>`의 선택 "
           f"체크포인트를 쓴다(seed {'·'.join(str(s) for s in seeds)}). A·C는 임계값 {THRESHOLD}, B·D는 `{VAL_SET}`에서 고른 "
           "seed별 임계값(`thresholds.json`)으로 판정하며, 판정 규칙은 모두 float32 인젝션 점수 ≥ 임계값이다. "
           f"`{VAL_SET}` 값은 학습 때 저장한 선택 step의 점수로, 사례 값은 체크포인트로 다시 예측한 점수로 계산했다. "
           "값은 seed 평균 ± 표본 표준편차(n−1)이고, 하나라도 N/A인 seed가 있으면 seed별 값을 적는다. 차이는 뒤 조건 − 앞 "
           "조건이며 괄호는 원문 그룹 단위 짝지은 bootstrap 95% CI(2,000회, seed 314159, 원문 레이블 층화, 정규화 키가 같은 "
           "그룹은 합침)다.", "",
           "**개발 결과다.** validation과 개발용 사례는 임계값 선택과 증강 설계에 쓴 개발 자료이므로, 아래 판정은 README 1.1 "
           "규칙을 개발 단계에 적용해 본 것이며 최종 판정이 아니다. 최종 판정은 `test_clean`과 최종 추가 평가 사례로 한다. "
           "전체 값은 `runs.csv`·`metrics.csv`·`models.csv`·`bootstrap.csv`·`predictions.csv`·`summary.json`에 있다.", "",
           "## 조건별 실행과 임계값", "",
           "| 조건 | seed | 실행 | 학습 시도 | 선택 step | 임계값 | logit | FPR 상한 |",
           "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    for r in runs:
        logit = "-" if r["logit"] is None else f"{r['logit'] + 0.0:+.2f}"
        cap = "-" if r["cap_met"] is None else "충족" if r["cap_met"] else "미충족"
        out.append(f"| {r['condition']} | {r['seed']} | `{r['run_id']}` | "
                   f"{'-' if r['attempt'] is None else r['attempt']} | {r['selected_step']} | "
                   f"{r['threshold']:.6f} | {logit} | {cap} |")
    b, aug, code = rec["budget"], rec["augmentation"], rec["code"]
    rerun = sorted({r["run_id"] for r in runs if type(r["attempt"]) is int and r["attempt"] > 1})
    not_met = [f"{r['condition']} seed {r['seed']}" for r in runs if r["cap_met"] is False]
    differing = rec["differing_between_A_and_C"]
    out += ["", "- `학습 시도`는 `run.json`의 `attempt`이며, 2 이상이면 다시 학습한 실행이다(재실행 기록, README 4.2·5.3). "
            "2 이상인 실행: " + (", ".join(f"`{x}`" for x in rerun) if rerun else "없음") + ".",
            f"- `FPR 상한`은 B·D의 임계값을 고를 때 FPR 상한(같은 seed A 조건의 `{VAL_SET}` FPR + 1%p, README 5.3)을 "
            "만족하는 후보가 있었는지다(`thresholds.json`의 `cap_met`). 미충족이면 FPR이 가장 낮은 후보를 고른 것이다. "
            "미충족인 조건: " + (", ".join(not_met) if not_met else "없음") + ".",
            f"- A와 C 실행은 seed마다 학습 예산(N = {b['N']:,} update, 평가 시점 "
            f"{' / '.join(f'{s:,}' for s in b['eval_steps'])})·설정·공통 목록 파일·데이터 revision·사전학습 가중치·정밀도·"
            "결정론 설정이 같고, 환경 기록은 `run.json`의 `env` 전체(Python·OS·torch·CUDA·cuDNN·라이브러리 버전·GPU 이름과 "
            "드라이버·`requirements.txt` SHA-256 등 모든 항목)가 같다. 하나라도 다르면 이 스크립트가 거부한다. 선택 step은 "
            "달라도 된다(IMP-03).",
            "- A와 C 사이에 달라도 되는 기록은 " + ", ".join(MAY_DIFFER_TEXT[n] for n, _ in MAY_DIFFER) + "이다"
            "(C 학습 전에 `build_lists.py`와 `train.py`가 바뀔 수 있어 허용하며, 두 조건의 값을 모두 적는다). 이 가운데 "
            "실제로 다른 것: " + (", ".join(MAY_DIFFER_TEXT[n] for n in differing) if differing else "없음")
            + f". 학습 코드 SHA-256이 A와 다른 파일: {len(code['files_differing'])}개"
            + (f"(`{'`, `'.join(code['files_differing'])}`)" if code["files_differing"] else "")
            + ". A와 C의 값은 `summary.json`의 `records`에 있다.",
            f"- C의 학습 데이터는 원본 {aug['train_clean_rows']:,}행 + 증강 {aug['rows']:,}행 = {aug['train_rows']:,}행이다.",
            f"- `thresholds.json`은 여기서 쓴 실행과 대조했다: 기록된 실행 파일 SHA-256 "
            f"{res['thresholds_info']['run_file_sha256_checked']}개가 같고, 선택한 임계값에서의 `{VAL_SET}` "
            f"TP·FP·FN·TN 기록 {res['thresholds_info']['selection_counts_checked']}건이 여기서 계산한 값과 같다"
            f"(여섯 실행의 `{'`·`'.join(PINNED_RUN_FILES)}` SHA-256과 임계값마다의 TP·FP·FN·TN 기록이 없거나 다르면 이 "
            "스크립트가 거부한다).",
            "", f"## `{VAL_SET}` (개발 자료)", "",
            "| 조건 | F1 | Precision | Recall | FPR | ROC-AUC | FP | FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in CONDITIONS:
        out.append(f"| {c} | " + " | ".join(pm(c, VAL_SET, "all", "original", "all", q)
                                             for q in ("f1", "precision", "recall", "fpr", "roc_auc"))
                   + f" | {pm(c, VAL_SET, 'all', 'original', 'all', 'fp', 1)} | "
                   f"{pm(c, VAL_SET, 'all', 'original', 'all', 'fn', 1)} |")
    out += ["", f"- B·D의 임계값은 이 `{VAL_SET}`에서 골랐으므로 B·D의 값은 선택에 쓴 자료에서 잰 값이다.",
            "", "## 개발용 사례: 조건별 변형 탐지율과 FPR", "",
            "| 평가 자료 | 하위 유형 | 사례 수 (공격/정상) | 탐지율 A | 탐지율 B | 탐지율 C | 탐지율 D | FPR A | FPR B | FPR C | "
            "FPR D |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for es, sub, _ in cells:
        out.append(f"| {es} | {sub} | {n(es, sub, 'variant', '1')}/{n(es, sub, 'variant', '0')} | "
                   + " | ".join(pm(c, es, sub, "variant", "1", "recall") for c in CONDITIONS) + " | "
                   + " | ".join(pm(c, es, sub, "variant", "0", "fpr") for c in CONDITIONS) + " |")
    out += ["", f"- `{POOLED_SET}` main은 ROB-01(case·space·char)·ROB-02·ROB-03의 평가군 사례를 합친 것이다. ROB-01 "
            "homoglyph는 보류한 생성 방식이라 따로 둔다.",
            "", "## 개발용 사례: 원문과 변형, 판정 변화율", "",
            "| 조건 | 평가 자료 | 하위 유형 | 탐지율 원문 | 탐지율 변형 | FPR 원문 | FPR 변형 | 판정 변화율 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for c in CONDITIONS:
        for es, sub, _ in cells:
            if es in ("ROB-04", POOLED_SET):
                continue
            out.append(f"| {c} | {es} | {sub} | {pm(c, es, sub, 'original', '1', 'recall')} | "
                       f"{pm(c, es, sub, 'variant', '1', 'recall')} | {pm(c, es, sub, 'original', '0', 'fpr')} | "
                       f"{pm(c, es, sub, 'variant', '0', 'fpr')} | {pm(c, es, sub, 'change', 'all', 'change_rate')} |")
    targets = [(VAL_SET, "all", m) for m in VAL_METRICS] + [(es, sub, m) for es, sub, _ in cells
                                                            for m in E.BOOT_METRICS]
    label = {"recall": "탐지율", "fpr": "FPR", "f1": "F1"}
    for title, pairs in (("## 주요 비교: A→B, A→C, A→D", PRIMARY), ("## 보조 비교: B→D, C→D, B→C", SECONDARY)):
        out += ["", title, "", "| 비교 | 평가 자료 | 하위 유형 | 지표 | 앞 조건 | 뒤 조건 | 차이 [CI] |",
                "| --- | --- | --- | --- | ---: | ---: | ---: |"]
        for a, b_ in pairs:
            for es, sub, metric in targets:
                r = bs.get((a, b_, es, sub, metric))
                if r is not None:
                    out.append(f"| {a}→{b_} | {es} | {sub} | {label[metric]} | {E._f4(r['estimate_a'])} | "
                               f"{E._f4(r['estimate_b'])} | {ci(r)} |")
    out += ["", "- 사례의 탐지율은 공격 변형, FPR은 정상 변형에서 잰 값이다. 값은 3-seed 평균이다.",
            "", "## RQ-04 규칙의 개발 단계 적용 (최종 판정 아님)", "",
            f"① `{POOLED_SET}` main 공격 변형의 탐지율 차이 ≥ {DELTA_R:.2f}이고 CI 하한 > 0, ② `{VAL_SET}` FPR 증가의 CI "
            f"상한 ≤ {DELTA_F:.2f}, ③ `{VAL_SET}` F1 감소의 CI 상한 ≤ {DELTA_O:.2f}. 셋 다 충족하면 \"개선 확인\", ①은 "
            "충족하나 ② 또는 ③을 충족하지 못하면 \"상충 있는 개선\", ①을 충족하지 못하면 \"개선 확인 못함\"이다(README 1.1의 "
            f"규칙을 `test_clean`과 최종 사례 대신 `{VAL_SET}`과 개발용 사례에 적용). CI가 정의되지 않으면 그 조건은 "
            "미충족으로 둔다.", "",
            "| 비교 | ① 탐지율 차이 [CI] | ① | ② FPR 증가 [CI] | ② | ③ F1 감소 [CI] | ③ | 개발 단계 판정 |",
            "| --- | ---: | --- | ---: | --- | ---: | --- | --- |"]
    for v in ev["verdicts"]["rq04_development"]:
        out.append(f"| {v['condition_a']}→{v['condition_b']} | "
                   f"{E._ci_text(v['recall_diff'], v['recall_diff_low'], v['recall_diff_high'])} | "
                   f"{ok(v['c1_recall'])} | "
                   f"{E._ci_text(v['fpr_increase'], v['fpr_increase_low'], v['fpr_increase_high'])} | "
                   f"{ok(v['c2_fpr'])} | "
                   f"{E._ci_text(v['f1_decrease'], v['f1_decrease_low'], v['f1_decrease_high'])} | {ok(v['c3_f1'])} | "
                   f"{v['text']} |")
    out += ["", "## 보조 결과: 정상 변형과 ROB-04의 FPR 변화 (사전 지정)", "",
            "FPR 차이(뒤 − 앞)의 CI 상한 < 0이면 \"오탐 감소 확인\", CI 하한 > 0이면 \"오탐 증가 확인\", 그 외는 \"변화 확인 "
            "못함\"이다.", "",
            "| 구분 | 비교 | 평가 자료 | 하위 유형 | FPR 앞 조건 | FPR 뒤 조건 | 차이 [CI] | 판정 |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- |"]
    for v in ev["verdicts"]["benign_fpr_change"]:
        out.append(f"| {'주요' if v['kind'] == 'primary' else '보조'} | {v['condition_a']}→{v['condition_b']} | "
                   f"{v['eval_set']} | {v['subtype']} | {E._f4(v['estimate_a'])} | {E._f4(v['estimate_b'])} | "
                   f"{E._ci_text(v['diff'], v['diff_low'], v['diff_high'])} | {v['text']} |")
    out += ["", "## ROB-04 FPR 수준", "", "| 조건 | 사례 수 | FPR | FPR 95% CI |", "| --- | ---: | ---: | ---: |"]
    for c in CONDITIONS:
        r = bs.get((c, "", "ROB-04", "benign_security", "fpr"))
        lv = "-" if r is None or r["a_low"] is None else f"[{r['a_low']:.4f}, {r['a_high']:.4f}]"
        out.append(f"| {c} | {n('ROB-04', 'benign_security', 'variant', '0')} | "
                   f"{pm(c, 'ROB-04', 'benign_security', 'variant', '0', 'fpr')} | {lv} |")
    out += ["", "- 다중 비교 보정은 하지 않았으므로 신뢰구간은 기술적 근거로 해석한다(README 6.1). 이 구간은 학습된 모델을 "
            "고정했을 때 평가 원문 표집의 불확실성만 나타내며, seed 간 변동은 표준편차로 따로 적었다.",
            "- 사례의 원문 예측은 validation 전체를 학습 때와 같은 순서·배치로 다시 계산했고, 여섯 실행 모두 "
            "`predictions_validation.csv`의 선택 step 예측과 같았다. B·D 임계값에서도 사례 원문의 판정은 저장된 점수와 다시 "
            "계산한 점수에서 같았다(다르면 이 스크립트가 거부한다).",
            f"- 정규화 키가 같아 합친 그룹이 있는 비교: {ev['grouping']['comparisons_with_nk_merges']}개"
            "(`bootstrap.csv`의 `nk_merges`).", ""]
    return "\n".join(out)


def summary_json(res, script_sha):
    ev = res["eval"]
    runs = []
    for row, cr in zip(run_rows(res), res["cruns"]):
        vals = [res["P"][cr["run_id"]]["val"][sid] for sid, _, _ in res["val_items"]]
        m = eval_lib.metrics([y for _, y, _ in res["val_items"]], [v[2] for v in vals])
        runs.append({**row, "exact": {k: m[f"{k}_exact"] for k in ("f1", "precision", "recall", "fpr")},
                     "thresholded_original_check": res["threshold_checks"][cr["run_id"]]})
    doc = {
        "schema": 1,
        "readme": "README 5.3 (conditions A-D, IMP-01~04), 6.1 (metrics, paired bootstrap), 1.1 (RQ-04 rules, "
                  "applied at the development stage), 5.2 (ROB-01~04); stage-4 design D6; written by " + SCRIPT_REL,
        "development": True,
        "rules": {
            "stage": "development result: validation_clean and the development cases were used for the threshold "
                     "selection and the augmentation design; the verdicts are not the final verdict",
            "conditions": f"A = {source_run_id('A', '<seed>')} at {THRESHOLD}; B = the same checkpoints at the B "
                          f"thresholds; C = {source_run_id('C', '<seed>')} at {THRESHOLD}; D = the same C "
                          "checkpoints at the D thresholds",
            "decision": "float32(score) >= threshold (eval_lib.predictions on a float32 tensor) everywhere",
            "source_runs": "A pinned by results/s2 (eval_cases.load_stage2); C verified like summarize_s2 "
                           "(completed, official lists, clean code state, budget, model pins, augmentation "
                           "recorded, reload check, checkpoint size and SHA-256, selection rule, predictions "
                           "reproducing metrics.json at every eval step, condition C, threshold 0.5)",
            "records": "equal in the A and C run of a seed: " + ", ".join(n for n, _ in SAME_AC)
                       + " (environment = the whole env record of run.json, no entry exempt); may differ between "
                       "A and C, both values reported, one value per trained condition: "
                       + ", ".join(n for n, _ in MAY_DIFFER) + "; equal in the three C runs: "
                       + ", ".join(n for n, _ in SAME_C) + "; attempt = the attempt number of run.json",
            "thresholds": "entries of thresholds.json = JSON objects with the keys "
                          + ", ".join(THRESHOLD_KEYS) + f"; one per seed and condition B|D; a float32 value in "
                          f"(0, 1) equal to sigmoid(logit) within {SIGMOID_TOL:g}, logit on the grid k/"
                          f"{LOGIT_STEPS} within -{LOGIT_MAX:g}..{LOGIT_MAX:g}; required: cap_met (boolean), "
                          "chosen tp, fp, fn, tn (integers, equal to the validation_clean counts used here) and "
                          f"the SHA-256 of {' and '.join(PINNED_RUN_FILES)} of every source run (equal to the "
                          "files used here); when present, seed, condition, selected_step, checkpoint_sha256 "
                          "and the SHA-256 of other run files must agree with the runs used here",
            "validation_clean": "in_clean rows of predictions_validation.csv at the selected step, stored scores "
                                "and margins",
            "cases": "status assigned with review decision keep or relabel (label = new_label); pending or "
                     "unapplied exclude refused; predicted with eval_cases.predict_runs for the A and C "
                     "checkpoints, B and D by re-thresholding the float32 scores",
            "pooled": f"{POOLED_SET} main = main-group cases of {', '.join(POOLED_CONDITIONS)} (ROB-01 case, "
                      "space, char; homoglyph is held out)",
            "metrics": "eval_lib.metrics; label all only with both labels, 1: recall, 0: FPR; N/A is null; ROC-AUC "
                       "on the margin (the same for A and B, and for C and D)",
            "seeds": "mean and sample std (n - 1) over the seeds; null when any seed is N/A",
            "bootstrap": f"groups = group_id (a validation row is its own group), merged when texts share "
                         f"build_lists.nk, strata = original label (mixed if several), numpy default_rng("
                         f"{E.BOOT_SEED}) restarted per comparison, {E.N_RESAMPLES} resamples, the same indices "
                         "for both conditions; per resample the metric per seed (recall, FPR: ratio of summed "
                         "counts; F1: sum of 2TP over sum of 2TP + FP + FN), 3-seed mean, then b - a; "
                         "numpy.percentile 2.5 / 97.5 (linear); null CI when any resample is undefined",
            "comparisons": "primary " + ", ".join(f"{a}->{b}" for a, b in PRIMARY) + "; secondary "
                           + ", ".join(f"{a}->{b}" for a, b in SECONDARY),
            "rq04_development": f"(1) {POOLED_SET} main variant attack recall difference >= {DELTA_R} and CI "
                                f"lower > 0; (2) validation_clean FPR increase CI upper <= {DELTA_F}; (3) "
                                f"validation_clean F1 decrease CI upper <= {DELTA_O}; all: confirmed, (1) only: "
                                "tradeoff, not (1): not_confirmed; an undefined CI satisfies nothing",
            "benign_fpr_change": "FPR difference on the benign variants of "
                                 + ", ".join(f"{es} {sub}" for es, sub in FPR_TARGETS)
                                 + ": CI upper < 0 fpr_decrease, CI lower > 0 fpr_increase, else no_change",
            "tolerance": "comparisons with 0 and with the limits allow 1e-9 for float rounding",
            "numbers": f"non-integers rounded to {summarize_s2.DP} decimals, except "
                       f"{', '.join(FULL_PRECISION)}",
        },
        "settings": res["settings"], "device": res["device"], "script_sha256_lf": script_sha,
        "inputs_sha256": res["inputs"], "representative": REPRESENTATIVE, "conditions": ev["conditions"],
        "seeds": ev["seeds"],
        "thresholds": {"file": res["thresholds_file"], **res["thresholds_info"],
                       "values": [{"condition": c, "seed": s, **res["thresholds"][(s, c)]}
                                  for c in TUNED for s in ev["seeds"]]},
        "source_runs": [{"run_id": r["run_id"], "trained_condition": r["trained"], "seed": r["seed"],
                         "attempt": r["attempt"], "selected_step": r["step"], "checkpoint_sha256": r["sha256"],
                         "checkpoint_size": r["size"], "original_check": res["checks"][r["run_id"]]}
                        for r in res["sources"]],
        "records": res["records"], "runs": runs, "cases": dict(res["cases"]),
        "metrics": out_metric_rows(res), "models_summary": out_model_rows(res), "changes": out_change_rows(res),
        "bootstrap": ev["bootstrap"], "grouping": ev["grouping"],
        "verdicts": {"stage": "development", **ev["verdicts"]},
    }
    return train._dump(_round(doc))


def write_outputs(out_dir, res, script_sha):
    seeds = res["eval"]["seeds"]
    model_cols = (("condition",) + CELL_COLUMNS + ("side", "label", "n", "quantity", "mean", "std")
                  + tuple(f"s{s}" for s in seeds))
    csv_text = summarize_s2.csv_text
    files = {
        "runs.csv": csv_text(RUN_COLUMNS, _text_threshold(run_rows(res))),
        "metrics.csv": csv_text(METRIC_COLUMNS, _text_threshold(out_metric_rows(res))),
        "models.csv": csv_text(model_cols, out_model_rows(res)),
        "bootstrap.csv": csv_text(BOOT_COLUMNS, res["eval"]["bootstrap"]),
        "predictions.csv": csv_text(PRED_COLUMNS, _text_threshold(prediction_rows(res))),
        "summary.json": summary_json(res, script_sha),
        "summary.md": summary_md(res),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = {n: out_dir / (n + ".tmp") for n in files}
    try:
        for n, text in files.items():  # every file is written before the first one is moved into place
            tmp[n].write_bytes(text.encode("utf-8"))
        for n in files:
            os.replace(tmp[n], out_dir / n)
    finally:
        for p in tmp.values():
            p.unlink(missing_ok=True)
    return list(files)


def output_path_error(out, root):
    """An O01 message when the output path is an existing file or lies below one, else None."""
    p = Path(out).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    if p.is_dir():
        return None
    return (f"O01 the output folder {E._rel(out, root)} is an existing file or lies below one (the outputs "
            "are written into a folder)")


# ---------------------------------------------------------------- main

def load_cases(root, cdir, val):
    """(records, info, {input name: SHA-256}, errors): the reviewed cases read and checked as eval_cases
    does, without --allow-incomplete (pending and unapplied exclude are refused)."""
    crel, rrel = f"{E._rel(cdir, root)}/cases.csv", f"{E._rel(cdir, root)}/review.csv"
    craw, crows, errs = E.read_table(cdir / "cases.csv", E.CASES_COLUMNS, crel, "C01")
    try:
        rraw = (cdir / "review.csv").read_bytes()
    except OSError as exc:
        return None, None, None, errs + [f"R01 {rrel}: cannot read ({type(exc).__name__})"]
    if crows is None:
        return None, None, None, errs
    cases, e = E.check_cases(crows, val)
    errs += e
    review, e = E.check_review(rraw, cases)
    errs += e
    if errs:
        return None, None, None, errs
    recs, info, errs = E.select_cases(cases, review, val)
    if info["pending"]:
        errs.append(f"P01 review.csv: {len(info['pending'])} assigned case(s) are still pending (first "
                    f"{info['pending'][0]}); finish the review")
    if info["unapplied_excludes"]:
        errs.append(f"P02 review.csv: {len(info['unapplied_excludes'])} assigned case(s) have decision exclude "
                    f"(first {info['unapplied_excludes'][0]}); run scripts/build_dev_cases.py --apply-review so "
                    "the replacements are made and reviewed")
    if not recs and not errs:
        errs.append("N01 no case to evaluate (no keep or relabel decision)")
    return recs, info, {crel: data_lists.sha256_bytes(craw), rrel: data_lists.sha256_bytes(rraw)}, errs


def run_eval(root, cdir, thr_path):
    """(errors, result). Nothing is written here."""
    s2raw, s2doc, errs = E.read_s2_summary(root)
    errs = errs or E.check_config_pin(root, s2doc)
    if errs:
        return errs, None
    try:
        cfg_bytes = (root / CONFIG_REL).read_bytes()
        cfg = json.loads(cfg_bytes)
        settings = {"threshold": cfg["threshold"], "max_length": cfg["max_length"], "eval_batch": cfg["eval"]["batch"],
                    "eval_precision": cfg["eval"]["precision"], "seeds": cfg["seeds"],
                    "attn_implementation": cfg["attn_implementation"], "determinism": cfg["determinism"],
                    "expected_updates": cfg["expected_updates"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"E00 {CONFIG_REL}: unreadable or missing keys ({type(exc).__name__})"], None
    seeds = settings["seeds"]
    if settings["eval_precision"] != "fp32" or settings["threshold"] != THRESHOLD:
        return [f"E00 {CONFIG_REL}: eval precision must be fp32 and threshold {THRESHOLD} (README 4.2)"], None
    if not (isinstance(seeds, list) and len(seeds) == summarize_s2.N_SEEDS and len(set(seeds)) == len(seeds)
            and all(type(s) is int for s in seeds) and type(settings["expected_updates"]) is int):
        return [f"E00 {CONFIG_REL}: seeds must be {summarize_s2.N_SEEDS} distinct integers and expected_updates "
                "an integer"], None
    data_lists.install_test_guard()
    try:
        val = data_lists.load_rows()["validation"]
    except data_lists.ListError as exc:
        return [f"E01 {e}" for e in exc.errors], None
    recs, info, case_shas, errs = load_cases(root, cdir, val)
    a_all, shas, e = E.load_stage2(root, list(model_specs.MODELS), seeds, s2raw, s2doc)
    errs += e
    sources = []
    for run in ([] if e else a_all):  # the nine stage-2 runs are pinned together; the representative's are used
        if run["model"] == REPRESENTATIVE:
            src, e2 = a_source(root, run, shas)
            errs += e2
            sources += [src] if src else []
    for s in seeds:
        src, e2 = load_c_run(root, s, cfg)
        errs += e2
        sources += [src] if src else []
    records = None
    if len(sources) == 2 * len(seeds):
        e, records = compare_records(sources)
        errs += e
    file_shas = {k: v for r in sources for k, v in r["files"].items()}
    trel = E._rel(thr_path, root)
    try:
        traw = Path(thr_path).read_bytes()
    except OSError as exc:
        traw, thresholds, tinfo = None, None, None
        errs.append(f"T01 {trel}: cannot read ({type(exc).__name__}); run scripts/select_thresholds.py first")
    else:
        used = {r["run_id"]: {"selected_step": r["step"], "checkpoint_sha256": r["sha256"]} for r in sources}
        thresholds, tinfo, e = read_thresholds(traw, seeds, used, file_shas)
        if not e:
            e, tinfo["selection_counts_checked"] = selection_count_errors(thresholds, sources)
        errs += e
    if errs:
        return errs, None
    device, dev_info = E.choose_device(cfg)
    try:
        P6, checks = E.predict_runs(root, sources, recs, val, cfg, device)
    except E.Refused as exc:
        return exc.errors, None
    cruns = condition_runs(seeds, thresholds)
    P, tchecks, errs = rethreshold(cruns, sources, P6)
    if errs:
        return errs, None
    first = sources[0]
    val_items = [(sid, y, ("val", sid)) for sid, y, c in zip(first["ids"], first["labels"], first["in_clean"]) if c]
    val_nk = {key: build_lists.nk(val[sid][1]) for sid, _, key in val_items}
    inputs = {CONFIG_REL: data_lists.sha256_bytes(cfg_bytes), **case_shas, **file_shas,
              trel: data_lists.sha256_bytes(traw),
              **{k: v for k, v in shas.items() if k.startswith(E.S2_REL + "/")}}
    return [], {"settings": settings, "device": dev_info, "inputs": dict(sorted(inputs.items())),
                "sources": sources, "records": records, "thresholds": thresholds, "thresholds_info": tinfo,
                "thresholds_file": trel, "cruns": cruns, "checks": checks, "threshold_checks": tchecks,
                "recs": recs, "cases": info, "val_items": val_items, "P": P,
                "eval": evaluate(recs, cruns, P, val_items, val_nk)}


def parse_args(argv):
    p = argparse.ArgumentParser(description="Compare conditions A-D of the representative model on "
                                            "validation_clean and the reviewed development cases (README 5.3, "
                                            "6.1) into results/s4/.")
    p.add_argument("--dir", help=f"cases folder with cases.csv and review.csv (default: <root>/{DIR_REL})")
    p.add_argument("--thresholds", help=f"thresholds file (default: <root>/{THRESHOLDS_REL})")
    p.add_argument("--out", help=f"output folder (default: <root>/{OUT_REL}; never {' or '.join(PROTECTED_OUT)})")
    p.add_argument("--root", help="repository root holding runs/, results/ and configs/; must be the root "
                                  "eval_lib.load_run_model uses (eval_lib.ROOT), so only tests change it")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root).resolve() if args.root else ROOT
    cdir = Path(args.dir) if args.dir else root / DIR_REL
    out = Path(args.out) if args.out else root / OUT_REL
    thr = Path(args.thresholds) if args.thresholds else root / THRESHOLDS_REL
    protected = [(root / p).resolve() for p in PROTECTED_OUT]
    if root != Path(eval_lib.ROOT).resolve():
        errors, res = ["E04 --root must be the repository root that eval_lib.load_run_model uses "
                       "(eval_lib.ROOT); other roots are not supported"], None
    elif any(out.resolve() == p or p in out.resolve().parents for p in protected):
        errors, res = [f"O01 the output folder must not be {' or '.join(PROTECTED_OUT)} or inside them (reviewed "
                       "records with the same file names)"], None
    elif output_path_error(out, root):
        errors, res = [output_path_error(out, root)], None
    else:
        errors, res = run_eval(root, cdir, thr)
    if errors:
        for e in errors:
            print(train._ascii("FAIL " + e))
        print(f"refused: {len(errors)} error(s); nothing written")
        return 1
    names = write_outputs(out, res, data_lists.sha256_lf(ROOT / SCRIPT_REL))
    print(train._ascii(f"OK conditions {', '.join(CONDITIONS)} x {len(res['eval']['seeds'])} seeds: "
                       f"{len(res['val_items'])} {VAL_SET} rows and {len(res['recs'])} cases on "
                       f"{res['device']['type']}; wrote {E._rel(out, root)}/: {', '.join(names)}"))
    for rid, c in res["checks"].items():
        print(f"{rid}: originals reproduced ({c['validation_rows']} rows, 0 flips; 9-digit margins differ in "
              f"{c['margin_rows_differing_9_digits']} rows, max {c['max_abs_margin_diff_9_digits']:.3g})")
    for v in res["eval"]["verdicts"]["rq04_development"]:
        print(f"development RQ-04 {v['comparison']}: {v['code']} (recall {v['c1_recall']}, FPR {v['c2_fpr']}, "
              f"F1 {v['c3_f1']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
