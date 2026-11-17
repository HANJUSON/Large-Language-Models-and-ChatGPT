"""Stage-3 evaluation of the reviewed development cases with the nine stage-2 models (README 5.2, 6.1).

Run from Git Bash at the repo root only after the review in data/cases/dev/review.csv is finished:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/eval_cases.py
Options: --dir DIR cases folder (default data/cases/dev); --out DIR output folder (default results/s3);
--allow-incomplete leaves pending cases out, writes to results/s3_exploratory/ (never inside results/s3/) and
marks every output exploratory; --root DIR must be the repository root that eval_lib uses (tests only).

Rules:
  - Cases (spec-stage3 "Case ids and records"; formats and constants of scripts/build_dev_cases.py):
    cases.csv (UTF-8, csv module, exact header) and review.csv parsed by build_dev_cases.parse_review (BOM,
    CRLF and fully blank rows accepted; R01-R09 as there: decision values, new_label only for relabel,
    no ROB-04 relabel, reason for relabel/exclude, reviewer one of build_dev_cases.REVIEWERS and empty
    when pending, text_sha256_12 on decided rows). On top: every assigned case has exactly one row, other
    rows only for review_excluded cases and decided (E10); a relabel's new_label differs from the label
    (R06); a decided row's text_sha256_12 equals build_lists.h12 of the case text in cases.csv (R09,
    stale decision). Refused (exit 1, nothing written) on any of these, an invalid cases.csv (C01-C03),
    an assigned case still pending (P01) or with decision exclude not yet replaced by build_dev_cases.py
    --apply-review (P02). --allow-incomplete leaves P01/P02 cases out instead (a ROB-01 exclude takes all
    ROB-01 cases of that original out). Used: assigned with keep or relabel (label = new_label). Each
    derived case's label must equal its validation original's label; ROB-03 texts must hold the original
    at their position.
  - Config (E03): configs/train.json SHA-256 (LF-normalised) must equal the stage-2 record (results/s2/
    summary.json inputs_sha256 and records.config_sha256); eval settings (threshold, max_length, eval batch
    and precision, attn_implementation, determinism) are read only after that check.
  - Models: runs/s2-<model>-A-s<seed> for model_specs.MODELS x configs/train.json seeds. Pins come from the
    reviewed record results/s2/: runs.csv (selected step, checkpoint SHA-256; must agree with
    summary.json) and summary.json inputs_sha256 (run.json, metrics.json, predictions_validation.csv must
    have these SHA-256s, so the run.json checkpoint size is pinned too). eval_lib.load_run_model with
    expected_sha256, expected_size and the config's attn_implementation. One model in memory at a time
    (freed before the next).
  - Prediction (README 4.2): CUDA if available (FP32, train.apply_determinism), else CPU; eval_lib.encode
    (max_length of the config) and eval_lib.predict (eval batch of the config, dynamic padding); score >=
    threshold (0.5). Originals: all validation rows (data_lists.load_rows, test guard installed) in
    sentence-ID order, as in training; every prediction must equal predictions_validation.csv at the
    selected step (M05; the margin difference is recorded). Cases in case_id order.
  - Cells: (condition, subtype) per README 5.2, plus ROB-01 "main" (case+space+char pooled; homoglyph is
    the held-out group) and ROB-03 slices position=prefix|suffix|both and length=short|long (long: a
    context side has more than LONG_WORDS whitespace words). Sides: variant (the cases, case labels) and
    original (each original once, original labels; not ROB-04).
  - Metrics (README 6.1, eval_lib.metrics): per run x cell x side x label. label "all" only when both labels
    are present (F1, precision, recall, FPR, ROC-AUC on the margin); "1" reports recall, "0" FPR. N/A
    (zero denominator) is an empty cell / null. Seed mean and sample std (n - 1) only when all seeds are
    defined (models.csv).
  - Change rate (README 6.1): pairs (case, its original) of a cell; changed = predictions differ; rate per
    seed, then mean and std over seeds. Label slices by the ORIGINAL label (as the strata).
  - Paired bootstrap (README 6.1): resampling unit = group_id, and within each comparison groups whose
    texts (cases or originals) share a normalised key (build_lists.nk) are merged into one (README 6.1
    item 1; merged id = smallest group_id; counted in bootstrap.csv nk_merges and summary.json grouping);
    strata = proposed label (cases.csv label: the original label; ROB-04 its own), a group with several is
    "mixed"; groups in group_id order,
    strata in sorted order, each drawn with replacement to its size by numpy default_rng(314159) (restarted
    for every comparison, so the same groups give the same resamples); 2,000 resamples; per resample the
    metric (ratio of summed counts) per seed, then the 3-seed mean, then B - A; CI = numpy.percentile 2.5 /
    97.5 (linear), N/A when any resample is undefined. Comparisons: variant - original (recall, FPR) per
    model x condition/subtype; ROB-01 homoglyph - main on the shared originals; secondary: model b - model
    a on the variants per cell; ROB-04 FPR level CI.
  - Outputs: predictions.csv, metrics.csv, models.csv, changes.csv, bootstrap.csv, summary.json,
    summary.md, each written to <name>.tmp then os.replace. IDs and numbers only (never text or absolute
    paths), LF, non-integers with 6 decimals (the margin check keeps full precision), no timestamps: the
    same inputs and device give the same bytes. Exploratory outputs carry an "exploratory" column / flag.
"""
import argparse
import csv
import gc
import io
import itertools
import json
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

import build_dev_cases  # noqa: E402
import build_lists  # noqa: E402
import eval_lib  # noqa: E402
import model_specs  # noqa: E402
import summarize_s2  # noqa: E402
import train  # noqa: E402

SCRIPT_REL = "scripts/eval_cases.py"
DIR_REL = "data/cases/dev"
OUT_REL = "results/s3"
OUT_EXPLORATORY_REL = "results/s3_exploratory"
S2_REL = "results/s2"
RUN_FILES = summarize_s2.INPUT_FILES
CASES_COLUMNS = build_dev_cases.CASE_COLUMNS
REVIEW_COLUMNS = build_dev_cases.REVIEW_COLUMNS
STATUSES = build_dev_cases.STATUSES
DECISIONS = build_dev_cases.DECISIONS
CONDITIONS = ("ROB-01", "ROB-02", "ROB-03", "ROB-04")
SUBTYPES = {"ROB-01": ("case", "space", "char", "homoglyph"), "ROB-02": ("paraphrase",), "ROB-03": ("context",),
            "ROB-04": ("benign_security",)}
HELD_OUT = (("ROB-01", "homoglyph"),)
REVIEWED = ("assigned", "review_excluded")  # statuses that may have a review row
USED = ("keep", "relabel")
POSITIONS = ("prefix", "suffix", "both")
LENGTHS = ("short", "long")
LONG_WORDS = 42  # ROB-03 prompt v1: short passages 15-35 words, long 50-90; a side with more words is long
N_RESAMPLES = 2000
BOOT_SEED = 314159
PERCENTILES = (2.5, 97.5)
BOOT_METRICS = ("recall", "fpr")
COUNTS = ("tp", "fp", "fn", "tn")
METRICS = ("f1", "precision", "recall", "fpr", "roc_auc")
REPORTED = {"all": COUNTS + METRICS, "1": ("tp", "fn", "recall"), "0": ("fp", "tn", "fpr")}
CELL_KEYS = ("condition", "subtype", "slice")
METRIC_COLUMNS = (("model", "seed", "run_id") + CELL_KEYS + ("side", "label", "n") + COUNTS + METRICS)
CHANGE_COLUMNS = (("model", "seed", "run_id") + CELL_KEYS
                  + ("label", "n_pairs", "changed", "to_attack", "to_benign", "change_rate"))
BOOT_COLUMNS = ("kind", "condition", "subtype_a", "subtype_b", "side_a", "side_b", "model_a", "model_b", "metric",
                "n_groups", "strata", "estimate_a", "estimate_b", "diff", "diff_low", "diff_high", "ci_vs_0",
                "a_low", "a_high", "b_low", "b_high", "undefined_resamples", "nk_merges")
PRED_COLUMNS = ("case_id", "run_id", "margin", "score", "prediction")
NAMES = summarize_s2.NAMES
FULL_PRECISION = ("max_abs_margin_diff_9_digits",)  # summary.json keys not rounded to 6 decimals


class Refused(Exception):
    def __init__(self, errors):
        super().__init__(f"{len(errors)} error(s)")
        self.errors = list(errors)


# ---------------------------------------------------------------- cases and review

def read_table(path, columns, name, code):
    """(raw bytes, [row dicts], errors) of a UTF-8 csv-module file whose header is exactly columns."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        return None, None, [f"{code} {name}: cannot read ({type(exc).__name__})"]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw, None, [f"{code} {name}: not UTF-8"]
    if text.startswith("\ufeff"):
        return raw, None, [f"{code} {name}: starts with a BOM"]
    try:
        records = list(csv.reader(io.StringIO(text, newline="")))
    except csv.Error as exc:
        return raw, None, [f"{code} {name}: csv error ({exc})"]
    if not records or tuple(records[0]) != columns:
        return raw, None, [f"{code} {name}: header is not {','.join(columns)}"]
    rows, errs = [], []
    for n, rec in enumerate(records[1:], start=1):
        if len(rec) != len(columns):
            errs.append(f"{code} {name}: record {n} has {len(rec)} fields, not {len(columns)}")
        else:
            rows.append(dict(zip(columns, rec)))
    return raw, rows, errs


def check_cases(rows, val):
    """({case_id: row}, errors); val is {sentence_id: (label str, text)} of validation."""
    cases, errs, seen = {}, [], {}
    for r in rows:
        cid, cond, sub = r["case_id"], r["condition"], r["subtype"]
        tag = f"C02 cases.csv {cid or '(empty case_id)'}:"
        if not cid or cid in cases:
            errs.append(f"{tag} empty or duplicate case_id")
            continue
        cases[cid] = r
        bad = []
        if cond not in SUBTYPES or sub not in SUBTYPES[cond]:
            bad.append("condition/subtype")
        if r["purpose"] != "dev":
            bad.append("purpose (not dev)")
        if r["label"] not in ("0", "1"):
            bad.append("label")
        if r["status"] not in STATUSES:
            bad.append("status")
        if r["eval_group"] != ("held_out" if (cond, sub) in HELD_OUT else "main"):
            bad.append("eval_group")
        if (r["position"] not in POSITIONS) if cond == "ROB-03" else r["position"] != "":
            bad.append("position")
        if cond == "ROB-04":
            if r["source_split"] != "new" or r["group_id"] != cid or r["label"] != "0":
                bad.append("ROB-04 source_split/group_id/label (new, the case_id, 0)")
        elif cond in SUBTYPES:
            if r["source_split"] != "validation" or r["group_id"] != r["source_id"]:
                bad.append("source_split/group_id (validation, the source_id)")
            elif r["source_id"] not in val:
                bad.append("source_id is not a validation row")
            elif r["label"] in ("0", "1") and val[r["source_id"]][0] != r["label"]:
                bad.append("label differs from the original's validation label")
            elif r["status"] == "assigned":
                prev = seen.setdefault((cond, sub, r["source_id"]), cid)
                if prev != cid:
                    bad.append(f"same original as {prev} in {cond} {sub}")
        if bad:
            errs.append(f"{tag} invalid {', '.join(bad)}")
    return cases, errs


def check_review(data, cases):
    """({case_id: row} of the assigned cases, errors) from the review.csv bytes.

    build_dev_cases.parse_review (R01-R09 of the builder), then: every assigned case has a row, other rows
    belong to review_excluded cases and are decided (E10), a relabel's new_label differs from the label
    (R06), a decided row's text_sha256_12 matches the case text (R09)."""
    rows, errs = build_dev_cases.parse_review(data)
    review = {}
    assigned = {cid for cid, r in cases.items() if r["status"] == "assigned"}
    for cid, r in rows.items():
        c, d = cases.get(cid), r["decision"]
        if c is None or c["status"] not in REVIEWED:
            errs.append(f"E10 review.csv {cid}: not an assigned or review_excluded case of cases.csv")
            continue
        if d == "pending" and cid not in assigned:
            errs.append(f"E10 review.csv {cid}: pending row of a review_excluded case")
        if d == "relabel" and r["new_label"] == c["label"]:
            errs.append(f"R06 {cid}: new_label equals the case label")
        if d in DECISIONS and d != "pending" and r["text_sha256_12"] != build_lists.h12(c["text"]):
            errs.append(f"R09 {cid}: text_sha256_12 differs from the case text in cases.csv (stale decision)")
        if cid in assigned:
            review[cid] = r
    missing = sorted(assigned - set(rows))
    if missing:
        errs.append(f"E10 review.csv: {len(missing)} assigned case(s) have no row (first {missing[0]})")
    return review, errs


def rob03_words(text, orig, position):
    """(context words before, after) the original in a ROB-03 text, or None when it is not at its position."""
    if position == "prefix":
        i = len(text) - len(orig)
        ok = i > 0 and text.endswith(orig)
    elif position == "suffix":
        i, ok = 0, text.startswith(orig)
    else:
        i = text.find(orig, 1)
        ok = i > 0
    if not ok:
        return None
    before, after = len(text[:i].split()), len(text[i + len(orig):].split())
    need = {"prefix": (before,), "suffix": (after,), "both": (before, after)}[position]
    return None if min(need) == 0 else (before, after)


def select_cases(cases, review, val):
    """(records of the used cases in case_id order, info, errors)."""
    dropped_groups = sorted({cases[c]["group_id"] for c, r in review.items()
                             if r["decision"] == "exclude" and cases[c]["condition"] == "ROB-01"})
    recs, dropped, errs = [], [], []
    for cid in sorted(review):
        c, d = cases[cid], review[cid]["decision"]
        if d not in USED:
            continue
        if c["condition"] == "ROB-01" and c["group_id"] in dropped_groups:
            dropped.append(cid)
            continue
        derived = c["condition"] != "ROB-04"
        rec = {"case_id": cid, "group_id": c["group_id"], "condition": c["condition"], "subtype": c["subtype"],
               "eval_group": c["eval_group"], "decision": d, "stratum": c["label"],
               "label": int(review[cid]["new_label"] if d == "relabel" else c["label"]),
               "source_id": c["source_id"] if derived else None,
               "orig_label": int(val[c["source_id"]][0]) if derived else None,
               "position": c["position"], "text": c["text"], "nk": build_lists.nk(c["text"]),
               "orig_nk": build_lists.nk(val[c["source_id"]][1]) if derived else None}
        if c["condition"] == "ROB-03":
            w = rob03_words(c["text"], val[c["source_id"]][1], c["position"])
            if w is None:
                errs.append(f"C03 cases.csv {cid}: the original is not at position {c['position']} of the text")
                continue
            rec.update(words_before=w[0], words_after=w[1], length="long" if max(w) > LONG_WORDS else "short")
        recs.append(rec)
    count = {}
    for cid, r in review.items():
        c = cases[cid]
        k = (c["condition"], c["subtype"], c["label"])
        count.setdefault(k, dict.fromkeys(DECISIONS + ("used",), 0))[r["decision"]] += 1
    for r in recs:
        count[(r["condition"], r["subtype"], cases[r["case_id"]]["label"])]["used"] += 1
    status = {s: sum(c["status"] == s for c in cases.values()) for s in STATUSES}
    info = {"status_counts": status, "rob01_groups_dropped": dropped_groups, "rob01_cases_dropped": dropped,
            "relabeled": [r["case_id"] for r in recs if r["decision"] == "relabel"],
            "pending": sorted(c for c, r in review.items() if r["decision"] == "pending"),
            "unapplied_excludes": sorted(c for c, r in review.items() if r["decision"] == "exclude"),
            "counts": [{"condition": k[0], "subtype": k[1], "proposed_label": int(k[2]), **v}
                       for k, v in sorted(count.items(), key=lambda kv: (kv[0][0], SUBTYPES[kv[0][0]].index(kv[0][1]),
                                                                        kv[0][2]))]}
    return recs, info, errs


# ---------------------------------------------------------------- stage-2 records

def read_s2_summary(root):
    """(raw bytes, document, errors) of results/s2/summary.json (inputs_sha256 and runs checked)."""
    try:
        raw = (root / S2_REL / "summary.json").read_bytes()
        doc = json.loads(raw)
        if not isinstance(doc["inputs_sha256"], dict) or not isinstance(doc["runs"], list):
            raise TypeError("inputs_sha256 / runs")
        return raw, doc, []
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, None, [f"M01 {S2_REL}/summary.json: unreadable or missing keys ({type(exc).__name__})"]


def check_config_pin(root, s2doc):
    """errors (E03) unless configs/train.json (LF SHA-256) equals the stage-2 record."""
    try:
        got = data_lists.sha256_lf(root / CONFIG_REL)
    except OSError as exc:
        return [f"E03 {CONFIG_REL}: cannot read ({type(exc).__name__})"]
    want = s2doc["inputs_sha256"].get(CONFIG_REL)
    rec = (s2doc.get("records") or {}).get("config_sha256", want)
    if want is None or got != want or rec != want:
        return [f"E03 {CONFIG_REL}: SHA-256 (LF) {got[:12]} differs from the stage-2 record in "
                f"{S2_REL}/summary.json ({str(want)[:12]}); the evaluation settings must be the frozen ones"]
    return []


def load_stage2(root, models, seeds, s2raw, s2doc):
    """([run dicts in model, seed order], input SHA-256s, errors)."""
    s2 = root / S2_REL
    raw_csv, rows, errs = read_table(s2 / "runs.csv", summarize_s2.RUN_COLUMNS, f"{S2_REL}/runs.csv", "M01")
    raw_json, inputs = s2raw, s2doc["inputs_sha256"]
    try:
        sj_runs = {r["run_id"]: (r["selected_step"], r["checkpoint_sha256"]) for r in s2doc["runs"]}
    except (KeyError, TypeError) as exc:
        return None, None, errs + [f"M01 {S2_REL}/summary.json: runs unreadable ({type(exc).__name__})"]
    if errs or rows is None:
        return None, None, errs
    want = [(m, s, f"s2-{m}-A-s{s}") for m in models for s in seeds]
    by = {r["run_id"]: r for r in rows}
    if len(rows) != len(want) or set(by) != {w[2] for w in want}:
        return None, None, [f"M01 {S2_REL}/runs.csv: runs are not exactly {', '.join(w[2] for w in want)}"]
    shas = {f"{S2_REL}/runs.csv": data_lists.sha256_bytes(raw_csv),
            f"{S2_REL}/summary.json": data_lists.sha256_bytes(raw_json)}
    runs = []
    for m, s, rid in want:
        r, tag = by[rid], f"{rid}:"
        try:
            step, pin = int(r["selected_step"]), r["checkpoint_sha256"]
        except ValueError:
            errs.append(f"M01 {tag} runs.csv selected_step is not an integer")
            continue
        if (r["model"], r["seed"]) != (m, str(s)) or sj_runs.get(rid) != (step, pin):
            errs.append(f"M01 {tag} runs.csv model/seed, or selected step / checkpoint SHA-256 differ from "
                        f"{S2_REL}/summary.json")
            continue
        raw, bad = {}, []
        for n in RUN_FILES:
            rel = f"runs/{rid}/{n}"
            try:
                raw[n] = (root / rel).read_bytes()
            except OSError:
                bad.append(f"{n} missing")
                continue
            shas[rel] = data_lists.sha256_bytes(raw[n])
            if shas[rel] != inputs.get(rel):
                bad.append(f"{n} SHA-256 differs from {S2_REL}/summary.json")
        if bad:
            errs.append(f"M02 {tag} " + "; ".join(bad))
            continue
        try:
            run = json.loads(raw["run.json"])
            ck, steps = run["checkpoint"], run["budget"]["eval_steps"]
            if run["selection"]["selected_step"] != step or ck["step"] != step or ck["sha256"] != pin:
                errs.append(f"M03 {tag} run.json selected step / checkpoint differ from {S2_REL}/runs.csv")
                continue
            size = ck["size"]
        except (ValueError, KeyError, TypeError) as exc:
            errs.append(f"M03 {tag} run.json unreadable or missing keys ({type(exc).__name__})")
            continue
        P, perrs = summarize_s2.read_predictions(raw["predictions_validation.csv"], steps)
        if perrs or step not in P["pred"]:
            errs.append(f"M03 {tag} predictions_validation.csv: {'; '.join(perrs) or 'no selected step'}")
            continue
        runs.append({"model": m, "seed": s, "run_id": rid, "step": step, "sha256": pin, "size": size,
                     "ids": P["ids"], "labels": P["labels"], "pred": P["pred"][step], "margin": P["margin"][step]})
    return runs, shas, errs


# ---------------------------------------------------------------- prediction

def choose_device(cfg):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        det = train.apply_determinism(cfg)
        return device, {"type": "cuda", "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
                        "determinism": det}
    return torch.device("cpu"), {"type": "cpu", "gpu": None, "torch": torch.__version__, "determinism": None}


def free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def predict_runs(root, runs, recs, val, cfg, device):
    """({run_id: {"case": {case_id: (margin, score, pred)}, "orig": {sentence_id: (...)}}}, checks); raises Refused."""
    sids = sorted(val)
    val_texts, val_labels = [val[s][1] for s in sids], [int(val[s][0]) for s in sids]
    cids, case_texts = [r["case_id"] for r in recs], [r["text"] for r in recs]
    need = {r["source_id"] for r in recs if r["source_id"]}
    thr, max_len, batch = cfg["threshold"], cfg["max_length"], cfg["eval"]["batch"]
    enc, P, checks = {}, {}, {}
    for run in runs:
        rid, key = run["run_id"], run["model"]
        if run["ids"] != sids or run["labels"] != val_labels:
            raise Refused([f"M05 {rid}: validation IDs or labels differ from predictions_validation.csv"])
        try:
            model, tok, _ = eval_lib.load_run_model(root / "runs" / rid, cfg["attn_implementation"],
                                                    expected_sha256=run["sha256"], expected_size=run["size"])
        except (RuntimeError, OSError, KeyError, TypeError, ValueError) as exc:
            raise Refused([f"M04 {rid}: checkpoint refused ({train._exc_brief(exc)})"]) from None
        try:
            if key not in enc:
                enc[key] = (eval_lib.encode(tok, val_texts, max_len), eval_lib.encode(tok, case_texts, max_len))
            (vids, vstats), (cenc, cstats) = enc[key]
            model.to(device)
            vm, vs = eval_lib.predict(model, vids, tok.pad_token_id, batch, device)
            cm, cs = eval_lib.predict(model, cenc, tok.pad_token_id, batch, device)
        finally:
            del model
            free(device)
        if not (torch.isfinite(vm).all() and torch.isfinite(cm).all()):
            raise Refused([f"M06 {rid}: non-finite margins"])
        vp, cp = eval_lib.predictions(vs, thr).tolist(), eval_lib.predictions(cs, thr).tolist()
        vm, vs, cm, cs = vm.tolist(), vs.tolist(), cm.tolist(), cs.tolist()
        flips = sum(a != b for a, b in zip(vp, run["pred"]))
        if flips:
            raise Refused([f"M05 {rid}: {flips} of {len(vp)} original (validation) predictions differ from "
                           f"predictions_validation.csv at the selected step {run['step']}"])
        vm9 = [float(eval_lib.fmt9(a)) for a in vm]  # stored margins have 9 significant digits
        checks[rid] = {
            "validation_rows": len(vp), "prediction_flips": flips,
            "max_abs_margin_diff_9_digits": max(abs(a - b) for a, b in zip(vm9, run["margin"])),
            "margin_rows_differing_9_digits": sum(a != b for a, b in zip(vm9, run["margin"])),
            "truncation": {"validation": vstats, "cases": cstats}}
        P[rid] = {"case": {c: (cm[i], cs[i], cp[i]) for i, c in enumerate(cids)},
                  "orig": {s: (vm[i], vs[i], vp[i]) for i, s in enumerate(sids) if s in need}}
    return P, checks


# ---------------------------------------------------------------- aggregation

def cell_defs(recs):
    """[(condition, subtype, slice, records)] in report order (only non-empty cells)."""
    out = []

    def add(cond, sub, sl, rs):
        if rs:
            out.append((cond, sub, sl, rs))

    for cond in CONDITIONS:
        rc = [r for r in recs if r["condition"] == cond]
        for sub in SUBTYPES[cond]:
            if cond == "ROB-01" and sub == "homoglyph":
                add(cond, "main", "all", [r for r in rc if r["eval_group"] == "main"])
            add(cond, sub, "all", [r for r in rc if r["subtype"] == sub])
        if cond == "ROB-03":
            for p in POSITIONS:
                add(cond, "context", f"position={p}", [r for r in rc if r["position"] == p])
            for ln in LENGTHS:
                add(cond, "context", f"length={ln}", [r for r in rc if r["length"] == ln])
    return out


def sides(cond):
    return ("variant",) if cond == "ROB-04" else ("original", "variant")


def side_items(rs, side):
    """[(group_id, label, (kind, id))]: the cases, or each original once (original label)."""
    if side == "variant":
        return [(r["group_id"], r["label"], ("case", r["case_id"])) for r in rs]
    seen = {}
    for r in rs:
        seen.setdefault(r["source_id"], (r["group_id"], r["orig_label"], ("orig", r["source_id"])))
    return [seen[s] for s in sorted(seen)]


def label_sets(items, with_all_always=False):
    present = {y for _, y, _ in items}
    out = ["all"] if with_all_always or len(present) == 2 else []
    return out + [lab for lab in ("1", "0") if int(lab) in present]


def _pick(items, lab):
    return [it for it in items if lab == "all" or it[1] == int(lab)]


def metric_rows(recs, runs, P):
    rows = []
    for run in runs:
        pr = P[run["run_id"]]
        for cond, sub, sl, rs in cell_defs(recs):
            for side in sides(cond):
                items = side_items(rs, side)
                for lab in label_sets(items):
                    sel = _pick(items, lab)
                    vals = [pr[k][i] for _, _, (k, i) in sel]
                    m = eval_lib.metrics([y for _, y, _ in sel], [v[2] for v in vals], [v[0] for v in vals])
                    rows.append({"model": run["model"], "seed": run["seed"], "run_id": run["run_id"],
                                 "condition": cond, "subtype": sub, "slice": sl, "side": side, "label": lab,
                                 "n": m["n"], **{q: m[q] if q in REPORTED[lab] else None for q in COUNTS + METRICS}})
    return rows


def change_rows(recs, runs, P):
    rows = []
    for run in runs:
        pr = P[run["run_id"]]
        for cond, sub, sl, rs in cell_defs(recs):
            if cond == "ROB-04":
                continue
            for lab in label_sets(side_items(rs, "original"), with_all_always=True):  # original labels
                pairs = [(pr["orig"][r["source_id"]][2], pr["case"][r["case_id"]][2])
                         for r in rs if lab == "all" or r["orig_label"] == int(lab)]
                changed = sum(o != v for o, v in pairs)
                rows.append({"model": run["model"], "seed": run["seed"], "run_id": run["run_id"], "condition": cond,
                             "subtype": sub, "slice": sl, "label": lab, "n_pairs": len(pairs), "changed": changed,
                             "to_attack": sum(o == 0 and v == 1 for o, v in pairs),
                             "to_benign": sum(o == 1 and v == 0 for o, v in pairs),
                             "change_rate": changed / len(pairs)})
    return rows


def seed_stat(vals):
    ok = all(v is not None for v in vals)
    return {"mean": statistics.fmean(vals) if ok else None, "std": statistics.stdev(vals) if ok else None,
            "values": vals}


def model_rows(mrows, crows, models, seeds):
    """Seed mean / sample std per model x cell x side x label x quantity (side "change": change rate)."""
    groups = {}
    for r in mrows:
        k = (r["model"],) + tuple(r[c] for c in CELL_KEYS) + (r["side"], r["label"])
        groups.setdefault(k, {})[r["seed"]] = r
    for r in crows:
        k = (r["model"],) + tuple(r[c] for c in CELL_KEYS) + ("change", r["label"])
        groups.setdefault(k, {})[r["seed"]] = {"n": r["n_pairs"], "change_rate": r["change_rate"]}
    cell_rank = {}
    for k in groups:
        cell_rank.setdefault(k[1:4], len(cell_rank))
    order = {"original": 0, "variant": 1, "change": 2, "all": 0, "1": 1, "0": 2}
    out = []
    for k, by_seed in sorted(groups.items(), key=lambda kv: (models.index(kv[0][0]), cell_rank[kv[0][1:4]],
                                                             order[kv[0][4]], order[kv[0][5]])):
        if sorted(by_seed) != sorted(seeds):
            raise RuntimeError(f"model_rows: {k} lacks a seed")
        per = [by_seed[s] for s in seeds]
        if len({p["n"] for p in per}) != 1:
            raise RuntimeError(f"model_rows: {k} n differs between seeds")
        for q in (("change_rate",) if k[4] == "change" else REPORTED[k[5]]):
            st = seed_stat([p[q] for p in per])
            out.append({"model": k[0], "condition": k[1], "subtype": k[2], "slice": k[3], "side": k[4],
                        "label": k[5], "n": per[0]["n"], "quantity": q, "mean": st["mean"], "std": st["std"],
                        **{f"s{s}": v for s, v in zip(seeds, st["values"])}})
    return out


def resample_indices(strata, n_resamples=N_RESAMPLES, seed=BOOT_SEED):
    """(n_resamples, n_groups) group indices: each stratum (sorted) drawn with replacement to its own size."""
    rng = np.random.default_rng(seed)
    parts = []
    for s in sorted(set(strata)):
        members = np.array([i for i, x in enumerate(strata) if x == s], dtype=np.int64)
        parts.append(members[rng.integers(0, len(members), size=(n_resamples, len(members)))])
    return np.concatenate(parts, axis=1)


def group_counts(items, run_ids, P, metric, groups):
    """(num (seeds x groups), den (groups)): recall counts label-1 items, FPR label-0 items; num = predicted 1."""
    want, gi = (1 if metric == "recall" else 0), {g: i for i, g in enumerate(groups)}
    num = np.zeros((len(run_ids), len(groups)), dtype=np.int64)
    den = np.zeros(len(groups), dtype=np.int64)
    for g, y, (kind, i) in items:
        if y == want:
            den[gi[g]] += 1
            for j, rid in enumerate(run_ids):
                num[j, gi[g]] += P[rid][kind][i][2]
    return num, den


def merge_groups(items, nk_of):
    """{group_id: merged group_id}: groups whose texts share a normalised key are one group (README 6.1
    item 1); the merged id is the smallest group_id of the component. Empty keys never merge."""
    parent = {g: g for g, _, _ in items}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    first = {}
    for g, _, key in items:
        k = nk_of.get(key)
        if k:
            a, b = find(g), find(first.setdefault(k, g))
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {g: find(g) for g in parent}


def bootstrap_dist(items_a, runs_a, items_b, runs_b, metric, stratum, P, nk_of=None):
    """{"groups", "strata", "merged", "a": (estimate, dist), "b": ... or None, "diff": dist or None}.

    Groups of the comparison sharing a normalised key (nk_of: item key -> build_lists.nk) are merged first;
    a merged group's stratum is its members' stratum, "mixed" if they differ. estimate: 3-seed mean of the
    full-sample metric (None when a seed is undefined); dist: per-resample 3-seed mean (NaN when undefined)."""
    merged = merge_groups(items_a + (items_b or []), nk_of or {})
    items_a = [(merged[g], y, k) for g, y, k in items_a]
    items_b = None if items_b is None else [(merged[g], y, k) for g, y, k in items_b]
    members = {}
    for g, m in merged.items():
        members.setdefault(m, []).append(g)
    groups = sorted(members)
    strata = []
    for m in groups:
        s = {stratum[g] for g in members[m]}
        strata.append(next(iter(s)) if len(s) == 1 else "mixed")
    idx = resample_indices(strata)
    out = {"groups": groups, "strata": strata, "b": None, "diff": None,
           "merged": [sorted(members[m]) for m in groups if len(members[m]) > 1]}
    for name, items, rids in (("a", items_a, runs_a), ("b", items_b, runs_b)):
        if items is None:
            continue
        num, den = group_counts(items, rids, P, metric, groups)
        total = int(den.sum())
        est = statistics.fmean(int(num[j].sum()) / total for j in range(len(rids))) if total else None
        rnum, rden = num[:, idx].sum(axis=2), den[idx].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(rden > 0, rnum / np.where(rden > 0, rden, 1), np.nan)
        out[name] = (est, rate.mean(axis=0))
    if out["b"] is not None:
        out["diff"] = out["b"][1] - out["a"][1]
    return out


def _ci(dist):
    if dist is None or np.isnan(dist).any():
        return None, None
    lo, hi = np.percentile(dist, PERCENTILES)
    return float(lo), float(hi)


def bootstrap_row(kind, cond, sub_a, sub_b, side_a, side_b, model_a, model_b, metric, res):
    (ea, da), b = res["a"], res["b"]
    eb, db = b if b is not None else (None, None)
    dl, dh = _ci(res["diff"])
    al, ah = _ci(da)
    bl, bh = _ci(db)
    count = {s: res["strata"].count(s) for s in sorted(set(res["strata"]))}
    undefined = res["diff"] if res["diff"] is not None else da
    return {"kind": kind, "condition": cond, "subtype_a": sub_a, "subtype_b": sub_b, "side_a": side_a,
            "side_b": side_b, "model_a": model_a, "model_b": model_b, "metric": metric,
            "n_groups": len(res["groups"]), "strata": ";".join(f"{s}:{n}" for s, n in count.items()),
            "estimate_a": ea, "estimate_b": eb, "diff": None if ea is None or eb is None else eb - ea,
            "diff_low": dl, "diff_high": dh,
            "ci_vs_0": "" if dl is None else "<0" if dh < 0 else ">0" if dl > 0 else "includes_0",
            "a_low": al, "a_high": ah, "b_low": bl, "b_high": bh,
            "undefined_resamples": int(np.isnan(undefined).sum()),
            "nk_merges": sum(len(g) - 1 for g in res["merged"])}


def bootstrap_rows(recs, runs, P, merged_sets=None):
    """Bootstrap rows; merged_sets (a set) collects the nk-merged group sets of every comparison."""
    models = list(dict.fromkeys(r["model"] for r in runs))
    nk_of = {}
    for r in recs:
        nk_of[("case", r["case_id"])] = r["nk"]
        if r["source_id"]:
            nk_of[("orig", r["source_id"])] = r["orig_nk"]
    rids = {m: [r["run_id"] for r in runs if r["model"] == m] for m in models}
    labels = {}
    for r in recs:
        labels.setdefault(r["group_id"], set()).add(r["stratum"])
    stratum = {g: next(iter(s)) if len(s) == 1 else "mixed" for g, s in labels.items()}
    want = {"recall": 1, "fpr": 0}
    cells = [c for c in cell_defs(recs) if c[2] == "all"]
    rows = []
    merged_sets = set() if merged_sets is None else merged_sets

    def has(items, metric):
        return any(y == want[metric] for _, y, _ in items)

    def boot(*args):
        res = bootstrap_dist(*args, nk_of=nk_of)
        merged_sets.update(tuple(g) for g in res["merged"])
        return res

    for m in models:
        for cond, sub, _, rs in cells:
            if cond == "ROB-04":
                items = side_items(rs, "variant")
                if has(items, "fpr"):
                    res = boot(items, rids[m], None, None, "fpr", stratum, P)
                    rows.append(bootstrap_row("level", cond, sub, "", "variant", "", m, "", "fpr", res))
                continue
            a, b = side_items(rs, "original"), side_items(rs, "variant")
            for metric in BOOT_METRICS:
                if has(a, metric) and has(b, metric):
                    res = boot(a, rids[m], b, rids[m], metric, stratum, P)
                    rows.append(bootstrap_row("variant_vs_original", cond, sub, sub, "original", "variant", m, m,
                                              metric, res))
        main = [r for r in recs if r["condition"] == "ROB-01" and r["eval_group"] == "main"]
        held = [r for r in recs if r["condition"] == "ROB-01" and r["eval_group"] == "held_out"]
        shared = {r["group_id"] for r in main} & {r["group_id"] for r in held}
        a = side_items([r for r in main if r["group_id"] in shared], "variant")
        b = side_items([r for r in held if r["group_id"] in shared], "variant")
        for metric in BOOT_METRICS:
            if has(a, metric) and has(b, metric):
                res = boot(a, rids[m], b, rids[m], metric, stratum, P)
                rows.append(bootstrap_row("held_out_vs_main", "ROB-01", "main", "homoglyph", "variant", "variant",
                                          m, m, metric, res))
    for ma, mb in itertools.combinations(models, 2):
        for cond, sub, _, rs in cells:
            items = side_items(rs, "variant")
            for metric in BOOT_METRICS:
                if has(items, metric):
                    res = boot(items, rids[ma], items, rids[mb], metric, stratum, P)
                    rows.append(bootstrap_row("model_vs_model", cond, sub, sub, "variant", "variant", ma, mb,
                                              metric, res))
    return rows


def evaluate(recs, runs, P):
    models = list(dict.fromkeys(r["model"] for r in runs))
    seeds = list(dict.fromkeys(r["seed"] for r in runs))
    mrows, crows = metric_rows(recs, runs, P), change_rows(recs, runs, P)
    merged = set()
    boot = bootstrap_rows(recs, runs, P, merged)
    return {"models": models, "seeds": seeds, "metrics": mrows, "changes": crows,
            "model_rows": model_rows(mrows, crows, models, seeds), "bootstrap": boot,
            "grouping": {"nk_merged_group_sets": [list(g) for g in sorted(merged)],
                         "comparisons_with_nk_merges": sum(r["nk_merges"] > 0 for r in boot)}}


# ---------------------------------------------------------------- output

def _round(obj, key=None):
    if isinstance(obj, float):
        return obj if key in FULL_PRECISION else round(obj, summarize_s2.DP)
    if isinstance(obj, dict):
        return {k: _round(v, k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, key) for v in obj]
    return obj


def csv_out(columns, rows, exploratory):
    if exploratory:
        columns, rows = columns + ("exploratory",), [{**r, "exploratory": 1} for r in rows]
    return summarize_s2.csv_text(columns, rows)


def prediction_rows(runs, recs, P):
    return [{"case_id": r["case_id"], "run_id": run["run_id"], "margin": eval_lib.fmt9(P[run["run_id"]]["case"][
        r["case_id"]][0]), "score": eval_lib.fmt9(P[run["run_id"]]["case"][r["case_id"]][1]),
             "prediction": P[run["run_id"]]["case"][r["case_id"]][2]} for run in runs for r in recs]


def _f4(v):
    return "N/A" if v is None else f"{v:.4f}"


def _ci_text(v, lo, hi):
    if v is None:
        return "N/A"
    return f"{v:+.4f} [{lo:+.4f}, {hi:+.4f}]" if lo is not None else f"{v:+.4f} [N/A]"


def summary_md(res):
    ev, models, seeds = res["eval"], res["eval"]["models"], res["eval"]["seeds"]
    st = {(r["model"], r["condition"], r["subtype"], r["slice"], r["side"], r["label"], r["quantity"]): r
          for r in ev["model_rows"]}
    bs = {(r["kind"], r["model_a"], r["model_b"], r["condition"], r["subtype_b"], r["metric"]): r
          for r in ev["bootstrap"]}

    def pm(m, c, sub, sl, side, lab, q):
        r = st.get((m, c, sub, sl, side, lab, q))
        if r is None:
            return "-"
        return summarize_s2._pm({"mean": r["mean"], "std": r["std"], "values": [r[f"s{s}"] for s in seeds]}, 4)

    def n(m, c, sub, sl, side, lab):
        r = st.get((m, c, sub, sl, side, lab, "recall" if lab == "1" else "fpr"))
        return r["n"] if r else 0

    def ci(kind, m, mb, c, sub, metric):
        r = bs.get((kind, m, mb, c, sub, metric))
        return "-" if r is None else _ci_text(r["diff"], r["diff_low"], r["diff_high"])

    name = {m: NAMES.get(m, m) for m in models}
    cells = [(c, sub, sl) for c, sub, sl, _ in res["cells"]]
    info = res["cases"]
    out = ["# 3단계 개발용 추가 평가 결과", ""]
    if res["exploratory"]:
        out += [f"**탐색적 결과**: 검수가 끝나지 않은 사례(pending {len(info['pending'])}개, 대체가 적용되지 않은 exclude "
                f"{len(info['unapplied_excludes'])}개)를 빼고 계산했다(`--allow-incomplete`). 판정 근거로 쓰지 않는다.", ""]
    out += [f"`{SCRIPT_REL}`가 검수한 개발용 사례(keep·relabel)를 2단계 {len(res['runs'])}개 실행(seed "
            f"{'·'.join(str(s) for s in seeds)})의 선택 체크포인트로 임계값 {res['settings']['threshold']}에서 판정한 결과다. "
            "값은 seed 평균 ± 표본 표준편차(n−1)이고, 하나라도 N/A인 seed가 있으면 seed별 값을 적는다. "
            "차이는 변형 − 원문이며 괄호는 원문 그룹 단위 짝지은 bootstrap 95% CI(2,000회, seed 314159, 원문 레이블 층화, "
            "정규화 키가 같은 그룹은 합침)다. 판정 변화율은 원문과 변형의 예측이 다른 쌍의 비율이며 레이블 구분은 원문 레이블이다. "
            "전체 값은 `metrics.csv`·`models.csv`·`changes.csv`·`bootstrap.csv`·`summary.json`에 있다.", "",
            "## 사례 수", "", "| 조건 | 하위 유형 | 제안 레이블 | 배정 | keep | relabel | exclude | pending | 평가 사용 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in info["counts"]:
        out.append(f"| {c['condition']} | {c['subtype']} | {c['proposed_label']} | "
                   f"{sum(c[d] for d in DECISIONS)} | {c['keep']} | {c['relabel']} | {c['exclude']} | {c['pending']} | "
                   f"{c['used']} |")
    out += ["", f"- 배정 사례만 센다. 검수 exclude로 대체된 사례(`review_excluded`): "
            f"{info['status_counts']['review_excluded']}개. relabel 사례: {len(info['relabeled'])}개. "
            f"대체 전 exclude 때문에 ROB-01 네 하위 유형을 모두 뺀 원문: {len(info['rob01_groups_dropped'])}개.", "",
            "## 변형 전후 비교 (RQ-02, 개발용)", "",
            "| 모델 | 조건 | 하위 유형 | 원문 수 (공격/정상) | 탐지율 원문 | 탐지율 변형 | 탐지율 차이 [CI] | FPR 원문 | FPR 변형 | "
            "FPR 차이 [CI] | 판정 변화율 |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in models:
        for c, sub, sl in cells:
            if c == "ROB-04" or sl != "all":
                continue
            out.append(f"| {name[m]} | {c} | {sub} | {n(m, c, sub, sl, 'original', '1')}/"
                       f"{n(m, c, sub, sl, 'original', '0')} | {pm(m, c, sub, sl, 'original', '1', 'recall')} | "
                       f"{pm(m, c, sub, sl, 'variant', '1', 'recall')} | "
                       f"{ci('variant_vs_original', m, m, c, sub, 'recall')} | "
                       f"{pm(m, c, sub, sl, 'original', '0', 'fpr')} | {pm(m, c, sub, sl, 'variant', '0', 'fpr')} | "
                       f"{ci('variant_vs_original', m, m, c, sub, 'fpr')} | "
                       f"{pm(m, c, sub, sl, 'change', 'all', 'change_rate')} |")
    def ho(m, metric):
        r = bs.get(("held_out_vs_main", m, m, "ROB-01", "homoglyph", metric))
        if r is None:
            return "- | - | -"
        return (f"{_f4(r['estimate_a'])} | {_f4(r['estimate_b'])} | "
                f"{_ci_text(r['diff'], r['diff_low'], r['diff_high'])}")

    out += ["", "## ROB-01 보류군(homoglyph)과 같은 유형 평가군(case·space·char) (같은 원문)", "",
            "| 모델 | 원문 수 | 탐지율 main | 탐지율 homoglyph | 차이 [CI] | FPR main | FPR homoglyph | 차이 [CI] |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in models:
        r = bs.get(("held_out_vs_main", m, m, "ROB-01", "homoglyph", "recall")) or \
            bs.get(("held_out_vs_main", m, m, "ROB-01", "homoglyph", "fpr"))
        out.append(f"| {name[m]} | {r['n_groups'] if r else 0} | {ho(m, 'recall')} | {ho(m, 'fpr')} |")
    out += ["", "- 두 하위 유형이 모두 있는 원문만 쓴다. 값은 3-seed 평균이고, main은 같은 원문의 case·space·char 사례를 합친 값이다. "
            "CI는 homoglyph − main이며 같은 원문을 함께 재표집한다.",
            "", f"## ROB-03 위치·길이별 결과 (긴 문맥: 한쪽 문맥이 {LONG_WORDS}단어 초과)", "",
            "| 모델 | 구분 | 사례 수 (공격/정상) | 탐지율 원문 | 탐지율 변형 | FPR 원문 | FPR 변형 | 판정 변화율 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in models:
        for c, sub, sl in cells:
            if c != "ROB-03":
                continue
            out.append(f"| {name[m]} | {sl} | {n(m, c, sub, sl, 'variant', '1')}/{n(m, c, sub, sl, 'variant', '0')} | "
                       f"{pm(m, c, sub, sl, 'original', '1', 'recall')} | "
                       f"{pm(m, c, sub, sl, 'variant', '1', 'recall')} | "
                       f"{pm(m, c, sub, sl, 'original', '0', 'fpr')} | {pm(m, c, sub, sl, 'variant', '0', 'fpr')} | "
                       f"{pm(m, c, sub, sl, 'change', 'all', 'change_rate')} |")
    out += ["", "## ROB-04 어려운 정상 입력 (RQ-03, 개발용)", "", "| 모델 | 사례 수 | FPR | FPR 95% CI |",
            "| --- | ---: | ---: | ---: |"]
    for m in models:
        r = bs.get(("level", m, "", "ROB-04", "", "fpr"))
        lv = "-" if r is None or r["a_low"] is None else f"[{r['a_low']:.4f}, {r['a_high']:.4f}]"
        out.append(f"| {name[m]} | {n(m, 'ROB-04', 'benign_security', 'all', 'variant', '0')} | "
                   f"{pm(m, 'ROB-04', 'benign_security', 'all', 'variant', '0', 'fpr')} | {lv} |")
    out += ["", "- 모델 간 차이(보조 결과)는 `bootstrap.csv`의 `model_vs_model` 행(모델 b − 모델 a, 변형 사례)에 있다.",
            "- 원문 예측은 validation 전체를 학습 때와 같은 순서·배치로 다시 계산했고, 실행마다 `predictions_validation.csv`의 "
            "선택 step 예측과 모두 같았다(마진 차이는 `summary.json`).", ""]
    return "\n".join(out)


def summary_json(res, script_sha):
    ev = res["eval"]
    doc = {
        "schema": 1,
        "readme": "README 5.2 (ROB-01~04), 6.1 (metrics, change rate, paired bootstrap), 4.2 (threshold, FP32 "
                  "eval, batch, dynamic padding, max length); spec-stage3 Evaluation; written by " + SCRIPT_REL,
        "exploratory": res["exploratory"],
        "rules": {
            "cases": "status assigned with review decision keep or relabel (label = new_label); pending or "
                     "exclude (not yet replaced by build_dev_cases.py --apply-review) refused unless "
                     "--allow-incomplete, which leaves them out (a ROB-01 exclude takes the whole original out)",
            "models": "nine stage-2 runs; checkpoint SHA-256 and selected step pinned from results/s2/runs.csv, "
                      "run files' SHA-256 from results/s2/summary.json; eval_lib.load_run_model",
            "originals": "all validation rows predicted in sentence-ID order (eval batch), predictions equal to "
                         "predictions_validation.csv at the selected step (else refused)",
            "metrics": "eval_lib.metrics; label all only with both labels (F1, precision, recall, FPR, ROC-AUC "
                       "on the margin), 1: recall, 0: FPR; N/A is null",
            "seeds": "mean and sample std (n - 1) over the seeds; null when any seed is N/A",
            "change_rate": "pairs (case, original) with different predictions / pairs, per seed; label slice = "
                           "the original label",
            "rob03": f"position from cases.csv; length long when a context side has more than {LONG_WORDS} "
                     "whitespace words, else short",
            "bootstrap": f"groups = group_id, merged within each comparison when texts share build_lists.nk "
                         "(merged id = smallest group_id), strata = proposed label (mixed if several), groups "
                         "sorted, strata "
                         f"sorted, numpy default_rng({BOOT_SEED}) restarted per comparison, integers(0, n_s, "
                         f"({N_RESAMPLES}, n_s)) per stratum; per resample the ratio of summed counts per seed, "
                         "3-seed mean, then B - A; numpy.percentile 2.5 / 97.5 (linear); null CI when any "
                         "resample is undefined",
            "config": "configs/train.json SHA-256 (LF) equal to the stage-2 record before any setting is read",
            "numbers": f"non-integers rounded to {summarize_s2.DP} decimals, except {', '.join(FULL_PRECISION)}",
        },
        "settings": res["settings"], "device": res["device"], "script_sha256_lf": script_sha,
        "inputs_sha256": res["inputs"], "models": ev["models"], "seeds": ev["seeds"],
        "runs": [{"run_id": r["run_id"], "model": r["model"], "seed": r["seed"], "selected_step": r["step"],
                  "checkpoint_sha256": r["sha256"], "checkpoint_size": r["size"],
                  "original_check": res["checks"][r["run_id"]]} for r in res["runs"]],
        "cases": {k: v for k, v in res["cases"].items()},
        "rob03_cases": [{"case_id": r["case_id"], "position": r["position"], "words_before": r["words_before"],
                         "words_after": r["words_after"], "length": r["length"]}
                        for r in res["recs"] if r["condition"] == "ROB-03"],
        "metrics": ev["metrics"], "models_summary": ev["model_rows"], "changes": ev["changes"],
        "bootstrap": ev["bootstrap"], "grouping": ev["grouping"],
    }
    return train._dump(_round(doc))


def write_outputs(out_dir, res, script_sha):
    ex, ev = res["exploratory"], res["eval"]
    seeds = ev["seeds"]
    model_cols = (("model",) + CELL_KEYS + ("side", "label", "n", "quantity", "mean", "std")
                  + tuple(f"s{s}" for s in seeds))
    files = {
        "predictions.csv": csv_out(PRED_COLUMNS, prediction_rows(res["runs"], res["recs"], res["P"]), ex),
        "metrics.csv": csv_out(METRIC_COLUMNS, ev["metrics"], ex),
        "models.csv": csv_out(model_cols, ev["model_rows"], ex),
        "changes.csv": csv_out(CHANGE_COLUMNS, ev["changes"], ex),
        "bootstrap.csv": csv_out(BOOT_COLUMNS, ev["bootstrap"], ex),
        "summary.json": summary_json(res, script_sha),
        "summary.md": summary_md(res),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, text in files.items():
        tmp = out_dir / (n + ".tmp")
        tmp.write_bytes(text.encode("utf-8"))
        os.replace(tmp, out_dir / n)
    return list(files)


# ---------------------------------------------------------------- main

def _rel(path, root):
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).name


def run_eval(root, cdir, allow_incomplete):
    """(errors, result). Nothing is written here."""
    s2raw, s2doc, errs = read_s2_summary(root)
    errs = errs or check_config_pin(root, s2doc)
    if errs:
        return errs, None
    try:
        cfg_bytes = (root / CONFIG_REL).read_bytes()
        cfg = json.loads(cfg_bytes)
        settings = {"threshold": cfg["threshold"], "max_length": cfg["max_length"], "eval_batch": cfg["eval"]["batch"],
                    "eval_precision": cfg["eval"]["precision"], "seeds": cfg["seeds"],
                    "attn_implementation": cfg["attn_implementation"], "determinism": cfg["determinism"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"E00 {CONFIG_REL}: unreadable or missing keys ({type(exc).__name__})"], None
    if settings["eval_precision"] != "fp32" or settings["threshold"] != 0.5:
        return [f"E00 {CONFIG_REL}: eval precision must be fp32 and threshold 0.5 (README 4.2)"], None
    data_lists.install_test_guard()
    try:
        val = data_lists.load_rows()["validation"]
    except data_lists.ListError as exc:
        return [f"E01 {e}" for e in exc.errors], None
    crel, rrel = f"{_rel(cdir, root)}/cases.csv", f"{_rel(cdir, root)}/review.csv"
    craw, crows, errs = read_table(cdir / "cases.csv", CASES_COLUMNS, crel, "C01")
    try:
        rraw = (cdir / "review.csv").read_bytes()
    except OSError as exc:
        return errs + [f"R01 {rrel}: cannot read ({type(exc).__name__})"], None
    if crows is None:
        return errs, None
    cases, e = check_cases(crows, val)
    errs += e
    review, e = check_review(rraw, cases)
    errs += e
    if errs:
        return errs, None
    recs, info, errs = select_cases(cases, review, val)
    if info["pending"] and not allow_incomplete:
        errs.append(f"P01 review.csv: {len(info['pending'])} assigned case(s) are still pending (first "
                    f"{info['pending'][0]}); finish the review, or use --allow-incomplete for exploratory output")
    if info["unapplied_excludes"] and not allow_incomplete:
        errs.append(f"P02 review.csv: {len(info['unapplied_excludes'])} assigned case(s) have decision exclude "
                    f"(first {info['unapplied_excludes'][0]}); run scripts/build_dev_cases.py --apply-review so the "
                    "replacements are made and reviewed, or use --allow-incomplete for exploratory output")
    if not recs and not errs:
        errs.append("N01 no case to evaluate (no keep or relabel decision)")
    models = list(model_specs.MODELS)
    runs, shas, e = load_stage2(root, models, settings["seeds"], s2raw, s2doc)
    errs += e
    if errs:
        return errs, None
    device, dev_info = choose_device(cfg)
    try:
        P, checks = predict_runs(root, runs, recs, val, cfg, device)
    except Refused as exc:
        return exc.errors, None
    inputs = {CONFIG_REL: data_lists.sha256_bytes(cfg_bytes), crel: data_lists.sha256_bytes(craw),
              rrel: data_lists.sha256_bytes(rraw), **shas}
    return [], {"exploratory": bool(allow_incomplete), "settings": settings, "device": dev_info,
                "inputs": dict(sorted(inputs.items())), "runs": runs, "checks": checks, "recs": recs, "cases": info,
                "cells": cell_defs(recs), "P": P, "eval": evaluate(recs, runs, P)}


def parse_args(argv):
    p = argparse.ArgumentParser(description="Evaluate the reviewed development cases with the nine stage-2 models "
                                            "(README 5.2, 6.1) into results/s3/.")
    p.add_argument("--dir", help=f"cases folder with cases.csv and review.csv (default: <root>/{DIR_REL})")
    p.add_argument("--out", help=f"output folder (default: <root>/{OUT_REL}, or <root>/{OUT_EXPLORATORY_REL} with "
                                 "--allow-incomplete)")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="leave pending cases out and mark every output exploratory (never into results/s3)")
    p.add_argument("--root", help="repository root holding runs/, results/s2/ and configs/; must be the root "
                                  "eval_lib.load_run_model uses (eval_lib.ROOT), so only tests change it")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root).resolve() if args.root else ROOT
    cdir = Path(args.dir) if args.dir else root / DIR_REL
    out = Path(args.out) if args.out else root / (OUT_EXPLORATORY_REL if args.allow_incomplete else OUT_REL)
    s3 = (root / OUT_REL).resolve()
    if root != Path(eval_lib.ROOT).resolve():
        errors, res = [f"E04 --root must be the repository root that eval_lib.load_run_model uses "
                       f"(eval_lib.ROOT); other roots are not supported"], None
    elif args.allow_incomplete and (out.resolve() == s3 or s3 in out.resolve().parents):
        errors, res = [f"E02 --allow-incomplete output must not be inside {OUT_REL}"], None
    else:
        errors, res = run_eval(root, cdir, args.allow_incomplete)
    if errors:
        for e in errors:
            print(train._ascii("FAIL " + e))
        print(f"refused: {len(errors)} error(s); nothing written")
        return 1
    names = write_outputs(out, res, data_lists.sha256_lf(ROOT / SCRIPT_REL))
    ex = (f" (EXPLORATORY: {len(res['cases']['pending'])} pending and {len(res['cases']['unapplied_excludes'])} "
          "unapplied exclude case(s) left out)") if res["exploratory"] else ""
    print(train._ascii(f"OK {len(res['recs'])} cases x {len(res['runs'])} runs on {res['device']['type']}{ex}; "
                       f"wrote {_rel(out, root)}/: {', '.join(names)}"))
    for rid, c in res["checks"].items():
        print(f"{rid}: originals reproduced ({c['validation_rows']} rows, 0 flips; 9-digit margins differ in "
              f"{c['margin_rows_differing_9_digits']} rows, max {c['max_abs_margin_diff_9_digits']:.3g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
