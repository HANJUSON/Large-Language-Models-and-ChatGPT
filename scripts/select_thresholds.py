"""Per-seed decision thresholds for conditions B and D -> results/s4/ (README 5.3 item 1, IMP-01).

Run from Git Bash at the repo root after the three condition-C runs have finished:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/select_thresholds.py
Options: --check selects again in memory and compares with the files already written (exit 1 on
any difference, nothing written); --root DIR / --out DIR for tests and dry runs.

Rules:
  - Runs: the representative model (REPRESENTATIVE = "roberta", README 5.1 EXP-04) and every seed
    of configs/train.json. B is tuned on runs/s2-roberta-A-s<seed>, D on runs/s4-roberta-C-s<seed>
    (IMP-01: A and B share one checkpoint, C and D share one). Both runs of every seed are
    required; nothing is written for a subset.
  - Data: predictions_validation.csv of the run, the rows of the selected step (run.json
    selection.selected_step, which must be the step of the kept checkpoint) with in_clean 1, the
    `score` column (softmax probability of label 1). Derived cases are never read.
  - Candidates: logit t_k = k/20 for k = -160..160 (-8.00..8.00, step 0.05, 321 candidates);
    threshold_k = float32(1 / (1 + exp(-t_k))), so k = 0 is exactly 0.5. The thresholds are
    strictly increasing in k.
  - Decision rule, the same for selection, the stored-row check and the reported metrics:
    attack when float32(score) >= threshold (eval_lib.predictions on a float32 tensor). A stored
    score must be the 9-digit text (eval_lib.fmt9) of a float32 value in [0, 1], so the text names
    one float32 value. The margin column is parsed but not used.
  - FPR cap (exact Fractions): validation_clean FPR of the same seed's A run at threshold 0.5
    plus 1/100. The same cap is used for B (A scores) and D (C scores).
  - Selection: among the candidates with FPR <= cap, the highest recall, then the lowest FPR.
    The candidates with that (TP, FP) outcome are contiguous in k (k_lo..k_hi); the one at index
    (lo + hi) // 2 is chosen (the lower middle of an even run). When no candidate meets the cap:
    the lowest FPR, then the highest recall, the same middle rule, and cap_met is false.
  - Outputs (results/s4/, LF, numbers and codes only, no sentence IDs, no timestamps or absolute
    paths; rows in config seed order, then B, D):
    thresholds.csv  seed, condition, run_id, selected_step, n_clean, n_pos, n_neg, cap_exact, cap,
                    cap_met (1/0), k, logit (k/20, 2 decimals), threshold (repr of the float32 value as a
                    Python float), k_lo, k_hi, then tp, fp, fn, tn, recall, fpr, f1 at the chosen
                    threshold and base_* = the same at threshold 0.5 on the same scores (condition
                    A for B, C for D); non-integers with 6 decimals.
    thresholds.json the same per (seed, condition) with exact fractions, the checkpoint SHA-256
                    recorded in run.json, the SHA-256 of every input file and of this script.
    Each file is written to <name>.tmp, then os.replace.

Refused (exit 1, `FAIL <code> ...` lines, nothing written):
  T00 configs/train.json unreadable, seeds not a non-empty list of distinct integers, or its
      threshold is not 0.5;
  T01 a run folder or one of run.json, metrics.json, predictions_validation.csv is missing, or
      runs/<id>.lock exists (the run is in progress, or was killed);
  T02 run.json or metrics.json is not JSON, misses a key, or the run identity (run ID, stage,
      model, condition, seed) differs from the folder name;
  T03 run.json status is not completed;
  T04 the selected step is not one of the run's eval steps, or the checkpoint is not kept, has
      no recorded SHA-256, or is not the checkpoint of the selected step;
  T05 predictions_validation.csv is malformed: not ASCII, header or final LF differs, a line
      does not parse, a step is not an eval step, label/in_clean/prediction not 0/1, the
      condition column differs from the run's condition, a score is not the 9-digit text of a
      float32 in [0, 1], sentence IDs not strictly ascending, rows differ between steps, or the
      row counts differ from run.json data;
  T06 the threshold column is not 0.5, or a stored prediction != (float32(score) >= 0.5);
  T07 metrics.json run ID, threshold 0.5, eval steps or selected step differ from run.json, or
      the stored scores do not reproduce its validation_clean counts at the selected step;
  T08 validation_clean has no label-0 or no label-1 row (FPR or recall undefined), or the
      validation rows (IDs, labels, in_clean) differ between the runs (one cap for A and C needs
      the same rows);
  T09 --check: an output file is missing or differs from the fresh selection.
"""
import argparse
import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path

import data_lists

if __name__ == "__main__":
    data_lists.install_test_guard()  # before torch and the rest are imported (as in train.py)

import torch  # noqa: E402

import eval_lib  # noqa: E402
import summarize_s2  # noqa: E402
import train  # noqa: E402

ROOT = data_lists.ROOT
OUT_REL = "results/s4"
SCRIPT_REL = "scripts/select_thresholds.py"
OUT_FILES = ("thresholds.csv", "thresholds.json")
REPRESENTATIVE = "roberta"  # README 5.1 EXP-04
SOURCES = (("B", "A", "s2"), ("D", "C", "s4"))  # tuned condition, condition of its run, stage of that run
CAP_CONDITION = "A"  # the cap of both B and D comes from the same seed's A run
DEFAULT_THRESHOLD = 0.5
K_MIN, K_MAX, K_PER_LOGIT = -160, 160, 20  # t_k = k / 20: -8.00..8.00 in steps of 0.05
CAP_MARGIN = Fraction(1, 100)  # README 5.3: +1 percentage point
INPUT_FILES = summarize_s2.INPUT_FILES  # run.json, metrics.json, predictions_validation.csv
COUNT_KEYS = ("n", "n_pos", "n_neg", "tn", "fp", "fn", "tp")
METRIC_KEYS = ("tp", "fp", "fn", "tn", "recall", "fpr", "f1")
COLUMNS = (("seed", "condition", "run_id", "selected_step", "n_clean", "n_pos", "n_neg", "cap_exact", "cap",
            "cap_met", "k", "logit", "threshold", "k_lo", "k_hi") + METRIC_KEYS
           + tuple(f"base_{k}" for k in METRIC_KEYS))


# ---------------------------------------------------------------- candidates and the decision rule

def f32(x):
    """The float32 value nearest to x, as a Python float."""
    return torch.tensor(x, dtype=torch.float32).item()


def candidates():
    """[(k, threshold)] for k = K_MIN..K_MAX; threshold = float32(1 / (1 + exp(-k/20))) as a Python float."""
    return [(k, f32(1.0 / (1.0 + math.exp(-(k / K_PER_LOGIT))))) for k in range(K_MIN, K_MAX + 1)]


def decide(scores, threshold):
    """0/1 tensor: float32(score) >= threshold (eval_lib.predictions on a float32 tensor)."""
    return eval_lib.predictions(torch.as_tensor(scores, dtype=torch.float32), threshold)


def candidate_counts(scores, labels):
    """[(TP, FP)] of every candidate, in k order."""
    s = torch.as_tensor(scores, dtype=torch.float32)  # converted once; decide() keeps a float32 tensor as it is
    y = torch.as_tensor(labels, dtype=torch.int64)
    out = []
    for _, thr in candidates():
        p = decide(s, thr)
        out.append((int((p * y).sum()), int((p * (1 - y)).sum())))
    return out


def fpr_cap(fp_a, n_neg):
    """Exact cap: FPR of the A run at threshold 0.5 (fp_a / n_neg) + 1/100."""
    return Fraction(fp_a, n_neg) + CAP_MARGIN


def select_threshold(scores, labels, cap):
    """README 5.3 selection on one run's validation_clean scores and labels (0/1); cap is a Fraction.

    Returns {"k", "logit", "threshold", "k_lo", "k_hi", "cap_met", "tp", "fp"}.
    """
    labels = [int(y) for y in labels]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        raise ValueError("selection needs both labels (recall and FPR defined)")
    cand, counts = candidates(), candidate_counts(scores, labels)
    idx = range(len(cand))
    ok = [i for i in idx if Fraction(counts[i][1], n_neg) <= cap]
    if ok:  # highest recall (TP / n_pos), then lowest FPR (FP / n_neg)
        best = max(ok, key=lambda i: (counts[i][0], -counts[i][1]))
    else:  # lowest FPR, then highest recall
        best = min(idx, key=lambda i: (counts[i][1], -counts[i][0]))
    same = [i for i in idx if counts[i] == counts[best]]
    lo, hi = same[0], same[-1]
    if same != list(range(lo, hi + 1)):
        raise RuntimeError("candidates with the same (TP, FP) are not contiguous in k")
    k, thr = cand[(lo + hi) // 2]
    return {"k": k, "logit": k / K_PER_LOGIT, "threshold": thr, "k_lo": cand[lo][0], "k_hi": cand[hi][0],
            "cap_met": bool(ok), "tp": counts[best][0], "fp": counts[best][1]}


def metrics_at(scores, labels, threshold):
    """eval_lib.metrics of the decisions at one threshold (no ROC-AUC)."""
    return eval_lib.metrics(labels, decide(scores, threshold).tolist())


def _frac(v):
    return f"{v.numerator}/{v.denominator}"


def threshold_row(seed, condition, rec, a_rec):
    """The output record of one tuned condition: rec is its run, a_rec the same seed's A run (the cap)."""
    cap = fpr_cap(a_rec["base"]["fp"], a_rec["base"]["n_neg"])
    sel = select_threshold(rec["scores"], rec["labels"], cap)
    chosen = metrics_at(rec["scores"], rec["labels"], sel["threshold"])
    if (chosen["tp"], chosen["fp"]) != (sel["tp"], sel["fp"]):
        raise RuntimeError("metrics at the chosen threshold differ from the candidate table")
    base = rec["base"]
    return {"seed": seed, "condition": condition, "source_condition": rec["condition"], "run_id": rec["run_id"],
            "selected_step": rec["selected_step"], "checkpoint_sha256": rec["checkpoint_sha256"],
            "n_clean": base["n"], "n_pos": base["n_pos"], "n_neg": base["n_neg"],
            "cap": cap, "cap_run_id": a_rec["run_id"], "cap_fp": a_rec["base"]["fp"],
            "cap_n_neg": a_rec["base"]["n_neg"], "cap_met": sel["cap_met"],
            "k": sel["k"], "logit": sel["logit"], "threshold": sel["threshold"], "k_lo": sel["k_lo"],
            "k_hi": sel["k_hi"], "chosen": chosen, "base": base}


# ---------------------------------------------------------------- reading one run

def read_predictions(raw, steps, condition):
    """({"ids", "labels", "in_clean", "score": {step: float32 tensor}}, errors) from the CSV bytes."""
    try:
        lines = raw.decode("ascii").split("\n")
    except UnicodeDecodeError:
        return None, ["T05 not ASCII"]
    if lines[0] != train.PRED_HEADER or lines[-1] != "":
        return None, ["T05 header or final LF differs"]
    by = {s: [] for s in steps}
    for n, ln in enumerate(lines[1:-1], start=2):
        try:
            step, sid, y, c, m, text, p, thr, cond = ln.split(",")
            step, y, c, p, m, sc, thr = int(step), int(y), int(c), int(p), float(m), float(text), float(thr)
        except ValueError:
            return None, [f"T05 line {n} is malformed"]
        if (step not in by or y not in (0, 1) or c not in (0, 1) or p not in (0, 1) or cond != condition
                or not 0.0 <= sc <= 1.0):
            return None, [f"T05 line {n}: step not an eval step, label/in_clean/prediction not 0/1, condition not "
                          f"{condition}, or score not in [0, 1]"]
        if thr != DEFAULT_THRESHOLD:
            return None, [f"T06 line {n}: threshold is not {DEFAULT_THRESHOLD}"]
        by[step].append((sid, y, c, p, text, sc, n))
    first = [r[:3] for r in by[steps[0]]]
    ids = [r[0] for r in first]
    if not ids or any(not b > a for a, b in zip(ids, ids[1:])):
        return None, [f"T05 step {steps[0]}: no rows, or sentence IDs not strictly ascending"]
    score = {}
    for s in steps:
        if [r[:3] for r in by[s]] != first:
            return None, [f"T05 step {s}: sentence IDs, labels or in_clean differ from step {steps[0]}"]
        t = torch.tensor([r[5] for r in by[s]], dtype=torch.float32)
        for r, v, q in zip(by[s], t.tolist(), decide(t, DEFAULT_THRESHOLD).tolist()):
            if eval_lib.fmt9(v) != r[4]:
                return None, [f"T05 line {r[6]}: score is not the 9-digit text of a float32 value"]
            if q != r[3]:
                return None, [f"T06 line {r[6]}: stored prediction != (float32(score) >= {DEFAULT_THRESHOLD})"]
        score[s] = t
    return {"ids": ids, "labels": [r[1] for r in first], "in_clean": [r[2] for r in first], "score": score}, []


def check_run(run_id, stage, condition, seed, run, met, raw, errs):
    """Checks of one parsed run (appends to errs); returns the record or None. May raise KeyError/TypeError."""
    tag = f"{run_id}:"
    ident = {"run_id": run_id, "stage": stage, "model_key": REPRESENTATIVE, "condition": condition, "seed": seed}
    bad = [k for k, v in ident.items() if run.get(k) != v]
    if bad:
        errs.append(f"T02 {tag} run.json {', '.join(bad)} do not match the run ID")
    if run.get("status") != "completed":
        errs.append(f"T03 {tag} status is {run.get('status')!r}, not 'completed'")
        return None
    steps, sel, ck = run["budget"]["eval_steps"], run["selection"]["selected_step"], run["checkpoint"]
    if (not isinstance(steps, list) or not steps or any(type(s) is not int for s in steps)
            or len(set(steps)) != len(steps) or type(sel) is not int or sel not in steps):
        errs.append(f"T04 {tag} selected step {sel!r} is not one of the eval steps {steps!r}")
        return None
    if ck.get("kept") is not True or not isinstance(ck.get("sha256"), str) or ck.get("step") != sel:
        errs.append(f"T04 {tag} checkpoint not kept, SHA-256 not recorded, or its step {ck.get('step')!r} != "
                    f"selected step {sel}")
    if (met.get("run_id") != run_id or met.get("threshold") != DEFAULT_THRESHOLD or met.get("eval_steps") != steps
            or [e["step"] for e in met["evals"]] != steps or met.get("selected_step") != sel):
        errs.append(f"T07 {tag} metrics.json run_id, threshold, eval steps or selected step differ from run.json")
        return None
    P, perrs = read_predictions(raw["predictions_validation.csv"], steps, condition)
    if perrs:
        errs += [f"{e[:3]} {tag} predictions_validation.csv: {e[4:]}" for e in perrs]
        return None
    data = run["data"]
    if len(P["ids"]) != data["validation_rows"] or sum(P["in_clean"]) != data["validation_clean_rows"]:
        errs.append(f"T05 {tag} predictions_validation.csv: row counts differ from run.json data")
    keep = [i for i, c in enumerate(P["in_clean"]) if c]
    labels, scores = [P["labels"][i] for i in keep], P["score"][sel][keep]
    base = metrics_at(scores, labels, DEFAULT_THRESHOLD)
    stored = met["evals"][steps.index(sel)]["validation_clean"]
    diff = [k for k in COUNT_KEYS if base[k] != stored.get(k)]
    if diff:
        errs.append(f"T07 {tag} the stored scores do not reproduce metrics.json validation_clean at step {sel} "
                    f"({', '.join(diff)})")
    if not base["n_pos"] or not base["n_neg"]:
        errs.append(f"T08 {tag} validation_clean has {base['n_pos']} label-1 and {base['n_neg']} label-0 rows "
                    "(recall or FPR undefined)")
    if errs:
        return None
    return {"run_id": run_id, "condition": condition, "seed": seed, "selected_step": sel,
            "checkpoint_sha256": ck["sha256"], "labels": labels, "scores": scores, "base": base,
            "rows": list(zip(P["ids"], P["labels"], P["in_clean"])),
            "sha256": {f"runs/{run_id}/{n}": data_lists.sha256_bytes(raw[n]) for n in INPUT_FILES}}


def load_run(root, stage, condition, seed):
    """(record, errors) of runs/<stage>-roberta-<condition>-s<seed>; the record is None when any check fails."""
    run_id = f"{stage}-{REPRESENTATIVE}-{condition}-s{seed}"
    d, tag = Path(root) / "runs" / run_id, f"{run_id}:"
    if train.lock_path(d).exists():
        return None, [f"T01 {tag} runs/{run_id}.lock exists (the run is in progress, or was killed)"]
    if not d.is_dir():
        return None, [f"T01 {tag} runs/{run_id}/ is missing"]
    missing = [n for n in INPUT_FILES if not (d / n).is_file()]
    if missing:
        return None, [f"T01 {tag} missing {', '.join(missing)}"]
    raw = {n: (d / n).read_bytes() for n in INPUT_FILES}
    try:
        run, met = json.loads(raw["run.json"]), json.loads(raw["metrics.json"])
    except ValueError:
        return None, [f"T02 {tag} run.json or metrics.json is not valid JSON"]
    errs = []
    try:
        rec = check_run(run_id, stage, condition, seed, run, met, raw, errs)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        return None, errs + [f"T02 {tag} run.json or metrics.json: missing key or wrong type "
                             f"({type(exc).__name__}: {str(exc)[:80]})"]
    return (None, errs) if errs else (rec, [])


# ---------------------------------------------------------------- selection for all seeds

def compute(root):
    """(errors, result). result holds everything the output files are made of; nothing is written here."""
    root = Path(root)
    try:
        cfg_bytes = (root / train.CONFIG_REL).read_bytes()
        cfg = json.loads(cfg_bytes)
        seeds, thr = cfg["seeds"], cfg["threshold"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"T00 {train.CONFIG_REL}: cannot read seeds and threshold ({type(exc).__name__})"], None
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds)
            or len(set(seeds)) != len(seeds)):
        return [f"T00 {train.CONFIG_REL}: seeds must be a non-empty list of distinct integers; seeds {seeds!r}"], None
    if type(thr) is not float or thr != DEFAULT_THRESHOLD:
        return [f"T00 {train.CONFIG_REL}: threshold is {thr!r}, not {DEFAULT_THRESHOLD} (conditions A and C)"], None
    errs, recs = [], {}
    for seed in seeds:
        for _, cond, stage in SOURCES:
            recs[(seed, cond)], e = load_run(root, stage, cond, seed)
            errs += e
    if errs:
        return errs, None
    ordered = list(recs.values())
    other = [r["run_id"] for r in ordered if r["rows"] != ordered[0]["rows"]]
    if other:
        return [f"T08 validation rows (IDs, labels, in_clean) differ from {ordered[0]['run_id']}: "
                f"{', '.join(other)}"], None
    rows = [threshold_row(seed, tuned, recs[(seed, cond)], recs[(seed, CAP_CONDITION)])
            for seed in seeds for tuned, cond, _ in SOURCES]
    inputs = {train.CONFIG_REL: data_lists.sha256_bytes(cfg_bytes)}
    for r in ordered:
        inputs.update(r["sha256"])
    return [], {"seeds": seeds, "runs": ordered, "rows": rows, "inputs": inputs}


# ---------------------------------------------------------------- output

def csv_row(r):
    row = {"seed": r["seed"], "condition": r["condition"], "run_id": r["run_id"],
           "selected_step": r["selected_step"], "n_clean": r["n_clean"], "n_pos": r["n_pos"], "n_neg": r["n_neg"],
           "cap_exact": _frac(r["cap"]), "cap": float(r["cap"]), "cap_met": int(r["cap_met"]), "k": r["k"],
           "logit": f"{r['logit']:.2f}", "threshold": repr(r["threshold"]), "k_lo": r["k_lo"], "k_hi": r["k_hi"]}
    for k in METRIC_KEYS:
        row[k], row[f"base_{k}"] = r["chosen"][k], r["base"][k]
    return row


def _metrics_json(m):
    out = {k: m[k] for k in ("tp", "fp", "fn", "tn")}
    for k in ("recall", "fpr", "f1", "precision"):
        out[k], out[f"{k}_exact"] = m[k], m[f"{k}_exact"]
    return out


def thresholds_json(res, script_sha):
    rows = []
    for r in res["rows"]:
        rows.append({
            "seed": r["seed"], "condition": r["condition"], "source_condition": r["source_condition"],
            "run_id": r["run_id"], "selected_step": r["selected_step"], "checkpoint_sha256": r["checkpoint_sha256"],
            "n_clean": r["n_clean"], "n_pos": r["n_pos"], "n_neg": r["n_neg"],
            "cap": {"exact": _frac(r["cap"]), "value": float(r["cap"]), "margin": _frac(CAP_MARGIN),
                    "a_run_id": r["cap_run_id"], "a_fp": r["cap_fp"], "a_n_neg": r["cap_n_neg"]},
            "cap_met": r["cap_met"], "k": r["k"], "logit": r["logit"], "threshold": r["threshold"],
            "k_lo": r["k_lo"], "k_hi": r["k_hi"],
            "chosen": _metrics_json(r["chosen"]), "base": _metrics_json(r["base"])})
    doc = {
        "schema": 1,
        "readme": "README 5.3 item 1 (threshold adjustment), IMP-01; written by " + SCRIPT_REL,
        "rules": {
            "data": "validation_clean rows (in_clean 1) of predictions_validation.csv at the run's selected step, "
                    "score column; development data only, no derived cases",
            "candidates": f"logit t = k/{K_PER_LOGIT} for k = {K_MIN}..{K_MAX}; threshold = float32(1 / (1 + "
                          "exp(-t))), stored as the Python float of that float32 value",
            "decision": "attack when float32(score) >= threshold (eval_lib.predictions on a float32 tensor)",
            "cap": f"validation_clean FPR of the same seed's condition-{CAP_CONDITION} run at threshold "
                   f"{DEFAULT_THRESHOLD} + {_frac(CAP_MARGIN)} (exact fractions), for B and D alike",
            "selection": "among candidates with FPR <= cap: highest recall, then lowest FPR; the candidates with "
                         "that (TP, FP) are k_lo..k_hi and index (lo + hi) // 2 is chosen; when none meets the "
                         "cap: lowest FPR, then highest recall, same middle rule, cap_met false",
            "metrics": f"chosen = eval_lib.metrics at the chosen threshold, base = at {DEFAULT_THRESHOLD} on the "
                       "same scores (condition A for B, C for D)",
        },
        "model": REPRESENTATIVE, "seeds": res["seeds"], "default_threshold": DEFAULT_THRESHOLD,
        "candidates": {"k_min": K_MIN, "k_max": K_MAX, "k_per_logit": K_PER_LOGIT, "count": K_MAX - K_MIN + 1},
        "script_sha256_lf": script_sha, "inputs_sha256": res["inputs"],
        "runs": [{"run_id": r["run_id"], "condition": r["condition"], "seed": r["seed"],
                  "selected_step": r["selected_step"], "checkpoint_sha256": r["checkpoint_sha256"],
                  "validation_rows": len(r["rows"]), "validation_clean_rows": r["base"]["n"]} for r in res["runs"]],
        "thresholds": rows,
    }
    return train._dump(doc)


def render(res, script_sha):
    """{file name: text} of OUT_FILES."""
    csv_name, json_name = OUT_FILES
    return {csv_name: summarize_s2.csv_text(COLUMNS, [csv_row(r) for r in res["rows"]]),
            json_name: thresholds_json(res, script_sha)}


def check_outputs(out_dir, files):
    """--check: T09 errors for every file that is missing or differs from the fresh bytes."""
    errs = []
    for n, text in files.items():
        if not (out_dir / n).is_file():
            errs.append(f"T09 --check: {n} is missing")
        elif (out_dir / n).read_bytes() != text.encode("utf-8"):
            errs.append(f"T09 --check: {n} differs from the fresh selection")
    return errs


def write_outputs(out_dir, files):
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, text in files.items():
        tmp = out_dir / (n + ".tmp")
        tmp.write_bytes(text.encode("utf-8"))
        os.replace(tmp, out_dir / n)


# ---------------------------------------------------------------- terminal

def describe(r):
    c, b = r["chosen"], r["base"]
    return (f"seed {r['seed']} {r['condition']} ({r['run_id']} step {r['selected_step']}): logit {r['logit']:.2f} "
            f"(k {r['k']} of {r['k_lo']}..{r['k_hi']}), threshold {r['threshold']!r}; cap {_frac(r['cap'])} "
            f"{'met' if r['cap_met'] else 'NOT MET'}; recall {b['recall_exact']} -> {c['recall_exact']}, "
            f"FPR {b['fpr_exact']} -> {c['fpr_exact']}, F1 {b['f1']:.4f} -> {c['f1']:.4f}")


def parse_args(argv):
    p = argparse.ArgumentParser(description="Select the per-seed thresholds of conditions B and D into results/s4/ "
                                            "(README 5.3).")
    p.add_argument("--root", help="repository root holding runs/ and configs/ (default: this repository)")
    p.add_argument("--out", help=f"output folder (default: <root>/{OUT_REL})")
    p.add_argument("--check", action="store_true", help="compare with the written files instead of writing")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root) if args.root else ROOT
    out = Path(args.out) if args.out else root / OUT_REL
    errors, res = compute(root)
    if not errors:
        files = render(res, data_lists.sha256_lf(ROOT / SCRIPT_REL))
        if args.check:
            errors = check_outputs(out, files)
    if errors:
        for e in errors:
            print(train._ascii("FAIL " + e))
        print(f"refused: {len(errors)} error(s); nothing written")
        return 1
    try:
        shown = out.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        shown = str(out)
    if args.check:
        print(train._ascii(f"OK --check: {', '.join(files)} in {shown}/ equal the fresh selection"))
        return 0
    write_outputs(out, files)
    print(train._ascii(f"OK {len(res['runs'])} runs checked; wrote {shown}/: {', '.join(files)}"))
    for r in res["rows"]:
        print(train._ascii(describe(r)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
