"""Stage-2 result summary: nine runs -> results/s2/ (README 5.1 EXP-01~03, 6.1, 8, TRAIN-04).

Run from Git Bash at the repo root after the nine stage-2 runs have finished:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/summarize_s2.py
Options: --out DIR writes elsewhere (e.g. a temp folder for a dry run); --show-text N
also prints, to the terminal only, up to N model-error sentences per model.

Rules:
  - Inputs: runs/s2-<model>-A-s<seed>/{run.json, metrics.json, predictions_validation.csv}
    for every model in model_specs.MODELS and each of the 3 distinct seeds in
    configs/train.json. Refused (exit 1, nothing written; codes S00-S09) when:
    S00 the config has no 3 distinct integer seeds or no expected_updates;
    S01 a run folder or file is missing, runs/<id>.lock exists, or runs/ holds another
    s2-* folder (so each model has exactly the config's 3 seeds);
    S02 run.json/metrics.json is not JSON, misses keys, or its identity differs from the
    folder name; S03 status is not completed;
    S04 the reload check did not pass, the checkpoint is not kept, its path is not
    runs/<id>/checkpoint/model.safetensors, the file is missing, or its size or
    SHA-256 (streamed, model_specs.sha256_file) differs from run.json, or its step is
    not the selected step;
    S05 metrics.json run ID, threshold 0.5, eval steps or selected step differ from
    run.json, or the selected step does not follow the selection rule (eval_lib.select);
    S06 predictions_validation.csv is malformed, inconsistent (threshold, condition A,
    prediction = score >= threshold, same rows at every step, IDs ascending), has
    other row counts than run.json, a non-finite margin, or does not reproduce
    metrics.json at every eval step (eval_lib.metrics);
    S07 resources (update times, eval times, peak memory) or attempt not recorded;
    S08 the code, config, list, environment (train.RERUN_SAME), lists_meta, review or
    budget records differ between runs, a model's record differs between its seeds,
    the config SHA-256 differs from configs/train.json, or the validation rows (IDs,
    labels, in_clean) differ between runs;
    S09 a stage-2 precondition is not recorded: official lists with the git-tracked
    check, code.git_dirty empty (named code/config/list files clean; the other-path
    count may be nonzero because it also counts untracked files), updates_run = N =
    expected_updates, eval steps ceil(N/3), ceil(2N/3), N, and the model key, repo,
    revision, weights SHA-256 and tokenizer fingerprint equal to model_specs.
  - Metrics (README 6.1): eval_lib.metrics at each run's selected step (4.2
    checkpoint selection on validation_clean), threshold 0.5, for validation_clean
    (in_clean rows) and validation (all rows). N/A stays None (empty cell).
  - Resources: update_seconds_total and update_seconds_median from run.json (6.1
    training time); eval_seconds_mean = mean of run.json resources.eval_s, the time of
    one validation prediction pass (all rows, already tokenized, FP32, eval batch of the
    config), not the 6.1 inference-time measurement; peak allocated/reserved bytes.
  - Seeds (TRAIN-04): mean and sample standard deviation (statistics.stdev, n - 1)
    over the 3 seeds; None when any seed value is N/A (README 6.1).
  - Errors (EXP-02/03), validation_clean only: a run gets a sentence wrong when its
    prediction at the selected step differs from the label; FP label 0, FN label 1.
    Model error: wrong in >= 2 of the model's 3 seeds. shared_by lists the models
    whose model error it is; common = all three, individual = one model only.
  - Outputs (results/s2/): runs.csv, models.csv, errors.csv, summary.json,
    summary.md. Sentence IDs and numbers only, never text; LF; rows in model, seed
    and sentence-ID order; non-integers with 6 decimals (summary.md: rounded for
    reading); no timestamps or absolute paths, so the same inputs give the same bytes.
  - --show-text N: after writing, joins IDs to the validation text with
    data_lists.load_rows() (test is never read; the test.csv guard is installed first)
    and prints up to N model-error sentences per model (control characters escaped).
    Never written to a file.
"""
import argparse
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

import data_lists

if __name__ == "__main__":
    data_lists.install_test_guard()  # before torch, sklearn and the rest are imported (as in train.py)

import eval_lib  # noqa: E402
import model_specs  # noqa: E402
import train  # noqa: E402

ROOT = data_lists.ROOT
OUT_REL = "results/s2"
SCRIPT_REL = "scripts/summarize_s2.py"
CONDITION = "A"
THRESHOLD = 0.5
N_SEEDS = 3
MIN_SEEDS_WRONG = 2  # README 5.1 EXP-03: wrong in >= 2 of the 3 seeds
AUC_TOL = 1e-9  # recomputed ROC-AUC vs metrics.json (9-digit margins keep the float32 order)
DP = 6
INPUT_FILES = ("run.json", "metrics.json", "predictions_validation.csv")
SUBSETS = ("validation_clean", "validation")
METRICS = ("f1", "precision", "recall", "fpr", "roc_auc", "tp", "fp", "fn", "tn")
EXACT_KEYS = ("n", "n_pos", "n_neg", "tn", "fp", "fn", "tp", "precision_exact", "recall_exact", "fpr_exact",
              "f1_exact")
RESOURCES = ("update_seconds_total", "update_seconds_median", "eval_seconds_mean", "peak_memory_allocated_bytes",
             "peak_memory_reserved_bytes")
RUN_COLUMNS = (("model", "seed", "run_id", "selected_step")
               + tuple(f"{s}_{k}" for s in SUBSETS for k in ("n",) + METRICS) + RESOURCES
               + ("attempt", "checkpoint_sha256"))
QUANTITIES = ("selected_step",) + tuple(f"{s}_{k}" for s in SUBSETS for k in METRICS) + RESOURCES
# Records that must be equal in all nine runs (train.RERUN_SAME: code, config, lists, environment).
SAME = train.RERUN_SAME + (("lists_meta SHA-256", ("lists", "lists_meta_sha256")),
                           ("train_review SHA-256", ("lists", "review_sha256")), ("budget", ("budget",)))
NAMES = {"bert": "BERT", "roberta": "RoBERTa", "deberta": "DeBERTa-v3"}


# ---------------------------------------------------------------- reading one run

def read_predictions(raw, steps):
    """({"ids", "labels", "in_clean", "pred": {step: []}, "margin": {step: []}}, errors) from the CSV bytes."""
    try:
        lines = raw.decode("ascii").split("\n")
    except UnicodeDecodeError:
        return None, ["not ASCII"]
    if lines[0] != train.PRED_HEADER or lines[-1] != "":
        return None, ["header or final LF differs"]
    by = {s: [] for s in steps}
    for n, ln in enumerate(lines[1:-1], start=2):
        try:
            step, sid, y, c, m, sc, p, thr, cond = ln.split(",")
            step, y, c, p, m, sc, thr = int(step), int(y), int(c), int(p), float(m), float(sc), float(thr)
        except ValueError:
            return None, [f"line {n} is malformed"]
        if (step not in by or y not in (0, 1) or c not in (0, 1) or thr != THRESHOLD or cond != CONDITION
                or p != int(sc >= thr)):
            return None, [f"line {n}: step not an eval step, label/in_clean not 0/1, threshold not {THRESHOLD}, "
                          f"condition not {CONDITION}, or prediction != (score >= threshold)"]
        by[step].append((sid, y, c, p, m))
    first = [r[:3] for r in by[steps[0]]]
    ids = [r[0] for r in first]
    if not ids or any(not b > a for a, b in zip(ids, ids[1:])):
        return None, [f"step {steps[0]}: no rows, or sentence IDs not strictly ascending"]
    for s in steps[1:]:
        if [r[:3] for r in by[s]] != first:
            return None, [f"step {s}: sentence IDs, labels or in_clean differ from step {steps[0]}"]
    return {"ids": ids, "labels": [r[1] for r in first], "in_clean": [r[2] for r in first],
            "pred": {s: [r[3] for r in by[s]] for s in steps},
            "margin": {s: [r[4] for r in by[s]] for s in steps}}, []


def metrics_at(P, step):
    y, p, m = P["labels"], P["pred"][step], P["margin"][step]
    k = [i for i, c in enumerate(P["in_clean"]) if c]
    return {"validation": eval_lib.metrics(y, p, m),
            "validation_clean": eval_lib.metrics([y[i] for i in k], [p[i] for i in k], [m[i] for i in k])}


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_run(d, run_id, model, seed, run, met, raw, cfg, errs):
    """Checks of one parsed run (appends to errs); returns the record or None. May raise KeyError/TypeError."""
    tag = f"{run_id}:"
    ident = {"run_id": run_id, "stage": "s2", "model_key": model, "condition": CONDITION, "seed": seed}
    bad = [k for k, v in ident.items() if run.get(k) != v]
    if bad:
        errs.append(f"S02 {tag} run.json {', '.join(bad)} do not match the run ID")
    if run.get("status") != "completed":
        errs.append(f"S03 {tag} status is {run.get('status')!r}, not 'completed'")
        return None
    budget, lists, mrec = run["budget"], run["lists"], run["model"]
    sel, steps, n_total = run["selection"]["selected_step"], budget["eval_steps"], budget["N"]
    if lists.get("mode") != "official" or lists.get("git_tracked_checked") is not True:
        errs.append(f"S09 {tag} lists are not official with the git-tracked check (mode {lists.get('mode')!r}, "
                    f"git_tracked_checked {lists.get('git_tracked_checked')!r})")
    if run["code"].get("git_dirty") != []:
        errs.append(f"S09 {tag} code.git_dirty is not empty (code, config or list files not clean at the start)")
    if budget.get("updates_run") != n_total or n_total != cfg["expected_updates"] or steps != train.eval_steps(n_total):
        errs.append(f"S09 {tag} budget: updates_run {budget.get('updates_run')}, N {n_total} (config expected_updates "
                    f"{cfg['expected_updates']}), eval steps {steps} (want ceil(N/3), ceil(2N/3), N)")
    pins = (model, *model_specs.MODELS[model], model_specs.WEIGHTS[model][1], model_specs.TOKENIZER_SHA256[model])
    if (mrec.get("key"), mrec.get("repo"), mrec.get("revision"), (mrec.get("weights") or {}).get("sha256"),
            mrec.get("tokenizer_fingerprint")) != pins:
        errs.append(f"S09 {tag} model record differs from model_specs (key, repo, revision, weights sha256, "
                    "tokenizer fingerprint)")
    ck = run["checkpoint"]
    rel = f"runs/{run_id}/checkpoint/{model_specs.CHECKPOINT_FILE}"
    if (ck.get("reload_check") or {}).get("passed") is not True:
        errs.append(f"S04 {tag} the checkpoint reload check did not pass")
    if (ck.get("kept") is not True or ck.get("path") != rel or not isinstance(ck.get("sha256"), str)
            or not isinstance(ck.get("size"), int)):
        errs.append(f"S04 {tag} checkpoint not kept, path not {rel}, or sha256/size not recorded")
    else:
        f = d / "checkpoint" / model_specs.CHECKPOINT_FILE
        if not f.is_file() or f.stat().st_size != ck["size"]:
            errs.append(f"S04 {tag} {rel} is missing or its size differs from run.json")
        elif model_specs.sha256_file(f) != ck["sha256"]:
            errs.append(f"S04 {tag} {rel} SHA-256 differs from run.json")
    if ck.get("step") != sel:
        errs.append(f"S04 {tag} checkpoint step {ck.get('step')} != selected step {sel}")
    if met.get("run_id") != run_id or met.get("threshold") != THRESHOLD:
        errs.append(f"S05 {tag} metrics.json run_id or threshold ({met.get('threshold')}) differs")
    if met.get("eval_steps") != steps or [e["step"] for e in met["evals"]] != steps:
        errs.append(f"S05 {tag} metrics.json eval steps differ from run.json budget {steps}")
        return None
    if met.get("selected_step") != sel:
        errs.append(f"S05 {tag} metrics.json selected_step {met.get('selected_step')} != run.json {sel}")
    else:
        try:
            if eval_lib.select(met["evals"])["step"] != sel:
                errs.append(f"S05 {tag} selected step {sel} does not follow the selection rule (eval_lib.select)")
        except ValueError as exc:
            errs.append(f"S05 {tag} selection key undefined in metrics.json ({exc})")
    P, perrs = read_predictions(raw["predictions_validation.csv"], steps)
    if perrs:
        errs += [f"S06 {tag} predictions_validation.csv: {e}" for e in perrs]
        return None
    data = run["data"]
    if len(P["ids"]) != data["validation_rows"] or sum(P["in_clean"]) != data["validation_clean_rows"]:
        errs.append(f"S06 {tag} prediction row counts differ from run.json data")
    computed = {}
    for e in met["evals"]:
        try:
            c = computed[e["step"]] = metrics_at(P, e["step"])
        except eval_lib.NonFiniteError as exc:
            errs.append(f"S06 {tag} step {e['step']}: {exc}")
            return None
        except ValueError as exc:  # sklearn input validation
            errs.append(f"S06 {tag} step {e['step']}: metrics failed ({type(exc).__name__}: {exc})")
            return None
        for s in SUBSETS:
            diff = [k for k in EXACT_KEYS if c[s][k] != e[s].get(k)]
            a, b = c[s]["roc_auc"], e[s].get("roc_auc")
            if (a is None) != (b is None) or (a is not None and abs(a - b) > AUC_TOL):
                diff.append("roc_auc")
            if diff:
                errs.append(f"S06 {tag} step {e['step']} {s}: predictions do not reproduce metrics.json "
                            f"({', '.join(diff)})")
    res, eval_s = run["resources"], run["resources"].get("eval_s")
    resources = {"update_seconds_total": res.get("update_seconds_total"),
                 "update_seconds_median": res.get("update_seconds_median"),
                 "eval_seconds_mean": statistics.fmean(eval_s) if isinstance(eval_s, list) and len(eval_s) == len(
                     steps) and all(_number(v) for v in eval_s) else None,
                 "peak_memory_allocated_bytes": res.get("peak_memory_allocated_bytes"),
                 "peak_memory_reserved_bytes": res.get("peak_memory_reserved_bytes")}
    if not all(_number(v) for v in resources.values()) or not isinstance(run.get("attempt"), int):
        errs.append(f"S07 {tag} resources (update times, {len(steps)} eval times, peak memory) or attempt not "
                    "recorded as numbers")
    if errs:
        return None
    pred = P["pred"][sel]
    return {"model": model, "seed": seed, "run_id": run_id, "run": run, "selected_step": sel,
            "metrics": computed[sel], "rows": list(zip(P["ids"], P["labels"], P["in_clean"])),
            "wrong": {sid for sid, y, c, p in zip(P["ids"], P["labels"], P["in_clean"], pred) if c and p != y},
            "resources": resources, "attempt": run["attempt"], "checkpoint_sha256": ck["sha256"],
            "saved_steps": run["selection"].get("saved_steps"),
            "sha256": {f"runs/{run_id}/{n}": data_lists.sha256_bytes(raw[n]) for n in INPUT_FILES}}


def load_run(root, model, seed, cfg):
    """(record, errors) of one run; the record is None when any check fails."""
    run_id = f"s2-{model}-{CONDITION}-s{seed}"
    d, tag = root / "runs" / run_id, f"{run_id}:"
    if train.lock_path(d).exists():
        return None, [f"S01 {tag} runs/{run_id}.lock exists (the run is in progress, or was killed)"]
    if not d.is_dir():
        return None, [f"S01 {tag} runs/{run_id}/ is missing"]
    missing = [n for n in INPUT_FILES if not (d / n).is_file()]
    if missing:
        return None, [f"S01 {tag} missing {', '.join(missing)}"]
    raw = {n: (d / n).read_bytes() for n in INPUT_FILES}
    try:
        run, met = json.loads(raw["run.json"]), json.loads(raw["metrics.json"])
    except ValueError:
        return None, [f"S02 {tag} run.json or metrics.json is not valid JSON"]
    errs = []
    try:
        rec = check_run(d, run_id, model, seed, run, met, raw, cfg, errs)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        return None, errs + [f"S02 {tag} run.json or metrics.json: missing key or wrong type "
                             f"({type(exc).__name__}: {str(exc)[:80]})"]
    return (None, errs) if errs else (rec, [])


def check_same(recs, cfg_sha):
    errs = []
    for name, path in SAME:
        groups = {}
        for r in recs:
            groups.setdefault(json.dumps(train._get(r["run"], path), sort_keys=True), []).append(r["run_id"])
        if len(groups) > 1 or "null" in groups:
            errs.append(f"S08 {name} differs between runs or is not recorded: "
                        + " | ".join(", ".join(ids) for ids in groups.values()))
    for m in dict.fromkeys(r["model"] for r in recs):
        if len({json.dumps(r["run"]["model"], sort_keys=True) for r in recs if r["model"] == m}) > 1:
            errs.append(f"S08 model record of {m} differs between its seeds")
    other = [r["run_id"] for r in recs if r["run"]["config"]["sha256"] != cfg_sha]
    if other:
        errs.append(f"S08 config SHA-256 differs from the current configs/train.json: {', '.join(other)}")
    other = [r["run_id"] for r in recs if r["rows"] != recs[0]["rows"]]
    if other:
        errs.append(f"S08 validation rows (IDs, labels, in_clean) differ from {recs[0]['run_id']}: {', '.join(other)}")
    return errs


# ---------------------------------------------------------------- aggregation

def run_row(r):
    row = {"model": r["model"], "seed": r["seed"], "run_id": r["run_id"], "selected_step": r["selected_step"]}
    for s in SUBSETS:
        for k in ("n",) + METRICS:
            row[f"{s}_{k}"] = r["metrics"][s][k]
    row.update(r["resources"], attempt=r["attempt"], checkpoint_sha256=r["checkpoint_sha256"])
    return row


def model_stats(rows):
    """{quantity: {"mean", "std", "values"}} over one model's seed rows (std: n - 1; None if any N/A)."""
    out = {}
    for q in QUANTITIES:
        vals = [r[q] for r in rows]
        ok = all(v is not None for v in vals)
        out[q] = {"mean": statistics.fmean(vals) if ok else None, "std": statistics.stdev(vals) if ok else None,
                  "values": vals}
    return out


def error_rows(recs, models):
    label = {sid: y for sid, y, c in recs[0]["rows"] if c}
    count = {m: {} for m in models}
    for r in recs:
        for sid in r["wrong"]:
            count[r["model"]][sid] = count[r["model"]].get(sid, 0) + 1
    rows = []
    for sid in sorted(set().union(*count.values())):
        row = {"sentence_id": sid, "label": label[sid], "error_type": "FP" if label[sid] == 0 else "FN"}
        row.update({f"{m}_seeds_wrong": count[m].get(sid, 0) for m in models})
        shared = [m for m in models if count[m].get(sid, 0) >= MIN_SEEDS_WRONG]
        row.update({f"{m}_model_error": int(m in shared) for m in models})
        row.update(n_models_error=len(shared), shared_by=";".join(shared) or "none")
        rows.append(row)
    return rows


def fpfn(rows):
    fp = sum(r["error_type"] == "FP" for r in rows)
    return {"total": len(rows), "fp": fp, "fn": len(rows) - fp}


def error_summary(rows, models, n_seeds):
    return {
        "per_model": {m: {
            "model_error": fpfn([r for r in rows if r[f"{m}_model_error"]]),
            "only_this_model": fpfn([r for r in rows if r["shared_by"] == m]),
            "by_seeds_wrong": {str(k): fpfn([r for r in rows if r[f"{m}_seeds_wrong"] == k])
                               for k in range(1, n_seeds + 1)}} for m in models},
        "all_models": fpfn([r for r in rows if r["n_models_error"] == len(models)]),
        "exactly_two": {";".join(p): fpfn([r for r in rows if r["shared_by"] == ";".join(p)])
                        for p in itertools.combinations(models, 2)},
        "any_model": fpfn([r for r in rows if r["n_models_error"] >= 1]),
        "any_run": fpfn(rows),
    }


def summarize(root):
    """(errors, result). result holds everything the output files are made of."""
    root = Path(root)
    try:
        cfg_bytes = (root / train.CONFIG_REL).read_bytes()
        cfg = json.loads(cfg_bytes)
        seeds, expected_updates = cfg["seeds"], cfg["expected_updates"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"S00 {train.CONFIG_REL}: cannot read seeds and expected_updates ({type(exc).__name__})"], None
    if not (isinstance(seeds, list) and len(seeds) == N_SEEDS and len(set(seeds)) == N_SEEDS
            and all(type(s) is int for s in seeds) and type(expected_updates) is int):
        return [f"S00 {train.CONFIG_REL}: seeds must be {N_SEEDS} distinct integers (EXP-03: {MIN_SEEDS_WRONG} of "
                f"{N_SEEDS}) and expected_updates an integer; seeds {seeds!r}"], None
    models = list(model_specs.MODELS)
    expected = {f"s2-{m}-{CONDITION}-s{s}" for m in models for s in seeds}
    runs_dir = root / "runs"
    extra = sorted(p.name for p in runs_dir.glob("s2-*") if p.is_dir() and p.name not in expected) \
        if runs_dir.is_dir() else []
    errs = [f"S01 unexpected stage-2 run folders (model or seed not in model_specs / config): {', '.join(extra)}"] \
        if extra else []
    recs = []
    for m in models:
        for s in seeds:
            r, e = load_run(root, m, s, cfg)
            errs += e
            if r:
                recs.append(r)
    if errs:
        return errs, None
    cfg_sha = data_lists.sha256_bytes(cfg_bytes)
    errs = check_same(recs, cfg_sha)
    if errs:
        return errs, None
    rows = [run_row(r) for r in recs]
    erows = error_rows(recs, models)
    run0 = recs[0]["run"]
    inputs = {train.CONFIG_REL: cfg_sha}
    for r in recs:
        inputs.update(r["sha256"])
    return [], {
        "models": models, "seeds": seeds, "recs": recs, "rows": rows,
        "stats": {m: model_stats([r for r in rows if r["model"] == m]) for m in models},
        "error_rows": erows, "errors": error_summary(erows, models, len(seeds)), "inputs": inputs,
        "records": {"code_files_sha256_lf": run0["code"]["files_sha256_lf"], "config_sha256": cfg_sha,
                    "lists_files_sha256": run0["lists"]["files_sha256"],
                    "lists_meta_sha256": run0["lists"]["lists_meta_sha256"],
                    "review_sha256": run0["lists"]["review_sha256"], "budget": run0["budget"],
                    "env": {"requirements_sha256": run0["env"]["requirements_sha256"], "torch": run0["env"]["torch"],
                            "cuda": run0["env"]["cuda"], "gpu": run0["env"]["gpu"]["name"]},
                    "git_commits": sorted({str(r["run"]["code"].get("git_commit")) for r in recs})},
    }


# ---------------------------------------------------------------- output

def cell(v):
    if v is None:
        return ""
    return f"{v:.{DP}f}" if isinstance(v, float) else str(v)


def jnum(v):
    return round(v, DP) if isinstance(v, float) else v


def csv_text(columns, rows):
    return "".join(",".join(cell(r[c]) for c in columns) + "\n" for r in [dict(zip(columns, columns))] + rows)


def error_columns(models):
    return (("sentence_id", "label", "error_type") + tuple(f"{m}_seeds_wrong" for m in models)
            + tuple(f"{m}_model_error" for m in models) + ("n_models_error", "shared_by"))


def models_csv(res):
    cols = ("model", "quantity", "mean", "std") + tuple(f"s{s}" for s in res["seeds"])
    rows = []
    for m in res["models"]:
        for q in QUANTITIES:
            st = res["stats"][m][q]
            rows.append({"model": m, "quantity": q, "mean": st["mean"], "std": st["std"],
                         **{f"s{s}": v for s, v in zip(res["seeds"], st["values"])}})
    return csv_text(cols, rows)


def summary_json(res, script_sha):
    runs = []
    for r, row in zip(res["recs"], res["rows"]):
        runs.append({**{c: jnum(row[c]) for c in RUN_COLUMNS}, "saved_steps": r["saved_steps"],
                     "exact": {s: {k: r["metrics"][s][f"{k}_exact"] for k in ("f1", "precision", "recall", "fpr")}
                               for s in SUBSETS}})
    stats = {m: {q: {"mean": jnum(st["mean"]), "std": jnum(st["std"]), "values": [jnum(v) for v in st["values"]]}
                 for q, st in res["stats"][m].items()} for m in res["models"]}
    doc = {
        "schema": 1,
        "readme": "README 5.1 EXP-01~03, 6.1, 8 (stage 2), TRAIN-04; written by " + SCRIPT_REL,
        "rules": {
            "metrics": "eval_lib.metrics at each run's selected step (validation_clean F1 max, then lower FPR, "
                       "then earlier step), threshold 0.5; ROC-AUC on the margin l1 - l0; N/A is null",
            "seeds": "mean and sample standard deviation (n - 1) over the seeds; null when any seed is N/A",
            "resources": "update_seconds_total/median: run.json training time (README 6.1); eval_seconds_mean: mean "
                         "of run.json resources.eval_s, one validation prediction pass (all rows, tokenized, FP32, "
                         "config eval batch), not the README 6.1 inference-time measurement; peak memory: run.json",
            "checkpoint": "checkpoint file size and SHA-256 verified against run.json; reload check passed",
            "errors": "validation_clean rows with prediction != label at the selected step; FP label 0, FN label 1; "
                      f"model error = wrong in >= {MIN_SEEDS_WRONG} of the {len(res['seeds'])} seeds of a model",
            "numbers": f"non-integers rounded to {DP} decimals; exact fractions under runs[].exact",
        },
        "models": res["models"], "seeds": res["seeds"], "threshold": THRESHOLD, "condition": CONDITION,
        "script_sha256_lf": script_sha, "inputs_sha256": res["inputs"], "records": res["records"],
        "runs": runs, "models_summary": stats, "errors": res["errors"],
    }
    return train._dump(doc)


def _f(v, dp, scale=1):
    return "N/A" if v is None else f"{v / scale:.{dp}f}"


def _pm(st, dp, scale=1):
    if st["mean"] is None:
        return "seed별 " + " / ".join(_f(v, dp, scale) for v in st["values"])
    return f"{st['mean'] / scale:.{dp}f} ± {st['std'] / scale:.{dp}f}"


def _c(d):
    return f"{d['total']} ({d['fp']} / {d['fn']})"


def summary_md(res):
    models, seeds, stats, E = res["models"], res["seeds"], res["stats"], res["errors"]
    name = {m: NAMES.get(m, m) for m in models}
    n = {s: res["rows"][0][f"{s}_n"] for s in SUBSETS}
    budget = res["records"]["budget"]
    seed_txt = "·".join(str(s) for s in seeds)
    out = [
        "# 2단계 결과 요약", "",
        f"`{SCRIPT_REL}`가 `runs/s2-<모델>-{CONDITION}-s<seed>/`의 {len(res['rows'])}개 실행 기록에서 만든다. "
        f"validation만 쓴다(`validation_clean` {n['validation_clean']:,}행, validation {n['validation']:,}행). "
        f"각 실행의 선택 step(`validation_clean` F1 최대 → FPR 낮음 → 이른 step)에서 임계값 {THRESHOLD}로 판정한다. "
        f"평균 ± 표준편차는 seed {seed_txt}의 표본 표준편차(n−1)다. 표의 수는 반올림했고 전체 값은 "
        "`runs.csv`·`models.csv`·`summary.json`에 있다.", "",
        f"모든 실행의 학습 예산은 N = {budget['N']:,} update, 평가 시점은 "
        f"{' / '.join(f'{s:,}' for s in budget['eval_steps'])}이다(TRAIN-04).", "",
        "## 모델별 성능 (EXP-01, seed 평균 ± 표준편차)", "",
        "| 모델 | 평가셋 | F1 | Precision | Recall | FPR | ROC-AUC | FP | FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in models:
        for s in SUBSETS:
            out.append(f"| {name[m]} | `{s}` | " + " | ".join(
                _pm(stats[m][f"{s}_{k}"], 4) for k in ("f1", "precision", "recall", "fpr", "roc_auc"))
                + f" | {_pm(stats[m][f'{s}_fp'], 1)} | {_pm(stats[m][f'{s}_fn'], 1)} |")
    out += ["", "## 자원과 선택 step (EXP-01, TRAIN-04)", "",
            f"| 모델 | 선택 step (seed {' / '.join(str(s) for s in seeds)}) | 학습 시간(초, update 합계) | "
            "update당 시간 중앙값(초) | validation 평가 1회(초) | 최대 할당 메모리(GB) | 최대 예약 메모리(GB) | 시도 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for m in models:
        st = stats[m]
        att = [r["attempt"] for r in res["rows"] if r["model"] == m]
        out.append(f"| {name[m]} | {' / '.join(str(v) for v in st['selected_step']['values'])} | "
                   f"{_pm(st['update_seconds_total'], 1)} | {_pm(st['update_seconds_median'], 4)} | "
                   f"{_pm(st['eval_seconds_mean'], 2)} | {_pm(st['peak_memory_allocated_bytes'], 3, 1e9)} | "
                   f"{_pm(st['peak_memory_reserved_bytes'], 3, 1e9)} | {' / '.join(str(a) for a in att)} |")
    out += ["", "- 학습 시간: update 1회의 벽시계 시간(6.1)의 실행별 합계와 중앙값.",
            f"- validation 평가 1회: 평가 시점 {len(budget['eval_steps'])}번의 평균. 토큰화된 validation "
            f"{n['validation']:,}행을 FP32·평가 배치로 예측하는 시간이며(토큰화·지표 계산 제외, 워밍업 구분 없음) "
            "6.1의 추론 시간 측정이 아니다.",
            "- GB = 10⁹바이트. 시도는 seed별 `run.json`의 attempt(1이면 재실행 없음).", "",
            "## 실행별 결과 (`validation_clean`)", "",
            "| 실행 ID | 선택 step | F1 | Precision | Recall | FPR | ROC-AUC | FP | FN | validation F1 | 학습 시간(초) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in res["rows"]:
        out.append(f"| `{r['run_id']}` | {r['selected_step']} | " + " | ".join(
            _f(r[f"validation_clean_{k}"], 4) for k in ("f1", "precision", "recall", "fpr", "roc_auc"))
            + f" | {r['validation_clean_fp']} | {r['validation_clean_fn']} | {_f(r['validation_f1'], 4)} | "
            f"{_f(r['update_seconds_total'], 1)} |")
    out += ["", "## 오류 사례 (EXP-02·03, `validation_clean`)", "",
            f"모델 오류는 그 모델의 seed {len(seeds)}개 중 {MIN_SEEDS_WRONG}개 이상에서 틀린 문장이다. "
            "FP는 정상(0)을 인젝션으로, FN은 인젝션(1)을 정상으로 판정한 경우다. "
            "문장 ID 목록은 `errors.csv`(문장 없음). 칸은 문장 수 (FP / FN)이다.", "",
            "| 모델 | 모델 오류 | 이 모델만 | " + " | ".join(f"{k}개 seed에서 틀림" for k in range(1, len(seeds) + 1))
            + " |", "| --- | ---: | ---: | " + " | ".join("---:" for _ in seeds) + " |"]
    for m in models:
        pm_ = E["per_model"][m]
        out.append(f"| {name[m]} | {_c(pm_['model_error'])} | {_c(pm_['only_this_model'])} | "
                   + " | ".join(_c(pm_["by_seeds_wrong"][str(k)]) for k in range(1, len(seeds) + 1)) + " |")
    out += ["", "| 범위 | 문장 수 (FP / FN) |", "| --- | ---: |",
            f"| 세 모델 공통 모델 오류 | {_c(E['all_models'])} |"]
    for pair, d in E["exactly_two"].items():
        out.append(f"| {'·'.join(name[m] for m in pair.split(';'))}만의 모델 오류 | {_c(d)} |")
    out += [f"| 한 모델 이상의 모델 오류 (합집합) | {_c(E['any_model'])} |",
            f"| 한 실행 이상에서 틀림 (`errors.csv` 행) | {_c(E['any_run'])} |", ""]
    return "\n".join(out)


def write_outputs(out_dir, res, script_sha):
    files = {
        "runs.csv": csv_text(RUN_COLUMNS, res["rows"]),
        "models.csv": models_csv(res),
        "errors.csv": csv_text(error_columns(res["models"]), res["error_rows"]),
        "summary.json": summary_json(res, script_sha),
        "summary.md": summary_md(res),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, text in files.items():
        (out_dir / n).write_bytes(text.encode("utf-8"))
    return list(files)


# ---------------------------------------------------------------- terminal

def _printable(text):
    return "".join(ch if ch.isprintable() else ch.encode("unicode_escape").decode("ascii") for ch in text)


def show_text(res, n):
    """Print up to n model-error sentences per model (terminal only). Returns an exit code."""
    data_lists.install_test_guard()
    try:
        val = data_lists.load_rows()["validation"]
    except data_lists.ListError as exc:
        for e in exc.errors:
            print("FAIL " + e)
        return 1
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    models = res["models"]
    print(f"model-error sentences (terminal only, never written): up to {n} per model, sentence ID order")
    for m in models:
        rows = [r for r in res["error_rows"] if r[f"{m}_model_error"]]
        print(f"{m}: {len(rows)} model errors")
        for r in rows[:n]:
            label, text = val[r["sentence_id"]]
            seeds = ", ".join(f"{k} {r[f'{k}_seeds_wrong']}" for k in models)
            warn = "" if int(label) == r["label"] else " [label differs from the raw split]"
            print(f"  {r['sentence_id']} {r['error_type']} (seeds wrong: {seeds}){warn}: {_printable(text)}")
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description="Summarise the nine stage-2 runs into results/s2/ (README 5.1, 6.1, 8).")
    p.add_argument("--out", help=f"output folder (default: <root>/{OUT_REL})")
    p.add_argument("--root", help="repository root holding runs/ and configs/ (default: this repository)")
    p.add_argument("--show-text", type=int, default=0, metavar="N",
                   help="also print, to the terminal only, up to N model-error sentences per model")
    args = p.parse_args(argv)
    if args.show_text < 0:
        p.error("--show-text must be >= 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root) if args.root else ROOT
    out = Path(args.out) if args.out else root / OUT_REL
    errors, res = summarize(root)
    if errors:
        for e in errors:
            print(train._ascii("FAIL " + e))
        print(f"refused: {len(errors)} error(s); nothing written")
        return 1
    names = write_outputs(out, res, data_lists.sha256_lf(ROOT / SCRIPT_REL))
    try:
        shown = out.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        shown = str(out)
    print(train._ascii(f"OK {len(res['rows'])} runs checked; wrote {shown}/: {', '.join(names)}"))
    for m in res["models"]:
        f1, me = res["stats"][m]["validation_clean_f1"], res["errors"]["per_model"][m]["model_error"]
        print(train._ascii(f"{m}: validation_clean F1 {_pm(f1, 4).replace('±', '+-')}; model errors "
                           f"{me['total']} (FP {me['fp']}, FN {me['fn']})"))
    E = res["errors"]
    print(f"model errors common to all models {E['all_models']['total']}, any model {E['any_model']['total']}, "
          f"wrong in any run {E['any_run']['total']}")
    return show_text(res, args.show_text) if args.show_text else 0


if __name__ == "__main__":
    sys.exit(main())
