"""Fine-tune one model on the common lists: stage-1 smoke runs, stage-2 runs (condition A) and stage-4
runs (condition C: train_clean + augmentation) (README 4.2, 5.3, 6.1, 7.1).

Run from Git Bash at the repo root:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/train.py --model bert --smoke --lists trial
        -> runs/s1-bert-smoke-s0/ (first 100 updates of the N schedule, no kept checkpoint)
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/train.py --model roberta --seed 42
        -> runs/s2-roberta-A-s42/ (official lists only)
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/train.py --model roberta --seed 42 --condition C
        -> runs/s4-roberta-C-s42/ (official lists + data/cases/train/train_aug.csv, same N)
    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/train.py --smoke-summary
Add --rerun to repeat a run: the previous small files move to runs/<id>/attempts/<k>/,
its checkpoint is deleted (the archived run.json then says checkpoint.kept false,
deleted_by_rerun true), and the same-seed difference is recorded (rerun_diff).
Add --preflight-only to run the preflight of a run and exit (0: it would start).

Settings come from configs/train.json only (no CLI overrides). Rules:
  - Startup: test.csv audit guard (data_lists.install_test_guard), then
    CUBLAS_WORKSPACE_CONFIG from the config, then torch. Preflight creates
    nothing: config sanity, pinned model key, --lists trial only with
    --smoke, seed in the config, HF_HUB_OFFLINE=1, run folder absent (or
    present with --rerun) and a plain direct child of runs/ (no symlink or
    junction), no runs/<id>.lock, list verification (data_lists.verify_lists;
    the official mode runs build_lists.py --check as a subprocess), N ==
    expected_updates, CUDA. Stage 2 also needs the official lists, every
    imported project file and the config git-tracked and unmodified and no
    modified tracked file outside runs/; smoke runs skip only the git checks
    (lists block git_tracked_checked false) and record the dirty state of the
    code, config and list files (other paths as a count).
  - Lock: runs/<id>.lock (O_EXCL, gitignored) is held from before the run
    folder is touched until run.json is final; a killed run leaves it behind.
  - N = epochs * ceil(train_clean rows / 16), epochs 3; evals at ceil(N/3), ceil(2N/3), N.
    Epoch plan: randperm per epoch from a CPU torch.Generator(seed) separate
    from the global RNG, chunked into updates of micro_batch * grad_accum rows
    (the last one partial), each split into micro-batches; generic in the
    dataset size M, stops at N updates.
  - seed_everything(seed) right before model_specs.load_model (eager
    attention, FP32 master weights). BF16 autocast for the forward pass, no
    GradScaler. Loss per micro-batch = CE(logits.float(), y, sum) / m with m
    the update's example count, so the update loss is the exact mean. AdamW,
    weight decay only on parameters with ndim >= 2; clip_grad_norm_ (pre-clip
    norm logged); LambdaLR max(0, (N - d) / N) stepped after every optimizer
    step (update u runs at lr * (N - u + 1) / N). A non-finite loss or grad
    norm fails the run.
  - Eval (eval_lib): FP32, batch 64, all validation rows in sentence-ID
    order; metrics for validation and validation_clean (in_clean rows);
    checkpoint saved (model_specs.save_checkpoint) only on a strict
    improvement of the selection key (F1, -FPR, -step) on validation_clean.
    Stage 2 saves into runs/<id>/checkpoint.staging/ (gitignored).
  - End of run: peak memory read, training model and optimizer freed, the
    checkpoint reloaded with model_specs.load_finetuned; every tensor's
    model_specs.tensor_digest must equal the one taken while it was written
    and the validation predictions of the selected step must not flip (max
    |margin diff| recorded). Only then is the staging folder renamed to
    runs/<id>/checkpoint/ and the checkpoint counts as validated.
  - Failures: run.json status failed/interrupted with phase, update,
    exception type and line (message only for determinism, non-finite and
    reload-check errors); a checkpoint that was not validated is deleted; exit 1. A
    validated stage-2 checkpoint is kept even if the run is interrupted later.
  - Smoke (stage s1, condition smoke, seed 0): first 100 updates of the N
    schedule, eval at 100, checkpoint in the fixed temp folder
    %TEMP%/<repo hash>-<run id>-ckpt outside the repo (a leftover is removed
    first), reload-checked, SHA recorded, deleted. Pass: finite losses and
    grad norms, mean loss of the last 20 updates < first 20,
    max_memory_allocated <= 7.0e9 B, reload check passed. Timing: median
    update time over updates 6-100; one-run estimate T = N*t + 4*t_eval +
    t_load + 3*t_save + t_reload (excludes data preparation and preflight).
    Exit 0 only if everything passes.
  - Rerun difference: comparable only when the previous attempt completed
    with identical code, config, list files, requirements, torch, CUDA and
    GPU name, and every update and prediction row has a counterpart.
  - runs/<id>/: config.json (byte copy), run.json, metrics.json,
    train_log.csv, predictions_validation.csv, log.txt, checkpoint/ (stage 2).
    JSON files are written to <name>.tmp, then os.replace.
    IDs, labels and numbers only, never sentence text; repo-relative POSIX
    paths; LF; stdout ASCII.
  - Condition C (--condition C; stage s4, run ID s4-<model>-C-s<seed>;
    README 5.3, IMP-02): a new run from the pretrained weights on train_clean
    plus the augmentation rows; never with --smoke. Everything stage 2 checks
    (official lists, the git checks P08 and P09) applies, plus P13 on the
    augmentation set data/cases/train/: cases.csv, review.csv, train_aug.csv,
    cases_meta.json, test_overlap.json and scripts/build_train_aug.py exist,
    are git-tracked and unmodified; cases_meta.json has ready_for_training
    true, review.pending 0 and test_overlap.status "applied"; the SHA-256 of
    cases.csv, review.csv and train_aug.csv equal its "outputs" and the
    LF-normalised SHA-256 of build_train_aug.py equals its "script"; the
    augmentation rows join; `build_train_aug.py --check` as a subprocess exits
    0 (run last, only when nothing else failed). An "outputs" entry that is
    missing or has no SHA-256 is reported by file name, next to the state of
    the set: the builder records train_aug.csv only for a set that is ready.
    A refusal is bounded: the first ten per-row join errors, then one count
    per code; the first five FAIL lines of --check; a line that carries an ID
    or a FAIL line is cut at 300 characters.
    Augmentation rows: train_aug.csv (data_lists.read_list) joined with the
    case texts of cases.csv (csv module) by data_lists.join: IDs strictly
    ascending, label and text_sha256_12 equal. Every row must be a case with
    status assigned, purpose train, a non-empty text and not the held-out
    subtype homoglyph; an empty train_aug.csv is refused. The same bytes are
    hashed and parsed; the run rereads the files and fails (FAIL J.. lines in
    log.txt, no data block in run.json) when a SHA-256 differs from the
    preflight's. Loading draws no random number.
    N stays epochs * ceil(train_clean rows / 16) and the eval steps stay the
    same: the rows are appended after the train_clean rows, in train_aug.csv
    order, once the budget recheck on train_clean is done, so the epoch plan
    permutes train_clean + augmentation and stops at N updates.
    run.json data then has train_clean_rows, augmentation_rows, augmentation
    {dir, files_sha256, label_counts, build_train_aug_sha256_lf} and
    train_rows = all rows trained on; condition A and smoke records have none
    of these keys. A rerun is comparable only when the augmentation files
    SHA-256 are also equal.
  - Refusal codes (FAIL <code> ..., exit 1, nothing created): P01 config,
    P02 model, P03 seed, --lists or --condition, P04 offline flag, P05 CUBLAS
    variable, P06 run folder, P07 lists (carries L01-L08 of data_lists), P08
    project files untracked or modified, P09 other modified tracked files,
    P10 N, P11 CUDA, P12 lock, P13 augmentation set (condition C only; may
    carry a join code). Join codes (data_lists.ListError): J00-J05 of
    data_lists, J06 augmentation file unreadable or malformed, J07 a
    train_aug.csv row is not a usable train case, J08 train_aug.csv has no
    rows, J09 an augmentation file changed after the preflight.
"""
import argparse
import csv
import gc
import io
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import data_lists

ROOT = data_lists.ROOT
CONFIG_REL = "configs/train.json"

if __name__ == "__main__":
    data_lists.install_test_guard()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = json.loads(
        (ROOT / CONFIG_REL).read_bytes())["determinism"]["cublas_workspace_config"]

import torch  # noqa: E402  (after the guard and CUBLAS_WORKSPACE_CONFIG)
import torch.nn.functional as F  # noqa: E402

import build_lists  # noqa: E402
import eval_lib  # noqa: E402
import model_specs  # noqa: E402

CODE_FILES = ("scripts/train.py", "scripts/data_lists.py", "scripts/eval_lib.py", "scripts/model_specs.py",
              "scripts/build_lists.py")
SMALL_FILES = ("config.json", "run.json", "metrics.json", "train_log.csv", "predictions_validation.csv", "log.txt")
STAGING = "checkpoint.staging"  # stage 2: checkpoint written here, renamed to checkpoint/ after the reload check
TRAIN_LOG_HEADER = "update,epoch,examples,lr,loss,grad_norm,seconds"
PRED_HEADER = "step,sentence_id,label,in_clean,margin,score,prediction,threshold,condition"
UPDATE_SIZE = 16  # README 4.2: effective batch 16
MEMORY_WINDOW = ("torch.cuda.max_memory_allocated / max_memory_reserved after reset_peak_memory_stats right after "
                 "CUDA init; covers tokenizer, model load, all updates and evals; read before the reload check")
TIME_WINDOW = ("per update: wall time of one optimizer update (all micro-batches, clip, step, schedule) between "
               "cuda.synchronize calls; evals, saves and logging excluded")
T_FORMULA = "T = N*t + 4*t_eval + t_load + 3*t_save + t_reload (estimate; excludes data preparation and preflight)"
# Config keys the code does not read but implements; check_config asserts their values.
FIXED_CONFIG = {"padding": "dynamic", "inputs": ["input_ids", "attention_mask"],
                "schedule": "linear decay to 0 over N updates, no warmup: LambdaLR max(0, (N - d) / N)"}
EPOCHS = 3  # README 4.2: eval_steps are ceil(N/3), ceil(2N/3), N, i.e. the ends of 3 epochs
# Trained conditions and their stage (README 5.3): B and D reuse the A and C checkpoints with another threshold.
STAGE_OF = {"A": "s2", "C": "s4"}
CONDITIONS = tuple(STAGE_OF)
# Condition C augmentation set (README 7.1), built by scripts/build_train_aug.py.
AUG_DIR_REL = "data/cases/train"
AUG_CASES, AUG_LIST, AUG_META = "cases.csv", "train_aug.csv", "cases_meta.json"
AUG_FILES = (AUG_CASES, "review.csv", AUG_LIST, AUG_META, "test_overlap.json")
AUG_PINNED = (AUG_CASES, "review.csv", AUG_LIST)  # SHA-256 recorded in cases_meta.json "outputs"
AUG_CASE_COLUMNS = ("case_id", "purpose", "subtype", "label", "status", "text")  # cases.csv columns read here
AUG_HELD_OUT = ("homoglyph",)  # README 5.2: the held-out ROB-01 method is never trained on
BUILD_TRAIN_AUG = "scripts/build_train_aug.py"

NonFiniteError = eval_lib.NonFiniteError


class ReloadCheckError(RuntimeError):
    pass


def _ascii(s):
    return str(s).encode("ascii", "backslashreplace").decode("ascii")


def say(msg):
    print(_ascii(msg), flush=True)


def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dump(obj):
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def write_json(path, obj):
    """Atomic: <name>.tmp, then os.replace (a crash never leaves a truncated record)."""
    text, tmp = _dump(obj), Path(path).with_name(Path(path).name + ".tmp")
    with open(tmp, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def _exc_brief(exc):
    """'<ExcType>: <message>' without file system paths (OSError: strerror only), ASCII, at most 300 chars."""
    msg = (exc.strerror or "") if isinstance(exc, OSError) else str(exc)
    for p in (str(ROOT), ROOT.as_posix()):
        msg = msg.replace(p, "<repo>")
    return _ascii(f"{build_lists.exc_name(exc)}: {msg}")[:300]


# ---------------------------------------------------------------- schedule

def budget(rows, epochs, update_size=UPDATE_SIZE):
    """N = epochs * ceil(rows / update_size) (README 4.2)."""
    return epochs * -(-rows // update_size)


def eval_steps(n):
    return [-(-n // 3), -(-2 * n // 3), n]


def epoch_plan(m, n_updates, seed, update_size=UPDATE_SIZE, micro_batch=4):
    """[(epoch, [micro-batch index lists])] for n_updates updates over a dataset of size m.

    Each epoch is torch.randperm(m) from one CPU generator seeded with seed;
    epochs repeat until n_updates updates exist (stops mid-epoch when needed).
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    plan, epoch = [], 0
    while len(plan) < n_updates:
        epoch += 1
        perm = torch.randperm(m, generator=g).tolist()
        for i in range(0, m, update_size):
            idx = perm[i:i + update_size]
            plan.append((epoch, [idx[j:j + micro_batch] for j in range(0, len(idx), micro_batch)]))
            if len(plan) == n_updates:
                break
    return plan


def make_optimizer(model, ocfg):
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= ocfg["weight_decay_min_ndim"] else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": ocfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=ocfg["lr"], betas=tuple(ocfg["betas"]), eps=ocfg["eps"])
    info = {"decay_tensors": len(decay), "decay_elements": sum(p.numel() for p in decay),
            "no_decay_tensors": len(no_decay), "no_decay_elements": sum(p.numel() for p in no_decay),
            "foreach": opt.defaults.get("foreach"), "fused": opt.defaults.get("fused")}
    return opt, info


def make_scheduler(optimizer, n_total):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda d: max(0.0, (n_total - d) / n_total))


def seed_everything(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)


def apply_determinism(cfg):
    d = cfg["determinism"]
    torch.use_deterministic_algorithms(d["use_deterministic_algorithms"], warn_only=d["warn_only"])
    torch.backends.cudnn.deterministic = d["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = d["cudnn_benchmark"]
    # One API only: on torch 2.14 reading the legacy cudnn.allow_tf32 after the new
    # fp32_precision API was used raises, so the legacy flags are neither set nor read.
    torch.backends.fp32_precision = "tf32" if d["tf32"] else "ieee"
    return {
        "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "fp32_precision": {"global": torch.backends.fp32_precision,
                           "cuda_matmul": torch.backends.cuda.matmul.fp32_precision,
                           "cudnn_conv": torch.backends.cudnn.conv.fp32_precision,
                           "cudnn_rnn": torch.backends.cudnn.rnn.fp32_precision},
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------- training

def eval_record(step, labels, in_clean, margins, preds):
    clean = [i for i, c in enumerate(in_clean) if c]
    return {"step": step,
            "validation": eval_lib.metrics(labels, preds.tolist(), margins.tolist()),
            "validation_clean": eval_lib.metrics([labels[i] for i in clean], preds[clean].tolist(),
                                                 margins[clean].tolist())}


def fit(model, key, data, cfg, *, seed, n_total, n_run, eval_at, device, pad_id, ckpt_dir, run_id,
        log=print, on_update=None, on_eval=None, progress=None):
    """Train for n_run updates of the n_total schedule; returns log rows, evals and the best checkpoint.

    data: train_ids, train_labels, val_ids, val_labels, val_in_clean.
    """
    progress = {} if progress is None else progress
    micro = cfg["micro_batch"]
    plan = epoch_plan(len(data["train_ids"]), n_total, seed, micro * cfg["grad_accum"], micro)[:n_run]
    optimizer, opt_info = make_optimizer(model, cfg["optimizer"])
    scheduler = make_scheduler(optimizer, n_total)
    labels_t = torch.tensor(data["train_labels"], dtype=torch.long)
    use_bf16 = cfg["precision"] == "bf16"
    threshold = cfg["threshold"]
    log_every = max(1, n_run // 20)
    rows, evals, saved, t_eval, t_save, best = [], [], [], [], [], None
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for u, (epoch, micro_batches) in enumerate(plan, start=1):
        progress.update(phase="train", update=u)
        m = sum(len(b) for b in micro_batches)
        lr = optimizer.param_groups[0]["lr"]
        _sync(device)
        t0 = time.perf_counter()
        loss_acc = torch.zeros((), dtype=torch.float64, device=device)
        for idx in micro_batches:
            ids, mask = eval_lib.collate([data["train_ids"][i] for i in idx], pad_id)
            y = labels_t[idx].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = F.cross_entropy(logits.float(), y, reduction="sum") / m
            loss.backward()
            loss_acc += loss.detach().double()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
        loss_v, gn_v = loss_acc.item(), grad_norm.item()
        if not (math.isfinite(loss_v) and math.isfinite(gn_v)):
            raise NonFiniteError(f"update {u}: loss {loss_v}, grad_norm {gn_v}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        _sync(device)
        seconds = time.perf_counter() - t0
        row = {"update": u, "epoch": epoch, "examples": m, "lr": lr, "loss": loss_v, "grad_norm": gn_v,
               "seconds": seconds}
        rows.append(row)
        if on_update:
            on_update(row)
        if u % log_every == 0 or u == n_run:
            log(f"update {u}/{n_run} epoch {epoch} loss {loss_v:.4f} grad_norm {gn_v:.3f} lr {lr:.3e} "
                f"{seconds:.3f}s")
        if u not in eval_at:
            continue
        progress.update(phase="eval")
        t0 = time.perf_counter()
        margins, scores = eval_lib.predict(model, data["val_ids"], pad_id, cfg["eval"]["batch"], device)
        preds = eval_lib.predictions(scores, threshold)
        t_eval.append(time.perf_counter() - t0)
        rec = eval_record(u, data["val_labels"], data["val_in_clean"], margins, preds)
        key_u = eval_lib.selection_key(rec["validation_clean"], u)
        rec["improved"] = best is None or key_u > best["key"]
        evals.append(rec)
        vc = rec["validation_clean"]
        log(f"eval step {u}: validation_clean F1 {vc['f1']} FPR {vc['fpr']} ROC-AUC {vc['roc_auc']}; "
            f"validation F1 {rec['validation']['f1']}; {t_eval[-1]:.1f}s"
            + ("; improved, saving checkpoint" if rec["improved"] else ""))
        if rec["improved"]:
            progress.update(phase="save")
            t0 = time.perf_counter()
            info = model_specs.save_checkpoint(model, key, ckpt_dir, run_id, u)
            t_save.append(time.perf_counter() - t0)
            best = {"key": key_u, "step": u, "sha256": info["sha256"], "size": info["size"],
                    "tensors": info["tensors"], "margins": margins, "preds": preds}
            saved.append(u)
        if on_eval:
            on_eval(rec, margins, scores, preds)
    if evals and eval_lib.select(evals)["step"] != best["step"]:
        raise RuntimeError("selected step differs from the last saved checkpoint")
    return {"log": rows, "evals": evals, "saved": saved, "best": best, "eval_s": t_eval, "save_s": t_save,
            "optimizer": opt_info}


def reload_check(key, ckpt_file, best, data, cfg, device, pad_id):
    """Reload the saved checkpoint; compare per-tensor digests with those taken while it was written and
    predictions with the selected step's."""
    t0 = time.perf_counter()
    model = model_specs.load_finetuned(key, ckpt_file, best["sha256"], size=best["size"],
                                       attn_implementation=cfg["attn_implementation"])
    sd = model.state_dict()
    equal = {k: model_specs.tensor_digest(t) for k, t in sd.items()} == best["tensors"]
    model.to(device)
    margins, scores = eval_lib.predict(model, data["val_ids"], pad_id, cfg["eval"]["batch"], device)
    preds = eval_lib.predictions(scores, cfg["threshold"])
    flips = int((preds != best["preds"]).sum())
    diff = float((margins - best["margins"]).abs().max())
    del model, sd
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"state_dict_equal": equal, "prediction_flips": flips, "max_abs_margin_diff": diff,
            "passed": equal and flips == 0, "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------- preflight

def check_config(cfg):
    errors = []

    def need(ok, msg):
        if not ok:
            errors.append(f"P01 {CONFIG_REL}: {msg}")

    try:
        for k, want in FIXED_CONFIG.items():
            need(cfg[k] == want, f"{k} must be {json.dumps(want)} (implemented by the code, not read)")
        need(isinstance(cfg["max_length"], int) and cfg["max_length"] > 0, "max_length")
        need(cfg["micro_batch"] * cfg["grad_accum"] == UPDATE_SIZE,
             f"micro_batch * grad_accum must be {UPDATE_SIZE} (README 4.2, TRAIN-03)")
        need(cfg["precision"] in ("bf16", "fp32"), "precision must be bf16 or fp32")
        need(cfg["eval"]["precision"] == "fp32" and cfg["eval"]["batch"] > 0, "eval")
        o = cfg["optimizer"]
        need(o["name"] == "AdamW" and o["lr"] > 0 and len(o["betas"]) == 2 and o["eps"] > 0
             and o["weight_decay"] >= 0 and o["weight_decay_min_ndim"] >= 1, "optimizer")
        need(cfg["grad_clip_norm"] > 0, "grad_clip_norm")
        need(cfg["epochs"] == EPOCHS and type(cfg["epochs"]) is int,
             f"epochs must be {EPOCHS} (eval_steps ceil(N/3), ceil(2N/3), N are the epoch ends; README 4.2)")
        need(isinstance(cfg["expected_updates"], int), "expected_updates")
        need(0 < cfg["threshold"] < 1, "threshold")
        need(cfg["attn_implementation"] in model_specs.ATTN_IMPLEMENTATIONS, "attn_implementation")
        d = cfg["determinism"]
        need(all(isinstance(d[k], bool) for k in ("use_deterministic_algorithms", "warn_only",
                                                  "cudnn_deterministic", "cudnn_benchmark", "tf32"))
             and isinstance(d["cublas_workspace_config"], str), "determinism")
        need(all(isinstance(s, int) for s in cfg["seeds"]), "seeds")
        s = cfg["smoke"]
        need(isinstance(s["seed"], int) and s["seed"] not in cfg["seeds"], "smoke.seed must differ from seeds")
        need(0 < s["updates"] <= cfg["expected_updates"] and all(0 < e <= s["updates"] for e in s["eval_at"])
             and s["memory_limit_bytes"] > 0 and 0 <= s["timing_skip"] < s["updates"] - 1, "smoke")
    except (KeyError, TypeError, IndexError) as exc:
        errors.append(f"P01 {CONFIG_REL}: missing or malformed key ({type(exc).__name__})")
    return errors


def run_identity(args, cfg):
    """(stage, condition, seed, run ID): --smoke -> s1 / smoke / the config's smoke seed; otherwise the
    stage of the condition (A -> s2, C -> s4; A when args has no condition) with --seed.

    Raises ValueError for an unknown condition and for a condition other than A with --smoke.
    """
    cond = getattr(args, "condition", "A")
    if cond not in STAGE_OF or (args.smoke and cond != "A"):
        raise ValueError(f"no run identity for condition {cond!r}" + (" with --smoke" if args.smoke else ""))
    stage, cond, seed = ("s1", "smoke", cfg["smoke"]["seed"]) if args.smoke else (STAGE_OF[cond], cond, args.seed)
    return stage, cond, seed, f"{stage}-{args.model}-{cond}-s{seed}"


# ---------------------------------------------------------------- augmentation (condition C)

class _Blob:
    """Path stand-in: lets data_lists.read_list parse bytes that were already read and hashed."""

    def __init__(self, name, data):
        self.name, self._data = name, data

    def read_bytes(self):
        return self._data


def _clip(line, limit=300):
    """line as ASCII, at most limit characters (a cut line ends in "..."): bounds a refusal line that carries
    file content (an ID of train_aug.csv, a FAIL line of --check)."""
    line = _ascii(line)
    return line if len(line) <= limit else line[:limit - 3] + "..."


def parse_cases(data):
    """{case_id: {column: value}} from cases.csv bytes (UTF-8, csv module; line breaks inside a text are kept).

    Raises data_lists.ListError (J06). Messages carry record numbers and counts only.
    """
    try:
        recs = list(csv.reader(io.StringIO(data.decode("utf-8"), newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise data_lists.ListError([f"J06 {AUG_CASES}: {build_lists.exc_name(exc)} (not UTF-8 csv)"]) from None
    head = recs[0] if recs else []
    if len(set(head)) != len(head) or not set(AUG_CASE_COLUMNS) <= set(head):
        raise data_lists.ListError([f"J06 {AUG_CASES}: the header repeats a column or lacks one of "
                                    f"{', '.join(AUG_CASE_COLUMNS)}"])
    cases, errors = {}, []
    for n, rec in enumerate(recs[1:], start=1):
        if len(rec) != len(head):
            errors.append(f"J06 {AUG_CASES} data record {n}: {len(rec)} fields, the header has {len(head)}")
            continue
        r = dict(zip(head, rec))
        if r["case_id"] in cases:
            errors.append(f"J06 {AUG_CASES} data record {n}: duplicate case_id")
        elif r["label"] not in ("0", "1"):
            errors.append(f"J06 {AUG_CASES} data record {n}: label is not 0 or 1")
        else:
            cases[r["case_id"]] = r
    if errors:
        more = [f"J06 {AUG_CASES}: {len(errors) - 10} more malformed record(s)"] if len(errors) > 10 else []
        raise data_lists.ListError(errors[:10] + more)
    return cases


def join_augmentation(cases_bytes, list_bytes):
    """[(case_id, label int, text)] in train_aug.csv order from the bytes of cases.csv and train_aug.csv.

    Raises data_lists.ListError: J00-J04 (data_lists.read_list / join), J06 malformed file, J07 a row that
    is not a usable train case, J08 no rows. The refusal is bounded: the first ten J01-J04 errors (one per
    row), then one count per code; one J07 line per rule; every line at most 300 characters.
    """
    cases = parse_cases(cases_bytes)
    try:
        listed = data_lists.read_list(_Blob(AUG_LIST, list_bytes))
    except UnicodeDecodeError:
        raise data_lists.ListError([f"J06 {AUG_LIST}: not ASCII"]) from None
    raw = list_bytes.split(b"\n")
    if raw[0] != data_lists.TRAIN_VAL_HEADER.encode("ascii") or any(ln.count(b",") != 2 for ln in raw[1:-1]):
        raise data_lists.ListError([f"J06 {AUG_LIST}: the header is not {data_lists.TRAIN_VAL_HEADER} or a row "
                                    "does not have 3 fields"])
    if not listed:
        raise data_lists.ListError([f"J08 {AUG_LIST}: no rows (condition C needs augmentation rows)"])
    joined, errors = data_lists.join(listed, {cid: (c["label"], c["text"]) for cid, c in cases.items()}, AUG_LIST)
    if len(errors) > 10:  # as parse_cases: a list of 10,000 wrong rows must not give 10,000 lines
        codes = [e.split(" ", 1)[0] for e in errors[10:]]
        errors = errors[:10] + [f"{code} {AUG_LIST}: {codes.count(code)} more error(s) of this kind"
                                for code in sorted(set(codes))]
    for what, wrong in (("status is not assigned", lambda c: c["status"] != "assigned"),
                        ("purpose is not train", lambda c: c["purpose"] != "train"),
                        ("held-out subtype " + "/".join(AUG_HELD_OUT), lambda c: c["subtype"] in AUG_HELD_OUT),
                        ("empty text", lambda c: c["text"] == "")):
        bad = [cid for cid, _, _ in listed if cid in cases and wrong(cases[cid])]
        if bad:
            errors.append(f"J07 {AUG_LIST}: {what} in {AUG_CASES}: {len(bad)} row(s) "
                          f"({', '.join(bad[:5])}{', ...' if len(bad) > 5 else ''})")
    if errors:
        raise data_lists.ListError([_clip(e) for e in errors])
    return joined


def label_counts(rows):
    """{"0": n, "1": n} for rows shaped (id, label int, text)."""
    return {str(y): sum(r[1] == y for r in rows) for y in (0, 1)}


def load_augmentation(directory, files_sha256):
    """Augmentation rows [(case_id, label int, text)] in train_aug.csv order for a condition C run.

    Every AUG_FILES file is read once; a SHA-256 that differs from files_sha256 (the preflight's) raises
    data_lists.ListError J09 before anything is parsed; then join_augmentation. No random number is drawn.
    """
    try:
        blobs = {n: (Path(directory) / n).read_bytes() for n in AUG_FILES}
    except OSError as exc:
        raise data_lists.ListError([f"J06 {AUG_DIR_REL}: cannot read ({build_lists.exc_name(exc)})"]) from None
    changed = [n for n in AUG_FILES if data_lists.sha256_bytes(blobs[n]) != files_sha256.get(n)]
    if changed:
        raise data_lists.ListError([f"J09 {AUG_DIR_REL}: changed after the preflight: {', '.join(changed)}"])
    return join_augmentation(blobs[AUG_CASES], blobs[AUG_LIST])


def _brief(value):
    return _ascii(json.dumps(value))[:40]


def verify_augmentation(root=ROOT):
    """P13 (condition C). Return (errors, info); info is the run.json data.augmentation block (None on errors).

    Reads only. Each file is read once and the same bytes are hashed, compared with cases_meta.json and
    joined. `build_train_aug.py --check` runs last, as a subprocess, only when nothing else failed.
    The state of the set (ready_for_training, review.pending, test_overlap.status) is reported before the
    pins, and a missing "outputs" entry by file name: a train_aug.csv left next to the meta of a set that
    is not ready is refused with both.
    """
    d = root / AUG_DIR_REL
    rels = [f"{AUG_DIR_REL}/{n}" for n in AUG_FILES] + [BUILD_TRAIN_AUG]
    missing = [p for p in rels if not (root / p).is_file()]
    if missing:
        return [f"P13 augmentation: missing: {', '.join(missing)} ({BUILD_TRAIN_AUG} writes {AUG_LIST} only when "
                "the review is complete and the test overlap is applied)"], None
    errors = []
    untracked = sorted(set(rels) - data_lists.git_tracked(root, rels))
    if untracked:
        errors.append(f"P13 augmentation: not tracked by git: {', '.join(untracked)}")
    dirty = data_lists.git_status(root, rels)
    if dirty is None:
        errors.append("P13 augmentation: git status failed")
    elif dirty:
        errors.append(f"P13 augmentation: modified in the working tree or index: {', '.join(p for _, p in dirty)}")
    try:
        blobs = {n: (d / n).read_bytes() for n in AUG_FILES}
        script_sha = data_lists.sha256_lf(root / BUILD_TRAIN_AUG)
    except OSError as exc:
        return errors + [f"P13 augmentation: cannot read ({build_lists.exc_name(exc)})"], None
    files = {n: data_lists.sha256_bytes(b) for n, b in blobs.items()}
    meta_rel = f"{AUG_DIR_REL}/{AUG_META}"
    try:
        meta = json.loads(blobs[AUG_META].decode("utf-8"))
        ready, pending = meta["ready_for_training"], meta["review"]["pending"]
        overlap = meta["test_overlap"]["status"]
        outputs, meta_script = meta["outputs"], meta["script"]["sha256_lf"]
        if not isinstance(outputs, dict) or not isinstance(meta_script, str):
            raise TypeError
    except (ValueError, KeyError, TypeError):
        return errors + [f"P13 {meta_rel}: unreadable or missing keys"], None
    if ready is not True:
        errors.append(f"P13 {meta_rel}: ready_for_training is {_brief(ready)}, condition C needs true")
    if type(pending) is not int or pending != 0:
        errors.append(f"P13 {meta_rel}: review.pending is {_brief(pending)}, condition C needs 0 "
                      f"(decide every case in {AUG_DIR_REL}/review.csv, then rebuild)")
    if overlap != "applied":
        errors.append(f"P13 {meta_rel}: test_overlap.status is {_brief(overlap)}, condition C needs \"applied\" "
                      f"(run scripts/check_cases_vs_test.py, then {BUILD_TRAIN_AUG})")
    # The pins, one file at a time: the builder writes no train_aug.csv entry for a set that is not ready, and
    # the state errors above must be reported next to that, not hidden by it.
    for n in AUG_PINNED:
        entry = outputs.get(n)
        want = entry.get("sha256") if isinstance(entry, dict) else None
        if entry is None:
            errors.append(f"P13 {meta_rel}: outputs has no {n} entry" + (
                f" ({BUILD_TRAIN_AUG} records it only for a set that is ready for training)" if n == AUG_LIST
                else ""))
        elif not isinstance(want, str):
            errors.append(f"P13 {meta_rel}: outputs {n} has no sha256 string")
        elif files[n] != want:
            errors.append(f"P13 {AUG_DIR_REL}/{n}: sha256 {files[n][:12]} != cases_meta {_ascii(want)[:12]}")
    if script_sha != meta_script:
        errors.append(f"P13 {BUILD_TRAIN_AUG}: sha256_lf {script_sha[:12]} != cases_meta script "
                      f"{_ascii(meta_script)[:12]} (script changed after the augmentation set was built: "
                      "rebuild it)")
    rows = None
    if not errors:
        try:
            rows = join_augmentation(blobs[AUG_CASES], blobs[AUG_LIST])
        except data_lists.ListError as exc:
            errors += [f"P13 {e}" for e in exc.errors]
    if not errors:
        try:
            r = subprocess.run([sys.executable, str(root / BUILD_TRAIN_AUG), "--check"], cwd=str(root),
                               capture_output=True, timeout=data_lists.CHECK_TIMEOUT_S,
                               env={**os.environ, "PYTHONUTF8": "1"})
        except subprocess.TimeoutExpired:
            errors.append(f"P13 {BUILD_TRAIN_AUG} --check: no result within {data_lists.CHECK_TIMEOUT_S} s")
        else:
            if r.returncode != 0:
                fails = [ln for ln in r.stdout.decode("ascii", "replace").splitlines() if ln.startswith("FAIL")]
                errors.append(f"P13 {BUILD_TRAIN_AUG} --check exit {r.returncode}: "
                              + " | ".join(_clip(ln) for ln in fails[:5]))
    if errors:
        return errors, None
    return [], {"dir": AUG_DIR_REL, "files_sha256": files, "label_counts": label_counts(rows),
                "build_train_aug_sha256_lf": script_sha}


def preflight(args, cfg, root=ROOT):
    """Return (errors, ctx). Reads only; creates no file or folder."""
    errors = check_config(cfg)
    if errors:
        return errors, None
    condition = getattr(args, "condition", "A")
    if condition not in CONDITIONS:
        return [f"P03 condition {_ascii(repr(condition))[:40]} is not one of {', '.join(CONDITIONS)} (README 5.3: "
                "B and D reuse the A and C checkpoints)"], None
    if args.smoke and condition != "A":
        return [f"P03 --condition {condition} is not allowed with --smoke (a smoke run is stage 1 on the "
                "common lists only)"], None
    official = f"stage {STAGE_OF[condition][1:]}"  # "stage 2" (A) or "stage 4" (C) in the messages below
    if args.model not in model_specs.WEIGHTS:
        errors.append(f"P02 model {args.model!r} has no pinned weights in model_specs.WEIGHTS")
    if args.lists == "trial" and not args.smoke:
        errors.append(f"P03 --lists trial is allowed only with --smoke (README 7.1: {official} uses the official "
                      "lists)")
    if not args.smoke and args.seed not in cfg["seeds"]:
        errors.append(f"P03 seed {args.seed} is not in {CONFIG_REL} seeds {cfg['seeds']}")
    stage, cond, seed, run_id = run_identity(args, cfg)
    run_dir = root / "runs" / run_id
    from huggingface_hub import constants
    if not constants.HF_HUB_OFFLINE:
        errors.append("P04 HF_HUB_OFFLINE=1 is required (set before Python starts)")
    want = cfg["determinism"]["cublas_workspace_config"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != want:
        errors.append(f"P05 CUBLAS_WORKSPACE_CONFIG is not {want} (train.py sets it when run as a script)")
    why = eval_lib.run_dir_problem(run_dir, root)
    if why:
        errors.append(f"P06 runs/{run_id} {why}; refused")
    elif run_dir.exists() and not args.rerun:
        errors.append(f"P06 runs/{run_id}/ exists; pass --rerun to archive it and run again")
    elif args.rerun and not run_dir.is_dir():
        errors.append(f"P06 --rerun: runs/{run_id}/ does not exist")
    if lock_path(run_dir).exists():
        errors.append(f"P12 runs/{run_id}.lock exists: a run of {run_id} is in progress, or one was killed before "
                      f"it finished. If no train.py process is running, delete runs/{run_id}.lock and retry "
                      f"(with --rerun if runs/{run_id}/ exists)")
    lerrs, lists = data_lists.verify_lists(args.lists, root, require_tracked=not args.smoke)
    errors += [f"P07 {e}" for e in lerrs]
    if stage in STAGE_OF.values():  # s2 and s4; smoke runs record the dirty state instead
        rels = list(CODE_FILES) + [CONFIG_REL]
        untracked = sorted(set(rels) - data_lists.git_tracked(root, rels))
        if untracked:
            errors.append(f"P08 {official} needs git-tracked project files; untracked: {', '.join(untracked)}")
        st = data_lists.git_status(root)
        if st is None:
            errors.append("P08 git status failed")
        else:
            mod = sorted(p for _, p in st if p in rels)
            if mod:
                errors.append(f"P08 {official} needs unmodified project files; modified: {', '.join(mod)}")
            other = sorted(p for _, p in st if p not in rels and not p.startswith("runs/"))
            if other:
                errors.append(f"P09 {official} refuses modified tracked files outside runs/: {len(other)} "
                              f"({', '.join(_ascii(p) for p in other[:5])}{', ...' if len(other) > 5 else ''})")
    augmentation = None
    if cond == "C":
        aerrs, augmentation = verify_augmentation(root)
        errors += aerrs
    n_total = None
    if lists is not None:
        n_total = budget(lists["train_clean_rows"], cfg["epochs"])
        if n_total != cfg["expected_updates"]:
            errors.append(f"P10 N = {n_total} from {lists['train_clean_rows']} train_clean rows != "
                          f"expected_updates {cfg['expected_updates']}")
    if not torch.cuda.is_available():
        errors.append("P11 CUDA is not available")
    if errors:
        return errors, None
    smoke = cfg["smoke"]
    ctx = {"stage": stage, "condition": cond, "seed": seed, "run_id": run_id, "run_dir": run_dir,
           "lists": lists, "list_dir": data_lists.list_dir(args.lists, root), "n_total": n_total,
           "eval_at": smoke["eval_at"] if args.smoke else eval_steps(n_total),
           "n_run": smoke["updates"] if args.smoke else n_total}
    if cond == "C":  # N, eval steps and updates above come from train_clean alone (README IMP-02)
        ctx.update(augmentation=augmentation, aug_dir=root / AUG_DIR_REL)
    return [], ctx


# ---------------------------------------------------------------- run folder and records

def lock_path(run_dir):
    return run_dir.parent / f"{run_dir.name}.lock"


def smoke_ckpt_dir(run_id, root=ROOT):
    """Fixed temp folder of a smoke run's checkpoint, outside the repo and unique per clone and run ID."""
    tag = data_lists.sha256_bytes(str(Path(root).resolve()).encode("utf-8"))[:12]
    return Path(tempfile.gettempdir()) / f"{tag}-{run_id}-ckpt"


def prepare_run_dir(run_dir, rerun):
    """Create the run folder, or archive the previous attempt. Returns (attempt, archived folder or None).

    On --rerun the old checkpoint and any staging folder are deleted; when a
    checkpoint was deleted the archived run.json is rewritten with
    checkpoint.kept false and deleted_by_rerun true.
    """
    if not rerun:
        run_dir.mkdir(parents=True)
        return 1, None
    k = 1
    while (run_dir / "attempts" / str(k)).exists():
        k += 1
    dest = run_dir / "attempts" / str(k)
    dest.mkdir(parents=True)
    for name in SMALL_FILES:
        if (run_dir / name).exists():
            os.replace(run_dir / name, dest / name)
    if (run_dir / STAGING).exists():
        shutil.rmtree(run_dir / STAGING)
    if (run_dir / "checkpoint").exists():
        shutil.rmtree(run_dir / "checkpoint")
        try:
            prev = json.loads((dest / "run.json").read_text(encoding="ascii"))
        except (OSError, ValueError):
            prev = None
        if isinstance(prev, dict) and isinstance(prev.get("checkpoint"), dict):
            prev["checkpoint"].update(kept=False, deleted_by_rerun=True)
            write_json(dest / "run.json", prev)
    return k + 1, dest


class RunFiles:
    def __init__(self, run_dir):
        self.dir = run_dir
        self.log_f = open(run_dir / "log.txt", "w", encoding="ascii", newline="\n")
        self.train_f = open(run_dir / "train_log.csv", "w", encoding="ascii", newline="\n")
        self.train_f.write(TRAIN_LOG_HEADER + "\n")
        self.pred_f = open(run_dir / "predictions_validation.csv", "w", encoding="ascii", newline="\n")
        self.pred_f.write(PRED_HEADER + "\n")

    def log(self, msg):
        say(msg)
        self.log_f.write(_ascii(msg) + "\n")
        self.log_f.flush()

    def update(self, r):
        self.train_f.write(f"{r['update']},{r['epoch']},{r['examples']},{r['lr']!r},{r['loss']!r},"
                           f"{r['grad_norm']!r},{r['seconds']:.6f}\n")
        self.train_f.flush()

    def predictions(self, step, sids, labels, in_clean, margins, scores, preds, threshold, condition):
        for i, sid in enumerate(sids):
            self.pred_f.write(f"{step},{sid},{labels[i]},{int(in_clean[i])},{eval_lib.fmt9(margins[i])},"
                              f"{eval_lib.fmt9(scores[i])},{preds[i]},{threshold!r},{condition}\n")
        self.pred_f.flush()

    def close(self):
        for f in (self.log_f, self.train_f, self.pred_f):
            f.close()


def nvidia_driver():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, timeout=30)
        return (r.stdout.decode("ascii", "replace").strip() or None) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def env_info(device):
    import huggingface_hub
    import safetensors
    import sklearn
    import tokenizers
    import transformers
    props = torch.cuda.get_device_properties(device)
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
            "safetensors": safetensors.__version__, "huggingface_hub": huggingface_hub.__version__,
            "scikit_learn": sklearn.__version__,
            "gpu": {"name": props.name, "capability": list(torch.cuda.get_device_capability(device)),
                    "total_memory_bytes": props.total_memory, "driver": nvidia_driver()},
            "requirements_sha256": data_lists.sha256_bytes((ROOT / "requirements.txt").read_bytes())}


def code_record(root=ROOT):
    """Commit, code SHA-256s and git state; paths are named only for the code, config and list files
    (other modified or untracked paths outside runs/ are only counted)."""
    st = data_lists.git_status(root, untracked=True)
    named = set(CODE_FILES) | {CONFIG_REL, data_lists.REVIEW} | {
        f"data/processed/{n}" for n in data_lists.LIST_FILES + (data_lists.META,)}
    rest = None if st is None else [(xy, p) for xy, p in st if not p.startswith("runs/")]
    return {"git_commit": data_lists.git_head(root),
            "git_dirty": None if rest is None else [{"status": xy, "path": p} for xy, p in rest if p in named],
            "git_dirty_other_count": None if rest is None else sum(p not in named for _, p in rest),
            "files_sha256_lf": {p: data_lists.sha256_lf(root / p) for p in CODE_FILES}}


def failure_info(exc, progress):
    frames = traceback.extract_tb(exc.__traceback__)
    ours = [f for f in frames if Path(f.filename).name in {Path(p).name for p in CODE_FILES}]
    info = {"phase": progress.get("phase"), "update": progress.get("update"),
            "exception": build_lists.exc_name(exc),
            "where": f"{Path(ours[-1].filename).name}:{ours[-1].lineno}" if ours else None,
            "deepest": f"{Path(frames[-1].filename).name}:{frames[-1].lineno}" if frames else None}
    msg = str(exc)
    if isinstance(exc, (NonFiniteError, ReloadCheckError)) or "deterministic" in msg:
        info["message"] = _ascii(msg)[:500]
    return info


def _read_csv(path):
    lines = path.read_text(encoding="ascii").rstrip("\n").split("\n")
    head = lines[0].split(",")
    return [dict(zip(head, ln.split(","))) for ln in lines[1:]]


# Records that must be equal for a same-seed difference to mean anything (README 4.2).
RERUN_SAME = (("code files SHA-256", ("code", "files_sha256_lf")), ("config SHA-256", ("config", "sha256")),
              ("list files SHA-256", ("lists", "files_sha256")),
              ("requirements SHA-256", ("env", "requirements_sha256")), ("torch", ("env", "torch")),
              ("CUDA", ("env", "cuda")), ("GPU", ("env", "gpu", "name")))
# Condition C only (absent from A and smoke records, so not part of RERUN_SAME, which other scripts read).
RERUN_SAME_AUG = ("augmentation files SHA-256", ("data", "augmentation", "files_sha256"))


def _get(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _equal_or_none(a, b):
    return None if a is None or b is None else a == b


def rerun_diff(prev_dir, record, log_rows, pred_path):
    """Same-seed difference against the archived attempt (README 4.2).

    Comparable only when the previous attempt completed and every RERUN_SAME
    value is recorded and equal, and every update row and prediction row of this
    attempt has a counterpart (the previous files have the same row counts).
    """
    k = int(prev_dir.name)
    try:
        prev = json.loads((prev_dir / "run.json").read_text(encoding="ascii"))
    except (OSError, ValueError):
        return {"previous_attempt": k, "comparable": False, "reason": "previous run.json unreadable"}
    out = {"previous_attempt": k, "previous_status": prev.get("status")}
    if prev.get("status") != "completed":
        return {**out, "comparable": False, "reason": "the previous attempt did not complete"}
    differ = [name for name, path in RERUN_SAME
              if _get(prev, path) is None or _get(prev, path) != _get(record, path)]
    if _get(prev, RERUN_SAME_AUG[1]) != _get(record, RERUN_SAME_AUG[1]):  # both None for A and smoke
        differ.append(RERUN_SAME_AUG[0])
    if differ:
        return {**out, "comparable": False,
                "reason": f"differs from the previous attempt or not recorded there: {', '.join(differ)}"}
    prev_rows = _read_csv(prev_dir / "train_log.csv") if (prev_dir / "train_log.csv").exists() else []
    prev_log = {int(r["update"]): r for r in prev_rows}
    first, max_loss, n = None, 0.0, 0
    for r in log_rows:
        p = prev_log.get(r["update"])
        if p is None:
            continue
        n += 1
        d = abs(float(p["loss"]) - r["loss"])
        max_loss = max(max_loss, d)
        if first is None and (float(p["loss"]) != r["loss"] or float(p["grad_norm"]) != r["grad_norm"]):
            first = r["update"]
    flips, max_margin, compared = 0, 0.0, 0
    old_rows = _read_csv(prev_dir / "predictions_validation.csv") \
        if (prev_dir / "predictions_validation.csv").exists() else []
    old = {(r["step"], r["sentence_id"]): r for r in old_rows}
    cur_rows = _read_csv(pred_path)
    for r in cur_rows:
        o = old.get((r["step"], r["sentence_id"]))
        if o is None:
            continue
        compared += 1
        flips += o["prediction"] != r["prediction"]
        max_margin = max(max_margin, abs(float(o["margin"]) - float(r["margin"])))
    if not (n == len(log_rows) == len(prev_rows) and compared == len(cur_rows) == len(old_rows)):
        return {**out, "comparable": False, "updates_compared": n, "predictions_compared": compared,
                "reason": f"incomplete overlap: updates {n} of {len(log_rows)} (previous {len(prev_rows)}), "
                          f"prediction rows {compared} of {len(cur_rows)} (previous {len(old_rows)})"}
    return {**out, "comparable": True, "updates_compared": n, "first_differing_update": first,
            "max_abs_loss_diff": max_loss,
            "selected_step_equal": _equal_or_none(_get(prev, ("selection", "selected_step")),
                                                  _get(record, ("selection", "selected_step"))),
            "predictions_compared": compared, "prediction_flips": flips, "max_abs_margin_diff": max_margin,
            "checkpoint_sha256_equal": _equal_or_none(_get(prev, ("checkpoint", "sha256")),
                                                      _get(record, ("checkpoint", "sha256")))}


def smoke_block(cfg, res, peak_alloc, reload, load_s, n_total):
    s = cfg["smoke"]
    losses = [r["loss"] for r in res["log"]]
    first, last = statistics.fmean(losses[:20]), statistics.fmean(losses[-20:])
    finite = all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in res["log"])
    t = statistics.median(r["seconds"] for r in res["log"][s["timing_skip"]:])
    t_eval, t_save = max(res["eval_s"]), max(res["save_s"])
    est = n_total * t + 4 * t_eval + load_s + 3 * t_save + reload["seconds"]
    crit = {"finite": finite, "loss_first20_mean": first, "loss_last20_mean": last, "loss_decreased": last < first,
            "peak_memory_allocated_bytes": peak_alloc, "memory_limit_bytes": s["memory_limit_bytes"],
            "memory_ok": peak_alloc <= s["memory_limit_bytes"], "reload_ok": reload["passed"]}
    passed = finite and crit["loss_decreased"] and crit["memory_ok"] and crit["reload_ok"]
    return {"passed": passed, "criteria": crit,
            "timing": {"update_median_s": t, "timed_updates": [s["timing_skip"] + 1, len(losses)],
                       "eval_s": t_eval, "load_s": load_s, "save_s": t_save, "reload_s": reload["seconds"],
                       "run_estimate_s": est, "formula": T_FORMULA}}


# ---------------------------------------------------------------- run

def run(args, cfg, cfg_bytes, ctx):
    key, run_id, run_dir = args.model, ctx["run_id"], ctx["run_dir"]
    smoke, cond = ctx["stage"] == "s1", ctx["condition"]
    repo, rev = model_specs.MODELS[key]
    fname, wsha, wsize = model_specs.WEIGHTS[key]
    record = {
        "schema": 1, "run_id": run_id, "stage": ctx["stage"], "model_key": key, "condition": cond,
        "seed": ctx["seed"], "attempt": None, "status": "running", "started_utc": _utc(), "finished_utc": None,
        "code": None, "config": {"path": CONFIG_REL, "sha256": data_lists.sha256_bytes(cfg_bytes),
                                 "copy": f"runs/{run_id}/config.json"},
        "budget": {"update_size": cfg["micro_batch"] * cfg["grad_accum"], "micro_batch": cfg["micro_batch"],
                   "grad_accum": cfg["grad_accum"], "epochs": cfg["epochs"], "N": ctx["n_total"],
                   "eval_steps": ctx["eval_at"], "updates_run": ctx["n_run"]},
        "lists": ctx["lists"],
        "model": {"key": key, "repo": repo, "revision": rev,
                  "weights": {"file": fname, "sha256": wsha, "size": wsize},
                  "attn_implementation": cfg["attn_implementation"], "precision": cfg["precision"]},
        "checkpoint": None, "selection": None, "resources": None, "failure": None,
    }
    lock = lock_path(run_dir)
    run_dir.parent.mkdir(exist_ok=True)
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        say(f"FAIL runs/{run_id}.lock appeared after the preflight: another run of {run_id} is in progress; "
            "nothing created")
        return 1
    files, prev_dir, ckpt_dir, checkpoint_ok = None, None, None, False
    progress = {"phase": "setup", "update": None}
    nondet = []
    orig_show = warnings.showwarning

    def show(message, category, filename, lineno, file=None, line=None):
        if "deterministic" in str(message) and len(nondet) < 20:
            nondet.append(_ascii(message)[:300])
        orig_show(message, category, filename, lineno, file, line)

    warnings.showwarning = show
    status, code, wall0 = "failed", 1, time.perf_counter()
    try:
        os.write(lock_fd, f"pid {os.getpid()} started {record['started_utc']}\n".encode("ascii"))
        record["attempt"], prev_dir = prepare_run_dir(run_dir, args.rerun)
        (run_dir / "config.json").write_bytes(cfg_bytes)
        files = RunFiles(run_dir)
        record["code"] = code_record()
        write_json(run_dir / "run.json", record)
        files.log(f"run {run_id} attempt {record['attempt']}: lists {ctx['lists']['mode']} ({ctx['lists']['dir']}"
                  f", git tracked check {ctx['lists']['git_tracked_checked']}), N {ctx['n_total']}, updates "
                  f"{ctx['n_run']}, eval at {ctx['eval_at']}")
        if smoke:
            ckpt_dir = smoke_ckpt_dir(run_id)
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)  # left by a killed smoke run
        else:
            ckpt_dir = run_dir / STAGING
        progress["phase"] = "data"
        td = data_lists.load_training_data(ctx["list_dir"], data_lists.load_rows())
        train, val = td["train"], td["validation"]
        if budget(len(train), cfg["epochs"]) != ctx["n_total"]:
            raise RuntimeError("train_clean rows changed after preflight")
        # Condition C: read after the budget recheck, so N and the eval steps come from train_clean alone
        # (README IMP-02), and before the data record, so a refused set leaves no data block.
        aug = None
        if cond == "C":
            try:
                aug = load_augmentation(ctx["aug_dir"], ctx["augmentation"]["files_sha256"])
            except data_lists.ListError as exc:
                for e in exc.errors:  # IDs, file names and counts only
                    files.log(f"FAIL {e}")
                raise
        record["data"] = {"train_rows": len(train), "validation_rows": len(val),
                          "validation_clean_rows": sum(r[3] for r in val)}
        if aug is not None:
            counts = label_counts(aug)
            record["data"].update(train_rows=len(train) + len(aug), train_clean_rows=len(train),
                                  augmentation_rows=len(aug),
                                  augmentation={**ctx["augmentation"], "label_counts": counts})
            files.log(f"augmentation: {len(aug)} rows of {AUG_DIR_REL}/{AUG_LIST} (label 1: {counts['1']}, label "
                      f"0: {counts['0']}) appended after {len(train)} train_clean rows; N stays {ctx['n_total']}")
            train = train + aug  # train_clean rows first, then the augmentation rows in train_aug.csv order
        del aug
        progress["phase"] = "setup"
        record["determinism"] = apply_determinism(cfg)
        device = torch.device("cuda")
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        record["env"] = env_info(device)
        tok = model_specs.load_tokenizer(key)
        record["model"]["tokenizer_fingerprint"] = model_specs.check_tokenizer(tok, key)
        pad_id = tok.pad_token_id
        train_ids, st_train = eval_lib.encode(tok, [r[2] for r in train], cfg["max_length"])
        val_ids, st_val = eval_lib.encode(tok, [r[2] for r in val], cfg["max_length"])
        record["data"]["truncation"] = {"train": st_train, "validation": st_val}
        data = {"train_ids": train_ids, "train_labels": [r[1] for r in train], "val_ids": val_ids,
                "val_labels": [r[1] for r in val], "val_in_clean": [r[3] for r in val]}
        val_sids = [r[0] for r in val]
        del td, train, val
        write_json(run_dir / "run.json", record)
        files.log(f"data: train {len(train_ids)} rows, validation {len(val_ids)} rows "
                  f"({record['data']['validation_clean_rows']} in_clean); tokens max train "
                  f"{st_train['max_tokens']} validation {st_val['max_tokens']}, over {cfg['max_length']}: "
                  f"{st_train['rows_over_max_length']} / {st_val['rows_over_max_length']}")

        progress["phase"] = "load"
        seed_everything(ctx["seed"])
        _sync(device)
        t0 = time.perf_counter()
        model = model_specs.load_model(key, attn_implementation=cfg["attn_implementation"])
        model.to(device)
        _sync(device)
        load_s = time.perf_counter() - t0
        record["model"]["parameters"] = sum(p.numel() for p in model.parameters())
        files.log(f"model {repo}@{rev[:7]} loaded in {load_s:.1f}s ({record['model']['parameters']} parameters)")

        metrics_doc = {"run_id": run_id, "threshold": cfg["threshold"], "eval_steps": ctx["eval_at"],
                       "selection_rule": "validation_clean F1 max, then lower FPR, then earlier step "
                                         "(exact fractions)", "evals": [], "selected_step": None}

        def on_eval(rec, margins, scores, preds):
            files.predictions(rec["step"], val_sids, data["val_labels"], data["val_in_clean"], margins.tolist(),
                              scores.tolist(), preds.tolist(), cfg["threshold"], cond)
            metrics_doc["evals"].append(rec)
            write_json(run_dir / "metrics.json", metrics_doc)

        res = fit(model, key, data, cfg, seed=ctx["seed"], n_total=ctx["n_total"], n_run=ctx["n_run"],
                  eval_at=ctx["eval_at"], device=device, pad_id=pad_id, ckpt_dir=ckpt_dir, run_id=run_id,
                  log=files.log, on_update=files.update, on_eval=on_eval, progress=progress)
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_res = torch.cuda.max_memory_reserved(device)
        best = res["best"]
        metrics_doc["selected_step"] = best["step"]
        write_json(run_dir / "metrics.json", metrics_doc)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        progress.update(phase="reload", update=None)
        ckpt_file = ckpt_dir / model_specs.CHECKPOINT_FILE
        reload = reload_check(key, ckpt_file, best, data, cfg, device, pad_id)
        files.log(f"reload check: state_dict equal {reload['state_dict_equal']}, prediction flips "
                  f"{reload['prediction_flips']}, max |margin diff| {reload['max_abs_margin_diff']:.3g}, "
                  f"{reload['seconds']:.1f}s")
        upd = [r["seconds"] for r in res["log"]]
        record["optimizer"] = res["optimizer"]
        record["selection"] = {"selected_step": best["step"], "saved_steps": res["saved"]}
        record["checkpoint"] = {"kept": not smoke, "path": None if smoke else
                                f"runs/{run_id}/checkpoint/{model_specs.CHECKPOINT_FILE}",
                                "sha256": best["sha256"], "size": best["size"], "step": best["step"],
                                "reload_check": reload}
        record["resources"] = {"peak_memory_allocated_bytes": peak_alloc, "peak_memory_reserved_bytes": peak_res,
                               "memory_window": MEMORY_WINDOW, "time_window": TIME_WINDOW,
                               "update_seconds_total": sum(upd), "update_seconds_median": statistics.median(upd),
                               "load_s": load_s, "eval_s": res["eval_s"], "save_s": res["save_s"],
                               "reload_s": reload["seconds"]}
        files.log(f"peak memory allocated {peak_alloc} B, reserved {peak_res} B")
        if not reload["passed"]:
            raise ReloadCheckError(f"state_dict_equal {reload['state_dict_equal']}, "
                                   f"prediction_flips {reload['prediction_flips']}")
        if not smoke:
            os.replace(ckpt_dir, run_dir / "checkpoint")
        checkpoint_ok = True
        code = 0
        if smoke:
            record["smoke"] = smoke_block(cfg, res, peak_alloc, reload, load_s, ctx["n_total"])
            c, t = record["smoke"]["criteria"], record["smoke"]["timing"]
            files.log(f"smoke: loss mean updates 1-20 {c['loss_first20_mean']:.4f}, "
                      f"{len(upd) - 19}-{len(upd)} {c['loss_last20_mean']:.4f} (decreased {c['loss_decreased']}); "
                      f"finite {c['finite']}; peak allocated {peak_alloc / 1e9:.3f} GB <= "
                      f"{c['memory_limit_bytes'] / 1e9:.1f} GB {c['memory_ok']}; reload ok {c['reload_ok']}")
            files.log(f"smoke timing: median {t['update_median_s']:.4f} s/update over updates "
                      f"{t['timed_updates'][0]}-{t['timed_updates'][1]}; eval {t['eval_s']:.1f}s, load "
                      f"{t['load_s']:.1f}s, save {t['save_s']:.1f}s, reload {t['reload_s']:.1f}s; one run "
                      f"T = {t['run_estimate_s'] / 3600:.2f} h ({T_FORMULA})")
            files.log("SMOKE PASSED" if record["smoke"]["passed"] else "SMOKE FAILED")
            code = 0 if record["smoke"]["passed"] else 1
        status = "completed"
        if prev_dir is not None:
            # A diagnostic: its failure neither changes the status nor touches the checkpoint.
            try:
                files.pred_f.flush()
                record["rerun"] = rerun_diff(prev_dir, record, res["log"], run_dir / "predictions_validation.csv")
            except Exception as exc:
                record["rerun"] = {"previous_attempt": int(prev_dir.name), "comparable": False,
                                   "reason": _exc_brief(exc)}
            files.log(f"rerun vs attempt {prev_dir.name}: " + ", ".join(
                f"{k} {v}" for k, v in record["rerun"].items() if k != "previous_attempt"))
    except KeyboardInterrupt as exc:
        status, code = "interrupted", 1
        record["failure"] = failure_info(exc, progress)
    except Exception as exc:
        status, code = "failed", 1
        record["failure"] = failure_info(exc, progress)
        sys.stderr.write(_ascii("".join(traceback.format_exception(exc))))
    finally:
        try:
            warnings.showwarning = orig_show
            if ckpt_dir is not None and (smoke or not checkpoint_ok):
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                if not smoke:
                    shutil.rmtree(run_dir / "checkpoint", ignore_errors=True)
                if record.get("checkpoint"):
                    record["checkpoint"].update(kept=False, path=None, deleted=True)
            record.setdefault("determinism", {})["nondeterministic_warnings"] = nondet
            record["status"] = status
            record["finished_utc"] = _utc()
            record["wall_s"] = time.perf_counter() - wall0
            log = files.log if files else say
            if record["failure"]:
                log(f"FAILED: status {status}, " + ", ".join(f"{k} {v}" for k, v in record["failure"].items()))
            if run_dir.is_dir():
                write_json(run_dir / "run.json", record)
            if files:
                files.close()
        finally:
            os.close(lock_fd)
            lock.unlink(missing_ok=True)
    return code


# ---------------------------------------------------------------- smoke summary

def smoke_summary(root=ROOT):
    rows, missing, unreadable = [], [], []
    for key in model_specs.MODELS:
        p = root / "runs" / f"s1-{key}-smoke-s0" / "run.json"
        if not p.exists():
            missing.append(key)
            continue
        try:
            rows.append((key, json.loads(p.read_text(encoding="ascii"))))
        except (OSError, ValueError) as exc:
            unreadable.append(f"{key} ({build_lists.exc_name(exc)})")
    say(f"{'model':8} {'status':11} {'pass':5} {'loss1-20':>9} {'loss81-100':>10} {'alloc_GB':>8} {'resv_GB':>8} "
        f"{'t/upd_s':>8} {'eval_s':>7} {'load_s':>7} {'save_s':>7} {'reload_s':>8} {'T_h':>6}  rerun")
    T = {}
    for key, r in rows:
        s, res = r.get("smoke") or {}, r.get("resources") or {}
        c, t = s.get("criteria") or {}, s.get("timing") or {}
        rr = r.get("rerun")
        rerun = "-" if rr is None else ("not comparable" if not rr.get("comparable") else
                                        f"first diff update {rr['first_differing_update']}, max loss diff "
                                        f"{rr['max_abs_loss_diff']:.3g}, flips {rr['prediction_flips']}")
        if t:
            T[key] = t["run_estimate_s"]
        say(f"{key:8} {r.get('status', '?'):11} {str(s.get('passed', '-')):5} "
            f"{c.get('loss_first20_mean', float('nan')):9.4f} {c.get('loss_last20_mean', float('nan')):10.4f} "
            f"{res.get('peak_memory_allocated_bytes', float('nan')) / 1e9:8.3f} "
            f"{res.get('peak_memory_reserved_bytes', float('nan')) / 1e9:8.3f} "
            f"{t.get('update_median_s', float('nan')):8.4f} {t.get('eval_s', float('nan')):7.1f} "
            f"{t.get('load_s', float('nan')):7.1f} {t.get('save_s', float('nan')):7.1f} "
            f"{t.get('reload_s', float('nan')):8.1f} {t.get('run_estimate_s', float('nan')) / 3600:6.2f}  {rerun}")
    say(f"one run: {T_FORMULA}")
    if missing:
        say(f"missing smoke runs: {', '.join(missing)}; 12-run estimate needs all three")
    if unreadable:
        say(f"unreadable run.json: {', '.join(unreadable)}")
    if len(T) == len(model_specs.MODELS):
        base = sum(3 * v for v in T.values())
        lo, hi = base + 3 * min(T.values()), base + 3 * max(T.values())
        say(f"12 runs: sum 3*T_m = {base / 3600:.2f} h (9 stage-2 runs) + 3*[min, max] T_m "
            f"= [{lo / 3600:.2f}, {hi / 3600:.2f}] h (estimate; excludes data preparation and preflight)")
    ok = not missing and not unreadable and all((r.get("smoke") or {}).get("passed") for _, r in rows)
    return 0 if ok else 1


# ---------------------------------------------------------------- main

def parse_args(argv):
    p = argparse.ArgumentParser(description="Fine-tune one model (README 4.2); settings from configs/train.json.")
    p.add_argument("--model", choices=sorted(model_specs.MODELS))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="stage-1 smoke run (seed and updates from the config)")
    g.add_argument("--seed", type=int, help="stage-2 run (condition A) or stage-4 run (condition C) with this "
                                            "seed (must be in the config seeds)")
    g.add_argument("--smoke-summary", action="store_true", help="summarise runs/s1-*-smoke-s0/run.json")
    p.add_argument("--condition", choices=CONDITIONS, default="A",
                   help="A (default): train_clean, runs/s2-<model>-A-s<seed>; C: train_clean + the augmentation "
                        f"rows of {AUG_DIR_REL}/{AUG_LIST} with the same N, runs/s4-<model>-C-s<seed> (not with "
                        "--smoke)")
    p.add_argument("--lists", choices=("official", "trial"), default="official",
                   help="common lists (trial only with --smoke)")
    p.add_argument("--rerun", action="store_true", help="archive the previous attempt and run again")
    p.add_argument("--preflight-only", action="store_true",
                   help="run the preflight of this run, create nothing, exit 0 if it would start")
    args = p.parse_args(argv)
    if args.smoke_summary and (args.model or args.rerun or args.lists != "official" or args.preflight_only
                               or args.condition != "A"):
        p.error("--smoke-summary takes no other option")
    if not args.smoke_summary and not args.model:
        p.error("--model is required")
    if args.smoke and args.condition != "A":
        p.error(f"--condition {args.condition} is not allowed with --smoke")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.smoke_summary:
        return smoke_summary()
    cfg_bytes = (ROOT / CONFIG_REL).read_bytes()
    cfg = json.loads(cfg_bytes)
    errors, ctx = preflight(args, cfg, ROOT)
    if errors:
        for e in errors:
            say("FAIL " + e)
        say(f"preflight failed: {len(errors)} error(s); nothing created")
        return 1
    if args.preflight_only:
        aug = (f", augmentation {sum(ctx['augmentation']['label_counts'].values())} rows"
               if ctx["condition"] == "C" else "")
        say(f"preflight passed: {ctx['run_id']} would start (lists {ctx['lists']['mode']}, git tracked check "
            f"{ctx['lists']['git_tracked_checked']}, N {ctx['n_total']}, updates {ctx['n_run']}, eval at "
            f"{ctx['eval_at']}{aug}); --preflight-only: nothing created")
        return 0
    return run(args, cfg, cfg_bytes, ctx)


if __name__ == "__main__":
    sys.exit(main())
