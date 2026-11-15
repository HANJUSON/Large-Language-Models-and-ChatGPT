"""Encoding, prediction, metrics and checkpoint selection shared by training and evaluation.

Rules (README 4.2 settings, 6.1 metrics, 7.1 REPRO-02):
  - encode: the model's own tokenizer (model_specs.load_tokenizer), special
    tokens added, truncation to max_length; truncation statistics are taken
    before truncation (rows over max_length, max and median token count).
  - collate: dynamic right padding to the longest sequence in the batch with
    the tokenizer's pad id; inputs are input_ids and attention_mask only.
  - predict: eval mode, inference_mode, FP32 (autocast off), batches in list
    (sentence ID) order. margin = l1 - l0 and score = softmax(logits)[1] come
    from the same float32 logits; prediction = score >= threshold (0.5 -> 1).
    Stored values use 9 significant digits (round-trips float32).
  - metrics: TN/FP/FN/TP and n; precision, recall, FPR and F1 =
    2TP / (2TP + FP + FN) are None (N/A) when the denominator is 0; exact
    fractions kept as "a/b" strings; ROC-AUC on the margin (monotone in the
    score, no float32 saturation ties), None when only one label is present;
    a non-finite margin raises NonFiniteError.
  - selection_key: (F1, -FPR, -step) on validation_clean as Fractions, so
    ties compare exactly (higher F1, then lower FPR, then earlier step).
  - load_run_model: run folders only as plain direct children of runs/; the
    checkpoint path is always runs/<id>/checkpoint/model.safetensors.
"""
import json
import math
import statistics
from fractions import Fraction
from pathlib import Path

import torch

import model_specs

ROOT = Path(__file__).resolve().parents[1]


class NonFiniteError(RuntimeError):
    pass


def fmt9(x):
    return f"{x:.9g}"


def encode(tok, texts, max_length):
    """Return (list of token-id lists, truncation stats)."""
    full = tok(list(texts), add_special_tokens=True, truncation=False)["input_ids"]
    lengths = [len(x) for x in full]
    over = sum(n > max_length for n in lengths)
    ids = full
    if over:
        ids = tok(list(texts), add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
    if any(len(x) > max_length for x in ids):
        raise RuntimeError("encode: a sequence is longer than max_length after truncation")
    stats = {"rows": len(lengths), "max_length": max_length, "rows_over_max_length": over,
             "max_tokens": max(lengths), "median_tokens": statistics.median(lengths)}
    return ids, stats


def collate(seqs, pad_id):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return ids, mask


def predict(model, seqs, pad_id, batch_size, device):
    """Return (margins, scores) as float32 CPU tensors in input order."""
    was_training = model.training
    model.eval()
    margins, scores = [], []
    try:
        with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            for i in range(0, len(seqs), batch_size):
                ids, mask = collate(seqs[i:i + batch_size], pad_id)
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float()
                margins.append((logits[:, 1] - logits[:, 0]).cpu())
                scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    finally:
        if was_training:
            model.train()
    return torch.cat(margins), torch.cat(scores)


def predictions(scores, threshold):
    return (scores >= threshold).to(torch.int64)


def _ratio(a, b):
    return None if b == 0 else Fraction(a, b)


def metrics(labels, preds, margins=None):
    if margins is not None:
        bad = sum(not math.isfinite(float(m)) for m in margins)
        if bad:
            raise NonFiniteError(f"metrics: {bad} of {len(margins)} margins are not finite")
    labels, preds = [int(y) for y in labels], [int(p) for p in preds]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, preds))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, preds))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, preds))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, preds))
    out = {"n": len(labels), "n_pos": tp + fn, "n_neg": tn + fp, "tn": tn, "fp": fp, "fn": fn, "tp": tp}
    for name, v in (("precision", _ratio(tp, tp + fp)), ("recall", _ratio(tp, tp + fn)),
                    ("fpr", _ratio(fp, fp + tn)), ("f1", _ratio(2 * tp, 2 * tp + fp + fn))):
        out[name] = None if v is None else float(v)
        out[name + "_exact"] = None if v is None else f"{v.numerator}/{v.denominator}"
    out["roc_auc"] = None
    if margins is not None and out["n_pos"] and out["n_neg"]:
        from sklearn.metrics import roc_auc_score
        out["roc_auc"] = float(roc_auc_score(labels, [float(m) for m in margins]))
    return out


def selection_key(m, step):
    """(F1, -FPR, -step) as Fractions from a metrics dict of validation_clean."""
    f1, fpr = _ratio(2 * m["tp"], 2 * m["tp"] + m["fp"] + m["fn"]), _ratio(m["fp"], m["fp"] + m["tn"])
    if f1 is None or fpr is None:
        raise ValueError("selection needs both labels in validation_clean (F1 and FPR defined)")
    return (f1, -fpr, -Fraction(step))


def select(evals):
    """The eval dict ({"step", "validation_clean", ...}) with the largest selection key."""
    return max(evals, key=lambda e: selection_key(e["validation_clean"], e["step"]))


def run_dir_problem(run_dir, root=ROOT):
    """None when run_dir is a plain direct child of <root>/runs (not a symlink or junction), else the reason."""
    p = Path(run_dir)
    if p.is_symlink() or p.is_junction():
        return "is a symlink or junction"
    if p.resolve().parent != Path(root).resolve() / "runs":
        return "does not resolve to a direct child of runs/"
    return None


def load_run_model(run_dir, attn_implementation="eager", *, expected_sha256=None, expected_size=None):
    """(model, tokenizer, run record) of a stage-2 run with a kept checkpoint (CPU, eval mode).

    run_dir must be a plain direct child of runs/; the checkpoint is always
    run_dir/checkpoint/model.safetensors (run.json "path", if set, must name
    exactly that) and run.json must record its size. It is loaded only via
    model_specs.load_finetuned (size and SHA-256 checked before parsing); the
    tokenizer only via model_specs.load_tokenizer and its fingerprint must
    equal the run record. run.json can be edited, so later eval/demo code must
    pass expected_sha256 and expected_size from a reviewed record (freeze-1);
    when given, run.json must agree with them, so the file is checked against
    the pins.
    """
    why = run_dir_problem(run_dir, ROOT)
    if why:
        raise RuntimeError(f"load_run_model: the run folder {why}")
    rd = Path(run_dir).resolve()
    run = json.loads((rd / "run.json").read_text(encoding="utf-8"))
    key, ckpt = run["model"]["key"], run.get("checkpoint") or {}
    rel = f"runs/{rd.name}/checkpoint/{model_specs.CHECKPOINT_FILE}"
    if not ckpt.get("kept"):
        raise RuntimeError("load_run_model: the run has no kept checkpoint")
    if ckpt.get("path") is not None and ckpt["path"] != rel:
        raise RuntimeError(f"load_run_model: run.json checkpoint path is not {rel}")
    if not isinstance(ckpt.get("size"), int) or not isinstance(ckpt.get("sha256"), str):
        raise RuntimeError("load_run_model: run.json records no checkpoint size or sha256")
    for name, pin in (("sha256", expected_sha256), ("size", expected_size)):
        if pin is not None and pin != ckpt[name]:
            raise RuntimeError(f"load_run_model: run.json checkpoint {name} differs from the pinned value")
    model = model_specs.load_finetuned(key, rd / "checkpoint" / model_specs.CHECKPOINT_FILE, ckpt["sha256"],
                                       size=ckpt["size"], attn_implementation=attn_implementation)
    tok = model_specs.load_tokenizer(key)
    if model_specs.tokenizer_fingerprint(tok) != run["model"]["tokenizer_fingerprint"]:
        raise RuntimeError("load_run_model: tokenizer fingerprint differs from the run record")
    return model, tok, run
