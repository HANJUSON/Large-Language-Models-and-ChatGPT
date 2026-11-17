"""CPU tests for the training code (scripts/train.py, data_lists.py, eval_lib.py, model_specs checkpoints).

Offline, no pretrained weights, no dataset text: models are tiny and built from
config, lists and git repositories are synthetic temp copies.

Run from Git Bash at the repo root:
    PYTHONUTF8=1 .venv/Scripts/python.exe -m unittest tests/test_train_cpu.py -v
"""
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import torch  # noqa: E402
from transformers import (AutoModelForSequenceClassification, BertConfig, DebertaV2Config,  # noqa: E402
                          RobertaConfig)

import data_lists  # noqa: E402
import eval_lib  # noqa: E402
import model_specs  # noqa: E402
import train  # noqa: E402

CFG = json.loads((ROOT / "configs" / "train.json").read_bytes())
CPU = torch.device("cpu")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def tiny_config():
    return BertConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                      intermediate_size=64, max_position_embeddings=64, num_labels=2, pad_token_id=0)


def tiny_model(seed=0):
    train.seed_everything(seed)
    return AutoModelForSequenceClassification.from_config(tiny_config(), dtype=torch.float32,
                                                          attn_implementation="eager")


def synthetic(n, seed, clean_skip=()):
    """Token-id sequences whose label is visible in the tokens (5 for 1, 6 for 0)."""
    rng = random.Random(seed)
    ids, labels = [], []
    for i in range(n):
        y = i % 2
        body = [5 if y else 6] * rng.randint(1, 4) + [rng.randint(10, 60) for _ in range(rng.randint(3, 15))]
        rng.shuffle(body)
        ids.append([2] + body + [3])
        labels.append(y)
    return ids, labels, [i not in clean_skip for i in range(n)]


def offline_tiny():
    """Patches so load_finetuned builds the tiny config instead of the pinned one (no network, no weights)."""
    return (mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True),
            mock.patch("transformers.AutoConfig.from_pretrained", side_effect=lambda *a, **k: tiny_config()))


def reference_checkpoint_bytes(model, key, run_id, step):
    """The previous writer: safetensors.torch.save of an FP32 CPU copy, then __metadata__ keys sorted in place."""
    from safetensors.torch import save
    repo, rev = model_specs.MODELS[key]
    state = {k: t.detach().to("cpu", copy=True).contiguous() for k, t in model.state_dict().items()}
    data = save(state, metadata={"model_key": key, "repo": repo, "revision": rev, "run_id": str(run_id),
                                 "step": str(step)})
    n = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + n])
    meta = header.pop("__metadata__")
    new = json.dumps({"__metadata__": dict(sorted(meta.items())), **header}, separators=(",", ":"),
                     ensure_ascii=True).encode("ascii")
    assert len(new) <= n
    return data[:8] + new + b" " * (n - len(new)) + data[8 + n:]


class OddNames(torch.nn.Module):
    """Parameter names whose insertion order, case and digits differ from sorted order; a 0-dim tensor."""

    def __init__(self):
        super().__init__()
        for n in ("zeta", "Alpha", "a_10", "a_9", "b", "B2"):
            self.register_parameter(n, torch.nn.Parameter(torch.randn(3, 5)))
        self.scalar = torch.nn.Parameter(torch.tensor(1.5))
        self.sub = torch.nn.Linear(7, 3)


STANDIN_BUILD_LISTS = b"import sys\nsys.exit(0)\n"  # build_lists.py --check stand-in for temp repos (exit 0)


class TestBudgetAndPlan(unittest.TestCase):
    def test_budget(self):
        for rows in (11049, 11050, 11051):
            self.assertEqual(train.budget(rows, 3), 2073, rows)
        self.assertEqual(train.eval_steps(2073), [691, 1382, 2073])
        self.assertEqual(train.eval_steps(12), [4, 8, 12])

    def test_epoch_plan_full(self):
        m, n = 11050, 2073
        plan = train.epoch_plan(m, n, 42)
        self.assertEqual(len(plan), n)
        for e in (1, 2, 3):
            ups = [mb for ep, mb in plan if ep == e]
            self.assertEqual(len(ups), 691)
            sizes = [sum(len(b) for b in u) for u in ups]
            self.assertEqual(sizes, [16] * 690 + [10])
            self.assertEqual([len(b) for b in ups[0]], [4, 4, 4, 4])
            self.assertEqual([len(b) for b in ups[-1]], [4, 4, 2])
            idx = [i for u in ups for b in u for i in b]
            self.assertEqual(sorted(idx), list(range(m)))  # each epoch is a permutation
        firsts = [[i for b in mb for i in b] for ep, mb in plan if ep in (1, 2)][::691]
        self.assertNotEqual(firsts[0], firsts[1])
        self.assertEqual(train.epoch_plan(m, n, 42), plan)
        self.assertNotEqual(train.epoch_plan(m, n, 123)[0], plan[0])
        self.assertEqual(train.epoch_plan(m, 100, 42), plan[:100])  # smoke = prefix of the N schedule

    def test_epoch_plan_generic(self):
        plan = train.epoch_plan(40, 10, 0)  # 3 updates per epoch (16, 16, 8); stops mid-epoch 4
        self.assertEqual([ep for ep, _ in plan], [1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
        self.assertEqual([len(b) for b in plan[2][1]], [4, 4])
        plan = train.epoch_plan(11050, 691, 0, 16, 2)  # OOM fallback 2 x 8
        self.assertEqual([len(b) for b in plan[-1][1]], [2, 2, 2, 2, 2])

    def test_condition_c_plan_keeps_n(self):
        """11,050 train_clean + 1,700 augmentation rows: N and the eval steps still come from 11,050 (IMP-02)."""
        clean, aug = 11050, 1700
        n = train.budget(clean, 3)
        self.assertEqual(n, 2073)
        self.assertEqual(train.budget(clean + aug, 3), 2391)  # 3 * ceil(12750 / 16): what N must NOT become
        self.assertEqual(train.eval_steps(n), [691, 1382, 2073])
        plan = train.epoch_plan(clean + aug, n, 42)
        self.assertEqual(len(plan), 2073)
        # 12750 = 796 * 16 + 14 -> 797 updates per epoch; 2073 - 2 * 797 = 479 updates of epoch 3
        self.assertEqual([sum(ep == e for ep, _ in plan) for e in (1, 2, 3, 4)], [797, 797, 479, 0])
        for e in (1, 2):
            ups = [mb for ep, mb in plan if ep == e]
            self.assertEqual([sum(len(b) for b in u) for u in ups], [16] * 796 + [14])
            idx = [i for u in ups for b in u for i in b]
            self.assertEqual(sorted(idx), list(range(clean + aug)))  # clean and augmentation rows, once each
        last = [i for ep, mb in plan if ep == 3 for b in mb for i in b]
        self.assertEqual((len(last), len(set(last))), (479 * 16, 479 * 16))  # epoch 3 stops at N: 7,664 rows
        self.assertTrue(any(i >= clean for i in last) and any(i < clean for i in last))
        # the first 11,050 indices are the train_clean rows of condition A, but the permutation differs from A's
        self.assertNotEqual(train.epoch_plan(clean, n, 42)[0], plan[0])


class TestOptimizerAndSchedule(unittest.TestCase):
    def test_lr_schedule(self):
        n = 2073
        model = torch.nn.Linear(2, 2)
        opt, _ = train.make_optimizer(model, CFG["optimizer"])
        sched = train.make_scheduler(opt, n)
        lrs = []
        for _ in range(n):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        self.assertEqual(lrs[0], 2e-5)
        self.assertTrue(math.isclose(lrs[99], 2e-5 * 1974 / 2073, rel_tol=1e-12))
        for u in (1, 100, 691, 1382, 2073):
            self.assertTrue(math.isclose(lrs[u - 1], 2e-5 * (n - u + 1) / n, rel_tol=1e-12), u)
        self.assertEqual([g["lr"] for g in opt.param_groups], [0.0, 0.0])

    def test_weight_decay_groups(self):
        model = torch.nn.Sequential(torch.nn.Embedding(5, 4), torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
        opt, info = train.make_optimizer(model, CFG["optimizer"])
        decay, no_decay = opt.param_groups
        self.assertEqual(decay["weight_decay"], 0.01)
        self.assertEqual(no_decay["weight_decay"], 0.0)
        self.assertTrue(all(p.ndim >= 2 for p in decay["params"]))
        self.assertTrue(all(p.ndim < 2 for p in no_decay["params"]))
        self.assertEqual((info["decay_tensors"], info["no_decay_tensors"]), (2, 3))
        self.assertEqual((decay["betas"], decay["eps"]), ((0.9, 0.999), 1e-8))


class Stub(torch.nn.Module):
    """Dropout-free mean-of-embeddings classifier with the HF output shape (.logits)."""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(16, 2)
        self.bias = torch.nn.Parameter(torch.tensor([0.3, -0.2]))

    def forward(self, input_ids, attention_mask):
        m = attention_mask.unsqueeze(-1).to(self.emb.weight.dtype)
        return types.SimpleNamespace(logits=(self.emb(input_ids) * m).sum(1) / m.sum(1) + self.bias)


class TestLossNormalisation(unittest.TestCase):
    def test_update_loss_is_full_batch_mean(self):
        torch.manual_seed(0)
        stub = Stub()
        rng = random.Random(3)
        ids = [[rng.randint(1, 15) for _ in range(rng.randint(2, 9))] for _ in range(10)]
        labels = [rng.randint(0, 1) for _ in range(10)]
        ref = copy.deepcopy(stub).double()
        x, mask = eval_lib.collate(ids, 0)
        y = torch.tensor(labels)
        loss64 = torch.nn.functional.cross_entropy(ref(x, mask).logits, y)
        loss64.backward()
        gn64 = torch.sqrt(sum((p.grad ** 2).sum() for p in ref.parameters())).item()
        with torch.no_grad():
            lg = ref(x, mask).logits
            naive = sum(torch.nn.functional.cross_entropy(lg[i:j], y[i:j]).item() for i, j in ((0, 4), (4, 8), (8, 10))) / 3
        cfg = {**CFG, "precision": "fp32"}
        data = {"train_ids": ids, "train_labels": labels, "val_ids": [], "val_labels": [], "val_in_clean": []}
        res = train.fit(stub, "bert", data, cfg, seed=0, n_total=1, n_run=1, eval_at=[], device=CPU, pad_id=0,
                        ckpt_dir=None, run_id="t", log=lambda m: None)
        row = res["log"][0]
        self.assertEqual(row["examples"], 10)
        self.assertTrue(math.isclose(row["loss"], loss64.item(), rel_tol=1e-6))
        self.assertTrue(math.isclose(row["grad_norm"], gn64, rel_tol=1e-5))
        self.assertFalse(math.isclose(naive, loss64.item(), rel_tol=1e-4))  # mean of micro-batch means differs


class TestReferenceLoop(unittest.TestCase):
    """fit vs an independent float64 loop (hand-written AdamW): lr, loss, grad norm and all parameters after
    each of 3 updates. Deleting optimizer.step, scheduler.step or the per-update zero_grad fails this."""

    def test_three_updates_match_float64_reference(self):
        torch.manual_seed(0)
        stub = Stub()
        with torch.no_grad():
            stub.emb.weight.mul_(0.1)  # |p| < 0.5: float32 rounding stays far below one Adam step (~lr)
        ref = copy.deepcopy(stub).double()
        rng = random.Random(5)
        ids = [[rng.randint(1, 15) for _ in range(rng.randint(2, 9))] for _ in range(40)]
        labels = [rng.randint(0, 1) for _ in range(40)]
        data = {"train_ids": ids, "train_labels": labels, "val_ids": [], "val_labels": [], "val_in_clean": []}
        n, snaps = 3, []
        res = train.fit(stub, "bert", data, {**CFG, "precision": "fp32"}, seed=11, n_total=n, n_run=n, eval_at=[],
                        device=CPU, pad_id=0, ckpt_dir=None, run_id="t", log=lambda m: None,
                        on_update=lambda r: snaps.append({k: p.detach().double().clone()
                                                          for k, p in stub.named_parameters()}))
        o = CFG["optimizer"]
        b1, b2 = o["betas"]
        moments = {k: (torch.zeros_like(p), torch.zeros_like(p)) for k, p in ref.named_parameters()}
        before = {k: p.detach().clone() for k, p in ref.named_parameters()}
        plan = train.epoch_plan(len(ids), n, 11)
        self.assertEqual([sum(len(b) for b in mb) for _, mb in plan], [16, 16, 8])
        for u, (_, mbs) in enumerate(plan, start=1):
            idx = [i for b in mbs for i in b]
            x, mask = eval_lib.collate([ids[i] for i in idx], 0)
            ref.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(ref(x, mask).logits, torch.tensor([labels[i] for i in idx]))
            loss.backward()
            gn = math.sqrt(sum((p.grad ** 2).sum().item() for p in ref.parameters()))
            coef = min(1.0, CFG["grad_clip_norm"] / (gn + 1e-6))
            lr = 2e-5 * (n - u + 1) / n
            row = res["log"][u - 1]
            self.assertTrue(math.isclose(row["lr"], lr, rel_tol=1e-12), (u, row["lr"], lr))
            self.assertTrue(math.isclose(row["loss"], loss.item(), rel_tol=1e-6), u)
            self.assertTrue(math.isclose(row["grad_norm"], gn, rel_tol=1e-5), u)
            with torch.no_grad():
                for k, p in ref.named_parameters():
                    g = p.grad * coef
                    m, v = moments[k]
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    if p.ndim >= o["weight_decay_min_ndim"]:
                        p.mul_(1 - lr * o["weight_decay"])
                    p.sub_(lr * (m / (1 - b1 ** u)) / ((v / (1 - b2 ** u)).sqrt() + o["eps"]))
            for k, p in ref.named_parameters():
                self.assertLess((snaps[u - 1][k] - p).abs().max().item(), 5e-7, (u, k))
                self.assertGreater((p - before[k]).abs().max().item(), 1e-6, (u, k))  # the reference moved
                before[k] = p.detach().clone()
        self.assertEqual([r["lr"] for r in res["log"]], [2e-5 * (3 / 3), 2e-5 * (2 / 3), 2e-5 * (1 / 3)])


class TestSelection(unittest.TestCase):
    def test_improve_tie_improve_saves_twice(self):
        labels = [1, 1, 0, 0]

        def out(preds):  # (margins, scores) that give these predictions at threshold 0.5
            s = torch.tensor([0.9 if p else 0.1 for p in preds])
            return torch.log(s / (1 - s)), s

        # F1 2/3 FPR 0, then F1 2/3 FPR 0 at a later step (tie: not an improvement), then F1 1
        calls = [out([1, 0, 0, 0]), out([0, 1, 0, 0]), out([1, 1, 0, 0])]
        torch.manual_seed(0)
        rng = random.Random(1)
        ids = [[rng.randint(1, 15) for _ in range(4)] for _ in range(48)]
        data = {"train_ids": ids, "train_labels": [i % 2 for i in range(48)], "val_ids": [[1, 2]] * 4,
                "val_labels": labels, "val_in_clean": [True] * 4}
        tmp = Path(tempfile.mkdtemp())
        try:
            with mock.patch.object(eval_lib, "predict", side_effect=calls), \
                    mock.patch.object(model_specs, "save_checkpoint", wraps=model_specs.save_checkpoint) as spy:
                res = train.fit(Stub(), "bert", data, {**CFG, "precision": "fp32"}, seed=0, n_total=3, n_run=3,
                                eval_at=[1, 2, 3], device=CPU, pad_id=0, ckpt_dir=tmp / "ck", run_id="t-sel",
                                log=lambda m: None)
            self.assertEqual([e["validation_clean"]["f1_exact"] for e in res["evals"]], ["2/3", "2/3", "1/1"])
            self.assertEqual([e["improved"] for e in res["evals"]], [True, False, True])
            self.assertEqual(res["saved"], [1, 3])
            self.assertEqual([c.args[4] for c in spy.call_args_list], [1, 3])
            self.assertEqual(res["best"]["step"], 3)
            data_bytes = (tmp / "ck" / "model.safetensors").read_bytes()
            self.assertEqual(model_specs.checkpoint_metadata(data_bytes)["step"], "3")
            self.assertEqual(sha(data_bytes), res["best"]["sha256"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestOtherArchitectures(unittest.TestCase):
    """Tiny RoBERTa and DeBERTa-v2 from config: eval mid-run, then training continues in train mode."""

    def check(self, config, pad_id):
        train.seed_everything(0)
        model = AutoModelForSequenceClassification.from_config(config, dtype=torch.float32,
                                                               attn_implementation="eager")
        tr_ids, tr_y, _ = synthetic(40, 1)
        va_ids, va_y, va_clean = synthetic(12, 2)
        data = {"train_ids": tr_ids, "train_labels": tr_y, "val_ids": va_ids, "val_labels": va_y,
                "val_in_clean": va_clean}
        cfg = copy.deepcopy(CFG)
        cfg["optimizer"]["lr"] = 5e-4
        modes, snaps = [], []

        def on_update(row):
            modes.append(model.training)
            snaps.append({k: p.detach().clone() for k, p in model.named_parameters()})

        tmp = Path(tempfile.mkdtemp())
        try:
            res = train.fit(model, "bert", data, cfg, seed=3, n_total=6, n_run=6, eval_at=[3], device=CPU,
                            pad_id=pad_id, ckpt_dir=tmp, run_id="t-arch", log=lambda m: None, on_update=on_update)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual([e["step"] for e in res["evals"]], [3])
        self.assertEqual(modes, [True] * 6)  # predict() restored train mode after the eval
        self.assertTrue(all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in res["log"]))
        for u in (4, 5, 6):
            moved = max((snaps[u - 1][k] - snaps[u - 2][k]).abs().max().item() for k in snaps[0])
            self.assertGreater(moved, 0, u)

    def test_roberta(self):
        self.check(RobertaConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                 intermediate_size=64, max_position_embeddings=40, pad_token_id=1, num_labels=2), 1)

    def test_deberta_v2(self):
        self.check(DebertaV2Config(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                   intermediate_size=64, max_position_embeddings=64, type_vocab_size=0,
                                   relative_attention=True, position_buckets=16, max_relative_positions=-1,
                                   pos_att_type=["p2c", "c2p"], norm_rel_ebd="layer_norm", share_att_key=True,
                                   position_biased_input=False, pad_token_id=0, num_labels=2), 0)


class TestMetricsAndSelection(unittest.TestCase):
    def test_metrics(self):
        m = eval_lib.metrics([1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0], [3, 2, -1, 1, -2, -3, -4])
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"], m["n"]), (3, 1, 1, 2, 7))
        self.assertEqual(m["f1_exact"], "2/3")
        self.assertEqual(m["precision_exact"], "2/3")
        self.assertEqual(m["recall_exact"], "2/3")
        self.assertEqual(m["fpr_exact"], "1/4")
        self.assertAlmostEqual(m["roc_auc"], 11 / 12)

    def test_not_applicable(self):
        neg = eval_lib.metrics([0, 0, 0], [0, 0, 0], [1.0, 2.0, 3.0])
        self.assertIsNone(neg["precision"])
        self.assertIsNone(neg["recall"])
        self.assertIsNone(neg["f1"])
        self.assertEqual(neg["fpr"], 0.0)
        self.assertIsNone(neg["roc_auc"])
        pos = eval_lib.metrics([1, 1], [0, 0], [0.0, 1.0])
        self.assertEqual(pos["f1"], 0.0)  # positives present, none detected: 0, not N/A
        self.assertIsNone(pos["fpr"])
        self.assertIsNone(pos["precision"])
        self.assertIsNone(pos["roc_auc"])

    def test_threshold_and_format(self):
        s = torch.tensor([0.5, 0.49999997, 0.50000006, 1.0, 0.0], dtype=torch.float32)
        self.assertEqual(eval_lib.predictions(s, 0.5).tolist(), [1, 0, 1, 1, 0])
        for v in torch.randn(1000, dtype=torch.float32).tolist() + s.tolist():
            self.assertEqual(float(torch.tensor(float(eval_lib.fmt9(v)), dtype=torch.float32)), v)

    def test_selection_ties(self):
        def m(tp, fp, fn, tn):
            return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        evals = [{"step": 4, "validation_clean": m(2, 1, 1, 6)},   # F1 2/3, FPR 1/7
                 {"step": 8, "validation_clean": m(4, 2, 2, 12)},  # F1 4/6 = 2/3, FPR 1/7: tie -> earlier
                 {"step": 12, "validation_clean": m(2, 2, 0, 5)}]  # F1 2/3, FPR 2/7
        self.assertEqual(eval_lib.select(evals)["step"], 4)
        evals[1]["validation_clean"] = m(4, 1, 3, 13)            # F1 8/12 = 2/3, FPR 1/14: lower FPR wins
        self.assertEqual(eval_lib.select(evals)["step"], 8)
        self.assertEqual(eval_lib.selection_key(m(1, 0, 0, 1), 3), (Fraction(1), Fraction(0), Fraction(-3)))
        evals.append({"step": 13, "validation_clean": m(9, 0, 0, 9)})
        self.assertEqual(eval_lib.select(evals)["step"], 13)
        with self.assertRaises(ValueError):
            eval_lib.selection_key(m(0, 0, 0, 3), 1)


class TestTinyLoop(unittest.TestCase):
    def run_fit(self, ckpt_dir, data, cfg):
        model = tiny_model(0)
        res = train.fit(model, "bert", data, cfg, seed=7, n_total=train.budget(50, 3), n_run=12,
                        eval_at=train.eval_steps(12), device=CPU, pad_id=0, ckpt_dir=ckpt_dir, run_id="t-run",
                        log=lambda m: None)
        return res

    def test_loop(self):
        tr_ids, tr_y, _ = synthetic(50, 1)
        va_ids, va_y, va_clean = synthetic(30, 2, clean_skip=(3, 8))
        data = {"train_ids": tr_ids, "train_labels": tr_y, "val_ids": va_ids, "val_labels": va_y,
                "val_in_clean": va_clean}
        cfg = copy.deepcopy(CFG)
        cfg["optimizer"]["lr"] = 5e-4
        tmp = Path(tempfile.mkdtemp())
        try:
            res = self.run_fit(tmp / "a", data, cfg)
            self.assertEqual([r["examples"] for r in res["log"]], [16, 16, 16, 2] * 3)
            self.assertEqual([r["epoch"] for r in res["log"]], [1] * 4 + [2] * 4 + [3] * 4)
            self.assertEqual([e["step"] for e in res["evals"]], [4, 8, 12])
            for e in res["evals"]:
                self.assertEqual(e["validation"]["n"], 30)
                self.assertEqual(e["validation_clean"]["n"], 28)
            # which evals improve is data-dependent here; TestSelection pins improve -> tie -> improve
            self.assertEqual(res["saved"][0], 4)  # the first eval always improves
            self.assertEqual([e["step"] for e in res["evals"] if e["improved"]], res["saved"])
            self.assertEqual(res["best"]["step"], res["saved"][-1])
            self.assertEqual(os.listdir(tmp / "a"), ["model.safetensors"])
            data_bytes = (tmp / "a" / "model.safetensors").read_bytes()
            meta = model_specs.checkpoint_metadata(data_bytes)
            self.assertEqual(meta["step"], str(res["saved"][-1]))
            self.assertEqual(meta["model_key"], "bert")
            self.assertEqual(sha(data_bytes), res["best"]["sha256"])
            # same seed -> identical losses, grad norms, metrics and checkpoint bytes
            res2 = self.run_fit(tmp / "b", data, cfg)
            self.assertEqual([(r["loss"], r["grad_norm"]) for r in res["log"]],
                             [(r["loss"], r["grad_norm"]) for r in res2["log"]])
            self.assertEqual(res["evals"], res2["evals"])
            self.assertEqual(res["best"]["sha256"], res2["best"]["sha256"])
            # end-of-run reload check
            p1, p2 = offline_tiny()
            with p1, p2:
                rc = train.reload_check("bert", tmp / "a" / "model.safetensors", res["best"], data, cfg, CPU, 0)
            self.assertTrue(rc["state_dict_equal"])
            self.assertEqual(rc["prediction_flips"], 0)
            self.assertEqual(rc["max_abs_margin_diff"], 0.0)
            self.assertTrue(rc["passed"])
            # the reload check must fail when one tensor digest or one prediction differs
            bad_t = dict(res["best"], tensors=dict(res["best"]["tensors"]))
            first = sorted(bad_t["tensors"])[0]
            bad_t["tensors"][first] = ["tampered"]
            preds = res["best"]["preds"].clone()
            preds[0] = ~preds[0] if preds.dtype == torch.bool else 1 - preds[0]
            bad_p = dict(res["best"], preds=preds)
            with p1, p2:
                rt = train.reload_check("bert", tmp / "a" / "model.safetensors", bad_t, data, cfg, CPU, 0)
                rp = train.reload_check("bert", tmp / "a" / "model.safetensors", bad_p, data, cfg, CPU, 0)
            self.assertFalse(rt["state_dict_equal"])
            self.assertFalse(rt["passed"])
            self.assertTrue(rp["state_dict_equal"])
            self.assertEqual(rp["prediction_flips"], 1)
            self.assertFalse(rp["passed"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_appended_rows_keep_n_and_eval_steps(self):
        """Condition C shape: 50 rows + 30 appended rows, N still 3 * ceil(50 / 16) = 12 (README IMP-02)."""
        tr_ids, tr_y, _ = synthetic(50, 1)
        au_ids, au_y, _ = synthetic(30, 3)
        va_ids, va_y, va_clean = synthetic(30, 2, clean_skip=(3, 8))
        data = {"train_ids": tr_ids + au_ids, "train_labels": tr_y + au_y, "val_ids": va_ids, "val_labels": va_y,
                "val_in_clean": va_clean}
        cfg = copy.deepcopy(CFG)
        cfg["optimizer"]["lr"] = 5e-4
        tmp = Path(tempfile.mkdtemp())
        try:
            res = self.run_fit(tmp / "c", data, cfg)
            self.assertEqual(len(res["log"]), 12)  # not 3 * ceil(80 / 16) = 15
            self.assertEqual([r["epoch"] for r in res["log"]], [1] * 5 + [2] * 5 + [3] * 2)  # 80 / 16 = 5 per epoch
            self.assertEqual([r["examples"] for r in res["log"]], [16] * 12)
            self.assertEqual([e["step"] for e in res["evals"]], [4, 8, 12])  # the same eval steps as 50 rows
            for u in (1, 5, 12):  # the schedule decays over N = 12 updates: lr * (N - u + 1) / N
                self.assertTrue(math.isclose(res["log"][u - 1]["lr"], 5e-4 * (12 - u + 1) / 12, rel_tol=1e-12), u)
            used = {i for _, mbs in train.epoch_plan(80, 12, 7)[:5] for b in mbs for i in b}
            self.assertEqual(used, set(range(80)))  # epoch 1 covers the appended rows 50..79 too
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestCheckpointGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.model = tiny_model(0)
        self.info = model_specs.save_checkpoint(self.model, "bert", self.tmp / "ck", "r1", 5)
        self.path = self.tmp / "ck" / "model.safetensors"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def load(self, path=None, sha256=None, key="bert", size=None):
        p1, p2 = offline_tiny()
        with p1, p2:
            return model_specs.load_finetuned(key, path or self.path, sha256 or self.info["sha256"], size=size)

    def rewrite(self, state, meta=None):
        from safetensors.torch import save_file
        p = self.tmp / "other.safetensors"
        save_file(state, str(p), metadata=meta or {"model_key": "bert", "repo": model_specs.MODELS["bert"][0],
                                                  "revision": model_specs.MODELS["bert"][1]})
        return p, sha(p.read_bytes())

    def test_save_and_load(self):
        self.assertEqual(os.listdir(self.tmp / "ck"), ["model.safetensors"])
        self.assertEqual(self.info["size"], self.path.stat().st_size)
        self.assertEqual(self.info["sha256"], sha(self.path.read_bytes()))
        model = self.load(size=self.info["size"])
        saved = self.model.state_dict()
        self.assertEqual(set(model.state_dict()), set(saved))
        for k, t in model.state_dict().items():
            self.assertTrue(torch.equal(t, saved[k]), k)
        self.assertEqual({k: model_specs.tensor_digest(t) for k, t in model.state_dict().items()},
                         self.info["tensors"])
        with torch.no_grad():  # one changed element changes that tensor's digest only
            model.classifier.bias[0] += 1
        got = {k: model_specs.tensor_digest(t) for k, t in model.state_dict().items()}
        self.assertEqual([k for k in got if got[k] != self.info["tensors"][k]], ["classifier.bias"])

    def test_streaming_writer_matches_previous_writer(self):
        torch.manual_seed(1)
        for model in (self.model, OddNames()):
            for run_id, step in (("r1", 5), ("s1-bert-smoke-s0", 100), ("s2-roberta-A-s2026", 2073)):
                d = self.tmp / f"w-{type(model).__name__}-{step}"
                info = model_specs.save_checkpoint(model, "bert", d, run_id, step)
                got = (d / "model.safetensors").read_bytes()
                self.assertEqual(got, reference_checkpoint_bytes(model, "bert", run_id, step), (run_id, step))
                self.assertEqual((info["sha256"], info["size"]), (sha(got), len(got)))
                self.assertEqual(info["tensors"], {k: model_specs.tensor_digest(t)
                                                   for k, t in model.state_dict().items()})

    def test_sha_and_size_guard_before_parse(self):
        with mock.patch("safetensors.torch.load") as spy:
            with self.assertRaisesRegex(RuntimeError, "sha256"):
                self.load(sha256="0" * 64)
            with self.assertRaisesRegex(RuntimeError, "size"):
                self.load(size=self.info["size"] + 1)
            spy.assert_not_called()

    def test_offline_required(self):
        with mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", False):
            with self.assertRaisesRegex(RuntimeError, "HF_HUB_OFFLINE"):
                model_specs.load_finetuned("bert", self.path, self.info["sha256"])

    def test_metadata_key(self):
        with self.assertRaisesRegex(RuntimeError, "model_key"):
            self.load(key="roberta")

    def test_strict_keys_and_dtype(self):
        state = {k: t.detach().clone() for k, t in self.model.state_dict().items()}
        p, h = self.rewrite({**state, "extra.weight": torch.zeros(2)})
        with self.assertRaisesRegex(RuntimeError, "key set"):
            self.load(p, h)
        missing = dict(state)
        missing.pop("classifier.bias")
        p, h = self.rewrite(missing)
        with self.assertRaisesRegex(RuntimeError, "key set"):
            self.load(p, h)
        half = dict(state)
        half["classifier.bias"] = half["classifier.bias"].half()
        p, h = self.rewrite(half)
        with self.assertRaisesRegex(RuntimeError, "float32"):
            self.load(p, h)

    def test_folder_only_checkpoint(self):
        (self.tmp / "ck" / "config.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "other files"):
            model_specs.save_checkpoint(tiny_model(0), "bert", self.tmp / "ck", "r1", 6)


def make_link(target, link):
    """Directory junction on Windows (no privilege needed), symlink elsewhere."""
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


class TestLoadRunModel(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.rd = self.root / "runs" / "s2-bert-A-s42"
        self.info = model_specs.save_checkpoint(tiny_model(0), "bert", self.rd / "checkpoint", "s2-bert-A-s42", 4)
        self.write(path=f"runs/s2-bert-A-s42/checkpoint/{model_specs.CHECKPOINT_FILE}")
        self.patches = [mock.patch.object(eval_lib, "ROOT", self.root), *offline_tiny(),
                        mock.patch.object(model_specs, "load_tokenizer", return_value=object()),
                        mock.patch.object(model_specs, "tokenizer_fingerprint", return_value="fp")]

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self, **ckpt):
        rec = {"model": {"key": "bert", "tokenizer_fingerprint": "fp"},
               "checkpoint": {"kept": True, "sha256": self.info["sha256"], "size": self.info["size"], **ckpt}}
        (self.rd / "run.json").write_text(json.dumps(rec), encoding="ascii")

    def load(self, run_dir=None, **kw):
        with contextlib.ExitStack() as st:
            for p in self.patches:
                st.enter_context(p)
            return eval_lib.load_run_model(run_dir or self.rd, **kw)

    def test_loads_with_and_without_pins(self):
        model, _, run = self.load()
        self.assertEqual(run["checkpoint"]["size"], self.info["size"])
        self.assertFalse(model.training)
        self.load(expected_sha256=self.info["sha256"], expected_size=self.info["size"])

    def test_refusals(self):
        with mock.patch.object(model_specs, "load_finetuned") as spy:
            with self.assertRaisesRegex(RuntimeError, "pinned"):
                self.load(expected_sha256="0" * 64)
            with self.assertRaisesRegex(RuntimeError, "pinned"):
                self.load(expected_size=self.info["size"] + 1)
            self.write(path="runs/other/checkpoint/model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "path is not"):
                self.load()
            self.write(path=None, size=None)
            with self.assertRaisesRegex(RuntimeError, "no checkpoint size"):
                self.load()
            self.write(kept=False)
            with self.assertRaisesRegex(RuntimeError, "no kept checkpoint"):
                self.load()
            outside = self.root / "elsewhere" / "s2-bert-A-s42"
            shutil.copytree(self.rd, outside)
            with self.assertRaisesRegex(RuntimeError, "direct child"):
                self.load(outside)
            make_link(outside, self.root / "runs" / "linked")
            with self.assertRaisesRegex(RuntimeError, "symlink or junction"):
                self.load(self.root / "runs" / "linked")
            spy.assert_not_called()


class TestRecords(unittest.TestCase):
    def test_check_config(self):
        self.assertEqual(train.check_config(CFG), [])
        for k, v in (("padding", "max_length"), ("inputs", ["input_ids", "attention_mask", "token_type_ids"]),
                     ("schedule", "cosine"), ("epochs", 2), ("epochs", 3.0)):
            errors = train.check_config({**CFG, k: v})
            self.assertEqual(len(errors), 1, (k, v, errors))
            self.assertIn(k, errors[0])
        cfg = dict(CFG)
        del cfg["schedule"]
        self.assertIn("missing or malformed key", train.check_config(cfg)[0])

    def test_nvidia_driver(self):
        def fake(rc, out):
            return mock.patch.object(train.subprocess, "run", return_value=types.SimpleNamespace(returncode=rc,
                                                                                               stdout=out))
        with fake(0, b"610.88\r\n"):
            self.assertEqual(train.nvidia_driver(), "610.88")
        with fake(0, b" \n"):
            self.assertIsNone(train.nvidia_driver())
        with fake(9, b"610.88\n"):
            self.assertIsNone(train.nvidia_driver())
        with mock.patch.object(train.subprocess, "run", side_effect=OSError):
            self.assertIsNone(train.nvidia_driver())

    def test_non_finite_margins(self):
        self.assertIs(train.NonFiniteError, eval_lib.NonFiniteError)
        for bad in (float("nan"), float("inf")):
            with mock.patch("sklearn.metrics.roc_auc_score") as spy:
                with self.assertRaises(eval_lib.NonFiniteError):
                    eval_lib.metrics([1, 0, 0], [1, 0, 0], [1.0, bad, -1.0])
                spy.assert_not_called()
        with self.assertRaises(eval_lib.NonFiniteError):  # also with one label (no ROC-AUC)
            eval_lib.metrics([0, 0], [0, 0], torch.tensor([0.0, float("nan")]).tolist())

    def test_write_json_atomic(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            p = tmp / "run.json"
            train.write_json(p, {"a": 1})
            with mock.patch.object(train.os, "replace", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    train.write_json(p, {"a": 2})
            self.assertEqual(json.loads(p.read_text()), {"a": 1})  # the old record is intact
            train.write_json(p, {"a": 3})
            self.assertEqual((json.loads(p.read_text()), sorted(os.listdir(tmp))), ({"a": 3}, ["run.json"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_smoke_summary_unreadable_run_json(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "runs" / "s1-bert-smoke-s0").mkdir(parents=True)
            (tmp / "runs" / "s1-bert-smoke-s0" / "run.json").write_text("{broken", encoding="ascii")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(train.smoke_summary(tmp), 1)
            self.assertIn("unreadable run.json: bert (json.decoder.JSONDecodeError)", out.getvalue())
            self.assertIn("missing smoke runs: roberta, deberta", out.getvalue())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_smoke_ckpt_dir(self):
        a = train.smoke_ckpt_dir("s1-bert-smoke-s0", ROOT)
        self.assertEqual(a, train.smoke_ckpt_dir("s1-bert-smoke-s0", ROOT))
        self.assertEqual(a.parent, Path(tempfile.gettempdir()))
        self.assertTrue(a.name.endswith("-s1-bert-smoke-s0-ckpt"))
        self.assertNotEqual(a, train.smoke_ckpt_dir("s1-bert-smoke-s0", ROOT / "other"))
        self.assertNotEqual(a, train.smoke_ckpt_dir("s1-roberta-smoke-s0", ROOT))


class TestRerunDiff(unittest.TestCase):
    ENV = {"requirements_sha256": "r", "torch": "2.14.1", "cuda": "13.0", "gpu": {"name": "RTX"}}

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.prev, self.cur = self.tmp / "attempts" / "1", self.tmp / "cur"
        self.prev.mkdir(parents=True)
        self.cur.mkdir()
        self.rows = [{"update": u, "epoch": 1, "examples": 16, "lr": 2e-5, "loss": 1 / u, "grad_norm": 0.5,
                      "seconds": 0.1} for u in range(1, 6)]
        self.record = {"status": "completed", "code": {"files_sha256_lf": {"scripts/train.py": "c"}},
                       "config": {"sha256": "k"}, "lists": {"files_sha256": {"train_clean.csv": "l"}},
                       "env": copy.deepcopy(self.ENV), "selection": {"selected_step": 5},
                       "checkpoint": {"sha256": "s"}}
        self.files(self.cur, self.rows)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def files(self, d, rows, margins=(1.5, -2.0)):
        f = train.RunFiles(d)
        for r in rows:
            f.update(r)
        f.predictions(5, ["validation:000000", "validation:000001"], [1, 0], [True, False], list(margins),
                      [0.8, 0.1], [1, 0], 0.5, "smoke")
        f.close()

    def diff(self, prev_record=None, prev_rows=None, margins=(1.5, -2.0)):
        self.files(self.prev, self.rows if prev_rows is None else prev_rows, margins)
        train.write_json(self.prev / "run.json", self.record if prev_record is None else prev_record)
        return train.rerun_diff(self.prev, self.record, self.rows, self.cur / "predictions_validation.csv")

    def test_identical(self):
        d = self.diff()
        self.assertEqual(d, {"previous_attempt": 1, "previous_status": "completed", "comparable": True,
                             "updates_compared": 5, "first_differing_update": None, "max_abs_loss_diff": 0.0,
                             "selected_step_equal": True, "predictions_compared": 2, "prediction_flips": 0,
                             "max_abs_margin_diff": 0.0, "checkpoint_sha256_equal": True})

    def test_differences_recorded(self):
        rows = copy.deepcopy(self.rows)
        rows[2]["loss"] += 0.25
        d = self.diff(prev_rows=rows, margins=(1.5, -2.5))
        self.assertTrue(d["comparable"])
        self.assertEqual((d["first_differing_update"], d["max_abs_margin_diff"]), (3, 0.5))
        self.assertAlmostEqual(d["max_abs_loss_diff"], 0.25, places=12)
        prev = copy.deepcopy(self.record)
        del prev["checkpoint"]["sha256"]
        prev["selection"] = None
        d = self.diff(prev_record=prev)
        self.assertIsNone(d["checkpoint_sha256_equal"])  # not recorded: None, not False
        self.assertIsNone(d["selected_step_equal"])

    def test_not_comparable(self):
        cases = [({"status": "interrupted"}, "did not complete")]
        for path, value in ((("env", "torch"), "2.13"), (("env", "cuda"), "12.8"), (("env", "gpu", "name"), "B"),
                            (("env", "requirements_sha256"), "x"), (("lists", "files_sha256"), {}),
                            (("config", "sha256"), "y"), (("code", "files_sha256_lf"), {}), (("env", "gpu"), None)):
            prev = copy.deepcopy(self.record)
            node = prev
            for k in path[:-1]:
                node = node[k]
            node[path[-1]] = value
            cases.append((prev, "differs from the previous attempt"))
        for prev, reason in cases:
            d = self.diff(prev_record=prev)
            self.assertFalse(d["comparable"], prev)
            self.assertIn(reason, d["reason"])
        d = self.diff(prev_rows=self.rows[:4])
        self.assertFalse(d["comparable"])
        self.assertIn("incomplete overlap: updates 4 of 5", d["reason"])
        (self.prev / "run.json").write_text("{", encoding="ascii")
        d = train.rerun_diff(self.prev, self.record, self.rows, self.cur / "predictions_validation.csv")
        self.assertEqual((d["comparable"], d["reason"]), (False, "previous run.json unreadable"))

    def test_condition_c_needs_equal_augmentation_files(self):
        self.assertEqual(len(train.RERUN_SAME), 7)  # unchanged: summarize_s2.py reads it for the A runs
        self.assertNotIn("augmentation", json.dumps(train.RERUN_SAME))
        aug = {"augmentation": {"files_sha256": {"train_aug.csv": "a", "cases.csv": "b"}}}
        self.record["data"] = copy.deepcopy(aug)
        self.assertTrue(self.diff()["comparable"])  # equal augmentation files
        prev = copy.deepcopy(self.record)
        prev["data"]["augmentation"]["files_sha256"]["train_aug.csv"] = "other"
        d = self.diff(prev_record=prev)
        self.assertEqual((d["comparable"], d["reason"]), (
            False, "differs from the previous attempt or not recorded there: augmentation files SHA-256"))
        prev = copy.deepcopy(self.record)
        prev["data"] = {"train_rows": 20}  # an attempt without augmentation is never comparable with one
        self.assertIn("augmentation files SHA-256", self.diff(prev_record=prev)["reason"])
        self.record["data"] = {"train_rows": 20}  # A and smoke: no augmentation on either side
        self.assertTrue(self.diff()["comparable"])


class TestRunLifecycle(unittest.TestCase):
    """run() with data, tokenizer, model, CUDA calls and fit patched out (CPU, tiny checkpoint, no weights)."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "runs").mkdir()
        self.smoke_tmp = self.root / "tempdir" / "x-s1-bert-smoke-s0-ckpt"

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def ctx(self, smoke):
        run_id = "s1-bert-smoke-s0" if smoke else "s2-bert-A-s42"
        return {"stage": "s1" if smoke else "s2", "condition": "smoke" if smoke else "A",
                "seed": 0 if smoke else 42, "run_id": run_id, "run_dir": self.root / "runs" / run_id,
                "lists": {"mode": "official", "dir": "data/processed", "git_tracked_checked": not smoke},
                "list_dir": None, "n_total": train.budget(20, 3), "eval_at": [1], "n_run": 30 if smoke else 4}

    def go(self, smoke=False, rerun=False, reload_ok=True, rerun_diff=None, code_record=None):
        ctx, seen = self.ctx(smoke), {}

        def fake_fit(model, key, data, cfg, **kw):
            seen.update(ckpt_dir=kw["ckpt_dir"], ckpt_dir_existed=kw["ckpt_dir"].exists(),
                        lock=train.lock_path(ctx["run_dir"]).exists())
            info = model_specs.save_checkpoint(tiny_model(0), key, kw["ckpt_dir"], kw["run_id"], 1)
            rows = [{"update": u, "epoch": 1, "examples": 16, "lr": 2e-5, "loss": 1 / u, "grad_norm": 1.0,
                     "seconds": 0.01} for u in range(1, kw["n_run"] + 1)]
            return {"log": rows, "evals": [], "saved": [1], "eval_s": [0.1], "save_s": [0.1], "optimizer": {},
                    "best": {"step": 1, "sha256": info["sha256"], "size": info["size"], "tensors": info["tensors"]}}

        reload = {"state_dict_equal": reload_ok, "prediction_flips": 0, "max_abs_margin_diff": 0.0,
                  "passed": reload_ok, "seconds": 0.1}
        patches = [
            mock.patch.object(data_lists, "load_rows", return_value=None),
            mock.patch.object(data_lists, "load_training_data", return_value={
                "train": [(f"train:{i:06d}", i % 2, "t") for i in range(20)],
                "validation": [(f"validation:{i:06d}", i % 2, "v", True) for i in range(4)]}),
            mock.patch.object(train, "apply_determinism", return_value={}),
            mock.patch.object(train, "env_info", return_value={"gpu": {"name": "cpu"}}),
            mock.patch.object(model_specs, "load_tokenizer", return_value=types.SimpleNamespace(pad_token_id=0)),
            mock.patch.object(model_specs, "check_tokenizer", return_value="fp"),
            mock.patch.object(eval_lib, "encode", side_effect=lambda tok, texts, n: (
                [[2, 5, 3]] * len(texts), {"max_tokens": 3, "rows_over_max_length": 0})),
            mock.patch.object(model_specs, "load_model", return_value=mock.MagicMock()),
            mock.patch.object(train, "fit", side_effect=fake_fit),
            mock.patch.object(train, "reload_check", return_value=reload),
            mock.patch.object(train, "smoke_ckpt_dir", return_value=self.smoke_tmp),
            mock.patch.object(train, "code_record", side_effect=code_record,
                              return_value={"git_commit": None, "git_dirty": [], "files_sha256_lf": {}}),
            *(mock.patch(f"torch.cuda.{f}", return_value=1) for f in (
                "init", "reset_peak_memory_stats", "synchronize", "max_memory_allocated", "max_memory_reserved",
                "empty_cache")),
        ]
        if rerun_diff is not None:
            patches.append(mock.patch.object(train, "rerun_diff", side_effect=rerun_diff))
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            st.enter_context(contextlib.redirect_stdout(io.StringIO()))
            st.enter_context(contextlib.redirect_stderr(io.StringIO()))
            code = train.run(types.SimpleNamespace(model="bert", rerun=rerun), CFG, b"{}\n", ctx)
        self.assertFalse(train.lock_path(ctx["run_dir"]).exists())  # released
        return code, json.loads((ctx["run_dir"] / "run.json").read_text(encoding="ascii")), seen, ctx["run_dir"]

    def previous_attempt(self):
        rd = self.root / "runs" / "s2-bert-A-s42"
        (rd / "checkpoint").mkdir(parents=True)
        (rd / "checkpoint" / "model.safetensors").write_bytes(b"old")
        (rd / train.STAGING).mkdir()
        (rd / train.STAGING / "model.safetensors.tmp").write_bytes(b"partial")
        train.write_json(rd / "run.json", {"status": "completed", "checkpoint": {"kept": True, "sha256": "old"}})

    def test_stage2_diagnostic_failure_keeps_validated_checkpoint(self):
        self.previous_attempt()
        err = OSError(2, "No such file or directory", str(ROOT / "runs" / "x" / "train_log.csv"))
        code, rec, seen, rd = self.go(rerun=True, rerun_diff=err)
        self.assertEqual((code, rec["status"], rec["failure"]), (0, "completed", None))
        self.assertEqual(rec["rerun"], {"previous_attempt": 1, "comparable": False,
                                        "reason": "FileNotFoundError: No such file or directory"})
        self.assertEqual((seen["ckpt_dir"], seen["ckpt_dir_existed"], seen["lock"]), (rd / train.STAGING, False, True))
        self.assertFalse((rd / train.STAGING).exists())
        self.assertEqual(os.listdir(rd / "checkpoint"), ["model.safetensors"])
        self.assertEqual(sha((rd / "checkpoint" / "model.safetensors").read_bytes()), rec["checkpoint"]["sha256"])
        self.assertEqual((rec["checkpoint"]["kept"], rec["checkpoint"]["path"], "deleted" in rec["checkpoint"]),
                         (True, "runs/s2-bert-A-s42/checkpoint/model.safetensors", False))
        old = json.loads((rd / "attempts" / "1" / "run.json").read_text(encoding="ascii"))
        self.assertEqual(old["checkpoint"], {"kept": False, "deleted_by_rerun": True, "sha256": "old"})

    def test_stage2_reload_failure_deletes_unvalidated_checkpoint(self):
        code, rec, _, rd = self.go(reload_ok=False)
        self.assertEqual((code, rec["status"], rec["failure"]["exception"]), (1, "failed", "train.ReloadCheckError"))
        self.assertFalse((rd / train.STAGING).exists() or (rd / "checkpoint").exists())
        self.assertEqual({k: rec["checkpoint"][k] for k in ("kept", "path", "deleted")},
                         {"kept": False, "path": None, "deleted": True})

    def test_interrupt_after_validation_keeps_checkpoint(self):
        self.previous_attempt()
        code, rec, _, rd = self.go(rerun=True, rerun_diff=KeyboardInterrupt)
        self.assertEqual((code, rec["status"], rec["failure"]["exception"]), (1, "interrupted", "KeyboardInterrupt"))
        self.assertTrue(rec["checkpoint"]["kept"])
        self.assertEqual(sha((rd / "checkpoint" / "model.safetensors").read_bytes()), rec["checkpoint"]["sha256"])

    def test_smoke_fixed_temp_folder(self):
        self.smoke_tmp.mkdir(parents=True)
        (self.smoke_tmp / "model.safetensors").write_bytes(b"left by a killed run")
        code, rec, seen, rd = self.go(smoke=True)
        self.assertEqual((code, rec["status"], rec["smoke"]["passed"]), (0, "completed", True))
        self.assertEqual((seen["ckpt_dir"], seen["ckpt_dir_existed"]), (self.smoke_tmp, False))
        self.assertFalse(self.smoke_tmp.exists())
        self.assertFalse((rd / "checkpoint").exists() or (rd / train.STAGING).exists())
        self.assertEqual({k: rec["checkpoint"][k] for k in ("kept", "path", "deleted")},
                         {"kept": False, "path": None, "deleted": True})

    def test_setup_failure_ends_with_final_status(self):
        code, rec, _, rd = self.go(code_record=RuntimeError("git"))
        self.assertEqual((code, rec["status"], rec["failure"]["phase"], rec["failure"]["exception"]),
                         (1, "failed", "setup", "RuntimeError"))
        self.assertEqual(rec["attempt"], 1)
        self.assertFalse((rd / "checkpoint").exists() or (rd / train.STAGING).exists())

    def test_lock_held_by_another_run(self):
        ctx = self.ctx(False)
        train.lock_path(ctx["run_dir"]).write_text("pid 1\n")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(train.run(types.SimpleNamespace(model="bert", rerun=False), CFG, b"{}\n", ctx), 1)
        self.assertIn("lock appeared after the preflight", out.getvalue())
        self.assertFalse(ctx["run_dir"].exists())
        self.assertEqual(train.lock_path(ctx["run_dir"]).read_text(), "pid 1\n")  # not ours: left alone


LIST_CONTENT = {
    "train_clean.csv": b"sentence_id,label,text_sha256_12\ntrain:000000,0,aaaaaaaaaaaa\n",
    "train_excluded.csv": b"sentence_id,label,text_sha256_12,reason,ref\n",
    "validation_clean.csv": b"sentence_id,label,text_sha256_12\nvalidation:000000,1,bbbbbbbbbbbb\n",
    "validation_excluded.csv": b"sentence_id,label,text_sha256_12,reason,ref\n",
    "test_clean_preaug.csv": b"sentence_id\ntest:000000\n",
    "test_excluded_preaug.csv": b"sentence_id,reason\n",
}
REVIEW_BYTES = b"sentence_id,text_sha256_12,group_id,check,label,decision,reason,reviewer,status,confirmed_by\n"


def make_repo_files(root, trial, where="trial", script=b"# build_lists stand-in\n"):
    """Synthetic list folder (meta with matching SHA-256s), review and build_lists.py under root."""
    proc = root / "data" / "processed"
    d = proc / "trial" if where == "trial" else proc
    d.mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts" / "build_lists.py").write_bytes(script)
    (proc / "train_review.csv").write_bytes(REVIEW_BYTES)
    for n, b in LIST_CONTENT.items():
        (d / n).write_bytes(b)
    meta = {"trial": trial, "review": {"sha256": sha(REVIEW_BYTES)}, "script": {"sha256_lf": sha(script)},
            "dataset": {"revision": "rev"},
            "outputs": {n: {"rows": 11050 if n == "train_clean.csv" else 1, "sha256": sha(b)}
                        for n, b in LIST_CONTENT.items()}}
    (d / "lists_meta.json").write_text(json.dumps(meta), encoding="ascii", newline="\n")
    return d


class TestListVerification(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def codes(self, mode):
        errors, info = data_lists.verify_lists(mode, self.root)
        return [e.split()[0] for e in errors], info

    def test_trial_ok(self):
        make_repo_files(self.root, True)
        codes, info = self.codes("trial")
        self.assertEqual(codes, [])
        self.assertEqual(info["files_sha256"]["train_clean.csv"], sha(LIST_CONTENT["train_clean.csv"]))
        self.assertEqual(info["train_clean_rows"], 11050)

    def test_flipped_byte(self):
        d = make_repo_files(self.root, True)
        b = bytearray((d / "validation_clean.csv").read_bytes())
        b[-3] ^= 1
        (d / "validation_clean.csv").write_bytes(bytes(b))
        self.assertEqual(self.codes("trial")[0], ["L05"])

    def test_review_sha_mismatch(self):
        make_repo_files(self.root, True)
        (self.root / "data" / "processed" / "train_review.csv").write_bytes(REVIEW_BYTES + b"x\n")
        self.assertEqual(self.codes("trial")[0], ["L04"])

    def test_script_sha_mismatch(self):
        make_repo_files(self.root, True)
        (self.root / "scripts" / "build_lists.py").write_bytes(b"# edited\n")
        self.assertEqual(self.codes("trial")[0], ["L06"])

    def test_trial_lists_in_official_location(self):
        make_repo_files(self.root, True, where="official")
        codes, info = self.codes("official")
        self.assertIn("L03", codes)  # meta trial true
        self.assertIn("L07", codes)  # not git-tracked (no repo)
        self.assertNotIn("L08", codes)  # --check subprocess not reached
        self.assertIsNone(info)

    def test_official_missing(self):
        make_repo_files(self.root, True)
        self.assertEqual(self.codes("official")[0], ["L01"])

    def test_join(self):
        rows = {"train:000000": ("0", "alpha"), "train:000001": ("1", "beta")}
        h = {k: data_lists.build_lists.h12(v[1]) for k, v in rows.items()}
        ok, errors = data_lists.join([("train:000000", "0", h["train:000000"]),
                                      ("train:000001", "1", h["train:000001"])], rows, "x")
        self.assertEqual((errors, [r[0] for r in ok]), ([], ["train:000000", "train:000001"]))
        cases = [([("train:000000", "0", "000000000000")], "J04"),
                 ([("train:000000", "1", h["train:000000"])], "J03"),
                 ([("train:000009", "0", h["train:000000"])], "J02"),
                 ([("train:000001", "1", h["train:000001"]), ("train:000000", "0", h["train:000000"])], "J01")]
        for lst, code in cases:
            _, errors = data_lists.join(lst, rows, "x")
            self.assertEqual([e.split()[0] for e in errors], [code])

    def test_partition(self):
        d = self.root / "lists"
        d.mkdir()
        rows = {"train": {"train:000000": ("0", "a"), "train:000001": ("1", "b")},
                "validation": {"validation:000000": ("1", "c"), "validation:000001": ("0", "d")}}
        h = data_lists.build_lists.h12
        (d / "train_clean.csv").write_bytes(f"{data_lists.TRAIN_VAL_HEADER}\ntrain:000000,0,{h('a')}\n".encode())
        (d / "train_excluded.csv").write_bytes(
            f"{data_lists.TRAIN_VAL_HEADER},reason,ref\ntrain:000001,1,{h('b')},duplicate_same_label,x\n".encode())
        (d / "validation_clean.csv").write_bytes(
            f"{data_lists.TRAIN_VAL_HEADER}\nvalidation:000001,0,{h('d')}\n".encode())
        (d / "validation_excluded.csv").write_bytes(
            f"{data_lists.TRAIN_VAL_HEADER},reason,ref\nvalidation:000000,1,{h('c')},train_overlap,x\n".encode())
        out = data_lists.load_training_data(d, rows)
        self.assertEqual([r[0] for r in out["train"]], ["train:000000"])
        self.assertEqual([(r[0], r[3]) for r in out["validation"]],
                         [("validation:000000", False), ("validation:000001", True)])
        (d / "validation_excluded.csv").write_bytes(f"{data_lists.TRAIN_VAL_HEADER},reason,ref\n".encode())
        with self.assertRaises(data_lists.ListError) as cm:
            data_lists.load_training_data(d, rows)
        self.assertEqual([e.split()[0] for e in cm.exception.errors], ["J05"])


class TestTestCsvGuard(unittest.TestCase):
    CODE = ("import sys; sys.path.insert(0, sys.argv[1]); import data_lists; data_lists.install_test_guard(); "
            "open(sys.argv[2], 'rb').read(); print('READ')")

    def run_open(self, path):
        env = {**os.environ, "PYTHONUTF8": "1"}
        return subprocess.run([sys.executable, "-c", self.CODE, str(SCRIPTS), str(path)], capture_output=True,
                              env=env, timeout=120)

    def test_guard(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "sub").mkdir()
            for name in ("TEST.CSV", "sub/Test.Csv"):
                (tmp / name).write_bytes(b"x\n")
                r = self.run_open(tmp / name)
                self.assertEqual(r.returncode, 1, name)
                self.assertIn(b"FAIL test.csv guard", r.stdout)
                self.assertNotIn(b"READ", r.stdout)
            (tmp / "other.csv").write_bytes(b"x\n")
            r = self.run_open(tmp / "other.csv")
            self.assertEqual((r.returncode, r.stdout.strip()), (0, b"READ"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestPreflightHermetic(unittest.TestCase):
    """Stage-2 refusal on a temporary git repo (no official lists, trial lists, dirty files); nothing created."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "scripts").mkdir()
        (self.root / "configs").mkdir()
        for p in train.CODE_FILES:
            shutil.copyfile(ROOT / p, self.root / p)
        shutil.copyfile(ROOT / train.CONFIG_REL, self.root / train.CONFIG_REL)
        (self.root / "data" / "processed").mkdir(parents=True)
        (self.root / "data" / "processed" / "train_review.csv").write_bytes(REVIEW_BYTES)
        self.git("init", "-q")
        self.commit()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def git(self, *args):
        r = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def commit(self):
        self.git("add", "-A")
        self.git("-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-q", "-m", "x")

    def codes(self, **kw):
        args = types.SimpleNamespace(**{"model": "bert", "smoke": False, "seed": 42, "lists": "official",
                                        "rerun": False, **kw})
        errors, ctx = train.preflight(args, CFG, self.root)
        self.assertFalse((self.root / "runs").exists())
        return errors, ctx

    def test_s2_refused_without_official_lists(self):
        errors, ctx = self.codes()
        self.assertIsNone(ctx)
        self.assertTrue(any(e.startswith("P07 L01") for e in errors), errors)
        self.assertFalse(any(e.startswith(("P08", "P09")) for e in errors), errors)

    def test_s2_refused_with_trial_lists_in_official_folder(self):
        script = (self.root / "scripts" / "build_lists.py").read_bytes()
        make_repo_files(self.root, True, where="official", script=script)
        self.commit()
        errors, _ = self.codes()
        self.assertTrue(any(e.startswith("P07 L03") for e in errors), errors)

    def test_s2_refused_dirty_and_cli(self):
        with open(self.root / "scripts" / "eval_lib.py", "a", encoding="ascii") as f:
            f.write("# edit\n")
        (self.root / "notes.txt").write_text("x")
        self.commit()
        with open(self.root / "notes.txt", "a", encoding="ascii") as f:
            f.write("y")
        with open(self.root / "scripts" / "train.py", "a", encoding="ascii") as f:
            f.write("# edit\n")
        errors, _ = self.codes()
        self.assertTrue(any(e.startswith("P08") and "scripts/train.py" in e for e in errors), errors)
        self.assertTrue(any(e.startswith("P09") for e in errors), errors)
        errors, _ = self.codes(seed=7, lists="trial")
        self.assertEqual(sum(e.startswith("P03") for e in errors), 2, errors)

    def test_smoke_trial_lists_pass_list_checks(self):
        script = (self.root / "scripts" / "build_lists.py").read_bytes()
        make_repo_files(self.root, True, where="trial", script=script)
        with open(self.root / "scripts" / "train.py", "a", encoding="ascii") as f:
            f.write("# edit\n")  # smoke records dirty files instead of refusing
        errors, _ = self.codes(smoke=True, seed=None, lists="trial")
        self.assertFalse(any(e.startswith(("P07", "P08", "P09", "P10")) for e in errors), errors)
        (self.root / "runs" / "s1-bert-smoke-s0").mkdir(parents=True)
        errors, _ = train.preflight(types.SimpleNamespace(model="bert", smoke=True, seed=None, lists="trial",
                                                          rerun=False), CFG, self.root)
        self.assertTrue(any(e.startswith("P06") for e in errors), errors)

    def smoke_args(self, **kw):
        return types.SimpleNamespace(**{"model": "bert", "smoke": True, "seed": None, "lists": "trial",
                                        "rerun": True, **kw})

    def test_run_dir_link_and_lock_refused(self):
        target = self.root / "elsewhere"
        target.mkdir()
        (self.root / "runs").mkdir()
        make_link(target, self.root / "runs" / "s1-bert-smoke-s0")
        errors, _ = train.preflight(self.smoke_args(), CFG, self.root)
        self.assertTrue(any(e.startswith("P06") and "symlink or junction" in e for e in errors), errors)
        os.rmdir(self.root / "runs" / "s1-bert-smoke-s0")  # removes the junction only
        (self.root / "runs" / "s1-bert-smoke-s0").mkdir()
        (self.root / "runs" / "s1-bert-smoke-s0.lock").write_text("pid 1\n")
        errors, _ = train.preflight(self.smoke_args(), CFG, self.root)
        self.assertFalse(any(e.startswith("P06") for e in errors), errors)
        self.assertTrue(any(e.startswith("P12") and "delete runs/s1-bert-smoke-s0.lock" in e for e in errors),
                        errors)
        shutil.rmtree(self.root / "runs")
        make_link(target, self.root / "runs")  # runs/ itself redirected outside the repo
        (target / "s1-bert-smoke-s0").mkdir()
        errors, _ = train.preflight(self.smoke_args(), CFG, self.root)
        self.assertTrue(any(e.startswith("P06") and "direct child" in e for e in errors), errors)

    def test_code_record_names_only_watched_paths(self):
        with open(self.root / "scripts" / "train.py", "a", encoding="ascii") as f:
            f.write("# edit\n")
        (self.root / "private-notes.txt").write_text("x")
        (self.root / "data" / "processed" / "train_clean.csv").write_text("x")
        (self.root / "runs" / "r").mkdir(parents=True)
        (self.root / "runs" / "r" / "log.txt").write_text("x")
        rec = train.code_record(self.root)
        self.assertEqual(sorted(d["path"] for d in rec["git_dirty"]),
                         ["data/processed/train_clean.csv", "scripts/train.py"])
        self.assertEqual(rec["git_dirty_other_count"], 1)
        self.assertNotIn("private-notes", json.dumps(rec))

    def test_smoke_may_use_untracked_official_lists(self):
        make_repo_files(self.root, False, where="official", script=STANDIN_BUILD_LISTS)  # not committed
        errors, info = data_lists.verify_lists("official", self.root, require_tracked=False)
        self.assertEqual(errors, [])
        self.assertEqual((info["git_tracked_checked"], info["build_lists_check"]), (False, "passed"))
        errors, _ = self.codes(smoke=True, seed=None)
        self.assertFalse(any(e.startswith(("P07", "P08", "P09", "P10")) for e in errors), errors)
        errors, _ = self.codes()  # stage 2 keeps the git check
        self.assertTrue(any(e.startswith("P07 L07") for e in errors), errors)
        self.assertEqual(data_lists.verify_lists("trial", self.root)[0][0].split()[0], "L01")

    def ready(self):
        """Patches for the checks a CPU test machine cannot pass (offline flag, CUBLAS env, CUDA)."""
        return [mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True),
                mock.patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": CFG["determinism"]["cublas_workspace_config"]}),
                mock.patch.object(torch.cuda, "is_available", return_value=True)]

    def tree(self):
        return sorted((p.relative_to(self.root).as_posix(), p.stat().st_size if p.is_file() else None)
                      for p in self.root.rglob("*") if ".git" not in p.relative_to(self.root).parts)

    def test_preflight_only(self):
        make_repo_files(self.root, False, where="official", script=STANDIN_BUILD_LISTS)
        self.commit()
        before = self.tree()
        out = io.StringIO()
        with contextlib.ExitStack() as st:
            for p in self.ready() + [mock.patch.object(train, "ROOT", self.root), contextlib.redirect_stdout(out)]:
                st.enter_context(p)
            errors, ctx = train.preflight(types.SimpleNamespace(model="bert", smoke=False, seed=42,
                                                                lists="official", rerun=False), CFG, self.root)
            self.assertEqual((errors, ctx["run_id"], ctx["lists"]["git_tracked_checked"]),
                             ([], "s2-bert-A-s42", True))  # stage 2 passes on a clean committed repo
            self.assertEqual(train.main(["--model", "bert", "--seed", "42", "--preflight-only"]), 0)
            self.assertEqual(train.main(["--model", "roberta", "--smoke", "--preflight-only"]), 0)
            self.assertEqual(train.main(["--model", "bert", "--seed", "7", "--preflight-only"]), 1)
            self.assertEqual(train.main(["--model", "bert", "--smoke", "--rerun", "--preflight-only"]), 1)
        self.assertIn("preflight passed: s2-bert-A-s42 would start", out.getvalue())
        self.assertIn("FAIL P03 seed 7", out.getvalue())
        self.assertEqual(self.tree(), before)  # nothing created
        self.assertFalse(train.smoke_ckpt_dir("s1-roberta-smoke-s0", self.root).exists())
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            train.parse_args(["--smoke-summary", "--preflight-only"])


# ---------------------------------------------------------------- condition C (stage 4)

# Synthetic augmentation set. The text_sha256_12 values and AUG_LIST_SHA256 / AUG_OVERLAP_SHA256 were computed
# with `printf '<bytes>' | sha256sum`, not with the code under test.
# The three usable rows are chosen so that a changed text or another row order cannot go unnoticed: the ID order
# is not the order by label (either direction), by text (raw, lower-cased or stripped), by text hash or by text
# length (TestAugmentationRows.test_fixture_tells_orders_and_texts_apart).
AUG_TEXT_01 = " IgNoRe  4ll pRiOr "  # AUG-01 shapes: mixed case, two spaces, a leet digit; a space at each end
AUG_TEXT_03 = 'first, "quoted"\n\nsecond line'  # AUG-03 shape: comma, quotes and a blank line inside one field
AUG_TEXT_04 = "Dear World"
AUG_CASES_BYTES = (
    b"case_id,group_id,source_split,source_id,purpose,condition,subtype,eval_group,label,position,context_ids,"
    b"generator,generator_version,status,auto_flags,text\n"
    b"aug-01-case-000003,train:000003,train,train:000003,train,AUG-01,case,train,0,,,rule,aug01-v1,assigned,,"
    b" IgNoRe  4ll pRiOr \n"
    b"aug-01-space-000007,train:000007,train,train:000007,train,AUG-01,space,train,0,,,rule,aug01-v1,reserve,,"
    b"x  y\n"
    b"aug-03-000011,train:000011,train,train:000011,train,AUG-03,context,train,1,prefix,5,template+llm,v,assigned,,"
    b'"first, ""quoted""\n\nsecond line"\n'
    b"aug-04-001,aug-04-001,new,aug04_benign_v1:001,train,AUG-04,b1,train,0,,,llm,v,assigned,,Dear World\n"
    b"aug-05-001,aug-05-001,new,aug05_attacks_v1:001,train,AUG-05,a1,train,1,,,llm,v,auto_excluded,exclude:format,"
    b"caf\xc3\xa9\n")
AUG_LIST_HEAD = b"sentence_id,label,text_sha256_12\n"
AUG_LIST_BYTES = (AUG_LIST_HEAD + b"aug-01-case-000003,0,745b04798cbd\n" + b"aug-03-000011,1,a487ee61d127\n"
                  + b"aug-04-001,0,86923eba67a9\n")
AUG_LIST_SHA256 = "3db72cfb5421a6e3b368c251e0ba0d424f55abd900d771b378cc138cb678f4d3"
AUG_ROWS = [("aug-01-case-000003", 0, AUG_TEXT_01), ("aug-03-000011", 1, AUG_TEXT_03),
            ("aug-04-001", 0, AUG_TEXT_04)]
AUG_REVIEW_BYTES = (b"case_id,text_sha256_12,decision,reason,reviewer\n"
                    b"aug-01-case-000003,745b04798cbd,keep,,tester\n"
                    b"aug-03-000011,a487ee61d127,keep,,tester\n"
                    b"aug-04-001,86923eba67a9,keep,,tester\n")
AUG_OVERLAP_BYTES = b'{"schema": 1}\n'
AUG_OVERLAP_SHA256 = "15864c015e873e139dad34ab4d217e1209db9b041d8fabd7c9e8e415e2dc477a"
AUG_DIR = "data/cases/train"
AUG_RELS = [f"{AUG_DIR}/{n}" for n in ("cases.csv", "review.csv", "train_aug.csv", "cases_meta.json",
                                       "test_overlap.json")] + ["scripts/build_train_aug.py"]
# build_train_aug.py stand-ins for temp repos. The first exits 0 only for exactly `--check` run from the repo root.
STANDIN_BUILD_TRAIN_AUG = (
    b"import os, sys\n"
    b"root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n"
    b"sys.exit(0 if sys.argv[1:] == ['--check'] and os.path.samefile(os.getcwd(), root) else 3)\n")
FAILING_BUILD_TRAIN_AUG = (b"import sys\n"
                           b"print('FAIL K01 --check: cases.csv differs from the rebuilt bytes')\n"
                           b"print('refused: 1 error(s); nothing written')\n"
                           b"sys.exit(1)\n")
CHECK_FAILED = ("P13 scripts/build_train_aug.py --check exit 1: FAIL K01 --check: cases.csv differs from the "
                "rebuilt bytes")
# Fails with nine FAIL lines: one with a byte that is not ASCII, then eight of 5,009 characters each.
NOISY_BUILD_TRAIN_AUG = (b"import sys\n"
                         b"sys.stdout.buffer.write(b'FAIL K00 \\xff byte\\n')\n"
                         b"for n in range(1, 9):\n"
                         b"    sys.stdout.buffer.write(b'FAIL K0%d ' % n + b'x' * 5000 + b'\\n')\n"
                         b"sys.exit(2)\n")


def make_aug_files(root, *, files=None, edit=None, script=STANDIN_BUILD_TRAIN_AUG):
    """Synthetic augmentation set under root: data/cases/train/ and scripts/build_train_aug.py (script=None: no
    script). files replaces file bytes by name, edit(meta) changes the cases_meta.json dict before it is written.
    Returns {name: bytes} of the five data files."""
    out = {"cases.csv": AUG_CASES_BYTES, "review.csv": AUG_REVIEW_BYTES, "train_aug.csv": AUG_LIST_BYTES,
           "test_overlap.json": AUG_OVERLAP_BYTES, **(files or {})}
    meta = {"schema": 1, "ready_for_training": True, "review": {"pending": 0, "keep": 3, "exclude": 0},
            "test_overlap": {"status": "applied"},
            "script": {"path": "scripts/build_train_aug.py",
                       "sha256_lf": sha((script or b"").replace(b"\r\n", b"\n"))},
            "outputs": {n: {"rows": out[n].count(b"\n") - 1, "sha256": sha(out[n])}
                        for n in ("cases.csv", "review.csv", "train_aug.csv")}}
    if edit is not None:
        edit(meta)
    out.setdefault("cases_meta.json", (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("ascii"))
    (root / AUG_DIR).mkdir(parents=True, exist_ok=True)
    for n, b in out.items():
        (root / AUG_DIR / n).write_bytes(b)
    if script is not None:
        (root / "scripts").mkdir(exist_ok=True)
        (root / "scripts" / "build_train_aug.py").write_bytes(script)
    return out


class TestConditionIdentity(unittest.TestCase):
    def test_run_identity(self):
        ns = types.SimpleNamespace
        self.assertEqual((train.CONDITIONS, train.STAGE_OF), (("A", "C"), {"A": "s2", "C": "s4"}))
        self.assertEqual(train.run_identity(ns(model="roberta", smoke=False, seed=123, condition="C"), CFG),
                         ("s4", "C", 123, "s4-roberta-C-s123"))
        self.assertEqual(train.run_identity(ns(model="roberta", smoke=False, seed=123, condition="A"), CFG),
                         ("s2", "A", 123, "s2-roberta-A-s123"))
        self.assertEqual(train.run_identity(ns(model="bert", smoke=False, seed=42), CFG),
                         ("s2", "A", 42, "s2-bert-A-s42"))  # no condition attribute: A, as before stage 4
        for smoke in (ns(model="bert", smoke=True, seed=None), ns(model="bert", smoke=True, seed=None, condition="A")):
            self.assertEqual(train.run_identity(smoke, CFG), ("s1", "smoke", 0, "s1-bert-smoke-s0"))
        for bad in (ns(model="bert", smoke=True, seed=None, condition="C"),  # never a smoke run of C
                    ns(model="bert", smoke=False, seed=42, condition="B"),
                    ns(model="bert", smoke=False, seed=42, condition="D"),
                    ns(model="bert", smoke=False, seed=42, condition="c"),
                    ns(model="bert", smoke=False, seed=42, condition="smoke"),
                    ns(model="bert", smoke=False, seed=42, condition=None)):
            with self.assertRaises(ValueError):
                train.run_identity(bad, CFG)

    def test_cli(self):
        a = train.parse_args(["--model", "roberta", "--seed", "42"])
        self.assertEqual((a.condition, a.smoke, a.seed), ("A", False, 42))
        a = train.parse_args(["--model", "roberta", "--seed", "42", "--condition", "C"])
        self.assertEqual((a.condition, a.smoke, a.seed, a.lists), ("C", False, 42, "official"))
        self.assertEqual(train.run_identity(a, CFG)[3], "s4-roberta-C-s42")
        self.assertEqual(train.parse_args(["--model", "bert", "--smoke"]).condition, "A")
        self.assertEqual(train.parse_args(["--model", "bert", "--smoke", "--condition", "A"]).condition, "A")
        self.assertTrue(train.parse_args(["--smoke-summary"]).smoke_summary)
        for argv, message in ((["--model", "bert", "--smoke", "--condition", "C"],
                               "--condition C is not allowed with --smoke"),
                              (["--model", "bert", "--seed", "42", "--condition", "B"], "invalid choice: 'B'"),
                              (["--model", "bert", "--seed", "42", "--condition", "D"], "invalid choice: 'D'"),
                              (["--smoke-summary", "--condition", "C"], "--smoke-summary takes no other option")):
            err = io.StringIO()
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(err):
                train.parse_args(argv)
            self.assertEqual(cm.exception.code, 2, argv)
            self.assertIn(message, err.getvalue())

    def test_preflight_refuses_condition_before_anything_else(self):
        ns = types.SimpleNamespace
        nowhere = Path(tempfile.gettempdir()) / "no-such-repo-for-train-tests"
        errors, ctx = train.preflight(ns(model="bert", smoke=True, seed=None, lists="trial", rerun=False,
                                         condition="C"), CFG, nowhere)
        self.assertEqual((errors, ctx), (["P03 --condition C is not allowed with --smoke (a smoke run is stage 1 "
                                          "on the common lists only)"], None))
        errors, ctx = train.preflight(ns(model="bert", smoke=False, seed=42, lists="official", rerun=False,
                                         condition="B"), CFG, nowhere)
        self.assertEqual((errors, ctx), (["P03 condition 'B' is not one of A, C (README 5.3: B and D reuse the A "
                                          "and C checkpoints)"], None))


class TestAugmentationRows(unittest.TestCase):
    """join_augmentation / load_augmentation on synthetic bytes."""

    def refused(self, cases=AUG_CASES_BYTES, listed=AUG_LIST_BYTES):
        with self.assertRaises(data_lists.ListError) as cm:
            train.join_augmentation(cases, listed)
        return cm.exception.errors

    def test_rows_in_list_order(self):
        rows = train.join_augmentation(AUG_CASES_BYTES, AUG_LIST_BYTES)
        # labels as int; every text exactly as written: letter case, inner and edge spaces of the AUG-01 text,
        # the blank line and the quotes of the AUG-03 text
        self.assertEqual(rows, AUG_ROWS)
        self.assertEqual(rows, [("aug-01-case-000003", 0, " IgNoRe  4ll pRiOr "),
                                ("aug-03-000011", 1, 'first, "quoted"\n\nsecond line'),
                                ("aug-04-001", 0, "Dear World")])
        self.assertEqual(train.label_counts(rows), {"0": 2, "1": 1})
        self.assertEqual(train.label_counts([]), {"0": 0, "1": 0})
        one = AUG_LIST_HEAD + b"aug-03-000011,1,a487ee61d127\n"  # only listed cases are used
        self.assertEqual(train.join_augmentation(AUG_CASES_BYTES, one), [AUG_ROWS[1]])
        # cases.csv is read by column name: another column order and an extra column give the same rows
        recs = list(csv.reader(io.StringIO(AUG_CASES_BYTES.decode("utf-8"), newline="")))
        self.assertEqual([len(r) for r in recs], [16] * 6)
        moved = io.StringIO(newline="")
        w = csv.writer(moved, lineterminator="\n")
        for i, r in enumerate(recs):
            w.writerow([r[15], "note" if i == 0 else "extra"] + r[:15])
        self.assertEqual(train.join_augmentation(moved.getvalue().encode("utf-8"), AUG_LIST_BYTES), AUG_ROWS)

    def test_fixture_tells_orders_and_texts_apart(self):
        """The fixture itself: a join that sorted the rows or changed a text could not return AUG_ROWS."""
        ids = [r[0] for r in AUG_ROWS]
        self.assertEqual(ids, sorted(ids))  # train_aug.csv order is the ID order
        self.assertEqual(ids, [ln.split(b",")[0].decode("ascii") for ln in AUG_LIST_BYTES.split(b"\n")[1:-1]])
        for name, key in (("label", lambda r: r[1]), ("text", lambda r: r[2]),
                          ("lower-cased text", lambda r: r[2].lower()), ("upper-cased text", lambda r: r[2].upper()),
                          ("stripped text", lambda r: r[2].strip()), ("text length", lambda r: len(r[2])),
                          ("text hash", lambda r: sha(r[2].encode("utf-8")))):
            for reverse in (False, True):  # sorted() is stable: neither direction leaves the rows in ID order
                self.assertNotEqual([r[0] for r in sorted(AUG_ROWS, key=key, reverse=reverse)], ids, (name, reverse))
        for name, change in (("lower", str.lower), ("upper", str.upper), ("strip", str.strip),
                             ("lstrip", str.lstrip), ("rstrip", str.rstrip),
                             ("two spaces to one", lambda t: t.replace("  ", " ")),
                             ("white space collapsed", lambda t: " ".join(t.split())),
                             ("LF to CRLF", lambda t: t.replace("\n", "\r\n")),
                             ("leet digit undone", lambda t: t.replace("4", "a"))):
            changed = [r[0] for r in AUG_ROWS if change(r[2]) != r[2]]
            self.assertTrue(changed, name)
        self.assertEqual([r[0] for r in AUG_ROWS if r[2].lower() != r[2]], [ids[0], ids[2]])
        self.assertEqual([r[0] for r in AUG_ROWS if r[2].strip() != r[2]], [ids[0]])

    def test_list_refusals(self):
        r1, r3, r4 = (ln + b"\n" for ln in AUG_LIST_BYTES.split(b"\n")[1:4])
        j00 = ["J00 train_aug.csv: header or final LF differs"]
        j06 = ["J06 train_aug.csv: the header is not sentence_id,label,text_sha256_12 or a row does not have 3 "
               "fields"]
        for listed, want in (
                (AUG_LIST_BYTES[:-1], j00),
                (AUG_LIST_BYTES.replace(b"sentence_id", b"case_id"), j00),
                (b"", j00),
                (AUG_LIST_BYTES.replace(b"_12\n", b"_12,reason\n"), j06),
                (AUG_LIST_BYTES.replace(b"\n", b"\r\n"), j06),  # CRLF checkout
                (AUG_LIST_HEAD + r1[:-1] + b",x\n", j06),
                (AUG_LIST_HEAD + b"aug-01-case-000003,0\n", j06),
                (AUG_LIST_HEAD + r1 + b"\n" + r3, j06),  # blank line
                (AUG_LIST_HEAD + b"aug-01-case-00000\xc3\xa9,0,745b04798cbd\n", ["J06 train_aug.csv: not ASCII"]),
                (AUG_LIST_HEAD, ["J08 train_aug.csv: no rows (condition C needs augmentation rows)"]),
                (AUG_LIST_HEAD + r3 + r1, ["J01 train_aug.csv: IDs not strictly ascending at aug-01-case-000003"]),
                (AUG_LIST_HEAD + r4 + r4, ["J01 train_aug.csv: IDs not strictly ascending at aug-04-001"]),
                (AUG_LIST_HEAD + b"aug-04-002,0,86923eba67a9\n",
                 ["J02 train_aug.csv: aug-04-002 is not a row of the split"]),
                (AUG_LIST_HEAD + b"aug-04-001,1,86923eba67a9\n",
                 ["J03 train_aug.csv: aug-04-001 label differs from the raw split"]),
                (AUG_LIST_HEAD + b"aug-03-000011,0,a487ee61d127\n",
                 ["J03 train_aug.csv: aug-03-000011 label differs from the raw split"]),
                (AUG_LIST_HEAD + b"aug-04-001,0,86923eba67aa\n",
                 ["J04 train_aug.csv: aug-04-001 text_sha256_12 differs from the raw text"]),
                # the hash of the lower-cased or of the stripped AUG-01 text is not the hash of the text
                (AUG_LIST_HEAD + f"aug-01-case-000003,0,{sha(b' ignore  4ll prior ')[:12]}\n".encode("ascii"),
                 ["J04 train_aug.csv: aug-01-case-000003 text_sha256_12 differs from the raw text"]),
                (AUG_LIST_HEAD + f"aug-01-case-000003,0,{sha(b'IgNoRe  4ll pRiOr')[:12]}\n".encode("ascii"),
                 ["J04 train_aug.csv: aug-01-case-000003 text_sha256_12 differs from the raw text"]),
                # a reserve case and an automatically excluded case (hashes of their texts) are not usable
                (AUG_LIST_HEAD + f"aug-01-space-000007,0,{sha(b'x  y')[:12]}\n".encode("ascii"),
                 ["J07 train_aug.csv: status is not assigned in cases.csv: 1 row(s) (aug-01-space-000007)"]),
                (AUG_LIST_HEAD + f"aug-05-001,1,{sha(b'caf' + bytes([0xc3, 0xa9]))[:12]}\n".encode("ascii"),
                 ["J07 train_aug.csv: status is not assigned in cases.csv: 1 row(s) (aug-05-001)"])):
            self.assertEqual(self.refused(listed=listed), want, listed)

    def test_case_refusals(self):
        head = "J06 cases.csv: the header repeats a column or lacks one of case_id, purpose, subtype, label, status, text"
        last = AUG_CASES_BYTES.split(b"\n")[-2] + b"\n"
        for cases, want in (
                (AUG_CASES_BYTES.replace(b"caf\xc3\xa9", b"caf\xe9"),
                 ["J06 cases.csv: UnicodeDecodeError (not UTF-8 csv)"]),
                (AUG_CASES_BYTES.replace(b",assigned,, IgNoRe  4ll pRiOr \n", b',assigned,,"IgNoRe"x\n'),
                 ["J06 cases.csv: csv.Error (not UTF-8 csv)"]),
                (b"", [head]),
                (AUG_CASES_BYTES.replace(b",status,", b",state,", 1), [head]),
                (AUG_CASES_BYTES.replace(b",auto_flags,", b",position,", 1), [head]),
                (b"\xef\xbb\xbf" + AUG_CASES_BYTES, [head]),  # BOM: the first column is no longer case_id
                (AUG_CASES_BYTES + b"aug-05-002,x\n", ["J06 cases.csv data record 6: 2 fields, the header has 16"]),
                (AUG_CASES_BYTES + last, ["J06 cases.csv data record 6: duplicate case_id"]),
                (AUG_CASES_BYTES.replace(b",b1,train,0,", b",b1,train,2,"),
                 ["J06 cases.csv data record 4: label is not 0 or 1"]),
                (AUG_CASES_BYTES + b"x\n" * 12,
                 [f"J06 cases.csv data record {n}: 1 fields, the header has 16" for n in range(6, 16)]
                 + ["J06 cases.csv: 2 more malformed record(s)"]),
                # a usable row must be an assigned train case of a trained subtype with a text
                (AUG_CASES_BYTES.replace(b",llm,v,assigned,,Dear", b",llm,v,review_excluded,,Dear"),
                 ["J07 train_aug.csv: status is not assigned in cases.csv: 1 row(s) (aug-04-001)"]),
                (AUG_CASES_BYTES.replace(b":001,train,AUG-04,", b":001,dev,AUG-04,"),
                 ["J07 train_aug.csv: purpose is not train in cases.csv: 1 row(s) (aug-04-001)"]),
                (AUG_CASES_BYTES.replace(b",AUG-01,case,train,0,", b",AUG-01,homoglyph,train,0,"),
                 ["J07 train_aug.csv: held-out subtype homoglyph in cases.csv: 1 row(s) (aug-01-case-000003)"]),
                # the text is hashed as written: a CRLF inside the field is another text
                (AUG_CASES_BYTES.replace(b'""\n\nsecond', b'""\r\n\r\nsecond'),
                 ["J04 train_aug.csv: aug-03-000011 text_sha256_12 differs from the raw text"]),
                (AUG_CASES_BYTES.replace(b",Dear World\n", b",Dear  World\n"),
                 ["J04 train_aug.csv: aug-04-001 text_sha256_12 differs from the raw text"]),
                # ... and so is another letter case, one space instead of two, or a missing edge space
                (AUG_CASES_BYTES.replace(b",Dear World\n", b",dear world\n"),
                 ["J04 train_aug.csv: aug-04-001 text_sha256_12 differs from the raw text"]),
                (AUG_CASES_BYTES.replace(b",, IgNoRe  4ll pRiOr \n", b",, IgNoRe 4ll pRiOr \n"),
                 ["J04 train_aug.csv: aug-01-case-000003 text_sha256_12 differs from the raw text"]),
                (AUG_CASES_BYTES.replace(b",, IgNoRe  4ll pRiOr \n", b",,IgNoRe  4ll pRiOr \n"),
                 ["J04 train_aug.csv: aug-01-case-000003 text_sha256_12 differs from the raw text"]),
                (AUG_CASES_BYTES.replace(b",, IgNoRe  4ll pRiOr \n", b",, IgNoRe  4ll pRiOr\n"),
                 ["J04 train_aug.csv: aug-01-case-000003 text_sha256_12 differs from the raw text"])):
            self.assertNotEqual(cases, AUG_CASES_BYTES)  # the replacement found its bytes
            self.assertEqual(self.refused(cases=cases), want, cases[-60:])
        # an empty text is refused even when its hash (sha256 of nothing) is listed
        empty = AUG_LIST_BYTES.replace(b"86923eba67a9", b"e3b0c44298fc")
        self.assertNotEqual(empty, AUG_LIST_BYTES)
        self.assertEqual(self.refused(AUG_CASES_BYTES.replace(b",Dear World\n", b",\n"), empty),
                         ["J07 train_aug.csv: empty text in cases.csv: 1 row(s) (aug-04-001)"])
        # several violations: every rule reports, at most five IDs each
        head_line = AUG_CASES_BYTES.split(b"\n")[0] + b"\n"
        rows = "".join(f"aug-04-{n:03d},aug-04-{n:03d},new,s,dev,AUG-04,b1,train,0,,,llm,v,reserve,,abc\n"
                       for n in range(1, 7))
        listed = AUG_LIST_HEAD + "".join(f"aug-04-{n:03d},0,ba7816bf8f01\n" for n in range(1, 7)).encode("ascii")
        ids = "aug-04-001, aug-04-002, aug-04-003, aug-04-004, aug-04-005, ..."
        self.assertEqual(self.refused(head_line + rows.encode("ascii"), listed),
                         [f"J07 train_aug.csv: status is not assigned in cases.csv: 6 row(s) ({ids})",
                          f"J07 train_aug.csv: purpose is not train in cases.csv: 6 row(s) ({ids})"])

    def test_refusals_stay_short(self):
        """A refusal is bounded: ten per-row errors, then one count per code; no line over 300 characters."""
        def unknown(n):
            return b"".join(f"aug-00-{i:06d},0,000000000000\n".encode("ascii") for i in range(1, n + 1))

        j02 = [f"J02 train_aug.csv: aug-00-{i:06d} is not a row of the split" for i in range(1, 11)]
        self.assertEqual(self.refused(listed=AUG_LIST_HEAD + unknown(10)), j02)  # ten errors: all listed
        self.assertEqual(self.refused(listed=AUG_LIST_HEAD + unknown(11)),
                         j02 + ["J02 train_aug.csv: 1 more error(s) of this kind"])
        self.assertEqual(self.refused(listed=AUG_LIST_HEAD + unknown(10000)),
                         j02 + ["J02 train_aug.csv: 9990 more error(s) of this kind"])
        # other codes after the tenth error are counted per code; the J07 lines (one per rule) are kept
        mixed = (AUG_LIST_HEAD + unknown(11) + b"aug-01-case-000003,1,745b04798cbd\n"
                 + f"aug-01-space-000007,0,{sha(b'x  y')[:12]}\n".encode("ascii")
                 + b"aug-03-000011,1,000000000000\n" + b"aug-04-001,0,86923eba67a9\n" + b"aug-04-002,0,86923eba67a9\n")
        self.assertEqual(self.refused(listed=mixed), j02 + [
            "J02 train_aug.csv: 2 more error(s) of this kind",
            "J03 train_aug.csv: 1 more error(s) of this kind",
            "J04 train_aug.csv: 1 more error(s) of this kind",
            "J07 train_aug.csv: status is not assigned in cases.csv: 1 row(s) (aug-01-space-000007)"])
        # an ID of any length cannot make a long line: cut at 300 characters, marked with "..."
        long_id = "aug-09-" + "9" * 400
        row = f"{long_id},0,000000000000\n".encode("ascii")
        j02_cut = ("J02 train_aug.csv: " + long_id)[:297] + "..."
        self.assertEqual(self.refused(listed=AUG_LIST_HEAD + row), [j02_cut])
        self.assertEqual(self.refused(listed=AUG_LIST_HEAD + row + row), [
            j02_cut, ("J01 train_aug.csv: IDs not strictly ascending at " + long_id)[:297] + "...", j02_cut])
        self.assertEqual(len(j02_cut), 300)
        cases = AUG_CASES_BYTES.replace(b"aug-05-001,aug-05-001,", long_id.encode("ascii") + b",aug-05-001,")
        listed = AUG_LIST_HEAD + f"{long_id},1,{sha(b'caf' + bytes([0xc3, 0xa9]))[:12]}\n".encode("ascii")
        self.assertEqual(self.refused(cases, listed), [
            ("J07 train_aug.csv: status is not assigned in cases.csv: 1 row(s) (" + long_id)[:297] + "..."])
        # 300 characters exactly are not cut
        fits = "aug-09-" + "9" * (300 - len("J02 train_aug.csv:  is not a row of the split") - len("aug-09-"))
        self.assertEqual(self.refused(listed=AUG_LIST_HEAD + f"{fits},0,000000000000\n".encode("ascii")),
                         [f"J02 train_aug.csv: {fits} is not a row of the split"])
        self.assertEqual(len(f"J02 train_aug.csv: {fits} is not a row of the split"), 300)

    def test_load_from_folder(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            files = make_aug_files(tmp, script=None)
            d = tmp / AUG_DIR
            pins = {n: sha(b) for n, b in files.items()}
            self.assertEqual(sorted(pins), sorted(train.AUG_FILES))
            self.assertEqual((pins["train_aug.csv"], pins["test_overlap.json"]), (AUG_LIST_SHA256, AUG_OVERLAP_SHA256))
            random.seed(5)
            torch.manual_seed(5)
            before = (random.getstate(), torch.get_rng_state().clone())
            self.assertEqual(train.load_augmentation(d, pins), AUG_ROWS)
            self.assertEqual(random.getstate(), before[0])  # no draw from the global generators
            self.assertTrue(torch.equal(torch.get_rng_state(), before[1]))
            for n in sorted(files):  # every file is pinned, also the ones that are not parsed
                with self.assertRaises(data_lists.ListError) as cm:
                    train.load_augmentation(d, {**pins, n: "0" * 64})
                self.assertEqual(cm.exception.errors, [f"J09 data/cases/train: changed after the preflight: {n}"])
            with self.assertRaises(data_lists.ListError) as cm:
                train.load_augmentation(d, {n: h for n, h in pins.items() if n != "review.csv"})
            self.assertEqual(cm.exception.errors, ["J09 data/cases/train: changed after the preflight: review.csv"])
            # edited after the preflight: refused by the pin before the (now malformed) file is parsed
            (d / "cases.csv").write_bytes(files["cases.csv"] + b"x\n")
            (d / "train_aug.csv").write_bytes(files["train_aug.csv"] + b"aug-05-001,1,000000000000\n")
            with self.assertRaises(data_lists.ListError) as cm:
                train.load_augmentation(d, pins)
            self.assertEqual(cm.exception.errors,
                             ["J09 data/cases/train: changed after the preflight: cases.csv, train_aug.csv"])
            (d / "test_overlap.json").unlink()
            with self.assertRaises(data_lists.ListError) as cm:
                train.load_augmentation(d, pins)
            self.assertEqual(cm.exception.errors, ["J06 data/cases/train: cannot read (FileNotFoundError)"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestAugmentationPreflight(unittest.TestCase):
    """P13 and the stage-4 git checks on a temporary git repo (synthetic augmentation set, stand-in builders)."""

    setUp, tearDown = TestPreflightHermetic.setUp, TestPreflightHermetic.tearDown
    git, ready, tree = TestPreflightHermetic.git, TestPreflightHermetic.ready, TestPreflightHermetic.tree
    META = f"{AUG_DIR}/cases_meta.json"

    def commit(self):
        self.git("add", "-A")
        self.git("-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-q", "--allow-empty",
                 "-m", "x")

    def p13(self, commit=True, **kw):
        """Write the augmentation set, commit it, return train.verify_augmentation(root)."""
        make_aug_files(self.root, **kw)
        if commit:
            self.commit()
        return train.verify_augmentation(self.root)

    def test_verified_block(self):
        files = make_aug_files(self.root)
        self.commit()
        before = self.tree()
        errors, info = train.verify_augmentation(self.root)
        self.assertEqual(errors, [])
        self.assertEqual(info, {
            "dir": "data/cases/train",
            "files_sha256": {"cases.csv": sha(AUG_CASES_BYTES), "review.csv": sha(AUG_REVIEW_BYTES),
                             "train_aug.csv": AUG_LIST_SHA256, "cases_meta.json": sha(files["cases_meta.json"]),
                             "test_overlap.json": AUG_OVERLAP_SHA256},
            "label_counts": {"0": 2, "1": 1},
            "build_train_aug_sha256_lf": sha(STANDIN_BUILD_TRAIN_AUG)})
        self.assertEqual(self.tree(), before)  # reads only
        # the stand-in really tells `--check` from the repo root apart from anything else
        script = str(self.root / "scripts" / "build_train_aug.py")
        self.assertEqual(subprocess.run([sys.executable, script], cwd=str(self.root)).returncode, 3)
        self.assertEqual(subprocess.run([sys.executable, script, "--check"], cwd=str(self.root / "scripts")).returncode,
                         3)

    def test_missing_files(self):
        make_aug_files(self.root)
        self.commit()
        for rel in AUG_RELS:
            data = (self.root / rel).read_bytes()
            (self.root / rel).unlink()
            errors, info = train.verify_augmentation(self.root)
            self.assertEqual((len(errors), info), (1, None), rel)
            self.assertTrue(errors[0].startswith(f"P13 augmentation: missing: {rel} (scripts/build_train_aug.py "
                                                 "writes train_aug.csv only when"), errors)
            (self.root / rel).write_bytes(data)
        self.assertEqual(train.verify_augmentation(self.root)[0], [])
        for rel in (AUG_RELS[4], AUG_RELS[5]):  # a folder in the place of a file is a missing file
            data = (self.root / rel).read_bytes()
            (self.root / rel).unlink()
            (self.root / rel).mkdir()
            errors, info = train.verify_augmentation(self.root)
            self.assertEqual((len(errors), info), (1, None), rel)
            self.assertTrue(errors[0].startswith(f"P13 augmentation: missing: {rel} ("), errors)
            (self.root / rel).rmdir()
            (self.root / rel).write_bytes(data)
        self.assertEqual(train.verify_augmentation(self.root)[0], [])
        shutil.rmtree(self.root / "data" / "cases")
        errors, _ = train.verify_augmentation(self.root)
        self.assertTrue(errors[0].startswith("P13 augmentation: missing: " + ", ".join(AUG_RELS[:5]) + " ("), errors)

    def test_git_tracked_and_unmodified(self):
        errors, info = self.p13(commit=False, script=FAILING_BUILD_TRAIN_AUG)  # nothing committed yet
        self.assertEqual((errors, info),
                         (["P13 augmentation: not tracked by git: " + ", ".join(sorted(AUG_RELS))], None))
        self.assertEqual(self.p13()[0], [])
        with open(self.root / AUG_DIR / "test_overlap.json", "ab") as f:
            f.write(b"\n")
        want = [f"P13 augmentation: modified in the working tree or index: {AUG_DIR}/test_overlap.json"]
        self.assertEqual(train.verify_augmentation(self.root), (want, None))
        self.git("add", "-A")  # staged but not committed: still refused
        self.assertEqual(train.verify_augmentation(self.root), (want, None))
        self.commit()
        self.assertEqual(train.verify_augmentation(self.root)[0], [])
        for n in ("cases.csv", "review.csv", "train_aug.csv"):  # a pinned file: the git check and the pin report
            data = (self.root / AUG_DIR / n).read_bytes()
            (self.root / AUG_DIR / n).write_bytes(data + b"x\n")
            self.assertEqual(train.verify_augmentation(self.root), ([
                f"P13 augmentation: modified in the working tree or index: {AUG_DIR}/{n}",
                f"P13 {AUG_DIR}/{n}: sha256 {sha(data + bytes([0x78, 0x0a]))[:12]} != cases_meta {sha(data)[:12]}"],
                None))
            (self.root / AUG_DIR / n).write_bytes(data)
        self.assertEqual(train.verify_augmentation(self.root)[0], [])
        with open(self.root / "scripts" / "build_train_aug.py", "ab") as f:
            f.write(b"# edit\n")
        errors, _ = train.verify_augmentation(self.root)
        self.assertEqual(errors[0], "P13 augmentation: modified in the working tree or index: "
                                    "scripts/build_train_aug.py")
        self.assertEqual(len(errors), 2)
        self.assertTrue(errors[1].startswith("P13 scripts/build_train_aug.py: sha256_lf "), errors)
        with mock.patch.object(data_lists, "git_status", return_value=None):
            self.assertIn("P13 augmentation: git status failed", train.verify_augmentation(self.root)[0])

    def test_meta_state(self):
        make_aug_files(self.root)
        self.commit()
        dirty = f"P13 augmentation: modified in the working tree or index: {self.META}, scripts/build_train_aug.py"
        for edit, want in (
                (lambda m: m.update(ready_for_training=False),
                 [f"P13 {self.META}: ready_for_training is false, condition C needs true"]),
                (lambda m: m.update(ready_for_training="true"),
                 [f'P13 {self.META}: ready_for_training is "true", condition C needs true']),
                (lambda m: m.update(ready_for_training=1),
                 [f"P13 {self.META}: ready_for_training is 1, condition C needs true"]),
                (lambda m: m["review"].update(pending=2),
                 [f"P13 {self.META}: review.pending is 2, condition C needs 0 (decide every case in "
                  "data/cases/train/review.csv, then rebuild)"]),
                (lambda m: m["review"].update(pending=False),
                 [f"P13 {self.META}: review.pending is false, condition C needs 0 (decide every case in "
                  "data/cases/train/review.csv, then rebuild)"]),
                (lambda m: m["test_overlap"].update(status="pending"),
                 [f'P13 {self.META}: test_overlap.status is "pending", condition C needs "applied" (run '
                  "scripts/check_cases_vs_test.py, then scripts/build_train_aug.py)"]),
                (lambda m: (m.update(ready_for_training=False), m["review"].update(pending=1),
                            m["test_overlap"].update(status="pending")),
                 [f"P13 {self.META}: ready_for_training is false, condition C needs true",
                  f"P13 {self.META}: review.pending is 1, condition C needs 0 (decide every case in "
                  "data/cases/train/review.csv, then rebuild)",
                  f'P13 {self.META}: test_overlap.status is "pending", condition C needs "applied" (run '
                  "scripts/check_cases_vs_test.py, then scripts/build_train_aug.py)"])):
            # not committed: the first error is the git one; the failing stand-in shows --check is not reached
            errors, info = self.p13(commit=False, edit=edit, script=FAILING_BUILD_TRAIN_AUG)
            self.assertEqual((errors, info), ([dirty] + want, None))
        unreadable = [f"P13 {self.META}: unreadable or missing keys"]
        for raw in (b"{", b"[]\n", b"\xff\n", b"null\n"):
            self.assertEqual(self.p13(files={"cases_meta.json": raw}), (unreadable, None), raw)
        for edit in (lambda m: m.pop("ready_for_training"), lambda m: m.pop("review"),
                     lambda m: m["review"].pop("pending"), lambda m: m["test_overlap"].pop("status"),
                     lambda m: m.update(test_overlap="applied"), lambda m: m.pop("outputs"),
                     lambda m: m.update(outputs=[]), lambda m: m.update(outputs=None),
                     lambda m: m.pop("script"), lambda m: m["script"].pop("sha256_lf"),
                     lambda m: m["script"].update(sha256_lf=7)):
            errors, info = self.p13(commit=False, edit=edit)
            self.assertEqual((errors[1:], info), (unreadable, None))
        # an "outputs" entry that is missing or has no SHA-256 is named, one line per file, and the other pins
        # are still compared (the failing stand-in shows --check is not reached)
        no_list = (f"P13 {self.META}: outputs has no train_aug.csv entry (scripts/build_train_aug.py records it "
                   "only for a set that is ready for training)")
        for edit, want in (
                (lambda m: m["outputs"].pop("train_aug.csv"), [no_list]),
                (lambda m: m["outputs"].update({"train_aug.csv": None}), [no_list]),
                (lambda m: m["outputs"].pop("cases.csv"), [f"P13 {self.META}: outputs has no cases.csv entry"]),
                (lambda m: m["outputs"]["review.csv"].pop("sha256"),
                 [f"P13 {self.META}: outputs review.csv has no sha256 string"]),
                (lambda m: m["outputs"]["cases.csv"].update(sha256=None),
                 [f"P13 {self.META}: outputs cases.csv has no sha256 string"]),
                (lambda m: m["outputs"]["train_aug.csv"].update(sha256=7),
                 [f"P13 {self.META}: outputs train_aug.csv has no sha256 string"]),
                (lambda m: m["outputs"].update({"review.csv": "0" * 64}),
                 [f"P13 {self.META}: outputs review.csv has no sha256 string"]),
                (lambda m: (m["outputs"].pop("review.csv"), m["outputs"].pop("train_aug.csv"),
                            m["outputs"]["cases.csv"].update(sha256="0" * 64)),
                 [f"P13 {AUG_DIR}/cases.csv: sha256 {sha(AUG_CASES_BYTES)[:12]} != cases_meta 000000000000",
                  f"P13 {self.META}: outputs has no review.csv entry", no_list])):
            errors, info = self.p13(commit=False, edit=edit, script=FAILING_BUILD_TRAIN_AUG)
            self.assertEqual((errors[1:], info), (want, None))

    def test_stale_list_next_to_a_meta_that_is_not_ready(self):
        """build_train_aug.py records outputs["train_aug.csv"] only for a set that is ready for training. A
        train_aug.csv left next to the meta of a set that is not ready is refused with the state of the set."""
        def not_ready(m):
            m.update(ready_for_training=False)
            m["review"].update(pending=3, keep=0)
            m["test_overlap"].update(status="pending")
            del m["outputs"]["train_aug.csv"]

        errors, info = self.p13(edit=not_ready, script=FAILING_BUILD_TRAIN_AUG)  # committed: no git error
        self.assertEqual((errors, info), ([
            f"P13 {self.META}: ready_for_training is false, condition C needs true",
            f"P13 {self.META}: review.pending is 3, condition C needs 0 (decide every case in "
            "data/cases/train/review.csv, then rebuild)",
            f'P13 {self.META}: test_overlap.status is "pending", condition C needs "applied" (run '
            "scripts/check_cases_vs_test.py, then scripts/build_train_aug.py)",
            f"P13 {self.META}: outputs has no train_aug.csv entry (scripts/build_train_aug.py records it only for "
            "a set that is ready for training)"], None))
        meta = json.loads((self.root / self.META).read_bytes())
        self.assertEqual(sorted(meta["outputs"]), ["cases.csv", "review.csv"])  # the shape under test

    def test_sha_pins(self):
        files = make_aug_files(self.root, script=None)
        for n in ("cases.csv", "review.csv", "train_aug.csv"):
            errors, info = self.p13(edit=lambda m: m["outputs"][n].update(sha256="0" * 64),
                                    script=FAILING_BUILD_TRAIN_AUG)
            self.assertEqual((errors, info), (
                [f"P13 {AUG_DIR}/{n}: sha256 {sha(files[n])[:12]} != cases_meta 000000000000"], None))
        self.assertEqual(sha(AUG_LIST_BYTES)[:12], "3db72cfb5421")
        errors, info = self.p13(edit=lambda m: m["script"].update(sha256_lf="1" * 64),
                                script=FAILING_BUILD_TRAIN_AUG)
        self.assertEqual((errors, info), ([
            f"P13 scripts/build_train_aug.py: sha256_lf {sha(FAILING_BUILD_TRAIN_AUG)[:12]} != cases_meta script "
            "111111111111 (script changed after the augmentation set was built: rebuild it)"], None))
        # the data files are pinned byte for byte: a CRLF copy of review.csv is not the file the meta records
        crlf_review = AUG_REVIEW_BYTES.replace(b"\n", b"\r\n")
        errors, info = self.p13(files={"review.csv": crlf_review},
                                edit=lambda m: m["outputs"]["review.csv"].update(sha256=sha(AUG_REVIEW_BYTES)),
                                script=FAILING_BUILD_TRAIN_AUG)
        self.assertEqual((errors, info), ([f"P13 {AUG_DIR}/review.csv: sha256 {sha(crlf_review)[:12]} != cases_meta "
                                           f"{sha(AUG_REVIEW_BYTES)[:12]}"], None))
        self.assertEqual((self.root / AUG_DIR / "review.csv").read_bytes(), crlf_review)
        # LF-normalised: a CRLF checkout of the builder still equals the LF hash recorded in the meta
        crlf = STANDIN_BUILD_TRAIN_AUG.replace(b"\n", b"\r\n")
        self.assertNotEqual(sha(crlf), sha(STANDIN_BUILD_TRAIN_AUG))
        errors, info = self.p13(script=crlf)
        self.assertEqual((errors, info["build_train_aug_sha256_lf"]), ([], sha(STANDIN_BUILD_TRAIN_AUG)))

    def test_check_subprocess_and_join(self):
        self.assertEqual(self.p13(script=FAILING_BUILD_TRAIN_AUG), ([CHECK_FAILED], None))
        # a join error is reported with its code and stops before --check (the failing stand-in is not reached)
        bad = AUG_LIST_BYTES.replace(b"86923eba67a9", b"86923eba67aa")
        self.assertNotEqual(bad, AUG_LIST_BYTES)
        self.assertEqual(self.p13(files={"train_aug.csv": bad}, script=FAILING_BUILD_TRAIN_AUG),
                         (["P13 J04 train_aug.csv: aug-04-001 text_sha256_12 differs from the raw text"], None))
        self.assertEqual(self.p13(files={"train_aug.csv": AUG_LIST_HEAD}, script=FAILING_BUILD_TRAIN_AUG),
                         (["P13 J08 train_aug.csv: no rows (condition C needs augmentation rows)"], None))
        # 10,000 rows that are not cases: ten lines and one count, not 10,000 lines
        many = AUG_LIST_HEAD + b"".join(f"zz-{i:06d},0,000000000000\n".encode("ascii") for i in range(10000))
        self.assertEqual(self.p13(files={"train_aug.csv": many}, script=FAILING_BUILD_TRAIN_AUG), (
            [f"P13 J02 train_aug.csv: zz-{i:06d} is not a row of the split" for i in range(10)]
            + ["P13 J02 train_aug.csv: 9990 more error(s) of this kind"], None))
        # the FAIL lines of a failing --check: the first five, each ASCII and cut at 300 characters
        errors, info = self.p13(script=NOISY_BUILD_TRAIN_AUG)
        self.assertEqual((errors, info), ([
            "P13 scripts/build_train_aug.py --check exit 2: FAIL K00 \\ufffd byte | "
            + " | ".join(f"FAIL K0{n} " + "x" * 288 + "..." for n in range(1, 5))], None))
        self.assertEqual(len(errors[0]), len("P13 scripts/build_train_aug.py --check exit 2: FAIL K00 \\ufffd byte")
                         + 4 * (3 + 300))
        self.assertEqual(self.p13()[0], [])
        real, calls = subprocess.run, []

        def spy(cmd, *a, **kw):
            if cmd[-1] == "--check":
                calls.append((list(cmd), a, kw))
            return real(cmd, *a, **kw)

        # the check runs like L08: the same interpreter, from the repo root, in UTF-8 mode, at most 600 s
        with mock.patch.object(subprocess, "run", side_effect=spy), mock.patch.dict(os.environ, {"PYTHONUTF8": "0"}):
            self.assertEqual(train.verify_augmentation(self.root)[0], [])
        self.assertEqual(len(calls), 1)
        cmd, a, kw = calls[0]
        self.assertEqual((cmd, a), ([sys.executable, str(self.root / "scripts" / "build_train_aug.py"), "--check"], ()))
        self.assertEqual((kw["cwd"], kw["timeout"], kw["capture_output"], kw["env"]["PYTHONUTF8"]),
                         (str(self.root), 600, True, "1"))
        self.assertEqual(data_lists.CHECK_TIMEOUT_S, 600)

        def slow(cmd, *a, **kw):
            if cmd[-1] == "--check":
                raise subprocess.TimeoutExpired(cmd, kw["timeout"])
            return real(cmd, *a, **kw)

        with mock.patch.object(subprocess, "run", side_effect=slow):
            self.assertEqual(train.verify_augmentation(self.root),
                             (["P13 scripts/build_train_aug.py --check: no result within 600 s"], None))

    # ---- the whole preflight

    def official(self, **kw):
        d = make_repo_files(self.root, False, where="official", script=STANDIN_BUILD_LISTS)
        # 11,056 = 691 * 16 train_clean rows: N = 2073, and counting three more rows would already give 2076
        meta = json.loads((d / "lists_meta.json").read_text(encoding="ascii"))
        meta["outputs"]["train_clean.csv"]["rows"] = 11056
        (d / "lists_meta.json").write_bytes(json.dumps(meta).encode("ascii"))
        make_aug_files(self.root, **kw)
        self.commit()

    def preflight(self, **kw):
        args = types.SimpleNamespace(**{"model": "bert", "smoke": False, "seed": 42, "lists": "official",
                                        "rerun": False, "condition": "C", **kw})
        with contextlib.ExitStack() as st:
            for p in self.ready():
                st.enter_context(p)
            errors, ctx = train.preflight(args, CFG, self.root)
        self.assertFalse((self.root / "runs").exists())
        return errors, ctx

    def test_condition_c_passes_with_the_budget_of_a(self):
        self.official()
        before = self.tree()
        errors, ctx = self.preflight()
        self.assertEqual(errors, [])
        self.assertEqual((ctx["stage"], ctx["condition"], ctx["seed"], ctx["run_id"], ctx["run_dir"]),
                         ("s4", "C", 42, "s4-bert-C-s42", self.root / "runs" / "s4-bert-C-s42"))
        # N from the 11,056 train_clean rows of lists_meta.json alone: 3 * (11056 / 16) = 3 * 691 = 2073;
        # with the 3 augmentation rows it would be 3 * ceil(11059 / 16) = 3 * 692 = 2076 (and P10 would refuse)
        self.assertEqual((ctx["lists"]["train_clean_rows"], train.budget(11056 + 3, 3)), (11056, 2076))
        self.assertEqual((ctx["n_total"], ctx["eval_at"], ctx["n_run"]), (2073, [691, 1382, 2073], 2073))
        self.assertEqual(ctx["aug_dir"], self.root / "data" / "cases" / "train")
        self.assertEqual(sorted(ctx["augmentation"]),
                         ["build_train_aug_sha256_lf", "dir", "files_sha256", "label_counts"])
        self.assertEqual((ctx["augmentation"]["label_counts"], ctx["augmentation"]["files_sha256"]["train_aug.csv"]),
                         ({"0": 2, "1": 1}, AUG_LIST_SHA256))
        errors, a = self.preflight(condition="A")  # condition A on the same repo: as before stage 4
        self.assertEqual((errors, a["stage"], a["condition"], a["run_id"]), ([], "s2", "A", "s2-bert-A-s42"))
        self.assertEqual(sorted(a), ["condition", "eval_at", "list_dir", "lists", "n_run", "n_total", "run_dir",
                                     "run_id", "seed", "stage"])
        self.assertEqual(sorted(set(ctx) - set(a)), ["aug_dir", "augmentation"])
        same = ("seed", "lists", "list_dir", "n_total", "eval_at", "n_run")
        self.assertEqual({k: ctx[k] for k in same}, {k: a[k] for k in same})
        self.assertEqual(self.tree(), before)  # nothing created

    def test_p13_is_for_condition_c_only(self):
        make_repo_files(self.root, False, where="official", script=STANDIN_BUILD_LISTS)
        self.commit()  # no augmentation set at all
        self.assertEqual(self.preflight(condition="A")[0], [])
        errors, ctx = self.preflight()
        self.assertEqual((len(errors), ctx), (1, None))
        self.assertTrue(errors[0].startswith("P13 augmentation: missing: data/cases/train/cases.csv, "), errors)
        make_aug_files(self.root, script=FAILING_BUILD_TRAIN_AUG)  # a set that fails its --check
        self.commit()
        self.assertEqual(self.preflight(condition="A")[0], [])
        self.assertEqual(self.preflight(), ([CHECK_FAILED], None))

    def test_p13_is_reported_next_to_a_list_failure(self):
        errors, ctx = self.preflight()  # neither official lists nor an augmentation set: both are reported
        self.assertEqual(([e.split()[0] for e in errors], ctx), (["P07", "P13"], None))
        self.assertTrue(errors[0].startswith("P07 L01 official lists: missing in data/processed/: "), errors)
        self.assertTrue(errors[1].startswith("P13 augmentation: missing: data/cases/train/cases.csv, "), errors)
        make_aug_files(self.root, script=FAILING_BUILD_TRAIN_AUG)  # still no lists; the set fails its --check
        self.commit()
        errors, ctx = self.preflight()
        self.assertEqual((errors[1:], ctx), ([CHECK_FAILED], None))
        self.assertTrue(errors[0].startswith("P07 L01 "), errors)

    def test_stage4_git_checks(self):
        self.official()
        (self.root / "notes.txt").write_text("x")
        self.commit()
        with open(self.root / "notes.txt", "a", encoding="ascii") as f:
            f.write("y")
        with open(self.root / "scripts" / "train.py", "a", encoding="ascii") as f:
            f.write("# edit\n")
        self.assertEqual(self.preflight()[0], [
            "P08 stage 4 needs unmodified project files; modified: scripts/train.py",
            "P09 stage 4 refuses modified tracked files outside runs/: 1 (notes.txt)"])
        self.assertEqual(self.preflight(condition="A")[0], [  # the stage-2 wording is unchanged
            "P08 stage 2 needs unmodified project files; modified: scripts/train.py",
            "P09 stage 2 refuses modified tracked files outside runs/: 1 (notes.txt)"])
        self.commit()
        self.git("rm", "-q", "--cached", "scripts/eval_lib.py")  # the file stays on disk, no longer tracked
        self.git("-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-q", "-m", "untrack")
        self.assertEqual(self.preflight()[0],
                         ["P08 stage 4 needs git-tracked project files; untracked: scripts/eval_lib.py"])
        self.assertEqual(self.preflight(condition="A")[0],
                         ["P08 stage 2 needs git-tracked project files; untracked: scripts/eval_lib.py"])

    def test_condition_c_keeps_the_stage2_rules(self):
        self.official()
        errors, _ = self.preflight(lists="trial", seed=7)
        self.assertIn("P03 --lists trial is allowed only with --smoke (README 7.1: stage 4 uses the official lists)",
                      errors)
        self.assertIn(f"P03 seed 7 is not in configs/train.json seeds {CFG['seeds']}", errors)
        errors, _ = self.preflight(lists="trial", condition="A")
        self.assertIn("P03 --lists trial is allowed only with --smoke (README 7.1: stage 2 uses the official lists)",
                      errors)
        (self.root / "runs" / "s4-bert-C-s42").mkdir(parents=True)
        (self.root / "runs" / "s4-bert-C-s42.lock").write_text("pid 1\n")
        args = types.SimpleNamespace(model="bert", smoke=False, seed=42, lists="official", rerun=False, condition="C")
        with contextlib.ExitStack() as st:
            for p in self.ready():
                st.enter_context(p)
            errors, _ = train.preflight(args, CFG, self.root)
            self.assertEqual([e.split()[0] for e in errors], ["P06", "P12"])
            self.assertIn("runs/s4-bert-C-s42/ exists; pass --rerun", errors[0])
            errors, a = train.preflight(types.SimpleNamespace(**{**vars(args), "condition": "A"}), CFG, self.root)
            self.assertEqual((errors, a["run_id"]), ([], "s2-bert-A-s42"))  # another run ID: not blocked

    def test_preflight_only_cli(self):
        self.official()
        before = self.tree()
        out = io.StringIO()
        with contextlib.ExitStack() as st:
            for p in self.ready() + [mock.patch.object(train, "ROOT", self.root), contextlib.redirect_stdout(out)]:
                st.enter_context(p)
            self.assertEqual(train.main(["--model", "bert", "--seed", "42", "--condition", "C", "--preflight-only"]), 0)
            self.assertEqual(train.main(["--model", "bert", "--seed", "42", "--preflight-only"]), 0)
            self.assertEqual(self.tree(), before)  # nothing created
            (self.root / AUG_DIR / "train_aug.csv").unlink()
            self.assertEqual(train.main(["--model", "bert", "--seed", "42", "--condition", "C", "--preflight-only"]), 1)
            self.assertEqual(train.main(["--model", "bert", "--seed", "42", "--preflight-only"]), 1)  # P09
        text = out.getvalue()
        self.assertIn("preflight passed: s4-bert-C-s42 would start (lists official, git tracked check True, N 2073, "
                      "updates 2073, eval at [691, 1382, 2073], augmentation 3 rows); --preflight-only: nothing "
                      "created\n", text)
        self.assertIn("preflight passed: s2-bert-A-s42 would start (lists official, git tracked check True, N 2073, "
                      "updates 2073, eval at [691, 1382, 2073]); --preflight-only: nothing created\n", text)
        self.assertIn("FAIL P13 augmentation: missing: data/cases/train/train_aug.csv (", text)
        self.assertIn("preflight failed: 2 error(s); nothing created\n", text)
        self.assertFalse((self.root / "runs").exists())


RUN_JSON_KEYS = ["attempt", "budget", "checkpoint", "code", "condition", "config", "data", "determinism", "env",
                 "failure", "finished_utc", "lists", "model", "model_key", "optimizer", "resources", "run_id",
                 "schema", "seed", "selection", "stage", "started_utc", "status", "wall_s"]


class TestRunConditions(unittest.TestCase):
    """run() for conditions A and C: 16 train_clean rows, the real augmentation loader, patched model and fit."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "runs").mkdir()
        self.files = make_aug_files(self.root, script=None)
        self.pins = {n: sha(b) for n, b in self.files.items()}

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def go(self, condition):
        stage = {"A": "s2", "C": "s4"}[condition]
        run_id = f"{stage}-bert-{condition}-s42"
        # N = 3 * ceil(16 / 16) = 3; with the 3 augmentation rows counted it would be 3 * ceil(19 / 16) = 6
        ctx = {"stage": stage, "condition": condition, "seed": 42, "run_id": run_id,
               "run_dir": self.root / "runs" / run_id,
               "lists": {"mode": "official", "dir": "data/processed", "git_tracked_checked": True},
               "list_dir": None, "n_total": train.budget(16, 3), "eval_at": [1, 2, 3], "n_run": 3}
        if condition == "C":
            # label_counts here are not the counts of the set: the record must carry those of the rows it loaded
            ctx.update(aug_dir=self.root / "data" / "cases" / "train",
                       augmentation={"dir": "data/cases/train", "files_sha256": dict(self.pins),
                                     "label_counts": {"0": 20, "1": 10}, "build_train_aug_sha256_lf": "f" * 64})
        seen = {"order": [], "encoded": []}
        real_load = train.load_augmentation

        def load(*a):
            seen["order"].append("load_augmentation")
            return real_load(*a)

        def encode(tok, texts, n):
            seen["encoded"].append(list(texts))
            return [[2, 5, 3]] * len(texts), {"max_tokens": 3, "rows_over_max_length": 0}

        def fake_fit(model, key, data, cfg, **kw):
            seen["order"].append("fit")
            seen.update(train_rows=len(data["train_ids"]), train_labels=list(data["train_labels"]),
                        n_total=kw["n_total"], n_run=kw["n_run"], eval_at=kw["eval_at"])
            info = model_specs.save_checkpoint(tiny_model(0), key, kw["ckpt_dir"], kw["run_id"], 1)
            scores = torch.tensor([0.75, 0.25, 0.5, 0.125])
            kw["on_eval"]({"step": 1}, torch.tensor([1.5, -2.0, 0.0, -3.0]), scores,
                          eval_lib.predictions(scores, cfg["threshold"]))
            rows = [{"update": u, "epoch": 1, "examples": 16, "lr": 2e-5, "loss": 1 / u, "grad_norm": 1.0,
                     "seconds": 0.01} for u in range(1, kw["n_run"] + 1)]
            return {"log": rows, "evals": [], "saved": [1], "eval_s": [0.1], "save_s": [0.1], "optimizer": {},
                    "best": {"step": 1, "sha256": info["sha256"], "size": info["size"], "tensors": info["tensors"]}}

        reload = {"state_dict_equal": True, "prediction_flips": 0, "max_abs_margin_diff": 0.0, "passed": True,
                  "seconds": 0.1}
        patches = [
            mock.patch.object(data_lists, "load_rows", return_value=None),
            mock.patch.object(data_lists, "load_training_data", return_value={
                "train": [(f"train:{i:06d}", i % 2, "t") for i in range(16)],
                "validation": [(f"validation:{i:06d}", i % 2, "v", True) for i in range(4)]}),
            mock.patch.object(train, "load_augmentation", side_effect=load),
            mock.patch.object(train, "seed_everything",
                              side_effect=lambda s: seen["order"].append(f"seed_everything({s})")),
            mock.patch.object(train, "apply_determinism", return_value={}),
            mock.patch.object(train, "env_info", return_value={"gpu": {"name": "cpu"}}),
            mock.patch.object(model_specs, "load_tokenizer", return_value=types.SimpleNamespace(pad_token_id=0)),
            mock.patch.object(model_specs, "check_tokenizer", return_value="fp"),
            mock.patch.object(eval_lib, "encode", side_effect=encode),
            mock.patch.object(model_specs, "load_model",
                              side_effect=lambda *a, **k: seen["order"].append("load_model") or mock.MagicMock()),
            mock.patch.object(train, "fit", side_effect=fake_fit),
            mock.patch.object(train, "reload_check", return_value=reload),
            mock.patch.object(train, "code_record",
                              return_value={"git_commit": None, "git_dirty": [], "files_sha256_lf": {}}),
            *(mock.patch(f"torch.cuda.{f}", return_value=1) for f in (
                "init", "reset_peak_memory_stats", "synchronize", "max_memory_allocated", "max_memory_reserved",
                "empty_cache")),
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            st.enter_context(contextlib.redirect_stdout(io.StringIO()))
            st.enter_context(contextlib.redirect_stderr(io.StringIO()))
            code = train.run(types.SimpleNamespace(model="bert", rerun=False), CFG, b"{}\n", ctx)
        self.assertFalse(train.lock_path(ctx["run_dir"]).exists())
        return code, json.loads((ctx["run_dir"] / "run.json").read_text(encoding="ascii")), seen, ctx["run_dir"]

    def test_condition_c_appends_the_augmentation_rows(self):
        code, rec, seen, rd = self.go("C")
        self.assertEqual((code, rec["status"], rec["failure"]), (0, "completed", None))
        self.assertEqual((rec["run_id"], rec["stage"], rec["condition"], rec["seed"]), ("s4-bert-C-s42", "s4", "C", 42))
        # trained rows: the 16 train_clean rows first, then the augmentation rows in train_aug.csv order
        # (texts exactly as in cases.csv: letter case, double and edge spaces, the blank line)
        self.assertEqual(seen["encoded"], [["t"] * 16 + [" IgNoRe  4ll pRiOr ", 'first, "quoted"\n\nsecond line',
                                                         "Dear World"], ["v"] * 4])
        self.assertEqual(seen["encoded"][0][16:], [AUG_TEXT_01, AUG_TEXT_03, AUG_TEXT_04])
        self.assertEqual(seen["train_labels"], [i % 2 for i in range(16)] + [0, 1, 0])
        self.assertEqual(seen["train_rows"], 19)
        # the budget is the one of the 16 train_clean rows
        self.assertEqual((seen["n_total"], seen["n_run"], seen["eval_at"]), (3, 3, [1, 2, 3]))
        self.assertEqual(rec["budget"], {"update_size": 16, "micro_batch": CFG["micro_batch"],
                                         "grad_accum": CFG["grad_accum"], "epochs": 3, "N": 3,
                                         "eval_steps": [1, 2, 3], "updates_run": 3})
        stats = {"max_tokens": 3, "rows_over_max_length": 0}
        self.assertEqual(rec["data"], {
            "train_rows": 19, "train_clean_rows": 16, "augmentation_rows": 3,
            "augmentation": {"dir": "data/cases/train",
                             "files_sha256": {"cases.csv": sha(AUG_CASES_BYTES), "review.csv": sha(AUG_REVIEW_BYTES),
                                              "train_aug.csv": AUG_LIST_SHA256,
                                              "cases_meta.json": sha(self.files["cases_meta.json"]),
                                              "test_overlap.json": AUG_OVERLAP_SHA256},
                             "label_counts": {"0": 2, "1": 1}, "build_train_aug_sha256_lf": "f" * 64},
            "validation_rows": 4, "validation_clean_rows": 4, "truncation": {"train": stats, "validation": stats}})
        self.assertEqual(sorted(rec), RUN_JSON_KEYS)  # the same top-level keys as a condition A record
        # read in the data phase, before the seed is set for the model load (no effect on the head initialisation)
        self.assertEqual(seen["order"][:4], ["load_augmentation", "seed_everything(42)", "load_model", "fit"])
        self.assertEqual(seen["order"].count("load_augmentation"), 1)
        self.assertEqual((rec["checkpoint"]["kept"], rec["checkpoint"]["path"]),
                         (True, "runs/s4-bert-C-s42/checkpoint/model.safetensors"))
        self.assertEqual(sha((rd / "checkpoint" / "model.safetensors").read_bytes()), rec["checkpoint"]["sha256"])
        preds = (rd / "predictions_validation.csv").read_text(encoding="ascii").split("\n")
        self.assertEqual(preds, ["step,sentence_id,label,in_clean,margin,score,prediction,threshold,condition",
                                 "1,validation:000000,0,1,1.5,0.75,1,0.5,C", "1,validation:000001,1,1,-2,0.25,0,0.5,C",
                                 "1,validation:000002,0,1,0,0.5,1,0.5,C", "1,validation:000003,1,1,-3,0.125,0,0.5,C",
                                 ""])
        log = (rd / "log.txt").read_text(encoding="ascii")
        self.assertIn("augmentation: 3 rows of data/cases/train/train_aug.csv (label 1: 1, label 0: 2) appended "
                      "after 16 train_clean rows; N stays 3\n", log)
        self.assertIn("data: train 19 rows, validation 4 rows (4 in_clean)", log)
        for name in train.SMALL_FILES:  # IDs and numbers only
            text = (rd / name).read_text(encoding="ascii")
            self.assertFalse(any(t in text.lower() for t in ("ignore", "prior", "dear world", "second line",
                                                             "quoted")), name)

    def test_condition_a_is_unchanged(self):
        shutil.rmtree(self.root / "data")  # condition A never looks at the augmentation set
        code, rec, seen, rd = self.go("A")
        self.assertEqual((code, rec["status"], rec["failure"]), (0, "completed", None))
        self.assertEqual((rec["run_id"], rec["stage"], rec["condition"]), ("s2-bert-A-s42", "s2", "A"))
        stats = {"max_tokens": 3, "rows_over_max_length": 0}
        self.assertEqual(rec["data"], {"train_rows": 16, "validation_rows": 4, "validation_clean_rows": 4,
                                       "truncation": {"train": stats, "validation": stats}})  # no new key
        self.assertEqual(sorted(rec), RUN_JSON_KEYS)
        self.assertEqual(seen["encoded"], [["t"] * 16, ["v"] * 4])
        self.assertEqual((seen["train_rows"], seen["train_labels"]), (16, [i % 2 for i in range(16)]))
        self.assertEqual((seen["n_total"], seen["n_run"], seen["eval_at"]), (3, 3, [1, 2, 3]))
        self.assertEqual(seen["order"][:3], ["seed_everything(42)", "load_model", "fit"])
        self.assertNotIn("load_augmentation", seen["order"])
        self.assertEqual(rec["checkpoint"]["path"], "runs/s2-bert-A-s42/checkpoint/model.safetensors")
        preds = (rd / "predictions_validation.csv").read_text(encoding="ascii").split("\n")
        self.assertEqual(preds[1:], ["1,validation:000000,0,1,1.5,0.75,1,0.5,A", "1,validation:000001,1,1,-2,0.25,0,0.5,A",
                                     "1,validation:000002,0,1,0,0.5,1,0.5,A", "1,validation:000003,1,1,-3,0.125,0,0.5,A",
                                     ""])
        log = (rd / "log.txt").read_text(encoding="ascii")
        self.assertIn("data: train 16 rows, validation 4 rows (4 in_clean)", log)
        self.assertNotIn("augmentation", log + (rd / "run.json").read_text(encoding="ascii"))

    def test_condition_c_stops_when_the_set_changed_after_the_preflight(self):
        with open(self.root / AUG_DIR / "review.csv", "ab") as f:
            f.write(b"aug-05-001,000000000000,keep,,tester\n")
        code, rec, seen, rd = self.go("C")
        self.assertEqual((code, rec["status"], rec["failure"]["phase"], rec["failure"]["exception"]),
                         (1, "failed", "data", "data_lists.ListError"))
        self.assertEqual((seen["order"], seen["encoded"]), (["load_augmentation"], []))  # nothing trained
        self.assertIn("FAIL J09 data/cases/train: changed after the preflight: review.csv\n",
                      (rd / "log.txt").read_text(encoding="ascii"))
        self.assertNotIn("data", rec)
        self.assertFalse((rd / "checkpoint").exists() or (rd / train.STAGING).exists())
        self.assertIsNone(rec["checkpoint"])

    def test_condition_c_join_refusal_is_short_in_the_log(self):
        # 500 listed rows that are not cases, with the pin of exactly these bytes (as if a preflight had passed)
        many = AUG_LIST_HEAD + b"".join(f"zz-{i:06d},0,000000000000\n".encode("ascii") for i in range(500))
        (self.root / AUG_DIR / "train_aug.csv").write_bytes(many)
        self.pins["train_aug.csv"] = sha(many)
        code, rec, seen, rd = self.go("C")
        self.assertEqual((code, rec["status"], rec["failure"]["phase"], rec["failure"]["exception"]),
                         (1, "failed", "data", "data_lists.ListError"))
        self.assertEqual((seen["order"], seen["encoded"]), (["load_augmentation"], []))  # nothing trained
        log = (rd / "log.txt").read_text(encoding="ascii")
        self.assertEqual(log.count("FAIL J"), 11)  # ten rows and one count, not 500 lines
        for i in range(10):
            self.assertIn(f"FAIL J02 train_aug.csv: zz-{i:06d} is not a row of the split\n", log)
        self.assertIn("FAIL J02 train_aug.csv: 490 more error(s) of this kind\n", log)
        self.assertNotIn("data", rec)


if __name__ == "__main__":
    unittest.main()
