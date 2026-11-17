"""Tests for scripts/make_freeze.py on synthetic temporary git repositories (no models, no dataset text).

The synthetic tree (make_tree) holds records in the formats of train.py, build_lists.py, build_train_aug.py,
select_thresholds.py and fetch_external.py. Repo.make() writes it to a new temp folder and commits it with
git (git fast-import, then the index is filled from HEAD; user, autocrlf and signing are set in the
repository's own config and the global and system git configs are switched off for the whole module).
Everything about git, --check, the output bytes and one refusal of every code runs on such a repository.
The many variants of a content rule (one changed record each) use Repo.stub(): the same tree in a shared
plain temp folder, with make_freeze.git_state replaced by a state that says "every file tracked and clean",
because a repository per variant costs about half a second on Windows. Checkpoints are 35 to 38 byte files, so the
pointer-size limit of the script is lowered to LIMIT for the module.

Expected values do not come from the script: expected() rebuilds the whole record from the tree with hashlib,
the hash rule is written out there (L = CRLF -> LF, B = bytes), and the SHA_* constants below were computed
outside Python with `printf '<bytes>' | sha256sum`. The dataset manifest of the synthetic tree records
SHA-256 values that are not the hashes of its csv files: the record must copy them, not recompute them.
The synthetic dataset file named test.csv is created by renaming a file of another name and is never opened,
by the tests or by the script; data_lists.install_test_guard (a process-wide audit hook) is mocked for the
whole module, and one test runs the script in a subprocess with the real guard.

Run from Git Bash at the repo root:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m unittest tests/test_make_freeze.py -v
"""
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import data_lists  # noqa: E402
import make_freeze as M  # noqa: E402

DATE = "2026-11-17"
REAL_LIMIT = M.POINTER_LIMIT  # read before setUpModule lowers it
LIMIT = 16  # pointer-size limit of the module while the tests run (the synthetic checkpoints have 35 bytes)
RECORDED = 104  # files of the synthetic tree that the record names: 120 files minus the 16 marked "not recorded"
SEEDS = [42, 123, 2026]
DS = "datasets/s-labs-prompt-injection"
DSM = DS + "/download_manifest.json"
LD = "data/processed"
META = LD + "/lists_meta.json"
AUG = "data/cases/train"
AUG_META = AUG + "/cases_meta.json"
FIN = "data/cases/final"
SPEC = FIN + "/final_spec.json"
DEV = "data/cases/dev"
EXT = "data/external"
RULES = EXT + "/external_rules.json"
THR = "results/s4/thresholds.json"
A42, C42 = "s2-roberta-A-s42", "s4-roberta-C-s42"
# The twelve runs of the freeze, written out by hand: run ID, stage, model, condition, seed.
RUNS = (
    ("s2-bert-A-s42", "s2", "bert", "A", 42), ("s2-bert-A-s123", "s2", "bert", "A", 123),
    ("s2-bert-A-s2026", "s2", "bert", "A", 2026),
    ("s2-deberta-A-s42", "s2", "deberta", "A", 42), ("s2-deberta-A-s123", "s2", "deberta", "A", 123),
    ("s2-deberta-A-s2026", "s2", "deberta", "A", 2026),
    ("s2-roberta-A-s42", "s2", "roberta", "A", 42), ("s2-roberta-A-s123", "s2", "roberta", "A", 123),
    ("s2-roberta-A-s2026", "s2", "roberta", "A", 2026),
    ("s4-roberta-C-s42", "s4", "roberta", "C", 42), ("s4-roberta-C-s123", "s4", "roberta", "C", 123),
    ("s4-roberta-C-s2026", "s4", "roberta", "C", 2026),
)
RUN_FILES = ("run.json", "metrics.json", "predictions_validation.csv")
LIST6 = ("train_clean.csv", "train_excluded.csv", "validation_clean.csv", "validation_excluded.csv",
         "test_clean_preaug.csv", "test_excluded_preaug.csv")
LIST9 = LIST6 + ("similar_pairs.csv", "test_clean.csv", "test_excluded.csv")
# Data rows of the synthetic lists, counted by hand from the bytes in leaf_files().
LIST_ROWS = {"train_clean.csv": 2, "train_excluded.csv": 0, "validation_clean.csv": 1, "validation_excluded.csv": 0,
             "test_clean_preaug.csv": 3, "test_excluded_preaug.csv": 2, "similar_pairs.csv": 1, "test_clean.csv": 2,
             "test_excluded.csv": 3, "train_review.csv": 0}
AUG5 = ("cases.csv", "review.csv", "train_aug.csv", "cases_meta.json", "test_overlap.json")
# Thresholds of the synthetic thresholds.json: candidate index k (logit k / 20) per seed and condition.
K = {(42, "B"): -27, (42, "D"): -13, (123, "B"): 0, (123, "D"): 5, (2026, "B"): -40, (2026, "D"): 12}
CAP_NOT_MET = {(123, "D")}
DS_REV = "d" * 40
DS_SHA = {"train": "1a" * 32, "validation": "2b" * 32, "test": "3c" * 32}  # manifest values, not the file hashes
README_TEXT = ("README 3.2 (test use procedure, step 1: design freeze) and 7.1 (freeze records); written by "
               "scripts/make_freeze.py")

# printf '<bytes>' | sha256sum (Git Bash), not computed by the script or by Python:
SHA_README_LF = "c1dff118e0ead7629dd3e40dfc4c3e35176c815b98107dddce0ca49371e0c78d"    # '# spec\nline two\n'
SHA_README_CRLF = "13205ebb16bf21b01abe48e4cd6443ad3ca732604c07db08c33ac4756f91e382"  # '# spec\r\nline two\r\n'
SHA_REQUIREMENTS = "2c4ca257b4c4fca3687bf53000c01bd3bb8c86ca566e3035061ace27150d0a38"  # 'torch==0.0\n'
# 'CKPT\r\ns2-bert-A-s42' + 8 x '\000\377' (35 bytes); with the CRLF turned into LF it would be 035a2f42...
SHA_CKPT_BERT42 = "4a3229b94ad579c692f3b21fd4b698ea464ade45e46cfdfef4a487194ea105ab"
SHA_CKPT_BERT42_LF = "035a2f425757b62d4de459ef6b91cc27b40a9fd4625090185e2a6abc4dcea5d2"
SHA_ALPHA_CARD = "e7194d6f1f0646fb36f1c5eec08cd415fd1a7f3fedc84b01d2b04fe69a6ee292"     # 'card\r\nalpha\r\n'
SHA_ALPHA_CARD_LF = "11796008c8795ba7fb848f38fdf8a9aa76f35680ab5be91eca4f3f6085c06969"  # 'card\nalpha\n'
SHA_ALPHA_PARQUET = "c9d0ff2c0866190c549dec97bb59374a3f043e06ad2e961de6edeb2d210ea1ec"  # 'PAR1\r\n\000alpha'
# 'synthetic prompt\nwith CRLF\n' and '# synthetic build_train_aug\n' (the LF forms of two CRLF files):
SHA_AUG_PROMPT_LF = "a38278cac243ffc1015ffc73c0ba7a6b8d58555f667caa44935e3a0f322e51a3"
SHA_BUILD_AUG_LF = "23d007ba103e673379a4e7127dd1c5bae3b938c7eb00265ec040cdb28b5460f4"
SHA_FINAL_PROMPT4_LF = "044d98dac3fb06cecd76df09ab4e9a2c2c4ac7fcf556c23f59c794209a160477"  # 'prompt four\n'

FAKE_HEAD = "f" * 40  # code_commit of the stubbed git state
GUARD_CALLS = []
_PATCHERS = []
_SHARED = []  # the shared plain folder of Repo.stub(), created on first use


def setUpModule():
    _PATCHERS.extend((
        mock.patch.object(M.data_lists, "install_test_guard", side_effect=lambda: GUARD_CALLS.append("guard")),
        mock.patch.object(M, "POINTER_LIMIT", LIMIT),
        mock.patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}),
    ))
    for p in _PATCHERS:
        p.start()


def tearDownModule():
    while _PATCHERS:
        _PATCHERS.pop().stop()
    while _SHARED:
        rmtree(_SHARED.pop())


# ---------------------------------------------------------------- hashes (independent of the script)

def sha(data):
    return hashlib.sha256(data).hexdigest()


def lf(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def J(obj):
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("ascii")


def f32(x):
    return float(np.float32(x))


def threshold_of(k):
    return f32(1.0 / (1.0 + math.exp(-(k / 20))))


def checkpoint_bytes(run_id):
    return b"CKPT\r\n" + run_id.encode("ascii") + b"\x00\xff" * 8


# ---------------------------------------------------------------- the synthetic tree

def leaf_files():
    """{repo-relative path: bytes} of the files that pin nothing (the records are added by make_tree)."""
    t = {
        "README.md": b"# spec\r\nline two\r\n",
        "requirements.txt": b"torch==0.0\n",
        "configs/train.json": J({"seeds": SEEDS, "threshold": 0.5, "epochs": 3}),
        "notes/unrelated.txt": b"not recorded\n",
        DS + "/README.md": b"not recorded\n",
        DS + "/data/train.csv": b"text,label\r\nsyn train row,0\r\n",
        DS + "/data/validation.csv": b"text,label\r\nsyn validation row,1\r\n",
        DS + "/data/test.csv": b"text,label\r\nsyn third split row,1\r\n",
        LD + "/train_clean.csv": b"sentence_id,label,text_sha256_12\ntrain:000000,0,aaaaaaaaaaaa\n"
                                 b"train:000001,1,bbbbbbbbbbbb\n",
        LD + "/train_excluded.csv": b"sentence_id,label,text_sha256_12,reason,ref\n",
        LD + "/validation_clean.csv": b"sentence_id,label,text_sha256_12\nvalidation:000000,1,cccccccccccc\n",
        LD + "/validation_excluded.csv": b"sentence_id,label,text_sha256_12,reason,ref\n",
        LD + "/test_clean_preaug.csv": b"sentence_id\ntest:000000\ntest:000002\ntest:000003\n",
        LD + "/test_excluded_preaug.csv": b"sentence_id,reason\ntest:000001,validation_overlap\n"
                                          b"test:000004,validation_overlap\n",
        LD + "/similar_pairs.csv": b"level,sentence_id,other_id\npunct_key,train:000000,train:000001\n",
        LD + "/test_clean.csv": b"sentence_id\ntest:000000\ntest:000002\n",
        LD + "/test_excluded.csv": b"sentence_id,reason\ntest:000001,validation_overlap\n"
                                   b"test:000003,augmentation_overlap\ntest:000004,validation_overlap\n",
        LD + "/train_review.csv": b"sentence_id,text_sha256_12,group_id,check,label,decision,reason,reviewer,"
                                  b"status,confirmed_by\n",
        AUG + "/originals.csv": b"original_id,label\ntrain:000000,0\n",
        AUG + "/cases.csv": b"case_id,purpose,subtype,label,status,text\n"
                            b"aug-01-case-000000,train,case,1,assigned,SyN cAsE oNe\n"
                            b"aug-04-001,train,benign,0,assigned,syn benign two\n"
                            b"aug-05-001,train,attack,1,assigned,syn attack three\n",
        AUG + "/review.csv": b"case_id,text_sha256_12,decision,new_label,reason,reviewer\n",
        AUG + "/train_aug.csv": b"sentence_id,label,text_sha256_12\naug-01-case-000000,1,111111111111\n"
                                b"aug-04-001,0,222222222222\naug-05-001,1,333333333333\n",
        AUG + "/test_overlap.json": J({"schema": 1, "counts": {"exact": 0}}),
        AUG + "/prompts/aug03_contexts_v1.md": b"synthetic prompt\r\nwith CRLF\r\n",
        AUG + "/prompts/aug_review_v1.md": b"synthetic review prompt\n",
        AUG + "/llm/aug03_contexts_v1.json": J({"generator": {"model": "syn"}, "items": []}),
        FIN + "/RULES.md": b"# synthetic rules\n",
        FIN + "/prompts/rob02_final_v1.md": b"prompt two\n",
        FIN + "/prompts/rob04_final_v1.md": b"prompt four\r\n",
        DEV + "/originals.csv": b"original_id\nvalidation:000000\n",
        DEV + "/cases.csv": b"case_id,text\ndev-rob04-001,syn dev case\n",
        DEV + "/review.csv": b"case_id,decision\ndev-rob04-001,keep\n",
        DEV + "/cases_meta.json": J({"schema": 1, "purpose": "dev"}),
        DEV + "/REVIEW_GUIDE.md": b"not recorded\n",
        EXT + "/alpha/README.md": b"card\r\nalpha\r\n",
        EXT + "/alpha/data/part.parquet": b"PAR1\r\n\x00alpha",
        EXT + "/beta/README.md": b"card beta\n",
        EXT + "/beta/data/one.parquet": b"PAR1 beta one",
        EXT + "/beta/data/two.parquet": b"PAR1 beta two",
        "results/s2/summary.json": J({"schema": 1, "stage": 2}),
        "results/s2/runs.csv": b"not recorded\n",
        "results/s3/summary.json": J({"schema": 1, "stage": 3}),
        "results/s3/error_summary.json": J({"schema": 1, "errors": 0}),
        "results/s4/summary.json": J({"schema": 1, "stage": 4}),
        "results/s4/dev_cases_vs_test.json": J({"schema": 1, "matches": []}),
        "scripts/build_lists.py": b"# synthetic build_lists\n",
        "scripts/build_train_aug.py": b"# synthetic build_train_aug\r\n",
        "scripts/build_dev_cases.py": b"ROB01 = {}\n\n\ndef assemble(position, ctx_texts, orig):\n    return orig\n"
                                      b"\n\ndef plan_rob03(origs, pool, token_len):\n    return []\n",
        "scripts/external_lang.py": b"# synthetic external_lang\n",
        "scripts/fetch_external.py": b"# synthetic fetch_external\n",
        "scripts/review/notes.md": b"synthetic notes\n",
        "tests/test_synthetic.py": b"# synthetic test\n",
    }
    for run_id, _stage, _model, _cond, _seed in RUNS:
        t[f"runs/{run_id}/metrics.json"] = J({"run_id": run_id, "selected_step": 20})
        t[f"runs/{run_id}/predictions_validation.csv"] = f"step,sentence_id\n20,validation:000000,{run_id}\n".encode()
        t[f"runs/{run_id}/checkpoint/model.safetensors"] = checkpoint_bytes(run_id)
        t[f"runs/{run_id}/log.txt"] = b"not recorded\n"
    return t


def run_json(t, run_id, stage, model, cond, seed):
    ck = t[f"runs/{run_id}/checkpoint/model.safetensors"]
    doc = {"schema": 1, "run_id": run_id, "stage": stage, "model_key": model, "condition": cond, "seed": seed,
           "status": "completed", "selection": {"selected_step": 20, "saved_steps": [10, 20]},
           "checkpoint": {"kept": True, "path": f"runs/{run_id}/checkpoint/model.safetensors", "sha256": sha(ck),
                          "size": len(ck), "step": 20},
           "config": {"path": "configs/train.json", "sha256": sha(t["configs/train.json"])},
           "env": {"requirements_sha256": sha(t["requirements.txt"])},
           "lists": {"files_sha256": {n: sha(t[f"{LD}/{n}"]) for n in LIST6}},
           "data": {"train_rows": 2}}
    if cond == "C":
        doc["data"]["augmentation"] = {"dir": AUG, "files_sha256": {n: sha(t[f"{AUG}/{n}"]) for n in AUG5},
                                       "label_counts": {"0": 1, "1": 2}}
    return doc


def thresholds_json(t):
    entries, inputs = [], {"configs/train.json": sha(t["configs/train.json"])}
    for seed in SEEDS:
        for cond, run_id in (("B", f"s2-roberta-A-s{seed}"), ("D", f"s4-roberta-C-s{seed}")):
            k = K[(seed, cond)]
            entries.append({"seed": seed, "condition": cond, "run_id": run_id, "selected_step": 20,
                            "checkpoint_sha256": sha(t[f"runs/{run_id}/checkpoint/model.safetensors"]),
                            "cap_met": (seed, cond) not in CAP_NOT_MET, "k": k, "logit": k / 20,
                            "threshold": threshold_of(k), "chosen": {"tp": 1, "fp": 0, "fn": 0, "tn": 1}})
            for n in RUN_FILES:
                inputs[f"runs/{run_id}/{n}"] = sha(t[f"runs/{run_id}/{n}"])
    return {"schema": 1, "model": "roberta", "seeds": SEEDS, "inputs_sha256": inputs, "thresholds": entries}


def manifest_json(t, name, dataset, revision):
    d = f"{EXT}/{name}/"
    return {"dataset": dataset, "revision": revision,
            "files": [{"path": p[len(d):], "size_bytes": len(t[p]), "sha256": sha(t[p])}
                      for p in sorted(t) if p.startswith(d) and not p.endswith("download_manifest.json")]}


def make_tree(tweak=None, leaves=None):
    """A consistent synthetic tree. leaves(t) changes the leaf files before any record pins them;
    tweak = {record path: fn(doc)} changes a record when it is written, before the records that pin it."""
    tweak = dict(tweak or {})
    t = leaf_files()
    if leaves:
        leaves(t)

    def record(path, doc):
        fn = tweak.pop(path, None)
        if fn:
            fn(doc)
        t[path] = J(doc)

    record(DSM, {"dataset": "syn/dataset", "revision": DS_REV,
                 "files": [{"path": "README.md", "size_bytes": len(t[DS + "/README.md"]), "sha256": "00" * 32}]
                 + [{"path": f"data/{s}.csv", "size_bytes": len(t[f"{DS}/data/{s}.csv"]), "sha256": DS_SHA[s]}
                    for s in ("train", "validation", "test") if f"{DS}/data/{s}.csv" in t],
                 "splits": {s: {"file": f"data/{s}.csv", "rows": 1} for s in ("train", "validation", "test")}})
    record(AUG_META, {
        "schema": 1, "ready_for_training": True,
        "outputs": {n: {"sha256": sha(t[f"{AUG}/{n}"])}
                    for n in ("originals.csv", "cases.csv", "review.csv", "train_aug.csv") if f"{AUG}/{n}" in t},
        "script": {"path": "scripts/build_train_aug.py", "sha256_lf": lf(t["scripts/build_train_aug.py"])},
        "generators": {"aug03_contexts_v1": {
            "file": "llm/aug03_contexts_v1.json", "sha256_lf": lf(t[AUG + "/llm/aug03_contexts_v1.json"]),
            "prompt": "prompts/aug03_contexts_v1.md",
            "prompt_sha256_lf": lf(t.get(AUG + "/prompts/aug03_contexts_v1.md", b""))}}})
    record(META, {
        "schema": 1, "trial": False, "test_clean_stage": "final",
        "outputs": {n: {"rows": t[f"{LD}/{n}"].count(b"\n") - 1, "sha256": sha(t[f"{LD}/{n}"])} for n in LIST9},
        "review": {"path": LD + "/train_review.csv", "sha256": sha(t[LD + "/train_review.csv"])},
        "script": {"path": "scripts/build_lists.py", "sha256_lf": lf(t["scripts/build_lists.py"])},
        "dataset": {"revision": DS_REV},
        "augmentation": {"dir": AUG, "files": {"train_aug.csv": sha(t[AUG + "/train_aug.csv"]),
                                               "cases.csv": sha(t[AUG + "/cases.csv"])}, "rows": 3}})
    for run_id, stage, model, cond, seed in RUNS:
        record(f"runs/{run_id}/run.json", run_json(t, run_id, stage, model, cond, seed))
    record(THR, thresholds_json(t))
    record(SPEC, {"schema": 1, "extraction": {"seed": 20261117, "label_order": [1, 0]},
                  "targets": {"ROB-01": {"1": 50, "0": 50}, "ROB-04": {"0": 50}},
                  "generators": {"llm_model": "syn-model-1"},
                  "prompts_sha256_lf": {p: lf(t[p]) for p in sorted(t) if p.startswith(FIN + "/prompts/")}})
    sources = {}
    for name, dataset, revision, splits in (("alpha", "org/alpha", "1" * 40, ["train"]),
                                            ("beta", "org/beta", "2" * 40, ["one", "two"])):
        man = manifest_json(t, name, dataset, revision)
        path = f"{EXT}/{name}/download_manifest.json"
        record(path, man)
        man = json.loads(t[path])
        sources[name] = {"dataset": dataset, "revision": revision, "dir": f"{EXT}/{name}", "manifest": path,
                         "manifest_sha256": sha(t[path]), "adopted_splits": splits,
                         "files": [{k: f[k] for k in ("path", "size_bytes", "sha256") if k in f}
                                   for f in man["files"]]}
    record(RULES, {"schema": 1, "rules_id": "syn_rules_v1", "sources": sources})
    assert not tweak, f"tweak for a record that make_tree does not write: {sorted(tweak)}"
    return t


def expected(t, commit, date=DATE):
    """The record that make_freeze.py must write for tree t, built with hashlib only.
    L: SHA-256 after CRLF -> LF (text rule); B: SHA-256 of the bytes (checkpoints, parquet, external raw files)."""
    def L(path):
        return lf(t[path])

    def B(path):
        return sha(t[path])

    runs = {}
    for run_id, stage, model, cond, seed in RUNS:
        ck = f"runs/{run_id}/checkpoint/model.safetensors"
        runs[run_id] = {"run_id": run_id, "model_key": model, "stage": stage, "condition": cond, "seed": seed,
                        "selected_step": 20, "checkpoint": {"path": ck, "size": len(t[ck]), "sha256": B(ck)},
                        "files": {n: L(f"runs/{run_id}/{n}") for n in RUN_FILES}}
    by_seed = {str(s): {c: {"run_id": f"{'s2' if c == 'B' else 's4'}-roberta-{'A' if c == 'B' else 'C'}-s{s}",
                            "logit": K[(s, c)] / 20, "threshold": threshold_of(K[(s, c)]),
                            "cap_met": (s, c) not in CAP_NOT_MET} for c in ("B", "D")} for s in SEEDS}
    c_runs = {}
    for run_id, _stage, _model, cond, _seed in RUNS:
        if cond == "C":
            recorded = json.loads(t[f"runs/{run_id}/run.json"])["data"]["augmentation"]["files_sha256"]
            c_runs[run_id] = {"files_sha256": {n: recorded[n] for n in AUG5},
                              "differs_from_frozen": [n for n in ("cases_meta.json", "test_overlap.json")
                                                      if recorded[n] != L(f"{AUG}/{n}")]}
    external = {}
    for name, dataset, revision, splits in (("alpha", "org/alpha", "1" * 40, ["train"]),
                                            ("beta", "org/beta", "2" * 40, ["one", "two"])):
        d = f"{EXT}/{name}"
        raw = [p for p in sorted(t) if p.startswith(d + "/") and not p.endswith("download_manifest.json")]
        man = d + "/download_manifest.json"
        external[name] = {"dataset": dataset, "revision": revision, "dir": d, "adopted_splits": splits,
                          "manifest": {"path": man, "sha256": L(man)},
                          "files": {p: {"sha256": B(p), "size_bytes": len(t[p])} for p in raw}}
    return {
        "schema": 1,
        "hash_rule": M.HASH_RULE,  # explanatory text only
        "freeze": {"stage": 1, "tag": "freeze-1", "date": date, "readme": README_TEXT},
        "code_commit": commit,
        "spec": {"path": "README.md", "sha256": L("README.md")},
        "dataset": {"manifest": {"path": DSM, "sha256": L(DSM)}, "name": "syn/dataset", "revision": DS_REV,
                    "files": {s: {"path": f"{DS}/data/{s}.csv", "sha256": DS_SHA[s],
                                  "size_bytes": len(t[f"{DS}/data/{s}.csv"])}
                              for s in ("train", "validation", "test")}},
        "lists": {"dir": LD, "test_clean_stage": "final", "trial": False,
                  "files": {**{n: {"sha256": L(f"{LD}/{n}"), "rows": LIST_ROWS[n]} for n in LIST_ROWS},
                            "lists_meta.json": {"sha256": L(META)}},
                  "test_clean": {"rows": 2, "excluded_rows": 3, "excluded_by_reason": {
                      "train_overlap": 0, "augmentation_overlap": 1, "validation_overlap": 2}}},
        "config": {"path": "configs/train.json", "sha256": L("configs/train.json")},
        "environment": {"path": "requirements.txt", "sha256": L("requirements.txt")},
        "models": {"keys": ["bert", "deberta", "roberta"], "representative": "roberta", "seeds": [42, 123, 2026]},
        "runs": runs,
        "conditions": {
            "A": {"checkpoints": "s2-roberta-A-s<seed>", "threshold": 0.5},
            "B": {"checkpoints": "s2-roberta-A-s<seed>",
                  "thresholds": {str(s): threshold_of(K[(s, "B")]) for s in SEEDS}},
            "C": {"checkpoints": "s4-roberta-C-s<seed>", "threshold": 0.5},
            "D": {"checkpoints": "s4-roberta-C-s<seed>",
                  "thresholds": {str(s): threshold_of(K[(s, "D")]) for s in SEEDS}}},
        "thresholds": {"path": THR, "sha256": L(THR), "by_seed": by_seed},
        "augmentation": {
            "dir": AUG, "ready_for_training": True, "train_aug": {"rows": 3, "label_counts": {"0": 1, "1": 2}},
            "files": {p: L(p) for p in sorted(t) if p.startswith(AUG + "/")}, "c_runs": c_runs},
        "final_cases": {
            "dir": FIN, "files": {p: L(p) for p in sorted(t) if p.startswith(FIN + "/")},
            "generator_code": {"path": "scripts/build_dev_cases.py", "sha256": L("scripts/build_dev_cases.py"),
                               "provides": ["ROB01", "assemble", "plan_rob03"]},
            "extraction_seed": 20261117, "targets": {"ROB-01": {"1": 50, "0": 50}, "ROB-04": {"0": 50}},
            "llm_model": "syn-model-1"},
        "dev_cases": {"dir": DEV, "files": {f"{DEV}/{n}": L(f"{DEV}/{n}") for n in (
            "originals.csv", "cases.csv", "review.csv", "cases_meta.json")}},
        "external": {"rules": {"path": RULES, "sha256": L(RULES)}, "sources": external,
                     "scripts": {p: L(p) for p in ("scripts/external_lang.py", "scripts/fetch_external.py")}},
        "results": {p: L(p) for p in (
            "results/s2/summary.json", "results/s3/summary.json", "results/s3/error_summary.json",
            "results/s4/thresholds.json", "results/s4/summary.json", "results/s4/dev_cases_vs_test.json")},
        "scripts": {p: L(p) for p in sorted(t) if p.startswith(("scripts/", "tests/"))},
    }


def render(doc):
    return (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


# ---------------------------------------------------------------- repositories

def put(root, rel, data):
    """Write one file. A file named test.csv is written under another name and renamed: it is never opened."""
    p = Path(root).joinpath(*rel.split("/"))
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.name.lower() == "test.csv":
        part = p.with_name("third_split.part")
        part.write_bytes(data)
        os.replace(part, p)
    else:
        p.write_bytes(data)


def git(root, *args):
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if r.returncode != 0:
        raise AssertionError(f"git {' '.join(args)}: {r.stderr.decode('utf-8', 'replace')}")
    return r.stdout.decode("utf-8").strip()


def git_init(root):
    git(root, "init", "-q", "-b", "main")
    with open(Path(root) / ".git" / "config", "a", encoding="ascii", newline="\n") as f:
        f.write("[user]\n\tname = Synthetic Tester\n\temail = tester@example.invalid\n"
                "[core]\n\tautocrlf = false\n\tsafecrlf = false\n[commit]\n\tgpgsign = false\n")


def commit_tree(root, t):
    """First commit of a repository whose work tree already holds the files of t: one git fast-import
    (the blobs are the bytes of t, nothing is converted), then the index is filled from HEAD."""
    stream = [b"commit refs/heads/main\ncommitter Synthetic Tester <tester@example.invalid> 1700000000 +0000\n"
              b"data 9\nsynthetic\n"]
    for rel in sorted(t):
        stream.append(b"M 100644 inline %s\ndata %d\n%s\n" % (rel.encode("ascii"), len(t[rel]), t[rel]))
    r = subprocess.run(["git", "-C", str(root), "fast-import", "--quiet"], input=b"".join(stream),
                       capture_output=True)
    if r.returncode != 0:
        raise AssertionError(f"git fast-import: {r.stderr.decode('utf-8', 'replace')}")
    git(root, "reset", "-q")
    return git(root, "rev-parse", "HEAD")


def sync(root, t):
    """Make a plain folder hold exactly the files of t: what else is there (the output of an earlier
    test, a file where a folder must be, or the other way round) is removed first."""
    root = Path(root)
    want = {root.joinpath(*rel.split("/")): rel for rel in t}
    folders = {parent for path in want for parent in path.parents}
    for path in sorted(root.rglob("*"), reverse=True):  # children before their folder
        if path.is_dir():
            if path not in folders:
                path.rmdir()
        elif path not in want:
            path.unlink()
    for path, rel in want.items():
        if path.name.lower() == "test.csv" or not path.is_file() or path.read_bytes() != t[rel]:
            put(root, rel, t[rel])  # a file named test.csv is not read back: it is always written again


def commit_all(root, message="synthetic"):
    git(root, "add", "-A")
    git(root, "commit", "-q", "--allow-empty", "-m", message)
    return git(root, "rev-parse", "HEAD")


def run_script(root, *args):
    """(exit code, stdout lines) of make_freeze.main in a subprocess: the real test.csv guard (not mocked),
    the pointer limit lowered like in this module; the last line tells that the guard was installed."""
    prog = ("import sys; sys.path.insert(0, sys.argv[1]); import data_lists, make_freeze as M; "
            f"M.POINTER_LIMIT = {LIMIT}; code = M.main(sys.argv[2:]); "
            "print('guard installed:', data_lists._HOOK['installed']); sys.exit(code)")
    cmd = [sys.executable, "-c", prog, str(ROOT / "scripts"), "--stage", "1", "--root", str(root), *args]
    r = subprocess.run(cmd, capture_output=True, env={**os.environ, "PYTHONUTF8": "1"})
    return r.returncode, r.stdout.decode("ascii").splitlines()


def rmtree(path):
    def writable(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    shutil.rmtree(path, onexc=writable)


class Repo(unittest.TestCase):
    maxDiff = None

    def make(self, tweak=None, leaves=None, after=None, init=True, commit=True):
        """A new git repository in a temp folder that holds a synthetic tree, committed. after(t) changes
        the finished tree (records included) before it is written: an inconsistency that is committed."""
        self.assertNotIsInstance(M.git_state, mock.Mock, "a real repository after stub() in one test")
        self.t = make_tree(tweak, leaves)
        if after:
            after(self.t)
        self.tmp = Path(tempfile.mkdtemp(prefix="mkfreeze_")).resolve()
        self.addCleanup(rmtree, self.tmp)
        self.root = self.tmp / "repo"
        for rel, data in self.t.items():
            put(self.root, rel, data)
        self.head = None
        if init:
            git_init(self.root)
            if commit:
                self.head = commit_tree(self.root, self.t)
        self.out = self.root / "freeze" / "freeze-1.json"
        return self.root

    def stub(self, tweak=None, leaves=None, after=None):
        """The same tree in the shared plain folder, with the git layer of the script replaced for the
        rest of the test by a state in which every file of the tree is tracked and clean (HEAD FAKE_HEAD).
        For the variants of a content rule; the work tree must not be changed afterwards."""
        self.t = make_tree(tweak, leaves)
        if after:
            after(self.t)
        if not _SHARED:
            _SHARED.append(Path(tempfile.mkdtemp(prefix="mkfreeze_shared_")).resolve())
        self.tmp, self.root, self.head = _SHARED[0], _SHARED[0] / "tree", FAKE_HEAD
        sync(self.root, self.t)
        state = {"head": FAKE_HEAD, "tracked": set(self.t), "dirty": {}, "untracked_code": []}
        patcher = mock.patch.object(M, "git_state", return_value=([], state))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.out = self.root / "freeze" / "freeze-1.json"
        return self.root

    def main(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = M.main([*args, "--root", str(self.root)])
        return code, buf.getvalue().splitlines()

    def build(self, *extra):
        return self.main("--stage", "1", "--date", DATE, *extra)

    def check(self, *extra):
        return self.main("--stage", "1", "--check", *extra)

    def written(self):
        code, out = self.build()
        self.assertEqual(code, 0, out)
        return self.out.read_bytes()

    def fails(self, run):
        """FAIL lines of a refused run: exit 1, one summary line, nothing else printed."""
        code, out = run
        fails = [ln for ln in out if ln.startswith("FAIL ")]
        self.assertEqual(code, 1, out)
        self.assertEqual(out[-1], f"refused: {len(fails)} error(s); nothing written", out)
        self.assertEqual(len(out), len(fails) + 1, out)
        self.assertTrue(all(ln.isascii() for ln in out))
        return fails

    def refused(self, codes, *needles, real=False, **tree):
        """Run the freeze and require exactly these refusal codes (in order). With tree arguments the
        tree is built first: stubbed git by default, a real repository with real=True."""
        if tree or not hasattr(self, "root"):
            (self.make if real else self.stub)(**tree)
        fails = self.fails(self.build())
        self.assertEqual([ln.split()[1] for ln in fails], codes, fails)
        for needle in needles:
            self.assertIn(needle, "\n".join(fails))
        self.assertFalse(self.out.parent.exists(), "a refused run must not write")
        return fails


def setter(*keys, value):
    """fn(doc) that sets doc[k1][k2]... = value."""
    def fn(doc):
        for k in keys[:-1]:
            doc = doc[k]
        doc[keys[-1]] = value
    return fn


def remover(*keys):
    def fn(doc):
        for k in keys[:-1]:
            doc = doc[k]
        del doc[keys[-1]]
    return fn


# ---------------------------------------------------------------- the record

class TestRecord(Repo):
    def test_record_is_the_hand_built_one(self):
        self.make()
        code, out = self.build()
        self.assertEqual(code, 0, out)
        data = self.out.read_bytes()
        want = expected(self.t, self.head)
        self.assertEqual(json.loads(data), want)
        self.assertEqual(data, render(want))
        self.assertEqual(len(self.t), 120)
        self.assertEqual(sum(v == b"not recorded\n" for v in self.t.values()), 120 - RECORDED)
        self.assertEqual(out[0], f"OK wrote freeze/freeze-1.json: stage 1, date {DATE}, code_commit "
                                 f"{self.head[:12]}, {RECORDED} files recorded, sha256 {sha(data)[:12]}")
        self.assertIn("tag the commit freeze-1", out[1])
        self.assertEqual(len(out), 2)
        self.assertFalse(self.out.with_name("freeze-1.json.tmp").exists())

    def test_bytes_are_ascii_lf_without_paths_of_this_machine(self):
        self.make()
        data = self.written()
        self.assertTrue(data.isascii())
        self.assertNotIn(b"\r", data)
        self.assertTrue(data.endswith(b"}\n"))
        for needle in (str(self.root), self.root.as_posix(), str(self.tmp), "mkfreeze_"):
            self.assertNotIn(needle.encode(), data)
        doc = json.loads(data)
        self.assertEqual(list(doc), sorted(doc))  # sorted keys
        self.assertEqual(sorted(doc), sorted([
            "schema", "hash_rule", "freeze", "code_commit", "spec", "dataset", "lists", "config", "environment",
            "models", "runs", "conditions", "thresholds", "augmentation", "final_cases", "dev_cases", "external",
            "results", "scripts"]))

    def test_hashes_computed_with_sha256sum(self):
        self.make()
        doc = json.loads(self.written())
        # text rule: README.md is CRLF in the work tree, the record holds the hash of its LF form
        self.assertEqual(sha(self.t["README.md"]), SHA_README_CRLF)
        self.assertEqual(doc["spec"]["sha256"], SHA_README_LF)
        self.assertEqual(doc["environment"]["sha256"], SHA_REQUIREMENTS)
        self.assertEqual(doc["augmentation"]["files"][AUG + "/prompts/aug03_contexts_v1.md"], SHA_AUG_PROMPT_LF)
        self.assertEqual(doc["final_cases"]["files"][FIN + "/prompts/rob04_final_v1.md"], SHA_FINAL_PROMPT4_LF)
        self.assertEqual(doc["scripts"]["scripts/build_train_aug.py"], SHA_BUILD_AUG_LF)
        # binary rule: the CRLF inside a checkpoint, a parquet file or an external raw file is kept
        ck = doc["runs"]["s2-bert-A-s42"]["checkpoint"]
        self.assertEqual((ck["sha256"], ck["size"]), (SHA_CKPT_BERT42, 35))
        self.assertNotEqual(ck["sha256"], SHA_CKPT_BERT42_LF)
        alpha = doc["external"]["sources"]["alpha"]["files"]
        self.assertEqual(alpha[EXT + "/alpha/data/part.parquet"], {"sha256": SHA_ALPHA_PARQUET, "size_bytes": 12})
        self.assertEqual(alpha[EXT + "/alpha/README.md"], {"sha256": SHA_ALPHA_CARD, "size_bytes": 13})
        self.assertNotEqual(alpha[EXT + "/alpha/README.md"]["sha256"], SHA_ALPHA_CARD_LF)

    def test_text_rule_is_data_lists_sha256_lf(self):
        self.make(init=False)
        fz = M.Freeze(self.root)
        self.assertEqual(fz.sha("README.md"), data_lists.sha256_lf(self.root / "README.md"))
        self.assertEqual(fz.sha("README.md"), SHA_README_LF)
        self.assertEqual(fz.text("README.md"), b"# spec\nline two\n")
        self.assertEqual(fz.sha("runs/s2-bert-A-s42/checkpoint/model.safetensors"), SHA_CKPT_BERT42)
        self.assertEqual(fz.sha(EXT + "/alpha/data/part.parquet"), SHA_ALPHA_PARQUET)
        self.assertEqual(fz.recorded, {"README.md": "lf", "runs/s2-bert-A-s42/checkpoint/model.safetensors": "bytes",
                                       EXT + "/alpha/data/part.parquet": "bytes"})
        self.assertEqual(M.sha256_stream(self.root / "README.md"), (SHA_README_CRLF, 18))
        self.assertEqual(fz.errors, [])

    def test_dataset_hashes_are_copied_from_the_manifest(self):
        self.stub()
        files = json.loads(self.written())["dataset"]["files"]
        self.assertEqual(files["test"], {"path": DS + "/data/test.csv", "sha256": "3c" * 32, "size_bytes": 35})
        self.assertEqual(files["train"]["sha256"], "1a" * 32)
        self.assertEqual(files["validation"]["sha256"], "2b" * 32)
        self.assertNotEqual(sha(self.t[DS + "/data/train.csv"]), "1a" * 32)  # not the hash of the file

    def test_test_clean_counts_and_list_rows(self):
        self.stub()
        lists = json.loads(self.written())["lists"]
        self.assertEqual(lists["test_clean"], {"rows": 2, "excluded_rows": 3, "excluded_by_reason": {
            "train_overlap": 0, "augmentation_overlap": 1, "validation_overlap": 2}})
        self.assertEqual({n: v.get("rows") for n, v in lists["files"].items()},
                         {**LIST_ROWS, "lists_meta.json": None})

    def test_thresholds_and_conditions(self):
        self.stub()
        doc = json.loads(self.written())
        self.assertEqual(doc["conditions"]["B"]["thresholds"]["123"], 0.5)  # k = 0
        self.assertEqual(doc["thresholds"]["by_seed"]["42"]["B"], {
            "run_id": "s2-roberta-A-s42", "logit": -1.35, "threshold": f32(1 / (1 + math.exp(1.35))),
            "cap_met": True})
        self.assertEqual(doc["thresholds"]["by_seed"]["123"]["D"], {
            "run_id": "s4-roberta-C-s123", "logit": 0.25, "threshold": f32(1 / (1 + math.exp(-0.25))),
            "cap_met": False})
        self.assertEqual(doc["conditions"]["D"]["thresholds"]["2026"], f32(1 / (1 + math.exp(-0.6))))

    def test_same_tree_gives_the_same_bytes(self):
        self.make()
        first = self.written()
        other = self.tmp / "second.json"
        code, out = self.build("--out", str(other))
        self.assertEqual(code, 0, out)
        self.assertEqual(other.read_bytes(), first)
        self.assertIn(str(other), out[0])  # a path outside the root is shown as given

    def test_unrelated_dirty_and_untracked_files_do_not_matter(self):
        self.make()
        put(self.root, "notes/unrelated.txt", b"changed after the commit\n")
        put(self.root, "notes/new.txt", b"untracked\n")
        put(self.root, DEV + "/prompts/extra.md", b"dev prompts are not recorded\n")
        put(self.root, "runs/s1-bert-smoke-s0/run.json", b"{}\n")
        self.assertEqual(json.loads(self.written()), expected(self.t, self.head))

    def test_threshold_entries_are_found_at_any_depth(self):
        def nest(doc):
            doc["by_condition"] = {"B": [e for e in doc["thresholds"] if e["condition"] == "B"],
                                   "D": {"entries": [e for e in doc["thresholds"] if e["condition"] == "D"]}}
            del doc["thresholds"]
        self.stub(tweak={THR: nest})
        self.assertEqual(json.loads(self.written()), expected(self.t, self.head))

    def test_c_run_with_another_cases_meta_is_recorded_with_both_values(self):
        def other(doc):
            doc["data"]["augmentation"]["files_sha256"].update({"cases_meta.json": "ab" * 32,
                                                                "test_overlap.json": "cd" * 32})
        self.stub(tweak={f"runs/{C42}/run.json": other})
        aug = json.loads(self.written())["augmentation"]
        self.assertEqual(aug["c_runs"][C42]["differs_from_frozen"], ["cases_meta.json", "test_overlap.json"])
        self.assertEqual(aug["c_runs"][C42]["files_sha256"]["cases_meta.json"], "ab" * 32)
        self.assertEqual(aug["files"][AUG_META], lf(self.t[AUG_META]))
        self.assertEqual(aug["c_runs"]["s4-roberta-C-s123"]["differs_from_frozen"], [])
        self.assertEqual(json.loads(self.out.read_bytes()), expected(self.t, self.head))

    def test_lists_meta_augmentation_files_may_be_objects(self):
        def objects(doc):
            files = doc["augmentation"]["files"]
            doc["augmentation"]["files"] = {n: {"sha256": v, "rows": 3} for n, v in files.items()}
            del doc["augmentation"]["dir"]
        self.stub(tweak={META: objects})
        self.assertEqual(json.loads(self.written()), expected(self.t, self.head))

    def test_generator_file_outside_the_patterns_is_recorded(self):
        def leaves(t):
            t[AUG + "/llm/notes.txt"] = b"generator output\n"

        def gen(doc):
            doc["generators"]["extra_v1"] = {"file": "llm/notes.txt", "sha256_lf": lf(b"generator output\n"),
                                             "prompt": "prompts/aug_review_v1.md",
                                             "prompt_sha256_lf": lf(b"synthetic review prompt\n")}
        self.stub(leaves=leaves, tweak={AUG_META: gen})
        self.assertEqual(json.loads(self.written())["augmentation"]["files"][AUG + "/llm/notes.txt"],
                         lf(b"generator output\n"))

    def test_guard_is_installed_before_anything_else(self):
        self.make(init=False)
        del GUARD_CALLS[:]
        code, out = self.main("--stage", "2", "--date", "not a date")
        self.assertEqual((code, GUARD_CALLS), (1, ["guard"]))
        self.build()
        self.assertEqual(GUARD_CALLS, ["guard", "guard"])

    def test_real_guard_in_a_subprocess_and_default_paths(self):
        """The script with the real test.csv guard: it must finish, so it never opened the synthetic
        test.csv (the guard ends the process with exit 1 on any open of such a path)."""
        self.make()
        code, out = run_script(self.root, "--date", DATE)
        self.assertEqual(code, 0, out)
        self.assertEqual(out[-1], "guard installed: True")
        self.assertEqual(self.out.read_bytes(), render(expected(self.t, self.head)))
        code, out = run_script(self.root, "--check")
        self.assertEqual(code, 0, out)
        self.assertEqual(out, [f"OK --check: freeze/freeze-1.json equals the fresh rebuild; {RECORDED} files "
                               f"re-hashed, code_commit {self.head[:12]}, HEAD {self.head[:12]}",
                               "guard installed: True"])

    def test_pointer_limit_is_one_mebibyte(self):
        self.assertEqual(REAL_LIMIT, 1 << 20)
        self.assertEqual(M.POINTER_LIMIT, LIMIT)  # lowered for the module


class TestLineEndings(Repo):
    def test_crlf_checkout_of_every_text_file_verifies(self):
        """After the freeze every text file (and the freeze file) is rewritten with CRLF and committed:
        --check must pass, because text is hashed in its LF form; binary and raw external files are not touched."""
        self.make()
        data = self.written()
        commit_all(self.root, "freeze file")
        changed = 0
        for rel, raw in self.t.items():
            binary = rel.endswith((".safetensors", ".parquet")) or rel.startswith((EXT + "/alpha/", EXT + "/beta/"))
            if not binary and not rel.startswith(DS + "/data/") and b"\r\n" not in raw:
                put(self.root, rel, raw.replace(b"\n", b"\r\n"))
                changed += 1
        self.out.write_bytes(data.replace(b"\n", b"\r\n"))
        self.assertEqual(changed, 94)  # 120 files - 12 checkpoints - 7 external - 3 dataset csv - 4 already CRLF
        commit_all(self.root, "crlf checkout")
        code, out = self.check()
        self.assertEqual(code, 0, out)

    def test_crlf_in_a_binary_file_is_not_normalised(self):
        """A checkpoint whose CRLF became LF has another hash: binary files are hashed as bytes."""
        def strip(t):
            p = f"runs/{A42}/checkpoint/model.safetensors"
            t[p] = t[p].replace(b"\r\n", b"\n") + b"\x00"  # same size, the LF form plus one byte
        self.refused(["F04"], f"runs/{A42}/checkpoint/model.safetensors: sha256 on disk", after=strip)


class TestEveryCodeOnARealRepository(Repo):
    """One refusal per code with real git (F00, F02 and F09 have their own classes below)."""

    def test_one_refusal_of_every_code(self):
        ck = f"runs/{A42}/checkpoint/model.safetensors"
        prompt = FIN + "/prompts/rob02_final_v1.md"

        def gamma(t):
            t[EXT + "/gamma/download_manifest.json"] = J({"dataset": "org/gamma", "revision": "3" * 40, "files": []})

        def drop(doc):
            doc["thresholds"] = doc["thresholds"][1:]
        cases = (
            ("F01", {"after": lambda t: t.pop("results/s4/summary.json")}, "results/s4/summary.json: missing"),
            ("F03", {"leaves": lambda t: t.update({ck: b"pointer\n"})}, f"{ck}: 8 bytes, smaller than 16"),
            ("F04", {"after": lambda t: t.update({ck: t[ck][:-1] + b"\x01"})}, f"{ck}: sha256 on disk"),
            ("F05", {"tweak": {META: setter("test_clean_stage", value="pre-augmentation")}}, "test_clean_stage"),
            ("F06", {"tweak": {f"runs/{C42}/run.json": setter("data", "augmentation", "files_sha256", "cases.csv",
                                                             value="34" * 32)}}, "files_sha256 > cases.csv"),
            ("F07", {"tweak": {THR: drop}}, "no entry"),
            ("F08", {"after": lambda t: t.update({prompt: b"edited\n"})}, f"prompts_sha256_lf > {prompt}"),
            ("F10", {"tweak": {META: remover("trial")}}, "key trial is missing"),
            ("F11", {"tweak": {f"runs/{A42}/run.json": setter("status", value="interrupted")}}, "status is"),
            ("F12", {"leaves": gamma}, "does not adopt: gamma"),
        )
        for code, tree, needle in cases:
            with self.subTest(code=code):
                self.refused([code], needle, real=True, **tree)
                self.assertRegex(self.head, "^[0-9a-f]{40}$")
                self.assertNotEqual(self.head, FAKE_HEAD)


# ---------------------------------------------------------------- arguments (F00)

class TestArguments(Repo):
    def test_stage_2_is_refused(self):
        self.make(init=False)
        fails = self.fails(self.main("--stage", "2", "--date", DATE))
        self.assertEqual([ln.split()[1] for ln in fails], ["F00"])
        self.assertIn("only stage 1", fails[0])

    def test_date_is_required_and_validated(self):
        self.make(init=False)
        fails = self.fails(self.main("--stage", "1"))
        self.assertEqual(fails, ["FAIL F00 --date YYYY-MM-DD is required (the date of the freeze; never taken from "
                                 "the clock)"])
        for bad in ("20261117", "2026-13-01", "2026-02-30", "2026-11-17T00:00", "17-11-2026", "2026-1-7", ""):
            with self.subTest(date=bad):
                fails = self.fails(self.main("--stage", "1", "--date", bad))
                self.assertEqual(fails, ["FAIL F00 --date is not a YYYY-MM-DD date"])

    def test_date_with_check_is_refused(self):
        self.make()
        self.written()
        fails = self.fails(self.check("--date", DATE))
        self.assertEqual([ln.split()[1] for ln in fails], ["F00"])
        self.assertIn("taken from the existing freeze file", fails[0])

    def test_output_inside_datasets_or_named_like_the_test_file(self):
        self.make()
        for out, needle in ((self.root / "datasets" / "freeze-1.json", "inside datasets/"),
                            (self.root / DS / "data" / "x.json", "inside datasets/"),
                            (self.tmp / "TEST.csv", "named test.csv"),
                            (self.root / "freeze", "is a folder")):
            with self.subTest(out=out.name):
                if needle == "is a folder":
                    out.mkdir()
                fails = self.fails(self.build("--out", str(out)))
                self.assertEqual([ln.split()[1] for ln in fails], ["F00"])
                self.assertIn(needle, fails[0])
        self.assertEqual(list((self.root / "freeze").iterdir()), [])

    def test_existing_freeze_file_is_never_overwritten(self):
        self.make()
        first = self.written()
        put(self.root, "freeze/freeze-1.json", b"{}\n")
        fails = self.fails(self.build())
        self.assertEqual([ln.split()[1] for ln in fails], ["F00"])
        self.assertIn("never overwritten", fails[0])
        self.assertEqual(self.out.read_bytes(), b"{}\n")
        self.assertNotEqual(first, b"{}\n")

    def test_a_recorded_file_named_like_the_test_file_is_not_opened(self):
        def leaves(t):
            t["tests/test.csv"] = b"synthetic, tracked under tests/\n"
        self.refused(["F00"], "tests/test.csv: a file named test.csv is never opened", leaves=leaves)


# ---------------------------------------------------------------- git precondition (F02) and missing files (F01)

class TestGit(Repo):
    def test_untracked_recorded_file(self):
        self.make()
        git(self.root, "rm", "-q", "--cached", "results/s4/summary.json")
        git(self.root, "commit", "-q", "-m", "untrack one result")
        self.refused(["F02"], "FAIL F02 results/s4/summary.json: not tracked by git")

    def test_new_file_that_was_never_added(self):
        self.make()
        extra = AUG + "/prompts/later.md"
        put(self.root, extra, b"written after the commit\n")
        self.refused(["F02"], f"FAIL F02 {extra}: not tracked by git")

    def test_modified_in_the_working_tree(self):
        self.make()
        put(self.root, "README.md", b"# spec\r\nline two, edited\r\n")
        self.refused(["F02"], "FAIL F02 README.md: modified in the working tree or index (git status ' M')")

    def test_modified_in_the_index(self):
        self.make()
        put(self.root, "results/s3/summary.json", J({"schema": 1, "stage": 3, "edited": True}))
        git(self.root, "add", "results/s3/summary.json")
        self.refused(["F02"], "FAIL F02 results/s3/summary.json: modified in the working tree or index (git status "
                              "'M ')")

    def test_staged_new_file_is_not_at_head(self):
        self.make()
        put(self.root, "tests/test_later.py", b"# added, not committed\n")
        git(self.root, "add", "tests/test_later.py")
        self.refused(["F02"], "FAIL F02 tests/test_later.py: modified in the working tree or index (git status 'A ')")

    def test_untracked_file_under_scripts_or_tests(self):
        self.make()
        put(self.root, "scripts/review/later.py", b"# not committed\n")
        put(self.root, "tests/test_other.py", b"# not committed\n")
        fails = self.refused(["F02", "F02"])
        self.assertIn("FAIL F02 scripts/review/later.py: untracked file under scripts/ or tests/", fails[0])
        self.assertIn("FAIL F02 tests/test_other.py: untracked file under scripts/ or tests/", fails[1])

    def test_ignored_file_under_scripts_is_accepted(self):
        def leaves(t):
            t[".gitignore"] = b"__pycache__/\n"
        self.make(leaves=leaves)
        put(self.root, "scripts/__pycache__/x.pyc", b"\x00")
        self.assertEqual(json.loads(self.written()), expected(self.t, self.head))

    def test_folder_that_is_not_a_repository(self):
        self.make(init=False)
        self.refused(["F02"], "FAIL F02 git: the root is not inside a git work tree")

    def test_repository_without_a_commit(self):
        self.make(commit=False)
        self.refused(["F02"], "FAIL F02 git: no HEAD commit")

    def test_root_below_the_top_level(self):
        t = make_tree()
        self.tmp = Path(tempfile.mkdtemp(prefix="mkfreeze_")).resolve()
        self.addCleanup(rmtree, self.tmp)
        self.root = self.tmp / "outer" / "inner"
        for rel, data in t.items():
            put(self.root, rel, data)
        git_init(self.tmp / "outer")
        commit_all(self.tmp / "outer")
        self.out = self.root / "freeze" / "freeze-1.json"
        fails = self.fails(self.build())
        self.assertEqual(fails, ["FAIL F02 git: the root is not the top level of the work tree"])

    def test_git_not_available(self):
        self.make()
        with mock.patch.object(M.subprocess, "run", side_effect=FileNotFoundError("git")):
            fails = self.fails(self.build())
        self.assertEqual(fails, ["FAIL F02 git is not available (the freeze needs the git state of every recorded "
                                 "file)"])

    def test_git_command_fails(self):
        self.make()
        real = subprocess.run

        def run(cmd, **kw):
            r = real(cmd, **kw)
            if "status" in cmd:
                r.returncode = 128
            return r
        with mock.patch.object(M.subprocess, "run", side_effect=run):
            fails = self.fails(self.build())
        self.assertEqual(fails, ["FAIL F02 git: ls-files or status failed"])

    def test_git_is_only_read(self):
        """Every git call is one of the read-only commands and carries --no-optional-locks."""
        self.make()
        self.written()
        seen, real = [], subprocess.run

        def run(cmd, **kw):
            seen.append(cmd)
            return real(cmd, **kw)
        with mock.patch.object(M.subprocess, "run", side_effect=run):
            self.assertEqual(self.check()[0], 0)
        self.assertTrue(seen)
        for cmd in seen:
            self.assertEqual(cmd[:4], ["git", "--no-optional-locks", "-C", str(self.root)])
            self.assertIn(cmd[4], ("rev-parse", "ls-files", "status", "cat-file", "merge-base"))

    def test_deleted_tracked_file(self):
        self.make()
        for rel in ("results/s2/summary.json", DEV + "/review.csv", f"runs/{A42}/metrics.json", "requirements.txt",
                    DS + "/data/test.csv", f"runs/s2-bert-A-s123/checkpoint/model.safetensors",
                    EXT + "/beta/data/two.parquet", "tests/test_synthetic.py"):
            (self.root / rel).unlink()
        fails = self.refused(["F01"] * 8)
        self.assertEqual(sorted(ln.split()[2] for ln in fails), sorted(p + ":" for p in (
            "results/s2/summary.json", DEV + "/review.csv", f"runs/{A42}/metrics.json", "requirements.txt",
            DS + "/data/test.csv", "runs/s2-bert-A-s123/checkpoint/model.safetensors",
            EXT + "/beta/data/two.parquet", "tests/test_synthetic.py")))

    def test_every_results_file_must_exist(self):
        for rel in ("results/s2/summary.json", "results/s3/summary.json", "results/s3/error_summary.json",
                    "results/s4/summary.json", "results/s4/dev_cases_vs_test.json"):
            with self.subTest(rel=rel):
                self.refused(["F01"], f"FAIL F01 {rel}: missing", after=lambda t, rel=rel: t.pop(rel))

    def test_missing_folders_and_patterns(self):
        def strip(t):
            for p in [p for p in t if p.startswith((f"runs/{C42}/", FIN + "/prompts/", AUG + "/llm/"))]:
                del t[p]
        fails = self.refused(["F01", "F01", "F01", "F01", "F08"], after=strip)
        self.assertIn(f"FAIL F01 runs/{C42}/: the run folder is missing", fails)
        self.assertIn(f"FAIL F01 {AUG}/llm/*.json: no file", fails)
        self.assertIn(f"FAIL F01 {AUG}/llm/aug03_contexts_v1.json: missing", fails)
        self.assertIn(f"FAIL F01 {FIN}/prompts/*.md: no file", fails)


# ---------------------------------------------------------------- checkpoints and runs (F03, F04, F11, F10)

class TestRuns(Repo):
    CK = f"runs/{A42}/checkpoint/model.safetensors"
    RUN = f"runs/{A42}/run.json"

    def test_lfs_pointer(self):
        pointer = b"version x\nsize 9\n"[:15]

        def leaves(t):
            t[self.CK] = pointer
        self.assertLess(len(pointer), LIMIT)
        self.refused(["F03"], f"FAIL F03 {self.CK}: 15 bytes, smaller than 16: a Git LFS pointer", leaves=leaves)

    def test_file_of_exactly_the_limit_is_a_checkpoint(self):
        def leaves(t):
            t[self.CK] = b"0123456789abcdef"
        self.stub(leaves=leaves)
        ck = json.loads(self.written())["runs"][A42]["checkpoint"]
        self.assertEqual(ck, {"path": self.CK, "size": 16, "sha256": sha(b"0123456789abcdef")})

    def test_with_the_real_limit_every_synthetic_checkpoint_is_a_pointer(self):
        self.make()
        with mock.patch.object(M, "POINTER_LIMIT", REAL_LIMIT):
            fails = self.refused(["F03"] * 12)
        self.assertIn("35 bytes, smaller than 1048576", fails[0])

    def test_checkpoint_size_differs_from_run_json(self):
        def grow(t):
            t[self.CK] += b"\x00"
        self.refused(["F04"], f"FAIL F04 {self.CK}: size 39 != run.json 38", after=grow)

    def test_checkpoint_hash_differs_from_run_json(self):
        def flip(t):
            t[self.CK] = t[self.CK][:-1] + b"\x01"
        fails = self.refused(["F04"], after=flip)
        self.assertEqual(fails[0], f"FAIL F04 {self.CK}: sha256 on disk: {sha(self.t[self.CK])[:12]} != "
                                   f"{sha(checkpoint_bytes(A42))[:12]} (run.json checkpoint > sha256)")

    def test_run_json_names_another_checkpoint_hash(self):
        self.refused(["F04", "F07"], f"{self.CK}: sha256 on disk",
                     tweak={self.RUN: setter("checkpoint", "sha256", value="ab" * 32)})

    def test_identity_differs_from_the_run_id(self):
        for key, value in (("seed", 123), ("seed", True), ("condition", "C"), ("model_key", "bert"),
                           ("stage", "s4"), ("run_id", "s2-roberta-A-s123")):
            with self.subTest(key=key, value=value):
                self.refused(["F11"], f"FAIL F11 {self.RUN}: {key} do not match the run ID",
                             tweak={self.RUN: setter(key, value=value)})

    def test_status_not_completed(self):
        self.refused(["F11"], f"FAIL F11 {self.RUN}: status is \"failed\", not \"completed\"",
                     tweak={self.RUN: setter("status", value="failed")})

    def test_checkpoint_record(self):
        for key, value in (("kept", False), ("step", 10),
                           ("path", f"runs/{A42}/checkpoint.staging/model.safetensors")):
            with self.subTest(key=key):
                self.refused(["F11"], f"FAIL F11 {self.RUN}: checkpoint is not kept",
                             tweak={self.RUN: setter("checkpoint", key, value=value)})

    def test_selected_step_differs_from_the_checkpoint_step(self):
        # thresholds.json records selected_step 20 for this run: the stale threshold is reported too
        self.refused(["F11", "F07"], "is not the selected step 10", "selected_step 20 disagrees with the frozen run",
                     tweak={self.RUN: setter("selection", "selected_step", value=10)})

    def test_lock_file(self):
        self.make()
        put(self.root, f"runs/{C42}.lock", b"pid 1\n")
        self.refused(["F11"], f"FAIL F11 runs/{C42}.lock exists")

    def test_missing_keys_and_wrong_types(self):
        for fn, needle in ((remover("selection"), "key selection > selected_step is missing"),
                           (setter("selection", "selected_step", value="20"),
                            "selection > selected_step is not an integer"),
                           (setter("selection", "selected_step", value=True),
                            "selection > selected_step is not an integer"),
                           (setter("checkpoint", value=None), "checkpoint is not an object"),
                           (setter("checkpoint", "sha256", value="AB" * 32), "checkpoint > sha256 is not a SHA-256"),
                           (remover("checkpoint", "size"), "key checkpoint > size is missing"),
                           (remover("config"), "key config > sha256 is missing"),
                           (remover("env", "requirements_sha256"), "key env > requirements_sha256 is missing"),
                           (remover("lists"), "key lists > files_sha256 is missing"),
                           (remover("lists", "files_sha256", "train_clean.csv"),
                            "lists > files_sha256 > train_clean.csv is not a SHA-256")):
            with self.subTest(needle=needle):
                self.refused(["F10"], f"FAIL F10 {self.RUN}: {needle}", tweak={self.RUN: fn})

    def test_run_json_is_not_json(self):
        def junk(t):
            t[self.RUN] = b"{not json\n"
        self.refused(["F10"], f"FAIL F10 {self.RUN}: not UTF-8 JSON", after=junk)

    def test_run_json_is_a_list(self):
        def junk(t):
            t[self.RUN] = b"[]\n"
        self.refused(["F10"], f"FAIL F10 {self.RUN}: not a JSON object", after=junk)

    def test_run_was_trained_with_another_config(self):
        self.refused(["F04"], f"FAIL F04 {self.RUN} config > sha256: abababababab != "
                              f"{lf(leaf_files()['configs/train.json'])[:12]} (configs/train.json on disk)",
                     tweak={self.RUN: setter("config", "sha256", value="ab" * 32)})

    def test_run_was_trained_with_other_requirements(self):
        self.refused(["F04"], f"FAIL F04 {self.RUN} env > requirements_sha256: cdcdcdcdcdcd != "
                              f"{SHA_REQUIREMENTS[:12]} (requirements.txt on disk)",
                     tweak={self.RUN: setter("env", "requirements_sha256", value="cd" * 32)})

    def test_run_was_trained_on_other_lists(self):
        for name in LIST6:
            with self.subTest(name=name):
                self.refused(["F04"], f"FAIL F04 {self.RUN} lists > files_sha256 > {name}: efefefefefef != ",
                             tweak={self.RUN: setter("lists", "files_sha256", name, value="ef" * 32)})

    def test_config_changed_after_the_runs(self):
        """A config edited after training: all twelve runs name the old hash."""
        def edit(t):
            t["configs/train.json"] = J({"seeds": SEEDS, "threshold": 0.5, "epochs": 4})
        self.refused(["F04"] * 12, "config > sha256", after=edit)

    def test_seeds_of_the_config(self):
        for seeds in ([], [42, 42], [42, "123"], [True], "42"):
            with self.subTest(seeds=seeds):
                def leaves(t, seeds=seeds):
                    t["configs/train.json"] = J({"seeds": seeds, "threshold": 0.5})
                fails = self.refused(["F10"], leaves=leaves)
                self.assertIn("FAIL F10 configs/train.json: seeds is not", fails[0])

    def test_runs_follow_the_seeds_of_the_config(self):
        def leaves(t):
            t["configs/train.json"] = J({"seeds": [42, 123, 2026, 7], "threshold": 0.5})
        fails = self.refused(["F01"] * 4 + ["F07"], leaves=leaves)
        self.assertEqual(fails[:4], [f"FAIL F01 runs/{r}/: the run folder is missing" for r in (
            "s2-bert-A-s7", "s2-deberta-A-s7", "s2-roberta-A-s7", "s4-roberta-C-s7")])
        self.assertIn("no entry (an object with the keys threshold, logit, run_id) for s2-roberta-A-s7, "
                      "s4-roberta-C-s7", fails[4])


# ---------------------------------------------------------------- dataset and lists (F04, F05, F06, F10)

class TestDatasetAndLists(Repo):
    def test_dataset_file_size_differs_from_the_manifest(self):
        def size(doc):
            doc["files"][3]["size_bytes"] += 1
        self.refused(["F04"], f"FAIL F04 {DS}/data/test.csv: size on disk: 35 != 36 (the manifest's size_bytes)",
                     tweak={DSM: size})

    def test_dataset_path_that_is_not_a_file(self):
        def leaves(t):
            del t[DS + "/data/validation.csv"]
            t[DS + "/data/validation.csv/part.txt"] = b"a folder where the split file should be\n"

        def size(doc):
            doc["files"].insert(2, {"path": "data/validation.csv", "size_bytes": 0, "sha256": "2b" * 32})
        self.refused(["F01"], f"FAIL F01 {DS}/data/validation.csv: not a regular file", leaves=leaves,
                     tweak={DSM: size})

    def test_dataset_manifest_keys(self):
        for fn, needle in ((remover("revision"), "key revision is missing"),
                           (remover("splits", "validation"), "key splits > validation > file is missing"),
                           (setter("splits", "test", "file", value="data/other.csv"),
                            "files has no safe entry for \"data/other.csv\" (split test)"),
                           (setter("files", value=[]),
                            "files has no safe entry for \"data/train.csv\" (split train)"),
                           (setter("files", value={}), "files is not a list")):
            with self.subTest(needle=needle):
                self.refused(["F10"], f"FAIL F10 {DSM}: {needle}", tweak={DSM: fn})

    def test_dataset_file_entry_without_a_hash(self):
        def strip(doc):
            del doc["files"][1]["sha256"]
        self.refused(["F10"], f"FAIL F10 {DSM} files[data/train.csv]: key sha256 is missing", tweak={DSM: strip})

    def test_lists_are_not_final(self):
        fails = self.refused(["F05"], tweak={META: setter("test_clean_stage", value="pre-augmentation")})
        self.assertIn(f"FAIL F05 {META}: test_clean_stage is \"pre-augmentation\", freeze 1 needs \"final\"",
                      fails[0])

    def test_trial_lists(self):
        for value in (True, None, 0, "false"):
            with self.subTest(value=value):
                self.refused(["F05"], f"FAIL F05 {META}: trial is {json.dumps(value)}, freeze 1 needs false",
                             tweak={META: setter("trial", value=value)})

    def test_test_clean_header(self):
        def leaves(t):
            t[LD + "/test_clean.csv"] = b"sentence_id,label\ntest:000000,1\n"
        self.refused(["F05"], f"FAIL F05 {LD}/test_clean.csv: the header is not sentence_id", leaves=leaves)

    def test_test_excluded_header_and_reasons(self):
        row = "a row is not sentence_id,reason"
        for data, needle in ((b"sentence_id,why\ntest:000001,validation_overlap\n",
                              "the header is not sentence_id,reason"),
                             (b"sentence_id,reason\ntest:000001,other_overlap\n", row),
                             (b"sentence_id,reason\ntest:000001,1,train_overlap\n", row),
                             (b"sentence_id,reason\ntest:000001\n", row)):
            with self.subTest(needle=needle, data=data):
                def leaves(t, data=data):
                    t[LD + "/test_excluded.csv"] = data
                self.refused(["F05"], f"FAIL F05 {LD}/test_excluded.csv: {needle}", leaves=leaves)

    def test_reason_counts_follow_the_file(self):
        def leaves(t):
            t[LD + "/test_excluded.csv"] = (b"sentence_id,reason\ntest:000001,train_overlap\n"
                                            b"test:000003,train_overlap\ntest:000004,augmentation_overlap\n"
                                            b"test:000005,train_overlap\n")
        self.stub(leaves=leaves)
        self.assertEqual(json.loads(self.written())["lists"]["test_clean"], {
            "rows": 2, "excluded_rows": 4,
            "excluded_by_reason": {"train_overlap": 3, "augmentation_overlap": 1, "validation_overlap": 0}})

    def test_list_file_differs_from_lists_meta(self):
        for name in ("similar_pairs.csv", "test_clean.csv", "test_excluded.csv"):
            with self.subTest(name=name):
                def edit(t, name=name):
                    more = b"test:000009,validation_overlap\n" if name == "test_excluded.csv" else b"x\n"
                    t[f"{LD}/{name}"] += more
                self.refused(["F04"], f"FAIL F04 {META} outputs > {name} > sha256: ", f"({LD}/{name} on disk)",
                             after=edit)

    def test_training_list_differs_from_lists_meta_and_from_the_runs(self):
        def edit(t):
            t[LD + "/train_clean.csv"] += b"train:000002,0,dddddddddddd\n"
        fails = self.refused(["F04"] * 13, after=edit)
        self.assertIn(f"FAIL F04 {META} outputs > train_clean.csv > sha256: ", fails[0])
        self.assertIn("lists > files_sha256 > train_clean.csv", fails[1])

    def test_review_and_script_and_revision_of_lists_meta(self):
        for path, value, needle in ((("review", "sha256"), "ab" * 32, f"review > sha256: abababababab != "),
                                    (("script", "sha256_lf"), "cd" * 32, "script > sha256_lf: cdcdcdcdcdcd != "),
                                    (("dataset", "revision"), "e" * 40, "dataset > revision: ")):
            with self.subTest(path=path):
                self.refused(["F04"], f"FAIL F04 {META} {needle}", tweak={META: setter(*path, value=value)})

    def test_build_lists_changed_after_the_lists_were_built(self):
        def edit(t):
            t["scripts/build_lists.py"] += b"# edited\n"
        self.refused(["F04"], f"FAIL F04 {META} script > sha256_lf: ", "(scripts/build_lists.py on disk)", after=edit)

    def test_final_test_clean_was_built_with_another_augmentation(self):
        for name in ("train_aug.csv", "cases.csv"):
            with self.subTest(name=name):
                self.refused(["F06"], f"FAIL F06 {META} augmentation > files > {name}: 121212121212 != ",
                             "test_clean was built with another augmentation",
                             tweak={META: setter("augmentation", "files", name, value="12" * 32)})

    def test_lists_meta_without_an_augmentation_record(self):
        for fn in (setter("augmentation", value=None), remover("augmentation"),
                   setter("augmentation", "files", value=None), setter("augmentation", "files", value={})):
            with self.subTest():
                fails = self.refused(["F06", "F06"], tweak={META: fn})
                self.assertIn(f"FAIL F06 {META}: augmentation > files records no SHA-256 for train_aug.csv", fails[0])
                self.assertIn("records no SHA-256 for cases.csv", fails[1])

    def test_lists_meta_augmentation_folder(self):
        self.refused(["F06"], f"FAIL F06 {META}: augmentation > dir is \"data/cases/other\", not data/cases/train",
                     tweak={META: setter("augmentation", "dir", value="data/cases/other")})

    def test_csv_without_a_final_lf(self):
        def leaves(t):
            t[LD + "/train_review.csv"] = t[LD + "/train_review.csv"][:-1]
        self.refused(["F10"], f"FAIL F10 {LD}/train_review.csv: no final LF", leaves=leaves)

    def test_lists_meta_unreadable_or_incomplete(self):
        for fn, needle in ((remover("trial"), "key trial is missing"),
                           (remover("test_clean_stage"), "key test_clean_stage is missing"),
                           (remover("outputs", "test_clean.csv"), "key outputs > test_clean.csv > sha256 is missing"),
                           (setter("outputs", "similar_pairs.csv", value={"rows": 1}),
                            "key outputs > similar_pairs.csv > sha256 is missing"),
                           (remover("review"), "key review > sha256 is missing"),
                           (remover("script"), "key script > sha256_lf is missing"),
                           (remover("dataset"), "key dataset > revision is missing")):
            with self.subTest(needle=needle):
                self.refused(["F10"], f"FAIL F10 {META}: {needle}", tweak={META: fn})

    def test_lists_meta_not_utf8(self):
        def junk(t):
            t[META] = b"\xff\xfe{}\n"
        self.refused(["F10"], f"FAIL F10 {META}: not UTF-8 JSON", after=junk)


# ---------------------------------------------------------------- augmentation (F06, F04, F10)

class TestAugmentation(Repo):
    def test_set_is_not_ready(self):
        for value in (False, None, "true", 1):
            with self.subTest(value=value):
                self.refused(["F06"], f"FAIL F06 {AUG_META}: ready_for_training is {json.dumps(value)}, freeze 1 "
                                      "needs true", tweak={AUG_META: setter("ready_for_training", value=value)})

    def test_c_run_was_trained_on_another_augmentation(self):
        for name in ("cases.csv", "review.csv", "train_aug.csv"):
            with self.subTest(name=name):
                fails = self.refused(["F06"], tweak={f"runs/{C42}/run.json": setter(
                    "data", "augmentation", "files_sha256", name, value="34" * 32)})
                self.assertEqual(fails[0], f"FAIL F06 runs/{C42}/run.json data > augmentation > files_sha256 > "
                                           f"{name}: 343434343434 != {sha(self.t[f'{AUG}/{name}'])[:12]} (the frozen "
                                           f"{AUG}/{name}: condition C was not trained on the frozen augmentation)")

    def test_augmentation_changed_after_the_c_runs(self):
        """review.csv edited after training: every C run and cases_meta.json name the old file."""
        def edit(t):
            t[AUG + "/review.csv"] += b"aug-04-001,222222222222,keep,,,tester\n"
        fails = self.refused(["F06", "F06", "F06", "F04"], after=edit)
        self.assertIn("runs/s4-roberta-C-s123/run.json data > augmentation > files_sha256 > review.csv", fails[0])
        self.assertIn(f"FAIL F04 {AUG_META} outputs > review.csv > sha256", fails[3])

    def test_c_run_without_an_augmentation_record(self):
        for fn, needle in ((remover("data", "augmentation"), "key data > augmentation > files_sha256 > cases.csv is "
                                                              "missing"),
                           (remover("data", "augmentation", "files_sha256", "test_overlap.json"),
                            "key data > augmentation > files_sha256 > test_overlap.json is missing"),
                           (setter("data", "augmentation", "files_sha256", "cases_meta.json", value=None),
                            "data > augmentation > files_sha256 > cases_meta.json is not a SHA-256")):
            with self.subTest(needle=needle):
                self.refused(["F10"], f"FAIL F10 runs/{C42}/run.json: {needle}", tweak={f"runs/{C42}/run.json": fn})

    def test_an_a_run_needs_no_augmentation_record(self):
        self.stub()
        self.assertNotIn("augmentation", json.loads(self.t[f"runs/{A42}/run.json"])["data"])
        self.assertEqual(sorted(json.loads(self.written())["augmentation"]["c_runs"]),
                         ["s4-roberta-C-s123", "s4-roberta-C-s2026", "s4-roberta-C-s42"])

    def test_train_aug_is_malformed_or_empty(self):
        head = b"sentence_id,label,text_sha256_12\n"
        row = b"aug-04-001,0,222222222222"
        for data in (head, b"sentence_id,label\naug-04-001,0\n", b"case_id,label,text_sha256_12\n" + row + b"\n",
                     head + b"aug-04-001,2,222222222222\n", head + b"aug-04-001,0\n", head + row + b",x\n",
                     head + row, head + row + b"\n" + row.replace(b"001", b"002"), head + row + b"\n\n"):
            with self.subTest(data=data):
                def leaves(t, data=data):
                    t[AUG + "/train_aug.csv"] = data
                self.refused(["F06"], f"FAIL F06 {AUG}/train_aug.csv: not a list of at least one row", leaves=leaves)

    def test_label_counts_follow_the_file(self):
        def leaves(t):
            t[AUG + "/train_aug.csv"] += b"aug-05-002,0,444444444444\naug-05-003,0,555555555555\n"
        self.stub(leaves=leaves)
        self.assertEqual(json.loads(self.written())["augmentation"]["train_aug"],
                         {"rows": 5, "label_counts": {"0": 3, "1": 2}})

    def test_cases_meta_outputs(self):
        for name in ("originals.csv", "cases.csv", "review.csv", "train_aug.csv"):
            with self.subTest(name=name):
                self.refused(["F04"], f"FAIL F04 {AUG_META} outputs > {name} > sha256: 565656565656 != ",
                             tweak={AUG_META: setter("outputs", name, "sha256", value="56" * 32)})

    def test_originals_changed_after_the_set_was_built(self):
        def edit(t):
            t[AUG + "/originals.csv"] += b"train:000001,1\n"
        self.refused(["F04"], f"FAIL F04 {AUG_META} outputs > originals.csv > sha256", after=edit)

    def test_build_train_aug_changed_after_the_set_was_built(self):
        def edit(t):
            t["scripts/build_train_aug.py"] = b"# synthetic build_train_aug, edited\r\n"
        self.refused(["F04"], f"FAIL F04 {AUG_META} script > sha256_lf: {SHA_BUILD_AUG_LF[:12]} != ", after=edit)

    def test_prompt_or_llm_output_changed_after_the_set_was_built(self):
        for rel, key in ((AUG + "/prompts/aug03_contexts_v1.md", "prompt_sha256_lf"),
                         (AUG + "/llm/aug03_contexts_v1.json", "sha256_lf")):
            with self.subTest(rel=rel):
                def edit(t, rel=rel):
                    t[rel] += b"edited\n"
                self.refused(["F04"], f"FAIL F04 {AUG_META} generators > aug03_contexts_v1 > {key}: ",
                             f"({rel} on disk)", after=edit)

    def test_cases_meta_keys(self):
        for fn, needle in ((remover("ready_for_training"), "key ready_for_training is missing"),
                           (remover("outputs", "train_aug.csv"), "key outputs > train_aug.csv > sha256 is missing"),
                           (remover("script"), "key script > sha256_lf is missing"),
                           (remover("generators"), "key generators is missing"),
                           (setter("generators", value=[]), "generators is not an object"),
                           (remover("generators", "aug03_contexts_v1", "prompt"),
                            "key generators > aug03_contexts_v1 > prompt is missing"),
                           (setter("generators", "aug03_contexts_v1", "file", value="../cases.csv"),
                            "generators > aug03_contexts_v1 > file is not a safe relative path")):
            with self.subTest(needle=needle):
                self.refused(["F10"], f"FAIL F10 {AUG_META}: {needle}", tweak={AUG_META: fn})

    def test_generator_names_a_missing_file(self):
        self.refused(["F01"], f"FAIL F01 {AUG}/llm/gone.json: missing",
                     tweak={AUG_META: setter("generators", "aug03_contexts_v1", "file", value="llm/gone.json")})

    def test_no_prompt_file(self):
        def leaves(t):
            del t[AUG + "/prompts/aug03_contexts_v1.md"], t[AUG + "/prompts/aug_review_v1.md"]
        fails = self.refused(["F01", "F01"], leaves=leaves)
        self.assertEqual(fails, [f"FAIL F01 {AUG}/prompts/*.md: no file",
                                 f"FAIL F01 {AUG}/prompts/aug03_contexts_v1.md: missing"])


# ---------------------------------------------------------------- thresholds (F07)

def entry(seed, cond):
    def pick(doc):
        return next(e for e in doc["thresholds"] if (e["seed"], e["condition"]) == (seed, cond))
    return pick


def on_entry(seed, cond, /, **values):
    def fn(doc):
        e = entry(seed, cond)(doc)
        for k, v in values.items():
            if v is remover:
                del e[k]
            else:
                e[k] = v
    return fn


class TestThresholds(Repo):
    def one(self, fn, *needles, codes=("F07",)):
        return self.refused(list(codes), *needles, tweak={THR: fn})

    def test_missing_entry(self):
        def drop(doc):
            doc["thresholds"] = [e for e in doc["thresholds"] if (e["seed"], e["condition"]) != (123, "D")]
        self.one(drop, f"FAIL F07 {THR}: no entry (an object with the keys threshold, logit, run_id) for "
                       "s4-roberta-C-s123")

    def test_missing_condition_of_every_seed(self):
        def drop(doc):
            doc["thresholds"] = [e for e in doc["thresholds"] if e["condition"] != "B"]
        self.one(drop, "for s2-roberta-A-s123, s2-roberta-A-s2026, s2-roberta-A-s42")

    def test_duplicate_entry(self):
        def twice(doc):
            doc["thresholds"].append(dict(entry(42, "B")(doc)))
        self.one(twice, f"FAIL F07 {THR} B s42: more than one entry for s2-roberta-A-s42")

    def test_entry_for_an_unexpected_run(self):
        for run_id in ("s2-bert-A-s42", "s2-roberta-A-s7", "s4-roberta-C-s43", 42):
            with self.subTest(run_id=run_id):
                def other(doc, run_id=run_id):
                    doc["thresholds"].append({**entry(42, "B")(doc), "run_id": run_id})
                self.one(other, f"FAIL F07 {THR}: entry for an unexpected run {json.dumps(run_id)} (want ")

    def test_entry_of_the_wrong_run(self):
        """The B entry of seed 42 names the C run: D has two entries and B none."""
        self.one(on_entry(42, "B", run_id=C42), "D s42: condition \"B\" disagrees with the frozen run",
                 "D s42: checkpoint_sha256", "D s42: more than one entry for s4-roberta-C-s42",
                 "no entry (an object with the keys threshold, logit, run_id) for s2-roberta-A-s42",
                 codes=["F07"] * 4)

    def test_cap_met_must_be_a_boolean(self):
        for value in (remover, None, 1, "true"):
            with self.subTest(value=value):
                self.one(on_entry(2026, "D", cap_met=value), f"FAIL F07 {THR} D s2026: cap_met must be true or false")

    def test_threshold_must_be_a_float32_in_the_open_interval(self):
        for value in (0.3, 0, 1, 0.0, 1.0, 1.5, -0.25, "0.5", None, True):
            with self.subTest(value=value):
                self.one(on_entry(42, "D", threshold=value),
                         f"FAIL F07 {THR} D s42: threshold is not a float32 value in (0, 1) stored as a float")

    def test_logit_must_be_a_candidate(self):
        for value in (0.33, 8.05, -8.05, 100, "0.5", None, True, float("nan")):
            with self.subTest(value=value):
                self.one(on_entry(42, "D", logit=value),
                         f"FAIL F07 {THR} D s42: logit is not a candidate k/20 within -8..8")

    def test_threshold_must_be_the_sigmoid_of_the_logit(self):
        self.one(on_entry(42, "D", logit=K[(42, "D")] / 20 + 0.05), f"FAIL F07 {THR} D s42: threshold is not "
                                                                    "sigmoid(logit) (float32)")
        self.one(on_entry(42, "D", threshold=threshold_of(K[(42, "D")] + 1)), "threshold is not sigmoid(logit)")

    def test_extreme_candidates_are_accepted(self):
        def edge(doc):
            entry(42, "B")(doc).update(logit=-8, threshold=threshold_of(-160))
            entry(42, "D")(doc).update(logit=8.0, threshold=threshold_of(160))
        self.stub(tweak={THR: edge})
        by_seed = json.loads(self.written())["thresholds"]["by_seed"]
        self.assertEqual(by_seed["42"]["B"]["logit"], -8)
        self.assertEqual(by_seed["42"]["D"]["threshold"], f32(1 / (1 + math.exp(-8))))

    def test_optional_keys_must_agree_with_the_frozen_run(self):
        ck = sha(checkpoint_bytes(C42))
        for key, value, shown in (("seed", 123, "123"), ("seed", True, "true"), ("condition", "B", "\"B\""),
                                  ("selected_step", 10, "10"), ("checkpoint_sha256", "ab" * 32, "abababababab")):
            with self.subTest(key=key, value=value):
                want = {"seed": "42", "condition": "\"D\"", "selected_step": "20", "checkpoint_sha256": ck[:12]}[key]
                self.one(on_entry(42, "D", **{key: value}), f"FAIL F07 {THR} D s42: {key} {shown} disagrees with the "
                                                            f"frozen run {C42} ({want})")

    def test_optional_keys_may_be_absent(self):
        def strip(doc):
            for e in doc["thresholds"]:
                for k in ("seed", "condition", "selected_step", "checkpoint_sha256", "k", "chosen"):
                    del e[k]
            del doc["model"], doc["inputs_sha256"]["configs/train.json"]
            for k in [k for k in doc["inputs_sha256"] if k.endswith("/metrics.json")]:
                del doc["inputs_sha256"][k]
        self.stub(tweak={THR: strip})
        self.assertEqual(json.loads(self.written()), expected(self.t, self.head))

    def test_thresholds_selected_on_other_run_files(self):
        for name in RUN_FILES:
            with self.subTest(name=name):
                self.one(setter("inputs_sha256", f"runs/{A42}/{name}", value="78" * 32),
                         f"FAIL F07 {THR}: the recorded SHA-256 of runs/{A42}/{name}: 787878787878 != ",
                         "the thresholds were selected on other run files")

    def test_run_file_changed_after_the_selection(self):
        def edit(t):
            t[f"runs/{C42}/predictions_validation.csv"] += b"20,validation:000001,x\n"
        self.refused(["F07"], f"the recorded SHA-256 of runs/{C42}/predictions_validation.csv", after=edit)

    def test_run_files_must_be_pinned(self):
        def strip(doc):
            del doc["inputs_sha256"][f"runs/{A42}/run.json"]
            del doc["inputs_sha256"][f"runs/{C42}/predictions_validation.csv"]
        self.one(strip, f"FAIL F07 {THR}: no SHA-256 is recorded for runs/{A42}/run.json, "
                        f"runs/{C42}/predictions_validation.csv")

    def test_model_must_be_the_representative_one(self):
        self.one(setter("model", value="bert"), f"FAIL F07 {THR}: model is \"bert\", not the representative model "
                                                "roberta")

    def test_config_threshold_must_be_one_half(self):
        for value in (0.4, 1, "0.5", None):
            with self.subTest(value=value):
                def leaves(t, value=value):
                    t["configs/train.json"] = J({"seeds": SEEDS, "threshold": value})
                self.refused(["F07"], f"FAIL F07 configs/train.json: threshold is {json.dumps(value)}, conditions A "
                                      "and C need 0.5", leaves=leaves)

    def test_config_without_a_threshold(self):
        def leaves(t):
            t["configs/train.json"] = J({"seeds": SEEDS})
        self.refused(["F10"], "FAIL F10 configs/train.json: key threshold is missing", leaves=leaves)

    def test_thresholds_file_missing_or_unreadable(self):
        self.refused(["F01"], f"FAIL F01 {THR}: missing", after=lambda t: t.pop(THR))
        self.refused(["F10"], f"FAIL F10 {THR}: not UTF-8 JSON", after=lambda t: t.update({THR: b"[1, 2\n"}))
        self.refused(["F10"], f"FAIL F10 {THR}: not a JSON object", after=lambda t: t.update({THR: b"[1, 2]\n"}))


# ---------------------------------------------------------------- final cases (F08) and external sources (F12, F04)

class TestFinalCases(Repo):
    P2, P4 = FIN + "/prompts/rob02_final_v1.md", FIN + "/prompts/rob04_final_v1.md"

    def test_prompt_changed_after_the_spec(self):
        def edit(t):
            t[self.P4] = b"prompt four, edited\r\n"
        fails = self.refused(["F08"], after=edit)
        new = lf(b"prompt four, edited\n")
        self.assertEqual(fails[0], f"FAIL F08 {SPEC} prompts_sha256_lf > {self.P4}: {SHA_FINAL_PROMPT4_LF[:12]} != "
                                   f"{new[:12]} (the prompt file)")

    def test_spec_names_another_hash(self):
        self.refused(["F08"], f"FAIL F08 {SPEC} prompts_sha256_lf > {self.P2}: 909090909090 != ",
                     tweak={SPEC: setter("prompts_sha256_lf", self.P2, value="90" * 32)})

    def test_spec_hash_is_the_lf_hash(self):
        """The spec must hold the LF hash of the CRLF prompt, not the hash of its bytes."""
        self.refused(["F08"], f"prompts_sha256_lf > {self.P4}",
                     tweak={SPEC: setter("prompts_sha256_lf", self.P4, value=sha(b"prompt four\r\n"))})

    def test_prompt_file_that_the_spec_does_not_name(self):
        def add(t):
            t[FIN + "/prompts/rob03_final_v1.md"] = b"prompt three\n"
        self.refused(["F08"], f"FAIL F08 {SPEC}: prompts_sha256_lf names 2 file(s), {FIN}/prompts/ holds 3: only in "
                              f"the spec [], only in the folder ['{FIN}/prompts/rob03_final_v1.md']", after=add)

    def test_spec_names_a_prompt_that_is_not_there(self):
        self.refused(["F08"], "names 3 file(s)", f"only in the spec ['{FIN}/prompts/gone.md']",
                     tweak={SPEC: setter("prompts_sha256_lf", FIN + "/prompts/gone.md", value="90" * 32)})

    def test_generator_code_must_define_the_three_names(self):
        base = leaf_files()["scripts/build_dev_cases.py"]
        for old, new, lacking in ((b"ROB01 = {}", b"ROB02 = {}", "ROB01"),
                                  (b"def assemble(", b"def assemble_all(", "assemble"),
                                  (b"def plan_rob03(", b"    def plan_rob03(", "plan_rob03"),
                                  (base, b"# empty\n", "ROB01, assemble, plan_rob03")):
            with self.subTest(lacking=lacking):
                def leaves(t, old=old, new=new):
                    t["scripts/build_dev_cases.py"] = t["scripts/build_dev_cases.py"].replace(old, new)
                self.refused(["F08"], f"FAIL F08 scripts/build_dev_cases.py: does not define {lacking} (",
                             leaves=leaves)

    def test_generator_code_with_crlf(self):
        def leaves(t):
            t["scripts/build_dev_cases.py"] = t["scripts/build_dev_cases.py"].replace(b"\n", b"\r\n")
        self.stub(leaves=leaves)
        self.assertEqual(json.loads(self.written()), expected(self.t, self.head))

    def test_spec_keys(self):
        for fn, needle in ((remover("prompts_sha256_lf"), "key prompts_sha256_lf is missing"),
                           (remover("extraction", "seed"), "key extraction > seed is missing"),
                           (setter("extraction", "seed", value="20261117"), "extraction > seed is not an integer"),
                           (remover("targets"), "key targets is missing"),
                           (setter("targets", value=[50]), "targets is not an object"),
                           (setter("generators", "llm_model", value=""), "generators > llm_model is not a non-empty "
                                                                         "string")):
            with self.subTest(needle=needle):
                self.refused(["F10"], f"FAIL F10 {SPEC}: {needle}", tweak={SPEC: fn})

    def test_copied_values_follow_the_spec(self):
        def other(doc):
            doc["extraction"]["seed"] = 7
            doc["targets"] = {"ROB-02": {"1": 3}}
            doc["generators"]["llm_model"] = "other-model"
        self.stub(tweak={SPEC: other})
        final = json.loads(self.written())["final_cases"]
        self.assertEqual((final["extraction_seed"], final["targets"], final["llm_model"]),
                         (7, {"ROB-02": {"1": 3}}, "other-model"))


class TestExternal(Repo):
    MAN = EXT + "/alpha/download_manifest.json"
    PARQUET = EXT + "/alpha/data/part.parquet"

    def test_raw_file_hash_differs_from_the_manifest(self):
        def flip(t):
            t[self.PARQUET] = b"PAR1\r\n\x00alphA"
        fails = self.refused(["F04"], after=flip)
        new = sha(b"PAR1\r\n\x00alphA")
        self.assertEqual(fails[0], f"FAIL F04 {self.PARQUET}: sha256 on disk: {new[:12]} != {SHA_ALPHA_PARQUET[:12]} "
                                   f"({self.MAN} sha256)")

    def test_raw_file_size_differs_from_the_manifest(self):
        def grow(t):
            t[EXT + "/beta/data/two.parquet"] += b"!"
        fails = self.refused(["F04", "F04"], after=grow)
        self.assertEqual(fails[0], f"FAIL F04 {EXT}/beta/data/two.parquet: size on disk: 14 != 13 "
                                   f"({EXT}/beta/download_manifest.json size_bytes)")
        self.assertIn("sha256 on disk", fails[1])

    def test_raw_text_file_is_hashed_as_bytes(self):
        """The CRLF card of a source turned into LF: the manifest's byte hash no longer matches."""
        def strip(t):
            t[EXT + "/alpha/README.md"] = b"card\nalpha\n"
        fails = self.refused(["F04", "F04"], after=strip)
        self.assertIn(f"{EXT}/alpha/README.md: size on disk: 11 != 13", fails[0])
        self.assertIn(f"sha256 on disk: {SHA_ALPHA_CARD_LF[:12]} != {SHA_ALPHA_CARD[:12]}", fails[1])

    def test_rules_name_another_manifest_hash(self):
        self.refused(["F04"], f"FAIL F04 {RULES} sources > alpha > manifest_sha256: abababababab != ",
                     f"({self.MAN} on disk)",
                     tweak={RULES: setter("sources", "alpha", "manifest_sha256", value="ab" * 32)})

    def test_rules_files_differ_from_the_manifest(self):
        def size(doc):
            doc["sources"]["beta"]["files"][1]["size_bytes"] += 1

        def drop(doc):
            del doc["sources"]["beta"]["files"][2]

        def twice(doc):
            doc["sources"]["beta"]["files"].append(doc["sources"]["beta"]["files"][0])

        def other(doc):
            doc["sources"]["beta"]["files"][0]["sha256"] = "ab" * 32

        def junk(doc):
            doc["sources"]["beta"]["files"][0] = "README.md"
        for fn in (size, drop, twice, other, junk):
            with self.subTest(fn=fn.__name__):
                self.refused(["F04"], f"FAIL F04 {RULES} sources > beta > files: path, size_bytes and sha256 differ "
                                      f"from the files of {EXT}/beta/download_manifest.json", tweak={RULES: fn})

    def test_rules_and_manifest_differ_in_dataset_or_revision(self):
        for key, value in (("revision", "9" * 40), ("dataset", "org/other")):
            with self.subTest(key=key):
                self.refused(["F12"],
                             f"FAIL F12 {self.MAN}: dataset or revision differs from {RULES} sources > alpha",
                             tweak={RULES: setter("sources", "alpha", key, value=value)})

    def test_copied_source_values(self):
        self.stub()
        alpha = json.loads(self.written())["external"]["sources"]["alpha"]
        self.assertEqual((alpha["dataset"], alpha["revision"], alpha["adopted_splits"], alpha["dir"]),
                         ("org/alpha", "1" * 40, ["train"], EXT + "/alpha"))

    def test_manifest_of_a_source_that_the_rules_do_not_adopt(self):
        def leaves(t):
            t[EXT + "/gamma/download_manifest.json"] = J({"dataset": "org/gamma", "revision": "3" * 40, "files": []})
        self.refused(["F12"], f"FAIL F12 {EXT}: download manifest of a source that {RULES} does not adopt: gamma",
                     leaves=leaves)

    def test_rules_without_sources(self):
        fails = self.refused(["F12", "F12"], tweak={RULES: setter("sources", value={})})
        self.assertEqual(fails, [
            f"FAIL F12 {RULES}: sources names no source",
            f"FAIL F12 {EXT}: download manifest of a source that {RULES} does not adopt: alpha, beta"])

    def test_source_folder_and_manifest_path(self):
        for key, value in (("dir", EXT + "/other"), ("manifest", EXT + "/alpha/manifest.json"),
                           ("dir", "data/elsewhere/alpha"), ("manifest", "../alpha/download_manifest.json")):
            with self.subTest(key=key, value=value):
                self.refused(["F12"], f"FAIL F12 {RULES} sources > alpha: dir and manifest must be {EXT}/alpha and "
                                      f"{self.MAN}", tweak={RULES: setter("sources", "alpha", key, value=value)})

    def test_source_name_that_is_not_a_folder_name(self):
        def rename(doc):
            doc["sources"]["../alpha"] = doc["sources"].pop("alpha")
        self.refused(["F12", "F12"], f"FAIL F12 {RULES}: source name \"../alpha\" is not a folder name",
                     "does not adopt: alpha", tweak={RULES: rename})

    def test_unsafe_or_forbidden_raw_path(self):
        for path in ("../beta/README.md", "/abs/file.parquet", "data\\part.parquet", "c:/x.parquet", "data/test.csv",
                     "data//part.parquet", "./README.md"):
            with self.subTest(path=path):
                def unsafe(doc, path=path):
                    doc["files"][0]["path"] = path
                fails = self.refused(["F12", "F04"], tweak={self.MAN: unsafe})
                self.assertEqual(fails[0], f"FAIL F12 {self.MAN} files[0]: path is not a safe relative path (or is "
                                           "named test.csv)")

    def test_files_named_like_the_test_file_are_refused_without_being_opened(self):
        """A download manifest that lists data/test.csv and a tracked tests/test.csv, both existing: the script
        (run with the real guard) must refuse them by name. An open would end the process with the guard's
        line and without the summary line."""
        def leaves(t):
            t[EXT + "/alpha/data/test.csv"] = b"synthetic external file\n"
            t["tests/test.csv"] = b"synthetic, tracked under tests/\n"
        self.make(leaves=leaves)
        self.assertIn("data/test.csv", [f["path"] for f in json.loads(self.t[self.MAN])["files"]])
        code, out = run_script(self.root, "--date", DATE)
        self.assertEqual(code, 1, out)
        self.assertEqual(out, [
            f"FAIL F12 {self.MAN} files[2]: path is not a safe relative path (or is named test.csv)",
            f"FAIL F04 {RULES} sources > alpha > files: path, size_bytes and sha256 differ from the files of "
            f"{self.MAN}",
            "FAIL F00 tests/test.csv: a file named test.csv is never opened",
            "refused: 3 error(s); nothing written", "guard installed: True"])

    def test_manifest_and_rules_keys(self):
        for path, fn, needle in (
                (RULES, remover("sources"), f"{RULES}: key sources is missing"),
                (RULES, setter("sources", value=[]), f"{RULES}: sources is not an object"),
                (RULES, remover("sources", "alpha", "manifest_sha256"),
                 f"{RULES} sources > alpha: key manifest_sha256 is missing"),
                (RULES, remover("sources", "alpha", "adopted_splits"),
                 f"{RULES} sources > alpha: key adopted_splits is missing"),
                (RULES, setter("sources", "alpha", "files", value=None),
                 f"{RULES} sources > alpha: files is not a list"),
                (RULES, setter("sources", "alpha", value="alpha"), f"{RULES} sources > alpha: key dir is missing")):
            with self.subTest(needle=needle):
                self.refused(["F10"], f"FAIL F10 {needle}", tweak={path: fn})

    def test_manifest_file_entry_keys(self):
        def no_size(doc):
            del doc["files"][1]["size_bytes"]

        def no_hash(doc):
            doc["files"][0]["sha256"] = "none"
        self.refused(["F10"], f"FAIL F10 {self.MAN} files[1]: key size_bytes is missing", tweak={self.MAN: no_size})
        self.refused(["F10"], f"FAIL F10 {self.MAN} files[0]: sha256 is not a SHA-256", tweak={self.MAN: no_hash})

    def test_external_scripts_are_recorded(self):
        self.refused(["F01"], "FAIL F01 scripts/external_lang.py: missing",
                     after=lambda t: t.pop("scripts/external_lang.py"))


# ---------------------------------------------------------------- --check (F09)

class TestCheck(Repo):
    def frozen(self, **tree):
        self.make(**tree)
        self.data = self.written()
        return self.data

    def test_check_right_after_the_build(self):
        self.frozen()
        code, out = self.check()
        self.assertEqual(code, 0, out)
        self.assertEqual(out, [f"OK --check: freeze/freeze-1.json equals the fresh rebuild; {RECORDED} files "
                               f"re-hashed, code_commit {self.head[:12]}, HEAD {self.head[:12]}"])
        self.assertEqual(self.out.read_bytes(), self.data)
        self.assertEqual(sorted(p.name for p in self.out.parent.iterdir()), ["freeze-1.json"])

    def test_check_after_the_freeze_file_is_committed_and_tagged(self):
        self.frozen()
        later = commit_all(self.root, "freeze 1")
        git(self.root, "tag", "freeze-1")
        code, out = self.check()
        self.assertEqual(code, 0, out)
        self.assertNotEqual(later, self.head)
        self.assertIn(f"code_commit {self.head[:12]}, HEAD {later[:12]}", out[0])
        self.assertEqual(json.loads(self.out.read_bytes())["code_commit"], self.head)

    def test_check_with_another_file(self):
        self.frozen()
        other = self.tmp / "copy.json"
        other.write_bytes(self.data)
        self.assertEqual(self.check("--out", str(other))[0], 0)
        self.out.unlink()
        fails = self.fails(self.check())
        self.assertEqual(fails, ["FAIL F09 --check: freeze/freeze-1.json cannot be read (FileNotFoundError); build "
                                 "it first with --date"])

    def test_drift_of_a_committed_file_is_named(self):
        self.frozen()
        put(self.root, "README.md", b"# spec\r\nline two\r\nline three\r\n")
        put(self.root, "scripts/new_tool.py", b"# added after the freeze\n")
        (self.root / "tests" / "test_synthetic.py").unlink()
        commit_all(self.root, "after the freeze")
        fails = self.fails(self.check())
        new = lf(b"# spec\nline two\nline three\n")
        self.assertEqual(fails, [
            "FAIL F09 --check: scripts > scripts/new_tool.py: only in the fresh rebuild",
            "FAIL F09 --check: scripts > tests/test_synthetic.py: only in the freeze file",
            f"FAIL F09 --check: spec > sha256: freeze file {SHA_README_LF[:12]} != fresh rebuild {new[:12]}"])
        self.assertEqual(self.out.read_bytes(), self.data)

    def test_frozen_state_verifies_on_a_checkout_of_the_tag(self):
        """After later commits the main work tree differs from the freeze (F09), a checkout of the tag does not."""
        self.frozen()
        commit_all(self.root, "freeze 1")
        git(self.root, "tag", "freeze-1")
        put(self.root, "README.md", b"# spec\r\nline two\r\nstatus: frozen\r\n")
        commit_all(self.root, "after the freeze")
        fails = self.fails(self.check())
        self.assertEqual([ln.split(":")[0] for ln in fails], ["FAIL F09 --check"])
        self.assertIn("spec > sha256", fails[0])
        at_tag = self.tmp / "at_tag"
        git(self.root, "worktree", "add", "-q", "--detach", str(at_tag), "freeze-1")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = M.main(["--stage", "1", "--check", "--root", str(at_tag)])
        self.assertEqual(code, 0, buf.getvalue())
        self.assertIn(f"code_commit {self.head[:12]}", buf.getvalue())

    def test_drift_of_a_checkpoint(self):
        self.frozen()
        ck = f"runs/{C42}/checkpoint/model.safetensors"
        put(self.root, ck, self.t[ck][:-1] + b"\x07")
        commit_all(self.root, "another checkpoint")
        fails = self.fails(self.check())
        self.assertEqual([ln.split()[1] for ln in fails], ["F04"])
        self.assertIn(f"{ck}: sha256 on disk", fails[0])

    def test_uncommitted_change_is_refused_by_the_precondition(self):
        self.frozen()
        put(self.root, "requirements.txt", b"torch==0.1\n")
        fails = self.fails(self.check())
        self.assertEqual([ln.split()[1] for ln in fails], ["F04"] * 12 + ["F02"])
        self.assertIn("FAIL F02 requirements.txt: modified in the working tree or index", fails[-1])

    def test_edited_freeze_file(self):
        self.frozen()
        doc = json.loads(self.data)
        doc["conditions"]["B"]["thresholds"]["42"] = 0.5
        doc["runs"][A42]["checkpoint"]["sha256"] = "ab" * 32
        doc["lists"]["test_clean"]["excluded_by_reason"]["train_overlap"] = 1
        doc["models"]["keys"].append("extra")
        doc["extra"] = {"note": 1}
        del doc["results"]["results/s4/summary.json"]
        self.out.write_bytes(render(doc))
        fails = self.fails(self.check())
        self.assertEqual(fails, [
            f"FAIL F09 --check: conditions > B > thresholds > 42: freeze file 0.5 != fresh rebuild "
            f"{threshold_of(-27)}",
            "FAIL F09 --check: extra: only in the freeze file",
            "FAIL F09 --check: lists > test_clean > excluded_by_reason > train_overlap: freeze file 1 != fresh "
            "rebuild 0",
            "FAIL F09 --check: models > keys: freeze file [\"bert\", \"deberta\", \"roberta\", \"extra\"] != fresh "
            "rebuild [\"bert\", \"deberta\", \"roberta\"]",
            "FAIL F09 --check: results > results/s4/summary.json: only in the fresh rebuild",
            f"FAIL F09 --check: runs > {A42} > checkpoint > sha256: freeze file abababababab != fresh rebuild "
            f"{sha(checkpoint_bytes(A42))[:12]}"])

    def test_date_is_taken_from_the_file(self):
        self.make()
        code, out = self.main("--stage", "1", "--date", "2026-12-01")
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads(self.out.read_bytes()), expected(self.t, self.head, "2026-12-01"))
        self.assertEqual(self.check()[0], 0)

    def test_formatting_difference(self):
        self.frozen()
        self.out.write_bytes(json.dumps(json.loads(self.data), indent=1, sort_keys=True).encode("ascii") + b"\n")
        fails = self.fails(self.check())
        self.assertEqual(fails, ["FAIL F09 --check: the entries are equal but the bytes differ (formatting)"])

    def test_true_is_not_one(self):
        self.frozen()
        doc = json.loads(self.data)
        doc["schema"] = True
        self.out.write_bytes(render(doc))
        fails = self.fails(self.check())
        self.assertEqual(fails, ["FAIL F09 --check: schema: freeze file true != fresh rebuild 1"])

    def test_many_differences_are_bounded(self):
        self.frozen()
        doc = json.loads(self.data)
        for run in doc["runs"].values():
            run["selected_step"] = 10
            run["seed"] += 1
        self.out.write_bytes(render(doc))
        fails = self.fails(self.check())
        self.assertEqual(len(fails), 21)
        self.assertEqual(fails[-1], "FAIL F09 --check: 4 more difference(s)")

    def test_not_a_freeze_record(self):
        self.frozen()
        doc = json.loads(self.data)
        for data in (b"", b"{}\n", b"[1]\n", b"\xff\n", b"{\"freeze\": 1}\n",
                     render({**doc, "freeze": {**doc["freeze"], "stage": 2}}),
                     render({**doc, "freeze": {**doc["freeze"], "tag": "freeze-2"}}),
                     render({**doc, "freeze": {**doc["freeze"], "date": "2026-02-30"}}),
                     render({**doc, "code_commit": self.head[:12]}), render({**doc, "code_commit": None}),
                     render({k: v for k, v in doc.items() if k != "code_commit"})):
            with self.subTest(data=data[:30]):
                self.out.write_bytes(data)
                fails = self.fails(self.check())
                self.assertEqual(fails, ["FAIL F09 --check: freeze/freeze-1.json is not a stage-1 freeze record "
                                         "(JSON with freeze stage, tag, date and code_commit)"])

    def test_code_commit_that_is_not_in_the_repository(self):
        self.frozen()
        doc = json.loads(self.data)
        doc["code_commit"] = "0123456789" * 4
        self.out.write_bytes(render(doc))
        fails = self.fails(self.check())
        self.assertEqual(fails, ["FAIL F09 --check: code_commit 012345678901 is not a commit of this repository"])

    def test_code_commit_that_is_not_an_ancestor_of_head(self):
        self.frozen()
        git(self.root, "commit", "-q", "--amend", "-m", "rewritten history")
        new = git(self.root, "rev-parse", "HEAD")
        self.assertNotEqual(new, self.head)
        fails = self.fails(self.check())
        self.assertEqual(fails, [f"FAIL F09 --check: code_commit {self.head[:12]} is not an ancestor of HEAD "
                                 f"{new[:12]}"])

    def test_code_commit_must_be_a_commit_object(self):
        self.frozen()
        tree = git(self.root, "rev-parse", "HEAD^{tree}")
        doc = json.loads(self.data)
        doc["code_commit"] = tree
        self.out.write_bytes(render(doc))
        fails = self.fails(self.check())
        self.assertEqual(fails, [f"FAIL F09 --check: code_commit {tree[:12]} is not a commit of this repository"])

    def test_check_never_writes(self):
        self.frozen()
        put(self.root, "freeze/freeze-1.json", self.data.replace(b"freeze-1", b"freeze-9", 1))
        before = {p.relative_to(self.root).as_posix(): p.read_bytes() for p in self.root.rglob("*")
                  if p.is_file() and ".git" not in p.parts and p.name.lower() != "test.csv"}
        self.fails(self.check())
        after = {p.relative_to(self.root).as_posix(): p.read_bytes() for p in self.root.rglob("*")
                 if p.is_file() and ".git" not in p.parts and p.name.lower() != "test.csv"}
        self.assertEqual(before, after)


# ---------------------------------------------------------------- small helpers

class TestHelpers(unittest.TestCase):
    def test_safe_rel(self):
        for good in ("a", "a/b.c", "data/x-1_y.parquet"):
            self.assertTrue(M._safe_rel(good), good)
        for bad in ("", "/a", "a/", "a//b", "../a", "a/../b", "./a", "a\\b", "c:/a", None, 3, ["a"]):
            self.assertFalse(M._safe_rel(bad), bad)

    def test_is_test_csv(self):
        for name in ("test.csv", "TEST.CSV", "a/b/test.csv", "a\\Test.csv", Path("x") / "test.csv"):
            self.assertTrue(M._is_test_csv(name), name)
        for name in ("test.csv.bak", "mytest.csv", "test_csv", "test.csv/x"):
            self.assertFalse(M._is_test_csv(name), name)

    def test_float32(self):
        self.assertTrue(M._is_float32(0.5))
        self.assertTrue(M._is_float32(float(np.float32(0.1))))
        self.assertFalse(M._is_float32(0.1))

    def test_expected_runs(self):
        self.assertEqual(M.expected_runs(SEEDS), list(RUNS))
        self.assertEqual(M.expected_runs([7]), [("s2-bert-A-s7", "s2", "bert", "A", 7),
                                                ("s2-deberta-A-s7", "s2", "deberta", "A", 7),
                                                ("s2-roberta-A-s7", "s2", "roberta", "A", 7),
                                                ("s4-roberta-C-s7", "s4", "roberta", "C", 7)])

    def test_diff(self):
        self.assertEqual(M.diff({"a": 1, "b": [1, {"c": 2}]}, {"a": 1, "b": [1, {"c": 2}]}), [])
        self.assertEqual(M.diff({"a": 1, "b": [1, {"c": 2}], "d": 0}, {"a": 2, "b": [1, {"c": 3}], "e": 0}), [
            "a: freeze file 1 != fresh rebuild 2", "b[1] > c: freeze file 2 != fresh rebuild 3",
            "d: only in the freeze file", "e: only in the fresh rebuild"])
        self.assertEqual(M.diff({"a": [1]}, {"a": [1, 2]}), ["a: freeze file [1] != fresh rebuild [1, 2]"])
        self.assertEqual(M.diff({"a": 1}, {"a": 1.0}), ["a: freeze file 1 != fresh rebuild 1.0"])
        self.assertEqual(M.diff({"a": {"b": 1}}, {"a": [1]}), ["a: freeze file {\"b\": 1} != fresh rebuild [1]"])

    def test_constants_follow_the_scripts_they_mirror(self):
        self.assertEqual(M.LIST_FILES, data_lists.LIST_FILES)
        self.assertEqual(M.ROOT, ROOT)
        self.assertEqual((M.STAGE, M.TAG, M.OUT_REL), (1, "freeze-1", "freeze/freeze-1.json"))


if __name__ == "__main__":
    unittest.main()
