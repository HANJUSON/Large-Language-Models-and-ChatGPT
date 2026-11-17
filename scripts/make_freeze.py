"""Freeze 1 record -> freeze/freeze-1.json (README 3.2 "test use procedure" step 1, 7.1 freeze records; design D10).

Run from Git Bash at the repo root, after everything the freeze records has been committed:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/make_freeze.py --stage 1 --date 2026-11-17
        -> freeze/freeze-1.json (then: commit it, tag the commit freeze-1, push; README 7.1)
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/make_freeze.py --stage 1 --check
        -> rebuilds the record in memory with the date and code_commit of the existing file, re-hashes every
           file on disk and compares the bytes; nothing is written
Options: --out PATH (the freeze file to write or to check; default <root>/freeze/freeze-1.json) and
--root DIR (repository root; tests and dry runs).

Standard library plus scripts/data_lists.py (test.csv guard, SHA-256 helpers). main() installs
data_lists.install_test_guard() first: this process never opens a test.csv path. The three dataset csv files
are not opened at all; their SHA-256 and size are copied from the dataset download manifest, and only their
size is compared with the file on disk (os.stat). No sentence text is read into the record or printed:
the record holds paths, hashes, sizes, counts, IDs, seeds and thresholds only.

Rules:
  - Hash rule. Binary files (runs/*/checkpoint/model.safetensors, *.parquet) and every raw file listed in a
    data/external download manifest: SHA-256 of the bytes, streamed (the manifest records the byte hash and
    data/external/** is -text in .gitattributes). Every other file: SHA-256 of the bytes with CRLF replaced
    by LF (the rule of data_lists.sha256_lf), so a checkout with other line endings verifies. Each file is
    read once; the bytes that are hashed are the bytes that are parsed.
  - Precondition. The root is the top level of a git work tree with a HEAD commit; every file the record
    names exists, is tracked by git and is unmodified in the working tree and in the index (git status);
    no untracked, non-ignored file lies under scripts/ or tests/. code_commit = `git rev-parse HEAD` at
    build time: the commit that holds every frozen file. The freeze file itself is committed afterwards.
    git is only read (ls-files, status with --no-optional-locks, rev-parse, cat-file, merge-base).
  - Checkpoints are Git LFS files in the work tree: a checkpoint smaller than POINTER_LIMIT bytes is an LFS
    pointer and is refused; otherwise its size and SHA-256 must equal run.json.
  - Content (top-level keys of the record):
    freeze        stage 1, tag freeze-1, date, readme
    hash_rule     the hash rule above, in words
    code_commit   HEAD at build time
    spec          README.md
    dataset       datasets/s-labs-prompt-injection/download_manifest.json: its hash, dataset name, revision,
                  and path, SHA-256 and size of train / validation / test csv copied from the manifest
    lists         data/processed/: the six list CSVs, similar_pairs.csv, test_clean.csv, test_excluded.csv
                  and train_review.csv with hash and rows, lists_meta.json with its hash; lists_meta must say
                  "test_clean_stage": "final" and "trial": false, its outputs / review / script / dataset
                  revision must equal the files, and its augmentation record must name the frozen
                  train_aug.csv and cases.csv; test_clean rows and the excluded rows by reason
                  (train_overlap, augmentation_overlap, validation_overlap) counted from test_excluded.csv
    config        configs/train.json            environment   requirements.txt
    models        the three model keys, the representative model (roberta) and the seeds of the config
    runs          s2-<model>-A-s<seed> (nine) and s4-roberta-C-s<seed> (three), keyed by run ID: run_id,
                  model_key, stage, condition, seed, selected_step, checkpoint {path, size, sha256} (the
                  run.json values, verified against the file) and the hashes of run.json, metrics.json and
                  predictions_validation.csv. run.json must be the completed run of that ID with a kept
                  checkpoint of the selected step, and its config, requirements and list-file SHA-256
                  must equal the frozen files
    conditions    A and C: checkpoint pattern and threshold 0.5 (configs/train.json); B and D: checkpoint
                  pattern and the per-seed thresholds
    thresholds    results/s4/thresholds.json: hash and per seed and condition B | D run_id, logit,
                  threshold, cap_met. Reader convention of eval_abcd.read_thresholds: an entry is any JSON
                  object holding "threshold", "logit" and "run_id"; s2-roberta-A-s<seed> is condition B,
                  s4-roberta-C-s<seed> condition D; exactly one entry per seed and condition; threshold a
                  float32 value in (0, 1) equal to sigmoid(logit) within 1e-6, logit a candidate k/20 in
                  -8..8, cap_met a boolean; the optional keys seed, condition, selected_step and
                  checkpoint_sha256 must agree with the frozen run; the document must record, in members
                  named runs/<id>/<file>, the SHA-256 of run.json and predictions_validation.csv of every
                  source run, equal to the frozen files (metrics.json when recorded)
    augmentation  data/cases/train/: originals.csv, cases.csv, review.csv, train_aug.csv, cases_meta.json,
                  test_overlap.json, prompts/*.md, llm/*.json; rows and label counts of train_aug.csv;
                  cases_meta.json must say ready_for_training true and its outputs, script and generators
                  hashes must equal the files; every C run's data.augmentation.files_sha256 must equal the
                  frozen cases.csv, review.csv and train_aug.csv; for cases_meta.json and test_overlap.json
                  the value of each run is recorded next to the frozen one and a difference is listed
    final_cases   data/cases/final/: RULES.md, final_spec.json, prompts/*.md; generator code
                  scripts/build_dev_cases.py (must define ROB01, assemble and plan_rob03); extraction seed,
                  targets and LLM model copied from final_spec.json, whose prompts_sha256_lf must name
                  exactly the prompt files with their hashes
    dev_cases     data/cases/dev/: originals.csv, cases.csv, review.csv, cases_meta.json
    external      data/external/external_rules.json, scripts/external_lang.py, scripts/fetch_external.py and
                  per adopted source (the "sources" of the rules, which must be exactly the folders that
                  hold a download_manifest.json): dataset, revision, adopted splits, the manifest and every
                  raw file of the manifest with size and SHA-256 verified on disk; rules and manifest must
                  agree on dataset, revision, manifest hash and files
    results       results/s2/summary.json, results/s3/summary.json, results/s3/error_summary.json,
                  results/s4/thresholds.json, results/s4/summary.json, results/s4/dev_cases_vs_test.json
    scripts       every git-tracked file under scripts/ and tests/
  - Output: UTF-8 JSON, ASCII, LF, indent 2, sorted keys, repo-relative POSIX paths, no timestamps: the
    same commit, date and files give the same bytes. Written to <name>.tmp, then os.replace. An existing
    freeze file is never overwritten.
  - --check: date and code_commit come from the existing file (CRLF in it is read as LF); code_commit must
    be a commit of the repository and an ancestor of HEAD; every check above runs again on the files as
    they are now; the differing entries are named. A recorded file that changed in a later commit (also
    README.md, or a file added under scripts/ or tests/) is a difference: to verify the frozen state
    itself, run --check with --root on a checkout of the freeze-1 tag.

Refused (exit 1, `FAIL <code> ...` lines, then `refused: N error(s); nothing written`):
  F00 arguments and output: --stage is not 1; --date missing, not a YYYY-MM-DD date, or given with --check;
      the output path is inside datasets/, named test.csv, a folder, or (without --check) exists already;
      a file to hash is named test.csv
  F01 a file or folder that the freeze records is missing or unreadable, or a pattern matches no file
  F02 git: not available, the root is not the top level of a work tree, no HEAD commit; a recorded file is
      not tracked or is modified in the working tree or index; an untracked file under scripts/ or tests/
  F03 a checkpoint is smaller than POINTER_LIMIT bytes (a Git LFS pointer, not the weights)
  F04 hash mismatch: a SHA-256, size or dataset revision recorded in a frozen record differs from the file
      on disk (run.json: checkpoint, config, requirements, list files; lists_meta.json: outputs, review,
      script, dataset revision; dataset manifest: file size; cases_meta.json: outputs, script, generators;
      external rules and manifests: manifest hash, raw file size and SHA-256, file lists)
  F05 lists not final: lists_meta.json test_clean_stage is not "final" or trial is not false, or
      test_clean.csv / test_excluded.csv have another header or an unknown reason
  F06 augmentation not ready or not the frozen one: cases_meta.json ready_for_training is not true;
      train_aug.csv is malformed or empty; a C run or lists_meta.json records another cases.csv,
      review.csv or train_aug.csv than the frozen file
  F07 thresholds inconsistent (see thresholds above), or configs/train.json threshold is not 0.5
  F08 final spec inconsistent: prompts_sha256_lf differs from the prompt files, or the generator code
      does not define ROB01, assemble or plan_rob03
  F09 --check: the freeze file is missing, unreadable or not a stage-1 record; its code_commit is not a
      commit or not an ancestor of HEAD; the fresh rebuild differs from the file
  F10 a record is unreadable: not UTF-8 JSON, not a JSON object, a required key missing or of the wrong
      type, or a csv file without a final LF
  F11 run record inconsistent: runs/<id>.lock exists; run ID, stage, model, condition or seed differ from
      the folder name; status not completed; checkpoint not kept, its path or step not the expected one
  F12 external records inconsistent: an unsafe path; a source folder or manifest path that is not
      data/external/<source>/download_manifest.json; rules and manifest differ in dataset or revision; a
      download manifest on disk whose source is not in the rules
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import stat
import struct
import subprocess
import sys
from pathlib import Path

import data_lists

ROOT = data_lists.ROOT
SCRIPT_REL = "scripts/make_freeze.py"
STAGE = 1
TAG = "freeze-1"
OUT_REL = "freeze/freeze-1.json"
SCHEMA = 1
POINTER_LIMIT = 1 << 20  # bytes; a checkpoint below this is a Git LFS pointer (tests lower it)
CHUNK = 1 << 20
BINARY_SUFFIXES = (".safetensors", ".parquet")
SHA_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
SHOWN_DIFFS = 20

README_REL = "README.md"
CONFIG_REL = "configs/train.json"
REQUIREMENTS_REL = "requirements.txt"
DATASET_DIR = "datasets/s-labs-prompt-injection"
DATASET_MANIFEST = f"{DATASET_DIR}/download_manifest.json"
SPLITS = ("train", "validation", "test")

LISTS_DIR = "data/processed"
LISTS_META = "lists_meta.json"
REVIEW = "train_review.csv"
LIST_FILES = ("train_clean.csv", "train_excluded.csv", "validation_clean.csv", "validation_excluded.csv",
              "test_clean_preaug.csv", "test_excluded_preaug.csv")  # data_lists.LIST_FILES: run.json lists
TEST_CLEAN, TEST_EXCLUDED = "test_clean.csv", "test_excluded.csv"
BUILD_OUTPUTS = LIST_FILES + ("similar_pairs.csv", TEST_CLEAN, TEST_EXCLUDED)  # lists_meta.json "outputs"
TEST_CLEAN_HEADER, TEST_EXCLUDED_HEADER = b"sentence_id", b"sentence_id,reason"
TEST_REASONS = ("train_overlap", "augmentation_overlap", "validation_overlap")
BUILD_LISTS = "scripts/build_lists.py"

MODEL_KEYS = ("bert", "deberta", "roberta")
REPRESENTATIVE = "roberta"  # README 5.1 EXP-04
DEFAULT_THRESHOLD = 0.5  # conditions A and C
RUN_FILES = ("run.json", "metrics.json", "predictions_validation.csv")
CHECKPOINT_FILE = "model.safetensors"
TUNED = (("B", "s2", "A"), ("D", "s4", "C"))  # tuned condition, stage and condition of its source run

THRESHOLDS_REL = "results/s4/thresholds.json"
THRESHOLD_KEYS = ("threshold", "logit", "run_id")  # eval_abcd.THRESHOLD_KEYS
PINNED_RUN_FILES = ("run.json", "predictions_validation.csv")  # eval_abcd.PINNED_RUN_FILES
LOGIT_MAX, LOGIT_STEPS, SIGMOID_TOL = 8.0, 20, 1e-6  # as in eval_abcd

AUG_DIR = "data/cases/train"
AUG_LIST, AUG_CASES, AUG_META = "train_aug.csv", "cases.csv", "cases_meta.json"
AUG_FILES = ("originals.csv", AUG_CASES, "review.csv", AUG_LIST, AUG_META, "test_overlap.json")
AUG_OUTPUTS = ("originals.csv", AUG_CASES, "review.csv", AUG_LIST)  # cases_meta.json "outputs"
AUG_PINNED = (AUG_CASES, "review.csv", AUG_LIST)  # a C run must have trained on exactly these files
AUG_MAY_DIFFER = (AUG_META, "test_overlap.json")  # recorded per run next to the frozen value
AUG_LIST_HEADER = b"sentence_id,label,text_sha256_12"
BUILD_TRAIN_AUG = "scripts/build_train_aug.py"

FINAL_DIR = "data/cases/final"
FINAL_FILES = ("RULES.md", "final_spec.json")
GENERATOR_REL = "scripts/build_dev_cases.py"
GENERATOR_DEFS = (("ROB01", rb"^ROB01 = "), ("assemble", rb"^def assemble\("), ("plan_rob03", rb"^def plan_rob03\("))
DEV_DIR = "data/cases/dev"
DEV_FILES = ("originals.csv", "cases.csv", "review.csv", "cases_meta.json")

EXT_DIR = "data/external"
EXT_RULES = f"{EXT_DIR}/external_rules.json"
EXT_MANIFEST = "download_manifest.json"
EXT_SCRIPTS = ("scripts/external_lang.py", "scripts/fetch_external.py")
RESULT_FILES = ("results/s2/summary.json", "results/s3/summary.json", "results/s3/error_summary.json",
                THRESHOLDS_REL, "results/s4/summary.json", "results/s4/dev_cases_vs_test.json")
CODE_DIRS = ("scripts", "tests")

HASH_RULE = {
    "bytes": "SHA-256 of the file bytes: runs/*/checkpoint/model.safetensors, *.parquet and every raw file "
             "listed in a data/external download manifest (size and SHA-256 as the manifest records them, "
             "verified on disk)",
    "lf": "every other file: SHA-256 of the bytes with CRLF replaced by LF (scripts/data_lists.py sha256_lf), "
          "so a checkout with other line endings gives the same value",
    "dataset": "the csv files under datasets/ are not opened: sha256 and size_bytes are copied from the "
               "download manifest and the size is compared with the file on disk",
}
KINDS = {
    "any": ("present", lambda v: True),
    "str": ("a non-empty string", lambda v: isinstance(v, str) and v != ""),
    "int": ("an integer", lambda v: type(v) is int),
    "dict": ("an object", lambda v: isinstance(v, dict)),
    "list": ("a list", lambda v: isinstance(v, list)),
    "sha": ("a SHA-256 (64 lowercase hex characters)", lambda v: isinstance(v, str) and bool(SHA_RE.fullmatch(v))),
}


class _Stop(Exception):
    """Leaves one section of the record after its error has been recorded."""


def _ascii(s):
    return str(s).encode("ascii", "backslashreplace").decode("ascii")


def _short(v):
    """A recorded value for a message: hashes cut to 12 characters, anything else to 40."""
    if isinstance(v, str) and SHA_RE.fullmatch(v):
        return v[:12]
    return _ascii(json.dumps(v))[:40]


def _safe_rel(path):
    """True for a relative POSIX path without empty, '.', '..' parts, backslashes or a drive."""
    return (isinstance(path, str) and path != "" and "\\" not in path and ":" not in path
            and all(part not in ("", ".", "..") for part in path.split("/")))


def _is_test_csv(name):
    return str(name).replace("\\", "/").rsplit("/", 1)[-1].lower() == "test.csv"


def sha256_stream(path):
    """(SHA-256, size) of the file bytes, read in chunks."""
    h, size = hashlib.sha256(), 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _dump(doc):
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


# ---------------------------------------------------------------- git (read only)

def git(root, *args):
    return subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args], capture_output=True)


def git_state(root):
    """(errors, state). state: head, tracked (set of paths), dirty {path: XY}, untracked_code [paths]."""
    try:
        r = git(root, "rev-parse", "--show-prefix")
    except OSError:
        return ["F02 git is not available (the freeze needs the git state of every recorded file)"], None
    if r.returncode != 0:
        return ["F02 git: the root is not inside a git work tree (git rev-parse failed)"], None
    if r.stdout.strip():
        return ["F02 git: the root is not the top level of the work tree"], None
    r = git(root, "rev-parse", "--verify", "HEAD^{commit}")
    head = r.stdout.decode("ascii", "replace").strip()
    if r.returncode != 0 or not COMMIT_RE.fullmatch(head):
        return ["F02 git: no HEAD commit (git rev-parse HEAD failed)"], None
    listed = git(root, "ls-files", "-z")
    status = git(root, "status", "--porcelain=v1", "-z", "--untracked-files=no")
    others = git(root, "ls-files", "-z", "--others", "--exclude-standard", "--", *CODE_DIRS)
    if listed.returncode != 0 or status.returncode != 0 or others.returncode != 0:
        return ["F02 git: ls-files or status failed"], None
    dirty, parts, i = {}, status.stdout.split(b"\0"), 0
    while i < len(parts):
        entry = parts[i]
        i += 1
        if not entry:
            continue
        xy, path = entry[:2].decode("ascii", "replace"), entry[3:].decode("utf-8", "replace")
        if xy[0] in "RC":
            i += 1  # the original path of a rename or copy follows
        dirty[path] = xy
    tracked, untracked = ([p.decode("utf-8", "replace") for p in r.stdout.split(b"\0") if p]
                          for r in (listed, others))
    return [], {"head": head, "tracked": set(tracked), "dirty": dirty, "untracked_code": sorted(untracked)}


def git_errors(recorded, state):
    """F02 lines for the recorded files (repo-relative paths) and for untracked code files."""
    errors = []
    for rel in sorted(recorded):
        if rel not in state["tracked"]:
            errors.append(f"F02 {rel}: not tracked by git")
        elif rel in state["dirty"]:
            errors.append(f"F02 {rel}: modified in the working tree or index (git status "
                          f"{state['dirty'][rel]!r})")
    for rel in state["untracked_code"]:
        if rel not in recorded:  # a recorded one has its line above
            errors.append(f"F02 {rel}: untracked file under scripts/ or tests/ (every file there is part of "
                          "the freeze: commit it or remove it)")
    return errors


# ---------------------------------------------------------------- files and records

class Freeze:
    """Reads, hashes and remembers the files of one record; errors are collected, never raised to main()."""

    def __init__(self, root):
        self.root = Path(root)
        self.errors = []
        self.recorded = {}  # repo-relative path -> "lf" | "bytes" | "manifest" (existing files only)
        self._sha = {}
        self._text = {}

    def fail(self, code, msg):
        self.errors.append(f"{code} {msg}")

    def path(self, rel):
        return self.root.joinpath(*rel.split("/"))

    def sha(self, rel, rule=None):
        """SHA-256 of a recorded file under the hash rule; None after its F00/F01 line (printed once)."""
        if rel in self._sha:
            return self._sha[rel]
        rule = rule or ("bytes" if rel.lower().endswith(BINARY_SUFFIXES) else "lf")
        value, p = None, self.path(rel)
        if _is_test_csv(rel):
            self.fail("F00", f"{rel}: a file named test.csv is never opened")
        elif not p.is_file():
            self.fail("F01", f"{rel}: missing")
        else:
            self.recorded[rel] = rule
            try:
                if rule == "bytes":
                    value, _ = sha256_stream(p)
                else:
                    data = p.read_bytes().replace(b"\r\n", b"\n")
                    value, self._text[rel] = data_lists.sha256_bytes(data), data
            except OSError as exc:
                self.fail("F01", f"{rel}: cannot read ({type(exc).__name__})")
        self._sha[rel] = value
        return value

    def text(self, rel):
        """LF-normalised bytes of a text file (the bytes that were hashed); None when it could not be read."""
        self.sha(rel, "lf")
        return self._text.get(rel)

    def load_json(self, rel):
        data = self.text(rel)
        if data is None:
            raise _Stop
        try:
            doc = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            self.fail("F10", f"{rel}: not UTF-8 JSON")
            raise _Stop from None
        if not isinstance(doc, dict):
            self.fail("F10", f"{rel}: not a JSON object")
            raise _Stop
        return doc

    def need(self, doc, rel, keys, kind):
        """doc[k1][k2]... of the given kind; F10 and _Stop otherwise."""
        value, shown = doc, " > ".join(keys)
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                self.fail("F10", f"{rel}: key {shown} is missing")
                raise _Stop
            value = value[k]
        name, ok = KINDS[kind]
        if not ok(value):
            self.fail("F10", f"{rel}: {shown} is not {name}")
            raise _Stop
        return value

    def same(self, code, what, got, want, note=""):
        """Compare a recorded value with the frozen one; a None side was reported before."""
        if got is not None and want is not None and got != want:
            self.fail(code, f"{what}: {_short(got)} != {_short(want)}" + (f" ({note})" if note else ""))

    def stat_size(self, rel):
        """Size of a file that is recorded without being opened (os.stat only); None after F01."""
        try:
            st = os.stat(self.path(rel))
        except OSError:
            self.fail("F01", f"{rel}: missing")
            return None
        if not stat.S_ISREG(st.st_mode):
            self.fail("F01", f"{rel}: not a regular file")
            return None
        self.recorded[rel] = "manifest"
        return st.st_size

    def glob(self, rel_dir, suffix):
        """Sorted paths of the plain files <rel_dir>/*<suffix> (not recursive); F01 when there is none."""
        d = self.path(rel_dir)
        try:
            names = sorted(n for n in os.listdir(d) if n.endswith(suffix) and (d / n).is_file())
        except OSError:
            names = []
        if not names:
            self.fail("F01", f"{rel_dir}/*{suffix}: no file")
        return [f"{rel_dir}/{n}" for n in names]

    def csv_rows(self, rel):
        """Data rows of a csv file (lines after the header); None when unreadable, F10 without a final LF."""
        data = self.text(rel)
        if data is None:
            return None
        if not data.endswith(b"\n"):
            self.fail("F10", f"{rel}: no final LF")
            return None
        return data.count(b"\n") - 1


def section(fz, fn, *args):
    """Result of one section, or None when it stopped (its errors are in fz.errors)."""
    try:
        return fn(fz, *args)
    except _Stop:
        return None


# ---------------------------------------------------------------- sections

def config_block(fz):
    """(block, seeds): configs/train.json with the seeds every run list is built from."""
    cfg = fz.load_json(CONFIG_REL)
    seeds = fz.need(cfg, CONFIG_REL, ("seeds",), "list")
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        fz.fail("F10", f"{CONFIG_REL}: seeds is not a non-empty list of distinct integers")
        raise _Stop
    thr = fz.need(cfg, CONFIG_REL, ("threshold",), "any")
    if type(thr) is not float or thr != DEFAULT_THRESHOLD:
        fz.fail("F07", f"{CONFIG_REL}: threshold is {_short(thr)}, conditions A and C need {DEFAULT_THRESHOLD}")
    return {"path": CONFIG_REL, "sha256": fz.sha(CONFIG_REL)}, seeds


def dataset_block(fz):
    m = fz.load_json(DATASET_MANIFEST)
    block = {"manifest": {"path": DATASET_MANIFEST, "sha256": fz.sha(DATASET_MANIFEST)},
             "name": fz.need(m, DATASET_MANIFEST, ("dataset",), "str"),
             "revision": fz.need(m, DATASET_MANIFEST, ("revision",), "str"), "files": {}}
    entries = fz.need(m, DATASET_MANIFEST, ("files",), "list")
    by_path = {f["path"]: f for f in entries if isinstance(f, dict) and isinstance(f.get("path"), str)}
    for split in SPLITS:
        name = fz.need(m, DATASET_MANIFEST, ("splits", split, "file"), "str")
        if name not in by_path or not _safe_rel(name):
            fz.fail("F10", f"{DATASET_MANIFEST}: files has no safe entry for {_short(name)} (split {split})")
            raise _Stop
        tag = f"{DATASET_MANIFEST} files[{name}]"
        sha = fz.need(by_path[name], tag, ("sha256",), "sha")
        size = fz.need(by_path[name], tag, ("size_bytes",), "int")
        rel = f"{DATASET_DIR}/{name}"
        fz.same("F04", f"{rel}: size on disk", fz.stat_size(rel), size, "the manifest's size_bytes")
        block["files"][split] = {"path": rel, "sha256": sha, "size_bytes": size}
    return block


def lists_block(fz, dataset):
    files = {}
    for name in BUILD_OUTPUTS + (REVIEW,):
        rel = f"{LISTS_DIR}/{name}"
        files[name] = {"sha256": fz.sha(rel), "rows": fz.csv_rows(rel)}
    meta_rel = f"{LISTS_DIR}/{LISTS_META}"
    files[LISTS_META] = {"sha256": fz.sha(meta_rel)}
    block = {"dir": LISTS_DIR, "files": files}

    clean_rel, excl_rel = f"{LISTS_DIR}/{TEST_CLEAN}", f"{LISTS_DIR}/{TEST_EXCLUDED}"
    clean, excluded = fz.text(clean_rel), fz.text(excl_rel)
    if clean is not None and clean.split(b"\n")[0] != TEST_CLEAN_HEADER:
        fz.fail("F05", f"{clean_rel}: the header is not {TEST_CLEAN_HEADER.decode()}")
    if excluded is not None:
        lines = excluded.split(b"\n")
        reasons = [ln.split(b",")[-1].decode("ascii", "replace") for ln in lines[1:] if ln]
        if lines[0] != TEST_EXCLUDED_HEADER:
            fz.fail("F05", f"{excl_rel}: the header is not {TEST_EXCLUDED_HEADER.decode()}")
        elif any(ln.count(b",") != 1 for ln in lines[1:] if ln) or set(reasons) - set(TEST_REASONS):
            fz.fail("F05", f"{excl_rel}: a row is not sentence_id,reason with a reason of "
                           f"{', '.join(TEST_REASONS)}")
        else:
            block["test_clean"] = {"rows": files[TEST_CLEAN]["rows"], "excluded_rows": files[TEST_EXCLUDED]["rows"],
                                   "excluded_by_reason": {k: reasons.count(k) for k in TEST_REASONS}}

    meta = fz.load_json(meta_rel)
    stage = fz.need(meta, meta_rel, ("test_clean_stage",), "any")
    trial = fz.need(meta, meta_rel, ("trial",), "any")
    if stage != "final":
        fz.fail("F05", f"{meta_rel}: test_clean_stage is {_short(stage)}, freeze 1 needs \"final\" (run "
                       f"{BUILD_LISTS} with the frozen augmentation)")
    if trial is not False:
        fz.fail("F05", f"{meta_rel}: trial is {_short(trial)}, freeze 1 needs false")
    block["test_clean_stage"], block["trial"] = stage, trial
    for name in BUILD_OUTPUTS:
        if stage != "final" and name in (TEST_CLEAN, TEST_EXCLUDED):
            continue  # the pre-augmentation build does not write them; F05 above says so
        fz.same("F04", f"{meta_rel} outputs > {name} > sha256",
                fz.need(meta, meta_rel, ("outputs", name, "sha256"), "sha"), files[name]["sha256"],
                f"{LISTS_DIR}/{name} on disk")
    fz.same("F04", f"{meta_rel} review > sha256", fz.need(meta, meta_rel, ("review", "sha256"), "sha"),
            files[REVIEW]["sha256"], f"{LISTS_DIR}/{REVIEW} on disk")
    fz.same("F04", f"{meta_rel} script > sha256_lf", fz.need(meta, meta_rel, ("script", "sha256_lf"), "sha"),
            fz.sha(BUILD_LISTS), f"{BUILD_LISTS} on disk")
    fz.same("F04", f"{meta_rel} dataset > revision", fz.need(meta, meta_rel, ("dataset", "revision"), "str"),
            dataset["revision"] if dataset else None, f"{DATASET_MANIFEST} revision")
    if stage == "final":  # the final test_clean must have been built with the frozen augmentation
        aug = meta.get("augmentation")
        recorded = aug.get("files") if isinstance(aug, dict) else None
        if isinstance(aug, dict) and aug.get("dir") not in (None, AUG_DIR):
            fz.fail("F06", f"{meta_rel}: augmentation > dir is {_short(aug.get('dir'))}, not {AUG_DIR}")
        for name in (AUG_LIST, AUG_CASES):
            got = recorded.get(name) if isinstance(recorded, dict) else None
            got = got.get("sha256") if isinstance(got, dict) else got
            if not KINDS["sha"][1](got):
                fz.fail("F06", f"{meta_rel}: augmentation > files records no SHA-256 for {name} (the final "
                               "test_clean must name the augmentation it was built with)")
            else:
                fz.same("F06", f"{meta_rel} augmentation > files > {name}", got, fz.sha(f"{AUG_DIR}/{name}"),
                        f"the frozen {AUG_DIR}/{name}: test_clean was built with another augmentation")
    return block


def expected_runs(seeds):
    """[(run_id, stage, model, condition, seed)]: the nine stage-2 runs and the three stage-4 runs."""
    runs = [(f"s2-{m}-A-s{s}", "s2", m, "A", s) for m in MODEL_KEYS for s in seeds]
    return runs + [(f"s4-{REPRESENTATIVE}-C-s{s}", "s4", REPRESENTATIVE, "C", s) for s in seeds]


def checkpoint_size(fz, rel):
    """Size of a checkpoint file that is not an LFS pointer; None after F01 (missing) or F03 (pointer)."""
    size = fz.stat_size(rel)
    if size is None:
        return None
    fz.recorded[rel] = "bytes"
    if size < POINTER_LIMIT:
        fz.fail("F03", f"{rel}: {size} bytes, smaller than {POINTER_LIMIT}: a Git LFS pointer, not the "
                       "checkpoint (git lfs pull)")
        return None
    return size


def run_block(fz, ident, pins):
    """(block, augmentation files_sha256 of a C run or None) of one run; pins = the frozen config,
    requirements and list-file hashes that run.json must name."""
    run_id, stage, model, cond, seed = ident
    base = f"runs/{run_id}"
    if fz.path(f"runs/{run_id}.lock").exists():
        fz.fail("F11", f"runs/{run_id}.lock exists (the run is in progress, or was killed)")
    if not fz.path(base).is_dir():
        fz.fail("F01", f"{base}/: the run folder is missing")
        raise _Stop
    files = {n: fz.sha(f"{base}/{n}") for n in RUN_FILES}
    ck_rel = f"{base}/checkpoint/{CHECKPOINT_FILE}"
    on_disk = checkpoint_size(fz, ck_rel)
    rel = f"{base}/run.json"
    doc = fz.load_json(rel)
    want = {"run_id": run_id, "stage": stage, "model_key": model, "condition": cond, "seed": seed}
    bad = [k for k, v in want.items() if type(doc.get(k)) is not type(v) or doc.get(k) != v]
    if bad:
        fz.fail("F11", f"{rel}: {', '.join(bad)} do not match the run ID")
    if doc.get("status") != "completed":
        fz.fail("F11", f"{rel}: status is {_short(doc.get('status'))}, not \"completed\"")
    step = fz.need(doc, rel, ("selection", "selected_step"), "int")
    ck = fz.need(doc, rel, ("checkpoint",), "dict")
    sha = fz.need(doc, rel, ("checkpoint", "sha256"), "sha")
    size = fz.need(doc, rel, ("checkpoint", "size"), "int")
    if ck.get("kept") is not True or ck.get("path") != ck_rel or ck.get("step") != step:
        fz.fail("F11", f"{rel}: checkpoint is not kept, its path is not {ck_rel}, or its step "
                       f"{_short(ck.get('step'))} is not the selected step {step}")
    if on_disk is not None and on_disk != size:
        fz.fail("F04", f"{ck_rel}: size {on_disk} != run.json {size}")
    elif on_disk is not None:  # hashed only when the size is right
        fz.same("F04", f"{ck_rel}: sha256 on disk", fz.sha(ck_rel), sha, "run.json checkpoint > sha256")
    fz.same("F04", f"{rel} config > sha256", fz.need(doc, rel, ("config", "sha256"), "sha"), pins["config"],
            f"{CONFIG_REL} on disk")
    fz.same("F04", f"{rel} env > requirements_sha256", fz.need(doc, rel, ("env", "requirements_sha256"), "sha"),
            pins["requirements"], f"{REQUIREMENTS_REL} on disk")
    listed = fz.need(doc, rel, ("lists", "files_sha256"), "dict")
    for name in LIST_FILES:
        if not KINDS["sha"][1](listed.get(name)):
            fz.fail("F10", f"{rel}: lists > files_sha256 > {name} is not {KINDS['sha'][0]}")
        else:
            fz.same("F04", f"{rel} lists > files_sha256 > {name}", listed[name], pins["lists"].get(name),
                    f"{LISTS_DIR}/{name} on disk")
    aug = None
    if cond == "C":
        aug = {n: fz.need(doc, rel, ("data", "augmentation", "files_sha256", n), "sha")
               for n in AUG_PINNED + AUG_MAY_DIFFER}
    block = {"run_id": run_id, "model_key": model, "stage": stage, "condition": cond, "seed": seed,
             "selected_step": step, "checkpoint": {"path": ck_rel, "size": size, "sha256": sha}, "files": files}
    return block, aug


def _objects(node):
    """Every JSON object of a parsed document, depth first in document order (eval_abcd._objects)."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _objects(v)
    elif isinstance(node, list):
        for v in node:
            yield from _objects(v)


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_float32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0] == v


def thresholds_block(fz, seeds, runs):
    """(block, {(seed, condition): entry}); runs = {run_id: frozen run block or None}."""
    rel = THRESHOLDS_REL
    doc = fz.load_json(rel)
    expected = {f"{stage}-{REPRESENTATIVE}-{source}-s{s}": (s, cond) for s in seeds for cond, stage, source in TUNED}
    out = {}
    for o in _objects(doc):
        if not all(k in o for k in THRESHOLD_KEYS):
            continue
        rid, thr, logit = o["run_id"], o["threshold"], o["logit"]
        if not isinstance(rid, str) or rid not in expected:
            fz.fail("F07", f"{rel}: entry for an unexpected run {_short(rid)} (want {', '.join(sorted(expected))})")
            continue
        seed, cond = expected[rid]
        tag = f"{rel} {cond} s{seed}:"
        if (seed, cond) in out:
            fz.fail("F07", f"{tag} more than one entry for {rid}")
            continue
        cap_met = o.get("cap_met")
        out[(seed, cond)] = {"run_id": rid, "logit": logit, "threshold": thr, "cap_met": cap_met}
        if not isinstance(cap_met, bool):
            fz.fail("F07", f"{tag} cap_met must be true or false")
        if not (isinstance(thr, float) and 0.0 < thr < 1.0 and _is_float32(thr)):
            fz.fail("F07", f"{tag} threshold is not a float32 value in (0, 1) stored as a float")
        elif not (_number(logit) and abs(logit) <= LOGIT_MAX + SIGMOID_TOL
                  and abs(logit * LOGIT_STEPS - round(logit * LOGIT_STEPS)) <= SIGMOID_TOL):
            fz.fail("F07", f"{tag} logit is not a candidate k/{LOGIT_STEPS} within -{LOGIT_MAX:g}..{LOGIT_MAX:g}")
        elif abs(thr - 1.0 / (1.0 + math.exp(-logit))) > SIGMOID_TOL:
            fz.fail("F07", f"{tag} threshold is not sigmoid(logit) (float32)")
        run = runs.get(rid)
        for key, want in (("seed", seed), ("condition", cond),
                          ("selected_step", run["selected_step"] if run else None),
                          ("checkpoint_sha256", run["checkpoint"]["sha256"] if run else None)):
            if key in o and want is not None and (o[key] != want or isinstance(o[key], bool)):
                fz.fail("F07", f"{tag} {key} {_short(o[key])} disagrees with the frozen run {rid} ({_short(want)})")
    missing = sorted(rid for rid, key in expected.items() if key not in out)
    if missing:
        fz.fail("F07", f"{rel}: no entry (an object with the keys {', '.join(THRESHOLD_KEYS)}) for "
                       f"{', '.join(missing)}")
    if "model" in doc and doc["model"] != REPRESENTATIVE:
        fz.fail("F07", f"{rel}: model is {_short(doc['model'])}, not the representative model {REPRESENTATIVE}")
    pins = {f"runs/{rid}/{n}": runs[rid]["files"][n] for rid in sorted(expected) if runs.get(rid) for n in RUN_FILES}
    recorded = set()
    for o in _objects(doc):
        for k, v in o.items():
            if k in pins and isinstance(v, str):
                recorded.add(k)
                fz.same("F07", f"{rel}: the recorded SHA-256 of {k}", v, pins[k],
                        "the frozen file: the thresholds were selected on other run files")
    unpinned = sorted(k for k in pins if k.rsplit("/", 1)[-1] in PINNED_RUN_FILES and k not in recorded)
    if unpinned:
        fz.fail("F07", f"{rel}: no SHA-256 is recorded for {', '.join(unpinned)} (the thresholds must name the "
                       "run files they were selected on)")
    by_seed = {str(s): {cond: out[(s, cond)] for cond, _, _ in TUNED if (s, cond) in out} for s in seeds}
    return {"path": rel, "sha256": fz.sha(rel), "by_seed": by_seed}, out


def conditions_block(seeds, thresholds):
    block = {}
    for cond, stage, source in TUNED:
        pattern = f"{stage}-{REPRESENTATIVE}-{source}-s<seed>"
        block[source] = {"checkpoints": pattern, "threshold": DEFAULT_THRESHOLD}
        block[cond] = {"checkpoints": pattern,
                       "thresholds": {str(s): thresholds[(s, cond)]["threshold"] for s in seeds}}
    return block


def train_aug_counts(fz, rel):
    """{"rows", "label_counts"} of train_aug.csv (sentence_id,label,text_sha256_12); F06 when malformed."""
    data = fz.text(rel)
    if data is None:
        return None
    lines = data.split(b"\n")
    labels = [ln.split(b",")[1] if ln.count(b",") == 2 else None for ln in lines[1:-1]]
    if lines[0] != AUG_LIST_HEADER or lines[-1] != b"" or not labels or set(labels) - {b"0", b"1"}:
        fz.fail("F06", f"{rel}: not a list of at least one row {AUG_LIST_HEADER.decode()} with labels 0 / 1 "
                       "and a final LF")
        return None
    return {"rows": len(labels), "label_counts": {k: labels.count(k.encode()) for k in ("0", "1")}}


def augmentation_block(fz, c_runs):
    """c_runs = {run_id: files_sha256 of run.json data.augmentation, or None when the run was refused}."""
    files = {f"{AUG_DIR}/{n}": fz.sha(f"{AUG_DIR}/{n}") for n in AUG_FILES}
    for rel in fz.glob(f"{AUG_DIR}/prompts", ".md") + fz.glob(f"{AUG_DIR}/llm", ".json"):
        files[rel] = fz.sha(rel)
    frozen = {n: files[f"{AUG_DIR}/{n}"] for n in AUG_PINNED + AUG_MAY_DIFFER}
    block = {"dir": AUG_DIR, "files": files, "train_aug": train_aug_counts(fz, f"{AUG_DIR}/{AUG_LIST}"),
             "c_runs": {}}
    for run_id in sorted(c_runs):
        recorded = c_runs[run_id]
        if recorded is None:
            continue
        for n in AUG_PINNED:
            fz.same("F06", f"runs/{run_id}/run.json data > augmentation > files_sha256 > {n}", recorded[n],
                    frozen[n], f"the frozen {AUG_DIR}/{n}: condition C was not trained on the frozen augmentation")
        block["c_runs"][run_id] = {"files_sha256": dict(recorded), "differs_from_frozen": [
            n for n in AUG_MAY_DIFFER if frozen[n] is not None and recorded[n] != frozen[n]]}
    meta_rel = f"{AUG_DIR}/{AUG_META}"
    meta = fz.load_json(meta_rel)
    ready = fz.need(meta, meta_rel, ("ready_for_training",), "any")
    if ready is not True:
        fz.fail("F06", f"{meta_rel}: ready_for_training is {_short(ready)}, freeze 1 needs true")
    block["ready_for_training"] = ready
    for n in AUG_OUTPUTS:
        fz.same("F04", f"{meta_rel} outputs > {n} > sha256", fz.need(meta, meta_rel, ("outputs", n, "sha256"), "sha"),
                files[f"{AUG_DIR}/{n}"], f"{AUG_DIR}/{n} on disk")
    fz.same("F04", f"{meta_rel} script > sha256_lf", fz.need(meta, meta_rel, ("script", "sha256_lf"), "sha"),
            fz.sha(BUILD_TRAIN_AUG), f"{BUILD_TRAIN_AUG} on disk")
    generators = fz.need(meta, meta_rel, ("generators",), "dict")
    for name in sorted(generators):
        for path_key, sha_key in (("file", "sha256_lf"), ("prompt", "prompt_sha256_lf")):
            part = fz.need(meta, meta_rel, ("generators", name, path_key), "str")
            want = fz.need(meta, meta_rel, ("generators", name, sha_key), "sha")
            if not _safe_rel(part):
                fz.fail("F10", f"{meta_rel}: generators > {name} > {path_key} is not a safe relative path")
                continue
            rel = f"{AUG_DIR}/{part}"
            if rel not in files:  # a generator file outside prompts/*.md and llm/*.json is recorded too
                files[rel] = fz.sha(rel)
            fz.same("F04", f"{meta_rel} generators > {name} > {sha_key}", want, files[rel], f"{rel} on disk")
    return block


def final_block(fz):
    files = {f"{FINAL_DIR}/{n}": fz.sha(f"{FINAL_DIR}/{n}") for n in FINAL_FILES}
    prompts = fz.glob(f"{FINAL_DIR}/prompts", ".md")
    for rel in prompts:
        files[rel] = fz.sha(rel)
    block = {"dir": FINAL_DIR, "files": files,
             "generator_code": {"path": GENERATOR_REL, "sha256": fz.sha(GENERATOR_REL),
                                "provides": [name for name, _ in GENERATOR_DEFS]}}
    source = fz.text(GENERATOR_REL)
    lacking = [] if source is None else [n for n, pattern in GENERATOR_DEFS if not re.search(pattern, source, re.M)]
    if lacking:
        fz.fail("F08", f"{GENERATOR_REL}: does not define {', '.join(lacking)} (the generator code that "
                       f"{FINAL_DIR}/RULES.md names)")
    spec_rel = f"{FINAL_DIR}/final_spec.json"
    spec = fz.load_json(spec_rel)
    recorded = fz.need(spec, spec_rel, ("prompts_sha256_lf",), "dict")
    if sorted(recorded) != prompts:
        fz.fail("F08", f"{spec_rel}: prompts_sha256_lf names {len(recorded)} file(s), {FINAL_DIR}/prompts/ "
                       f"holds {len(prompts)}: only in the spec {sorted(set(recorded) - set(prompts))}, only "
                       f"in the folder {sorted(set(prompts) - set(recorded))}")
    for rel in prompts:
        if rel in recorded:
            fz.same("F08", f"{spec_rel} prompts_sha256_lf > {rel}", recorded[rel], files[rel], "the prompt file")
    block["extraction_seed"] = fz.need(spec, spec_rel, ("extraction", "seed"), "int")
    block["targets"] = fz.need(spec, spec_rel, ("targets",), "dict")
    block["llm_model"] = fz.need(spec, spec_rel, ("generators", "llm_model"), "str")
    return block


def external_source(fz, name, src, tag):
    """The record of one adopted source of external_rules.json (rules entry src)."""
    d = fz.need(src, tag, ("dir",), "str")
    man_rel = fz.need(src, tag, ("manifest",), "str")
    if d != f"{EXT_DIR}/{name}" or man_rel != f"{d}/{EXT_MANIFEST}" or not _safe_rel(man_rel):
        fz.fail("F12", f"{tag}: dir and manifest must be {EXT_DIR}/{name} and {EXT_DIR}/{name}/{EXT_MANIFEST}")
        raise _Stop
    dataset, revision = fz.need(src, tag, ("dataset",), "str"), fz.need(src, tag, ("revision",), "str")
    splits = fz.need(src, tag, ("adopted_splits",), "list")
    rule_files = fz.need(src, tag, ("files",), "list")
    man_sha = fz.sha(man_rel)
    fz.same("F04", f"{tag} > manifest_sha256", fz.need(src, tag, ("manifest_sha256",), "sha"), man_sha,
            f"{man_rel} on disk")
    m = fz.load_json(man_rel)
    if fz.need(m, man_rel, ("dataset",), "str") != dataset or fz.need(m, man_rel, ("revision",), "str") != revision:
        fz.fail("F12", f"{man_rel}: dataset or revision differs from {tag}")
    files = {}
    for n, f in enumerate(fz.need(m, man_rel, ("files",), "list")):
        ftag = f"{man_rel} files[{n}]"
        path = fz.need(f, ftag, ("path",), "str")
        size, sha = fz.need(f, ftag, ("size_bytes",), "int"), fz.need(f, ftag, ("sha256",), "sha")
        if not _safe_rel(path) or _is_test_csv(path):
            fz.fail("F12", f"{ftag}: path is not a safe relative path (or is named test.csv)")
            continue
        rel = f"{d}/{path}"
        got = fz.sha(rel, "bytes")
        if got is not None:
            fz.same("F04", f"{rel}: size on disk", os.stat(fz.path(rel)).st_size, size, f"{man_rel} size_bytes")
            fz.same("F04", f"{rel}: sha256 on disk", got, sha, f"{man_rel} sha256")
        files[rel] = {"sha256": sha, "size_bytes": size}
    listed = {(f"{d}/{f['path']}", f["sha256"], f["size_bytes"]) for f in rule_files
              if isinstance(f, dict) and isinstance(f.get("path"), str) and isinstance(f.get("sha256"), str)
              and type(f.get("size_bytes")) is int}
    if len(listed) != len(rule_files) or listed != {(rel, v["sha256"], v["size_bytes"]) for rel, v in files.items()}:
        fz.fail("F04", f"{tag} > files: path, size_bytes and sha256 differ from the files of {man_rel}")
    return {"dataset": dataset, "revision": revision, "dir": d, "adopted_splits": splits,
            "manifest": {"path": man_rel, "sha256": man_sha}, "files": files}


def external_block(fz):
    block = {"rules": {"path": EXT_RULES, "sha256": fz.sha(EXT_RULES)},
             "scripts": {p: fz.sha(p) for p in EXT_SCRIPTS}, "sources": {}}
    base = fz.path(EXT_DIR)
    try:
        on_disk = sorted(n for n in os.listdir(base) if (base / n / EXT_MANIFEST).is_file())
    except OSError:
        on_disk = []
    rules = fz.load_json(EXT_RULES)
    sources = fz.need(rules, EXT_RULES, ("sources",), "dict")
    if not sources:
        fz.fail("F12", f"{EXT_RULES}: sources names no source")
    for name in sorted(sources):
        if not _safe_rel(name) or "/" in name:
            fz.fail("F12", f"{EXT_RULES}: source name {_short(name)} is not a folder name")
            continue
        block["sources"][name] = section(fz, external_source, name, sources[name], f"{EXT_RULES} sources > {name}")
    stray = [n for n in on_disk if n not in sources]
    if stray:
        fz.fail("F12", f"{EXT_DIR}: download manifest of a source that {EXT_RULES} does not adopt: "
                       f"{', '.join(stray)}")
    return block


# ---------------------------------------------------------------- the record

def build(root, date, commit=None):
    """(errors, record, info). commit = None takes HEAD (a new freeze); --check passes the frozen commit."""
    fz = Freeze(root)
    git_errs, state = git_state(root)
    doc = {"schema": SCHEMA, "hash_rule": HASH_RULE,
           "freeze": {"stage": STAGE, "tag": TAG, "date": date,
                      "readme": "README 3.2 (test use procedure, step 1: design freeze) and 7.1 (freeze "
                                f"records); written by {SCRIPT_REL}"},
           "code_commit": commit or (state["head"] if state else None),
           "spec": {"path": README_REL, "sha256": fz.sha(README_REL)}}
    doc["dataset"] = section(fz, dataset_block)
    doc["lists"] = section(fz, lists_block, doc["dataset"])
    cfg = section(fz, config_block)
    doc["config"], seeds = cfg if cfg else (None, None)
    doc["environment"] = {"path": REQUIREMENTS_REL, "sha256": fz.sha(REQUIREMENTS_REL)}
    c_runs = {}
    if seeds is not None:
        lists = doc["lists"]["files"] if doc["lists"] else {}
        pins = {"config": doc["config"]["sha256"], "requirements": doc["environment"]["sha256"],
                "lists": {n: lists[n]["sha256"] for n in LIST_FILES if n in lists}}
        runs = {}
        for ident in expected_runs(seeds):
            got = section(fz, run_block, ident, pins)
            runs[ident[0]] = got[0] if got else None
            if ident[3] == "C":
                c_runs[ident[0]] = got[1] if got else None
        doc["models"] = {"keys": list(MODEL_KEYS), "representative": REPRESENTATIVE, "seeds": list(seeds)}
        doc["runs"] = runs
        thr = section(fz, thresholds_block, seeds, runs)
        doc["thresholds"] = thr[0] if thr else None
        if thr and len(thr[1]) == len(seeds) * len(TUNED):
            doc["conditions"] = conditions_block(seeds, thr[1])
    doc["augmentation"] = section(fz, augmentation_block, c_runs)
    doc["final_cases"] = section(fz, final_block)
    doc["dev_cases"] = {"dir": DEV_DIR, "files": {f"{DEV_DIR}/{n}": fz.sha(f"{DEV_DIR}/{n}") for n in DEV_FILES}}
    doc["external"] = section(fz, external_block)
    doc["results"] = {p: fz.sha(p) for p in RESULT_FILES}
    if state:
        code = sorted(p for p in state["tracked"] if p.split("/", 1)[0] in CODE_DIRS and "/" in p)
        doc["scripts"] = {p: fz.sha(p) for p in code}
    errors = fz.errors + (git_errs if state is None else git_errors(fz.recorded, state))
    info = {"files": len(fz.recorded), "head": state["head"] if state else None}
    return errors, (None if errors else doc), info


# ---------------------------------------------------------------- --check

def diff(old, new, path=""):
    """Differences between two parsed records as text lines (paths of the differing entries)."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in sorted(set(old) | set(new)):
            where = f"{path} > {k}" if path else str(k)
            if k not in new:
                out.append(f"{where}: only in the freeze file")
            elif k not in old:
                out.append(f"{where}: only in the fresh rebuild")
            else:
                out += diff(old[k], new[k], where)
        return out
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [line for i, (a, b) in enumerate(zip(old, new)) for line in diff(a, b, f"{path}[{i}]")]
    if type(old) is type(new) and old == new:
        return []
    return [f"{path}: freeze file {_short(old)} != fresh rebuild {_short(new)}"]


def run_check(root, out, shown):
    """(errors, message) of --check."""
    try:
        existing = out.read_bytes().replace(b"\r\n", b"\n")
    except OSError as exc:
        return [f"F09 --check: {shown} cannot be read ({type(exc).__name__}); build it first with --date"], None
    try:
        old = json.loads(existing.decode("utf-8"))
        frozen = old["freeze"]
        date, commit = frozen["date"], old["code_commit"]
        if frozen["stage"] != STAGE or frozen["tag"] != TAG or not _valid_date(date) or not (
                isinstance(commit, str) and COMMIT_RE.fullmatch(commit)):
            raise ValueError
    except (UnicodeDecodeError, ValueError, KeyError, TypeError, RecursionError):
        return [f"F09 --check: {shown} is not a stage-{STAGE} freeze record (JSON with freeze stage, tag, date "
                "and code_commit)"], None
    errors, doc, info = build(root, date, commit)
    if info["head"] is not None:
        if git(root, "cat-file", "-e", commit + "^{commit}").returncode != 0:
            errors.append(f"F09 --check: code_commit {commit[:12]} is not a commit of this repository")
        elif git(root, "merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
            errors.append(f"F09 --check: code_commit {commit[:12]} is not an ancestor of HEAD {info['head'][:12]}")
    if doc is not None and _dump(doc).encode("ascii") != existing:
        lines = diff(old, doc) or ["the entries are equal but the bytes differ (formatting)"]
        errors += [f"F09 --check: {ln}" for ln in lines[:SHOWN_DIFFS]]
        if len(lines) > SHOWN_DIFFS:
            errors.append(f"F09 --check: {len(lines) - SHOWN_DIFFS} more difference(s)")
    if errors:
        return errors, None
    return [], (f"OK --check: {shown} equals the fresh rebuild; {info['files']} files re-hashed, code_commit "
                f"{commit[:12]}, HEAD {info['head'][:12]}")


# ---------------------------------------------------------------- main

def _valid_date(text):
    if not isinstance(text, str) or not DATE_RE.fullmatch(text):
        return False
    try:
        datetime.date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _inside(path, folder):
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
    except (ValueError, OSError):
        return False
    return True


def arg_errors(args, root, out):
    errors = []
    if args.stage != STAGE:
        errors.append(f"F00 --stage {args.stage}: only stage {STAGE} is implemented ({OUT_REL}; freeze 2 is "
                      "written in stage 5)")
    if args.check and args.date is not None:
        errors.append("F00 --date with --check: the date is taken from the existing freeze file")
    if not args.check and args.date is None:
        errors.append("F00 --date YYYY-MM-DD is required (the date of the freeze; never taken from the clock)")
    elif not args.check and not _valid_date(args.date):
        errors.append("F00 --date is not a YYYY-MM-DD date")
    if _is_test_csv(out) or _is_test_csv(os.path.realpath(out)):
        errors.append("F00 the output path is named test.csv")
    elif _inside(out, root / "datasets"):
        errors.append("F00 the output path is inside datasets/")
    elif out.is_dir():
        errors.append("F00 the output path is a folder")
    elif not args.check and out.exists():
        errors.append("F00 the output file exists already: a freeze file is never overwritten (verify it with "
                      "--check, or remove it to build the record again)")
    return errors


def _refuse(errors):
    for e in errors:
        print(_ascii("FAIL " + e))
    print(f"refused: {len(errors)} error(s); nothing written")
    return 1


def parse_args(argv):
    p = argparse.ArgumentParser(description="Write or check the freeze-1 record (see the module docstring).")
    p.add_argument("--stage", type=int, required=True, help="freeze stage (only 1)")
    p.add_argument("--date", help="date of the freeze, YYYY-MM-DD (required unless --check)")
    p.add_argument("--check", action="store_true", help="rebuild and compare with the existing freeze file")
    p.add_argument("--out", help=f"freeze file to write or check (default: <root>/{OUT_REL})")
    p.add_argument("--root", help="repository root (default: this repository)")
    return p.parse_args(argv)


def main(argv=None):
    data_lists.install_test_guard()  # before any file is opened: this process never opens test.csv
    args = parse_args(argv)
    root = Path(args.root) if args.root else ROOT
    out = Path(args.out) if args.out else root / OUT_REL
    errors = arg_errors(args, root, out)
    if errors:
        return _refuse(errors)
    try:
        shown = out.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        shown = str(out)
    if args.check:
        errors, message = run_check(root, out, shown)
        if errors:
            return _refuse(errors)
        print(_ascii(message))
        return 0
    errors, doc, info = build(root, args.date)
    if errors:
        return _refuse(errors)
    data = _dump(doc).encode("ascii")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, out)
    print(_ascii(f"OK wrote {shown}: stage {STAGE}, date {args.date}, code_commit {doc['code_commit'][:12]}, "
                 f"{info['files']} files recorded, sha256 {data_lists.sha256_bytes(data)[:12]}"))
    print(f"next: commit {OUT_REL}, tag the commit {TAG} and push (README 7.1); verify later with "
          f"--stage {STAGE} --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
