"""Build the common sentence lists (README 3.2 DATA-01..06, 7.1) from the pinned raw splits.

Reads datasets/s-labs-prompt-injection/download_manifest.json (its SHA-256 is
pinned), the three raw splits it lists and data/processed/train_review.csv;
checks every raw file against the manifest (revision, size, SHA-256) before
parsing; writes ID lists (never sentence text) and lists_meta.json. Standard
library only.

Rules (README line numbers as of the commit that adds this script):
  - Sentence ID `split:NNNNNN`: 0-based index of the data records read with the
    Python csv module (UTF-8, newline=''), header excluded (7.1 L460). train
    has multi-line records (train:000994, train:009878), so IDs are not lines.
  - Exact key = the raw `text` field, unchanged. Normalised key = NFKC ->
    strip and collapse whitespace -> casefold (3.1 L119-120); used only for
    the train review groups and the train/validation audit.
  - train_clean: drop review rows with decision `exclude`, then keep the
    lowest sentence ID of every exact-key group (3.2 L174; counts L154).
  - validation_clean: original validation minus rows whose exact key is in
    the ORIGINAL train (3.1 L113; counts L142, L148).
  - test_clean_preaug: test minus rows whose exact key is in the cleaned
    train, the augmentation cases (none before the augmentation freeze) or the
    whole original validation (DATA-05 L133; counts L143, L144, L156). One
    primary reason per excluded row by precedence train_overlap >
    augmentation_overlap > validation_overlap; non-exclusive overlap counts
    are in lists_meta.json "audit".
  - test use (3.2 L176-184): automated duplicate IDs and counts only.
  - Review record contract (7.1 L461): stop codes R01-R23 below. confirmed_by
    IDs compare case-insensitively; empty tokens are R09 and are not counted
    by R20/R23; R23 compares each row's sorted ID set, so the order is free.
  - Expected counts are constants (EXPECTED) citing README; a legitimate
    decision change edits README 3.2 and EXPECTED together (B02).
  - Contract gap: a normalised-key variant of a conflict text forms a 1-row
    normkey_only group whose label is not compared with the label kept in the
    conflict group (cannot occur at the pinned revision).

Outputs (UTF-8 without BOM, LF, header, trailing LF, sorted by sentence_id):
  train_clean.csv           sentence_id,label,text_sha256_12
  train_excluded.csv        sentence_id,label,text_sha256_12,reason,ref
                            (duplicate_same_label -> kept ID; review_exclude -> group_id)
  validation_clean.csv      sentence_id,label,text_sha256_12
  validation_excluded.csv   sentence_id,label,text_sha256_12,reason,ref
                            (train_overlap -> lowest original train ID)
  test_clean_preaug.csv     sentence_id
  test_excluded_preaug.csv  sentence_id,reason
  lists_meta.json           sorted keys, indent 2, ASCII; sources, review, rules,
                            counts by reason, audit, output SHA-256; no
                            timestamp, git commit or absolute path (paths are
                            repo-relative POSIX, or null outside the repo).
text_sha256_12 = first 12 lowercase hex chars of SHA-256(raw text, UTF-8).

Modes:
  - trial = any review row has status `proposed`. Without --allow-proposed the
    run stops with R22 and writes nothing. With it, the lists go to
    <out-dir>/trial/ only (gitignored), meta "trial": true, "usable_for":
    "trial-only"; they must not be used for stage 2 or freeze-1.
  - official = no proposed rows: lists go to <out-dir>/ with "trial": false;
    --allow-proposed is ignored (bytes identical with or without it).
  - --check: rebuild in memory and require the official files in <out-dir>/
    to be byte-identical and meta "trial": false; writes nothing. Stage-2 and
    freeze-1 code must require it to pass.
  Write sequence: validate everything in memory; remove a stale <name>.tmp of
  every output; delete <target>/lists_meta.json; write each file to <name>.tmp
  (exclusive create) and os.replace it; meta last; on an exception the .tmp
  files are removed and the run fails with B05 (no meta is left).
  Output locations (B06, checked before any raw file is read):
    - --out-dir never under datasets/ (each ancestor compared by file identity);
    - a write to the default out-dir (official, or its trial/) and --check on
      it need the default --review and --manifest;
    - --out-dir data/processed/trial/ is refused for every write (official
      lists never go to the trial folder);
    - with --allow-proposed, <out-dir>/trial must be data/processed/trial/ or
      outside the repo (checked before the review is read, so also when the
      review turns out to be confirmed).

Stop codes (errors are collected and printed as `FAIL <code> <ids or counts>`,
exit 1. Nothing is written, except that a B05 failure, an audit-hook refusal or
a B07 inside the write sequence can leave some lists replaced and
lists_meta.json deleted; a folder without lists_meta.json is incomplete, rerun):
  D01 manifest unreadable, SHA-256 or revision != pinned (then no raw file is read)
  D02 raw file size/SHA-256 != manifest
  D03 BOM, invalid UTF-8 or header != text,label  D04 record field count != 2 (or csv error)
  D05 label not 0/1                               D06 empty or whitespace-only text
  D07 rows or label counts != manifest splits     D08 exact-key duplicate inside validation/test
  R01 review BOM / invalid UTF-8 / unreadable     R02 CR in review
  R03 missing final LF or blank line              R04 header != the 10 columns
  R05 field count != 10 or surrounding whitespace R06 sentence_id not train:NNNNNN in range
  R07 duplicate sentence_id                       R08 rows not strictly ascending
  R09 value outside the allowed sets              R10 decision/reason pair not allowed for check
  R11 group_id prefix != check                    R12 detected group row missing
  R13 row not in any detected group               R14 check != detected check
  R15 group_id -> ID sets != detected partition   R16 label != train.csv
  R17 text_sha256_12 != train.csv text            R18 kept rows of a conflict group carry >= 2 labels
  R19 normkey_only group with differing labels    R20 confirmed_by invalid for the status
  R21 conflict group incoherent: all rows label_undeterminable, or >= 1 kept row,
      every row with the kept label conflict_label_kept, every other row
      conflict_label_rejected
  R23 rows of one group differ in status or confirmed_by
  R22 proposed rows without --allow-proposed (checked after everything else)
  If R05, R06, R07 or R09 occur, the group checks R12-R15, R18, R19, R21 and R23
  are skipped (a `note:` line says so), so one malformed row does not cascade.
  B01 exact-key group with >= 2 labels after review exclusion
  B02 count != EXPECTED (actual and expected printed)
  B03 invariant broken (partition by label, disjointness, unique train_clean
      keys, test_clean disjoint from validation and cleaned train, review
      exclude rows listed as review_exclude, refs)
  B04 output line grammar, or a non-ASCII stdout line
  B05 write failure (PermissionError etc.)
  B06 refused location or mode (see Output locations); --check with
      --allow-proposed, with proposed review rows, or with missing, trial or
      differing files (a differing lists_meta.json names up to 10 differing
      keys); an event refused by the audit hook
  B07 unexpected exception: type and line in this script only, message suppressed

Test-safety guarantees: test.csv is read once, by load_test() only, as bytes
whose size and SHA-256 are checked before parsing; texts are reduced to SHA-256
digests in memory (no normalised key on test) and are never printed, logged or
written; exceptions inside load_test are reported by type and record number
only. Test lists carry IDs and reasons only; stdout is ASCII, counts and IDs.
main() installs an audit hook (once) that, while main runs, refuses (Python-level
opens and the listed events only):
  - an open of a path whose basename is test.csv (case-insensitive, also after
    resolving) outside load_test's single read;
  - under datasets/ (file identity of each ancestor): write-mode opens (mode
    w/a/x/+ or flags O_WRONLY, O_RDWR, O_APPEND, O_CREAT, O_TRUNC, O_TEMPORARY)
    and os.rename, os.remove, os.rmdir, os.truncate, os.mkdir, os.link,
    os.symlink, os.chmod, shutil.rmtree;
  - shutil.copyfile and _winapi.CopyFile2 (used by shutil.copy2, copytree and
    Path.copy) from a test.csv path or into datasets/.
Limits: a development-hygiene guard, not a security boundary; native code,
other audit events (e.g. os.utime), other processes and hard links made earlier
under another name are not covered. README's normalised-key figures for test
(3.2 L144, L147) are not re-checked; test is compared by exact key only.

Downstream use (stage-2 training, freeze-1 and later code):
  - read the official lists in data/processed/ only (never trial/); require
    lists_meta.json "trial": false and its review.sha256 equal to the SHA-256
    of the current train_review.csv; run `build_lists.py --check` as a
    subprocess and require exit 0;
  - map sentence_id -> text by re-parsing the raw splits exactly as here
    (load_manifest, read_verified, parse_split; ID = csv record index), then
    verify label and text_sha256_12 of every mapped train/validation row;
  - editing this script, or another Python / Unicode version, changes
    lists_meta.json (script.sha256_lf, unicode_version) and fails --check:
    regenerate and commit the official lists (after freeze-1, re-record them
    in the freeze record too).

Run from Git Bash at the repo root:
    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/build_lists.py --allow-proposed
        (now: the review is proposed -> trial lists in data/processed/trial/)
    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/build_lists.py
        (after both members confirm the review -> official lists)
    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/build_lists.py --check
Options: --review PATH, --manifest PATH, --out-dir PATH (defaults resolve from
the repo root, not the cwd).

Out of scope: augmentation input. build_test() takes augmentation_digests, but
the CLI has none yet. At the augmentation freeze the final post-augmentation
lists are test_clean.csv / test_excluded.csv with lists_meta.json
"test_clean_stage": "final" (README 3.2 L156); this script writes only the
pre-augmentation test_clean_preaug.csv / test_excluded_preaug.csv
("test_clean_stage": "pre-augmentation").

Exit code 0 on success, 1 on any failure.
"""
import argparse
import codecs
import csv
import hashlib
import io
import json
import os
import re
import sys
import traceback
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
DEFAULT_MANIFEST = DATASETS / "s-labs-prompt-injection" / "download_manifest.json"
DEFAULT_REVIEW = ROOT / "data" / "processed" / "train_review.csv"
DEFAULT_OUT = ROOT / "data" / "processed"
TRIAL_DIR = "trial"
META = "lists_meta.json"
SCHEMA = 1
PINNED_REVISION = "002a9dd18514abd021869823d6b0429b38606d99"  # README 3.1 L101
# download_manifest.json as committed in 4e019a9 (datasets/** is -text, so the bytes are stable).
PINNED_MANIFEST_SHA256 = "c9c2791b10c2be6e57b65edfb8fd04323b07f49e47dc6fd7d1e83d2ad2ec7d99"
SPLITS = ("train", "validation", "test")
LABELS = ("0", "1")

# Expected counts. README is the source of truth: a legitimate decision change
# (e.g. keeping one row of the conflict group) edits README 3.2 and this block
# together, otherwise the run stops with B02.
EXPECTED = {
    "train_clean": {"rows": 11050, "0": 6267, "1": 4783},                       # 3.2 L154
    "train_excluded": {"rows": 39, "duplicate_same_label": 37, "review_exclude": 2},  # 3.2 L154
    "validation_clean": {"rows": 2097, "0": 1047, "1": 1050},                  # 3.2 L148
    "validation_excluded": {"rows": 4, "train_overlap": 4},                    # 3.2 L142
    "test_clean_preaug": {"rows": 1985, "0": 1043, "1": 942},                  # 3.2 L156
    "test_excluded_preaug": {"rows": 116, "train_overlap": 0,                  # 3.2 L143
                             "augmentation_overlap": 0, "validation_overlap": 116},  # 3.2 L144, L156
    "test_overlap": {"original_train": 0, "cleaned_train": 0,                  # 3.2 L143
                     "augmentation": 0, "validation": 116},                    # 3.2 L144
}

OUTPUTS = ("train_clean", "train_excluded", "validation_clean", "validation_excluded",
           "test_clean_preaug", "test_excluded_preaug")
HEADERS = {
    "train_clean": "sentence_id,label,text_sha256_12",
    "train_excluded": "sentence_id,label,text_sha256_12,reason,ref",
    "validation_clean": "sentence_id,label,text_sha256_12",
    "validation_excluded": "sentence_id,label,text_sha256_12,reason,ref",
    "test_clean_preaug": "sentence_id",
    "test_excluded_preaug": "sentence_id,reason",
}
LINE_GRAMMAR = {k: re.compile(v) for k, v in {
    "train_clean": r"train:[0-9]{6},[01],[0-9a-f]{12}",
    "train_excluded": r"train:[0-9]{6},[01],[0-9a-f]{12},"
                      r"(?:duplicate_same_label,train:[0-9]{6}|review_exclude,T-(?:CONFLICT|NORMKEY)-[1-9][0-9]*)",
    "validation_clean": r"validation:[0-9]{6},[01],[0-9a-f]{12}",
    "validation_excluded": r"validation:[0-9]{6},[01],[0-9a-f]{12},train_overlap,train:[0-9]{6}",
    "test_clean_preaug": r"test:[0-9]{6}",
    "test_excluded_preaug": r"test:[0-9]{6},(?:train_overlap|augmentation_overlap|validation_overlap)",
}.items()}
TEST_REASONS = ("train_overlap", "augmentation_overlap", "validation_overlap")  # precedence order
OVERLAP_KEYS = ("original_train", "cleaned_train", "augmentation", "validation")  # non-exclusive test overlap

REVIEW_COLUMNS = ("sentence_id", "text_sha256_12", "group_id", "check", "label",
                  "decision", "reason", "reviewer", "status", "confirmed_by")
REVIEW_HEADER = ",".join(REVIEW_COLUMNS)
CHECKS = ("exact_label_conflict", "normkey_only")
DECISIONS = ("keep", "exclude")
REVIEW_REASONS = ("label_undeterminable", "conflict_label_kept", "conflict_label_rejected",
                  "same_label_case_variant")
STATUSES = ("proposed", "confirmed")
ALLOWED_PAIRS = {  # (check, decision, reason), README 7.1 L461
    ("exact_label_conflict", "exclude", "label_undeterminable"),
    ("exact_label_conflict", "keep", "conflict_label_kept"),
    ("exact_label_conflict", "exclude", "conflict_label_rejected"),
    ("normkey_only", "keep", "same_label_case_variant"),
}
GROUP_PREFIX = {"exact_label_conflict": "T-CONFLICT-", "normkey_only": "T-NORMKEY-"}
TRAIN_SID_RE = re.compile(r"train:[0-9]{6}")
GROUP_RE = re.compile(r"T-(?:CONFLICT|NORMKEY)-[1-9][0-9]*")
H12_RE = re.compile(r"[0-9a-f]{12}")
GITHUB_ID_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}")

RULES = {
    "sentence_id": "split:NNNNNN = 0-based index of the data records read with Python csv "
                   "(UTF-8, newline=''), header excluded (README 7.1)",
    "exact_key": "raw text field, unchanged (README 3.1)",
    "normalised_key": "NFKC, strip and collapse whitespace, casefold (README 3.1); used for train "
                      "review groups and the train/validation audit only, never on test",
    "text_sha256_12": "first 12 lowercase hex characters of SHA-256 of the raw text field in UTF-8",
    "train_clean": "drop review rows with decision exclude, then keep the lowest sentence ID of "
                   "every exact-key group (README 3.2)",
    "validation_clean": "original validation minus rows whose exact key is in the original train "
                        "(README 3.1)",
    "test_clean_preaug": "test minus rows whose exact key is in the cleaned train, the augmentation "
                         "cases (none before the augmentation freeze) or the whole original "
                         "validation (README 3.2 DATA-05); one primary reason per row by precedence "
                         "train_overlap > augmentation_overlap > validation_overlap",
    "test_audit": "test is compared by exact key only (SHA-256 digests); README normalised-key "
                  "figures for test are not re-checked",
}


class StopError(Exception):
    """A build-stage stop condition; str() is '<code> <detail>'."""


class GuardError(BaseException):
    """Raised by the audit hook. BaseException so that `except Exception` cannot swallow it."""


# ---------------------------------------------------------------- guard + stdout

_GUARD = {"installed": False, "active": False, "test_reads": 0, "stdout_bad": False}
_WRITE_FLAGS = (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
                | getattr(os, "O_TEMPORARY", 0))
# Audit events checked on Python 3.14 (Windows): path arguments are args[0] and, for
# rename/link/symlink, args[1]; other args[1] values (mode, length, dir_fd) are not paths.
_MUTATING_EVENTS = frozenset({"os.rename", "os.remove", "os.rmdir", "os.truncate", "os.mkdir", "os.link",
                              "os.symlink", "os.chmod", "shutil.rmtree"})
_COPY_EVENTS = frozenset({"shutil.copyfile", "_winapi.CopyFile2"})  # (src, dst, ...)


def _real(path):
    """normcase(realpath(path)); falls back to abspath if resolving fails."""
    try:
        return os.path.normcase(os.path.realpath(path))
    except (OSError, ValueError):
        return os.path.normcase(os.path.abspath(path))


def _file_id(path):
    """(st_dev, st_ino) of an existing path, or None (missing, unreadable or no inode number)."""
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return None
    return (st.st_dev, st.st_ino) if st.st_ino else None


def _same(path, other):
    """Same file or folder by identity; normcase(abspath) equality when either does not exist."""
    fid = _file_id(other)
    if fid is not None and _file_id(path) == fid:
        return True
    return os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(other))


def _under(path, base):
    """True when path is base or inside it.

    abspath(path) and realpath(path) (which resolves junctions and symlinks) and
    each of their ancestors are compared with base by file identity
    (normcase(abspath) equality as the fallback), so extended-length and UNC
    admin-share forms, trailing-space or trailing-dot aliases and junctions are
    caught too.
    """
    try:
        starts = {os.path.abspath(path), os.path.realpath(path)}
    except (TypeError, ValueError, OSError):
        return False
    fid = _file_id(base)
    base_s = os.path.normcase(os.path.abspath(base))
    for p in starts:
        while True:
            if os.path.normcase(p) == base_s or (fid is not None and _file_id(p) == fid):
                return True
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
    return False


def _is_test_csv(*paths):
    for p in paths:
        base = os.path.basename(p).split(":")[0].rstrip(". ").lower()
        if base == "test.csv":
            return True
    return False


def _arg_path(args, i):
    """args[i] as a str path, or None when it is missing or not a path (e.g. a file descriptor)."""
    p = args[i] if len(args) > i else None
    if not isinstance(p, (str, bytes, os.PathLike)):
        return None
    try:
        return os.fsdecode(p)
    except Exception:
        return None


def _audit(event, args):
    if not _GUARD["active"]:
        return
    if event == "open":
        p = _arg_path(args, 0)
        if p is None:
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = bool((isinstance(mode, str) and any(c in mode for c in "wax+"))
                       or (isinstance(flags, int) and flags & _WRITE_FLAGS))
        if writing and _under(p, DATASETS):
            raise GuardError("refused a write-mode open under datasets/")
        if _is_test_csv(p, _real(p)):
            if writing or _GUARD["test_reads"] < 1:
                raise GuardError("refused an open of a test.csv path outside load_test")
            _GUARD["test_reads"] -= 1
    elif event in _MUTATING_EVENTS:
        for i in (0, 1):
            p = _arg_path(args, i)
            if p is not None and _under(p, DATASETS):
                raise GuardError(f"refused {event} under datasets/")
    elif event in _COPY_EVENTS:
        src, dst = _arg_path(args, 0), _arg_path(args, 1)
        if src is not None and _is_test_csv(src, _real(src)):
            raise GuardError(f"refused {event} of a test.csv path")
        if dst is not None and _under(dst, DATASETS):
            raise GuardError(f"refused {event} into datasets/")


def _install_guard():
    if not _GUARD["installed"]:
        sys.addaudithook(_audit)
        _GUARD["installed"] = True


def say(msg):
    """Print one stdout line; a non-ASCII line is replaced and makes the run fail."""
    if not msg.isascii():
        _GUARD["stdout_bad"] = True
        msg = "FAIL B04 a non-ASCII stdout line was suppressed"
    print(msg, flush=True)


def fmt_ids(ids, limit=10):
    ids = sorted(ids)
    shown = ", ".join(ids[:limit])
    more = f", ... (+{len(ids) - limit} more)" if len(ids) > limit else ""
    return f"{len(ids)} ({shown}{more})"


def fmt_counts(counts, keys):
    return ", ".join(f"{k}: {counts.get(k, 0)}" for k in keys)


def exc_name(exc):
    """Exception type name only (e.g. csv.Error, PermissionError); the message is never shown."""
    t = type(exc)
    return t.__qualname__ if t.__module__ == "builtins" else f"{t.__module__.lstrip('_')}.{t.__qualname__}"


# ---------------------------------------------------------------- keys

def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def digest(text):
    """Exact key as a full SHA-256 hex digest of the raw text (UTF-8)."""
    return sha256_hex(text.encode("utf-8"))


def h12(text):
    return digest(text)[:12]


def nk(text):
    """Normalised key: NFKC -> strip and collapse whitespace -> casefold (README 3.1)."""
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


# ---------------------------------------------------------------- paths

def rel(path):
    """Repo-relative POSIX path, or None when the path is outside the repo."""
    real = os.path.realpath(path)
    root = os.path.realpath(ROOT)
    if os.path.normcase(real).startswith(os.path.normcase(root).rstrip(os.sep) + os.sep):
        return real[len(root.rstrip(os.sep)) + 1:].replace(os.sep, "/")
    return None


def show(path):
    """Repo-relative path, or the absolute path outside the repo; ASCII only (stdout is ASCII)."""
    r = rel(path)
    shown = r if r is not None else os.path.abspath(path)
    return shown if shown.isascii() else "<non-ASCII path>"


# ---------------------------------------------------------------- dataset

def load_manifest(path):
    """Return (info, manifest_meta, errors); info[split] = path, sha256, size, rows, labels (D01).

    The caller reads no raw file when errors is not empty.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        return None, None, [f"D01 manifest: cannot read ({exc_name(exc)})"]
    errors = []
    if sha256_hex(raw) != PINNED_MANIFEST_SHA256:
        errors.append(f"D01 manifest sha256 {sha256_hex(raw)[:12]} is not the pinned "
                      f"{PINNED_MANIFEST_SHA256[:12]} (download_manifest.json at commit 4e019a9)")
    try:
        m = json.loads(raw.decode("utf-8"))
        files = {f["path"]: f for f in m["files"]}
        info = {}
        for split in SPLITS:
            s = m["splits"][split]
            f = files[s["file"]]
            labels = {str(k).split("_")[0]: v for k, v in s["label_counts"].items()}
            info[split] = {"path": Path(path).parent / f["path"], "sha256": f["sha256"],
                           "size": f["size_bytes"],
                           "expected": {"rows": s["rows"], "labels": {k: labels.get(k) for k in LABELS}}}
        meta = {"name": m["dataset"], "revision": m.get("revision"), "sha256": sha256_hex(raw)}
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return None, None, errors + [f"D01 manifest: unreadable or missing keys ({exc_name(exc)})"]
    if meta["revision"] != PINNED_REVISION:
        errors.append(f"D01 manifest revision is not the pinned {PINNED_REVISION}")
    return info, meta, errors


def read_verified(path, sha256, size, name):
    """Read a raw file once as bytes; size and SHA-256 must equal the manifest (D02).

    Returns (bytes or None, errors); bytes are None on any mismatch, so nothing is parsed.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as exc:
        return None, [f"D02 {name}: cannot read ({exc_name(exc)})"]
    errors = []
    if len(data) != size:
        errors.append(f"D02 {name}: size {len(data)} differs from the manifest")
    if sha256_hex(data) != sha256:
        errors.append(f"D02 {name}: sha256 {sha256_hex(data)[:12]} differs from the manifest")
    return (None if errors else data), errors


def parse_split(data, split, expected=None):
    """Parse raw split bytes into [(sid, label, text)] with D03-D08 checks.

    expected = {"rows": n, "labels": {"0": n, "1": n}} enables D07; D08 (exact-key
    duplicate inside the split) applies to validation and test. Error messages
    carry IDs and counts only, never field values.
    """
    name = f"{split}.csv"
    if data.startswith(codecs.BOM_UTF8):
        return [], [f"D03 {name}: starts with a UTF-8 BOM"]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [], [f"D03 {name}: not valid UTF-8 (message suppressed)"]
    rows, errors = [], []
    bad_fields, bad_labels, empty = [], [], []
    counts = Counter()
    n = 0
    try:
        reader = csv.reader(io.StringIO(text, newline=""))
        if next(reader, None) != ["text", "label"]:
            return [], [f"D03 {name}: header is not text,label"]
        for rec in reader:
            sid = f"{split}:{n:06d}"
            n += 1
            if len(rec) != 2:
                bad_fields.append(sid)
                continue
            txt, label = rec
            if label not in LABELS:
                bad_labels.append(sid)
                continue
            counts[label] += 1
            if not txt.strip():
                empty.append(sid)
                continue
            rows.append((sid, label, txt))
    except Exception as exc:
        return [], [f"D04 {name}: {exc_name(exc)} at record {split}:{n:06d} (message suppressed)"]
    if bad_fields:
        errors.append(f"D04 {name}: field count != 2 in {fmt_ids(bad_fields)}")
    if bad_labels:
        errors.append(f"D05 {name}: label not 0/1 in {fmt_ids(bad_labels)}")
    if empty:
        errors.append(f"D06 {name}: empty or whitespace-only text in {fmt_ids(empty)}")
    if expected is not None:
        got = {"rows": n, "labels": {k: counts[k] for k in LABELS}}
        if got != expected:
            errors.append(f"D07 {name}: rows {n} ({fmt_counts(counts, LABELS)}) != manifest rows "
                          f"{expected['rows']} ({fmt_counts(expected['labels'], LABELS)})")
    if split != "train":
        seen, dups = set(), []
        for sid, _, txt in rows:
            if txt in seen:
                dups.append(sid)
            seen.add(txt)
        if dups:
            errors.append(f"D08 {name}: exact-key duplicate rows inside the split (no rule defined): "
                          f"{fmt_ids(dups)}")
    return rows, errors


def load_test(path, sha256, size, expected=None):
    """The only reader of test.csv. Returns ([(sid, label, digest)], errors).

    One read (the audit hook allows exactly one open while this runs), SHA-256 check
    before parsing, texts reduced to SHA-256 digests; any exception is reported by
    type only, its message suppressed.
    """
    _GUARD["test_reads"] = 1
    try:
        data, errors = read_verified(path, sha256, size, "test.csv")
        _GUARD["test_reads"] = 0
        if data is None:
            return [], errors
        parsed, perrors = parse_split(data, "test", expected)
        return [(sid, label, digest(txt)) for sid, label, txt in parsed], errors + perrors
    except Exception as exc:
        return [], [f"D04 test.csv: {exc_name(exc)} while loading (message suppressed)"]
    finally:
        _GUARD["test_reads"] = 0


# ---------------------------------------------------------------- review

def find_groups(train):
    """Detected review groups of train (README 7.1 L461).

    exact_label_conflict: all rows of a raw text that carries >= 2 labels.
    normkey_only: all rows of a normalised key that covers >= 2 distinct raw texts,
    minus rows already in a conflict group (those are written as conflict rows);
    empty remainders form no group. Returns ({sid: check}, set of frozenset(sids)).
    """
    by_text, labels = {}, {}
    for sid, label, text in train:
        by_text.setdefault(text, []).append(sid)
        labels.setdefault(text, set()).add(label)
    groups, conflict = [], set()
    for text, sids in by_text.items():
        if len(labels[text]) > 1:
            groups.append(("exact_label_conflict", frozenset(sids)))
            conflict.update(sids)
    by_key = {}
    for sid, _, text in train:
        by_key.setdefault(nk(text), []).append((sid, text))
    for members in by_key.values():
        if len({t for _, t in members}) > 1:
            rest = frozenset(s for s, _ in members if s not in conflict)
            if rest:
                groups.append(("normkey_only", rest))
    return {sid: check for check, g in groups for sid in g}, {g for _, g in groups}


def validate_review(data, train):
    """Validate train_review.csv bytes against train rows: R01-R21 and R23 (R22 is in main).

    After R05, R06, R07 or R09 the group checks are skipped and a `note: ` entry
    (not a stop code) is appended. Returns (rows, errors); rows are dicts keyed by
    REVIEW_COLUMNS plus "line".
    """
    errors = []
    if data.startswith(codecs.BOM_UTF8):
        return [], ["R01 review: starts with a UTF-8 BOM"]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [], ["R01 review: not valid UTF-8"]
    if "\r" in text:
        errors.append("R02 review: contains CR characters (LF line endings only)")
    if not text.endswith("\n"):
        errors.append("R03 review: missing final LF")
    lines = [ln[:-1] if ln.endswith("\r") else ln for ln in text.split("\n")]
    if text.endswith("\n"):
        lines.pop()
    blank = [str(i) for i, ln in enumerate(lines, 1) if not ln.strip()]
    if blank:
        errors.append(f"R03 review: blank line(s) at line {', '.join(blank[:10])}")
    if not lines or lines[0] != REVIEW_HEADER:
        errors.append("R04 review: header is not " + REVIEW_HEADER)
        return [], errors

    n_train = len(train)
    train_map = {sid: (label, txt) for sid, label, txt in train}
    rows, valid, unsorted, prev = [], [], [], None
    row_level = False  # R05, R06, R07 or R09: the group checks below would only cascade
    for ln, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) != len(REVIEW_COLUMNS):
            errors.append(f"R05 review line {ln}: {len(fields)} fields, expected {len(REVIEW_COLUMNS)}")
            row_level = True
            continue
        spaced = [c for c, f in zip(REVIEW_COLUMNS, fields) if f != f.strip()]
        if spaced:
            errors.append(f"R05 review line {ln}: surrounding whitespace in {', '.join(spaced)}")
            row_level = True
            continue
        r = dict(zip(REVIEW_COLUMNS, fields))
        r["line"] = ln
        rows.append(r)
        sid = r["sentence_id"]
        if prev is not None and not sid > prev:
            unsorted.append(str(ln))
        prev = sid
        ok_sid = bool(TRAIN_SID_RE.fullmatch(sid)) and int(sid[6:]) < n_train
        tag = sid if ok_sid else f"line {ln}"
        if not ok_sid:
            errors.append(f"R06 review line {ln}: sentence_id is not train:NNNNNN below {n_train}")
            row_level = True
        tokens = r["confirmed_by"].split(";") if r["confirmed_by"] else []
        bad = [c for c, ok in (
            ("text_sha256_12", H12_RE.fullmatch(r["text_sha256_12"])),
            ("group_id", GROUP_RE.fullmatch(r["group_id"])),
            ("check", r["check"] in CHECKS),
            ("label", r["label"] in LABELS),
            ("decision", r["decision"] in DECISIONS),
            ("reason", r["reason"] in REVIEW_REASONS),
            ("reviewer", r["reviewer"] == "claude" or GITHUB_ID_RE.fullmatch(r["reviewer"])),
            ("status", r["status"] in STATUSES),
            ("confirmed_by", all(GITHUB_ID_RE.fullmatch(t) for t in tokens)),
        ) if not ok]
        if bad:
            errors.append(f"R09 review {tag}: value outside the allowed set in {', '.join(bad)}")
            row_level = True
        if r["check"] in CHECKS and r["decision"] in DECISIONS and r["reason"] in REVIEW_REASONS \
                and (r["check"], r["decision"], r["reason"]) not in ALLOWED_PAIRS:
            errors.append(f"R10 review {tag}: decision/reason pair not allowed for {r['check']}")
        if r["check"] in CHECKS and GROUP_RE.fullmatch(r["group_id"]) \
                and not r["group_id"].startswith(GROUP_PREFIX[r["check"]]):
            errors.append(f"R11 review {tag}: group_id prefix does not match check")
        if ok_sid and sid in train_map:
            t_label, t_text = train_map[sid]
            if r["label"] in LABELS and r["label"] != t_label:
                errors.append(f"R16 review {sid}: label differs from train.csv")
            if H12_RE.fullmatch(r["text_sha256_12"]) and r["text_sha256_12"] != h12(t_text):
                errors.append(f"R17 review {sid}: text_sha256_12 differs from train.csv")
        if r["status"] in STATUSES:
            ids = [t for t in tokens if t]  # empty tokens are R09
            lowered = [t.lower() for t in ids]
            why = []
            if r["status"] == "confirmed" and len(ids) != 2:
                why.append("confirmed needs exactly 2 member IDs")
            if r["status"] == "proposed" and len(ids) > 1:
                why.append("proposed allows at most 1 member ID")
            if len(set(lowered)) != len(lowered):
                why.append("IDs not distinct (case-insensitive)")
            if "claude" in lowered:
                why.append("claude is not a member")
            if why:
                errors.append(f"R20 review {tag}: confirmed_by invalid: {'; '.join(why)}")
        if ok_sid and not bad:
            valid.append(r)
    if unsorted:
        errors.append(f"R08 review: rows not strictly ascending by sentence_id at line(s) {', '.join(unsorted[:10])}")
    dup = [s for s, c in Counter(r["sentence_id"] for r in rows if TRAIN_SID_RE.fullmatch(r["sentence_id"])).items()
           if c > 1]
    if dup:
        errors.append(f"R07 review: duplicate sentence_id {fmt_ids(dup)}")
        row_level = True
    if row_level:
        errors.append("note: group checks R12-R15, R18, R19, R21 and R23 skipped; "
                      "fix the row-level errors above first")
        return rows, errors

    expected_check, partition = find_groups(train)
    listed = {r["sentence_id"] for r in valid}
    missing = set(expected_check) - listed
    if missing:
        errors.append(f"R12 review: rows of detected groups missing: {fmt_ids(missing)}")
    extra = listed - set(expected_check)
    if extra:
        errors.append(f"R13 review: rows not in any detected group: {fmt_ids(extra)}")
    wrong = [r["sentence_id"] for r in valid
             if r["sentence_id"] in expected_check and r["check"] != expected_check[r["sentence_id"]]]
    if wrong:
        errors.append(f"R14 review: check differs from the detected check: {fmt_ids(wrong)}")

    groups = {}
    for r in valid:
        groups.setdefault(r["group_id"], []).append(r)
    off = [g for g, rs in groups.items() if frozenset(r["sentence_id"] for r in rs) not in partition]
    if off:
        errors.append(f"R15 review: group_id ID sets differ from the detected groups: {fmt_ids(off)}")
    for gid in sorted(groups):
        rs = groups[gid]
        who = {tuple(sorted(t.lower() for t in r["confirmed_by"].split(";") if t)) for r in rs}
        if len({r["status"] for r in rs}) > 1 or len(who) > 1:
            errors.append(f"R23 review {gid}: rows differ in status or confirmed_by")
        if gid.startswith("T-NORMKEY-"):
            if len({r["label"] for r in rs}) > 1:
                errors.append(f"R19 review {gid}: normkey_only group with differing labels")
            continue
        kept = {r["label"] for r in rs if r["decision"] == "keep"}
        if len(kept) > 1:
            errors.append(f"R18 review {gid}: kept rows carry {len(kept)} labels")
            continue
        if all(r["reason"] == "label_undeterminable" for r in rs):
            continue
        coherent = len(kept) == 1 and all(
            r["reason"] == ("conflict_label_kept" if r["label"] in kept else "conflict_label_rejected")
            for r in rs)
        if not coherent:
            errors.append(f"R21 review {gid}: conflict group incoherent (all label_undeterminable, or the "
                          f"kept label conflict_label_kept and every other row conflict_label_rejected)")
    return rows, errors


# ---------------------------------------------------------------- build

def build_train(train, review_rows):
    """Cleaned train (README 3.2 L174). Returns (clean, excluded), sorted by sid.

    clean rows (sid, label, h12); excluded rows (sid, label, h12, reason, ref).
    Raises StopError B01 when an exact-key group keeps >= 2 labels after review exclusion.
    """
    review = {r["sentence_id"]: r for r in review_rows}
    excluded, members = [], {}
    for sid, label, text in train:
        r = review.get(sid)
        if r is not None and r["decision"] == "exclude":
            excluded.append((sid, label, h12(text), "review_exclude", r["group_id"]))
        else:
            members.setdefault(text, []).append((sid, label))
    mixed = [min(m)[0] for m in members.values() if len({lab for _, lab in m}) > 1]
    if mixed:
        raise StopError(f"B01 exact-key groups with >= 2 labels after review exclusion: {fmt_ids(mixed)}")
    clean = []
    for text, m in members.items():
        keep_sid, keep_label = min(m)
        clean.append((keep_sid, keep_label, h12(text)))
        excluded += [(sid, lab, h12(text), "duplicate_same_label", keep_sid) for sid, lab in m if sid != keep_sid]
    return sorted(clean), sorted(excluded)


def build_validation(validation, train_original):
    """validation_clean (README 3.1 L113): drop rows whose exact key is in the ORIGINAL train.

    Returns (clean [(sid, label, h12)], excluded [(sid, label, h12, "train_overlap", lowest train sid)]).
    """
    first = {}
    for sid, _, text in train_original:
        if text not in first or sid < first[text]:
            first[text] = sid
    clean, excluded = [], []
    for sid, label, text in validation:
        if text in first:
            excluded.append((sid, label, h12(text), "train_overlap", first[text]))
        else:
            clean.append((sid, label, h12(text)))
    return sorted(clean), sorted(excluded)


def build_test(test_digests, *, cleaned_train_digests, validation_digests, original_train_digests,
               augmentation_digests=frozenset()):
    """test_clean (DATA-05): drop rows whose exact key is in cleaned train, augmentation or validation.

    test_digests: [(sid, label, digest)] as returned by load_test (only the first and
    last fields are used). One primary reason by precedence train_overlap >
    augmentation_overlap > validation_overlap. Returns (clean [(sid,)],
    excluded [(sid, reason)], overlap) where overlap holds non-exclusive row counts
    against original_train, cleaned_train, augmentation and validation.
    """
    sources = tuple(zip(TEST_REASONS, (cleaned_train_digests, augmentation_digests, validation_digests)))
    pools = tuple(zip(OVERLAP_KEYS, (original_train_digests, cleaned_train_digests, augmentation_digests,
                                     validation_digests)))
    clean, excluded = [], []
    overlap = Counter()
    for row in test_digests:
        sid, dg = row[0], row[-1]
        for key, pool in pools:
            overlap[key] += dg in pool
        reason = next((name for name, pool in sources if dg in pool), None)
        if reason is None:
            clean.append((sid,))
        else:
            excluded.append((sid, reason))
    return sorted(clean), sorted(excluded), {k: overlap[k] for k in OVERLAP_KEYS}


def count_lists(lists, test_labels, overlap):
    """Counts compared with EXPECTED: rows and labels for clean lists, rows and reasons for excluded."""
    counts = {}
    for stem in OUTPUTS:
        rows = lists[stem]
        c = {"rows": len(rows)}
        if "_clean" in stem:
            labs = [test_labels[r[0]] for r in rows] if stem.startswith("test") else [r[1] for r in rows]
            c.update({k: labs.count(k) for k in LABELS})
        else:
            c.update(Counter(r[1] if stem.startswith("test") else r[3] for r in rows))
        counts[stem] = c
    counts["test_overlap"] = dict(overlap)
    return counts


def check_expected(counts, expected=EXPECTED):
    """B02: every count must equal the README constants (actual and expected printed)."""
    errors = []
    for stem, want in expected.items():
        got = counts.get(stem, {})
        diffs = [f"{k} {got.get(k, 0)} (expected {want.get(k, 0)})"
                 for k in sorted(set(want) | set(got)) if got.get(k, 0) != want.get(k, 0)]
        if diffs:
            errors.append(f"B02 {stem}: {', '.join(diffs)}; a legitimate decision change edits README 3.2 "
                          f"and EXPECTED in scripts/build_lists.py together")
    return errors


def check_invariants(train, validation, test, review_rows, lists):
    """B03 invariants on the built lists (train/validation rows (sid, label, text), test rows (sid, label, digest))."""
    errors = []
    tr = {sid: (label, text) for sid, label, text in train}
    va = {sid: (label, text) for sid, label, text in validation}
    te = {sid: (label, dg) for sid, label, dg in test}

    def partition(split, src, clean, excl):
        c_ids, x_ids = [r[0] for r in lists[clean]], [r[0] for r in lists[excl]]
        if len(set(c_ids)) != len(c_ids) or len(set(x_ids)) != len(x_ids) or set(c_ids) & set(x_ids):
            errors.append(f"B03 {split}: clean and excluded lists are not disjoint")
        if set(c_ids) | set(x_ids) != set(src):
            errors.append(f"B03 {split}: clean + excluded does not cover the source rows")
        if Counter(src[s][0] for s in c_ids + x_ids if s in src) != Counter(v[0] for v in src.values()):
            errors.append(f"B03 {split}: clean + excluded label counts differ from the source")
        if split != "test":
            bad = [r[0] for r in lists[clean] + lists[excl]
                   if r[0] not in src or r[1] != src[r[0]][0] or r[2] != h12(src[r[0]][1])]
            if bad:
                errors.append(f"B03 {split}: label or text_sha256_12 differs from the source: {fmt_ids(bad)}")

    partition("train", tr, "train_clean", "train_excluded")
    partition("validation", va, "validation_clean", "validation_excluded")
    partition("test", te, "test_clean_preaug", "test_excluded_preaug")

    clean_texts = [tr[r[0]][1] for r in lists["train_clean"] if r[0] in tr]
    if len(set(clean_texts)) != len(clean_texts):
        errors.append("B03 train_clean: exact keys are not unique")
    clean_ids = {r[0] for r in lists["train_clean"]}
    bad_ref = [r[0] for r in lists["train_excluded"] if r[3] == "duplicate_same_label"
               and (r[4] not in clean_ids or tr.get(r[4], (None, None))[1] != tr.get(r[0], (None, ""))[1])]
    if bad_ref:
        errors.append(f"B03 train_excluded: duplicate_same_label ref is not the kept row: {fmt_ids(bad_ref)}")
    review_excl = {r["sentence_id"] for r in review_rows if r["decision"] == "exclude"}
    listed = {r[0] for r in lists["train_excluded"] if r[3] == "review_exclude"}
    if review_excl != listed:
        errors.append(f"B03 train_excluded: review exclude rows != review_exclude rows: "
                      f"{fmt_ids(review_excl ^ listed)}")
    train_texts = {text for _, _, text in train}
    leak = [r[0] for r in lists["validation_clean"] if va.get(r[0], (None, None))[1] in train_texts]
    if leak:
        errors.append(f"B03 validation_clean: rows share an exact key with the original train: {fmt_ids(leak)}")
    first_train = {}
    for sid, _, text in train:
        if text not in first_train or sid < first_train[text]:
            first_train[text] = sid
    bad_ref = [r[0] for r in lists["validation_excluded"]
               if r[4] != first_train.get(va.get(r[0], (None, None))[1])]
    if bad_ref:
        errors.append(f"B03 validation_excluded: train_overlap ref is not the lowest original train ID "
                      f"with the same text: {fmt_ids(bad_ref)}")
    val_dg = {digest(text) for _, _, text in validation}
    cleaned_dg = {digest(tr[s][1]) for s in clean_ids if s in tr}
    test_clean_dg = {te[r[0]][1] for r in lists["test_clean_preaug"] if r[0] in te}
    if test_clean_dg & val_dg:
        errors.append(f"B03 test_clean_preaug: {len(test_clean_dg & val_dg)} row(s) share an exact key with validation")
    if test_clean_dg & cleaned_dg:
        errors.append(f"B03 test_clean_preaug: {len(test_clean_dg & cleaned_dg)} row(s) share an exact key "
                      f"with the cleaned train")
    return errors


def compute_audit(train, validation, test, partition_checks, cleaned_train_digests, overlap):
    """Non-exclusive counts for lists_meta.json (README 3.2 check table)."""
    t_texts = [text for _, _, text in train]
    by_text = {}
    for _, label, text in train:
        by_text.setdefault(text, set()).add(label)
    v_texts = [text for _, _, text in validation]
    train_set = set(t_texts)
    val_by_dg = {digest(text): label for _, label, text in validation}
    over = [(label, val_by_dg[dg]) for _, label, dg in test if dg in val_by_dg]
    return {
        "train": {
            "exact_duplicate_extra_rows": len(t_texts) - len(train_set),
            "conflict_texts": sum(len(v) > 1 for v in by_text.values()),
            "normkey_duplicate_extra_rows": len(t_texts) - len({nk(t) for t in t_texts}),
            "review_groups": dict(sorted(Counter(partition_checks).items())),
            "whitespace_edge_rows": sum(t != t.strip() for t in t_texts),
        },
        "validation": {
            "internal_exact_duplicate_rows": len(v_texts) - len(set(v_texts)),
            "internal_normkey_duplicate_rows": len(v_texts) - len({nk(t) for t in v_texts}),
            "whitespace_edge_rows": sum(t != t.strip() for t in v_texts),
            "original_train_overlap_rows": sum(t in train_set for t in v_texts),
            "cleaned_train_overlap_rows": sum(digest(t) in cleaned_train_digests for t in v_texts),
        },
        "test": {
            "internal_exact_duplicate_rows": len(test) - len({dg for _, _, dg in test}),
            "original_train_overlap_rows": overlap["original_train"],
            "cleaned_train_overlap_rows": overlap["cleaned_train"],
            "augmentation_overlap_rows": overlap["augmentation"],
            "validation_overlap_rows": overlap["validation"],
            "validation_overlap_labels": {k: sum(t == k for t, _ in over) for k in LABELS},
            "validation_overlap_labels_agree": all(t == v for t, v in over),
            "note": "exact key only; README normalised-key figures for test are not re-checked",
        },
    }


# ---------------------------------------------------------------- render + grammar

def render_csv(stem, rows):
    return ("\n".join([HEADERS[stem]] + [",".join(r) for r in rows]) + "\n").encode("utf-8")


def render_meta(meta):
    return (json.dumps(meta, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def check_grammar(name, data):
    """B04: ASCII, LF only, one trailing LF, exact header, line grammar, strictly sorted IDs."""
    if not data.isascii():
        return [f"B04 {name}: non-ASCII bytes"]
    errors = []
    if b"\r" in data or not data.endswith(b"\n") or data.endswith(b"\n\n"):
        errors.append(f"B04 {name}: line endings are not LF with exactly one trailing LF")
    if name == META:
        try:
            json.loads(data)
        except ValueError:
            errors.append(f"B04 {name}: not valid JSON")
        return errors
    stem = name[:-len(".csv")]
    lines = data.decode("ascii").rstrip("\n").split("\n")
    if lines[0] != HEADERS.get(stem):
        errors.append(f"B04 {name}: header differs")
    bad = [str(i) for i, ln in enumerate(lines[1:], start=2) if not LINE_GRAMMAR[stem].fullmatch(ln)]
    if bad:
        errors.append(f"B04 {name}: {len(bad)} line(s) violate the grammar (first at line {bad[0]})")
    sids = [ln.split(",")[0] for ln in lines[1:]]
    if any(not b > a for a, b in zip(sids, sids[1:])):
        errors.append(f"B04 {name}: rows not strictly ascending by sentence_id")
    return errors


# ---------------------------------------------------------------- write / check

def write_outputs(target, files):
    """Remove stale <name>.tmp files, delete meta, write each file to <name>.tmp (exclusive
    create) + os.replace, meta last. Returns None or a B05 error."""
    names = [n for n in files if n != META] + [META]
    tmps, current = [], "creating the target folder"
    try:
        target.mkdir(parents=True, exist_ok=True)
        for name in [f"{stem}.csv" for stem in OUTPUTS] + [META]:
            current = f"removing a stale {name}.tmp"
            (target / (name + ".tmp")).unlink(missing_ok=True)
        current = f"deleting {META}"
        if (target / META).exists():
            os.remove(target / META)
        for name in names:
            current = f"writing {name}"
            tmp = target / (name + ".tmp")
            with open(tmp, "xb") as f:
                tmps.append(tmp)
                f.write(files[name])
            os.replace(tmp, target / name)
        return None
    except Exception as exc:
        return (f"B05 {exc_name(exc)} while {current}; .tmp files removed and {META} "
                f"(written last) not replaced; fix the cause (e.g. close the file) and rerun")
    finally:
        for tmp in tmps:
            try:
                if tmp.exists():
                    os.remove(tmp)
            except OSError:
                pass


def _diff_keys(old, new, prefix=""):
    """Dotted keys whose values differ between two parsed JSON values."""
    if isinstance(old, dict) and isinstance(new, dict):
        keys = []
        for k in sorted(set(old) | set(new), key=str):
            path = f"{prefix}{k}"
            keys += _diff_keys(old[k], new[k], path + ".") if k in old and k in new else [path]
        return keys
    return [] if type(old) is type(new) and old == new else [prefix[:-1] or "<root>"]


def compare_official(out_dir, files):
    """--check: the official files in out_dir must equal the rebuilt bytes and meta trial must be false."""
    errors = []
    for name, data in files.items():
        try:
            with open(out_dir / name, "rb") as f:
                old = f.read()
        except OSError:
            errors.append(f"B06 --check: {name} is missing in the out-dir")
            continue
        if name != META:
            if old != data:
                errors.append(f"B06 --check: {name} differs from the rebuilt bytes")
            continue
        try:
            old_meta = json.loads(old.decode("utf-8"))
        except ValueError:
            old_meta = None
        if not isinstance(old_meta, dict) or old_meta.get("trial") is not False:
            errors.append(f"B06 --check: {META} does not have trial false")
        if old == data:
            continue
        if old_meta is None:
            errors.append(f"B06 --check: {META} differs from the rebuilt bytes (not valid JSON)")
            continue
        keys = _diff_keys(old_meta, json.loads(data))
        if not keys:
            errors.append(f"B06 --check: {META} differs from the rebuilt bytes in formatting only")
            continue
        msg = f"B06 --check: {META} differs from the rebuilt bytes at keys {fmt_ids(keys)}"
        if {"script.sha256_lf", "unicode_version"} & set(keys):
            msg += "; script or Python version changed: regenerate and commit the official lists"
        errors.append(msg)
    return errors


# ---------------------------------------------------------------- main

def _parse_args(argv):
    p = argparse.ArgumentParser(description="Build the common sentence lists (README 3.2, 7.1).")
    p.add_argument("--allow-proposed", action="store_true",
                   help="build trial lists in <out-dir>/trial/ while the review has proposed rows")
    p.add_argument("--check", action="store_true",
                   help="verify the official lists in <out-dir>/ without writing")
    p.add_argument("--review", help="review record (default data/processed/train_review.csv)")
    p.add_argument("--manifest", help="dataset manifest (default datasets/s-labs-prompt-injection/download_manifest.json)")
    p.add_argument("--out-dir", help="output folder (default data/processed)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    _install_guard()
    _GUARD.update(active=True, test_reads=0, stdout_bad=False)
    try:
        code = _run(args)
    except GuardError as exc:
        say(f"FAIL B06 audit hook {exc}")
        code = 1
    except Exception as exc:
        lines = [ln for frame, ln in traceback.walk_tb(exc.__traceback__)
                 if os.path.basename(frame.f_code.co_filename) == os.path.basename(__file__)]
        say(f"FAIL B07 unexpected {exc_name(exc)} at build_lists.py:{lines[-1] if lines else '?'} "
            f"(message suppressed)")
        code = 1
    finally:
        _GUARD.update(active=False, test_reads=0)
    return 1 if _GUARD["stdout_bad"] else code


def _fail(errors):
    """Print every error as FAIL (a `note: ` entry as is) and the summary; returns exit code 1."""
    for e in errors:
        say(e if e.startswith("note: ") else "FAIL " + e)
    say(f"FAILED: {sum(not e.startswith('note: ') for e in errors)} error(s); nothing written")
    return 1


def _run(args):
    errors = []
    out_dir = Path(args.out_dir).resolve() if args.out_dir else DEFAULT_OUT
    review_path = Path(args.review).resolve() if args.review else DEFAULT_REVIEW
    manifest_path = Path(args.manifest).resolve() if args.manifest else DEFAULT_MANIFEST

    # B06 output locations, before any raw file is read.
    if args.check and args.allow_proposed:
        errors.append("B06 --check and --allow-proposed are mutually exclusive")
    if _under(out_dir, DATASETS):
        errors.append("B06 --out-dir is under datasets/ (raw data stays read-only)")
    if _same(out_dir, DEFAULT_OUT) and not (_same(review_path, DEFAULT_REVIEW)
                                            and _same(manifest_path, DEFAULT_MANIFEST)):
        errors.append("B06 the default out-dir data/processed/ (official lists, trial/ and --check) needs the "
                      "default --review and --manifest; pass --out-dir for another build")
    if not args.check:
        trial_dir, default_trial = out_dir / TRIAL_DIR, DEFAULT_OUT / TRIAL_DIR
        if _same(out_dir, default_trial):
            errors.append("B06 --out-dir is the trial folder data/processed/trial/: official lists never go "
                          "there (trial lists go to <out-dir>/trial/)")
        elif args.allow_proposed and _under(trial_dir, ROOT) and not _same(trial_dir, default_trial):
            errors.append(f"B06 --allow-proposed: trial lists go to data/processed/trial/ or outside the repo "
                          f"only, not {show(trial_dir)}/")
    if errors:
        return _fail(errors)

    info, man, errs = load_manifest(manifest_path)
    errors += errs
    src = {}
    if info is not None and not errs:
        for split in ("train", "validation"):
            data, errs = read_verified(info[split]["path"], info[split]["sha256"], info[split]["size"], f"{split}.csv")
            if data is not None:
                rows, more = parse_split(data, split, info[split]["expected"])
                errs += more
                if not more:
                    src[split] = rows
            errors += errs
        rows, errs = load_test(info["test"]["path"], info["test"]["sha256"], info["test"]["size"],
                               info["test"]["expected"])
        errors += errs
        if not errs:
            src["test"] = rows
        for split in SPLITS:
            if split in src:
                c = Counter(r[1] for r in src[split])
                say(f"ok {split}.csv: size and sha256 match the manifest, rows {len(src[split])} "
                    f"({fmt_counts(c, LABELS)})")

    review_bytes, review_rows = None, None
    try:
        with open(review_path, "rb") as f:
            review_bytes = f.read()
    except OSError as exc:
        errors.append(f"R01 review: cannot read ({exc_name(exc)})")
    if review_bytes is not None and "train" in src:
        rows, errs = validate_review(review_bytes, src["train"])
        errors += errs
        if not errs:
            review_rows = rows
            status = Counter(r["status"] for r in rows)
            say(f"ok review {show(review_path)}: rows {len(rows)} (proposed {status['proposed']}, "
                f"confirmed {status['confirmed']}), groups {len({r['group_id'] for r in rows})}")

    files = None
    trial = review_rows is not None and any(r["status"] == "proposed" for r in review_rows)
    if len(src) == len(SPLITS) and review_rows is not None:
        files, errs = _build(src, review_rows, info, man, manifest_path, review_path, review_bytes, trial)
        errors += errs

    if args.check:
        if trial:
            errors.append("B06 --check needs a confirmed review (the review has proposed rows)")
        elif files is not None and not errors:
            errors += compare_official(out_dir, files)
    if trial and not args.allow_proposed and not args.check:
        proposed = [r["sentence_id"] for r in review_rows if r["status"] == "proposed"]
        errors.append(f"R22 review has proposed rows {fmt_ids(proposed)}: confirm the review first (status "
                      f"confirmed and two member IDs in confirmed_by on every row) and rerun for the official "
                      f"lists, or rerun with --allow-proposed for a trial list in <out-dir>/trial/ that must "
                      f"not be used for stage 2 or freeze-1")
    if errors:
        return _fail(errors)
    if args.check:
        say(f"ok --check: {len(files)} official files in {show(out_dir)} match the rebuilt bytes (trial false)")
        return 0
    if args.allow_proposed and not trial:
        say("note: --allow-proposed ignored (the review has no proposed rows); writing the official lists")
    target = out_dir / TRIAL_DIR if trial else out_dir
    err = write_outputs(target, files)
    if err:
        say("FAIL " + err)
        return 1
    say(f"wrote {show(target)}/: {', '.join(n for n in files if n != META)}, {META}")
    if trial:
        say(f"TRIAL LISTS ONLY ({show(target)}/, meta trial true): the review has proposed rows; "
            f"do not use these lists for stage 2 or freeze-1")
    else:
        say("OFFICIAL LISTS written (meta trial false)")
    return 0


def _build(src, review_rows, info, man, manifest_path, review_path, review_bytes, trial):
    """Build, check and render every output in memory. Returns (files or None, errors)."""
    train, validation, test = src["train"], src["validation"], src["test"]
    try:
        train_clean, train_excluded = build_train(train, review_rows)
    except StopError as exc:
        return None, [str(exc)]
    validation_clean, validation_excluded = build_validation(validation, train)
    clean_ids = {r[0] for r in train_clean}
    cleaned_dg = frozenset(digest(text) for sid, _, text in train if sid in clean_ids)
    test_clean, test_excluded, overlap = build_test(
        test, cleaned_train_digests=cleaned_dg,
        validation_digests=frozenset(digest(text) for _, _, text in validation),
        original_train_digests=frozenset(digest(text) for _, _, text in train))
    lists = {"train_clean": train_clean, "train_excluded": train_excluded,
             "validation_clean": validation_clean, "validation_excluded": validation_excluded,
             "test_clean_preaug": test_clean, "test_excluded_preaug": test_excluded}
    errors = check_invariants(train, validation, test, review_rows, lists)
    test_labels = {sid: label for sid, label, _ in test}
    counts = count_lists(lists, test_labels, overlap)
    errors += check_expected(counts)
    expected_check, partition = find_groups(train)
    audit = compute_audit(train, validation, test, [expected_check[next(iter(g))] for g in partition],
                          cleaned_dg, overlap)

    for stem in OUTPUTS:
        c = counts[stem]
        keys = LABELS if "_clean" in stem else sorted(k for k in c if k != "rows")
        say(f"{stem} {c['rows']} ({fmt_counts(c, keys)})")
    say(f"test overlap rows (non-exclusive): {fmt_counts(overlap, OVERLAP_KEYS)}; "
        f"validation overlap labels ({fmt_counts(audit['test']['validation_overlap_labels'], LABELS)}), "
        f"agree with validation: {audit['test']['validation_overlap_labels_agree']}")

    files = {f"{stem}.csv": render_csv(stem, lists[stem]) for stem in OUTPUTS}
    src_counts = {s: Counter(r[1] for r in src[s]) for s in SPLITS}

    def label_counts(ids, split):
        lab = {r[0]: r[1] for r in src[split]}
        return {k: sum(lab[i] == k for i in ids) for k in LABELS}

    splits_meta = {}
    for split, clean, excl, ridx in (("train", "train_clean", "train_excluded", 3),
                                     ("validation", "validation_clean", "validation_excluded", 3),
                                     ("test", "test_clean_preaug", "test_excluded_preaug", 1)):
        x_ids = [r[0] for r in lists[excl]]
        splits_meta[split] = {
            "source": {"rows": len(src[split]), "labels": {k: src_counts[split][k] for k in LABELS}},
            "clean": {"rows": len(lists[clean]), "labels": label_counts([r[0] for r in lists[clean]], split)},
            "excluded": {"rows": len(x_ids), "labels": label_counts(x_ids, split),
                         "by_reason": dict(sorted(Counter(r[ridx] for r in lists[excl]).items()))},
        }
    script_bytes = Path(__file__).read_bytes()
    meta = {
        "schema": SCHEMA,
        "trial": trial,
        "usable_for": "trial-only" if trial else "official",
        "script": {"path": rel(__file__), "sha256_lf": sha256_hex(script_bytes.replace(b"\r\n", b"\n"))},
        "unicode_version": unicodedata.unidata_version,
        "dataset": {
            "name": man["name"], "revision": man["revision"],
            "manifest": {"path": rel(manifest_path), "sha256": man["sha256"]},
            "files": {s: {"path": rel(info[s]["path"]), "sha256": info[s]["sha256"], "rows": len(src[s]),
                          "labels": {k: src_counts[s][k] for k in LABELS}} for s in SPLITS},
        },
        "review": {
            "path": rel(review_path), "sha256": sha256_hex(review_bytes), "rows": len(review_rows),
            "status": {k: sum(r["status"] == k for r in review_rows) for k in STATUSES},
            "groups": {k: len({r["group_id"] for r in review_rows if r["check"] == k}) for k in CHECKS},
            "decisions": {k: sum(r["decision"] == k for r in review_rows) for k in DECISIONS},
            "reasons": dict(sorted(Counter(r["reason"] for r in review_rows).items())),
        },
        "augmentation": None,
        "test_clean_stage": "pre-augmentation",
        "rules": RULES,
        "splits": splits_meta,
        "audit": audit,
        "outputs": {name: {"rows": data.count(b"\n") - 1, "sha256": sha256_hex(data)} for name, data in files.items()},
    }
    files[META] = render_meta(meta)
    for name, data in files.items():
        errors += check_grammar(name, data)
    return (None if errors else files), errors


if __name__ == "__main__":
    sys.exit(main())
