"""Common-list access for training and evaluation (README 3.2, 7.1).

Standard library plus scripts/build_lists.py (frozen; imported for its raw-split
readers and guard helpers, never for load_test). Nothing here prints or writes
sentence text: errors carry sentence IDs, file names and counts only.

Rules:
  - install_test_guard() installs an audit hook (once per process) that ends
    the process with exit code 1 on any open of a path whose basename is
    test.csv (case-insensitive, also after resolving; build_lists._is_test_csv
    and _real) and on shutil.copyfile / CopyFile2 from such a path. Training
    never reads test.csv; build_lists.py --check reads its digests in its own
    process (a subprocess of verify_lists).
  - verify_lists("official"): data/processed/ list files and train_review.csv
    git-tracked and unmodified, lists_meta.json "trial" false, its review
    SHA-256 equal to the current train_review.csv, every list file SHA-256
    equal to lists_meta "outputs", build_lists.py SHA-256 (LF-normalised)
    equal to lists_meta "script", then `build_lists.py --check` as a
    subprocess with exit 0 (README 7.1 common lists). require_tracked=False
    (stage-1 smoke) skips only the git checks.
  - verify_lists("trial"): data/processed/trial/ with "trial" true and the
    same review, file and script SHA checks (no git or --check). Trial lists
    are for stage-1 smoke runs only.
  - load_rows(): train and validation re-parsed from the pinned raw files with
    build_lists.load_manifest / read_verified / parse_split (sentence ID = csv
    record index). test is never loaded.
  - load_training_data(): joins the lists with the rows; every ID must exist,
    label and text_sha256_12 must match, IDs strictly ascending, and
    clean + excluded must partition the split (disjoint, union = all rows).
    Validation is predicted on all rows with an in_clean flag.
"""
import hashlib
import json
import os
import subprocess
import sys

import build_lists

ROOT = build_lists.ROOT
LIST_FILES = ("train_clean.csv", "train_excluded.csv", "validation_clean.csv", "validation_excluded.csv",
              "test_clean_preaug.csv", "test_excluded_preaug.csv")
META = build_lists.META
REVIEW = "data/processed/train_review.csv"
BUILD_LISTS = "scripts/build_lists.py"
TRAIN_VAL_HEADER = "sentence_id,label,text_sha256_12"
CHECK_TIMEOUT_S = 600

_HOOK = {"installed": False}


class ListError(Exception):
    """List verification or join failure; .errors holds the messages (IDs and counts only)."""

    def __init__(self, errors):
        super().__init__(f"{len(errors)} list error(s)")
        self.errors = list(errors)


# ---------------------------------------------------------------- test.csv guard

def _guard_hook(event, args):
    if event == "open" or event in ("shutil.copyfile", "_winapi.CopyFile2"):
        p = build_lists._arg_path(args, 0)
        if p is not None and build_lists._is_test_csv(p, build_lists._real(p)):
            sys.stdout.write(f"FAIL test.csv guard: refused {event} of a test.csv path; "
                             "this process never reads test.csv\n")
            sys.stdout.flush()
            os._exit(1)


def install_test_guard():
    if not _HOOK["installed"]:
        sys.addaudithook(_guard_hook)
        _HOOK["installed"] = True


# ---------------------------------------------------------------- hashes and git

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_lf(path):
    """SHA-256 of a text file with CRLF normalised to LF (stable across autocrlf checkouts)."""
    with open(path, "rb") as f:
        return sha256_bytes(f.read().replace(b"\r\n", b"\n"))


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True)


def git_tracked(root, paths):
    """Subset of the repo-relative POSIX paths that git tracks (empty when root is not a repo)."""
    r = git(root, "ls-files", "-z", "--", *paths)
    if r.returncode != 0:
        return set()
    return {p.decode("utf-8") for p in r.stdout.split(b"\0") if p}


def git_status(root, paths=(), untracked=False):
    """[(XY, path)] from `git status --porcelain`; None when git fails (e.g. not a repo)."""
    r = git(root, "status", "--porcelain=v1", "-z",
            "--untracked-files=" + ("all" if untracked else "no"), "--", *paths)
    if r.returncode != 0:
        return None
    out, parts, i = [], r.stdout.split(b"\0"), 0
    while i < len(parts):
        entry = parts[i]
        i += 1
        if not entry:
            continue
        xy, path = entry[:2].decode("ascii"), entry[3:].decode("utf-8")
        if xy[0] in "RC":
            i += 1  # the original path of a rename or copy follows
        out.append((xy, path))
    return out


def git_head(root):
    r = git(root, "rev-parse", "HEAD")
    return r.stdout.decode("ascii").strip() if r.returncode == 0 else None


# ---------------------------------------------------------------- list verification

def list_dir(mode, root=ROOT):
    base = root / "data" / "processed"
    return base / build_lists.TRIAL_DIR if mode == "trial" else base


def verify_lists(mode, root=ROOT, *, require_tracked=True):
    """Return (errors, info). info is the run.json lists block (None on errors).

    require_tracked=False skips only the official-mode git checks (L07; stage-1
    smoke runs before the lists are committed); every byte, SHA and meta check
    and build_lists.py --check still run. info["git_tracked_checked"] records it.
    """
    if mode not in ("official", "trial"):
        raise ValueError(mode)
    d = list_dir(mode, root)
    shown = d.relative_to(root).as_posix()
    names = LIST_FILES + (META,)
    missing = [n for n in names if not (d / n).is_file()]
    if missing:
        return [f"L01 {mode} lists: missing in {shown}/: {', '.join(missing)}"], None
    errors = []
    tracked_checked = mode == "official" and require_tracked
    if tracked_checked:
        rels = [f"{shown}/{n}" for n in names] + [REVIEW]
        untracked = sorted(set(rels) - git_tracked(root, rels))
        if untracked:
            errors.append(f"L07 official lists: not tracked by git: {', '.join(untracked)}")
        dirty = git_status(root, rels)
        if dirty is None:
            errors.append("L07 official lists: git status failed")
        elif dirty:
            errors.append(f"L07 official lists: modified in the working tree or index: "
                          f"{', '.join(p for _, p in dirty)}")
    meta_bytes = (d / META).read_bytes()
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
        meta_trial = meta["trial"]
        meta_review = meta["review"]["sha256"]
        meta_outputs = {n: meta["outputs"][n]["sha256"] for n in LIST_FILES}
        meta_script = meta["script"]["sha256_lf"]
        revision = meta["dataset"]["revision"]
        train_rows = meta["outputs"]["train_clean.csv"]["rows"]
    except (ValueError, KeyError, TypeError):
        return errors + [f"L02 {shown}/{META}: unreadable or missing keys"], None
    if meta_trial is not (mode == "trial"):
        errors.append(f"L03 {shown}/{META}: trial is {json.dumps(meta_trial)}, {mode} lists need "
                      f"{json.dumps(mode == 'trial')} (trial lists are never used for stage 2)")
    try:
        review_sha = sha256_bytes((root / REVIEW).read_bytes())
    except OSError:
        review_sha = None
        errors.append(f"L04 {REVIEW}: cannot read")
    if review_sha is not None and review_sha != meta_review:
        errors.append(f"L04 {REVIEW}: sha256 {review_sha[:12]} != lists_meta review {meta_review[:12]} "
                      "(the review changed after the lists were built: rebuild them)")
    files = {}
    for n in LIST_FILES:
        files[n] = sha256_bytes((d / n).read_bytes())
        if files[n] != meta_outputs[n]:
            errors.append(f"L05 {shown}/{n}: sha256 {files[n][:12]} != lists_meta {meta_outputs[n][:12]}")
    try:
        script_sha = sha256_lf(root / BUILD_LISTS)
    except OSError:
        script_sha = None
        errors.append(f"L06 {BUILD_LISTS}: cannot read")
    if script_sha is not None and script_sha != meta_script:
        errors.append(f"L06 {BUILD_LISTS}: sha256_lf {script_sha[:12]} != lists_meta script {meta_script[:12]} "
                      "(script changed after the lists were built: rebuild them)")
    check = "not run (trial lists)"
    if mode == "official" and not errors:
        r = subprocess.run([sys.executable, str(root / BUILD_LISTS), "--check"], cwd=str(root),
                           capture_output=True, timeout=CHECK_TIMEOUT_S,
                           env={**os.environ, "PYTHONUTF8": "1"})
        if r.returncode != 0:
            fails = [ln for ln in r.stdout.decode("ascii", "replace").splitlines() if ln.startswith("FAIL")]
            errors.append(f"L08 {BUILD_LISTS} --check exit {r.returncode}: " + " | ".join(fails[:5]))
        else:
            check = "passed"
    if errors:
        return errors, None
    return [], {
        "mode": mode, "dir": shown, "files_sha256": files, "lists_meta_sha256": sha256_bytes(meta_bytes),
        "review_sha256": review_sha, "build_lists_sha256_lf": script_sha, "dataset_revision": revision,
        "train_clean_rows": train_rows, "build_lists_check": check, "git_tracked_checked": tracked_checked,
    }


# ---------------------------------------------------------------- rows and joins

def load_rows(manifest=build_lists.DEFAULT_MANIFEST):
    """{split: {sid: (label str, text)}} for train and validation; raises ListError."""
    info, _, errors = build_lists.load_manifest(manifest)
    if errors:
        raise ListError(errors)
    rows = {}
    for split in ("train", "validation"):
        s = info[split]
        data, errs = build_lists.read_verified(s["path"], s["sha256"], s["size"], f"{split}.csv")
        if data is None:
            raise ListError(errs)
        parsed, errs = build_lists.parse_split(data, split, s["expected"])
        if errs:
            raise ListError(errs)
        rows[split] = {sid: (label, text) for sid, label, text in parsed}
    return rows


def read_list(path):
    """[(sid, label, text_sha256_12)] from a train/validation list file (first three columns)."""
    lines = path.read_bytes().decode("ascii").split("\n")
    if not lines[0].startswith(TRAIN_VAL_HEADER) or lines[-1] != "":
        raise ListError([f"J00 {path.name}: header or final LF differs"])
    return [tuple(ln.split(",")[:3]) for ln in lines[1:-1]]


def join(list_rows, rows, name):
    """Map list rows to (sid, label int, text); returns (joined, errors)."""
    out, errors, prev = [], [], None
    for sid, label, h in list_rows:
        if prev is not None and not sid > prev:
            errors.append(f"J01 {name}: IDs not strictly ascending at {sid}")
        prev = sid
        r = rows.get(sid)
        if r is None:
            errors.append(f"J02 {name}: {sid} is not a row of the split")
        elif r[0] != label:
            errors.append(f"J03 {name}: {sid} label differs from the raw split")
        elif build_lists.h12(r[1]) != h:
            errors.append(f"J04 {name}: {sid} text_sha256_12 differs from the raw text")
        else:
            out.append((sid, int(label), r[1]))
    return out, errors


def load_training_data(directory, rows):
    """{"train": [(sid, label, text)], "validation": [(sid, label, text, in_clean)]}; raises ListError."""
    errors, out = [], {}
    for split in ("train", "validation"):
        clean, e1 = join(read_list(directory / f"{split}_clean.csv"), rows[split], f"{split}_clean.csv")
        excl, e2 = join(read_list(directory / f"{split}_excluded.csv"), rows[split], f"{split}_excluded.csv")
        errors += e1 + e2
        c_ids, x_ids = {r[0] for r in clean}, {r[0] for r in excl}
        if c_ids & x_ids:
            errors.append(f"J05 {split}: {len(c_ids & x_ids)} IDs in both clean and excluded lists")
        if (c_ids | x_ids) != set(rows[split]):
            errors.append(f"J05 {split}: clean + excluded cover {len(c_ids | x_ids)} of {len(rows[split])} rows")
        if split == "train":
            out[split] = clean
        else:
            out[split] = sorted([r + (True,) for r in clean] + [r + (False,) for r in excl])
    if errors:
        raise ListError(errors)
    return out
