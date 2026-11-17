"""Compare a cases file with the test split by hash only (README 3.2 levels A-C, 3.3, 9; stage-4 design D3).

Run from Git Bash at the repo root:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/check_cases_vs_test.py
        (data/cases/train/cases.csv -> data/cases/train/test_overlap.json)
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/check_cases_vs_test.py --check
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/check_cases_vs_test.py \
        --cases data/cases/dev/cases.csv --out <file outside the repo>.json
Options: --cases PATH (csv with the columns case_id and text; default data/cases/train/cases.csv),
--out PATH (json; default <folder of the cases file>/test_overlap.json), --check (rebuild in memory and
compare with the file on disk; writes nothing).

Rules:
  - test.csv is read exactly once, by build_lists.load_test (size and SHA-256 checked against the pinned
    manifest before parsing; every test text is reduced there to its exact-key digest and its
    similar-sentence digests, so no test text reaches this module). main() activates the audit guard of
    build_lists the way build_lists.main does: any other open of a test.csv path and any write under
    datasets/ raises GuardError (T07). This process must NOT have data_lists.install_test_guard() installed
    (that hook ends the process at the test read): refused up front (T06). The cases file is read and
    validated before the manifest and test.csv are touched.
  - Cases file: UTF-8 without BOM, csv module (newline=''), a header with exactly one `case_id` and one
    `text` column (other columns are ignored; `label`, when present once, fills the label slot of the row
    shape and changes nothing), every record with the header's field count, at least one record. case_id
    must match CASE_ID_RE and be unique. The text is the csv field, unchanged (no strip); an empty text
    is allowed (it has no key and matches nothing). Every row counts, whatever its status: this is the pool.
  - Rows are (case_id, label, build_lists.digest(text), build_lists.sim_keys(text)), compared with the test
    rows returned by load_test (README 3.2 levels):
      exact           (A) the exact-key digests are equal (the raw text is the same).
      normalised_key  (B) different exact key, same punct key and same normalised key.
      punct_key       (B) different exact key, same punct key, different normalised key.
      one_word_edit   (C) punct-key word lists differ by one inserted, deleted or substituted word and the
                          shorter one has >= 5 words. Counted only: no IDs are written for this level.
    B and C come from build_lists.similar_pairs(cases, test, edit=True): a pair with equal exact keys is
    never reported as B, and a row with an empty punct key takes no part in B or C (the exact level still
    applies to it).
  - pool_sha256 = SHA-256 of "".join(f"{case_id},{text_sha256_12}\n") over all rows in ascending case_id
    order (text_sha256_12 = first 12 hex of SHA-256(text, UTF-8)); it depends on IDs and texts only, so a
    status change in the cases file keeps the output valid and a changed pool makes it stale.
  - Output json (UTF-8, ASCII, LF, sorted keys, no timestamp, no absolute path, no sentence text, no test
    digest): schema, tool {path, sha256_lf}, build_lists_sha256_lf, cases_file (repo-relative POSIX path,
    or the bare file name for a file outside the repo), n_cases, pool_sha256, test {rows, sha256 (of
    test.csv, from the manifest), revision}, matches [{case_id, text_sha256_12 (of the case text), level,
    test_ids [...]}] (levels A and B only; one entry per case and level, sorted by case_id then level,
    test_ids ascending), counts {exact, normalised_key, punct_key: cases with at least one test row at
    that level (= the matches entries of that level); one_word_edit_cases, one_word_edit_pairs}.
    Written to <name>.tmp, then os.replace. stdout: IDs and counts only, ASCII.
  - --check: the file on disk must equal the rebuilt bytes (it embeds the LF SHA-256 of this script and of
    build_lists.py, so editing either makes it stale).

Stop codes (printed as `FAIL <code> ...`, IDs, record numbers and counts only, then
`refused: N error(s); nothing written`, exit 1):
  T01 refused location: --cases under datasets/ or a test.csv path; --out under datasets/, a test.csv path,
      not a .json name, a folder, or the cases file itself
  T02 cases file missing or unreadable, UTF-8 BOM, not UTF-8, csv error
  T03 cases file header: not exactly one case_id and one text column (or more than one label column)
  T04 cases file: a record whose field count differs from the header (a blank line included), or no record
  T05 cases file: case_id empty or outside CASE_ID_RE, or duplicate
  T06 data_lists.install_test_guard() is installed in this process
  T07 the build_lists audit guard refused an operation
  D01-D08 manifest and test.csv checks of build_lists.load_manifest / load_test (passed through)
  K01 --check: file missing or different     W01 write failure
  X01 unexpected exception (type and line only; no summary line)
"""
import argparse
import codecs
import csv
import io
import json
import os
import re
import sys
import traceback
from pathlib import Path

import build_lists
import data_lists

ROOT = build_lists.ROOT
SCRIPT_REL = "scripts/check_cases_vs_test.py"
DEFAULT_CASES = ROOT / "data" / "cases" / "train" / "cases.csv"
DEFAULT_OUT_NAME = "test_overlap.json"
SCHEMA = 1
EXACT = "exact"                                # README 3.2 level A
B_LEVELS = ("normalised_key", "punct_key")     # level B, as named by build_lists.similar_pairs
C_LEVEL = "one_word_edit"                      # level C, counted only
LEVELS = (EXACT,) + B_LEVELS                   # levels listed in "matches", in this order per case
CASE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}")
REQUIRED_COLUMNS = ("case_id", "text")
SHOWN_IDS = 10


class Refused(Exception):
    """A stop: .errors holds the FAIL messages (IDs, record numbers and counts only)."""

    def __init__(self, errors):
        super().__init__(f"{len(errors)} error(s)")
        self.errors = list(errors)


def _ascii(s):
    return str(s).encode("ascii", "backslashreplace").decode("ascii")


def say(msg):
    print(_ascii(msg), flush=True)


def _fmt_nums(nums):
    shown = ", ".join(str(n) for n in nums[:SHOWN_IDS])
    more = f", ... (+{len(nums) - SHOWN_IDS} more)" if len(nums) > SHOWN_IDS else ""
    return f"{len(nums)} ({shown}{more})"


def shown_path(path):
    """Repo-relative POSIX path, or the bare file name for a path outside the repo (never an absolute path)."""
    r = build_lists.rel(path)
    return r if r is not None else Path(path).name


# ---------------------------------------------------------------- locations

def _is_test_path(path):
    return build_lists._is_test_csv(str(path), build_lists._real(str(path)))


def location_errors(cases, out):
    """T01 messages for the two paths; checked before any file is read."""
    errors = []
    if build_lists._under(cases, build_lists.DATASETS):
        errors.append("T01 --cases is under datasets/ (raw data is read by build_lists.load_test only)")
    if _is_test_path(cases):
        errors.append("T01 --cases is a test.csv path")
    if build_lists._under(out, build_lists.DATASETS):
        errors.append("T01 --out is under datasets/ (raw data stays read-only)")
    if _is_test_path(out):
        errors.append("T01 --out is a test.csv path")
    if not out.name.lower().endswith(".json"):
        errors.append("T01 --out must be a .json file name")
    if out.is_dir():
        errors.append("T01 --out is a folder")
    if build_lists._same(out, cases):
        errors.append("T01 --out is the cases file")
    return errors


# ---------------------------------------------------------------- cases file

def read_cases(path):
    """[(case_id, label, text)] in file order (T02-T05). text is the csv field, unchanged."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise Refused([f"T02 cases file: cannot read ({build_lists.exc_name(exc)})"])
    if data.startswith(codecs.BOM_UTF8):
        raise Refused(["T02 cases file: starts with a UTF-8 BOM"])
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise Refused(["T02 cases file: not valid UTF-8 (message suppressed)"])
    recs = []
    try:
        reader = csv.reader(io.StringIO(text, newline=""))
        header = next(reader, None)
        for rec in reader:
            recs.append(rec)
    except csv.Error as exc:
        raise Refused([f"T02 cases file: {build_lists.exc_name(exc)} after record {len(recs)} (message suppressed)"])
    if header is None:
        raise Refused(["T03 cases file: empty (no header)"])
    errors = [f"T03 cases file: the header needs exactly one `{c}` column (found {header.count(c)})"
              for c in REQUIRED_COLUMNS if header.count(c) != 1]
    if header.count("label") > 1:
        errors.append(f"T03 cases file: the header has {header.count('label')} `label` columns")
    if errors:
        raise Refused(errors)
    ci, ti = header.index("case_id"), header.index("text")
    li = header.index("label") if "label" in header else None
    rows, seen, bad_fields, bad_ids, dups = [], set(), [], [], set()
    for n, rec in enumerate(recs, 1):  # n = 1-based record number, header excluded
        if len(rec) != len(header):
            bad_fields.append(n)
            continue
        cid = rec[ci]
        if not CASE_ID_RE.fullmatch(cid):
            bad_ids.append(n)  # the value is never printed: it is not a checked ID
            continue
        if cid in seen:
            dups.add(cid)
        seen.add(cid)
        rows.append((cid, rec[li] if li is not None else "", rec[ti]))
    if bad_fields:
        errors.append(f"T04 cases file: field count differs from the header ({len(header)}) in record(s) "
                      f"{_fmt_nums(bad_fields)}")
    if not recs:
        errors.append("T04 cases file: no record")
    if bad_ids:
        errors.append(f"T05 cases file: case_id empty or not matching {CASE_ID_RE.pattern} in record(s) "
                      f"{_fmt_nums(bad_ids)}")
    if dups:
        errors.append(f"T05 cases file: duplicate case_id {build_lists.fmt_ids(dups, SHOWN_IDS)}")
    if errors:
        raise Refused(errors)
    return rows


def pool_sha256(pairs):
    """SHA-256 of "".join(f"{case_id},{text_sha256_12}\n") over (case_id, text_sha256_12) in case_id order."""
    return build_lists.sha256_hex("".join(f"{cid},{h}\n" for cid, h in sorted(pairs)).encode("ascii"))


# ---------------------------------------------------------------- comparison

def compare(cases, test):
    """(found, counts) for case rows x test rows; both are (id, label, exact digest, sim_keys).

    found = {(case_id, level): {test IDs}} for the levels A and B; level C is counted only.
    """
    by_exact = {}
    for sid, _, exact, _ in test:
        by_exact.setdefault(exact, set()).add(sid)
    found = {}
    for cid, _, exact, _ in cases:
        if exact in by_exact:
            found[(cid, EXACT)] = set(by_exact[exact])
    edit_cases, edit_pairs = set(), 0
    for (cid, sid), level in build_lists.similar_pairs(cases, test, edit=True).items():
        if level == C_LEVEL:
            edit_cases.add(cid)
            edit_pairs += 1
        elif level in B_LEVELS:
            found.setdefault((cid, level), set()).add(sid)
        else:
            raise RuntimeError("build_lists.similar_pairs returned an unknown level")
    counts = {lv: sum(1 for _, level in found if level == lv) for lv in LEVELS}
    counts["one_word_edit_cases"] = len(edit_cases)
    counts["one_word_edit_pairs"] = edit_pairs
    return found, counts


def render_json(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def build(cases_path):
    """(output bytes, doc, {level: [case IDs]}, number of cases without a punct key). Raises Refused.

    The cases file is validated first; test.csv is read once, inside build_lists.load_test.
    """
    raw = read_cases(cases_path)
    info, manifest, errors = build_lists.load_manifest(build_lists.DEFAULT_MANIFEST)
    if errors:
        raise Refused(errors)
    t = info["test"]
    test, errors = build_lists.load_test(t["path"], t["sha256"], t["size"], t["expected"])
    if errors:
        raise Refused(errors)
    cases = [(cid, label, build_lists.digest(text), build_lists.sim_keys(text)) for cid, label, text in raw]
    h12 = {cid: exact[:12] for cid, _, exact, _ in cases}  # = build_lists.h12(text)
    found, counts = compare(cases, test)
    order = sorted(found, key=lambda k: (k[0], LEVELS.index(k[1])))
    doc = {
        "schema": SCHEMA,
        "tool": {"path": SCRIPT_REL, "sha256_lf": data_lists.sha256_lf(__file__)},
        "build_lists_sha256_lf": data_lists.sha256_lf(build_lists.__file__),
        "cases_file": shown_path(cases_path),
        "n_cases": len(cases),
        "pool_sha256": pool_sha256(h12.items()),
        "test": {"rows": len(test), "sha256": t["sha256"], "revision": manifest["revision"]},
        "matches": [{"case_id": cid, "text_sha256_12": h12[cid], "level": level,
                     "test_ids": sorted(found[(cid, level)])} for cid, level in order],
        "counts": counts,
    }
    by_level = {lv: [cid for cid, level in order if level == lv] for lv in LEVELS}
    no_punct_key = sum(1 for _, _, _, keys in cases if keys[1] is None)
    return render_json(doc), doc, by_level, no_punct_key


# ---------------------------------------------------------------- write / check

def write_file(path, data):
    """<name>.tmp (exclusive create), then os.replace. Returns None or a W01 error."""
    tmp = path.with_name(path.name + ".tmp")
    current = "creating the folder"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        current = f"writing {path.name}.tmp"
        tmp.unlink(missing_ok=True)
        with open(tmp, "xb") as f:
            f.write(data)
        current = f"replacing {path.name}"
        os.replace(tmp, path)
        return None
    except Exception as exc:
        return f"W01 {build_lists.exc_name(exc)} while {current}; rerun after fixing the cause"
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def check_file(path, data):
    """--check: K01 messages when the file on disk is missing or differs from the rebuilt bytes."""
    name = path.name
    try:
        old = path.read_bytes()
    except OSError:
        return [f"K01 --check: {name} is missing"]
    if old == data:
        return []
    try:
        old_doc = json.loads(old.decode("utf-8"))
    except ValueError:
        return [f"K01 --check: {name} differs from the rebuilt bytes (not valid JSON)"]
    keys = build_lists._diff_keys(old_doc, json.loads(data))
    if not keys:
        return [f"K01 --check: {name} differs from the rebuilt bytes in formatting only"]
    msg = f"K01 --check: {name} differs from the rebuilt bytes at keys {build_lists.fmt_ids(keys, SHOWN_IDS)}"
    if {"tool.sha256_lf", "build_lists_sha256_lf"} & set(keys):
        msg += "; this script or build_lists.py changed: rerun without --check"
    return [msg]


# ---------------------------------------------------------------- main

def _parse_args(argv):
    p = argparse.ArgumentParser(description="Compare a cases file with test by hash only (README 3.2 levels A-C).")
    p.add_argument("--cases", help="cases csv with the columns case_id and text (default data/cases/train/cases.csv)")
    p.add_argument("--out", help="output json (default <folder of the cases file>/test_overlap.json)")
    p.add_argument("--check", action="store_true", help="rebuild in memory and compare with the file on disk")
    return p.parse_args(argv)


def _run(args):
    cases_path = Path(args.cases).resolve() if args.cases else DEFAULT_CASES
    out = Path(args.out).resolve() if args.out else cases_path.parent / DEFAULT_OUT_NAME
    errors = location_errors(cases_path, out)
    if errors:
        raise Refused(errors)
    if data_lists._HOOK["installed"]:
        raise Refused(["T06 data_lists.install_test_guard() is installed in this process: it would end the "
                       "process at the test read; run this script as its own process"])
    data, doc, by_level, no_punct_key = build(cases_path)
    counts = doc["counts"]
    say(f"cases: {doc['n_cases']} row(s) in {doc['cases_file']}, pool_sha256 {doc['pool_sha256'][:12]}, "
        f"{no_punct_key} without a punct key (exact level only)")
    say(f"test: {doc['test']['rows']} row(s), revision {str(doc['test']['revision'])[:12]}")
    for lv in LEVELS:
        ids = by_level[lv]
        more = f", ... (+{len(ids) - SHOWN_IDS} more)" if len(ids) > SHOWN_IDS else ""
        say(f"{lv}: {counts[lv]} case(s)" + (f" ({', '.join(ids[:SHOWN_IDS])}{more})" if ids else ""))
    say(f"{C_LEVEL}: {counts['one_word_edit_cases']} case(s), {counts['one_word_edit_pairs']} pair(s) "
        "(counted only)")
    if args.check:
        errors = check_file(out, data)
        if errors:
            raise Refused(errors)
        say(f"OK --check: {shown_path(out)} equals the rebuilt bytes")
        return 0
    err = write_file(out, data)
    if err:
        raise Refused([err])
    say(f"OK wrote {shown_path(out)} ({len(doc['matches'])} match entries)")
    return 0


def main(argv=None):
    args = _parse_args(argv)
    # The guard of build_lists, activated as in build_lists.main; never data_lists.install_test_guard().
    build_lists._install_guard()
    build_lists._GUARD.update(active=True, test_reads=0)
    try:
        return _run(args)
    except Refused as exc:
        for e in exc.errors:
            say("FAIL " + e)
        say(f"refused: {len(exc.errors)} error(s); nothing written")
        return 1
    except build_lists.GuardError as exc:
        say(f"FAIL T07 build_lists audit hook {exc}")
        say("refused: 1 error(s); nothing written")
        return 1
    except Exception as exc:
        lines = [ln for frame, ln in traceback.walk_tb(exc.__traceback__)
                 if os.path.basename(frame.f_code.co_filename) == os.path.basename(__file__)]
        say(f"FAIL X01 unexpected {build_lists.exc_name(exc)} at check_cases_vs_test.py:{lines[-1] if lines else '?'} "
            "(message suppressed)")
        return 1
    finally:
        build_lists._GUARD.update(active=False, test_reads=0)


if __name__ == "__main__":
    sys.exit(main())
