"""Download and verify the pinned external evaluation sources (README 3.4 EXT-01, 7.1; stage-4 design D7).

Standard library plus scripts/build_lists.py (path and hash helpers) and
scripts/data_lists.py (test.csv guard). File contents are never parsed, printed
or logged: bytes are measured (size), hashed (SHA-256, git blob SHA-1) and
written unchanged (EXT-02; parquet rows are not parsed in this environment).
Output lines carry names, sizes and hashes only. main() installs
data_lists.install_test_guard() first: this process never opens a test.csv path.

Sources (SOURCES; every value is a pin, nothing is taken from the network on trust):
  deepset    deepset/prompt-injections @ 4f61ecb0...: README.md and the train and test parquet files
  notinject  leolee99/NotInject @ 847ae76c...: README.md and the NotInject_one/two/three parquet files
  Each adopted file is pinned by path, size, SHA-256 and Hub tree oid (the git
  blob SHA-1 of the file, or of its Git LFS pointer for an LFS file). Files of
  the revision that are not downloaded are pinned too (path, oid, reason).

Modes:
  --download  online; refuses when HF_HUB_OFFLINE is set. Per source:
              1. GET https://huggingface.co/api/datasets/<repo>/tree/<revision>?recursive=true
                 and compare it with the pins: every adopted file has the pinned
                 size and oid (LFS files also the pinned SHA-256 and size as
                 lfs.oid / lfs.size), every excluded file has the pinned oid,
                 and the revision holds no other file;
              2. GET https://huggingface.co/datasets/<repo>/resolve/<revision>/<path>
                 for every adopted file (at most the pinned size is accepted)
                 and compare size, SHA-256 and git blob SHA-1 with the pins.
              Only when every selected source passes are the raw files written
              to data/external/<name>/<path> and download_manifest.json to
              data/external/<name>/. An existing identical file is left as it
              is; an existing different file is never overwritten.
              Requests carry one header set by this script, a fixed generic
              User-Agent; no token, cookie or account/project information.
  (default)   offline check, no network: every pinned raw file exists with the
              pinned size, SHA-256 and git blob SHA-1, download_manifest.json
              equals a fresh rendering byte for byte, a source folder holds no
              other file or folder (listed by name, never opened), the base
              folder holds no folder that is not a pinned source, and the
              "sources" block of data/external/external_rules.json agrees with
              the pins.
  Options: --source NAME [NAME ...] (default: all), --out DIR and --rules PATH (tests).

download_manifest.json has the shape of datasets/s-labs-prompt-injection/
download_manifest.json without its timestamps and "splits" block (rows are not
parsed): dataset, dataset_url, revision, license, license_note,
download_method, excluded_from_download, verification, all_checks_passed,
files (path, source_url, size_bytes, sha256, and git_blob_sha1 +
matches_hub_blob, or for LFS files lfs_pointer_git_blob_sha1 + matches_hub_lfs
+ matches_hub_blob), note. It is a function of the pins alone and is written
only after the checks passed, so the same pins always give the same bytes
(ASCII, LF, no timestamp, no absolute path).

Refusal codes (exit 1, "FAIL <code> ..." lines, "refused: N error(s); nothing written"):
  X01 environment, arguments or pin table: --download with HF_HUB_OFFLINE set; --out under datasets/;
      --rules under datasets/ or a test.csv path; --rules with --download; a pin that is malformed
      (hex length, unsafe or repeated path, a test.csv name, files not sorted by path) or an LFS oid
      that does not follow from the pinned SHA-256 and size
  X02 network: HTTP status other than 200, URL or socket error, or a body longer than expected
  X03 Hub tree listing unreadable: not a JSON list of entries (or nested too deeply), a repeated path,
      or a paginated listing
  X04 Hub tree entry of an adopted file missing or different from the pin (size, oid, LFS SHA-256/size)
  X05 Hub tree file set differs: a file that is neither adopted nor excluded, or an excluded file
      that is missing or has a different oid
  X06 downloaded bytes differ from the pin (size, SHA-256 or git blob SHA-1)
  X07 local raw file: --download finds different bytes already there; the check finds it missing or
      different, finds a file or folder in a source folder that is not pinned, or finds a folder in
      the base folder that is not a pinned source
  X08 local download_manifest.json: --download finds different bytes already there; the check finds it
      missing or different
  X09 external_rules.json (check): unreadable, not LF UTF-8 JSON (or nested too deeply), or its
      "sources" block differs from the pins (dataset, dataset_url, revision, license, dir, manifest,
      manifest_sha256, files, split_files)
  X10 a file could not be written (files written before the error stay; run the check afterwards)

Run from Git Bash at the repo root:
    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/fetch_external.py --download
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/fetch_external.py
"""
import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import build_lists
import data_lists

ROOT = build_lists.ROOT
OUT_REL = "data/external"
DEFAULT_OUT = ROOT / "data" / "external"
RULES_NAME = "external_rules.json"
MANIFEST = "download_manifest.json"
HUB = "https://huggingface.co"
USER_AGENT = "python-urllib"  # generic client name; no version, account or project information
TIMEOUT_S = 60
TREE_LIMIT = 1 << 20  # bytes accepted for one tree listing
OFFLINE_VALUES = ("1", "ON", "YES", "TRUE")  # the values huggingface_hub reads as HF_HUB_OFFLINE set

DOWNLOAD_METHOD = ("urllib from Hugging Face resolve URLs pinned to revision "
                   "(scripts/fetch_external.py --download)")
VERIFICATION = ("size and SHA-256 compared with the values pinned in scripts/fetch_external.py; git blob SHA-1 "
                "(for LFS files: LFS SHA-256, LFS size and the git blob SHA-1 of the LFS pointer) and the file "
                "set of the revision compared with the Hugging Face tree API at revision; rows not parsed "
                "(parquet is not parsed in this environment)")
NOTE = ("Raw files are byte-identical to the Hub. Do not edit them; write converted or processed data "
        "elsewhere. External rows are used for the final evaluation only (README 3.4, EXT-02).")

NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
PATH_RE = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")
SHA1_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _file(path, size, sha256, oid, lfs):
    return {"path": path, "size": size, "sha256": sha256, "oid": oid, "lfs": lfs}


def _exclude(path, oid, reason):
    return {"path": path, "oid": oid, "reason": reason}


# Pins. "files" are sorted by path. Sizes, SHA-256 and oids are the Hub values at the revision
# (tree API, 2026-11-17); --download compares them with the Hub again and with the downloaded bytes.
SOURCES = {
    "deepset": {
        "dataset": "deepset/prompt-injections",
        "revision": "4f61ecb038e9c3fb77e21034b22511b523772cdd",
        "license": "apache-2.0",
        "license_note": ("data card metadata at the revision: license apache-2.0; the dataset_info block of "
                         "the same card names cc-by-4.0 (README 3.4)"),
        "files": (
            _file("README.md", 500,
                  "d90b4518dfe06154deeec938243d1ec9119bdfbfd2105aee9cf1567999764b94",
                  "d3a63fc0a09231cd1563afb14facfd9d55adcd03", False),
            _file("data/test-00000-of-00001-701d16158af87368.parquet", 10892,
                  "39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e",
                  "8bf9d00201127a68fc3b0cf1e8da9ecff8b221e7", True),
            _file("data/train-00000-of-00001-9564e8b05b4757ab.parquet", 40323,
                  "2e10bc7ab30f542c97e4e83e2a5683000b5057d25ec10908784c631d44124c04",
                  "551b244bd3dc2e271865caf32d209bb7f6e249ba", True),
        ),
        "excluded": (
            _exclude(".gitattributes", "f4f3945bd7150d3e12988485c42da1f8c29c59f8", "Hub LFS config, not data"),
            _exclude("gitattributes.txt", "f4f3945bd7150d3e12988485c42da1f8c29c59f8",
                  "same git blob as .gitattributes, not data"),
        ),
    },
    "notinject": {
        "dataset": "leolee99/NotInject",
        "revision": "847ae76cf8fea5ed325429e569ae8cfef022d2e0",
        "license": "mit",
        "license_note": "data card metadata at the revision: license mit (README 3.4)",
        "files": (
            _file("README.md", 2973,
                  "ffaf685021522a526f94af2047e5099ec6b7439a462c67e0ecaca0d631a0d43f",
                  "87958379fac69a6ea287107a5c5e48baf1ebe1d8", False),
            _file("data/NotInject_one-00000-of-00001.parquet", 10859,
                  "5b1d3a8ec341205305efd8c6d2cabeb5ba0f594dd8726b08ee7fde1a46ddc6ec",
                  "4dc626d1c0a07b8604baf934b23daaee09785034", True),
            _file("data/NotInject_three-00000-of-00001.parquet", 13009,
                  "d5c685b123570c092a0f374ba6bdf324b2026d905a2751a243eb2bb360d411a8",
                  "8b0b7cf318071c63df75a63bca9a6fdee75b44e8", True),
            _file("data/NotInject_two-00000-of-00001.parquet", 11183,
                  "a5a8c1b3b851b0b5acc3b6caf8bfee26d7da631b2967dd31ca90358df8883feb",
                  "68d9cb25f3886a4587dbbd391ab8a71b7dbad3e1", True),
        ),
        "excluded": (
            _exclude(".gitattributes", "55cab133643a2a73e083373d2106533678d0edd5", "Hub LFS config, not data"),
            _exclude("data/test-00000-of-00001.parquet", "8b0b7cf318071c63df75a63bca9a6fdee75b44e8",
                  "same git blob (LFS pointer) as data/NotInject_three-00000-of-00001.parquet, "
                  "not a split of the data card"),
            _exclude("data/train-00000-of-00001.parquet", "4dc626d1c0a07b8604baf934b23daaee09785034",
                  "same git blob (LFS pointer) as data/NotInject_one-00000-of-00001.parquet, "
                  "not a split of the data card"),
            _exclude("data/validation-00000-of-00001.parquet", "68d9cb25f3886a4587dbbd391ab8a71b7dbad3e1",
                  "same git blob (LFS pointer) as data/NotInject_two-00000-of-00001.parquet, "
                  "not a split of the data card"),
        ),
    },
}


# ---------------------------------------------------------------- hashes and URLs

def git_blob_sha1(data):
    """SHA-1 that git gives a blob with these bytes (the Hub tree "oid" of a plain file)."""
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def lfs_pointer(sha256, size):
    """Bytes of the Git LFS pointer file (spec v1) that git stores for an LFS file."""
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {size}\n".encode("ascii")


def stored_oid(data, lfs):
    """Hub tree oid that follows from the file bytes: blob SHA-1 of the file, or of its LFS pointer."""
    if lfs:
        return git_blob_sha1(lfs_pointer(build_lists.sha256_hex(data), len(data)))
    return git_blob_sha1(data)


def dataset_url(src):
    return f"{HUB}/datasets/{src['dataset']}"


def source_url(src, path):
    return f"{HUB}/datasets/{src['dataset']}/resolve/{src['revision']}/{path}"


def tree_url(src):
    return f"{HUB}/api/datasets/{src['dataset']}/tree/{src['revision']}?recursive=true"


def _safe_path(path):
    return (isinstance(path, str) and PATH_RE.fullmatch(path) is not None
            and all(seg not in (".", "..") for seg in path.split("/")))


def validate_pins(sources):
    """X01 messages for a malformed or inconsistent pin table (no file and no network is used)."""
    errors = []
    for name, src in sources.items():
        if not (isinstance(name, str) and NAME_RE.fullmatch(name)):
            errors.append(f"X01 pins: source name {name!a} is not lower-case letters, digits and hyphens")
            continue
        if not REPO_RE.fullmatch(str(src.get("dataset"))):
            errors.append(f"X01 pins {name}: dataset is not <owner>/<name>")
        if not SHA1_RE.fullmatch(str(src.get("revision"))):
            errors.append(f"X01 pins {name}: revision is not a 40-character commit hash")
        for key in ("license", "license_note"):
            if not (isinstance(src.get(key), str) and src[key] and src[key].isascii()):
                errors.append(f"X01 pins {name}: {key} is not a non-empty ASCII string")
        files, excluded = src.get("files") or (), src.get("excluded") or ()
        if not files:
            errors.append(f"X01 pins {name}: no adopted file")
        seen = set()
        for f in tuple(files) + tuple(excluded):
            path = f.get("path")
            if not _safe_path(path):
                errors.append(f"X01 pins {name}: unsafe path {path!a}")
                continue
            if build_lists._is_test_csv(path) or path.lower() == MANIFEST:  # also "test.csv." (Windows alias)
                errors.append(f"X01 pins {name} {path}: reserved file name")
            if path.lower() in seen:
                errors.append(f"X01 pins {name} {path}: path repeated")
            seen.add(path.lower())
            if not SHA1_RE.fullmatch(str(f.get("oid"))):
                errors.append(f"X01 pins {name} {path}: oid is not a 40-character SHA-1")
        for f in files:
            path = f.get("path")
            if not _safe_path(path):
                continue
            size, sha, lfs = f.get("size"), str(f.get("sha256")), f.get("lfs")
            if not (isinstance(size, int) and not isinstance(size, bool) and size > 0):
                errors.append(f"X01 pins {name} {path}: size is not a positive integer")
            elif not SHA256_RE.fullmatch(sha):
                errors.append(f"X01 pins {name} {path}: sha256 is not 64 hex characters")
            elif not isinstance(lfs, bool):
                errors.append(f"X01 pins {name} {path}: lfs is not true or false")
            elif lfs and git_blob_sha1(lfs_pointer(sha, size)) != f.get("oid"):
                errors.append(f"X01 pins {name} {path}: oid is not the blob SHA-1 of the LFS pointer "
                              "for the pinned sha256 and size")
        for x in excluded:
            if not (isinstance(x.get("reason"), str) and x["reason"] and x["reason"].isascii()):
                errors.append(f"X01 pins {name} {x.get('path')!a}: exclusion reason is not a non-empty ASCII string")
        paths = [f.get("path") for f in files]
        if all(isinstance(p, str) for p in paths) and paths != sorted(paths):
            errors.append(f"X01 pins {name}: adopted files are not sorted by path")
    return errors


# ---------------------------------------------------------------- manifest

def render_manifest(src):
    """download_manifest.json bytes for one source (a function of the pins alone)."""
    files = []
    for f in src["files"]:
        rec = {"path": f["path"], "source_url": source_url(src, f["path"]),
               "size_bytes": f["size"], "sha256": f["sha256"]}
        if f["lfs"]:
            rec.update({"lfs_pointer_git_blob_sha1": f["oid"], "matches_hub_lfs": True, "matches_hub_blob": True})
        else:
            rec.update({"git_blob_sha1": f["oid"], "matches_hub_blob": True})
        files.append(rec)
    doc = {
        "dataset": src["dataset"],
        "dataset_url": dataset_url(src),
        "revision": src["revision"],
        "license": src["license"],
        "license_note": src["license_note"],
        "download_method": DOWNLOAD_METHOD,
        "excluded_from_download": [f"{x['path']} ({x['reason']})" for x in src["excluded"]],
        "verification": VERIFICATION,
        "all_checks_passed": True,
        "files": files,
        "note": NOTE,
    }
    return (json.dumps(doc, ensure_ascii=True, indent=2) + "\n").encode("ascii")


def rules_source_record(name, src):
    """The keys of external_rules.json "sources"[name] that must equal the pins (X09)."""
    return {
        "dataset": src["dataset"],
        "dataset_url": dataset_url(src),
        "revision": src["revision"],
        "license": src["license"],
        "dir": f"{OUT_REL}/{name}",
        "manifest": f"{OUT_REL}/{name}/{MANIFEST}",
        "manifest_sha256": build_lists.sha256_hex(render_manifest(src)),
        "files": [{"path": f["path"], "size_bytes": f["size"], "sha256": f["sha256"]} for f in src["files"]],
    }


# ---------------------------------------------------------------- network

def default_opener():
    return urllib.request.build_opener()


def _read_limited(resp, limit):
    """At most limit + 1 bytes of the body (one byte more than allowed shows that it is too long)."""
    chunks, total = [], 0
    while total <= limit:
        chunk = resp.read(min(65536, limit + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def http_get(opener, url, limit):
    """(body, Link header, None) or (None, None, reason); the body is at most limit bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
    try:
        with opener.open(req, timeout=TIMEOUT_S) as resp:
            status = getattr(resp, "status", None)
            if status != 200:
                return None, None, f"HTTP {status}"
            link = resp.headers.get("Link") or ""
            data = _read_limited(resp, limit)
    except urllib.error.HTTPError as exc:
        exc.close()  # the error body is never read
        return None, None, f"HTTP {exc.code}"
    except (OSError, http.client.HTTPException) as exc:  # URLError and socket timeouts are OSError
        return None, None, f"network error ({build_lists.exc_name(exc)})"
    if len(data) > limit:
        return None, None, f"body longer than {limit} bytes"
    return data, link, None


def parse_tree(name, data, link):
    """({path: entry} of the files in a tree listing, X03 errors)."""
    if 'rel="next"' in link:
        return None, [f"X03 {name}: the Hub tree listing is paginated"]
    try:
        doc = json.loads(data.decode("utf-8"))
    except ValueError:
        return None, [f"X03 {name}: the Hub tree listing is not UTF-8 JSON"]
    except RecursionError:
        return None, [f"X03 {name}: the Hub tree listing is nested too deeply"]
    if not isinstance(doc, list):
        return None, [f"X03 {name}: the Hub tree listing is not a JSON list"]
    files = {}
    for e in doc:
        ok = (isinstance(e, dict) and e.get("type") in ("file", "directory") and isinstance(e.get("path"), str)
              and isinstance(e.get("oid"), str) and isinstance(e.get("size"), int)
              and not isinstance(e.get("size"), bool))
        lfs = e.get("lfs") if ok else None
        if ok and lfs is not None:
            ok = (isinstance(lfs, dict) and isinstance(lfs.get("oid"), str) and isinstance(lfs.get("size"), int)
                  and not isinstance(lfs.get("size"), bool))
        if not ok:
            return None, [f"X03 {name}: the Hub tree listing has a malformed entry"]
        if e["type"] != "file":
            continue
        if e["path"] in files:
            return None, [f"X03 {name}: the Hub tree listing repeats {e['path']!a}"]
        files[e["path"]] = e
    return files, []


def compare_tree(name, src, files):
    """X04 / X05 messages: Hub tree entries against the pins."""
    errors = []
    for f in src["files"]:
        e = files.get(f["path"])
        if e is None:
            errors.append(f"X04 {name} {f['path']}: not in the Hub tree at the revision")
            continue
        diffs = []
        if e["size"] != f["size"]:
            diffs.append(f"size {e['size']}")
        if e["oid"] != f["oid"]:
            diffs.append(f"oid {e['oid'][:12]!a}")
        lfs = e.get("lfs")
        if f["lfs"] and lfs is None:
            diffs.append("not an LFS file")
        elif not f["lfs"] and lfs is not None:
            diffs.append("an LFS file")
        elif lfs is not None:
            if lfs["oid"] != f["sha256"]:
                diffs.append(f"LFS sha256 {lfs['oid'][:12]!a}")
            if lfs["size"] != f["size"]:
                diffs.append(f"LFS size {lfs['size']}")
        if diffs:
            errors.append(f"X04 {name} {f['path']}: the Hub tree differs from the pin ({', '.join(diffs)})")
    known = {f["path"] for f in src["files"]} | {x["path"] for x in src["excluded"]}
    extra = sorted(set(files) - known)
    if extra:
        errors.append(f"X05 {name}: {len(extra)} file(s) of the revision are neither adopted nor excluded "
                      f"(first: {extra[0][:80]!a})")
    for x in src["excluded"]:
        e = files.get(x["path"])
        if e is None:
            errors.append(f"X05 {name} {x['path']}: excluded file is not in the Hub tree at the revision")
        elif e["oid"] != x["oid"]:
            errors.append(f"X05 {name} {x['path']}: excluded file has oid {e['oid'][:12]!a}, not the pinned one")
    return errors


def bytes_diffs(f, data):
    """How the bytes differ from the pin of one file (empty list when they are the pinned bytes)."""
    diffs = []
    if len(data) != f["size"]:
        diffs.append(f"size {len(data)}")
    sha = build_lists.sha256_hex(data)
    if sha != f["sha256"]:
        diffs.append(f"sha256 {sha[:12]}")
    oid = stored_oid(data, f["lfs"])
    if oid != f["oid"]:
        diffs.append(f"git blob sha1 {oid[:12]}")
    return diffs


def fetch_source(name, src, opener):
    """(errors, {path: bytes}) for one source: the tree listing first, then every adopted file."""
    data, link, reason = http_get(opener, tree_url(src), TREE_LIMIT)
    if reason:
        return [f"X02 {name} tree listing: {reason}"], {}
    files, errors = parse_tree(name, data, link)
    if not errors:
        errors = compare_tree(name, src, files)
    if errors:
        return errors, {}
    blobs = {}
    for f in src["files"]:
        data, _, reason = http_get(opener, source_url(src, f["path"]), f["size"])
        if reason:
            errors.append(f"X02 {name} {f['path']}: {reason}")
            continue
        diffs = bytes_diffs(f, data)
        if diffs:
            errors.append(f"X06 {name} {f['path']}: downloaded bytes differ from the pin ({', '.join(diffs)})")
            continue
        blobs[f["path"]] = data
    return errors, blobs


# ---------------------------------------------------------------- local files

def _read_local(path):
    """(bytes, None), (None, None) when the file does not exist, or (None, reason)."""
    try:
        return Path(path).read_bytes(), None
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, f"cannot read ({build_lists.exc_name(exc)})"


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def plan_writes(base, sources, blobs):
    """(errors, [(path, bytes, label)], kept count): what --download writes; X07 / X08 for conflicts."""
    errors, plan, kept = [], [], 0
    for name, src in sources.items():
        targets = [(base / name / f["path"], blobs[name][f["path"]], f"{name} {f['path']}", "X07")
                   for f in src["files"]]
        targets.append((base / name / MANIFEST, render_manifest(src), f"{name} {MANIFEST}", "X08"))
        for path, data, label, code in targets:
            have, reason = _read_local(path)
            if reason:
                errors.append(f"{code} {label}: {reason}")
            elif have is None:
                plan.append((path, data, label))
            elif have != data:
                errors.append(f"{code} {label}: a different file is already there "
                              f"(size {len(have)}, sha256 {build_lists.sha256_hex(have)[:12]}); "
                              "it is not overwritten: remove it to download again")
            else:
                kept += 1
    return errors, plan, kept


def unpinned_entries(folder, src):
    """(names under a source folder that the pins do not account for, listing error or None).

    Names are POSIX paths relative to the folder; nothing is opened. A file counts unless it is a
    pinned file or download_manifest.json; a folder counts unless a pinned file lies below it, and
    such an extra folder is one entry (it is not listed further). Names are compared as the file
    system compares them (os.path.normcase).
    """
    key = os.path.normcase
    keep = [f["path"] for f in src["files"]] + [MANIFEST]
    files = {key(p) for p in keep}
    dirs = {key(p[:i]) for p in keep for i, c in enumerate(p) if c == "/"}
    extra, failed = [], []
    if not os.path.isdir(folder):
        return extra, None
    for top, dirnames, filenames in os.walk(folder, onerror=failed.append):
        rel = os.path.relpath(top, folder).replace(os.sep, "/")
        prefix = "" if rel == "." else rel + "/"
        extra += [prefix + n for n in filenames if key(prefix + n) not in files]
        extra += [prefix + n for n in dirnames if key(prefix + n) not in dirs]
        dirnames[:] = [n for n in dirnames if key(prefix + n) in dirs]
    return sorted(extra), (build_lists.exc_name(failed[0]) if failed else None)


def check_base(base, names):
    """X07 messages: folders directly under the base folder that are not pinned sources (names only)."""
    known = {os.path.normcase(n) for n in names}
    try:
        with os.scandir(base) as entries:
            extra = sorted(e.name for e in entries if e.is_dir() and os.path.normcase(e.name) not in known)
    except FileNotFoundError:
        return []  # every pinned file is then reported as missing
    except OSError as exc:
        return [f"X07 base folder: cannot list ({build_lists.exc_name(exc)})"]
    if extra:
        return [f"X07 base folder: {len(extra)} folder(s) are not pinned sources (first: {extra[0][:80]!a}); "
                "remove them"]
    return []


def check_local(base, name, src):
    """X07 / X08 messages of the offline check for one source."""
    errors = []
    for f in src["files"]:
        have, reason = _read_local(base / name / f["path"])
        if have is None:
            errors.append(f"X07 {name} {f['path']}: {reason or 'missing (run --download)'}")
            continue
        diffs = bytes_diffs(f, have)
        if diffs:
            errors.append(f"X07 {name} {f['path']}: differs from the pin ({', '.join(diffs)})")
    have, reason = _read_local(base / name / MANIFEST)
    if have is None:
        errors.append(f"X08 {name} {MANIFEST}: {reason or 'missing (run --download)'}")
    elif have != render_manifest(src):
        errors.append(f"X08 {name} {MANIFEST}: differs from a fresh rendering "
                      f"(sha256 {build_lists.sha256_hex(have)[:12]})")
    extra, reason = unpinned_entries(base / name, src)
    if reason:
        errors.append(f"X07 {name}: cannot list the source folder ({reason})")
    if extra:
        errors.append(f"X07 {name}: {len(extra)} file(s) or folder(s) in the source folder are not pinned "
                      f"(first: {extra[0][:80]!a}); remove them")
    return errors


def check_rules(path, sources):
    """X09 messages: the "sources" block of external_rules.json against the pins."""
    shown = build_lists.show(path)
    raw, reason = _read_local(path)
    if raw is None:
        return [f"X09 {shown}: {reason or 'missing'}"]
    if b"\r" in raw or not raw.endswith(b"\n"):
        return [f"X09 {shown}: not LF text with a final line break"]
    try:
        doc = json.loads(raw.decode("utf-8"))
    except ValueError:
        return [f"X09 {shown}: not UTF-8 JSON"]
    except RecursionError:
        return [f"X09 {shown}: JSON nested too deeply"]
    block = doc.get("sources") if isinstance(doc, dict) else None
    if not isinstance(block, dict) or sorted(block) != sorted(sources):
        return [f"X09 {shown}: \"sources\" does not hold exactly {', '.join(sorted(sources))}"]
    errors = []
    for name in sorted(sources):
        src, got = sources[name], block[name]
        if not isinstance(got, dict):
            errors.append(f"X09 {shown}: sources.{name} is not an object")
            continue
        for key, want in rules_source_record(name, src).items():
            if got.get(key) != want:
                errors.append(f"X09 {shown}: sources.{name}.{key} differs from the pins")
        split_files = got.get("split_files")
        data_paths = sorted(f["path"] for f in src["files"] if f["path"] != "README.md")
        if not isinstance(split_files, dict) or sorted(map(str, split_files.values())) != data_paths:
            errors.append(f"X09 {shown}: sources.{name}.split_files does not give every adopted data file "
                          "to exactly one split")
        elif got.get("adopted_splits") != list(split_files):
            errors.append(f"X09 {shown}: sources.{name}.adopted_splits differs from the split_files keys")
    return errors


# ---------------------------------------------------------------- main

def _say(line):
    print(line.encode("ascii", "backslashreplace").decode("ascii"))


def _refuse(errors):
    for e in errors:
        _say("FAIL " + e)
    print(f"refused: {len(errors)} error(s); nothing written")
    return 1


def parse_args(argv):
    p = argparse.ArgumentParser(description="Download or check the pinned external sources (see module docstring).")
    p.add_argument("--download", action="store_true", help="download from the Hub (online) and write the files")
    p.add_argument("--source", nargs="+", choices=sorted(SOURCES), metavar="NAME",
                   help="sources to handle (default: all)")
    p.add_argument("--out", help=f"base folder (default: <repo>/{OUT_REL})")
    p.add_argument("--rules", help=f"rules file for the check (default: <out>/{RULES_NAME})")
    return p.parse_args(argv)


def main(argv=None, opener=None):
    data_lists.install_test_guard()  # before any file is opened: this process never reads test.csv
    args = parse_args(argv)
    names = sorted(set(args.source)) if args.source else sorted(SOURCES)
    sources = {n: SOURCES[n] for n in names}
    base = Path(args.out) if args.out else DEFAULT_OUT
    errors = validate_pins(SOURCES)
    if build_lists._under(base, build_lists.DATASETS):
        errors.append("X01 --out is inside datasets/")
    if args.download and os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in OFFLINE_VALUES:
        errors.append("X01 --download needs network access: run it without HF_HUB_OFFLINE")
    if args.download and args.rules:
        errors.append("X01 --rules belongs to the check; --download does not read the rules file")
    if args.rules and build_lists._is_test_csv(args.rules, build_lists._real(args.rules)):
        errors.append("X01 --rules is a test.csv path")
    elif args.rules and build_lists._under(args.rules, build_lists.DATASETS):
        errors.append("X01 --rules is inside datasets/")
    if errors:
        return _refuse(errors)

    if not args.download:
        errors += check_base(base, SOURCES)
        for name, src in sources.items():
            errors += check_local(base, name, src)
        rules = Path(args.rules) if args.rules else base / RULES_NAME
        errors += check_rules(rules, SOURCES)
        if errors:
            return _refuse(errors)
        for name, src in sources.items():
            _say(f"ok {name}: {len(src['files'])} raw file(s) match the pins; {MANIFEST} sha256 "
                 f"{build_lists.sha256_hex(render_manifest(src))[:12]}")
        _say(f"ok {build_lists.show(rules)}: sources block matches the pins")
        print(f"check OK: {len(sources)} source(s), no network used")
        return 0

    if opener is None:
        opener = default_opener()
    blobs = {}
    for name, src in sources.items():
        errs, blobs[name] = fetch_source(name, src, opener)
        errors += errs
    if errors:
        return _refuse(errors)
    errors, plan, kept = plan_writes(base, sources, blobs)
    if errors:
        return _refuse(errors)
    for name, src in sources.items():
        for f in src["files"]:
            _say(f"ok {name} {f['path']}: {f['size']} bytes, sha256 {f['sha256'][:12]}, equal to the pin and the "
                 f"Hub tree ({'LFS sha256, LFS size, pointer blob' if f['lfs'] else 'git blob'})")
    written = 0
    for path, data, label in plan:
        try:
            _write(path, data)
        except OSError as exc:
            _say(f"FAIL X10 {label}: cannot write ({build_lists.exc_name(exc)})")
            print(f"stopped: 1 error(s); {written} file(s) were written before the error")
            return 1
        written += 1
    n_files = sum(len(s["files"]) for s in sources.values())
    print(f"OK {len(sources)} source(s), {n_files} raw file(s) verified against the Hub tree at the pinned "
          f"revisions; wrote {written} file(s), {kept} already present and identical")
    _say(f"folder: {build_lists.show(base)}")
    print("next: PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/fetch_external.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
