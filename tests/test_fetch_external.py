"""Tests for scripts/fetch_external.py with a fake opener (no network, synthetic bytes, no dataset text).

Expected hashes were computed outside the script: `sha256sum` for SHA-256 and `git hash-object --stdin`
for git blob SHA-1 (of the file, or of the Git LFS pointer text for LFS files).
data_lists.install_test_guard (a process-wide audit hook) is mocked for the whole module.

Run from Git Bash at the repo root:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m unittest tests/test_fetch_external.py -v
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_external as F  # noqa: E402

CARD = b"card\n"
HELLO = b"hello\n"
WORLD = b"hello world\n"
SHA_CARD = "ee5a8aebf57ef226f9b73e086350e212d563474aca1576db0f4960021004664f"     # printf 'card\n' | sha256sum
SHA_HELLO = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"    # printf 'hello\n' | sha256sum
SHA_WORLD = "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"    # printf 'hello world\n' | sha256sum
BLOB_EMPTY = "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"     # git hash-object --stdin < /dev/null
BLOB_CARD = "00c3bdd3af760ad499e18bacdee7430e609d143e"      # printf 'card\n' | git hash-object --stdin
BLOB_HELLO = "ce013625030ba8dba906f756967f9e9ca394464a"     # printf 'hello\n' | git hash-object --stdin
BLOB_X = "587be6b4c3f93f93c489c0111bba5596147a26cb"         # printf 'x\n' | git hash-object --stdin
# git hash-object of the LFS pointer text "version ...spec/v1\noid sha256:<sha>\nsize <n>\n"
PTR_WORLD = "0db788d9cca75b69bc4c2430024a654fafdb504e"      # sha SHA_WORLD, size 12
PTR_HELLO = "fa61e1ef71c9908c0926776a4aa3504b3a27f159"      # sha SHA_HELLO, size 6
REV_A = "1" * 40
REV_B = "2" * 40

SYN = {
    "alpha": {
        "dataset": "org-a/set-a", "revision": REV_A, "license": "mit", "license_note": "synthetic card",
        "files": (
            {"path": "README.md", "size": 5, "sha256": SHA_CARD, "oid": BLOB_CARD, "lfs": False},
            {"path": "data/part.parquet", "size": 12, "sha256": SHA_WORLD, "oid": PTR_WORLD, "lfs": True},
        ),
        "excluded": (
            {"path": ".gitattributes", "oid": BLOB_X, "reason": "config"},
            {"path": "data/copy.parquet", "oid": PTR_WORLD, "reason": "copy"},
        ),
    },
    "beta": {
        "dataset": "org-b/set-b", "revision": REV_B, "license": "apache-2.0", "license_note": "synthetic card",
        "files": (
            {"path": "README.md", "size": 6, "sha256": SHA_HELLO, "oid": BLOB_HELLO, "lfs": False},
            {"path": "data/one.parquet", "size": 6, "sha256": SHA_HELLO, "oid": PTR_HELLO, "lfs": True},
        ),
        "excluded": (),
    },
}
TREE_A = f"https://huggingface.co/api/datasets/org-a/set-a/tree/{REV_A}?recursive=true"
TREE_B = f"https://huggingface.co/api/datasets/org-b/set-b/tree/{REV_B}?recursive=true"
URL_A_CARD = f"https://huggingface.co/datasets/org-a/set-a/resolve/{REV_A}/README.md"
URL_A_PART = f"https://huggingface.co/datasets/org-a/set-a/resolve/{REV_A}/data/part.parquet"
URL_B_CARD = f"https://huggingface.co/datasets/org-b/set-b/resolve/{REV_B}/README.md"
URL_B_ONE = f"https://huggingface.co/datasets/org-b/set-b/resolve/{REV_B}/data/one.parquet"

MANIFEST_A = {
    "dataset": "org-a/set-a",
    "dataset_url": "https://huggingface.co/datasets/org-a/set-a",
    "revision": REV_A,
    "license": "mit",
    "license_note": "synthetic card",
    "download_method": "urllib from Hugging Face resolve URLs pinned to revision "
                       "(scripts/fetch_external.py --download)",
    "excluded_from_download": [".gitattributes (config)", "data/copy.parquet (copy)"],
    "verification": F.VERIFICATION,
    "all_checks_passed": True,
    "files": [
        {"path": "README.md", "source_url": URL_A_CARD, "size_bytes": 5, "sha256": SHA_CARD,
         "git_blob_sha1": BLOB_CARD, "matches_hub_blob": True},
        {"path": "data/part.parquet", "source_url": URL_A_PART, "size_bytes": 12, "sha256": SHA_WORLD,
         "lfs_pointer_git_blob_sha1": PTR_WORLD, "matches_hub_lfs": True, "matches_hub_blob": True},
    ],
    "note": F.NOTE,
}


GUARD_CALLS = []


def setUpModule():
    # main() installs the test.csv guard; the real one is a process-wide audit hook and must not be installed here
    patcher = mock.patch.object(F.data_lists, "install_test_guard", side_effect=lambda: GUARD_CALLS.append("guard"))
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


def tree_a():
    lfs = {"oid": SHA_WORLD, "size": 12, "pointerSize": 127}
    return [
        {"type": "directory", "oid": "0" * 40, "size": 0, "path": "data"},
        {"type": "file", "oid": BLOB_X, "size": 2, "path": ".gitattributes"},
        {"type": "file", "oid": BLOB_CARD, "size": 5, "path": "README.md"},
        {"type": "file", "oid": PTR_WORLD, "size": 12, "lfs": dict(lfs), "xetHash": "ab" * 32,
         "path": "data/copy.parquet"},
        {"type": "file", "oid": PTR_WORLD, "size": 12, "lfs": dict(lfs), "xetHash": "ab" * 32,
         "path": "data/part.parquet"},
    ]


def tree_b():
    return [
        {"type": "directory", "oid": "0" * 40, "size": 0, "path": "data"},
        {"type": "file", "oid": BLOB_HELLO, "size": 6, "path": "README.md"},
        {"type": "file", "oid": PTR_HELLO, "size": 6, "lfs": {"oid": SHA_HELLO, "size": 6, "pointerSize": 126},
         "path": "data/one.parquet"},
    ]


def routes(**changes):
    r = {TREE_A: json.dumps(tree_a()).encode(), URL_A_CARD: CARD, URL_A_PART: WORLD,
         TREE_B: json.dumps(tree_b()).encode(), URL_B_CARD: HELLO, URL_B_ONE: HELLO}
    r.update(changes)
    return r


class FakeResponse:
    def __init__(self, body, status=200, headers=None):
        self._buf = io.BytesIO(body)
        self.status = status
        self.headers = headers or {}
        self.served = 0

    def read(self, n=-1):
        chunk = self._buf.read(n)
        self.served += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Stands in for urllib's opener: routes[url] is a body, (body, status, headers) or an exception."""

    def __init__(self, table):
        self.table = table
        self.calls = []
        self.responses = {}

    def open(self, req, timeout=None):
        self.calls.append((req.full_url, req.get_method(), dict(req.header_items()), timeout))
        r = self.table.get(req.full_url)
        if r is None:
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
        if isinstance(r, BaseException):
            raise r
        resp = FakeResponse(*r) if isinstance(r, tuple) else FakeResponse(r)
        self.responses[req.full_url] = resp
        return resp

    def urls(self):
        return [c[0] for c in self.calls]


class Base(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp()) / "external"
        self.addCleanup(shutil.rmtree, self.out.parent, ignore_errors=True)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("HF_HUB_OFFLINE", None)          # the suite itself runs with HF_HUB_OFFLINE=1
        for p in (mock.patch.object(F, "SOURCES", copy.deepcopy(SYN)),
                  mock.patch.object(F, "default_opener", side_effect=AssertionError("network use in a test"))):
            p.start()
            self.addCleanup(p.stop)

    def run_main(self, *argv, table=None, opener=None):
        opener = opener or FakeOpener(routes() if table is None else table)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = F.main(["--out", str(self.out), *argv], opener=opener)
        return code, buf.getvalue(), opener

    def files(self):
        return sorted(p.relative_to(self.out).as_posix() for p in self.out.rglob("*") if p.is_file()) \
            if self.out.exists() else []

    def assert_refused(self, needle, *argv, written=(), **kw):
        code, out, opener = self.run_main(*argv, **kw)
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL " + needle, out)
        self.assertRegex(out, r"refused: \d+ error\(s\); nothing written\n$")
        self.assertTrue(out.isascii())
        self.assertEqual(self.files(), sorted(written))
        return out, opener

    def write_rules(self, change=None, raw=None):
        doc = {"sources": {
            "alpha": {"dataset": "org-a/set-a", "dataset_url": "https://huggingface.co/datasets/org-a/set-a",
                      "revision": REV_A, "license": "mit", "dir": "data/external/alpha",
                      "manifest": "data/external/alpha/download_manifest.json",
                      "manifest_sha256": hashlib.sha256((self.out / "alpha" / F.MANIFEST).read_bytes()).hexdigest(),
                      "files": [{"path": "README.md", "size_bytes": 5, "sha256": SHA_CARD},
                                {"path": "data/part.parquet", "size_bytes": 12, "sha256": SHA_WORLD}],
                      "adopted_splits": ["part"], "split_files": {"part": "data/part.parquet"}},
            "beta": {"dataset": "org-b/set-b", "dataset_url": "https://huggingface.co/datasets/org-b/set-b",
                     "revision": REV_B, "license": "apache-2.0", "dir": "data/external/beta",
                     "manifest": "data/external/beta/download_manifest.json",
                     "manifest_sha256": hashlib.sha256((self.out / "beta" / F.MANIFEST).read_bytes()).hexdigest(),
                     "files": [{"path": "README.md", "size_bytes": 6, "sha256": SHA_HELLO},
                               {"path": "data/one.parquet", "size_bytes": 6, "sha256": SHA_HELLO}],
                     "adopted_splits": ["one"], "split_files": {"one": "data/one.parquet"}},
        }}
        if change:
            change(doc)
        data = (json.dumps(doc, indent=1) + "\n").encode("utf-8") if raw is None else raw
        (self.out / F.RULES_NAME).write_bytes(data)


ALL_FILES = ["alpha/README.md", "alpha/data/part.parquet", "alpha/download_manifest.json",
             "beta/README.md", "beta/data/one.parquet", "beta/download_manifest.json"]


class HashTest(unittest.TestCase):
    def test_git_blob_sha1(self):
        self.assertEqual(F.git_blob_sha1(b""), BLOB_EMPTY)
        self.assertEqual(F.git_blob_sha1(HELLO), BLOB_HELLO)
        self.assertEqual(F.git_blob_sha1(CARD), BLOB_CARD)

    def test_lfs_pointer_text_and_oid(self):
        want = (b"version https://git-lfs.github.com/spec/v1\n"
                b"oid sha256:a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447\n"
                b"size 12\n")
        self.assertEqual(F.lfs_pointer(SHA_WORLD, 12), want)
        self.assertEqual(len(want), 127)
        self.assertEqual(F.git_blob_sha1(want), PTR_WORLD)
        self.assertEqual(F.stored_oid(WORLD, True), PTR_WORLD)
        self.assertEqual(F.stored_oid(HELLO, True), PTR_HELLO)
        self.assertEqual(F.stored_oid(HELLO, False), BLOB_HELLO)

    def test_bytes_diffs_names_every_difference(self):
        plain, lfs = SYN["alpha"]["files"]
        self.assertEqual(F.bytes_diffs(plain, CARD), [])
        self.assertEqual(F.bytes_diffs(lfs, WORLD), [])
        self.assertEqual(F.bytes_diffs(plain, b"cord\n"),          # same size, other bytes
                         ["sha256 " + hashlib.sha256(b"cord\n").hexdigest()[:12],
                          "git blob sha1 " + F.git_blob_sha1(b"cord\n")[:12]])
        self.assertEqual(F.bytes_diffs(lfs, WORLD[:-1])[0], "size 11")
        self.assertEqual(len(F.bytes_diffs(lfs, WORLD[:-1])), 3)
        # each pin field is compared on its own
        self.assertEqual(F.bytes_diffs({**plain, "size": 6}, CARD), ["size 5"])
        self.assertEqual(F.bytes_diffs({**plain, "sha256": "0" * 64}, CARD), ["sha256 " + SHA_CARD[:12]])
        self.assertEqual(F.bytes_diffs({**plain, "oid": "0" * 40}, CARD), ["git blob sha1 " + BLOB_CARD[:12]])
        self.assertEqual(F.bytes_diffs({**lfs, "oid": BLOB_HELLO}, WORLD), ["git blob sha1 " + PTR_WORLD[:12]])


class PinTest(unittest.TestCase):
    """The real pin table against the values of the design (D7) and of the Hub tree API (2026-11-17)."""

    def test_real_pins_are_consistent(self):
        self.assertEqual(F.validate_pins(F.SOURCES), [])
        self.assertEqual(sorted(F.SOURCES), ["deepset", "notinject"])

    def test_deepset_pins(self):
        s = F.SOURCES["deepset"]
        self.assertEqual(s["dataset"], "deepset/prompt-injections")
        self.assertEqual(s["revision"], "4f61ecb038e9c3fb77e21034b22511b523772cdd")
        self.assertEqual(s["license"], "apache-2.0")
        self.assertIn("cc-by-4.0", s["license_note"])
        files = {f["path"]: f for f in s["files"]}
        self.assertEqual(sorted(files), ["README.md", "data/test-00000-of-00001-701d16158af87368.parquet",
                                         "data/train-00000-of-00001-9564e8b05b4757ab.parquet"])
        train = files["data/train-00000-of-00001-9564e8b05b4757ab.parquet"]
        test = files["data/test-00000-of-00001-701d16158af87368.parquet"]
        self.assertEqual((train["size"], train["sha256"][:8], train["lfs"]), (40323, "2e10bc7a", True))
        self.assertEqual((test["size"], test["sha256"][:8], test["lfs"]), (10892, "39ac797c", True))
        self.assertEqual(train["oid"], "551b244bd3dc2e271865caf32d209bb7f6e249ba")     # Hub tree oid
        self.assertEqual(test["oid"], "8bf9d00201127a68fc3b0cf1e8da9ecff8b221e7")
        self.assertEqual((files["README.md"]["size"], files["README.md"]["lfs"]), (500, False))
        self.assertEqual(files["README.md"]["oid"], "d3a63fc0a09231cd1563afb14facfd9d55adcd03")
        self.assertEqual(F.source_url(s, train["path"]),
                         "https://huggingface.co/datasets/deepset/prompt-injections/resolve/"
                         "4f61ecb038e9c3fb77e21034b22511b523772cdd/data/train-00000-of-00001-9564e8b05b4757ab.parquet")
        self.assertEqual(F.tree_url(s), "https://huggingface.co/api/datasets/deepset/prompt-injections/tree/"
                                        "4f61ecb038e9c3fb77e21034b22511b523772cdd?recursive=true")
        self.assertEqual([x["path"] for x in s["excluded"]], [".gitattributes", "gitattributes.txt"])

    def test_notinject_pins(self):
        s = F.SOURCES["notinject"]
        self.assertEqual(s["dataset"], "leolee99/NotInject")
        self.assertEqual(s["revision"], "847ae76cf8fea5ed325429e569ae8cfef022d2e0")
        self.assertEqual(s["license"], "mit")
        files = {f["path"]: f for f in s["files"]}
        self.assertEqual(sorted(files), ["README.md", "data/NotInject_one-00000-of-00001.parquet",
                                         "data/NotInject_three-00000-of-00001.parquet",
                                         "data/NotInject_two-00000-of-00001.parquet"])
        self.assertEqual([(f["size"], f["lfs"]) for f in s["files"]],
                         [(2973, False), (10859, True), (13009, True), (11183, True)])
        self.assertEqual(files["data/NotInject_one-00000-of-00001.parquet"]["sha256"][:8], "5b1d3a8e")
        # the three excluded parquet files are the same git blobs as the adopted ones
        excluded = {x["path"]: x["oid"] for x in s["excluded"]}
        for copy_name, adopted in (("train", "one"), ("validation", "two"), ("test", "three")):
            self.assertEqual(excluded[f"data/{copy_name}-00000-of-00001.parquet"],
                             files[f"data/NotInject_{adopted}-00000-of-00001.parquet"]["oid"])
        self.assertEqual(sorted(excluded), [".gitattributes", "data/test-00000-of-00001.parquet",
                                            "data/train-00000-of-00001.parquet",
                                            "data/validation-00000-of-00001.parquet"])

    def test_generic_user_agent(self):
        self.assertEqual(F.USER_AGENT, "python-urllib")
        self.assertEqual(F.HUB, "https://huggingface.co")

    def test_validate_pins_reports_each_defect(self):
        def errors(change):
            src = copy.deepcopy(SYN)
            change(src)
            return F.validate_pins(src)

        self.assertEqual(F.validate_pins(SYN), [])
        files = lambda s, i: s["alpha"]["files"][i]  # noqa: E731
        cases = {
            "revision is not a 40-character": lambda s: s["alpha"].update(revision="main"),
            "dataset is not <owner>/<name>": lambda s: s["alpha"].update(dataset="set-a"),
            "license is not": lambda s: s["alpha"].update(license=""),
            "no adopted file": lambda s: s["alpha"].update(files=()),
            "unsafe path '../x'": lambda s: files(s, 0).update(path="../x"),
            "unsafe path 'a b.txt'": lambda s: files(s, 0).update(path="a b.txt"),
            "unsafe path '/etc/x'": lambda s: files(s, 0).update(path="/etc/x"),
            "unsafe path 'data\\\\x'": lambda s: files(s, 0).update(path="data\\x"),
            "data/Test.CSV: reserved file name": lambda s: files(s, 1).update(path="data/Test.CSV"),
            "data/test.csv.: reserved file name": lambda s: files(s, 1).update(path="data/test.csv."),  # Windows alias
            "download_manifest.json: reserved file name": lambda s: files(s, 1).update(path="download_manifest.json"),
            "path repeated": lambda s: files(s, 1).update(path="readme.md"),
            "data/copy.parquet: path repeated": lambda s: files(s, 1).update(path="data/copy.parquet"),
            "oid is not a 40-character": lambda s: files(s, 0).update(oid="abc"),
            "size is not a positive integer": lambda s: files(s, 0).update(size=0),
            "sha256 is not 64 hex": lambda s: files(s, 0).update(sha256=SHA_CARD.upper()),
            "lfs is not true or false": lambda s: files(s, 0).update(lfs=None),
            "exclusion reason": lambda s: s["alpha"]["excluded"][0].update(reason=""),
            "not sorted by path": lambda s: s["alpha"].update(files=s["alpha"]["files"][::-1]),
            "source name 'Alpha'": lambda s: s.update(Alpha=s.pop("alpha")),
        }
        for needle, change in cases.items():
            got = errors(change)
            self.assertTrue(any(e.startswith("X01 ") and needle in e for e in got), (needle, got))
        # an LFS oid must follow from the pinned SHA-256 and size
        for change in (lambda s: files(s, 1).update(size=13), lambda s: files(s, 1).update(sha256=SHA_HELLO),
                       lambda s: files(s, 1).update(oid=BLOB_HELLO)):
            got = errors(change)
            self.assertTrue(any("not the blob SHA-1 of the LFS pointer" in e for e in got), got)

    def test_main_refuses_a_bad_pin_table_before_any_request(self):
        bad = copy.deepcopy(SYN)
        bad["alpha"]["files"][1]["size"] = 13
        opener = FakeOpener(routes())
        buf = io.StringIO()
        out_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out_dir, ignore_errors=True)
        with mock.patch.object(F, "SOURCES", bad), mock.patch.dict(os.environ), contextlib.redirect_stdout(buf):
            os.environ.pop("HF_HUB_OFFLINE", None)
            code = F.main(["--download", "--out", str(out_dir / "x")], opener=opener)
        self.assertEqual(code, 1)
        self.assertIn("FAIL X01 pins alpha data/part.parquet", buf.getvalue())
        self.assertEqual(opener.calls, [])
        self.assertFalse((out_dir / "x").exists())

    def test_source_option_still_validates_the_whole_pin_table(self):
        bad = copy.deepcopy(SYN)
        bad["alpha"]["files"][1]["size"] = 13                               # the source that is not selected
        out_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out_dir, ignore_errors=True)
        for mode in (["--download"], []):
            opener = FakeOpener(routes())
            buf = io.StringIO()
            with mock.patch.object(F, "SOURCES", bad), mock.patch.dict(os.environ), contextlib.redirect_stdout(buf), \
                    mock.patch.object(F, "_read_local", side_effect=AssertionError("a file was read")):
                os.environ.pop("HF_HUB_OFFLINE", None)
                code = F.main([*mode, "--source", "beta", "--out", str(out_dir / "x")], opener=opener)
            self.assertEqual(code, 1, buf.getvalue())
            self.assertIn("FAIL X01 pins alpha data/part.parquet", buf.getvalue())
            self.assertIn("refused: 1 error(s); nothing written", buf.getvalue())
            self.assertEqual(opener.calls, [])
        self.assertFalse((out_dir / "x").exists())


class DownloadTest(Base):
    def test_download_writes_verified_bytes_and_manifest(self):
        code, out, opener = self.run_main("--download")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.files(), ALL_FILES)
        self.assertEqual((self.out / "alpha" / "README.md").read_bytes(), CARD)
        self.assertEqual((self.out / "alpha" / "data" / "part.parquet").read_bytes(), WORLD)
        self.assertEqual((self.out / "beta" / "README.md").read_bytes(), HELLO)
        self.assertEqual((self.out / "beta" / "data" / "one.parquet").read_bytes(), HELLO)
        raw = (self.out / "alpha" / F.MANIFEST).read_bytes()
        doc = json.loads(raw.decode("ascii"))
        self.assertEqual(doc, MANIFEST_A)
        self.assertEqual(list(doc), list(MANIFEST_A))                       # key order
        self.assertEqual([list(f) for f in doc["files"]], [list(f) for f in MANIFEST_A["files"]])
        self.assertTrue(raw.startswith(b'{\n  "dataset": "org-a/set-a",\n  "dataset_url"'))
        self.assertTrue(raw.endswith(b'}\n'))
        self.assertNotIn(b"\r", raw)
        for banned in (b"downloaded_at", b"revision_date", b"splits", str(self.out.parent).encode()):
            self.assertNotIn(banned, raw)
        self.assertIn("OK 2 source(s), 4 raw file(s) verified", out)
        self.assertIn("wrote 6 file(s), 0 already present", out)
        self.assertIn("ok alpha data/part.parquet: 12 bytes, sha256 a948904f2f0f, equal to the pin and the "
                      "Hub tree (LFS sha256, LFS size, pointer blob)", out)
        self.assertIn("ok beta README.md: 6 bytes, sha256 5891b5b522d5, equal to the pin and the Hub tree "
                      "(git blob)", out)

    def test_requests_are_the_pinned_urls_with_a_generic_user_agent_only(self):
        code, out, opener = self.run_main("--download")
        self.assertEqual(code, 0, out)
        self.assertEqual(opener.urls(), [TREE_A, URL_A_CARD, URL_A_PART, TREE_B, URL_B_CARD, URL_B_ONE])
        for url, method, headers, timeout in opener.calls:
            self.assertEqual(method, "GET")
            self.assertEqual(headers, {"User-agent": "python-urllib"})       # nothing else is sent
            self.assertEqual(timeout, 60)

    def test_output_has_no_file_content_and_no_absolute_path(self):
        code, out, _ = self.run_main("--download")
        self.assertEqual(code, 0, out)
        self.assertTrue(out.isascii())
        for text in ("hello", "card\n", "world"):
            self.assertNotIn(text, out)

    def test_second_download_is_identical_and_rewrites_nothing(self):
        self.run_main("--download")
        before = {n: (self.out / n).read_bytes() for n in self.files()}
        stamps = {n: (self.out / n).stat().st_mtime_ns for n in self.files()}
        with mock.patch.object(F, "_write", side_effect=AssertionError("rewrite")):
            code, out, _ = self.run_main("--download")
        self.assertEqual(code, 0, out)
        self.assertIn("wrote 0 file(s), 6 already present and identical", out)
        self.assertEqual({n: (self.out / n).read_bytes() for n in self.files()}, before)
        self.assertEqual({n: (self.out / n).stat().st_mtime_ns for n in self.files()}, stamps)

    def test_render_manifest_is_deterministic_and_lf(self):
        a, b = F.render_manifest(SYN["alpha"]), F.render_manifest(copy.deepcopy(SYN["alpha"]))
        self.assertEqual(a, b)
        self.assertEqual(json.loads(a), MANIFEST_A)
        self.assertEqual(a.count(b"\n"), len(a.splitlines()))
        self.assertTrue(a.isascii())

    def test_source_option_limits_requests_and_files(self):
        code, out, opener = self.run_main("--download", "--source", "beta")
        self.assertEqual(code, 0, out)
        self.assertEqual(opener.urls(), [TREE_B, URL_B_CARD, URL_B_ONE])
        self.assertEqual(self.files(), ALL_FILES[3:])

    def test_x01_offline_environment(self):
        for value in ("1", "true", "YES", " on "):
            os.environ["HF_HUB_OFFLINE"] = value
            out, opener = self.assert_refused("X01 --download needs network access", "--download")
            self.assertEqual(opener.calls, [])
        os.environ["HF_HUB_OFFLINE"] = "0"                                  # not set as offline
        code, out, _ = self.run_main("--download")
        self.assertEqual(code, 0, out)

    def test_x01_out_inside_datasets(self):
        self.assertEqual(F.build_lists.DATASETS, F.ROOT / "datasets")       # the folder that main() protects
        # A stand-in folder in the temporary directory: if the refusal ever fails, the files land there
        # (removed with the directory) and not in the real datasets/.
        datasets = self.out.parent / "datasets"
        for target in (datasets / "fetch-external-test", datasets):
            opener = FakeOpener(routes())
            buf = io.StringIO()
            with mock.patch.object(F.build_lists, "DATASETS", datasets), contextlib.redirect_stdout(buf):
                code = F.main(["--download", "--out", str(target)], opener=opener)
            self.assertEqual(code, 1)
            self.assertIn("FAIL X01 --out is inside datasets/", buf.getvalue())
            self.assertEqual(opener.calls, [])
            self.assertFalse(datasets.exists())

    def test_x01_rules_option_with_download(self):
        _, opener = self.assert_refused("X01 --rules belongs to the check", "--download", "--rules", "x.json")
        self.assertEqual(opener.calls, [])

    def test_x02_http_and_network_errors(self):
        table = routes()
        del table[URL_A_PART]                                               # the fake opener answers 404
        out, _ = self.assert_refused("X02 alpha data/part.parquet: HTTP 404", "--download", table=table)
        self.assertIn("refused: 1 error(s)", out)
        self.assert_refused("X02 beta tree listing: HTTP 503", "--download",
                            table=routes(**{TREE_B: urllib.error.HTTPError(TREE_B, 503, "x", {}, None)}))
        self.assert_refused("X02 alpha README.md: network error (urllib.error.URLError)", "--download",
                            table=routes(**{URL_A_CARD: urllib.error.URLError("no route")}))
        self.assert_refused("X02 alpha README.md: network error (TimeoutError)", "--download",
                            table=routes(**{URL_A_CARD: TimeoutError()}))
        self.assert_refused("X02 beta data/one.parquet: HTTP 206", "--download",
                            table=routes(**{URL_B_ONE: (HELLO, 206, {})}))

    def test_x02_body_longer_than_the_pin_is_not_read_to_the_end(self):
        big = WORLD + b"x" * 500000
        out, opener = self.assert_refused("X02 alpha data/part.parquet: body longer than 12 bytes", "--download",
                                          table=routes(**{URL_A_PART: big}))
        self.assertEqual(opener.responses[URL_A_PART].served, 13)
        huge_tree = b" " * (F.TREE_LIMIT + 10)
        out, opener = self.assert_refused(f"X02 alpha tree listing: body longer than {F.TREE_LIMIT} bytes",
                                          "--download", table=routes(**{TREE_A: huge_tree}))
        self.assertEqual(opener.urls(), [TREE_A, TREE_B, URL_B_CARD, URL_B_ONE])   # no alpha file was requested

    def test_read_limited_collects_short_reads(self):
        class Drip(FakeResponse):
            def read(self, n=-1):
                return super().read(min(n, 5))
        self.assertEqual(F._read_limited(Drip(WORLD), 12), WORLD)
        self.assertEqual(F._read_limited(Drip(WORLD), 100), WORLD)
        self.assertEqual(F._read_limited(Drip(WORLD), 4), WORLD[:5])       # limit + 1 bytes, then stop

    def test_x03_unreadable_tree(self):
        cases = {
            "not UTF-8 JSON": b"<html>busy</html>",
            "not a JSON list": json.dumps({"error": "x"}).encode(),
            "malformed entry": json.dumps(tree_a() + [{"type": "file", "path": "x"}]).encode(),
            "repeats 'README.md'": json.dumps(tree_a() + [tree_a()[2]]).encode(),
        }
        for needle, body in cases.items():
            out, opener = self.assert_refused("X03 alpha: the Hub tree listing", "--download",
                                              table=routes(**{TREE_A: body}))
            self.assertIn(needle, out)
            self.assertNotIn(URL_A_CARD, opener.urls(), needle)
        bad_lfs = tree_a()
        bad_lfs[4]["lfs"] = {"oid": SHA_WORLD}
        self.assert_refused("X03 alpha: the Hub tree listing has a malformed entry", "--download",
                            table=routes(**{TREE_A: json.dumps(bad_lfs).encode()}))
        self.assert_refused("X03 alpha: the Hub tree listing is paginated", "--download",
                            table=routes(**{TREE_A: (json.dumps(tree_a()).encode(), 200,
                                                     {"Link": '<https://huggingface.co/x?cursor=1>; rel="next"'})}))
        # json raises RecursionError (not ValueError) for a listing that is nested too deeply
        out, opener = self.assert_refused("X03 alpha: the Hub tree listing is nested too deeply", "--download",
                                          table=routes(**{TREE_A: b"[" * 200000}))
        self.assertIn("refused: 1 error(s)", out)
        self.assertNotIn(URL_A_CARD, opener.urls())

    def tree_case(self, needle, change, tree=tree_a, url=TREE_A):
        entries = tree()
        change(entries)
        out, opener = self.assert_refused(needle, "--download",
                                          table=routes(**{url: json.dumps(entries).encode()}))
        self.assertNotIn(URL_A_CARD if url == TREE_A else URL_B_CARD, opener.urls())   # no file is downloaded
        return out

    def test_x04_tree_entry_differs_from_the_pin(self):
        self.tree_case("X04 alpha README.md: not in the Hub tree", lambda t: t.pop(2))
        self.tree_case("X04 alpha README.md: the Hub tree differs from the pin (size 6)",
                       lambda t: t[2].update(size=6))
        self.tree_case(f"X04 alpha README.md: the Hub tree differs from the pin (oid '{BLOB_HELLO[:12]}')",
                       lambda t: t[2].update(oid=BLOB_HELLO))
        self.tree_case("X04 alpha README.md: the Hub tree differs from the pin (an LFS file)",
                       lambda t: t[2].update(lfs={"oid": SHA_CARD, "size": 5}))
        self.tree_case("X04 alpha data/part.parquet: the Hub tree differs from the pin (not an LFS file)",
                       lambda t: t[4].pop("lfs"))
        self.tree_case(f"X04 alpha data/part.parquet: the Hub tree differs from the pin (LFS sha256 '{SHA_HELLO[:12]}')",
                       lambda t: t[4]["lfs"].update(oid=SHA_HELLO))
        self.tree_case("X04 alpha data/part.parquet: the Hub tree differs from the pin (LFS size 13)",
                       lambda t: t[4]["lfs"].update(size=13))
        out = self.tree_case("X04 beta data/one.parquet: the Hub tree differs from the pin (size 7, LFS size 7)",
                             lambda t: (t[2].update(size=7), t[2]["lfs"].update(size=7)), tree=tree_b, url=TREE_B)
        self.assertNotIn("X04 alpha", out)

    def test_x05_file_set_of_the_revision(self):
        self.tree_case("X05 alpha: 1 file(s) of the revision are neither adopted nor excluded (first: 'data/new.parquet')",
                       lambda t: t.append({"type": "file", "oid": BLOB_X, "size": 2, "path": "data/new.parquet"}))
        self.tree_case("X05 alpha data/copy.parquet: excluded file is not in the Hub tree", lambda t: t.pop(3))
        self.tree_case(f"X05 alpha data/copy.parquet: excluded file has oid '{BLOB_X[:12]}'",
                       lambda t: t[3].update(oid=BLOB_X))
        self.tree_case("X05 alpha .gitattributes: excluded file has oid", lambda t: t[1].update(oid=BLOB_CARD))
        # a directory entry is not a file of the revision
        entries = tree_a() + [{"type": "directory", "oid": "0" * 40, "size": 0, "path": "docs"}]
        code, out, _ = self.run_main("--download", table=routes(**{TREE_A: json.dumps(entries).encode()}))
        self.assertEqual(code, 0, out)

    def test_x06_downloaded_bytes_differ(self):
        out, _ = self.assert_refused("X06 alpha README.md: downloaded bytes differ from the pin (sha256 ",
                                     "--download", table=routes(**{URL_A_CARD: b"cord\n"}))
        self.assertIn("git blob sha1 ", out)
        self.assert_refused("X06 alpha data/part.parquet: downloaded bytes differ from the pin (size 11, sha256 ",
                            "--download", table=routes(**{URL_A_PART: WORLD[:-1]}))
        self.assert_refused("X06 beta data/one.parquet: downloaded bytes differ from the pin (sha256 ",
                            "--download", table=routes(**{URL_B_ONE: b"hallo\n"}))
        # an error page of the pinned size is still refused
        self.assert_refused("X06 alpha README.md", "--download", table=routes(**{URL_A_CARD: b"<htm>"}))

    def test_one_failing_source_writes_nothing_for_any_source(self):
        out, opener = self.assert_refused("X06 beta README.md", "--download", table=routes(**{URL_B_CARD: b"HELLO\n"}))
        self.assertIn(URL_A_PART, opener.urls())                            # alpha itself was complete
        self.assertEqual(self.files(), [])

    def test_x07_existing_different_raw_file_is_not_overwritten(self):
        target = self.out / "alpha" / "data" / "part.parquet"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"local edit\n")
        self.assert_refused("X07 alpha data/part.parquet: a different file is already there (size 11, sha256 ",
                            "--download", written=["alpha/data/part.parquet"])
        self.assertEqual(target.read_bytes(), b"local edit\n")

    def test_x08_existing_different_manifest_is_not_overwritten(self):
        target = self.out / "beta" / F.MANIFEST
        target.parent.mkdir(parents=True)
        target.write_bytes(b"{}\n")
        self.assert_refused("X08 beta download_manifest.json: a different file is already there", "--download",
                            written=["beta/download_manifest.json"])
        self.assertEqual(target.read_bytes(), b"{}\n")

    def test_x10_write_error_is_reported(self):
        real, n = F._write, []

        def flaky(path, data):
            if len(n) == 2:
                raise PermissionError("denied")
            n.append(path)
            real(path, data)

        with mock.patch.object(F, "_write", side_effect=flaky):
            code, out, _ = self.run_main("--download")
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL X10 alpha download_manifest.json: cannot write (PermissionError)", out)
        self.assertIn("stopped: 1 error(s); 2 file(s) were written before the error", out)
        self.assertEqual(self.files(), ALL_FILES[:2])
        self.assertNotIn("OK 2 source(s)", out)

    def test_write_leaves_no_temporary_file(self):
        F._write(self.out / "a" / "b.bin", b"\x00\x01")
        self.assertEqual(self.files(), ["a/b.bin"])
        self.assertEqual((self.out / "a" / "b.bin").read_bytes(), b"\x00\x01")


class GuardTest(Base):
    """main() installs the test.csv guard before it reads a file or sends a request."""

    def run_recorded(self, *argv):
        events = []
        real_read, real_scandir, real_walk = F._read_local, os.scandir, os.walk

        class Recording(FakeOpener):
            def open(self, req, timeout=None):
                events.append("request")
                return super().open(req, timeout)

        def read(path):
            events.append("read")
            return real_read(path)

        def scandir(path):
            events.append("list")
            return real_scandir(path)

        def walk(top, **kw):
            events.append("list")
            return real_walk(top, **kw)

        with mock.patch.object(F.data_lists, "install_test_guard", side_effect=lambda: events.append("guard")), \
                mock.patch.object(F, "_read_local", side_effect=read), \
                mock.patch.object(F.os, "scandir", side_effect=scandir), \
                mock.patch.object(F.os, "walk", side_effect=walk):
            code, out, _ = self.run_main(*argv, opener=Recording(routes()))
        return code, out, events

    def test_guard_first_in_download_and_check(self):
        code, out, events = self.run_recorded("--download")
        self.assertEqual(code, 0, out)
        self.assertEqual(events[0], "guard")
        self.assertEqual(events.count("guard"), 1)
        self.assertEqual(events.count("request"), 6)
        self.write_rules()
        code, out, events = self.run_recorded()
        self.assertEqual(code, 0, out)
        self.assertEqual(events[0], "guard")
        self.assertEqual(events.count("guard"), 1)
        self.assertIn("read", events)
        self.assertIn("list", events)
        self.assertNotIn("request", events)

    def test_guard_is_installed_even_when_main_refuses(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        code, out, events = self.run_recorded("--download")
        self.assertEqual(code, 1, out)
        self.assertEqual(events, ["guard"])

    def test_module_uses_the_guard_of_data_lists(self):
        self.assertIs(F.data_lists, sys.modules["data_lists"])
        GUARD_CALLS.clear()
        code, out, _ = self.run_main()                                      # nothing downloaded: the check refuses
        self.assertEqual(code, 1, out)
        self.assertEqual(GUARD_CALLS, ["guard"])                            # the module-wide stand-in was called


class CheckTest(Base):
    def setUp(self):
        super().setUp()
        code, out, _ = self.run_main("--download")
        self.assertEqual(code, 0, out)
        self.write_rules()
        os.environ["HF_HUB_OFFLINE"] = "1"                                  # the check runs offline
        self.kept = ALL_FILES + [F.RULES_NAME]

    def check(self, *argv):
        code, out, opener = self.run_main(*argv, opener=FakeOpener({}))
        self.assertEqual(opener.calls, [], "the check must not use the network")
        return code, out

    def refused(self, needle, *argv):
        code, out = self.check(*argv)
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL " + needle, out)
        self.assertRegex(out, r"refused: \d+ error\(s\); nothing written\n$")
        return out

    def test_check_passes_after_download(self):
        before = {n: (self.out / n).read_bytes() for n in self.files()}
        code, out = self.check()
        self.assertEqual(code, 0, out)
        self.assertIn("ok alpha: 2 raw file(s) match the pins; download_manifest.json sha256 "
                      + hashlib.sha256(before["alpha/download_manifest.json"]).hexdigest()[:12], out)
        self.assertIn("check OK: 2 source(s), no network used", out)
        self.assertEqual({n: (self.out / n).read_bytes() for n in self.files()}, before)
        self.assertEqual(sorted(self.files()), sorted(self.kept))
        self.assertNotIn("hello", out)

    def test_x07_raw_file_missing_or_changed(self):
        target = self.out / "alpha" / "data" / "part.parquet"
        target.write_bytes(WORLD.replace(b"w", b"W"))
        self.refused("X07 alpha data/part.parquet: differs from the pin (sha256 ")
        target.write_bytes(WORLD + b"\n")
        self.refused("X07 alpha data/part.parquet: differs from the pin (size 13, sha256 ")
        target.unlink()
        self.refused("X07 alpha data/part.parquet: missing (run --download)")
        (self.out / "beta" / "README.md").write_bytes(b"hello\r\n")        # a CRLF checkout is a different file
        out = self.refused("X07 beta README.md: differs from the pin (size 7, sha256 ")
        self.assertIn("refused: 2 error(s)", out)

    def test_x07_file_or_folder_in_a_source_folder_that_is_not_pinned(self):
        opened = []
        real_read = F._read_local

        def read(path):
            opened.append(Path(path).relative_to(self.out).as_posix())
            return real_read(path)

        def refused_names_only(needle):
            with mock.patch.object(F, "_read_local", side_effect=read):
                out = self.refused(needle)
            self.assertIn("refused: 1 error(s)", out)
            return out

        stale = self.out / "alpha" / "data" / "old-00000.parquet"            # a file of an earlier revision
        stale.write_bytes(b"old bytes\n")
        out = refused_names_only("X07 alpha: 1 file(s) or folder(s) in the source folder are not pinned "
                                 "(first: 'data/old-00000.parquet'); remove them")
        self.assertNotIn("old bytes", out)
        stale.unlink()
        left = self.out / "alpha" / "README.md.tmp"                           # left by an interrupted write
        left.write_bytes(b"x")
        refused_names_only("X07 alpha: 1 file(s) or folder(s) in the source folder are not pinned "
                           "(first: 'README.md.tmp'); remove them")
        left.unlink()
        (self.out / "beta" / "extra" / "deep").mkdir(parents=True)            # a folder is one entry
        (self.out / "beta" / "extra" / "deep" / "a.bin").write_bytes(b"x")
        (self.out / "beta" / "data" / "sub").mkdir()                          # also when it is empty
        (self.out / "beta" / "zz.txt").write_bytes(b"x")
        refused_names_only("X07 beta: 3 file(s) or folder(s) in the source folder are not pinned "
                           "(first: 'data/sub'); remove them")
        self.assertEqual(sorted(set(opened)), sorted(self.kept))            # an unpinned file is never opened
        code, out = self.check("--source", "alpha")                         # only the selected source is listed
        self.assertEqual(code, 0, out)
        shutil.rmtree(self.out / "beta" / "extra")
        (self.out / "beta" / "data" / "sub").rmdir()
        (self.out / "beta" / "zz.txt").unlink()
        self.assertEqual(self.check()[0], 0)

    def test_unpinned_entries(self):
        src = SYN["alpha"]
        self.assertEqual(F.unpinned_entries(self.out / "alpha", src), ([], None))
        self.assertEqual(F.unpinned_entries(self.out / "no-such-source", src), ([], None))
        (self.out / "alpha" / "data" / "b.bin").write_bytes(b"x")
        (self.out / "alpha" / "a" / "b").mkdir(parents=True)
        (self.out / "alpha" / "a" / "b" / "c.bin").write_bytes(b"x")
        self.assertEqual(F.unpinned_entries(self.out / "alpha", src), (["a", "data/b.bin"], None))
        # a pin two folders deep: both folders on the way are accounted for
        deep = {"files": ({"path": "a/b/c.bin"},)}
        self.assertEqual(F.unpinned_entries(self.out / "alpha", deep),
                         (["README.md", "data"], None))

    def test_x07_folder_in_the_base_folder_that_is_not_a_pinned_source(self):
        (self.out / "stale-source").mkdir()
        (self.out / "stale-source" / F.MANIFEST).write_bytes(b"{}\n")
        out = self.refused("X07 base folder: 1 folder(s) are not pinned sources (first: 'stale-source'); remove them")
        self.assertIn("refused: 1 error(s)", out)
        self.refused("X07 base folder: 1 folder(s)", "--source", "beta")
        shutil.rmtree(self.out / "stale-source")
        code, out = self.check("--source", "beta")                          # alpha is a pinned source folder
        self.assertEqual(code, 0, out)
        (self.out / "notes.txt").write_bytes(b"x")                          # a file is not a source folder
        self.assertEqual(self.check()[0], 0)
        self.assertEqual(F.check_base(self.out / "no-such-folder", SYN), [])

    def test_x08_manifest_missing_or_changed(self):
        target = self.out / "alpha" / F.MANIFEST
        good = target.read_bytes()
        target.write_bytes(good.replace(b"\n", b"\r\n"))
        self.refused("X08 alpha download_manifest.json: differs from a fresh rendering")
        target.write_bytes(good.replace(b'"all_checks_passed": true', b'"all_checks_passed": false'))
        self.refused("X08 alpha download_manifest.json: differs from a fresh rendering")
        target.unlink()
        self.refused("X08 alpha download_manifest.json: missing (run --download)")

    def test_source_option_checks_only_that_source(self):
        (self.out / "alpha" / "README.md").unlink()
        code, out = self.check("--source", "beta")
        self.assertEqual(code, 0, out)
        self.assertNotIn("ok alpha", out)
        self.refused("X07 alpha README.md: missing", "--source", "alpha")

    def test_x09_rules_file_unreadable(self):
        rules = self.out / F.RULES_NAME
        good = rules.read_bytes()
        rules.unlink()
        self.refused("X09 ")
        self.write_rules(raw=good.replace(b"\n", b"\r\n"))
        self.refused("X09 ", )
        self.assertIn("not LF text", self.check()[1])
        self.write_rules(raw=good.rstrip(b"\n"))
        self.assertIn("not LF text", self.refused("X09 "))
        self.write_rules(raw=b"{not json\n")
        self.assertIn("not UTF-8 JSON", self.refused("X09 "))
        for raw in (b"[]\n", b'{"sources": []}\n', b'{"sources": {"alpha": {}}}\n'):
            self.write_rules(raw=raw)
            self.assertIn('"sources" does not hold exactly alpha, beta', self.refused("X09 "))
        self.write_rules(raw=b"[" * 200000 + b"\n")                         # json raises RecursionError here
        out = self.refused("X09 ")
        self.assertIn("JSON nested too deeply", out)
        self.assertIn("refused: 1 error(s)", out)

    def test_x09_every_pinned_key_is_compared(self):
        def mutate(change):
            self.write_rules(change)
            return self.refused("X09 ")

        a = lambda d: d["sources"]["alpha"]  # noqa: E731
        self.assertIn("sources.alpha.revision differs", mutate(lambda d: a(d).update(revision=REV_B)))
        self.assertIn("sources.alpha.dataset differs", mutate(lambda d: a(d).update(dataset="org-a/other")))
        self.assertIn("sources.alpha.dataset_url differs", mutate(lambda d: a(d).pop("dataset_url")))
        self.assertIn("sources.alpha.license differs", mutate(lambda d: a(d).update(license="apache-2.0")))
        self.assertIn("sources.alpha.dir differs", mutate(lambda d: a(d).update(dir="data/external/a")))
        self.assertIn("sources.alpha.manifest differs", mutate(lambda d: a(d).update(manifest="x")))
        self.assertIn("sources.alpha.manifest_sha256 differs", mutate(lambda d: a(d).update(manifest_sha256="0" * 64)))
        self.assertIn("sources.alpha.files differs", mutate(lambda d: a(d)["files"][1].update(sha256=SHA_HELLO)))
        self.assertIn("sources.alpha.files differs", mutate(lambda d: a(d)["files"][1].update(size_bytes=13)))
        self.assertIn("sources.alpha.files differs", mutate(lambda d: a(d)["files"].pop(0)))
        self.assertIn("sources.beta.files differs", mutate(lambda d: d["sources"]["beta"]["files"].reverse()))
        self.assertIn("sources.alpha is not an object", mutate(lambda d: d["sources"].update(alpha="x")))
        self.assertIn("sources.alpha.split_files does not give every adopted data file",
                      mutate(lambda d: a(d).update(split_files={})))
        self.assertIn("sources.alpha.split_files does not give every adopted data file",
                      mutate(lambda d: a(d).update(split_files={"part": "data/part.parquet", "card": "README.md"})))
        self.assertIn("sources.alpha.split_files does not give every adopted data file",
                      mutate(lambda d: a(d).pop("split_files")))
        self.assertIn("sources.alpha.adopted_splits differs", mutate(lambda d: a(d).update(adopted_splits=["x"])))
        self.write_rules()
        self.assertEqual(self.check()[0], 0)

    def test_rules_option_points_at_another_file(self):
        other = self.out.parent / "rules.json"
        shutil.copyfile(self.out / F.RULES_NAME, other)
        (self.out / F.RULES_NAME).unlink()
        code, out = self.check("--rules", str(other))
        self.assertEqual(code, 0, out)

    def test_x01_rules_path_is_a_test_csv_path_or_inside_datasets(self):
        # None of these paths exists and none is created; _read_local is replaced, so nothing can be opened.
        datasets = self.out.parent / "datasets"                             # stand-in for <repo>/datasets
        tmp = self.out.parent
        cases = {
            tmp / "test.csv": "X01 --rules is a test.csv path",
            tmp / "sub" / "Test.CSV": "X01 --rules is a test.csv path",
            str(tmp / "test.csv") + ".": "X01 --rules is a test.csv path",   # Windows opens test.csv for this name
            datasets / "x" / "test.csv": "X01 --rules is a test.csv path",
            datasets / "x" / "rules.json": "X01 --rules is inside datasets/",
            datasets: "X01 --rules is inside datasets/",
        }
        with mock.patch.object(F, "_read_local", side_effect=AssertionError("a file was read")), \
                mock.patch.object(F.build_lists, "DATASETS", datasets):
            for path, needle in cases.items():
                out = self.refused(needle, "--rules", str(path))
                self.assertIn("refused: 1 error(s)", out)
                out = self.refused(needle, "--rules", str(path), "--download")
                self.assertIn("FAIL X01 --rules belongs to the check", out)
        self.assertFalse(datasets.exists())
        self.assertEqual(sorted(self.files()), sorted(self.kept))


class RepoRecordsTest(unittest.TestCase):
    """Committed records under data/external/ against the real pins (JSON metadata only; raw files are not opened)."""

    def test_rules_file_agrees_with_the_pins(self):
        self.assertEqual(F.check_rules(F.DEFAULT_OUT / F.RULES_NAME, F.SOURCES), [])

    def test_manifests_equal_a_fresh_rendering(self):
        for name, src in F.SOURCES.items():
            raw = (F.DEFAULT_OUT / name / F.MANIFEST).read_bytes()
            self.assertEqual(raw, F.render_manifest(src), name)
            doc = json.loads(raw)
            self.assertEqual(doc["revision"], src["revision"])
            self.assertTrue(doc["all_checks_passed"])
            self.assertEqual([f["path"] for f in doc["files"]], [f["path"] for f in src["files"]])

    def test_rules_record_the_decisions_of_the_design(self):
        doc = json.loads((F.DEFAULT_OUT / F.RULES_NAME).read_bytes().decode("utf-8"))
        src = doc["sources"]
        self.assertEqual(src["deepset"]["adopted_splits"], ["train", "test"])
        self.assertEqual(src["deepset"]["split_files"],
                         {"train": "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
                          "test": "data/test-00000-of-00001-701d16158af87368.parquet"})
        self.assertEqual(src["notinject"]["adopted_splits"], ["NotInject_one", "NotInject_two", "NotInject_three"])
        self.assertEqual(src["deepset"]["use"], "evaluation only")
        self.assertEqual(src["notinject"]["use"], "evaluation only")
        labels = doc["labels"]
        self.assertEqual(labels["project"], {"0": "benign", "1": "injection"})
        self.assertEqual(labels["deepset"]["mapping"], {"0": 0, "1": 1})
        self.assertEqual(labels["notinject"]["input_field"], "prompt")
        self.assertEqual(labels["notinject"]["all_rows"], 0)
        dup = doc["duplicates"]
        self.assertEqual(dup["against_s_labs"]["removed_levels"], ["A", "B"])
        self.assertEqual(dup["against_s_labs"]["counted_levels"], ["C"])
        self.assertEqual(dup["against_s_labs"]["compared_with"], ["train", "validation", "test", "augmentation"])
        self.assertEqual(dup["within_source"]["keep"], "first")
        self.assertEqual(dup["cross_source"]["action"], "recorded")
        self.assertEqual(doc["length"]["max_length"], 256)
        self.assertEqual(doc["length"]["truncation"], True)
        self.assertEqual(doc["length"]["length_based_exclusion"], False)
        self.assertEqual(doc["sampling"]["rule"], "none")
        self.assertEqual(doc["conversion"]["tool"], "pyarrow")
        self.assertEqual(doc["deferred_to_stage_5"], ["conversion", "review", "list building"])


if __name__ == "__main__":
    unittest.main()
