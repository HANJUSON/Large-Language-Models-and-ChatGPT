"""Build the common sentence lists (README 3.2 DATA-01..06, 7.1) from the pinned raw splits.

Reads datasets/s-labs-prompt-injection/download_manifest.json (its SHA-256 is
pinned), the three raw splits it lists, data/processed/train_review.csv and,
when it exists, the frozen augmentation list data/cases/train/train_aug.csv
with its cases.csv (final stage, below); checks every raw file against the
manifest (revision, size, SHA-256) before parsing; writes ID lists (never
sentence text) and lists_meta.json. Standard library only.

Rules (README line numbers as of the commit that adds this script):
  - Sentence ID `split:NNNNNN`: 0-based index of the data records read with the
    Python csv module (UTF-8, newline=''), header excluded (7.1 L460). train
    has multi-line records (train:000994, train:009878), so IDs are not lines.
  - Exact key = the raw `text` field, unchanged. Normalised key = NFKC ->
    strip and collapse whitespace -> casefold (3.1 L119-120); used for the
    train review groups, the train/validation audit and the similar-sentence
    check. Punct key = normalised key with every Unicode P* (punctuation) and
    S* (symbol) character replaced by a space, whitespace collapsed; an empty
    punct key is no key (the row takes no part in the similar-sentence check).
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
  - Stages (README 3.2 DATA-05, 7.1). No train_aug.csv in the augmentation
    folder (--augmentation, default data/cases/train) = pre-augmentation
    stage: the eight files below, "test_clean_stage": "pre-augmentation",
    "augmentation": null. train_aug.csv exists = final stage (augmentation
    freeze): the same eight files with the same rows plus test_clean.csv and
    test_excluded.csv, "test_clean_stage": "final" and an "augmentation"
    record (folder, SHA-256 of train_aug.csv and cases.csv, rows, label
    counts, level C pair counts).
  - Augmentation input (final stage, stop code B09). train_aug.csv: ASCII, LF,
    header sentence_id,label,text_sha256_12, sentence_id = augmentation case
    ID (aug-01-<case|space|char>-NNNNNN, aug-03-NNNNNN, aug-04-NNN,
    aug-05-NNN), strictly ascending, no quoting. cases.csv: the 16 case
    columns (case_id ... text), read with the csv module (strict), text = the
    field unchanged. Every listed case must exist in cases.csv with status
    `assigned`, the same label and h12(text) equal to text_sha256_12.
    No listed case may equal a row of the whole original validation or of test
    at level A (exact key) or B (normalised key, punct key; decided by
    similar_pairs like every other B pair): scripts/build_train_aug.py must
    have removed such cases already, so a match is a stop, not an exclusion.
    Level C (one_word_edit) pairs augmentation-validation and
    augmentation-test are counted only (lists_meta.json "augmentation").
  - test_clean (final stage): test minus rows whose exact key is in the
    cleaned train, the augmentation cases or the whole original validation,
    same precedence of reasons. Because B09 allows no augmentation case that
    equals a test row, a successful final build has augmentation_overlap 0
    and test_clean.csv holds the rows of test_clean_preaug.csv; the
    similar-sentence statistics of test_clean_preaug therefore describe
    test_clean too. In the final stage the non-exclusive overlap counts
    ("audit" test.augmentation_overlap_rows, EXPECTED "test_overlap") are
    those of the final build, i.e. counted against the real augmentation
    cases.
  - test use (3.2 L176-184): automated duplicate IDs and counts only.
  - Similar-sentence check (README 3.2, DATA-06). Levels of a pair of rows
    whose exact keys differ, each pair at its first level: A exact key
    (automatic exclusion, above); B `normalised_key` (normalised keys equal)
    or `punct_key` (punct keys equal, normalised keys differ); C
    `one_word_edit` (punct-key word sequences differ by exactly one word
    inserted, deleted or substituted; the shorter has >= 5 words), a
    template-closeness statistic only. Decided from SHA-256 digests only
    (sim_keys): a substitution shares the digest of the words without word i
    at the same i and word count; an insertion has the shorter's punct-key
    digest among the longer's one-word-deletion digests.
    Scopes: train_clean internal (B group = the >= 2 rows sharing one punct
    key; one label -> all rows kept; >= 2 labels -> B08, the only way the
    check could remove train rows); validation_clean internal and vs every
    original train row; test_clean_preaug internal (B only) and vs every
    original train row and the whole original validation. Cross-split B and
    C rows are counted and listed for reporting, never excluded (the lists
    keep the exact-key definition). similar_pairs.csv lists every B and C
    pair except test C pairs (counted only). Counts are in lists_meta.json
    "similar" and pinned in EXPECTED.
  - Review record contract (7.1 L461): stop codes R01-R23 below. A confirmed
    row needs at least one member ID in confirmed_by (one member's sign-off is
    enough, 2026-11-13); a proposed row at most one. confirmed_by
    IDs compare case-insensitively; empty tokens are R09 and are not counted
    by R20/R23; R23 compares each row's sorted ID set, so the order is free.
  - Expected counts are constants (EXPECTED) citing README; a legitimate
    decision change edits README 3.2 and EXPECTED together (B02). The
    final-stage counts are in EXPECTED_FINAL; an entry that is still None is
    not pinned: the final build then stops with B02, prints the actual counts
    to pin and writes nothing (official lists never carry unpinned counts).
  - Contract gap: a normalised-key variant of a conflict text forms a 1-row
    normkey_only group whose label is not compared with the label kept in the
    conflict group by the review checks (cannot occur at the pinned revision);
    when both rows stay in train_clean with differing labels, B08 stops.

Outputs (UTF-8 without BOM, LF, header, trailing LF, sorted by sentence_id):
  train_clean.csv           sentence_id,label,text_sha256_12
  train_excluded.csv        sentence_id,label,text_sha256_12,reason,ref
                            (duplicate_same_label -> kept ID; review_exclude -> group_id)
  validation_clean.csv      sentence_id,label,text_sha256_12
  validation_excluded.csv   sentence_id,label,text_sha256_12,reason,ref
                            (train_overlap -> lowest original train ID)
  test_clean_preaug.csv     sentence_id
  test_excluded_preaug.csv  sentence_id,reason
  test_clean.csv            sentence_id          (final stage only)
  test_excluded.csv         sentence_id,reason   (final stage only)
  similar_pairs.csv         level,sentence_id,other_id (level normalised_key,
                            punct_key or one_word_edit; sentence_id is the
                            train_clean / validation_clean / test_clean_preaug
                            row, the lower ID for internal pairs; other_id the
                            row it is similar to; sorted by sentence_id, other_id)
  lists_meta.json           sorted keys, indent 2, ASCII; sources, review, rules,
                            counts by reason, audit, similar, output SHA-256; no
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
    freeze-1 code must require it to pass. Eight files in the
    pre-augmentation stage, ten in the final stage; in the pre-augmentation
    stage a test_clean.csv / test_excluded.csv left in <out-dir>/ fails it.
  Write sequence: validate everything in memory; remove a stale <name>.tmp of
  every output; delete <target>/lists_meta.json; in the pre-augmentation
  stage remove a test_clean.csv / test_excluded.csv of an earlier final-stage
  build; write each file to <name>.tmp (exclusive create) and os.replace it;
  meta last; on an exception the .tmp files are removed and the run fails
  with B05 (no meta is left).
  Output locations (B06, checked before any raw file is read):
    - --out-dir and --augmentation never under datasets/ (each ancestor
      compared by file identity);
    - a write to the default out-dir (official, or its trial/) and --check on
      it need the default --review, --manifest and --augmentation;
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
  B02 count != EXPECTED (actual and expected printed); final stage: count !=
      EXPECTED_FINAL, or an EXPECTED_FINAL entry not pinned yet (actual printed)
  B03 invariant broken (partition by label, disjointness, unique train_clean
      keys, test_clean disjoint from validation and cleaned train, review
      exclude rows listed as review_exclude, refs; final stage: test_clean +
      test_excluded partition test, test_clean disjoint from the cleaned
      train, the augmentation and validation)
  B04 output line grammar, or a non-ASCII stdout line
  B05 write failure (PermissionError etc.)
  B06 refused location or mode (see Output locations); --check with
      --allow-proposed, with proposed review rows, or with missing, trial or
      differing files (a differing lists_meta.json names up to 10 differing
      keys) or final-stage files left in a pre-augmentation out-dir; an event
      refused by the audit hook
  B07 unexpected exception: type and line in this script only, message suppressed
  B08 train_clean B group (normalised or punct key) with >= 2 labels: human
      review needed (README 3.2); train_review.csv has no check type for it yet
  B09 augmentation input (final stage): train_aug.csv or cases.csv unreadable
      or malformed (encoding, header, field count, case ID, label, hash
      format, order, canonical form, duplicate case_id); a listed case missing
      in cases.csv, not `assigned`, with another label, another text hash or
      an empty text; a listed case equal to a validation or test row at level
      A or B (case IDs only)

Test-safety guarantees: test.csv is read once, by load_test() only, as bytes
whose size and SHA-256 are checked before parsing; inside load_test each text
is reduced to SHA-256 digests (exact key, normalised key, punct key, its
one-word deletions with positions, plus the word count) and the texts are
dropped; they are never printed, logged or written; exceptions inside load_test
are reported by type and record number only. Test lists and similar_pairs.csv
carry test IDs (B levels) and reasons only; stdout is ASCII, counts and IDs.
The augmentation cases are compared with test through these digests only
(check_augmentation, build_test); a B09 match names augmentation case IDs.
Augmentation text is never printed or written either: reader errors carry
record numbers, or case IDs that passed the ID grammar.
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
under another name are not covered. README's whole-split normalised-key figures
for test (3.2 L144, L147) are not re-checked as such; the lists compare test by
exact key only, the similar block compares test_clean_preaug by digests.

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
        (while the review has proposed rows -> trial lists in data/processed/trial/)
    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/build_lists.py
        (after a member confirms every review row -> official lists)
    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/build_lists.py --check
Options: --review PATH, --manifest PATH, --out-dir PATH, --augmentation DIR
(defaults resolve from the repo root, not the cwd).

Pinning the final stage (once, at the augmentation freeze): run the default
build; it stops with one `FAIL B02 <key>: final-stage count not pinned; actual
{...}` line per EXPECTED_FINAL entry and writes nothing. Copy each printed
value into EXPECTED_FINAL (the only place), state the same counts in README
3.2, rerun the default build (ten files, "test_clean_stage": "final") and
commit the lists with this script.

Out of scope: dropping augmentation cases. A case whose exact, normalised or
punct key equals a validation or test row is removed by
scripts/build_train_aug.py with scripts/check_cases_vs_test.py; here such a
case stops the build (B09). The similar-sentence rule for external rows
(README 3.2: drop an external row whose key equals an S-Labs split row or an
augmentation case; count C) is applied when those inputs exist, not here.

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
DEFAULT_AUGMENTATION = ROOT / "data" / "cases" / "train"  # README 7.1; written by scripts/build_train_aug.py
AUG_LIST = "train_aug.csv"  # the usable cases; its presence selects the final stage
AUG_CASES = "cases.csv"     # case_id -> label, status, text
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
    # Similar-sentence check (README 3.2 similar-sentence block, DATA-06) = lists_meta.json "similar".
    "similar.train_clean": {"normalised_key": {"groups": 1, "rows": 2, "label_disagree_groups": 0},
                            "punct_key": {"groups": 18, "rows": 36, "label_disagree_groups": 0},
                            "one_word_edit": {"pairs": 851, "label_disagree_pairs": 1}},
    "similar.validation_clean": {"normalised_key": {"groups": 0, "rows": 0, "label_disagree_groups": 0},
                                 "punct_key": {"groups": 0, "rows": 0, "label_disagree_groups": 0},
                                 "one_word_edit": {"pairs": 78, "label_disagree_pairs": 0}},
    "similar.validation_clean_vs_original_train": {
        "normalised_key": {"rows": 0, "pairs": 0, "label_disagree_pairs": 0},
        "punct_key": {"rows": 2, "pairs": 2, "label_disagree_pairs": 0},
        "one_word_edit": {"rows": 12, "pairs": 12, "label_disagree_pairs": 0}},
    "similar.test_clean_preaug": {"normalised_key": {"groups": 0, "rows": 0, "label_disagree_groups": 0},
                                  "punct_key": {"groups": 0, "rows": 0, "label_disagree_groups": 0}},
    "similar.test_clean_preaug_vs_original_train": {
        "normalised_key": {"rows": 0, "pairs": 0, "label_disagree_pairs": 0},
        "punct_key": {"rows": 1, "pairs": 1, "label_disagree_pairs": 0},
        "one_word_edit": {"rows": 7, "pairs": 7, "label_disagree_pairs": 0}},
    "similar.test_clean_preaug_vs_validation": {
        "normalised_key": {"rows": 0, "pairs": 0, "label_disagree_pairs": 0},
        "punct_key": {"rows": 0, "pairs": 0, "label_disagree_pairs": 0},
        "one_word_edit": {"rows": 212, "pairs": 271, "label_disagree_pairs": 0}},
}

# Final-stage counts (README 3.2 DATA-05: test_clean.csv at the augmentation freeze), checked in
# addition to EXPECTED when <augmentation>/train_aug.csv exists. None = not pinned yet: the final build
# stops with B02, prints the actual counts of that entry and writes nothing. This dict is the only place
# to pin them (with README 3.2), from the B02 lines of one run on the frozen augmentation. B09 allows no
# augmentation case that equals a test row, so test_clean / test_excluded can only repeat the
# pre-augmentation counts above (augmentation_overlap 0).
EXPECTED_FINAL = {  # pinned 2026-11-17 on the reviewed augmentation (README 3.2, 5.3)
    "augmentation": {"rows": 1697, "0": 850, "1": 847},  # rows of train_aug.csv, by label
    "augmentation_level_c": {"validation": 0, "test": 0},  # one_word_edit pairs, counted only
    "test_clean": {"rows": 1985, "0": 1043, "1": 942},
    "test_excluded": {"rows": 116, "train_overlap": 0, "augmentation_overlap": 0, "validation_overlap": 116},
}

OUTPUTS = ("train_clean", "train_excluded", "validation_clean", "validation_excluded",
           "test_clean_preaug", "test_excluded_preaug")
FINAL_OUTPUTS = ("test_clean", "test_excluded")  # final stage only, in addition to OUTPUTS
SIMILAR = "similar_pairs"  # README 3.2 similar-sentence check; not a clean/excluded list
SIM_LEVELS = ("normalised_key", "punct_key", "one_word_edit")  # B, B, C
MIN_EDIT_WORDS = 5  # one_word_edit: the shorter word sequence has at least this many words
HEADERS = {
    "train_clean": "sentence_id,label,text_sha256_12",
    "train_excluded": "sentence_id,label,text_sha256_12,reason,ref",
    "validation_clean": "sentence_id,label,text_sha256_12",
    "validation_excluded": "sentence_id,label,text_sha256_12,reason,ref",
    "test_clean_preaug": "sentence_id",
    "test_excluded_preaug": "sentence_id,reason",
    "test_clean": "sentence_id",
    "test_excluded": "sentence_id,reason",
    SIMILAR: "level,sentence_id,other_id",
}
LINE_GRAMMAR = {k: re.compile(v) for k, v in {
    "train_clean": r"train:[0-9]{6},[01],[0-9a-f]{12}",
    "train_excluded": r"train:[0-9]{6},[01],[0-9a-f]{12},"
                      r"(?:duplicate_same_label,train:[0-9]{6}|review_exclude,T-(?:CONFLICT|NORMKEY)-[1-9][0-9]*)",
    "validation_clean": r"validation:[0-9]{6},[01],[0-9a-f]{12}",
    "validation_excluded": r"validation:[0-9]{6},[01],[0-9a-f]{12},train_overlap,train:[0-9]{6}",
    "test_clean_preaug": r"test:[0-9]{6}",
    "test_excluded_preaug": r"test:[0-9]{6},(?:train_overlap|augmentation_overlap|validation_overlap)",
    "test_clean": r"test:[0-9]{6}",
    "test_excluded": r"test:[0-9]{6},(?:train_overlap|augmentation_overlap|validation_overlap)",
    SIMILAR: r"(?:normalised_key|punct_key|one_word_edit),"
             r"(?:train:[0-9]{6},train|validation:[0-9]{6},(?:train|validation)):[0-9]{6}"
             r"|(?:normalised_key|punct_key),test:[0-9]{6},(?:train|validation|test):[0-9]{6}",
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

# Augmentation input (final stage, B09). The header, the case columns, the case ID grammar and the usable
# status are those of scripts/build_train_aug.py (TRAIN_AUG_HEADER, CASE_COLUMNS, CASE_ID_RE, `assigned`).
AUG_LIST_HEADER = "sentence_id,label,text_sha256_12"
AUG_CASE_COLUMNS = ("case_id", "group_id", "source_split", "source_id", "purpose", "condition", "subtype",
                    "eval_group", "label", "position", "context_ids", "generator", "generator_version",
                    "status", "auto_flags", "text")
AUG_ID_RE = re.compile(r"aug-01-(?:case|space|char)-[0-9]{6}|aug-03-[0-9]{6}|aug-0[45]-[0-9]{3}")
AUG_USABLE_STATUS = "assigned"
AUG_REFERENCES = ("validation", "test")  # whole original splits compared with the augmentation cases

RULES = {
    "sentence_id": "split:NNNNNN = 0-based index of the data records read with Python csv "
                   "(UTF-8, newline=''), header excluded (README 7.1)",
    "exact_key": "raw text field, unchanged (README 3.1)",
    "normalised_key": "NFKC, strip and collapse whitespace, casefold (README 3.1); used for train "
                      "review groups, the train/validation audit and the similar-sentence check; on test "
                      "only as a SHA-256 digest inside load_test",
    "punct_key": "normalised key with every Unicode P* (punctuation) and S* (symbol) character replaced by "
                 "a space, whitespace collapsed; an empty punct key is no key (README 3.2)",
    "similar_levels": "pairs whose exact keys differ, each at its first level: B normalised_key (normalised "
                      "keys equal), B punct_key (punct keys equal, normalised keys differ), C one_word_edit "
                      "(punct-key word sequences differ by one word inserted, deleted or substituted, the "
                      f"shorter has >= {MIN_EDIT_WORDS} words); decided from SHA-256 digests only (README 3.2)",
    "similar_scope": "train_clean internal: a B group (one punct key, >= 2 rows) with >= 2 labels stops the "
                     "build (B08), otherwise all rows are kept; validation_clean internal and vs every original "
                     "train row; test_clean_preaug internal (B only) and vs every original train row and the "
                     "whole original validation; cross-split pairs are counted and listed, never excluded; "
                     "similar_pairs.csv lists every B and C pair except test C pairs (counted only)",
    "text_sha256_12": "first 12 lowercase hex characters of SHA-256 of the raw text field in UTF-8",
    "train_clean": "drop review rows with decision exclude, then keep the lowest sentence ID of "
                   "every exact-key group (README 3.2)",
    "validation_clean": "original validation minus rows whose exact key is in the original train "
                        "(README 3.1)",
    "test_clean_preaug": "test minus rows whose exact key is in the cleaned train, the augmentation "
                         "cases (none before the augmentation freeze) or the whole original "
                         "validation (README 3.2 DATA-05); one primary reason per row by precedence "
                         "train_overlap > augmentation_overlap > validation_overlap",
    "test_audit": "the lists and the audit compare test by exact key only (SHA-256 digests); the similar "
                  "block compares test_clean_preaug by digests of the normalised key, the punct key and its "
                  "one-word deletions (no test text is kept)",
}
# Added to lists_meta.json "rules" in the final stage only (the pre-augmentation meta keeps RULES as is).
FINAL_RULES = {
    "augmentation": f"the cases listed in {AUG_LIST} of the augmentation folder (sentence_id = case ID, label, "
                    f"text_sha256_12; strictly ascending), text from {AUG_CASES}; every listed case has status "
                    f"{AUG_USABLE_STATUS} there with the same label and text hash; no listed case equals a row "
                    "of the whole original validation or of test at level A (exact key) or B (normalised_key, "
                    "punct_key), otherwise the build stops (B09); level C (one_word_edit) pairs with the whole "
                    "original validation and with test are counted only (README 3.2)",
    "test_clean": "final list at the augmentation freeze: test minus rows whose exact key is in the cleaned "
                  "train, the augmentation cases or the whole original validation (README 3.2 DATA-05); one "
                  "primary reason per row by precedence train_overlap > augmentation_overlap > "
                  "validation_overlap; no augmentation case equals a test row (B09), so the rows are those of "
                  "test_clean_preaug and its similar-sentence statistics apply",
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


def fmt_recs(numbers, limit=10):
    """Like fmt_ids for 1-based record numbers (header excluded), in file order."""
    shown = ", ".join(str(n) for n in numbers[:limit])
    more = f", ... (+{len(numbers) - limit} more)" if len(numbers) > limit else ""
    return f"{len(numbers)} (record {shown}{more})"


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


def pk(text):
    """Punct key: normalised key, every Unicode P*/S* character -> space, whitespace collapsed (README 3.2)."""
    return " ".join("".join(" " if unicodedata.category(c)[0] in "PS" else c for c in nk(text)).split())


def sim_keys(text):
    """Similar-sentence keys of one text, SHA-256 digests only (README 3.2).

    Returns (normalised-key digest, punct-key digest or None when the punct key is empty, word count of
    the punct key, ((i, digest of the punct-key words without word i), ...)). The deletion neighbourhood
    is kept for texts of at least MIN_EDIT_WORDS words only (shorter ones cannot form a one_word_edit pair).
    """
    words = pk(text).split()
    n = len(words)
    dels = tuple((i, digest(" ".join(words[:i] + words[i + 1:]))) for i in range(n)) if n >= MIN_EDIT_WORDS else ()
    return digest(nk(text)), (digest(" ".join(words)) if words else None), n, dels


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
    """The only reader of test.csv. Returns ([(sid, label, digest, sim_keys)], errors).

    One read (the audit hook allows exactly one open while this runs), SHA-256 check
    before parsing, each text reduced to its exact-key digest and its similar-sentence
    digests (sim_keys) here, so no text leaves this function; any exception is reported
    by type only, its message suppressed.
    """
    _GUARD["test_reads"] = 1
    try:
        data, errors = read_verified(path, sha256, size, "test.csv")
        _GUARD["test_reads"] = 0
        if data is None:
            return [], errors
        parsed, perrors = parse_split(data, "test", expected)
        return [(sid, label, digest(txt), sim_keys(txt)) for sid, label, txt in parsed], errors + perrors
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
            if r["status"] == "confirmed" and not ids:
                why.append("confirmed needs at least 1 member ID")
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

    test_digests: [(sid, label, digest, ...)] as returned by load_test (only the first and
    third fields are used). One primary reason by precedence train_overlap >
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
        sid, dg = row[0], row[2]
        for key, pool in pools:
            overlap[key] += dg in pool
        reason = next((name for name, pool in sources if dg in pool), None)
        if reason is None:
            clean.append((sid,))
        else:
            excluded.append((sid, reason))
    return sorted(clean), sorted(excluded), {k: overlap[k] for k in OVERLAP_KEYS}


def count_lists(lists, test_labels, overlap, stems=OUTPUTS):
    """Counts compared with EXPECTED: rows and labels for clean lists, rows and reasons for excluded."""
    counts = {}
    for stem in stems:
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


def check_expected(counts, expected=EXPECTED, name="EXPECTED"):
    """B02: every count must equal the README constants (actual and expected printed).

    An entry that is None (EXPECTED_FINAL before the augmentation freeze) is not pinned: B02 with the
    actual counts, so nothing is written until the entry holds them.
    """
    errors = []
    for stem, want in expected.items():
        got = counts.get(stem, {})
        if want is None:
            errors.append(f"B02 {stem}: final-stage count not pinned; actual {json.dumps(got)}; set "
                          f"{name}[\"{stem}\"] in scripts/build_lists.py to this value at the augmentation "
                          f"freeze, together with README 3.2, and rerun")
            continue
        diffs = [f"{k} {got.get(k, 0)} (expected {want.get(k, 0)})"
                 for k in sorted(set(want) | set(got)) if got.get(k, 0) != want.get(k, 0)]
        if diffs:
            errors.append(f"B02 {stem}: {', '.join(diffs)}; a legitimate decision change edits README 3.2 "
                          f"and {name} in scripts/build_lists.py together")
    return errors


def check_invariants(train, validation, test, review_rows, lists, augmentation_digests=None):
    """B03 invariants on the built lists (train/validation rows (sid, label, text), test rows (sid, label, digest, ...)).

    augmentation_digests (final stage only; exact-key digests of the augmentation cases) adds the checks
    of test_clean / test_excluded: partition of test, test_clean disjoint from the cleaned train, the
    augmentation and validation.
    """
    errors = []
    tr = {sid: (label, text) for sid, label, text in train}
    va = {sid: (label, text) for sid, label, text in validation}
    te = {sid: (label, dg) for sid, label, dg, *_ in test}

    def partition(split, src, clean, excl, name=None):
        name = name or split
        c_ids, x_ids = [r[0] for r in lists[clean]], [r[0] for r in lists[excl]]
        if len(set(c_ids)) != len(c_ids) or len(set(x_ids)) != len(x_ids) or set(c_ids) & set(x_ids):
            errors.append(f"B03 {name}: clean and excluded lists are not disjoint")
        if set(c_ids) | set(x_ids) != set(src):
            errors.append(f"B03 {name}: clean + excluded does not cover the source rows")
        if Counter(src[s][0] for s in c_ids + x_ids if s in src) != Counter(v[0] for v in src.values()):
            errors.append(f"B03 {name}: clean + excluded label counts differ from the source")
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
    pools = [("validation", val_dg), ("the cleaned train", cleaned_dg)]
    checks = [("test_clean_preaug", pools)]
    if augmentation_digests is not None:
        partition("test", te, "test_clean", "test_excluded", "test (final lists)")
        checks.append(("test_clean", pools + [("the augmentation", set(augmentation_digests))]))
    for stem, against in checks:
        test_clean_dg = {te[r[0]][1] for r in lists[stem] if r[0] in te}
        for name, pool in against:
            if test_clean_dg & pool:
                errors.append(f"B03 {stem}: {len(test_clean_dg & pool)} row(s) share an exact key with {name}")
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
    over = [(label, val_by_dg[dg]) for _, label, dg, *_ in test if dg in val_by_dg]
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
            "internal_exact_duplicate_rows": len(test) - len({r[2] for r in test}),
            "original_train_overlap_rows": overlap["original_train"],
            "cleaned_train_overlap_rows": overlap["cleaned_train"],
            "augmentation_overlap_rows": overlap["augmentation"],
            "validation_overlap_rows": overlap["validation"],
            "validation_overlap_labels": {k: sum(t == k for t, _ in over) for k in LABELS},
            "validation_overlap_labels_agree": all(t == v for t, v in over),
            "note": "exact key only; README normalised-key figures for test are not re-checked",
        },
    }


# ---------------------------------------------------------------- similar sentences (README 3.2)

def similar_pairs(subject, reference, edit=True):
    """{(sid, other_sid): level} for subject x reference rows whose exact keys differ.

    Rows are (sid, label, exact digest, sim_keys). Levels as in RULES["similar_levels"];
    one_word_edit only with edit. A row never pairs with itself; for an internal scope pass
    the same rows twice and keep sid < other_sid.
    """
    by_pk, by_sub, by_del, by_full = {}, {}, {}, {}
    for sid, _, exact, (nkd, p, n, dels) in reference:
        if p is None:
            continue
        by_pk.setdefault(p, []).append((sid, exact, nkd))
        if edit and n >= MIN_EDIT_WORDS:
            by_full.setdefault((n, p), []).append(sid)
            for i, h in dels:
                by_sub.setdefault((i, n, h), []).append((sid, p))
                by_del.setdefault((n, h), []).append(sid)
    pairs = {}
    for sid, _, exact, (nkd, p, n, dels) in subject:
        if p is None:
            continue
        for other, o_exact, o_nkd in by_pk.get(p, ()):
            if o_exact != exact:
                pairs[(sid, other)] = "normalised_key" if o_nkd == nkd else "punct_key"
        if not edit or n < MIN_EDIT_WORDS:
            continue
        near = set(by_del.get((n + 1, p), ()))  # the other has one word more
        for i, h in dels:
            near.update(o for o, o_p in by_sub.get((i, n, h), ()) if o_p != p)  # word i substituted
            near.update(by_full.get((n - 1, h), ()))  # the other has one word less
        for other in near:
            pairs[(sid, other)] = "one_word_edit"
    return pairs


def similar_check(train, validation, test, lists):
    """Similar-sentence check (README 3.2). Returns (similar_pairs.csv rows, stats, errors).

    train/validation rows (sid, label, text); test rows as returned by load_test (digests only).
    stats[scope][level] holds ints: internal scopes groups / rows / label_disagree_groups for the B
    levels (a punct_key group is one punct key over >= 2 normalised keys) and pairs /
    label_disagree_pairs for one_word_edit; cross scopes rows (subject rows with >= 1 pair) / pairs /
    label_disagree_pairs per level. errors: B08 for a train_clean B group with >= 2 labels.
    """
    items = {split: [(sid, label, digest(text), sim_keys(text)) for sid, label, text in rows]
             for split, rows in (("train", train), ("validation", validation))}
    items["test"] = list(test)
    label = {r[0]: r[1] for rows in items.values() for r in rows}

    def clean(split, stem):
        ids = {r[0] for r in lists[stem]}
        return [r for r in items[split] if r[0] in ids]

    train_c, val_c, test_c = (clean("train", "train_clean"), clean("validation", "validation_clean"),
                              clean("test", "test_clean_preaug"))
    rows_out, stats, errors = [], {}, []
    for scope, subject, reference, edit, listed in (
            ("train_clean", train_c, None, True, SIM_LEVELS),
            ("validation_clean", val_c, None, True, SIM_LEVELS),
            ("validation_clean_vs_original_train", val_c, items["train"], True, SIM_LEVELS),
            ("test_clean_preaug", test_c, None, False, SIM_LEVELS[:2]),
            ("test_clean_preaug_vs_original_train", test_c, items["train"], True, SIM_LEVELS[:2]),
            ("test_clean_preaug_vs_validation", test_c, items["validation"], True, SIM_LEVELS[:2])):
        pairs = similar_pairs(subject, subject if reference is None else reference, edit)
        if reference is None:
            pairs = {k: v for k, v in pairs.items() if k[0] < k[1]}
        rows_out += [(level, a, b) for (a, b), level in pairs.items() if level in listed]
        by_level = {level: [k for k, v in pairs.items() if v == level] for level in SIM_LEVELS[:3 if edit else 2]}
        if reference is not None:
            stats[scope] = {level: {"rows": len({a for a, _ in ks}), "pairs": len(ks),
                                    "label_disagree_pairs": sum(label[a] != label[b] for a, b in ks)}
                            for level, ks in by_level.items()}
            continue
        by_nk, by_pk = {}, {}
        for sid, _, _, (nkd, p, _, _) in subject:
            if p is not None:
                by_nk.setdefault(nkd, []).append(sid)
                by_pk.setdefault(p, []).append((sid, nkd))
        groups = {"normalised_key": [g for g in by_nk.values() if len(g) > 1],
                  "punct_key": [[s for s, _ in g] for g in by_pk.values() if len({n for _, n in g}) > 1]}
        mixed = {level: [g for g in gs if len({label[s] for s in g}) > 1] for level, gs in groups.items()}
        stats[scope] = {level: {"groups": len(gs), "rows": sum(map(len, gs)),
                                "label_disagree_groups": len(mixed[level])} for level, gs in groups.items()}
        if edit:
            ks = by_level["one_word_edit"]
            stats[scope]["one_word_edit"] = {"pairs": len(ks),
                                             "label_disagree_pairs": sum(label[a] != label[b] for a, b in ks)}
        bad = {s for gs in mixed.values() for g in gs for s in g}
        if scope == "train_clean" and bad:
            errors.append(f"B08 train_clean: B group(s) (normalised or punct key) with >= 2 labels in rows "
                          f"{fmt_ids(bad)}: human review needed (README 3.2); train_review.csv has no check "
                          f"type for punct-key groups yet")
    return sorted(rows_out, key=lambda r: (r[1], r[2])), stats, errors


# ---------------------------------------------------------------- augmentation (final stage, B09)

def parse_aug_list(data):
    """train_aug.csv bytes -> ([(case_id, label, text_sha256_12)], errors B09).

    ASCII, header AUG_LIST_HEADER, three fields per record (case ID, 0/1, 12 hex), IDs strictly
    ascending, and the bytes must equal the canonical rendering (LF, no quoting, one trailing LF), so
    that a line-based reader (data_lists.read_list) sees the same rows. Errors carry record numbers only.
    """
    if not data.isascii():
        return [], [f"B09 {AUG_LIST}: non-ASCII bytes"]
    try:
        recs = list(csv.reader(io.StringIO(data.decode("ascii"), newline=""), strict=True))
    except csv.Error as exc:
        return [], [f"B09 {AUG_LIST}: {exc_name(exc)} (message suppressed)"]
    if not recs or recs[0] != AUG_LIST_HEADER.split(","):
        return [], [f"B09 {AUG_LIST}: header is not {AUG_LIST_HEADER}"]
    rows, prev = [], None
    bad_fields, bad_ids, bad_labels, bad_hashes, unsorted = [], [], [], [], []
    for n, rec in enumerate(recs[1:], start=1):  # n = 1-based record number, header excluded
        if len(rec) != 3:
            bad_fields.append(n)
            continue
        cid, label, sha = rec
        if not AUG_ID_RE.fullmatch(cid):
            bad_ids.append(n)  # the value is never printed: it is not a checked ID
            continue
        if label not in LABELS:
            bad_labels.append(n)
        if not H12_RE.fullmatch(sha):
            bad_hashes.append(n)
        if prev is not None and not cid > prev:
            unsorted.append(n)
        prev = cid
        rows.append((cid, label, sha))
    errors = [f"B09 {AUG_LIST}: {what} at {fmt_recs(numbers)}" for what, numbers in (
        ("field count != 3", bad_fields), ("sentence_id is not an augmentation case ID", bad_ids),
        ("label not 0/1", bad_labels), ("text_sha256_12 is not 12 lowercase hex characters", bad_hashes),
        ("rows not strictly ascending by sentence_id", unsorted)) if numbers]
    if not errors and data != ("\n".join(",".join(rec) for rec in recs) + "\n").encode("ascii"):
        errors.append(f"B09 {AUG_LIST}: not in the canonical form (LF line endings, no quoting, exactly one "
                      f"trailing LF)")
    return rows, errors


def parse_aug_cases(data):
    """cases.csv bytes -> ({case_id: (label, status, text)}, errors B09).

    csv module, strict; UTF-8 without BOM; header = AUG_CASE_COLUMNS; every record has that many fields,
    a case ID of the augmentation grammar (unique) and label 0/1. text is the field, unchanged. A value
    is printed only when it is a case ID that passed the grammar; otherwise the record number.
    """
    if data.startswith(codecs.BOM_UTF8):
        return {}, [f"B09 {AUG_CASES}: starts with a UTF-8 BOM"]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {}, [f"B09 {AUG_CASES}: not valid UTF-8 (message suppressed)"]
    cases, n = {}, 0
    bad_fields, bad_ids, dups, bad_labels = [], [], set(), []
    idx = {c: AUG_CASE_COLUMNS.index(c) for c in ("case_id", "label", "status", "text")}
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        if next(reader, None) != list(AUG_CASE_COLUMNS):
            return {}, [f"B09 {AUG_CASES}: header is not the {len(AUG_CASE_COLUMNS)} case columns "
                        f"({AUG_CASE_COLUMNS[0]} ... {AUG_CASE_COLUMNS[-1]})"]
        for rec in reader:
            n += 1
            if len(rec) != len(AUG_CASE_COLUMNS):
                bad_fields.append(n)
                continue
            cid, label = rec[idx["case_id"]], rec[idx["label"]]
            if not AUG_ID_RE.fullmatch(cid):
                bad_ids.append(n)
                continue
            if cid in cases:
                dups.add(cid)
                continue
            if label not in LABELS:
                bad_labels.append(cid)
            cases[cid] = (label, rec[idx["status"]], rec[idx["text"]])
    except csv.Error as exc:
        return {}, [f"B09 {AUG_CASES}: {exc_name(exc)} after record {n} (message suppressed)"]
    errors = []
    if bad_fields:
        errors.append(f"B09 {AUG_CASES}: field count != {len(AUG_CASE_COLUMNS)} at {fmt_recs(bad_fields)}")
    if bad_ids:
        errors.append(f"B09 {AUG_CASES}: case_id is not an augmentation case ID at {fmt_recs(bad_ids)}")
    if dups:
        errors.append(f"B09 {AUG_CASES}: duplicate case_id {fmt_ids(dups)}")
    if bad_labels:
        errors.append(f"B09 {AUG_CASES}: label not 0/1 in {fmt_ids(bad_labels)}")
    return cases, errors


def read_augmentation(directory):
    """Augmentation input of the final stage (B09). Returns (record or None, errors).

    (None, []) = no train_aug.csv in the folder = pre-augmentation stage. Otherwise both files are read
    once as bytes (hashed and parsed from the same bytes) and every listed case must exist in cases.csv
    with status `assigned`, the same label, a non-empty text and h12(text) == text_sha256_12.
    record = {"dir", "files": {name: SHA-256}, "cases": rows of cases.csv,
              "rows": [(case_id, label, text)] in train_aug.csv order}.
    """
    directory = Path(directory)
    try:
        with open(directory / AUG_LIST, "rb") as f:
            list_bytes = f.read()
    except (FileNotFoundError, NotADirectoryError):  # no such file, also when the folder itself is missing
        return None, []
    except OSError as exc:
        return None, [f"B09 {AUG_LIST}: cannot read ({exc_name(exc)})"]
    try:
        with open(directory / AUG_CASES, "rb") as f:
            case_bytes = f.read()
    except OSError as exc:
        return None, [f"B09 {AUG_CASES}: cannot read ({exc_name(exc)}); {AUG_LIST} exists, so the final stage "
                      f"needs it"]
    listed, errors = parse_aug_list(list_bytes)
    cases, errs = parse_aug_cases(case_bytes)
    errors += errs
    if errors:
        return None, errors
    rows, missing, unusable, bad_labels, bad_hashes, empty = [], [], [], [], [], []
    for cid, label, sha in listed:
        case = cases.get(cid)
        if case is None:
            missing.append(cid)
            continue
        c_label, status, text = case
        if status != AUG_USABLE_STATUS:
            unusable.append(cid)
        if c_label != label:
            bad_labels.append(cid)
        if h12(text) != sha:
            bad_hashes.append(cid)
        if not text.strip():
            empty.append(cid)
        rows.append((cid, label, text))
    errors = [f"B09 {AUG_LIST}: listed case(s) {what}: {fmt_ids(ids)}" for what, ids in (
        (f"missing in {AUG_CASES}", missing), (f"status is not {AUG_USABLE_STATUS}", unusable),
        (f"label differs from {AUG_CASES}", bad_labels),
        (f"text_sha256_12 differs from the {AUG_CASES} text", bad_hashes),
        ("empty or whitespace-only text", empty)) if ids]
    if errors:
        return None, errors
    return {"dir": directory, "files": {AUG_LIST: sha256_hex(list_bytes), AUG_CASES: sha256_hex(case_bytes)},
            "cases": len(cases), "rows": rows}, []


def check_augmentation(aug, references):
    """B09: no augmentation case may equal a validation or test row at level A or B. Returns (errors, level_c).

    aug and every references[name] are digest rows (id, label, exact digest, sim_keys), so test takes
    part as returned by load_test (no text). Level A = equal exact digests; level B = similar_pairs'
    normalised_key / punct_key (exact digests differ; a row with an empty punct key has no B level).
    level_c[name] = number of one_word_edit pairs (case, row), counted only. Errors name case IDs.
    """
    errors, level_c = [], {}
    for name, rows in references.items():
        exact = {r[2] for r in rows}
        pairs = similar_pairs(aug, rows, edit=True)
        hits = {"A (exact key)": {r[0] for r in aug if r[2] in exact}}
        hits.update({f"B ({level})": {a for (a, _), found in pairs.items() if found == level}
                     for level in SIM_LEVELS[:2]})
        errors += [f"B09 augmentation: case(s) equal to a {name} row at level {level}: {fmt_ids(ids)}; "
                   f"scripts/build_train_aug.py must exclude them (README 3.2)"
                   for level, ids in hits.items() if ids]
        level_c[name] = sum(found == SIM_LEVELS[2] for found in pairs.values())
    return errors, level_c


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
    if stem == SIMILAR:
        sids = [tuple(ln.split(",")[1:3]) for ln in lines[1:]]
    else:
        sids = [ln.split(",")[0] for ln in lines[1:]]
    if any(not b > a for a, b in zip(sids, sids[1:])):
        errors.append(f"B04 {name}: rows not strictly ascending by sentence_id"
                      + (", other_id" if stem == SIMILAR else ""))
    return errors


# ---------------------------------------------------------------- write / check

def write_outputs(target, files):
    """Remove stale <name>.tmp files, delete meta, remove the final-stage lists this build does not
    write (pre-augmentation stage), write each file to <name>.tmp (exclusive create) + os.replace,
    meta last. Returns None or a B05 error."""
    names = [n for n in files if n != META] + [META]
    tmps, current = [], "creating the target folder"
    try:
        target.mkdir(parents=True, exist_ok=True)
        for name in [f"{stem}.csv" for stem in OUTPUTS + FINAL_OUTPUTS + (SIMILAR,)] + [META]:
            current = f"removing a stale {name}.tmp"
            (target / (name + ".tmp")).unlink(missing_ok=True)
        current = f"deleting {META}"
        if (target / META).exists():
            os.remove(target / META)
        for name in (f"{stem}.csv" for stem in FINAL_OUTPUTS):
            if name not in files:
                current = f"removing {name} of an earlier final-stage build"
                (target / name).unlink(missing_ok=True)
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
    """--check: the official files in out_dir must equal the rebuilt bytes and meta trial must be false.

    files holds eight entries in the pre-augmentation stage and ten in the final stage; a final-stage
    list left in out_dir by an earlier build fails a pre-augmentation check.
    """
    errors = [f"B06 --check: {name} is in the out-dir but this build is in the pre-augmentation stage "
              f"(no {AUG_LIST} in the augmentation folder)"
              for name in (f"{stem}.csv" for stem in FINAL_OUTPUTS)
              if name not in files and os.path.lexists(out_dir / name)]
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
    p.add_argument("--augmentation", metavar="DIR",
                   help="augmentation folder with train_aug.csv and cases.csv (default data/cases/train); "
                        "without train_aug.csv the build is in the pre-augmentation stage")
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
    aug_dir = Path(args.augmentation).resolve() if args.augmentation else DEFAULT_AUGMENTATION

    # B06 output locations, before any raw file is read.
    if args.check and args.allow_proposed:
        errors.append("B06 --check and --allow-proposed are mutually exclusive")
    if _under(out_dir, DATASETS):
        errors.append("B06 --out-dir is under datasets/ (raw data stays read-only)")
    if _under(aug_dir, DATASETS):
        errors.append("B06 --augmentation is under datasets/ (the augmentation set is not raw data)")
    if _same(out_dir, DEFAULT_OUT) and not (_same(review_path, DEFAULT_REVIEW)
                                            and _same(manifest_path, DEFAULT_MANIFEST)
                                            and _same(aug_dir, DEFAULT_AUGMENTATION)):
        errors.append("B06 the default out-dir data/processed/ (official lists, trial/ and --check) needs the "
                      "default --review, --manifest and --augmentation; pass --out-dir for another build")
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

    aug, aug_errors = read_augmentation(aug_dir)
    errors += aug_errors
    if aug is not None:
        c = Counter(r[1] for r in aug["rows"])
        say(f"ok augmentation {show(aug_dir)}: {AUG_LIST} rows {len(aug['rows'])} ({fmt_counts(c, LABELS)}), "
            f"{AUG_CASES} rows {aug['cases']}: final stage")
    elif not aug_errors:
        say(f"note: no {AUG_LIST} in {show(aug_dir)}: pre-augmentation stage "
            f"({', '.join(f'{stem}.csv' for stem in FINAL_OUTPUTS)} are not built)")

    files = None
    trial = review_rows is not None and any(r["status"] == "proposed" for r in review_rows)
    if len(src) == len(SPLITS) and review_rows is not None and not aug_errors:
        files, errs = _build(src, review_rows, info, man, manifest_path, review_path, review_bytes, trial, aug)
        errors += errs

    if args.check:
        if trial:
            errors.append("B06 --check needs a confirmed review (the review has proposed rows)")
        elif files is not None and not errors:
            errors += compare_official(out_dir, files)
    if trial and not args.allow_proposed and not args.check:
        proposed = [r["sentence_id"] for r in review_rows if r["status"] == "proposed"]
        errors.append(f"R22 review has proposed rows {fmt_ids(proposed)}: confirm the review first (status "
                      f"confirmed and member ID(s) in confirmed_by on every row) and rerun for the official "
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
        say("OFFICIAL LISTS written (meta trial false" + (", test_clean_stage final)" if aug is not None else ")"))
    return 0


def _build(src, review_rows, info, man, manifest_path, review_path, review_bytes, trial, aug=None):
    """Build, check and render every output in memory. Returns (files or None, errors).

    aug = None: pre-augmentation stage (eight files). aug = the record of read_augmentation(): final
    stage, the same eight files plus test_clean.csv / test_excluded.csv (README 3.2 DATA-05).
    """
    train, validation, test = src["train"], src["validation"], src["test"]
    try:
        train_clean, train_excluded = build_train(train, review_rows)
    except StopError as exc:
        return None, [str(exc)]
    validation_clean, validation_excluded = build_validation(validation, train)
    clean_ids = {r[0] for r in train_clean}
    cleaned_dg = frozenset(digest(text) for sid, _, text in train if sid in clean_ids)
    pools = {"cleaned_train_digests": cleaned_dg,
             "validation_digests": frozenset(digest(text) for _, _, text in validation),
             "original_train_digests": frozenset(digest(text) for _, _, text in train)}
    test_clean, test_excluded, overlap = build_test(test, **pools)
    lists = {"train_clean": train_clean, "train_excluded": train_excluded,
             "validation_clean": validation_clean, "validation_excluded": validation_excluded,
             "test_clean_preaug": test_clean, "test_excluded_preaug": test_excluded}
    stems, aug_rows, aug_dg, aug_errors, level_c = OUTPUTS, [], None, [], {}
    if aug is not None:  # final stage: the pre-augmentation lists above keep their rows
        stems = OUTPUTS + FINAL_OUTPUTS
        aug_rows = [(cid, label, digest(text), sim_keys(text)) for cid, label, text in aug["rows"]]
        aug_dg = frozenset(r[2] for r in aug_rows)
        val_rows = [(sid, label, digest(text), sim_keys(text)) for sid, label, text in validation]
        aug_errors, level_c = check_augmentation(aug_rows, dict(zip(AUG_REFERENCES, (val_rows, test))))
        lists["test_clean"], lists["test_excluded"], overlap = build_test(test, augmentation_digests=aug_dg,
                                                                          **pools)
    errors = check_invariants(train, validation, test, review_rows, lists, aug_dg)
    similar_rows, similar, errs = similar_check(train, validation, test, lists)
    errors += errs + aug_errors
    test_labels = {sid: label for sid, label, *_ in test}
    counts = count_lists(lists, test_labels, overlap, stems)
    counts.update({f"similar.{scope}": levels for scope, levels in similar.items()})
    errors += check_expected(counts)
    if aug is not None:
        c = Counter(r[1] for r in aug_rows)
        counts["augmentation"] = {"rows": len(aug_rows), **{k: c[k] for k in LABELS}}
        counts["augmentation_level_c"] = {k: level_c[k] for k in AUG_REFERENCES}
        counts["test_excluded"] = {"rows": counts["test_excluded"]["rows"],
                                   **{k: counts["test_excluded"].get(k, 0) for k in TEST_REASONS}}
        errors += check_expected(counts, EXPECTED_FINAL, "EXPECTED_FINAL")
    expected_check, partition = find_groups(train)
    audit = compute_audit(train, validation, test, [expected_check[next(iter(g))] for g in partition],
                          cleaned_dg, overlap)

    for stem in stems:
        c = counts[stem]
        keys = LABELS if "_clean" in stem else sorted(k for k in c if k != "rows")
        say(f"{stem} {c['rows']} ({fmt_counts(c, keys)})")
    say(f"test overlap rows (non-exclusive): {fmt_counts(overlap, OVERLAP_KEYS)}; "
        f"validation overlap labels ({fmt_counts(audit['test']['validation_overlap_labels'], LABELS)}), "
        f"agree with validation: {audit['test']['validation_overlap_labels_agree']}")
    for scope, levels in similar.items():
        say(f"similar {scope}: " + "; ".join(f"{level} " + ", ".join(f"{k} {v}" for k, v in d.items())
                                            for level, d in levels.items()))
    if aug is not None:
        say(f"augmentation {len(aug_rows)} ({fmt_counts(counts['augmentation'], LABELS)}); level C "
            f"one_word_edit pairs, counted only ({fmt_counts(level_c, AUG_REFERENCES)})")

    files = {f"{stem}.csv": render_csv(stem, lists[stem]) for stem in stems}
    files[f"{SIMILAR}.csv"] = render_csv(SIMILAR, similar_rows)
    src_counts = {s: Counter(r[1] for r in src[s]) for s in SPLITS}

    def label_counts(ids, split):
        lab = {r[0]: r[1] for r in src[split]}
        return {k: sum(lab[i] == k for i in ids) for k in LABELS}

    def clean_excluded(split, clean, excl, ridx):
        x_ids = [r[0] for r in lists[excl]]
        return {"clean": {"rows": len(lists[clean]), "labels": label_counts([r[0] for r in lists[clean]], split)},
                "excluded": {"rows": len(x_ids), "labels": label_counts(x_ids, split),
                             "by_reason": dict(sorted(Counter(r[ridx] for r in lists[excl]).items()))}}

    splits_meta = {}
    for split, clean, excl, ridx in (("train", "train_clean", "train_excluded", 3),
                                     ("validation", "validation_clean", "validation_excluded", 3),
                                     ("test", "test_clean_preaug", "test_excluded_preaug", 1)):
        splits_meta[split] = {
            "source": {"rows": len(src[split]), "labels": {k: src_counts[split][k] for k in LABELS}},
            **clean_excluded(split, clean, excl, ridx),
        }
    if aug is not None:  # splits.test.clean / .excluded stay the pre-augmentation lists
        splits_meta["test"]["final"] = clean_excluded("test", "test_clean", "test_excluded", 1)
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
        "augmentation": None if aug is None else {
            "dir": rel(aug["dir"]), "files": dict(aug["files"]), "rows": len(aug_rows),
            "label_counts": {k: counts["augmentation"][k] for k in LABELS},
            "level_c_pairs": dict(counts["augmentation_level_c"]),
        },
        "test_clean_stage": "pre-augmentation" if aug is None else "final",
        "rules": RULES if aug is None else {**RULES, **FINAL_RULES},
        "splits": splits_meta,
        "audit": audit,
        "similar": similar,
        "outputs": {name: {"rows": data.count(b"\n") - 1, "sha256": sha256_hex(data)} for name, data in files.items()},
    }
    files[META] = render_meta(meta)
    for name, data in files.items():
        errors += check_grammar(name, data)
    return (None if errors else files), errors


if __name__ == "__main__":
    sys.exit(main())
