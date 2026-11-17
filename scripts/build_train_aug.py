"""Stage-4 training augmentation set (README 3.3 DATA-07..12, 5.3 condition C): selection of train
originals, ROB-01 operations with new seeds, context assembly, LLM inputs, automatic checks, test
overlap, review sheet and the usable list -> data/cases/train/.

Run from Git Bash at the repo root:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/build_train_aug.py
        (build: originals.csv, cases.csv, review.csv, cases_meta.json and, when ready, train_aug.csv)
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/build_train_aug.py --check
Options: --dir DIR (default data/cases/train; holds prompts/, llm/, test_overlap.json and the outputs);
--print N prints, to the terminal only, up to N pending assigned cases per condition (with the train
original for derived cases); --view FILE also writes a read-only reading aid for the reviewer (UTF-8 with
BOM; every case that has a review row with its decision, text and train original) to FILE, which must be
outside the repo or a gitignored path (never in the cases folder). Decisions are made in review.csv only.

Rules (.omc/autopilot/stage4-design.md D1, D2):
  - Inputs. data/processed/train_clean.csv (its SHA-256 must equal lists_meta.json outputs, "trial"
    false) joined to the raw train rows with data_lists.load_rows() / join(); every raw train and
    validation row for the key checks. test.csv is never opened: main() installs
    data_lists.install_test_guard() before any data is read. Development material (README DATA-07, 12):
    every row of data/cases/dev/cases.csv and every item of the three dev LLM files, each verified against
    data/cases/dev/cases_meta.json (outputs / generators SHA-256). LLM outputs <dir>/llm/<name>.json with
    their prompts <dir>/prompts/<name>.md for aug03_contexts_v1 (items n = 1..240: 1..160 "short",
    161..240 "long"; keys n, length, kind, text), aug04_benign_v1 (n = 1..300, family blocks b1..b6 of 50;
    keys n, family, text) and aug05_attacks_v1 (n = 1..300, family blocks a1 60, a2 48, a3 48, a4..a7 36);
    each JSON is {"generator": {"model", "interface", "date", "prompt_file", "prompt_sha256", "notes"},
    "items": [...]} (generator fields are strings); prompt_sha256 must equal the LF-normalised SHA-256 of
    the prompt file and prompt_file must name it. An item text with a carriage return is refused (C04):
    the text of an AUG-04/05 item is written to cases.csv as it is, and the outputs are LF only. Token
    lengths use the three tokenizers of build_dev_cases.make_token_len (HF_HUB_OFFLINE=1).
  - Eligible original: ASCII text without a line break, 4 to 40 words (text.split()), and none of the
    surface patterns alternating_case, spaced_letters, multiple_spaces, leetspeak (the tests of
    build_dev_cases.train_patterns; leetspeak vocabulary = all-letter tokens of the raw train and
    validation rows) or all_caps_row (>= 4 cased characters and no lowercase character).
  - Selection. For label in (1, 0) the sorted train_clean IDs are shuffled once with
    random.Random(f"s4-aug:20261117:label{label}") and walked in that order; ineligible originals are
    passed over. Slot groups are filled in the order case, space, char (label 1, then label 0) and then
    context: each slot takes the next eligible original; an original whose case fails an automatic check
    is recorded (auto_excluded) and skipped. A group has 150 assigned slots; the next 15 passing
    candidates are its reserves. Every original is used for one case only. No eligible original left
    stops the run (C03).
  - AUG-01 (label 1 and 0; subtypes case, space, char): build_dev_cases.ROB01[subtype](text,
    random.Random(f"aug01-v1:{case_id}")): the ROB-01 v1 operations with new originals and new seeds.
    No eligible character -> exclude:no_eligible_char. The held-out method homoglyph is refused (C08).
  - AUG-03 (context; pair index i = 0..149 assigned, 150..164 reserve; the i-th label-1 and the i-th
    label-0 context original share their passages). Plan with rng = random.Random("aug03-v1"): position
    of pair i = (prefix, suffix, both)[i % 3]; in the assigned block rng.sample picks 25 of the 50
    prefix pairs and then 25 of the 50 suffix pairs for one long passage (one third of the 150 pairs),
    the other prefix/suffix pairs get one short passage; both gets two short passages; in the reserve
    block every prefix and suffix pair gets one long passage (the 160 short passages are planned for
    the assigned block, 150, and the five reserve both pairs, 10); then the usable short and the usable
    long passage numbers are shuffled into queues. Pairs are served in ascending order. Candidates: the
    queue of the planned class and then the other queue (both: pairs (a, b) with a before b over short
    queue + long queue, ordered by b and then a). The first of at most 20 candidates with which both
    cases have <= 256 tokens (special tokens included) for all three tokenizers is used and leaves its
    queue (a passage is used by one pair only). If none fits, each original whose case is over 256
    tokens with the first candidate is excluded (exclude:length_over_256) and the next eligible original
    of that label takes its place; the same replacement follows any other automatic exclusion of one
    side. Text: build_dev_cases.assemble ("\\n\\n" separators); the original must appear verbatim (C07).
    No passage left -> the remaining pairs stay empty (shortfall in the meta).
  - AUG-04 (label 0, families b1..b6) and AUG-05 (label 1, families a1..a7): the items of a family in
    file order; the first K passing items are assigned, later passing items are reserves. K: b1..b4 42,
    b5..b6 41 (250); a1 50, a2 40, a3 40, a4..a7 30 (250).
  - Automatic checks (keys build_lists.nk / pk; an empty punct key is no key; reasons in auto_flags as
    exclude:<reason>, flags as flag:<flag>, joined by ";"):
    1. any raw validation row: exact text -> exact=<ID>, normalised key -> normalised_key=<ID>, punct
       key -> punct_key=<ID> (lowest ID per level; README 3.2 levels A and B). Raw train rows: exact text
       -> exact=<ID>; normalised or punct key equal to a train row of a different label ->
       label_conflict=<ID> (lowest ID). Normalised or punct key equal to a train row of the same label:
       AUG-01/03 (train-derived; the case's own original is skipped at the key (B) level) -> flag
       train_normalised_key=<ID> / train_punct_key=<ID> only; AUG-04/05 (new LLM text) ->
       train_material=<ID> (lowest ID over the two key levels) and no flag: an item that repeats a
       training sentence is not new material, and the generation prompt forbids copying dataset
       sentences (lead decision 2026-11-17).
    2. development material: exact text, normalised key or punct key equal -> dev_material=<ID>;
       AUG-04/05 also a one-word edit (build_dev_cases.one_word_edit on punct-key words) ->
       dev_one_word_edit=<ID>. AUG-03 passages are checked on their own as well (see 5).
    3. inside the set: exact text, normalised key or punct key equal to an earlier passing case ->
       duplicate_of=<case_id> (checked as exact text or punct key: an equal normalised key always gives
       an equal punct key). Order: AUG-01 label 1 (case, space, char), AUG-01 label 0, AUG-03 pairs
       (label 1 then 0), AUG-04, AUG-05.
    4. more than 256 tokens for any tokenizer -> length_over_256.
    5. LLM text must be non-empty ASCII on one line; AUG-04 40-200 characters; AUG-05 3-40 words; else
       format. A passage failing this, equal (exact / normalised / punct key) to development material or
       to an earlier passage is not used (cases_meta.json aug03_passages.excluded).
    6. test overlap (README 3.2 levels A and B) from <dir>/test_overlap.json, written by
       scripts/check_cases_vs_test.py: test_exact / test_normalised_key / test_punct_key. An assigned
       case that matches becomes auto_excluded and the next reserve of its slot group that does not
       match takes its place (no reserve left -> shortfall). The file is applied only when its
       pool_sha256 equals the current pool: sha256 of "".join(f"{case_id},{text_sha256_12}\\n") over every
       cases.csv row in case_id order. The pool does not depend on test_overlap.json or review.csv.
       Missing file or another pool -> test_overlap.status "pending".
    Rule 6 and the review work per case: the other case of an AUG-03 pair keeps its status, so a passage
    can be left with assigned cases of one label only. cases_meta.json aug03_passages.single_label lists
    those passage numbers (label_1_only, label_0_only; statuses are not changed by this report).
  - Status: assigned | reserve | auto_excluded | review_excluded.
  - Case IDs: aug-01-{case|space|char}-NNNNNN, aug-03-NNNNNN (NNNNNN = train row number),
    aug-04-NNN, aug-05-NNN (NNN = LLM item n). group_id = the original's sentence ID, or the case ID
    for new cases (source_split "new", source_id <LLM name>:NNN). purpose and eval_group "train".
  - Review (review.csv: case_id, text_sha256_12, decision, reason, reviewer). decision pending | keep |
    exclude (no relabel: a case whose label is doubtful is excluded); reason required for exclude;
    reviewer empty when pending, else one of REVIEWERS (case-insensitive). text_sha256_12 is written by
    the script; a decided row whose value differs from the current case text is a stale decision (R09).
    A default build applies review.csv: an assigned case with decision exclude becomes review_excluded
    and is NOT replaced. Rows: every assigned case (new ones pending) and every decided row. A decided
    row stops the run (R09) when its case is gone, was never assigned (excluded by the checks 1-5, or a
    reserve that matches test) or is a reserve while the test overlap is applied; it is kept without
    effect when its assigned case was excluded by the test overlap after the review, or when its case
    is a reserve while the test overlap is pending (it may be assigned again once the overlap is applied).
  - train_aug.csv (header sentence_id,label,text_sha256_12; sentence_id = case_id, ascending): the
    usable cases = status assigned and decision keep. Written only when the test overlap is applied
    and no review row is pending; otherwise a stale file is deleted and cases_meta.json
    not_ready_reasons says why (ready_for_training false).
  - Outputs (UTF-8, LF, csv module quoting, sorted by case_id; originals.csv by label 1, 0 then order,
    one row per walked original, status of its case or `ineligible`; no timestamps or absolute paths).
    Write: every file to <name>.tmp, then os.replace in the order originals, cases, review, train_aug,
    meta. --check rebuilds in memory and compares every output byte for byte; a train_aug.csv on disk
    while the set is not ready also fails.

Stop codes (printed as `FAIL <code> ...`, IDs and counts only, exit 1, nothing written):
  C01 input missing or unreadable            C02 train_clean.csv or development material differs
  C03 no eligible train_clean original left  C04 LLM file schema (keys, types, numbering, families, prompt)
  C05 tokenizer environment (HF_HUB_OFFLINE) C06 --dir under datasets/ or data/cases/dev, or a bad --view
  C07 AUG-03 original not verbatim           C08 held-out method (homoglyph) or non-ASCII usable case
  R01 review.csv unreadable, not UTF-8 / csv R02 review header       R03 field count or line break
  R04 case_id malformed or duplicate         R05 decision value      R07 reason missing
  R08 reviewer rule                          R09 decided row: case gone, stale text hash, never reviewable
  T01 test_overlap.json unreadable or inconsistent with the pool
  K01 --check: file missing, different or unexpected            W01 write failure
  X01 unexpected exception
"""
import argparse
import codecs
import csv
import io
import itertools
import json
import os
import random
import re
import sys
import traceback
from pathlib import Path

import build_dev_cases as B
import build_lists
import data_lists

ROOT = data_lists.ROOT
SCRIPT_REL = "scripts/build_train_aug.py"
TRAIN_CLEAN_REL = "data/processed/train_clean.csv"
LISTS_META_REL = "data/processed/lists_meta.json"
DEV_REL = "data/cases/dev"
DEFAULT_DIR = ROOT / "data" / "cases" / "train"
SCHEMA = 1

SELECTION_SEED = 20261117
LABEL_ORDER = (1, 0)
N_MAIN = 150
N_RESERVE = 15
WORDS = (4, 40)  # words of an eligible original
MAX_TOKENS = B.MAX_TOKENS
CONDITIONS = ("AUG-01", "AUG-03", "AUG-04", "AUG-05")
AUG01_VERSION = "aug01-v1"
AUG03_VERSION = "aug03-v1"
AUG01_SUBTYPES = ("case", "space", "char")
HELD_OUT = "homoglyph"  # README 3.3: never used for training augmentation
AUG03_SHORT, AUG03_LONG = 160, 80
AUG03_WORDS = {"short": (15, 35), "long": (50, 90)}
AUG03_TRIES = 20  # candidates tried per attempt of a pair
AUG04_CHARS = (40, 200)
AUG05_WORDS = (3, 40)
LLM = {"AUG-03": "aug03_contexts_v1", "AUG-04": "aug04_benign_v1", "AUG-05": "aug05_attacks_v1"}
# (family, items in the file, K = assigned quota); the items of a family are one block of n, in this order.
FAMILIES = {
    "AUG-04": (("b1", 50, 42), ("b2", 50, 42), ("b3", 50, 42), ("b4", 50, 42), ("b5", 50, 41), ("b6", 50, 41)),
    "AUG-05": (("a1", 60, 50), ("a2", 48, 40), ("a3", 48, 40), ("a4", 36, 30), ("a5", 36, 30), ("a6", 36, 30),
               ("a7", 36, 30)),
}
LLM_LABEL = {"AUG-04": 0, "AUG-05": 1}
ITEM_KEYS = {"AUG-03": {"n", "length", "kind", "text"}, "AUG-04": {"n", "family", "text"},
             "AUG-05": {"n", "family", "text"}}
GENERATOR = {"AUG-01": "rule", "AUG-03": "template+llm", "AUG-04": "llm", "AUG-05": "llm"}
GENERATOR_KEYS = ("model", "interface", "date", "prompt_file", "prompt_sha256", "notes")
MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# Development LLM files (README DATA-07): name -> the item field that holds the generated text.
DEV_LLM = {"rob02_paraphrase_v1": "paraphrase", "rob03_contexts_v1": "text", "rob04_benign_v1": "text"}

ORIGINALS_COLUMNS = ("sentence_id", "label", "condition", "subtype", "order", "status")
CASE_COLUMNS = B.CASE_COLUMNS
REVIEW_COLUMNS = ("case_id", "text_sha256_12", "decision", "reason", "reviewer")
TRAIN_AUG_HEADER = data_lists.TRAIN_VAL_HEADER
# README DATA-10: HANJUSON, or the separate review session for the augmentation cases (prompt aug_review_v1).
REVIEWERS = ("HANJUSON", "llm:claude-opus-5-5:aug_review_v1")
DECISIONS = ("pending", "keep", "exclude")
STATUSES = B.STATUSES
META = B.META
TRAIN_AUG = "train_aug.csv"
OVERLAP = "test_overlap.json"
KEY_LEVELS = ("exact", "normalised_key", "punct_key")  # README 3.2: level A, then the two B keys
FILES = ("originals.csv", "cases.csv", "review.csv", TRAIN_AUG, META)  # write order; train_aug.csv only when ready
CASE_ID_RE = re.compile(r"aug-01-(?:case|space|char)-[0-9]{6}|aug-03-[0-9]{6}|aug-0[45]-[0-9]{3}")

# Surface patterns: the six of build_dev_cases.TRAIN_PATTERNS plus all_caps_row. SURFACE makes an original
# ineligible (cyrillic and non_ascii are covered by the ASCII rule).
ALL_CAPS = "all_caps_row"
PATTERNS = tuple(B.TRAIN_PATTERNS) + (ALL_CAPS,)
PATTERN_DEFS = {**B.TRAIN_PATTERNS, ALL_CAPS: ">= 4 cased characters and no lowercase character in the row"}
SURFACE = ("alternating_case", "spaced_letters", "multiple_spaces", "leetspeak", ALL_CAPS)

RULES = {
    "eligible_original": "ASCII, no line break, 4-40 words, none of the surface patterns alternating_case, "
                         "spaced_letters, multiple_spaces, leetspeak, all_caps_row",
    "selection": "per label (1, 0) the sorted train_clean IDs are shuffled once and walked; slot groups case, "
                 "space, char (label 1, then 0), then context; 150 assigned + 15 reserve passing candidates per "
                 "group; a failing candidate is auto_excluded and skipped; one case per original",
    "aug01": "build_dev_cases.ROB01[subtype] (ROB-01 v1 operations case, space, char) on new train originals with "
             "new seeds; the held-out method homoglyph is refused",
    "aug03": "pair i shares its passages across labels; position (prefix, suffix, both)[i % 3]; assigned block: "
             "25 of the 50 prefix and 25 of the 50 suffix pairs get one long passage (seeded sample), the others "
             "one short; reserve block: prefix and suffix pairs get one long passage; both = two short; first of "
             "at most 20 queue candidates (planned class, then the other) with <= 256 tokens for both cases and "
             "all three tokenizers; a passage is used by one pair only",
    "aug04_aug05": "items of a family in file order: the first K passing items are assigned, later passing "
                   "items are reserves",
    "checks": "validation rows: exact / normalised key / punct key -> exclude; train rows: exact -> exclude, "
              "key match with another label -> exclude (label_conflict), same label -> flag for AUG-01/03 "
              "(train-derived), exclude (train_material) for AUG-04/05 (new LLM text); development "
              "material: exact / key -> exclude, AUG-04/05 also a one-word edit; duplicate inside the set -> the "
              "later case excluded; > 256 tokens -> exclude; LLM text format -> exclude",
    "test_overlap": "levels A and B from test_overlap.json (scripts/check_cases_vs_test.py), applied only when "
                    "its pool_sha256 equals the current pool; a matching assigned case is replaced by the next "
                    "non-matching reserve of its slot group",
    "review": "pending | keep | exclude; exclude -> review_excluded, not replaced; usable = assigned and keep",
    "train_aug": "written only when the test overlap is applied and no review row is pending",
}

Refused = B.Refused
say = B.say


# ---------------------------------------------------------------- inputs

def load_dev_material(dev_dir):
    """([(ID, text)] of every development text, inputs record); files verified against the dev cases_meta.json."""
    try:
        cases_bytes = (dev_dir / "cases.csv").read_bytes()
        meta = json.loads((dev_dir / META).read_bytes())
        want = {"cases.csv": meta["outputs"]["cases.csv"]["sha256"],
                **{name: meta["generators"][name]["sha256_lf"] for name in DEV_LLM}}
        llm_bytes = {name: (dev_dir / "llm" / f"{name}.json").read_bytes().replace(b"\r\n", b"\n")
                     for name in DEV_LLM}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Refused([f"C01 development material ({DEV_REL}: cases.csv, {META}, llm/*.json): unreadable "
                       f"({build_lists.exc_name(exc)})"])
    errors, out = [], []
    got = data_lists.sha256_bytes(cases_bytes)
    if got != want["cases.csv"]:
        errors.append(f"C02 {DEV_REL}/cases.csv: sha256 {got[:12]} != its {META} outputs {want['cases.csv'][:12]}")
    try:
        recs = list(csv.DictReader(io.StringIO(cases_bytes.decode("utf-8"), newline="")))
        out += [(r["case_id"], r["text"]) for r in recs]
    except (UnicodeDecodeError, csv.Error, KeyError, TypeError) as exc:
        raise Refused([f"C01 {DEV_REL}/cases.csv: unreadable ({build_lists.exc_name(exc)})"])
    inputs = {"cases.csv": {"path": f"{DEV_REL}/cases.csv", "sha256": got, "rows": len(recs)}}
    for name, field in DEV_LLM.items():
        got = data_lists.sha256_bytes(llm_bytes[name])
        if got != want[name]:
            errors.append(f"C02 {DEV_REL}/llm/{name}.json: sha256_lf {got[:12]} != its {META} generators "
                          f"{want[name][:12]}")
        try:
            items = json.loads(llm_bytes[name].decode("utf-8"))["items"]
            texts = [(f"{name}:{it['id']}" if "id" in it else f"{name}:{it['n']:03d}", it[field]) for it in items]
            if not all(isinstance(t, str) for _, t in texts):
                raise TypeError
        except (ValueError, KeyError, TypeError) as exc:
            raise Refused([f"C01 {DEV_REL}/llm/{name}.json: unreadable ({build_lists.exc_name(exc)})"])
        out += texts
        inputs[f"llm/{name}.json"] = {"path": f"{DEV_REL}/llm/{name}.json", "sha256_lf": got, "items": len(texts)}
    if errors:
        raise Refused(errors)
    return out, inputs


def load_sources(root=None):
    """{"tclean": [(sid, label, text)], "train"/"validation": {sid: (label str, text)}, "dev": [(ID, text)],
    "inputs": {...}}."""
    root = Path(root) if root else ROOT
    try:
        rows = data_lists.load_rows()
    except data_lists.ListError as exc:
        raise Refused([f"C01 raw splits: {e}" for e in exc.errors])
    try:
        tc_bytes = (root / TRAIN_CLEAN_REL).read_bytes()
        lists_meta_bytes = (root / LISTS_META_REL).read_bytes()
        lists_meta = json.loads(lists_meta_bytes)
        want, trial = lists_meta["outputs"]["train_clean.csv"]["sha256"], lists_meta["trial"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Refused([f"C01 {TRAIN_CLEAN_REL} or {LISTS_META_REL}: unreadable ({build_lists.exc_name(exc)})"])
    errors = []
    got = data_lists.sha256_bytes(tc_bytes)
    if got != want:
        errors.append(f"C02 {TRAIN_CLEAN_REL}: sha256 {got[:12]} != lists_meta.json {want[:12]}")
    if trial is not False:
        errors.append(f"C02 {LISTS_META_REL}: trial is not false (trial lists are never used for augmentation)")
    try:
        joined, jerr = data_lists.join(data_lists.read_list(root / TRAIN_CLEAN_REL), rows["train"],
                                       "train_clean.csv")
    except (data_lists.ListError, UnicodeDecodeError, ValueError) as exc:
        joined, jerr = [], getattr(exc, "errors", [f"train_clean.csv: unreadable ({build_lists.exc_name(exc)})"])
    errors += [f"C02 {e}" for e in jerr]
    bad = [sid for sid, _, _ in joined if not build_lists.TRAIN_SID_RE.fullmatch(sid)]
    if bad:
        errors.append(f"C02 train_clean.csv: IDs that are not train:NNNNNN: {build_lists.fmt_ids(bad)}")
    if errors:
        raise Refused(errors)
    info, man, _ = build_lists.load_manifest(build_lists.DEFAULT_MANIFEST)
    dev, dev_inputs = load_dev_material(root / DEV_REL)
    inputs = {
        "train_clean.csv": {"path": TRAIN_CLEAN_REL, "sha256": got, "rows": len(joined)},
        "lists_meta.json": {"path": LISTS_META_REL, "sha256": data_lists.sha256_bytes(lists_meta_bytes)},
        "download_manifest.json": {"path": "datasets/s-labs-prompt-injection/download_manifest.json",
                                   "sha256": man["sha256"], "revision": man["revision"]},
        "development_material": dev_inputs,
    }
    for split in ("train", "validation"):
        inputs[f"{split}.csv"] = {"via": "download_manifest.json", "sha256": info[split]["sha256"],
                                  "size": info[split]["size"], "rows": len(rows[split])}
    return {"tclean": joined, "train": rows["train"], "validation": rows["validation"], "dev": dev,
            "inputs": inputs}


def load_llm(d):
    """{condition: {"generator", "items" (file order), "name", "file", "sha256_lf", "prompt", "prompt_sha256_lf"}}."""
    out, errors = {}, []
    for cond, name in LLM.items():
        f_rel, p_rel = f"llm/{name}.json", f"prompts/{name}.md"
        try:
            raw = (d / f_rel).read_bytes()
            prompt_sha = data_lists.sha256_lf(d / p_rel)
        except OSError:
            errors.append(f"C01 {f_rel} or {p_rel}: missing or unreadable")
            continue
        try:
            doc = json.loads(raw.decode("utf-8"))
            gen, items = doc["generator"], doc["items"]
            if not (isinstance(gen, dict) and all(isinstance(gen.get(k), str) for k in GENERATOR_KEYS)
                    and MODEL_RE.fullmatch(gen["model"]) and isinstance(items, list)):
                raise TypeError
        except (ValueError, KeyError, TypeError):
            errors.append(f"C04 {f_rel}: not JSON with a generator object (strings {', '.join(GENERATOR_KEYS)}; "
                          "model of letters, digits, . _ -) and an items list")
            continue
        if gen["prompt_sha256"] != prompt_sha:
            errors.append(f"C04 {f_rel}: generator.prompt_sha256 differs from the LF SHA-256 of {p_rel} "
                          f"({prompt_sha[:12]}): the prompt changed after generation, or the record is wrong")
        if not gen["prompt_file"].replace("\\", "/").endswith(p_rel):
            errors.append(f"C04 {f_rel}: generator.prompt_file does not name {p_rel}")
        errors += [f"C04 {f_rel}: {e}" for e in _item_errors(cond, items)]
        out[cond] = {"generator": gen, "items": items, "name": name, "file": f_rel,
                     "sha256_lf": data_lists.sha256_bytes(raw.replace(b"\r\n", b"\n")),
                     "prompt": p_rel, "prompt_sha256_lf": prompt_sha}
    if errors:
        raise Refused(errors)
    return out


def family_of(cond, n):
    """Family code of item n (1-based) by the block layout of FAMILIES, or None when n is out of range."""
    start = 1
    for fam, size, _ in FAMILIES[cond]:
        if start <= n < start + size:
            return fam
        start += size
    return None


def _item_errors(cond, items):
    """Schema errors of one LLM file: item k (0-based) must be item n = k + 1 with the keys of its file."""
    total = AUG03_SHORT + AUG03_LONG if cond == "AUG-03" else sum(size for _, size, _ in FAMILIES[cond])
    errors = []
    if len(items) != total:
        errors.append(f"{len(items)} items, expected {total}")
    for k, it in enumerate(items[:total]):
        n = k + 1
        if not (isinstance(it, dict) and set(it) == ITEM_KEYS[cond]):
            errors.append(f"item {n}: keys are not {', '.join(sorted(ITEM_KEYS[cond]))}")
        elif not (B._is_int(it["n"]) and it["n"] == n):
            errors.append(f"item {n}: n is not {n} (items are numbered 1..{total} in file order)")
        elif not isinstance(it["text"], str):
            errors.append(f"item {n}: text is not a string")
        elif "\r" in it["text"]:
            errors.append(f"item {n}: text has a carriage return (the outputs are LF only)")
        elif cond == "AUG-03":
            want = "short" if n <= AUG03_SHORT else "long"
            if it["length"] != want or not B._nonempty_str(it["kind"]):
                errors.append(f"item {n}: length is not {want} or kind is not a non-empty string")
        elif it["family"] != family_of(cond, n):
            errors.append(f"item {n}: family is not {family_of(cond, n)}")
    return errors[:20]


# ---------------------------------------------------------------- text rules

def one_line(text):
    return text.splitlines() == [text]


def format_ok(cond, text):
    """Rule 5: non-empty ASCII on one line; AUG-04 40-200 characters; AUG-05 3-40 words."""
    if not (text.strip() and text.isascii() and one_line(text)):
        return False
    if cond == "AUG-04":
        return AUG04_CHARS[0] <= len(text) <= AUG04_CHARS[1]
    if cond == "AUG-05":
        return AUG05_WORDS[0] <= len(text.split()) <= AUG05_WORDS[1]
    return True


def leet_vocab(train, validation):
    """Casefolded all-letter tokens of >= 3 letters in the raw train and validation rows (as B.train_patterns)."""
    return {t.casefold() for rows in (train, validation) for _, text in rows.values()
            for t in B._ALNUM.findall(text) if len(t) >= 3 and t.isalpha()}


def surface_patterns(text, vocab):
    """Names (PATTERNS order) of the surface patterns in text: the tests of B.train_patterns plus all_caps_row."""
    words = B._LETTERS.findall(text)
    cased = [c for c in text if c.isupper() or c.islower()]
    found = {
        "alternating_case": any(len(w) >= 4 and all(a.isupper() != b.isupper() for a, b in zip(w, w[1:]))
                                for w in words),
        "spaced_letters": bool(B._SPACED.search(text)),
        "multiple_spaces": bool(B._MULTI_SPACE.search(text)),
        "leetspeak": any(B.is_leet_token(t, vocab) for t in B._ALNUM.findall(text)),
        "cyrillic": bool(B._CYRILLIC.search(text)),
        "non_ascii": not text.isascii(),
        ALL_CAPS: len(cased) >= 4 and not any(c.islower() for c in cased),
    }
    return [p for p in PATTERNS if found[p]]


def ineligible_reason(text, vocab):
    """None when a train_clean original may be used, else the first rule it breaks."""
    if not text.isascii():
        return "non_ascii"
    if not one_line(text):
        return "line_break"
    if not WORDS[0] <= len(text.split()) <= WORDS[1]:
        return "word_count"
    found = surface_patterns(text, vocab)
    return next((p for p in SURFACE if p in found), None)


def check_held_out():
    """README 3.3: the held-out generation method can never be an augmentation subtype."""
    if HELD_OUT in AUG01_SUBTYPES or not set(AUG01_SUBTYPES) <= set(B.ROB01):
        raise Refused([f"C08 AUG-01 subtypes {', '.join(AUG01_SUBTYPES)}: only case, space and char are "
                       f"augmentation methods ({HELD_OUT} is held out, README 3.3)"])


def aug01_variant(subtype, case_id, text):
    """ROB-01 v1 operation on a train original with the augmentation seed; None = no eligible character."""
    if subtype == HELD_OUT or subtype not in AUG01_SUBTYPES:
        raise Refused([f"C08 AUG-01 subtype {subtype}: not an augmentation method ({HELD_OUT} is held out)"])
    out = B.ROB01[subtype](text, random.Random(f"{AUG01_VERSION}:{case_id}"))
    if out is not None and not out.isascii():
        raise Refused([f"C08 {case_id}: the variant of an ASCII original has non-ASCII characters (the held-out "
                       f"{HELD_OUT} substitution must never run here)"])
    return out


def plan_aug03():
    """(position per pair, planned passage classes per pair, the rng for the two queue shuffles)."""
    rng = random.Random(AUG03_VERSION)
    n = N_MAIN + N_RESERVE
    position = [B.POSITIONS[i % 3] for i in range(n)]
    classes = [("short", "short") if p == "both" else ("short",) for p in position]
    for pos in B.POSITIONS[:2]:
        idx = [i for i in range(N_MAIN) if position[i] == pos]
        for i in rng.sample(idx, len(idx) // 2):
            classes[i] = ("long",)
    for i in range(N_MAIN, n):
        if position[i] != "both":
            classes[i] = ("long",)
    return position, classes, rng


def aug03_candidates(position, classes, queues):
    """Passage tuples in trial order: the queue of the planned class, then the other queue (both: pairs
    (a, b) with a before b, ordered by b and then a, so one unusable passage cannot fill every trial)."""
    order = queues[classes[0]] + queues["short" if classes[0] == "long" else "long"]
    if position == "both":
        return ((a, order[y]) for y in range(1, len(order)) for a in order[:y])
    return ((n,) for n in order)


# ---------------------------------------------------------------- keys

def keys_of(text):
    """(exact text, normalised key, punct key); the two B keys are None when the punct key is empty."""
    p = build_lists.pk(text)
    return text, (build_lists.nk(text) if p else None), (p or None)


def key_matches(text, idx, skip=None):
    """IDs of a B.key_index with the same (exact text, normalised key only, punct key only); skip = own ID (B)."""
    exact = idx["exact"].get(text, [])
    _, n, p = keys_of(text)
    if p is None:
        return exact, [], []
    nk_all = idx["nk"].get(n, [])
    return (exact, [s for s in nk_all if s not in exact and s != skip],
            [s for s in idx["pk"].get(p, []) if s not in nk_all and s != skip])


def dup_keys(text):
    """Rule 3 keys (exact text, punct key or None when empty); an equal normalised key gives an equal punct key."""
    return text, (build_lists.pk(text) or None)


def first_seen(text, seen):
    """Value stored in seen = [{exact text: x}, {punct key: x}] for the first key text shares, else None."""
    return next((d[k] for d, k in zip(seen, dup_keys(text)) if k is not None and k in d), None)


def remember(text, value, seen):
    for d, k in zip(seen, dup_keys(text)):
        if k is not None:
            d.setdefault(k, value)


def pool_sha256(cases):
    """SHA-256 of "".join(f"{case_id},{text_sha256_12}\\n") over every case in case_id order (design D2)."""
    lines = "".join(f"{cid},{build_lists.h12(cases[cid]['text'])}\n" for cid in sorted(cases))
    return data_lists.sha256_bytes(lines.encode("utf-8"))


# ---------------------------------------------------------------- review sheet and test overlap

def parse_review(data):
    """({case_id: row}, errors R01-R09); BOM, CRLF and fully blank rows accepted, fields stripped."""
    if data.startswith(codecs.BOM_UTF8):
        data = data[len(codecs.BOM_UTF8):]
    try:
        recs = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        return {}, [f"R01 review.csv: {build_lists.exc_name(exc)} (not UTF-8 csv)"]
    if not recs or [f.strip() for f in recs[0]] != list(REVIEW_COLUMNS):
        return {}, [f"R02 review.csv: header is not {','.join(REVIEW_COLUMNS)}"]
    rows, errors = {}, []
    reviewers = {m.lower() for m in REVIEWERS}
    for line, rec in enumerate(recs[1:], start=2):
        if all(f.strip() == "" for f in rec):
            continue
        tag = f"review.csv line {line}"
        if len(rec) != len(REVIEW_COLUMNS):
            errors.append(f"R03 {tag}: {len(rec)} fields")
            continue
        if any("\n" in f or "\r" in f for f in rec):
            errors.append(f"R03 {tag}: line break inside a field")
            continue
        r = dict(zip(REVIEW_COLUMNS, (f.strip() for f in rec)))
        cid, dec = r["case_id"], r["decision"]
        if not CASE_ID_RE.fullmatch(cid):
            errors.append(f"R04 {tag}: case_id is not an augmentation case ID")
            continue
        if cid in rows:
            errors.append(f"R04 {tag}: duplicate case_id {cid}")
            continue
        if dec not in DECISIONS:
            errors.append(f"R05 {cid}: decision not one of {'|'.join(DECISIONS)} (no relabel: exclude a case "
                          "whose label is doubtful)")
        else:
            if dec == "exclude" and not r["reason"]:
                errors.append(f"R07 {cid}: exclude needs a reason")
            if dec == "pending" and r["reviewer"]:
                errors.append(f"R08 {cid}: a pending row has no reviewer")
            elif dec != "pending" and r["reviewer"].lower() not in reviewers:
                errors.append(f"R08 {cid}: reviewer must be one of {', '.join(REVIEWERS)} (README DATA-10)")
            if dec != "pending" and not build_lists.H12_RE.fullmatch(r["text_sha256_12"]):
                errors.append(f"R09 {cid}: a decided row needs the text_sha256_12 written by the script")
        rows[cid] = r
    return rows, errors


def read_overlap(path, cases):
    """(test_overlap record for the meta, {case_id: [levels in KEY_LEVELS order]}).

    status "pending" (no matches) when the file is missing or was made for another pool; a file that is
    not the output of scripts/check_cases_vs_test.py for this pool stops the run (T01).
    """
    info = {"file": OVERLAP, "status": "pending", "reason": None, "sha256": None,
            "pool_sha256": pool_sha256(cases), "counts": None}
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        info["reason"] = f"{OVERLAP} is missing"
        return info, {}
    except OSError as exc:
        raise Refused([f"T01 {OVERLAP}: unreadable ({build_lists.exc_name(exc)})"])
    info["sha256"] = data_lists.sha256_bytes(raw)
    try:
        doc = json.loads(raw.decode("utf-8"))
        ok = (isinstance(doc, dict) and doc.get("schema") == 1 and isinstance(doc.get("pool_sha256"), str)
              and isinstance(doc.get("matches"), list) and isinstance(doc.get("counts"), dict)
              and B._is_int(doc.get("n_cases")))
    except ValueError:
        ok = False
    if not ok:
        raise Refused([f"T01 {OVERLAP}: not the JSON written by scripts/check_cases_vs_test.py (schema 1 with "
                       "pool_sha256, n_cases, matches, counts)"])
    if doc["pool_sha256"] != info["pool_sha256"]:
        info["reason"] = f"{OVERLAP} was made for another pool (pool_sha256 {doc['pool_sha256'][:12]})"
        return info, {}
    errors, matches = [], {}
    if doc["n_cases"] != len(cases):
        errors.append(f"T01 {OVERLAP}: n_cases {doc['n_cases']} != {len(cases)} cases of the pool")
    for k, m in enumerate(doc["matches"]):
        if not (isinstance(m, dict) and isinstance(m.get("case_id"), str) and m.get("level") in KEY_LEVELS
                and isinstance(m.get("text_sha256_12"), str)):
            errors.append(f"T01 {OVERLAP}: match {k} needs case_id, text_sha256_12 and a level "
                          f"({'|'.join(KEY_LEVELS)})")
            continue
        cid, level = m["case_id"], m["level"]
        if cid not in cases or build_lists.h12(cases[cid]["text"]) != m["text_sha256_12"]:
            errors.append(f"T01 {OVERLAP}: match {k} ({B._ascii(cid)[:40]}) is not a case of the pool or its "
                          "text_sha256_12 differs")
        elif level in matches.get(cid, ()):
            errors.append(f"T01 {OVERLAP}: match {k} repeats {cid} {level}")
        else:
            matches.setdefault(cid, []).append(level)
    if errors:
        raise Refused(errors[:20])
    info.update(status="applied", counts=doc["counts"])
    return info, {cid: [lv for lv in KEY_LEVELS if lv in levels] for cid, levels in matches.items()}


# ---------------------------------------------------------------- build

class Builder:
    """One deterministic build: round 0 (selection and the automatic checks 1-5, which fix the pool), then
    the test overlap and the review, which change statuses only."""

    def __init__(self, src, llm, token_len):
        self.src, self.llm, self.token_len = src, llm, token_len
        self.train = src["train"]
        self.text = {sid: text for sid, _, text in src["tclean"]}
        dev = {i: ("", t) for i, t in src["dev"] if t}
        self.idx = {"validation": B.key_index({"train": {}, "validation": src["validation"]}),
                    "train": B.key_index({"train": src["train"], "validation": {}}),
                    "dev": B.key_index({"train": dev, "validation": {}})}
        self.dev_words = {}  # punct-key word count -> [(ID, words)] in ID order (one-word-edit candidates)
        for i in sorted(dev):
            words = build_lists.pk(dev[i][1]).split()
            self.dev_words.setdefault(len(words), []).append((i, words))
        self.vocab = leet_vocab(src["train"], src["validation"])
        self.walk = {}
        for label in LABEL_ORDER:
            ids = sorted(sid for sid, lab, _ in src["tclean"] if lab == label)
            random.Random(f"s4-aug:{SELECTION_SEED}:label{label}").shuffle(ids)
            self.walk[label] = {"ids": ids, "pos": 0}
        model = {c: v["generator"]["model"] for c, v in llm.items()}
        self.version = {"AUG-01": AUG01_VERSION, "AUG-03": f"{AUG03_VERSION}+llm:{model['AUG-03']}:{LLM['AUG-03']}",
                        "AUG-04": f"llm:{model['AUG-04']}:{LLM['AUG-04']}",
                        "AUG-05": f"llm:{model['AUG-05']}:{LLM['AUG-05']}"}
        self.cases = {}
        self.groups = []  # slot groups in filling order
        self.originals = []  # one row per walked original
        self.ineligible = {label: {} for label in LABEL_ORDER}
        self.seen = [{}, {}]  # exact text / punct key of the passing cases -> first case ID
        self.passage_excluded = []
        self.plan = []
        self.replacements = []
        self.overlap_hits = {"assigned": 0, "reserve": 0, "auto_excluded": 0}

    # -- originals

    def next_original(self, label, what):
        """(sentence ID, walk position) of the next eligible original of the label."""
        w = self.walk[label]
        while w["pos"] < len(w["ids"]):
            sid, order = w["ids"][w["pos"]], w["pos"]
            w["pos"] += 1
            why = ineligible_reason(self.text[sid], self.vocab)
            if why is None:
                return sid, order
            self.ineligible[label][why] = self.ineligible[label].get(why, 0) + 1
            self.originals.append({"sentence_id": sid, "label": label, "condition": "", "subtype": "",
                                   "order": order, "_case": None})
        raise Refused([f"C03 train_clean label {label}: no eligible original left for {what} "
                       f"({len(w['ids'])} rows walked)"])

    # -- records and checks

    def _record(self, cid, cond, subtype, label, text, *, group, source_split, source_id, position="",
                context_ids=""):
        if cid in self.cases:
            raise RuntimeError("case ID used twice")
        return {"case_id": cid, "group_id": group, "source_split": source_split, "source_id": source_id,
                "purpose": "train", "condition": cond, "subtype": subtype, "eval_group": "train",
                "label": str(label), "position": position, "context_ids": context_ids,
                "generator": GENERATOR[cond], "generator_version": self.version[cond], "status": "",
                "auto_flags": "", "text": text, "_reasons": [], "_flags": [], "_round0": ""}

    def fits(self, text):
        return max(self.token_len(text)) <= MAX_TOKENS

    def auto_checks(self, cond, label, text, own=None):
        """(exclusion reasons, flags) of one case text: rules 1, 2, 4 and 5 (see Rules)."""
        reasons, flags = [], []
        if cond in LLM_LABEL and not format_ok(cond, text):
            reasons.append("format")
        for level, ids in zip(KEY_LEVELS, key_matches(text, self.idx["validation"])):
            if ids:
                reasons.append(f"{level}={ids[0]}")
        exact, nk_ids, pk_ids = key_matches(text, self.idx["train"], skip=own)
        if exact:
            reasons.append(f"exact={exact[0]}")
        conflict = [s for s in nk_ids + pk_ids if self.train[s][0] != str(label)]
        if conflict:
            reasons.append(f"label_conflict={min(conflict)}")
        same = {name: [s for s in ids if self.train[s][0] == str(label)]
                for name, ids in (("train_normalised_key", nk_ids), ("train_punct_key", pk_ids))}
        if cond in LLM_LABEL:  # new LLM text that repeats a training sentence is not new material
            if any(same.values()):
                reasons.append(f"train_material={min(s for ids in same.values() for s in ids)}")
        else:  # train-derived case (its own original is skipped): a flag only
            flags += [f"{name}={ids[0]}" for name, ids in same.items() if ids]
        dev = [s for ids in key_matches(text, self.idx["dev"]) for s in ids]
        if dev:
            reasons.append(f"dev_material={min(dev)}")
        if cond in LLM_LABEL:
            words = build_lists.pk(text).split()
            near = [i for n in (len(words) - 1, len(words), len(words) + 1)
                    for i, w in self.dev_words.get(n, ()) if B.one_word_edit(words, w)]
            if near:
                reasons.append(f"dev_one_word_edit={min(near)}")
        if not self.fits(text):
            reasons.append("length_over_256")
        return reasons, flags

    def new_group(self, cond, subtype, label, quota):
        g = {"condition": cond, "subtype": subtype, "label": label, "quota": quota, "assigned": [], "reserve": []}
        self.groups.append(g)
        return g

    def settle(self, rec, g, original=None):
        """Store a checked record: auto_excluded when it has a reason, else assigned (up to the quota) or reserve."""
        cid = rec["case_id"]
        if rec["_reasons"]:
            rec["status"] = "auto_excluded"
        else:
            rec["status"] = "assigned" if len(g["assigned"]) < g["quota"] else "reserve"
            g[rec["status"]].append(cid)
            remember(rec["text"], cid, self.seen)
        rec["_round0"] = rec["status"]
        self.cases[cid] = rec
        if original is not None:
            self.originals.append({"sentence_id": original[0], "label": g["label"], "condition": rec["condition"],
                                   "subtype": rec["subtype"], "order": original[1], "_case": cid})

    def place(self, rec, g, original=None):
        """Rule 3 for a record that passed the other checks, then settle()."""
        if not rec["_reasons"]:
            dup = first_seen(rec["text"], self.seen)
            if dup:
                rec["_reasons"].append(f"duplicate_of={dup}")
        self.settle(rec, g, original)

    # -- round 0

    def round0(self):
        for label in LABEL_ORDER:
            for subtype in AUG01_SUBTYPES:
                self.fill_aug01(subtype, label)
        self.fill_aug03()
        for cond in ("AUG-04", "AUG-05"):
            self.fill_llm(cond)

    def fill_aug01(self, subtype, label):
        g = self.new_group("AUG-01", subtype, label, N_MAIN)
        while len(g["reserve"]) < N_RESERVE:
            sid, order = self.next_original(label, f"AUG-01 {subtype}")
            cid = f"aug-01-{subtype}-{sid[-6:]}"
            out = aug01_variant(subtype, cid, self.text[sid])
            rec = self._record(cid, "AUG-01", subtype, label, out or "", group=sid, source_split="train",
                               source_id=sid)
            if out is None:
                rec["_reasons"].append("no_eligible_char")
            else:
                rec["_reasons"], rec["_flags"] = self.auto_checks("AUG-01", label, out, own=sid)
            self.place(rec, g, (sid, order))

    def passages(self):
        """{n: (length class, text)} of the usable AUG-03 passages (rule 5); the others -> passage_excluded."""
        pool, seen = {}, [{}, {}]
        for it in self.llm["AUG-03"]["items"]:
            n, text, reasons = it["n"], it["text"], []
            if not format_ok("AUG-03", text):
                reasons.append("format")
            else:
                dev = [s for ids in key_matches(text, self.idx["dev"]) for s in ids]
                if dev:
                    reasons.append(f"dev_material={min(dev)}")
                dup = first_seen(text, seen)
                if dup is not None:
                    reasons.append(f"duplicate_of={LLM['AUG-03']}:{dup:03d}")
            if reasons:
                self.passage_excluded.append({"n": n, "reasons": reasons})
                continue
            pool[n] = (it["length"], text)
            remember(text, n, seen)
        return pool

    def fill_aug03(self):
        pool = self.passages()
        position, classes, rng = plan_aug03()
        queues = {c: sorted(n for n, (cls, _) in pool.items() if cls == c) for c in ("short", "long")}
        rng.shuffle(queues["short"])
        rng.shuffle(queues["long"])
        groups = {label: self.new_group("AUG-03", "context", label, N_MAIN) for label in LABEL_ORDER}
        for i in range(N_MAIN + N_RESERVE):
            pos = position[i]
            if next(aug03_candidates(pos, classes[i], queues), None) is None:
                break  # no passage left: the remaining pairs stay empty (shortfall in the meta)
            cur = {label: self.next_original(label, "AUG-03") for label in LABEL_ORDER}
            while True:
                origs = {label: self.text[cur[label][0]] for label in LABEL_ORDER}
                first = chosen = None
                for ctx in itertools.islice(aug03_candidates(pos, classes[i], queues), AUG03_TRIES):
                    first = first or ctx
                    if all(self.fits(B.assemble(pos, [pool[n][1] for n in ctx], o)) for o in origs.values()):
                        chosen = ctx
                        break
                ctx = chosen or first
                recs, pair_seen = {}, [{}, {}]
                for label in LABEL_ORDER:
                    sid = cur[label][0]
                    text = B.assemble(pos, [pool[n][1] for n in ctx], origs[label])
                    if origs[label] not in text:
                        raise Refused([f"C07 AUG-03 pair {i}: the original {sid} is not verbatim in its text"])
                    rec = self._record(f"aug-03-{sid[-6:]}", "AUG-03", "context", label, text, group=sid,
                                       source_split="train", source_id=sid, position=pos,
                                       context_ids=";".join(map(str, ctx)))
                    rec["_reasons"], rec["_flags"] = self.auto_checks("AUG-03", label, text, own=sid)
                    if not rec["_reasons"]:  # rule 3: earlier cases, then the other case of this pair
                        dup = first_seen(text, self.seen) or first_seen(text, pair_seen)
                        if dup:
                            rec["_reasons"].append(f"duplicate_of={dup}")
                        remember(text, rec["case_id"], pair_seen)
                    recs[label] = rec
                bad = [label for label in LABEL_ORDER if recs[label]["_reasons"]]
                if not bad:
                    for label in LABEL_ORDER:
                        self.settle(recs[label], groups[label], cur[label])
                    for n in ctx:
                        queues[pool[n][0]].remove(n)
                    self.plan.append({"pair": i, "block": "assigned" if i < N_MAIN else "reserve", "position": pos,
                                      "classes": list(classes[i]), "context_ids": list(ctx),
                                      "context_classes": [pool[n][0] for n in ctx],
                                      "originals": {f"label_{label}": cur[label][0] for label in LABEL_ORDER}})
                    break
                for label in bad:  # the failing side is recorded and replaced; the pair is tried again
                    self.settle(recs[label], groups[label], cur[label])
                    cur[label] = self.next_original(label, "AUG-03")

    def fill_llm(self, cond):
        label, name = LLM_LABEL[cond], LLM[cond]
        items = iter(self.llm[cond]["items"])
        for fam, size, quota in FAMILIES[cond]:
            g = self.new_group(cond, fam, label, quota)
            for it in itertools.islice(items, size):
                cid = f"aug-{cond[-2:]}-{it['n']:03d}"
                rec = self._record(cid, cond, fam, label, it["text"], group=cid, source_split="new",
                                   source_id=f"{name}:{it['n']:03d}")
                rec["_reasons"], rec["_flags"] = self.auto_checks(cond, label, it["text"])
                self.place(rec, g)

    # -- statuses after round 0

    def apply_overlap(self, matches):
        """Rule 6: matching cases are excluded; an assigned one is replaced by the next non-matching reserve."""
        for cid in sorted(matches):
            c = self.cases[cid]
            c["_reasons"] += [f"test_{level}" for level in matches[cid]]
            self.overlap_hits[c["status"]] += 1
        for g in self.groups:
            free = [cid for cid in g["reserve"] if cid not in matches]
            for cid in [c for c in g["assigned"] if c in matches]:
                self.cases[cid]["status"] = "auto_excluded"
                repl = free.pop(0) if free else None
                if repl is not None:
                    self.cases[repl]["status"] = "assigned"
                self.replacements.append({"condition": g["condition"], "subtype": g["subtype"], "label": g["label"],
                                          "excluded_case_id": cid, "replacement_case_id": repl})
            for cid in g["reserve"]:
                if cid in matches:
                    self.cases[cid]["status"] = "auto_excluded"

    def finish(self):
        """auto_flags strings; no usable or reserve case may hold a non-ASCII character (held-out method)."""
        bad = []
        for cid, c in self.cases.items():
            c["auto_flags"] = ";".join([f"exclude:{x}" for x in c["_reasons"]] + [f"flag:{x}" for x in c["_flags"]])
            if c["status"] != "auto_excluded" and not c["text"].isascii():
                bad.append(cid)
        if bad:
            raise Refused([f"C08 non-ASCII text in {build_lists.fmt_ids(bad)}: originals and LLM text are ASCII, "
                           f"so this is the held-out {HELD_OUT} substitution or a broken rule"])


def merge_review(b, review, applied):
    """(rows of the new review.csv, stats); applies exclude decisions (assigned -> review_excluded, no replacement).

    applied = the test overlap is applied. See Rules (Review) for the rows that are kept or refused.
    """
    errors, rows, stats = [], {}, {"dropped": 0, "retained_without_effect": 0}
    for cid, r in review.items():
        c = b.cases.get(cid)
        if r["decision"] == "pending":
            if c is not None and c["status"] == "assigned":
                rows[cid] = {**r, "text_sha256_12": build_lists.h12(c["text"])}
            else:
                stats["dropped"] += 1
            continue
        if c is None:
            errors.append(f"R09 {cid}: decided ({r['decision']}) but the case would disappear")
        elif r["text_sha256_12"] != build_lists.h12(c["text"]):
            errors.append(f"R09 {cid}: stale decision ({r['decision']}): text_sha256_12 {r['text_sha256_12']} != "
                          f"the current case text {build_lists.h12(c['text'])}; review the case again")
        elif c["status"] == "assigned":
            rows[cid] = r
            if r["decision"] == "exclude":
                c["status"] = "review_excluded"
        elif (c["status"], c["_round0"]) == ("auto_excluded", "assigned") or (c["status"] == "reserve"
                                                                             and not applied):
            rows[cid] = r  # excluded by the test overlap after the review, or waiting for it: kept, no effect
            stats["retained_without_effect"] += 1
        else:
            errors.append(f"R09 {cid}: decided ({r['decision']}) but the case is {c['status']} (never reviewable)")
    if errors:
        raise Refused(errors)
    for cid, c in b.cases.items():
        if c["status"] == "assigned" and cid not in rows:
            rows[cid] = {"case_id": cid, "text_sha256_12": build_lists.h12(c["text"]), "decision": "pending",
                         "reason": "", "reviewer": ""}
    return [rows[k] for k in sorted(rows)], stats


# ---------------------------------------------------------------- rendering

def render_originals(b):
    rows = [{**o, "status": b.cases[o["_case"]]["status"] if o["_case"] else "ineligible"} for o in b.originals]
    rows.sort(key=lambda o: (LABEL_ORDER.index(o["label"]), o["order"]))
    return B.render_csv(ORIGINALS_COLUMNS, rows), len(rows)


def usable_ids(b, rows):
    """Case IDs used for training: status assigned and decision keep, ascending."""
    return sorted(r["case_id"] for r in rows
                  if r["decision"] == "keep" and b.cases[r["case_id"]]["status"] == "assigned")


def render_train_aug(b, ids):
    lines = [TRAIN_AUG_HEADER] + [f"{cid},{b.cases[cid]['label']},{build_lists.h12(b.cases[cid]['text'])}"
                                  for cid in ids]
    return ("\n".join(lines) + "\n").encode("ascii")


def group_counts(b, rows):
    """One record per slot group (filling order): statuses, review decisions, usable cases, shortfall."""
    decision = {r["case_id"]: r["decision"] for r in rows}
    usable = set(usable_ids(b, rows))
    out = []
    for g in b.groups:
        cases = [c for c in b.cases.values()
                 if (c["condition"], c["subtype"], int(c["label"])) == (g["condition"], g["subtype"], g["label"])]
        rec = {"condition": g["condition"], "subtype": g["subtype"], "label": g["label"], "quota": g["quota"]}
        rec.update({st: sum(c["status"] == st for c in cases) for st in STATUSES})
        reviewed = [decision[c["case_id"]] for c in cases
                    if c["status"] in ("assigned", "review_excluded") and c["case_id"] in decision]
        rec.update({dec: reviewed.count(dec) for dec in DECISIONS})
        rec["usable"] = sum(c["case_id"] in usable for c in cases)
        rec["shortfall"] = g["quota"] - rec["assigned"] - rec["review_excluded"]
        out.append(rec)
    return out


def pattern_counts(rows, vocab):
    """{pattern: {"label_0": rows, "label_1": rows}} over (label, text) rows."""
    out = {p: {"label_0": 0, "label_1": 0} for p in PATTERNS}
    for label, text in rows:
        for p in surface_patterns(text, vocab):
            out[p][f"label_{label}"] += 1
    return out


def token_summary(b):
    out = {}
    for name in ("all",) + CONDITIONS:
        lens = [b.token_len(c["text"]) for c in b.cases.values()
                if c["status"] == "assigned" and name in ("all", c["condition"])]
        out[name] = {k: B._summary([x[j] for x in lens]) for j, k in enumerate(B.TOKENIZER_KEYS)}
    return out


def passage_flags(b):
    """Passages against the prompt format rules (numbers only, no exclusion; the reviewer decides)."""
    out = {"word_count_out_of_range": [], "forbidden_word_or_question": []}
    for it in b.llm["AUG-03"]["items"]:
        lo, hi = AUG03_WORDS[it["length"]]
        if not lo <= len(it["text"].split()) <= hi:
            out["word_count_out_of_range"].append(it["n"])
        if B.ROB03_SCAN.search(it["text"]):
            out["forbidden_word_or_question"].append(it["n"])
    return out


def build_meta(b, src, fingerprints, overlap, rows, stats, ids, not_ready, files, n_rows):
    cases = [c for _, c in sorted(b.cases.items())]
    groups = group_counts(b, rows)
    by_label = {f"label_{label}": sum(int(b.cases[i]["label"]) == label for i in ids) for label in LABEL_ORDER}
    used = sorted({int(n) for c in cases if c["condition"] == "AUG-03" and c["status"] != "auto_excluded"
                   for n in c["context_ids"].split(";")})
    excluded = {p["n"] for p in b.passage_excluded}
    # Passages whose assigned cases have one label only (a pair lost one side to the test overlap or the review).
    in_use = {label: {int(n) for c in cases if c["condition"] == "AUG-03" and c["status"] == "assigned"
                      and c["label"] == str(label) for n in c["context_ids"].split(";")} for label in LABEL_ORDER}
    single = {f"label_{label}_only": sorted(in_use[label] - in_use[1 - label]) for label in LABEL_ORDER}
    reviewers = {}
    for r in rows:
        if r["decision"] != "pending":
            name = next(m for m in REVIEWERS if m.lower() == r["reviewer"].lower())
            reviewers[name] = reviewers.get(name, 0) + 1
    usable = set(ids)
    aug = {"augmentation_assigned": [(c["label"], c["text"]) for c in cases if c["status"] == "assigned"],
           "augmentation_usable": [(c["label"], c["text"]) for c in cases if c["case_id"] in usable]}
    patterns = {"train_clean": pattern_counts([(lab, text) for _, lab, text in src["tclean"]], b.vocab),
                **{scope: pattern_counts(texts, b.vocab) for scope, texts in aug.items()}}
    patterns["after"] = {p: {k: patterns["train_clean"][p][k] + patterns["augmentation_usable"][p][k]
                             for k in ("label_0", "label_1")} for p in PATTERNS}
    here = Path(__file__).parent
    return {
        "schema": SCHEMA,
        "purpose": "train",
        "readme": "README 3.3 DATA-07..12, 5.3 (condition C), 3.2 keys; design .omc/autopilot/stage4-design.md D1, D2",
        "script": {"path": SCRIPT_REL, "sha256_lf": data_lists.sha256_lf(Path(__file__))},
        "inputs": {**src["inputs"], **{f"scripts/{n}": {"path": f"scripts/{n}",
                                                        "sha256_lf": data_lists.sha256_lf(here / n)}
                                       for n in ("build_dev_cases.py", "build_lists.py", "data_lists.py",
                                                 "model_specs.py")}},
        "rules": RULES,
        "seeds": {"selection_seed": SELECTION_SEED,
                  "selection": f"random.Random(f's4-aug:{SELECTION_SEED}:label{{label}}').shuffle(sorted IDs)",
                  "aug01": f"random.Random(f'{AUG01_VERSION}:{{case_id}}')",
                  "aug03": f"random.Random('{AUG03_VERSION}'): long-passage samples (assigned prefix pairs, "
                           "assigned suffix pairs), short queue, long queue"},
        "rule_versions": {"aug01": AUG01_VERSION, "aug03": AUG03_VERSION, "aug01_subtypes": list(AUG01_SUBTYPES),
                          "rob01_operations": f"scripts/build_dev_cases.py ROB01 ({B.ROB01_VERSION})",
                          "leet_map": B.LEET, "max_tokens": MAX_TOKENS, "aug03_tries": AUG03_TRIES,
                          "held_out": f"ROB-01 {HELD_OUT} (never used for training augmentation; refused, C08)"},
        "generators": {v["name"]: {"file": v["file"], "sha256_lf": v["sha256_lf"], "prompt": v["prompt"],
                                   "prompt_sha256_lf": v["prompt_sha256_lf"], "generator": v["generator"],
                                   "items": len(v["items"])} for v in b.llm.values()},
        "tokenizers": fingerprints,
        "allocation": {f"label_{label}": {"train_clean_rows": len(b.walk[label]["ids"]),
                                          "walked": b.walk[label]["pos"],
                                          "ineligible": dict(sorted(b.ineligible[label].items()))}
                       for label in LABEL_ORDER},
        "aug03_plan": b.plan,
        "aug03_passages": {"used": used, "excluded": b.passage_excluded,
                           "unused": [it["n"] for it in b.llm["AUG-03"]["items"]
                                      if it["n"] not in excluded and it["n"] not in used],
                           "single_label": single,
                           "prompt_format_flags": passage_flags(b)},
        "counts": {"groups": groups, "cases": len(cases),
                   "status": {st: sum(c["status"] == st for c in cases) for st in STATUSES},
                   "assigned": {f"label_{label}": sum(c["status"] == "assigned" and c["label"] == str(label)
                                                      for c in cases) for label in LABEL_ORDER},
                   "usable": {**by_label, "total": len(ids)},
                   "shortfall": sum(g["shortfall"] for g in groups)},
        "automatic_exclusions": [{"case_id": c["case_id"], "reasons": c["_reasons"]} for c in cases
                                 if c["status"] == "auto_excluded"],
        "flags": [{"case_id": c["case_id"], "flags": c["_flags"]} for c in cases
                  if c["_flags"] and c["status"] != "auto_excluded"],
        "token_lengths": token_summary(b),
        "test_overlap": {**overlap, "cases_matched": b.overlap_hits, "replacements": b.replacements},
        "review": {"rows": len(rows), **{dec: sum(r["decision"] == dec for r in rows) for dec in DECISIONS},
                   "reviewers": dict(sorted(reviewers.items())),
                   "retained_without_effect": stats["retained_without_effect"]},
        "ready_for_training": not not_ready,
        "not_ready_reasons": not_ready,
        "train_patterns_after": {"definitions": PATTERN_DEFS,
                                 "vocabulary": "leetspeak vocabulary: all-letter tokens of >= 3 letters in the raw "
                                               "train and validation rows (as build_dev_cases.train_patterns)",
                                 "scopes": "train_clean; augmentation_assigned (status assigned); "
                                           "augmentation_usable (assigned and keep); after = train_clean + "
                                           "augmentation_usable",
                                 "counts": patterns},
        "outputs": {n: {"sha256": data_lists.sha256_bytes(files[n]), "rows": n_rows[n]} for n in n_rows},
    }


def compute(src, d, token_len=None, fingerprints=None):
    """Every output for the folder d: ({name: bytes}, builder, review rows, meta)."""
    check_held_out()
    llm = load_llm(d)
    if token_len is None:
        token_len, fingerprints = B.make_token_len()
    review = {}
    try:
        review_bytes = (d / "review.csv").read_bytes()
    except FileNotFoundError:
        review_bytes = None  # first build: every assigned case gets a pending row
    except OSError as exc:
        raise Refused([f"R01 review.csv: unreadable ({build_lists.exc_name(exc)})"])
    if review_bytes is not None:
        review, errors = parse_review(review_bytes)
        if errors:
            raise Refused(errors)
    b = Builder(src, llm, token_len)
    b.round0()
    overlap, matches = read_overlap(d / OVERLAP, b.cases)  # the pool is fixed here
    b.apply_overlap(matches)
    applied = overlap["status"] == "applied"
    rows, stats = merge_review(b, review, applied)
    b.finish()
    pending = sum(r["decision"] == "pending" for r in rows)
    not_ready = ([] if applied else [f"test overlap pending: {overlap['reason']}"]) + (
        [f"review pending: {pending} row(s)"] if pending else [])
    ids = usable_ids(b, rows)
    cases = [{k: c[k] for k in CASE_COLUMNS} for _, c in sorted(b.cases.items())]
    originals, n_originals = render_originals(b)
    files = {"originals.csv": originals, "cases.csv": B.render_csv(CASE_COLUMNS, cases),
             "review.csv": B.render_csv(REVIEW_COLUMNS, rows)}
    n_rows = {"originals.csv": n_originals, "cases.csv": len(cases), "review.csv": len(rows)}
    if not not_ready:
        files[TRAIN_AUG], n_rows[TRAIN_AUG] = render_train_aug(b, ids), len(ids)
    meta = build_meta(b, src, fingerprints, overlap, rows, stats, ids, not_ready, files, n_rows)
    files[META] = B.render_json(meta)
    return files, b, rows, meta


# ---------------------------------------------------------------- write / check / report

def write_outputs(d, files):
    """Each file to <name>.tmp, then os.replace in FILES order (meta last); a train_aug.csv that is not
    in files is deleted first. Returns None or a W01 error."""
    tmps, current = [], "creating the folder"
    try:
        d.mkdir(parents=True, exist_ok=True)
        if TRAIN_AUG not in files:
            current = f"deleting the stale {TRAIN_AUG}"
            (d / TRAIN_AUG).unlink(missing_ok=True)
        for name in files:
            current = f"writing {name}.tmp"
            tmp = d / (name + ".tmp")
            tmp.unlink(missing_ok=True)
            with open(tmp, "xb") as f:
                tmps.append(tmp)
                f.write(files[name])
        for name in [n for n in FILES if n in files]:
            current = f"replacing {name}"
            os.replace(d / (name + ".tmp"), d / name)
        return None
    except Exception as exc:
        return f"W01 {build_lists.exc_name(exc)} while {current}; rerun after fixing the cause"
    finally:
        for tmp in tmps:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def check_outputs(d, files):
    errors = B.check_files(d, files)
    if TRAIN_AUG not in files and (d / TRAIN_AUG).exists():
        errors.append(f"K01 --check: {TRAIN_AUG} exists but the set is not ready for training")
    return errors


def original_text(c, src):
    """The train sentence a derived case comes from ('' for new cases)."""
    return src["train"][c["source_id"]][1] if c["source_split"] == "train" else ""


def render_view(b, rows, src):
    """Read-only reviewer aid (not an output, not an input): the cases of review.csv with text and original."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["case_id", "condition", "subtype", "label", "position", "status", "decision", "reason", "text",
                "original_text"])
    for r in rows:
        c = b.cases[r["case_id"]]
        w.writerow([c["case_id"], c["condition"], c["subtype"], c["label"], c["position"], c["status"],
                    r["decision"], r["reason"], c["text"], original_text(c, src)])
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


def report(b, rows, meta, print_n, src):
    for g in meta["counts"]["groups"]:
        short = f" (shortfall {g['shortfall']})" if g["shortfall"] else ""
        say(f"{g['condition']} {g['subtype']} label {g['label']}: assigned {g['assigned']}/{g['quota']}{short}, "
            f"reserve {g['reserve']}, auto_excluded {g['auto_excluded']}, review_excluded {g['review_excluded']}; "
            f"review pending {g['pending']}, keep {g['keep']}, exclude {g['exclude']}")
    c = meta["counts"]
    say(f"total: {c['cases']} cases; assigned {sum(c['assigned'].values())} (label 1: {c['assigned']['label_1']}, "
        f"label 0: {c['assigned']['label_0']}); usable {c['usable']['total']} (label 1: {c['usable']['label_1']}, "
        f"label 0: {c['usable']['label_0']}); AUG-03 passages not used: {len(meta['aug03_passages']['excluded'])} "
        f"excluded, {len(meta['aug03_passages']['unused'])} spare")
    single = meta["aug03_passages"]["single_label"]
    if any(single.values()):
        say(f"AUG-03 passages whose assigned cases have one label only: {len(single['label_1_only'])} (label 1), "
            f"{len(single['label_0_only'])} (label 0); see cases_meta.json aug03_passages.single_label")
    t = meta["test_overlap"]
    if t["status"] == "applied":
        say(f"test overlap: applied ({sum(t['cases_matched'].values())} matching case(s), "
            f"{len(t['replacements'])} assigned case(s) replaced or left empty)")
    else:
        say(f"test overlap: pending ({t['reason']}): run scripts/check_cases_vs_test.py, then build again")
    if meta["ready_for_training"]:
        say(f"{TRAIN_AUG}: {meta['outputs'][TRAIN_AUG]['rows']} usable cases")
    else:
        say(f"{TRAIN_AUG}: not written ({'; '.join(meta['not_ready_reasons'])})")
    if not print_n:
        return
    pending = {r["case_id"] for r in rows if r["decision"] == "pending"}
    for cond in CONDITIONS:
        pend = [c for k, c in sorted(b.cases.items()) if c["condition"] == cond and k in pending][:print_n]
        say(f"-- {cond}: first {len(pend)} pending case(s) (terminal only)")
        for c in pend:
            text = c["text"].replace("\r", "\\r").replace("\n", "\\n")
            say(f"{c['case_id']} label {c['label']} {c['position'] or c['subtype']}: {text}")
            if original_text(c, src):
                say(f"    original {c['source_id']}: {original_text(c, src)}")


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Build the stage-4 training augmentation set (README 3.3, 5.3).")
    p.add_argument("--check", action="store_true", help="rebuild in memory and compare with the files on disk")
    p.add_argument("--dir", help="cases folder (default data/cases/train)")
    p.add_argument("--print", type=int, default=0, metavar="N", dest="print_n",
                   help="print up to N pending cases per condition with their originals (terminal only)")
    p.add_argument("--view", metavar="FILE",
                   help="also write a read-only reviewer view (UTF-8 with BOM) to FILE outside the outputs")
    return p.parse_args(argv)


def _run(args):
    d = Path(args.dir).resolve() if args.dir else DEFAULT_DIR
    if build_lists._under(d, build_lists.DATASETS) or build_lists._under(d, ROOT / DEV_REL):
        raise Refused([f"C06 --dir is under datasets/ or {DEV_REL} (raw data and the development cases stay "
                       "read-only)"])
    view = Path(args.view).resolve() if args.view else None
    if view and (view.parent == d or view.name in FILES + (OVERLAP,)
                 or (build_lists._under(view, ROOT) and not B._ignored(view))):
        raise Refused([f"C06 --view {args.view}: write the view outside the repo or to a gitignored path, "
                       "never into the cases folder"])
    src = load_sources()
    files, b, rows, meta = compute(src, d)
    written = ", ".join(n for n in FILES if n in files)
    if args.check:
        errors = check_outputs(d, files)
        if errors:
            raise Refused(errors)
        say(f"OK --check: {written} equal the rebuilt bytes"
            + ("" if TRAIN_AUG in files else f"; no {TRAIN_AUG} (not ready)"))
    else:
        err = write_outputs(d, files)
        if err:
            raise Refused([err])
        say(f"OK wrote {written} ({len(b.cases)} cases, {len(rows)} review rows)")
    report(b, rows, meta, args.print_n, src)
    if view:
        view.parent.mkdir(parents=True, exist_ok=True)
        view.write_bytes(render_view(b, rows, src))
        say(f"OK wrote the read-only view {view.name} ({len(rows)} cases); decide in review.csv only")
    return 0


def main(argv=None):
    args = _parse_args(argv)
    data_lists.install_test_guard()
    try:
        return _run(args)
    except Refused as exc:
        for e in exc.errors:
            say("FAIL " + e)
        say(f"refused: {len(exc.errors)} error(s); nothing written")
        return 1
    except Exception as exc:
        lines = [ln for frame, ln in traceback.walk_tb(exc.__traceback__)
                 if os.path.basename(frame.f_code.co_filename) == os.path.basename(__file__)]
        say(f"FAIL X01 unexpected {build_lists.exc_name(exc)} at build_train_aug.py:{lines[-1] if lines else '?'} "
            "(message suppressed)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
