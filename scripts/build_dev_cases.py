"""Stage-3 development cases (README 3.3 DATA-07..12, 5.2 ROB-01..04): allocation, ROB-01 rules,
ROB-03 assembly, LLM outputs, automatic checks, review sheet and replacements -> data/cases/dev/.

Run from Git Bash at the repo root:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/build_dev_cases.py --originals
        (allocation only -> originals.csv; needs no LLM file and no tokenizer)
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/build_dev_cases.py
        (build: needs the three LLM files -> originals.csv, cases.csv, review.csv, cases_meta.json)
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/build_dev_cases.py --apply-review
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/build_dev_cases.py --check
Options: --dir DIR (default data/cases/dev; holds prompts/, llm/ and the outputs); --print N prints, to
the terminal only, up to N pending assigned cases per condition (with the original sentence for derived
cases) after a build, --apply-review or --check; --view FILE also writes a read-only reading aid for the
reviewer (UTF-8 with BOM so spreadsheets read it; every assigned case with its current decision, text
and original sentence) to FILE, which must be outside the repo or a gitignored path (never one of the
outputs). The view is not an input: decisions are made in review.csv only.

Rules (.omc/autopilot/spec-stage3.md):
  - Inputs. data/processed/validation_clean.csv (its SHA-256 must equal lists_meta.json outputs) joined
    to the raw validation rows with data_lists.load_rows() / join() (label and text_sha256_12 checked);
    all train and validation rows for the key checks. test.csv is never opened: main() installs
    data_lists.install_test_guard() before any data is read. LLM outputs <dir>/llm/<name>.json and their
    prompts <dir>/prompts/<name>.md for name in rob04_benign_v1, rob03_contexts_v1, rob02_paraphrase_v1;
    each JSON is {"generator": {"model": str, "prompt_sha256": hex, ...}, "items": [...]} with the
    item fields of its prompt; prompt_sha256 must equal the LF-normalised SHA-256 of the prompt file.
    ROB-03 token lengths use model_specs.load_tokenizer for bert, roberta and deberta (HF_HUB_OFFLINE=1).
    cases_meta.json "inputs" also records the LF SHA-256 of build_lists.py, data_lists.py, model_specs.py.
  - Allocation. For label in (1, 0): the sorted validation_clean IDs of that label are shuffled with
    random.Random(f"s3-dev:20261116:label{label}"); positions 0-49 -> rob01, 50-99 -> rob02,
    100-149 -> rob03, 150-164 -> reserve (shared by ROB-01..03, used in order). originals.csv `order`
    is that 0-based position. ROB-04: LLM items 1-50 main, 51-65 reserve (used in order).
  - Case IDs: dev-rob01-{case|space|char|homoglyph}-NNNNNN, dev-rob02-NNNNNN, dev-rob03-NNNNNN
    (NNNNNN = validation row number), dev-rob04-NNN (NNN = LLM item n). group_id = the original's
    sentence ID (all ROB-01 subtypes of one original share it); ROB-04 group_id = case_id and
    source_id = rob04_benign_v1:NNN.
  - ROB-01 v1, one random.Random(f"rob01-v1:{case_id}") per case. Letters are ASCII letters.
    case: each letter upper with p=0.5 else lower; the draw must change the casing of >= ceil(30%) of
    the letters; up to 20 redraws (21 draws), then letters still in their original casing are flipped
    in a seeded order until the threshold holds. space: words = maximal ASCII letter runs; k =
    rng.choice((1, 2)) of the longest words with >= 4 letters (ties: earlier first) are spelled out
    with single spaces; every inter-word whitespace run whose neighbouring token (maximal non-space
    run) contains a spelled-out word becomes 2 or 3 spaces (rng.choice per run, ascending), so the
    boundary of a spelled-out word stays visible; then round(g / 3) of the g other runs (rng.sample)
    become 2 or 3 spaces (rng.choice per run, ascending). char: a e i o s t (either case)
    -> 4 3 1 0 5 7 with p=0.5 each. homoglyph (HELD-OUT, eval_group held_out; never for augmentation):
    the Latin -> Cyrillic map HOMOGLYPH with p=0.5 each. char/homoglyph force one rng.choice position
    when the draws changed none. A subtype with no eligible character -> text "" and the whole
    original is automatically excluded (exclude:no_eligible_char).
  - ROB-02: the paraphrase of the original from rob02_paraphrase_v1.json (one per rob02 and reserve
    original; ids and labels must match originals.csv).
  - ROB-03 v1. Pair index i = 0-49 for the rob03 positions 100-149 and 50-64 for the reserve positions
    150-164; the attack and the benign original with the same index share one context assignment.
    rng = random.Random("rob03-v1"): (1) the 65 indices are shuffled and the j-th gets position
    (prefix, suffix, both)[j % 3]; in that order prefix and suffix each alternate short, long; both
    takes two short passages; (2) the short (1-80) and (3) the long (81-120) passage numbers are
    shuffled into queues. Indices are then served in ascending order: candidates are the queue
    passages in queue order (long class: long queue, then short queue; both: ordered pairs of short
    queue passages); the first candidate with which both originals of the index give <= 256 tokens
    (special tokens included) for all three tokenizers is used and removed from its queue (no reuse).
    If no candidate fits, the first candidate is used and each original that does not fit is
    excluded (exclude:length_over_256); no passage left -> exclude:no_context_passage. Text: prefix
    "{ctx}\n\n{orig}", suffix "{orig}\n\n{ctx}", both "{ctx1}\n\n{orig}\n\n{ctx2}"; context_ids are
    the passage numbers joined by ";". The original must appear verbatim (else C07). cases_meta
    rob03_plan records the planned classes and the classes actually used after fallback.
  - ROB-04: the LLM item text, label 0, subtype benign_security.
  - Automatic checks (keys build_lists.nk / pk; an empty punct key is no key). Against every train and
    validation row: exact text equal -> exclude:exact=<lowest ID> (all conditions); normalised key
    equal -> normalised_key=<ID>, else punct key equal -> punct_key=<ID>: a flag (flag:...) for
    ROB-01..03, an exclusion for ROB-04; the case's own original is skipped (ROB-01 equality with it is
    expected; for ROB-02 it is the next rule). ROB-02: same punct key as its original ->
    exclude:same_punct_key_as_original; a one-word edit of it (README 3.2 level C: punct-key word lists
    differ by one inserted, deleted or substituted word, the shorter has >= 5 words) ->
    exclude:one_word_edit_of_original. A case text equal to an earlier case text of the dev set ->
    exclude:duplicate_of=<case_id>. Order: ROB-01, ROB-02, ROB-03 (labels 1 then 0, slots in order,
    each replacement right after its slot), then ROB-04, then review replacements. Formal flags (no
    exclusion): flag:line_break (ROB-02/ROB-04 LLM text), flag:rob04_length (ROB-04 outside 40-200
    characters). One excluded case excludes its whole group (siblings get exclude:group=<case_id>).
    cases_meta.json "llm_item_flags" lists LLM items against prompt format rules (IDs only): ROB-03
    word count, line break, a forbidden word or a question mark; ROB-04 category, length, line break.
  - Replacement. An excluded group (automatic or review) is replaced by the next unused reserve
    original of the same label (ROB-04: the next unused reserve item); a reserve is used once it is
    tried, whatever the outcome. No reserve left -> the slot stays empty (shortfall in the meta).
    Unused reserves are listed with status reserve for every condition (ROB-01..03) and lose their
    other-condition rows once used.
  - Status: assigned | reserve | auto_excluded | review_excluded (every case of a group replaced after
    a review exclude; added to the spec's three values so that decided siblings are never evaluated).
    Evaluation uses status assigned with decision keep or relabel (label = new_label).
  - Review (review.csv: case_id, text_sha256_12, decision, new_label, reason, reviewer).
    text_sha256_12 = first 12 hex of SHA-256(case text, UTF-8), written by the script; a decided row
    whose value differs from the current case text is a stale decision (R09); pending rows get the
    current value. decision pending | keep | relabel | exclude; new_label 0/1 only with relabel and
    different from the case label; ROB-04 is benign-only (README 5.2), so relabel is refused there
    (use exclude); reason required for relabel and exclude; reviewer required for a decision, empty
    when pending, one of REVIEWERS (case-insensitive): HANJUSON (README DATA-10) or the separate
    Claude Code review session HANJUSON ordered on 2026-11-16 (llm:<model>:<prompt>, prompt in
    prompts/review_v1.md); a bare "claude" is refused. BOM, CRLF and fully blank rows are accepted
    on input, the file is rewritten canonically.
    Rows: every assigned case (new ones pending) plus every decided row of a case that still exists;
    pending rows of cases that are no longer assigned are dropped; a decided row of a case that
    would disappear, that is reserve or auto_excluded, or (build, --apply-review) whose text, label or
    status changed against the cases.csv on disk (except assigned -> review_excluded) stops the run.
  - Review exclusions are applied by --apply-review only: each group that is assigned and has a case
    with decision exclude becomes one event (sorted by condition, label 1 then 0, slot), appended to
    cases_meta.json "review_exclusions" with round = last round + 1. That log is the replacement
    history: every build replays it in order (each event must still exclude an assigned group and
    give the same replacement), so earlier replacements never move. A default build or --check with
    unapplied exclusions stops (R10). The history is never re-derived (build, --apply-review): after
    the replay, every review_excluded case of the cases.csv on disk must be review_excluded again,
    and when that file has any, every case it lists as assigned or review_excluded must have the same
    status again; otherwise cases_meta.json is missing or older than cases.csv and the run stops (R11).
  - Train patterns (README 5.2 last paragraph): obfuscation patterns counted in the original train rows
    (TRAIN_PATTERNS) and a map ROB-01 subtype -> patterns -> train rows by label, "present" (in train)
    or "new" (0 rows).
  - Outputs (UTF-8, LF, csv module quoting, sorted by case_id; originals.csv by label 1, 0 then
    order; no timestamps or absolute paths): originals.csv, cases.csv, review.csv, cases_meta.json
    (sorted keys, ASCII). Write: every file to <name>.tmp, then os.replace in the order originals,
    cases, review, meta. --check rebuilds in memory and compares the four files byte for byte.

Stop codes (printed as `FAIL <code> ...`, IDs and counts only, exit 1, nothing written):
  C01 input missing or unreadable          C02 validation_clean.csv SHA-256 or join differs
  C03 too few validation_clean rows          C04 LLM file schema (keys, types, numbering, ids, labels)
  C05 tokenizer environment (HF_HUB_OFFLINE)  C06 --dir under datasets/
  C07 ROB-03 original not verbatim in its assembled text
  R01 review.csv not UTF-8 / csv error       R02 review header       R03 field count or line break
  R04 case_id malformed or duplicate         R05 decision value      R06 new_label rule
  R07 reason missing                          R08 reviewer rule
  R09 decided case would disappear, or its text/label/status changed, or cases.csv unreadable
  R10 unapplied review exclusions (run --apply-review)
  R11 review_exclusions log inconsistent (unreadable, reverted decision, other replacement, no review.csv)
  K01 --check: file missing or different     W01 write failure        X01 unexpected exception
"""
import argparse
import codecs
import csv
import io
import json
import os
import random
import re
import statistics
import sys
import traceback
from pathlib import Path

import build_lists
import data_lists
import model_specs

ROOT = data_lists.ROOT
SCRIPT_REL = "scripts/build_dev_cases.py"
DEFAULT_DIR = ROOT / "data" / "cases" / "dev"
VALIDATION_CLEAN_REL = "data/processed/validation_clean.csv"
LISTS_META_REL = "data/processed/lists_meta.json"
SCHEMA = 1

EXTRACTION_SEED = 20261116
LABEL_ORDER = (1, 0)
N_MAIN = 50
N_RESERVE = 15
N_ALLOC = 3 * N_MAIN + N_RESERVE  # 165 originals per label
ALLOC_CONDITIONS = ("rob01", "rob02", "rob03")
CONDITIONS = ("ROB-01", "ROB-02", "ROB-03", "ROB-04")
COND_OF = {"rob01": "ROB-01", "rob02": "ROB-02", "rob03": "ROB-03"}
SLOT_START = {"ROB-01": 0, "ROB-02": 50, "ROB-03": 100}
RESERVE_START = 150

ROB01_VERSION = "rob01-v1"
ROB03_VERSION = "rob03-v1"
ROB01_SUBTYPES = ("case", "space", "char", "homoglyph")
CASE_REDRAWS = 20
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
HOMOGLYPH = {"a": "\u0430", "c": "\u0441", "e": "\u0435", "o": "\u043e", "p": "\u0440", "x": "\u0445",
             "y": "\u0443", "i": "\u0456", "A": "\u0410", "B": "\u0412", "C": "\u0421", "E": "\u0415",
             "H": "\u041d", "K": "\u041a", "M": "\u041c", "O": "\u041e", "P": "\u0420", "T": "\u0422",
             "X": "\u0425"}
POSITIONS = ("prefix", "suffix", "both")
MAX_TOKENS = 256
TOKENIZER_KEYS = tuple(model_specs.MODELS)

LLM = {"rob04": "rob04_benign_v1", "rob03": "rob03_contexts_v1", "rob02": "rob02_paraphrase_v1"}
ROB04_ITEMS, ROB04_MAIN = 65, 50
ROB04_CATEGORIES = ("security_course", "academic_research", "journalism", "quotation_analysis", "policy_writing",
                    "defensive_engineering", "red_team_planning", "concepts_history")
ROB04_CHARS = (40, 200)
ROB03_SHORT, ROB03_LONG = 80, 40
ROB03_WORDS = {"short": (15, 35), "long": (50, 90)}
ROB03_KINDS = ("email", "meeting notes", "product review", "recipe step", "travel plan", "study notes",
               "customer-support ticket", "news-style paragraph", "code comment or changelog line", "diary entry")
# Words the ROB-03 prompt forbids (rule 2), scanned for the meta flags only (the reviewer decides).
ROB03_SCAN = re.compile(r"\b(?:ai|assistants?|chatbots?|prompts?|instructions?|rules?|polic(?:y|ies)|security|"
                        r"hack\w*|passwords?|jailbreaks?|ignor\w*|disregard\w*)\b|\?", re.IGNORECASE)

CASE_COLUMNS = ("case_id", "group_id", "source_split", "source_id", "purpose", "condition", "subtype",
                "eval_group", "label", "position", "context_ids", "generator", "generator_version", "status",
                "auto_flags", "text")
REVIEW_COLUMNS = ("case_id", "text_sha256_12", "decision", "new_label", "reason", "reviewer")
# README DATA-10: HANJUSON, or the separate Claude Code review session HANJUSON ordered (2026-11-16).
REVIEWERS = ("HANJUSON", "llm:claude-opus-5-5:review_v1")
ORIGINALS_COLUMNS = ("sentence_id", "label", "condition", "order")
DECISIONS = ("pending", "keep", "relabel", "exclude")
STATUSES = ("assigned", "reserve", "auto_excluded", "review_excluded")
FILES = ("originals.csv", "cases.csv", "review.csv", "cases_meta.json")
META = "cases_meta.json"
CASE_ID_RE = re.compile(r"dev-rob01-(?:case|space|char|homoglyph)-[0-9]{6}|dev-rob0[23]-[0-9]{6}|dev-rob04-[0-9]{3}")
VAL_SID_RE = re.compile(r"validation:[0-9]{6}")

# Obfuscation patterns counted in the original train rows (README 5.2 last paragraph).
TRAIN_PATTERNS = {
    "alternating_case": "an ASCII letter run of >= 4 letters whose case alternates at every letter",
    "spaced_letters": ">= 4 single ASCII letters separated by single spaces",
    "multiple_spaces": "two or more spaces between non-space characters",
    "leetspeak": "a token (maximal ASCII letter/digit run) of letters and the digits 0 1 3 4 5 7 with at least "
                 "one of each whose de-leeted form (0->o, 1->i or l, 3->e, 4->a, 5->s, 7->t) is a word of >= 3 "
                 "letters occurring as an all-letter token in the train or validation texts (base64-like "
                 "fragments, codes and numbers such as mp3 or 4k do not count)",
    "cyrillic": "a character in U+0400-U+04FF",
    "non_ascii": "a non-ASCII character (README 3.2: all such train rows are injections, a shortcut candidate)",
}
# ROB-01 subtype -> the train patterns it produces (README 5.2: obfuscation already in train vs new).
SUBTYPE_PATTERNS = {"case": ("alternating_case",), "space": ("spaced_letters", "multiple_spaces"),
                    "char": ("leetspeak",), "homoglyph": ("cyrillic",)}
DELEET = {"0": ("o",), "1": ("i", "l"), "3": ("e",), "4": ("a",), "5": ("s",), "7": ("t",)}
_LETTERS = re.compile(r"[A-Za-z]+")
_ALNUM = re.compile(r"[A-Za-z0-9]+")
_SPACED = re.compile(r"(?<![A-Za-z])[A-Za-z](?: [A-Za-z]){3,}(?![A-Za-z])")
_MULTI_SPACE = re.compile(r"\S {2,}\S")
_CYRILLIC = re.compile("[\u0400-\u04ff]")

# Rule text recorded in cases_meta.json "rules" (the docstring has the full rules).
RULES = {
    "rob01_case": "each ASCII letter upper with p=0.5 else lower; >= ceil(30%) of the letters change case; up "
                  "to 20 redraws (21 draws), then letters still in their original case are flipped in a seeded order",
    "rob01_space": "k = rng.choice((1, 2)) longest words of >= 4 ASCII letters (ties: earlier) spelled out with "
                   "single spaces; every inter-word whitespace run next to a token holding a spelled-out word "
                   "becomes 2-3 spaces (rng.choice, ascending), so spelled-out word boundaries stay visible; "
                   "then round(g / 3) of the g other runs (rng.sample) become 2-3 spaces",
    "rob01_char": "a e i o s t (either case) -> 4 3 1 0 5 7, p=0.5 each, one rng.choice position if none changed",
    "rob01_homoglyph": "Latin -> Cyrillic look-alikes (rule_versions.homoglyph_map), p=0.5 each, one rng.choice "
                       "position if none changed; held-out (never used for augmentation)",
    "rob03": "pair index = rob03 slot 0-49 / reserve 50-64, shared by the attack and benign original; positions "
             "cycle prefix, suffix, both over a seeded order, prefix/suffix alternate short/long, both = two "
             "short; first queue passage (same class, then short) with <= 256 tokens for both originals and all "
             "three tokenizers; else the first candidate and the non-fitting original is excluded",
    "checks": "exact text = any train/validation row -> exclude; normalised/punct key -> flag (ROB-04: exclude); "
              "ROB-02 same punct key or README 3.2 one-word edit of its original -> exclude; duplicate dev text "
              "-> the later case excluded; one excluded case excludes its group",
    "replacement": "next unused reserve original of the same label (ROB-04: next reserve item), used once tried; "
                   "review exclusions only via --apply-review, recorded in review_exclusions and replayed",
}


class Refused(Exception):
    """A stop: .errors holds the FAIL messages (IDs and counts only)."""

    def __init__(self, errors):
        super().__init__(f"{len(errors)} error(s)")
        self.errors = list(errors)


def _ascii(s):
    return str(s).encode("ascii", "backslashreplace").decode("ascii")


def say(msg):
    print(_ascii(msg), flush=True)


# ---------------------------------------------------------------- inputs

def load_sources():
    """{"vclean": [(sid, label, text)], "train"/"validation": {sid: (label str, text)}, "inputs": {...}}."""
    try:
        rows = data_lists.load_rows()
    except data_lists.ListError as exc:
        raise Refused([f"C01 raw splits: {e}" for e in exc.errors])
    errors = []
    try:
        vc_bytes = (ROOT / VALIDATION_CLEAN_REL).read_bytes()
        lists_meta_bytes = (ROOT / LISTS_META_REL).read_bytes()
        want = json.loads(lists_meta_bytes)["outputs"]["validation_clean.csv"]["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Refused([f"C01 {VALIDATION_CLEAN_REL} or {LISTS_META_REL}: unreadable ({build_lists.exc_name(exc)})"])
    got = data_lists.sha256_bytes(vc_bytes)
    if got != want:
        errors.append(f"C02 {VALIDATION_CLEAN_REL}: sha256 {got[:12]} != lists_meta.json {want[:12]}")
    try:
        joined, jerr = data_lists.join(data_lists.read_list(ROOT / VALIDATION_CLEAN_REL), rows["validation"],
                                       "validation_clean.csv")
    except data_lists.ListError as exc:
        joined, jerr = [], exc.errors
    errors += [f"C02 {e}" for e in jerr]
    if errors:
        raise Refused(errors)
    info, man, _ = build_lists.load_manifest(build_lists.DEFAULT_MANIFEST)
    inputs = {
        "validation_clean.csv": {"path": VALIDATION_CLEAN_REL, "sha256": got, "rows": len(joined)},
        "lists_meta.json": {"path": LISTS_META_REL, "sha256": data_lists.sha256_bytes(lists_meta_bytes)},
        "download_manifest.json": {"path": "datasets/s-labs-prompt-injection/download_manifest.json",
                                   "sha256": man["sha256"], "revision": man["revision"]},
    }
    for split in ("train", "validation"):
        inputs[f"{split}.csv"] = {"via": "download_manifest.json", "sha256": info[split]["sha256"],
                                  "size": info[split]["size"], "rows": len(rows[split])}
    return {"vclean": joined, "train": rows["train"], "validation": rows["validation"], "inputs": inputs}


def make_token_len():
    """(token_len(text) -> token counts with special tokens per TOKENIZER_KEYS, tokenizer fingerprints)."""
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise Refused(["C05 tokenizers load offline only: run with HF_HUB_OFFLINE=1"])
    toks = {k: model_specs.load_tokenizer(k) for k in TOKENIZER_KEYS}
    cache = {}

    def token_len(text):
        if text not in cache:
            cache[text] = tuple(len(toks[k](text, add_special_tokens=True, truncation=False)["input_ids"])
                                for k in TOKENIZER_KEYS)
        return cache[text]

    return token_len, {k: model_specs.tokenizer_fingerprint(t) for k, t in toks.items()}


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _nonempty_str(v):
    return isinstance(v, str) and v.strip() != ""


def load_llm(d, expected_rob02):
    """{"rob04"/"rob03"/"rob02": {"generator", "items", "file", "sha256_lf", "prompt", "prompt_sha256_lf"}}.

    expected_rob02 = {sid: label} of the rob02 and reserve originals; items are keyed by n (rob04,
    rob03) or sentence ID (rob02).
    """
    out, errors = {}, []
    for key, name in LLM.items():
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
            if not (isinstance(gen, dict) and _nonempty_str(gen.get("model")) and isinstance(items, list)):
                raise TypeError
        except (ValueError, KeyError, TypeError):
            errors.append(f"C04 {f_rel}: not JSON with a generator object (non-empty model) and an items list")
            continue
        if gen.get("prompt_sha256") != prompt_sha:
            errors.append(f"C04 {f_rel}: generator.prompt_sha256 differs from the LF SHA-256 of {p_rel} "
                          f"({prompt_sha[:12]}): the prompt changed after generation, or the record is missing")
        parsed, errs = _parse_items(key, items, expected_rob02)
        errors += [f"C04 {f_rel}: {e}" for e in errs]
        out[key] = {"generator": gen, "items": parsed, "file": f_rel,
                    "sha256_lf": data_lists.sha256_bytes(raw.replace(b"\r\n", b"\n")),
                    "prompt": p_rel, "prompt_sha256_lf": prompt_sha}
    if errors:
        raise Refused(errors)
    return out


def _parse_items(key, items, expected_rob02):
    errors, parsed = [], {}
    if key == "rob02":
        for k, it in enumerate(items):
            if not (isinstance(it, dict) and isinstance(it.get("id"), str) and VAL_SID_RE.fullmatch(it["id"])
                    and it.get("label") in (0, 1) and _is_int(it.get("label")) and _nonempty_str(it.get("paraphrase"))):
                errors.append(f"item {k}: needs id validation:NNNNNN, label 0/1 and a non-empty paraphrase")
            elif it["id"] in parsed:
                errors.append(f"item {k}: duplicate id {it['id']}")
            else:
                parsed[it["id"]] = it
        missing, extra = sorted(set(expected_rob02) - set(parsed)), sorted(set(parsed) - set(expected_rob02))
        if missing:
            errors.append(f"no paraphrase for {build_lists.fmt_ids(missing)} (rob02 and reserve originals)")
        if extra:
            errors.append(f"ids not among the rob02 and reserve originals: {build_lists.fmt_ids(extra)}")
        wrong = [s for s in parsed if s in expected_rob02 and parsed[s]["label"] != expected_rob02[s]]
        if wrong:
            errors.append(f"label differs from the original in {build_lists.fmt_ids(wrong)}")
        return parsed, errors
    total = ROB04_ITEMS if key == "rob04" else ROB03_SHORT + ROB03_LONG
    for k, it in enumerate(items):
        n = it.get("n") if isinstance(it, dict) else None
        if not (_is_int(n) and 1 <= n <= total and _nonempty_str(it.get("text"))):
            errors.append(f"item {k}: needs n in 1..{total} and a non-empty text")
        elif key == "rob04" and not isinstance(it.get("category"), str):
            errors.append(f"item n={n}: category is not a string")
        elif key == "rob03" and not (isinstance(it.get("kind"), str)
                                     and it.get("length") == ("short" if n <= ROB03_SHORT else "long")):
            errors.append(f"item n={n}: kind not a string or length not {'short' if n <= ROB03_SHORT else 'long'}")
        elif n in parsed:
            errors.append(f"item {k}: duplicate n={n}")
        else:
            parsed[n] = it
    missing = sorted(set(range(1, total + 1)) - set(parsed))
    if missing and not errors:
        errors.append(f"missing n {', '.join(map(str, missing[:10]))}")
    return parsed, errors


# ---------------------------------------------------------------- allocation

def alloc_condition(pos):
    return ALLOC_CONDITIONS[pos // N_MAIN] if pos < RESERVE_START else "reserve"


def allocate(vclean):
    """[(sid, label, condition, order)] by label 1, 0 then order (README 5.2, seeded per label)."""
    out, errors = [], []
    for label in LABEL_ORDER:
        ids = sorted(sid for sid, lab, _ in vclean if lab == label)
        if len(ids) < N_ALLOC:
            errors.append(f"C03 validation_clean label {label}: {len(ids)} rows < {N_ALLOC}")
            continue
        random.Random(f"s3-dev:{EXTRACTION_SEED}:label{label}").shuffle(ids)
        out += [(sid, label, alloc_condition(pos), pos) for pos, sid in enumerate(ids[:N_ALLOC])]
    if errors:
        raise Refused(errors)
    return out


def alloc_counts(originals):
    return {lab: {c: sum(1 for _, l_, cc, _ in originals if l_ == lab and cc == c)
                  for c in ALLOC_CONDITIONS + ("reserve",)} for lab in LABEL_ORDER}


# ---------------------------------------------------------------- ROB-01 rules v1

def rob01_case(text, rng):
    idx = [i for i, c in enumerate(text) if c.isascii() and c.isalpha()]
    if not idx:
        return None
    need = -(-3 * len(idx) // 10)  # ceil(30% of the letters)
    for _ in range(1 + CASE_REDRAWS):
        up = [rng.random() < 0.5 for _ in idx]
        diff = sum(text[i].isupper() != u for i, u in zip(idx, up))
        if diff >= need:
            break
    else:
        same = [j for j, (i, u) in enumerate(zip(idx, up)) if text[i].isupper() == u]
        rng.shuffle(same)
        for j in same[:need - diff]:
            up[j] = not up[j]
    chars = list(text)
    for i, u in zip(idx, up):
        chars[i] = chars[i].upper() if u else chars[i].lower()
    return "".join(chars)


def rob01_space(text, rng):
    words = [m.span() for m in _LETTERS.finditer(text)]
    eligible = [w for w in words if w[1] - w[0] >= 4]
    if not eligible:
        return None
    k = min(len(eligible), rng.choice((1, 2)))
    split = set(sorted(eligible, key=lambda w: (w[0] - w[1], w[0]))[:k])
    tokens = [m.span() for m in re.finditer(r"\S+", text)]
    has_split = [any(a <= s and e <= b for s, e in split) for a, b in tokens]
    gaps = [(a[1], b[0]) for a, b in zip(tokens, tokens[1:])]  # inter-word whitespace runs
    touch = [g for j, g in enumerate(gaps) if has_split[j] or has_split[j + 1]]
    other = [g for j, g in enumerate(gaps) if not (has_split[j] or has_split[j + 1])]
    width = {g: rng.choice((2, 3)) for g in touch}
    widen = sorted(rng.sample(range(len(other)), round(len(other) / 3)))
    width.update({other[g]: rng.choice((2, 3)) for g in widen})
    out, i = [], 0
    for start, end in sorted(split | set(width)):
        out.append(text[i:start])
        out.append(" ".join(text[start:end]) if (start, end) in split else " " * width[(start, end)])
        i = end
    out.append(text[i:])
    return "".join(out)


def _substitute(text, rng, table):
    idx = [i for i, c in enumerate(text) if c in table]
    if not idx:
        return None
    pick = [i for i in idx if rng.random() < 0.5] or [rng.choice(idx)]
    chars = list(text)
    for i in pick:
        chars[i] = table[chars[i]]
    return "".join(chars)


LEET_TABLE = {**LEET, **{k.upper(): v for k, v in LEET.items()}}
ROB01 = {"case": rob01_case, "space": rob01_space,
         "char": lambda t, rng: _substitute(t, rng, LEET_TABLE),
         "homoglyph": lambda t, rng: _substitute(t, rng, HOMOGLYPH)}


def rob01_variant(subtype, case_id, text):
    return ROB01[subtype](text, random.Random(f"{ROB01_VERSION}:{case_id}"))


# ---------------------------------------------------------------- ROB-03 assembly v1

def assemble(position, ctx_texts, orig):
    if position == "prefix":
        return f"{ctx_texts[0]}\n\n{orig}"
    if position == "suffix":
        return f"{orig}\n\n{ctx_texts[0]}"
    return f"{ctx_texts[0]}\n\n{orig}\n\n{ctx_texts[1]}"


def plan_rob03(origs, pool, token_len):
    """Context assignment per pair index (see Rules).

    origs = {label: [original text of pair index i]}; pool = {n: (length class, text)}.
    Returns [{"position", "classes", "ctx": (n, ...) or (), "texts": {label: text}, "fail": {label: reason}}].
    """
    rng = random.Random(ROB03_VERSION)
    n_pairs = len(next(iter(origs.values())))
    order = list(range(n_pairs))
    rng.shuffle(order)
    position, classes, count = [None] * n_pairs, [None] * n_pairs, {"prefix": 0, "suffix": 0}
    for j, i in enumerate(order):
        pos = POSITIONS[j % 3]
        position[i] = pos
        if pos == "both":
            classes[i] = ("short", "short")
        else:
            classes[i] = ("short" if count[pos] % 2 == 0 else "long",)
            count[pos] += 1
    queues = {c: sorted(n for n, (cls, _) in pool.items() if cls == c) for c in ("short", "long")}
    rng.shuffle(queues["short"])
    rng.shuffle(queues["long"])

    def fits(text):
        return max(token_len(text)) <= MAX_TOKENS

    plan = []
    for i in range(n_pairs):
        pos = position[i]
        if pos == "both":
            q = queues["short"]
            cands = ((a, b) for x, a in enumerate(q) for b in q[x + 1:])
        else:
            order_q = (queues["long"] + queues["short"]) if classes[i] == ("long",) else queues["short"]
            cands = ((n,) for n in list(order_q))

        def texts(ctx):
            return {lab: assemble(pos, [pool[n][1] for n in ctx], o[i]) for lab, o in origs.items()}

        first, chosen = None, None
        for c in cands:
            first = first or c
            if all(fits(t) for t in texts(c).values()):
                chosen = c
                break
        ctx = chosen or first or ()
        for n in ctx:
            queues[pool[n][0]].remove(n)
        t = texts(ctx) if ctx else {lab: "" for lab in origs}
        fail = {lab: (None if ctx and fits(t[lab]) else
                      ("length_over_256" if ctx else "no_context_passage")) for lab in origs}
        bad = [lab for lab, o in origs.items() if ctx and o[i] not in t[lab]]
        if bad:
            raise Refused([f"C07 ROB-03 pair {i}: the label-{bad[0]} original is not verbatim in its text"])
        plan.append({"position": pos, "classes": classes[i], "ctx": ctx, "texts": t, "fail": fail,
                     "ctx_classes": tuple(pool[n][0] for n in ctx)})
    return plan


# ---------------------------------------------------------------- automatic checks

def key_index(src):
    """{"exact"/"nk"/"pk": {key: sorted IDs}} over every train and validation row."""
    idx = {"exact": {}, "nk": {}, "pk": {}}
    for split in ("train", "validation"):
        for sid, (_, text) in src[split].items():
            p = build_lists.pk(text)
            idx["exact"].setdefault(text, []).append(sid)
            if p:
                idx["nk"].setdefault(build_lists.nk(text), []).append(sid)
                idx["pk"].setdefault(p, []).append(sid)
    for d in idx.values():
        for v in d.values():
            v.sort()
    return idx


def one_word_edit(a, b):
    """README 3.2 level C on punct-key word lists: one word inserted, deleted or substituted, shorter >= 5 words."""
    if a == b or min(len(a), len(b)) < build_lists.MIN_EDIT_WORDS or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def key_checks(text, cond, own, idx, own_text=None):
    """(exclusion reasons, flags) of one case text (see Rules)."""
    reasons, flags = [], []
    exact = idx["exact"].get(text, [])
    if exact:
        reasons.append(f"exact={exact[0]}")
    p = build_lists.pk(text)
    if p:
        nk_all = idx["nk"].get(build_lists.nk(text), [])
        pk_all = idx["pk"].get(p, [])
        nk_ids = [s for s in nk_all if s not in exact and s != own]
        pk_ids = [s for s in pk_all if s not in nk_all and s != own]
        for level, ids in (("normalised_key", nk_ids), ("punct_key", pk_ids)):
            if ids:
                (reasons if cond == "ROB-04" else flags).append(f"{level}={ids[0]}")
    if cond == "ROB-02" and not exact:
        if p and p == build_lists.pk(own_text):
            reasons.append("same_punct_key_as_original")
        elif one_word_edit(p.split(), build_lists.pk(own_text).split()):
            reasons.append("one_word_edit_of_original")
    if cond in ("ROB-02", "ROB-04") and ("\n" in text or "\r" in text):
        flags.append("line_break")
    if cond == "ROB-04" and not ROB04_CHARS[0] <= len(text) <= ROB04_CHARS[1]:
        flags.append("rob04_length")
    return reasons, flags


def deleet_forms(token):
    """Every de-leeted, casefolded form of a token (1 -> i or l), at most 2 ** 6 forms."""
    forms = [""]
    for ch in token.casefold():
        forms = [f + x for f in forms for x in DELEET.get(ch, (ch,))][:64]
    return forms


def is_leet_token(token, vocab):
    digits = [c for c in token if c.isdigit()]
    return (len(token) >= 3 and any(c.isalpha() for c in token) and bool(digits)
            and all(c in DELEET for c in digits) and any(f in vocab for f in deleet_forms(token)))


def train_patterns(train, validation):
    """({pattern: {"rows", "label_0", "label_1"}} over the original train rows, subtype map) (README 5.2)."""
    vocab = {t.casefold() for rows in (train, validation) for _, text in rows.values()
             for t in _ALNUM.findall(text) if len(t) >= 3 and t.isalpha()}

    def found(text):
        words = _LETTERS.findall(text)
        return {
            "alternating_case": any(len(w) >= 4 and all(a.isupper() != b.isupper() for a, b in zip(w, w[1:]))
                                    for w in words),
            "spaced_letters": bool(_SPACED.search(text)),
            "multiple_spaces": bool(_MULTI_SPACE.search(text)),
            "leetspeak": any(is_leet_token(t, vocab) for t in _ALNUM.findall(text)),
            "cyrillic": bool(_CYRILLIC.search(text)),
            "non_ascii": not text.isascii(),
        }
    out = {p: {"rows": 0, "label_0": 0, "label_1": 0} for p in TRAIN_PATTERNS}
    for label, text in train.values():
        for p, hit in found(text).items():
            if hit:
                out[p]["rows"] += 1
                out[p][f"label_{label}"] += 1
    by_subtype = {st: {"patterns": {p: out[p] for p in pats},
                       "train": "present" if any(out[p]["rows"] for p in pats) else "new"}
                  for st, pats in SUBTYPE_PATTERNS.items()}
    return out, by_subtype


# ---------------------------------------------------------------- review sheet

def parse_review(data):
    """({case_id: row}, errors R01-R08); BOM and CRLF accepted, fields stripped."""
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
            continue  # fully blank row (e.g. a spreadsheet's trailing ",,,,,")
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
            errors.append(f"R04 {tag}: case_id is not a dev case ID")
            continue
        if cid in rows:
            errors.append(f"R04 {tag}: duplicate case_id {cid}")
            continue
        if dec not in DECISIONS:
            errors.append(f"R05 {cid}: decision not one of {'|'.join(DECISIONS)}")
        elif dec == "relabel" and cid.startswith("dev-rob04-"):
            errors.append(f"R06 {cid}: ROB-04 is benign-only (README 5.2): use exclude, not relabel")
        elif dec == "relabel" and r["new_label"] not in ("0", "1"):
            errors.append(f"R06 {cid}: relabel needs new_label 0 or 1")
        elif dec != "relabel" and r["new_label"]:
            errors.append(f"R06 {cid}: new_label only with relabel")
        if dec in ("relabel", "exclude") and not r["reason"]:
            errors.append(f"R07 {cid}: {dec} needs a reason")
        if dec == "pending" and r["reviewer"]:
            errors.append(f"R08 {cid}: a pending row has no reviewer")
        elif dec in DECISIONS and dec != "pending" and r["reviewer"].lower() not in reviewers:
            errors.append(f"R08 {cid}: reviewer must be one of {', '.join(REVIEWERS)} (README DATA-10)")
        if dec in DECISIONS and dec != "pending" and not build_lists.H12_RE.fullmatch(r["text_sha256_12"]):
            errors.append(f"R09 {cid}: a decided row needs the text_sha256_12 written by the script")
        rows[cid] = r
    return rows, errors


def read_old_cases(path):
    """{case_id: (label, status, text)} from the cases.csv on disk, or None when it does not exist."""
    if not path.exists():
        return None
    try:
        recs = list(csv.DictReader(io.StringIO(path.read_bytes().decode("utf-8"), newline="")))
        return {r["case_id"]: (r["label"], r["status"], r["text"]) for r in recs}
    except (OSError, UnicodeDecodeError, csv.Error, KeyError) as exc:
        raise Refused([f"R09 cases.csv on disk unreadable ({build_lists.exc_name(exc)}): cannot protect "
                       "decided cases; restore it from git"])


def read_log(path):
    """review_exclusions of the cases_meta.json on disk ([] when it does not exist)."""
    if not path.exists():
        return []
    try:
        log = json.loads(path.read_bytes())["review_exclusions"]
        ok = isinstance(log, list) and all(
            isinstance(e, dict) and _is_int(e.get("round")) and e["round"] >= 1 and e.get("condition") in CONDITIONS
            and isinstance(e.get("group_id"), str) for e in log)
    except (OSError, ValueError, KeyError, TypeError):
        ok = False
    if not ok:
        raise Refused([f"R11 {META} on disk: review_exclusions unreadable (the replacement history); "
                       "restore it from git"])
    return log


# ---------------------------------------------------------------- build

class Builder:
    """One deterministic build: round 0 (automatic checks), then the review_exclusions log, then
    (--apply-review) the new review exclusions."""

    def __init__(self, src, llm, token_len):
        self.src, self.llm, self.token_len = src, llm, token_len
        self.idx = key_index(src)
        self.vtext = {sid: text for sid, _, text in src["vclean"]}
        self.originals = allocate(src["vclean"])
        self.by_label = {lab: [s for s, l_, _, _ in self.originals if l_ == lab] for lab in LABEL_ORDER}
        self.label_of = {s: lab for s, lab, _, _ in self.originals}
        self.reserves = {lab: ids[RESERVE_START:] for lab, ids in self.by_label.items()}
        self.next_reserve = {lab: 0 for lab in LABEL_ORDER}
        self.next_rob04 = ROB04_MAIN + 1
        self.consumed = {}  # reserve original -> condition
        self.cases = {}
        self.group_cases = {}
        self.slots = {}
        self.slot_of = {}
        self.seen = {}
        self.tried = []
        model = {k: v["generator"]["model"] for k, v in llm.items()}
        self.version = {"ROB-01": ROB01_VERSION, "ROB-02": f"llm:{model['rob02']}:{LLM['rob02']}",
                        "ROB-03": f"{ROB03_VERSION}+llm:{model['rob03']}:{LLM['rob03']}",
                        "ROB-04": f"llm:{model['rob04']}:{LLM['rob04']}"}
        self.generator = {"ROB-01": "rule", "ROB-02": "llm", "ROB-03": "template+llm", "ROB-04": "llm"}
        pool = {n: (it["length"], it["text"]) for n, it in llm["rob03"]["items"].items()}
        self.pair_of = {}
        origs = {}
        for lab in LABEL_ORDER:
            ids = self.by_label[lab][SLOT_START["ROB-03"]:SLOT_START["ROB-03"] + N_MAIN] + self.reserves[lab]
            origs[lab] = [self.vtext[s] for s in ids]
            self.pair_of.update({s: i for i, s in enumerate(ids)})
        self.plan = plan_rob03(origs, pool, token_len)

    # -- candidates

    def _record(self, cid, group, cond, subtype, label, text, **kw):
        return {"case_id": cid, "group_id": group, "source_split": "new" if cond == "ROB-04" else "validation",
                "source_id": kw.get("source_id", group), "purpose": "dev", "condition": cond, "subtype": subtype,
                "eval_group": "held_out" if subtype == "homoglyph" else "main", "label": str(label),
                "position": kw.get("position", ""), "context_ids": kw.get("context_ids", ""),
                "generator": self.generator[cond], "generator_version": self.version[cond], "status": "",
                "auto_flags": "", "text": text, "_reasons": list(kw.get("reasons", ())), "_flags": []}

    def candidate(self, cond, group):
        """Case records of one group with their intrinsic exclusion reasons and flags."""
        if cond == "ROB-04":
            n = int(group[-3:])
            recs = [self._record(group, group, cond, "benign_security", 0, self.llm["rob04"]["items"][n]["text"],
                                 source_id=f"{LLM['rob04']}:{n:03d}")]
        else:
            num, label, orig = group.split(":")[1], self.label_of[group], self.vtext[group]
            if cond == "ROB-01":
                recs = []
                for st in ROB01_SUBTYPES:
                    cid = f"dev-rob01-{st}-{num}"
                    out = rob01_variant(st, cid, orig)
                    recs.append(self._record(cid, group, cond, st, label, out or "",
                                             reasons=() if out is not None else ("no_eligible_char",)))
            elif cond == "ROB-02":
                recs = [self._record(f"dev-rob02-{num}", group, cond, "paraphrase", label,
                                     self.llm["rob02"]["items"][group]["paraphrase"])]
            else:
                p = self.plan[self.pair_of[group]]
                fail = p["fail"][label]
                recs = [self._record(f"dev-rob03-{num}", group, cond, "context", label, p["texts"][label],
                                     position=p["position"], context_ids=";".join(map(str, p["ctx"])),
                                     reasons=(fail,) if fail else ())]
        for r in recs:
            if r["text"]:
                own = None if cond == "ROB-04" else group
                reasons, flags = key_checks(r["text"], cond, own, self.idx, own_text=self.vtext.get(group))
                r["_reasons"] += reasons
                r["_flags"] += flags
        return recs

    # -- placement and replacement

    def take_reserve(self, cond, label):
        if cond == "ROB-04":
            if self.next_rob04 > ROB04_ITEMS:
                return None
            self.next_rob04 += 1
            return f"dev-rob04-{self.next_rob04 - 1:03d}"
        k = self.next_reserve[label]
        if k >= len(self.reserves[label]):
            return None
        self.next_reserve[label] = k + 1
        self.consumed[self.reserves[label][k]] = cond
        return self.reserves[label][k]

    def place(self, cond, label, slot, group):
        """Put group into the slot; an excluded group is replaced by the next reserve until one passes."""
        while group is not None:
            recs = self.candidate(cond, group)
            local = {}
            for r in recs:
                if r["text"] and not r["_reasons"]:
                    dup = self.seen.get(r["text"]) or local.get(r["text"])
                    if dup:
                        r["_reasons"].append(f"duplicate_of={dup}")
                    local.setdefault(r["text"], r["case_id"])
            bad = [r for r in recs if r["_reasons"]]
            for r in recs:
                if bad and not r["_reasons"]:
                    r["_reasons"].append(f"group={bad[0]['case_id']}")
                r["status"] = "auto_excluded" if bad else "assigned"
                r["auto_flags"] = ";".join([f"exclude:{x}" for x in r["_reasons"]] + [f"flag:{x}" for x in r["_flags"]])
                self.cases[r["case_id"]] = r
            self.group_cases[(cond, group)] = [r["case_id"] for r in recs]
            self.tried.append({"condition": cond, "label": label, "slot": slot, "group_id": group,
                               "outcome": "auto_excluded" if bad else "assigned"})
            if not bad:
                self.seen.update(local)
                self.slots[(cond, label)][slot] = group
                self.slot_of[(cond, group)] = (label, slot)
                return
            group = self.take_reserve(cond, label)
        self.slots[(cond, label)][slot] = None

    def round0(self):
        for cond in CONDITIONS[:3]:
            for label in LABEL_ORDER:
                self.slots[(cond, label)] = [None] * N_MAIN
                start = SLOT_START[cond]
                for slot, group in enumerate(self.by_label[label][start:start + N_MAIN]):
                    self.place(cond, label, slot, group)
        self.slots[("ROB-04", 0)] = [None] * N_MAIN
        for slot in range(N_MAIN):
            self.place("ROB-04", 0, slot, f"dev-rob04-{slot + 1:03d}")

    def is_assigned(self, cond, group):
        return (cond, group) in self.slot_of

    def exclude_group(self, cond, group):
        label, slot = self.slot_of.pop((cond, group))
        for cid in self.group_cases[(cond, group)]:
            self.cases[cid]["status"] = "review_excluded"
        self.place(cond, label, slot, self.take_reserve(cond, label))
        return label, slot, self.slots[(cond, label)][slot]

    def add_reserves(self):
        for lab in LABEL_ORDER:
            for group in self.reserves[lab]:
                if group in self.consumed:
                    continue
                for cond in CONDITIONS[:3]:
                    for r in self.candidate(cond, group):
                        r["status"], r["auto_flags"] = "reserve", ""
                        self.cases[r["case_id"]] = r
        for n in range(self.next_rob04, ROB04_ITEMS + 1):
            r = self.candidate("ROB-04", f"dev-rob04-{n:03d}")[0]
            r["status"], r["auto_flags"] = "reserve", ""
            self.cases[r["case_id"]] = r


def excluded_groups(b, review):
    """{(cond, group): [excluded case IDs]} for assigned groups with a decision exclude."""
    out = {}
    for cid, r in review.items():
        c = b.cases.get(cid)
        if r["decision"] == "exclude" and c is not None and b.is_assigned(c["condition"], c["group_id"]):
            out.setdefault((c["condition"], c["group_id"]), []).append(cid)
    return out


def apply_log(b, review, log, apply_new, old_cases=None):
    """Replay the review_exclusions log, then (apply_new) the new exclusions. Returns the new log.

    old_cases (build, --apply-review: the cases.csv on disk) must be reproduced by the replay (see Rules),
    so a missing or older cases_meta.json stops the run instead of re-deriving the history.
    """
    errors, new_log = [], []
    for ev in log:
        cond, group = ev["condition"], ev["group_id"]
        cids = [c for c in b.group_cases.get((cond, group), []) if review.get(c, {}).get("decision") == "exclude"]
        if not b.is_assigned(cond, group):
            errors.append(f"R11 review_exclusions round {ev['round']}: {cond} {group} is not assigned at replay "
                          "(inputs changed after the review)")
            continue
        if not cids:
            errors.append(f"R11 review_exclusions round {ev['round']}: no case of {cond} {group} has decision "
                          "exclude any more (a decision was changed after its replacement)")
            continue
        label, slot, repl = b.exclude_group(cond, group)
        if "replacement_group_id" in ev and ev["replacement_group_id"] != repl:
            errors.append(f"R11 review_exclusions round {ev['round']}: {cond} {group} now gets {repl}, "
                          f"logged {ev['replacement_group_id']}")
        new_log.append({"round": ev["round"], "condition": cond, "label": label, "slot": slot, "group_id": group,
                        "excluded_case_ids": sorted(cids), "replacement_group_id": repl})
    if errors:
        raise Refused(errors)
    if old_cases:
        had_review = any(st == "review_excluded" for _, st, _ in old_cases.values())
        bad = [cid for cid, (_, st, _) in old_cases.items()
               if (st == "review_excluded" or (had_review and st == "assigned"))
               and b.cases.get(cid, {}).get("status") != st]
        if bad:
            raise Refused([f"R11 cases.csv on disk: {build_lists.fmt_ids(bad)} carry a review status or replacement "
                           f"that the {META} review_exclusions log does not reproduce ({META} missing or older "
                           "than cases.csv): restore the matching file from git; the history is never re-derived"])
    pending = excluded_groups(b, review)
    if pending and not apply_new:
        raise Refused([f"R10 review.csv: {len(pending)} excluded group(s) not replaced yet "
                       f"({build_lists.fmt_ids([g for _, g in pending])}): run --apply-review"])
    rnd = max([e["round"] for e in log], default=0) + 1
    order = sorted(pending, key=lambda k: (CONDITIONS.index(k[0]), LABEL_ORDER.index(b.slot_of[k][0]),
                                           b.slot_of[k][1]))
    for cond, group in order:
        cids = pending[(cond, group)]
        label, slot, repl = b.exclude_group(cond, group)
        new_log.append({"round": rnd, "condition": cond, "label": label, "slot": slot, "group_id": group,
                        "excluded_case_ids": sorted(cids), "replacement_group_id": repl})
    return new_log


def merge_review(b, review, old_cases, protect):
    """Rows of the new review.csv; refuses a decided row that would disappear or whose case changed.

    protect (writes): decided cases must keep the text, label and status of old_cases (the cases.csv on
    disk; None = missing), except assigned -> review_excluded. --check compares bytes instead.
    """
    errors, rows, dropped = [], {}, 0
    for cid, r in review.items():
        c = b.cases.get(cid)
        decided = r["decision"] != "pending"
        if c is None:
            if decided:
                errors.append(f"R09 {cid}: decided ({r['decision']}) but the case would disappear")
            else:
                dropped += 1
            continue
        if r["decision"] == "relabel" and r["new_label"] == c["label"]:
            errors.append(f"R06 {cid}: new_label equals the case label")
        if decided and c["status"] not in ("assigned", "review_excluded"):
            errors.append(f"R09 {cid}: decided ({r['decision']}) but the case is {c['status']} (never reviewable)")
        elif decided and r["text_sha256_12"] != build_lists.h12(c["text"]):
            errors.append(f"R09 {cid}: stale decision ({r['decision']}): text_sha256_12 {r['text_sha256_12']} != "
                          f"the current case text {build_lists.h12(c['text'])}; review the case again")
        elif decided and protect and old_cases is None:
            errors.append(f"R09 {cid}: decided but cases.csv is missing on disk (cannot verify the case text)")
        elif decided and protect and cid in old_cases:
            o_label, o_status, o_text = old_cases[cid]
            same_status = o_status == c["status"] or (o_status, c["status"]) == ("assigned", "review_excluded")
            if o_text != c["text"] or o_label != c["label"] or not same_status:
                errors.append(f"R09 {cid}: decided ({r['decision']}) but its text, label or status changed "
                              f"against cases.csv on disk ({o_status} -> {c['status']})")
        if decided:
            rows[cid] = r
        elif c["status"] == "assigned":
            rows[cid] = {**r, "text_sha256_12": build_lists.h12(c["text"])}
        else:
            dropped += 1
    if errors:
        raise Refused(errors)
    for cid, c in b.cases.items():
        if c["status"] == "assigned" and cid not in rows:
            rows[cid] = {"case_id": cid, "text_sha256_12": build_lists.h12(c["text"]), "decision": "pending",
                         "new_label": "", "reason": "", "reviewer": ""}
    return [rows[k] for k in sorted(rows)], dropped


# ---------------------------------------------------------------- rendering

def render_csv(columns, rows):
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for r in rows:
        w.writerow([r[c] for c in columns])
    return buf.getvalue().encode("utf-8")


def render_originals(originals):
    rows = [{"sentence_id": s, "label": lab, "condition": cond, "order": o} for s, lab, cond, o in originals]
    return render_csv(ORIGINALS_COLUMNS, rows)


def render_json(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def _summary(values):
    return {"n": len(values), "max": max(values), "median": statistics.median(values),
            "over_256": sum(v > MAX_TOKENS for v in values)} if values else {"n": 0}


def counts_of(b, review_rows):
    cases = {}
    for c in b.cases.values():
        d = cases.setdefault(c["condition"], {}).setdefault(c["subtype"], {}).setdefault(f"label_{c['label']}", {})
        d[c["status"]] = d.get(c["status"], 0) + 1
    decisions = {}
    for r in review_rows:
        c = b.cases[r["case_id"]]
        if c["status"] != "assigned":
            continue
        d = decisions.setdefault(c["condition"], {}).setdefault(f"label_{c['label']}", {k: 0 for k in DECISIONS})
        d[r["decision"]] += 1
    groups, shortfall = {}, {}
    for (cond, label), slots in sorted(b.slots.items()):
        groups.setdefault(cond, {})[f"label_{label}"] = sum(g is not None for g in slots)
        shortfall.setdefault(cond, {})[f"label_{label}"] = sum(g is None for g in slots)
    return {"cases": cases, "review_decisions_assigned": decisions, "assigned_originals": groups,
            "shortfall": shortfall}


def token_summary(b):
    out = {}
    for cond in CONDITIONS:
        texts = [c["text"] for c in b.cases.values() if c["condition"] == cond and c["status"] == "assigned"]
        lens = [b.token_len(t) for t in texts]
        out[cond] = {k: _summary([x[j] for x in lens]) for j, k in enumerate(TOKENIZER_KEYS)}
    for pos in POSITIONS:
        texts = [c["text"] for c in b.cases.values()
                 if c["condition"] == "ROB-03" and c["status"] == "assigned" and c["position"] == pos]
        lens = [b.token_len(t) for t in texts]
        out[f"ROB-03 {pos}"] = {k: _summary([x[j] for x in lens]) for j, k in enumerate(TOKENIZER_KEYS)}
    return out


def llm_flags(llm):
    rob03, rob04 = {"word_count_out_of_range": [], "line_break": [], "forbidden_word_or_question": []}, {
        "category_not_round_robin": [], "length_out_of_range": [], "line_break": []}
    for n, it in sorted(llm["rob03"]["items"].items()):
        lo, hi = ROB03_WORDS[it["length"]]
        if not lo <= len(it["text"].split()) <= hi:
            rob03["word_count_out_of_range"].append(n)
        if "\n" in it["text"] or "\r" in it["text"]:
            rob03["line_break"].append(n)
        if ROB03_SCAN.search(it["text"]):
            rob03["forbidden_word_or_question"].append(n)
    for n, it in sorted(llm["rob04"]["items"].items()):
        if it["category"] != ROB04_CATEGORIES[(n - 1) % len(ROB04_CATEGORIES)]:
            rob04["category_not_round_robin"].append(n)
        if not ROB04_CHARS[0] <= len(it["text"]) <= ROB04_CHARS[1]:
            rob04["length_out_of_range"].append(n)
        if "\n" in it["text"] or "\r" in it["text"]:
            rob04["line_break"].append(n)
    return {"rob03_contexts": rob03, "rob04_items": rob04}


def build_meta(b, src, llm, fingerprints, script_sha, log, review_rows, files, n_rows):
    excl = [{"case_id": c["case_id"], "reasons": c["_reasons"]} for c in b.cases.values()
            if c["status"] == "auto_excluded"]
    flags = [{"case_id": c["case_id"], "flags": c["_flags"]} for c in b.cases.values()
             if c["_flags"] and c["status"] in ("assigned", "review_excluded")]
    used_ctx = sorted({int(n) for c in b.cases.values() if c["condition"] == "ROB-03" and c["context_ids"]
                       and c["status"] in ("assigned", "review_excluded") for n in c["context_ids"].split(";")})
    patterns, by_subtype = train_patterns(src["train"], src["validation"])
    return {
        "schema": SCHEMA,
        "purpose": "dev",
        "readme": "README 3.3 DATA-07..12, 5.2 ROB-01..04, 3.2 keys; spec .omc/autopilot/spec-stage3.md",
        "script": {"path": SCRIPT_REL, "sha256_lf": script_sha},
        "inputs": {**src["inputs"], **{f"scripts/{n}": {"path": f"scripts/{n}", "sha256_lf": data_lists.sha256_lf(
            Path(__file__).parent / n)} for n in ("build_lists.py", "data_lists.py", "model_specs.py")}},
        "rules": RULES,
        "seeds": {"extraction_seed": EXTRACTION_SEED,
                  "allocation": f"random.Random(f's3-dev:{EXTRACTION_SEED}:label{{label}}').shuffle(sorted IDs)",
                  "rob01": f"random.Random(f'{ROB01_VERSION}:{{case_id}}')",
                  "rob03": f"random.Random('{ROB03_VERSION}'): pair order, short queue, long queue"},
        "rule_versions": {"rob01": ROB01_VERSION, "rob03": ROB03_VERSION, "max_tokens": MAX_TOKENS,
                          "homoglyph_map": {k: f"U+{ord(v):04X}" for k, v in HOMOGLYPH.items()}, "leet_map": LEET,
                          "held_out": "ROB-01 homoglyph (never used for training augmentation)"},
        "generators": {LLM[k]: {"file": v["file"], "sha256_lf": v["sha256_lf"], "prompt": v["prompt"],
                                "prompt_sha256_lf": v["prompt_sha256_lf"], "generator": v["generator"],
                                "items": len(v["items"])} for k, v in llm.items()},
        "tokenizers": fingerprints,
        "allocation": {"per_label": {f"label_{lab}": n for lab, n in alloc_counts(b.originals).items()},
                       "reserve_used": {f"label_{lab}": [{"group_id": g, "condition": b.consumed[g]}
                                                         for g in b.reserves[lab] if g in b.consumed]
                                        for lab in LABEL_ORDER},
                       "rob04_reserve_used": [f"dev-rob04-{n:03d}" for n in range(ROB04_MAIN + 1, b.next_rob04)],
                       "tried": b.tried},
        "rob03_plan": [{"pair": i, "position": p["position"], "classes": list(p["classes"]),
                        "context_ids": list(p["ctx"]), "context_classes": list(p["ctx_classes"]),
                        "originals": {f"label_{lab}": (b.by_label[lab][SLOT_START["ROB-03"] + i] if i < N_MAIN
                                                       else b.reserves[lab][i - N_MAIN]) for lab in LABEL_ORDER}}
                       for i, p in enumerate(b.plan)],
        "rob03_contexts_used": used_ctx,
        "counts": counts_of(b, review_rows),
        "automatic_exclusions": excl,
        "flags": flags,
        "llm_item_flags": llm_flags(llm),
        "review_exclusions": log,
        "token_lengths_assigned": token_summary(b),
        "train_patterns": {"definitions": TRAIN_PATTERNS, "scope": "original train rows (README 5.2)",
                           "counts": patterns, "rob01_subtypes": by_subtype},
        "outputs": {n: {"sha256": data_lists.sha256_bytes(files[n]), "rows": n_rows[n]} for n in n_rows},
    }


def compute(src, d, mode, token_len=None, fingerprints=None):
    """All four files for mode build | apply | check: ({name: bytes}, builder, review rows, stats)."""
    originals = allocate(src["vclean"])
    expected = {s: lab for s, lab, cond, _ in originals if cond in ("rob02", "reserve")}
    llm = load_llm(d, expected)
    if token_len is None:
        token_len, fingerprints = make_token_len()
    review_path = d / "review.csv"
    review = {}
    if review_path.exists():
        review, errors = parse_review(review_path.read_bytes())
        if errors:
            raise Refused(errors)
    elif mode == "apply":
        raise Refused(["R11 --apply-review needs review.csv (run the default build first)"])
    log = read_log(d / META)
    if log and not review_path.exists():
        raise Refused([f"R11 {META} logs review exclusions but review.csv is missing: restore it from git"])
    protect = mode != "check"
    old_cases = read_old_cases(d / "cases.csv") if protect else None
    b = Builder(src, llm, token_len)
    b.round0()
    new_log = apply_log(b, review, log, apply_new=(mode == "apply"), old_cases=old_cases)
    b.add_reserves()
    rows, dropped = merge_review(b, review, old_cases, protect)
    cases = [{k: c[k] for k in CASE_COLUMNS} for _, c in sorted(b.cases.items())]
    files = {"originals.csv": render_originals(originals), "cases.csv": render_csv(CASE_COLUMNS, cases),
             "review.csv": render_csv(REVIEW_COLUMNS, rows)}
    n_rows = {"originals.csv": len(originals), "cases.csv": len(cases), "review.csv": len(rows)}
    script_sha = data_lists.sha256_lf(Path(__file__))
    files[META] = render_json(build_meta(b, src, llm, fingerprints, script_sha, new_log, rows, files, n_rows))
    last = max([e["round"] for e in log], default=0)
    return files, b, rows, {"new_events": [e for e in new_log if e["round"] > last], "dropped": dropped}


# ---------------------------------------------------------------- write / check / report

def write_files(d, files):
    """Each file to <name>.tmp, then os.replace in FILES order (meta last). Returns None or a W01 error."""
    tmps, current = [], "creating the folder"
    try:
        d.mkdir(parents=True, exist_ok=True)
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


def check_files(d, files):
    errors = []
    for name, data in files.items():
        try:
            old = (d / name).read_bytes()
        except OSError:
            errors.append(f"K01 --check: {name} is missing")
            continue
        if old != data:
            errors.append(f"K01 --check: {name} differs from the rebuilt bytes")
    return errors


def original_text(c, src):
    """The validation sentence a derived case comes from ('' for ROB-04)."""
    if c["source_split"] != "validation":
        return ""
    return src["validation"][c["source_id"]][1]


def render_view(b, rows, src):
    """Read-only reviewer aid (not an output, not an input): assigned cases with decision, text and original."""
    decided = {r["case_id"]: r for r in rows}
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["case_id", "condition", "subtype", "eval_group", "label", "position", "decision", "new_label",
                "reason", "text", "original_text"])
    for k, c in sorted(b.cases.items()):
        if c["status"] != "assigned":
            continue
        r = decided.get(k, {})
        w.writerow([k, c["condition"], c["subtype"], c["eval_group"], c["label"], c["position"],
                    r.get("decision", ""), r.get("new_label", ""), r.get("reason", ""),
                    c["text"], original_text(c, src)])
    return ("﻿" + buf.getvalue()).encode("utf-8")


def report(b, rows, print_n, src=None):
    by = {}
    for r in rows:
        c = b.cases[r["case_id"]]
        if c["status"] == "assigned":
            by.setdefault((c["condition"], c["label"]), {k: 0 for k in DECISIONS})[r["decision"]] += 1
    for (cond, label), d in sorted(by.items(), key=lambda kv: (kv[0][0], LABEL_ORDER.index(int(kv[0][1])))):
        groups = sum(g is not None for g in b.slots[(cond, int(label))])
        short = N_MAIN - groups
        say(f"{cond} label {label}: {groups} originals assigned{f' (shortfall {short})' if short else ''}; "
            f"cases pending {d['pending']}, keep {d['keep']}, relabel {d['relabel']}, exclude {d['exclude']}")
    if not print_n:
        return
    decision = {r["case_id"]: r["decision"] for r in rows}
    for cond in CONDITIONS:
        pend = [c for k, c in sorted(b.cases.items()) if c["condition"] == cond and c["status"] == "assigned"
                and decision.get(k) == "pending"][:print_n]
        say(f"-- {cond}: first {len(pend)} pending case(s) (terminal only)")
        for c in pend:
            text = c["text"].replace("\r", "\\r").replace("\n", "\\n")
            say(f"{c['case_id']} label {c['label']} {c['position'] or c['subtype']}: {text}")
            orig = original_text(c, src) if src is not None else ""
            if orig:
                say(f"    original {c['source_id']}: {orig}")


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Build the stage-3 development cases (README 3.3, 5.2).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--originals", action="store_true", help="write originals.csv only (allocation)")
    g.add_argument("--apply-review", action="store_true", help="apply review exclusions with replacements")
    g.add_argument("--check", action="store_true", help="rebuild in memory and compare with the files on disk")
    p.add_argument("--dir", help="cases folder (default data/cases/dev)")
    p.add_argument("--print", type=int, default=0, metavar="N", dest="print_n",
                   help="print up to N pending cases per condition with their originals (terminal only)")
    p.add_argument("--view", metavar="FILE",
                   help="also write a read-only reviewer view (UTF-8 with BOM) to FILE outside the outputs")
    return p.parse_args(argv)


def _run(args):
    d = Path(args.dir).resolve() if args.dir else DEFAULT_DIR
    if build_lists._under(d, build_lists.DATASETS):
        raise Refused(["C06 --dir is under datasets/ (raw data stays read-only)"])
    view = Path(args.view).resolve() if args.view else None
    if view and (view.parent == d or view.name in FILES or (build_lists._under(view, ROOT) and not _ignored(view))):
        # checked before anything is computed or written
        raise Refused([f"C06 --view {args.view}: write the view outside the repo or to a gitignored path, "
                       "never into the cases folder"])
    src = load_sources()
    if args.originals:
        originals = allocate(src["vclean"])
        err = write_files(d, {"originals.csv": render_originals(originals)})
        if err:
            raise Refused([err])
        for lab, n in alloc_counts(originals).items():
            say(f"label {lab}: " + ", ".join(f"{c} {v}" for c, v in n.items()))
        say(f"OK wrote originals.csv ({len(originals)} originals)")
        return 0
    mode = "check" if args.check else "apply" if args.apply_review else "build"
    files, b, rows, stats = compute(src, d, mode)
    if mode == "check":
        errors = check_files(d, files)
        if errors:
            raise Refused(errors)
        say(f"OK --check: {', '.join(FILES)} equal the rebuilt bytes")
    else:
        err = write_files(d, files)
        if err:
            raise Refused([err])
        for e in stats["new_events"]:
            repl = e["replacement_group_id"] or "no reserve left (slot stays empty)"
            say(f"replaced {e['condition']} {e['group_id']} (round {e['round']}) -> {repl}")
        dropped = f", {stats['dropped']} pending row(s) of unassigned cases dropped" if stats["dropped"] else ""
        say(f"OK wrote {', '.join(FILES)} ({len(b.cases)} cases, {len(rows)} review rows{dropped})")
    report(b, rows, args.print_n, src)
    if view:
        view.parent.mkdir(parents=True, exist_ok=True)
        view.write_bytes(render_view(b, rows, src))
        say(f"OK wrote the read-only view {view.name} ({sum(c['status'] == 'assigned' for c in b.cases.values())}"
            " assigned cases); decide in review.csv only")
    return 0


def _ignored(path):
    """True when git ignores path (so a view inside the repo is never committed)."""
    import subprocess
    try:
        r = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=ROOT, capture_output=True)
    except OSError:
        return False
    return r.returncode == 0


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
        say(f"FAIL X01 unexpected {build_lists.exc_name(exc)} at build_dev_cases.py:{lines[-1] if lines else '?'} "
            "(message suppressed)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
