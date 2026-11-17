"""English / non-English rule for external sources without language metadata (README 3.4 EXT-06, stage-4 design D7).

Standard library plus scripts/build_lists.py (punct key) and scripts/data_lists.py
(--self-check only: raw train/validation rows and the test.csv guard). Nothing
here prints or writes sentence text: the self-check prints counts, and sentence
IDs only with --ids.

Rule (RULE_ID), applied to one text:
  1. Word tokens = the words of the punct key (build_lists.pk, README 3.2):
     NFKC, whitespace collapsed, casefold (lowercase; sharp s -> ss), every
     Unicode punctuation (P*) and symbol (S*) character -> space. "Don't" gives
     the tokens "don" and "t".
  2. en_hits = number of tokens (occurrences, not distinct words) that are in
     EN_WORDS; de_hits = the same for DE_WORDS.
  3. english       if en_hits > de_hits and en_hits >= 1
     german        if de_hits > en_hits
     undetermined  otherwise (equal counts, including 0 and 0)
  Main evaluation (MAIN_LANGS) = english and undetermined rows; german rows
  (SUPPLEMENTARY_LANGS: the rows the rule identifies as non-English) are
  reported as supplementary results (lead decision in design D7, option b).
  Reason: on S-Labs about 4% of the rows come out undetermined, nearly all of
  them without any list word and most of them injections (self-check counts),
  so leaving them out would bias the main evaluation by label. The number of
  undetermined rows in the main evaluation is reported with the results.
  Labels are never changed by this rule.

Word lists (fixed with the rule; their SHA-256 is recorded in
data/external/external_rules.json):
  - closed-class words only (articles, pronouns, prepositions, conjunctions,
    auxiliaries and modals, particles), written in punct-key form;
  - EN_WORDS and DE_WORDS are disjoint;
  - the words of AMBIGUOUS are in neither list. AMBIGUOUS is a hand-picked list
    of words that are ordinary words of both languages or of the other one (in,
    an, am, was, so, will, also, war, die, man, hat, bin, im, um, dir, nun,
    falls, us, such); it is not complete (see Limits);
  - DE_WORDS also holds the ae/oe/ue spellings of its umlaut words (fuer, ueber).
  The module refuses to import (ValueError) when a list breaks these properties,
  when a KNOWN_OVERLAPS word is not in the list it is recorded for, or when
  MAIN_LANGS and SUPPLEMENTARY_LANGS do not split LANGS.

Limits (recorded, not corrected):
  - The rule separates English from German only. A text in a third language
    comes out undetermined, or english when it happens to contain English list
    tokens (for example a, me, no, on); either way it is in the main evaluation.
  - A text without any list word comes out undetermined and is in the main
    evaluation, whatever its language: a short English command, and also a
    German text without a German list word.
  - Some list words are also words of the other language and are not in
    AMBIGUOUS, and one hit decides a short text: all, her, not (English list;
    German "all", "her", "Not") and den, mit (German list; English "den",
    "MIT"). "Komm her" comes out english, "MIT license" german and "Hilfe in der
    Not" undetermined (KNOWN_OVERLAPS; the lists are kept as they are).

Modes:
  --self-check  applies the rule to every raw S-Labs train and validation row
                (data_lists.load_rows(), after data_lists.install_test_guard();
                test is never read) and prints counts per split, per label and
                per reason. The counts, the main and supplementary language
                sets, the word-list record and the known overlaps are compared
                with the "language" -> "deepset" block of
                data/external/external_rules.json (--rules PATH; --no-rules
                skips the comparison). --ids also prints the sentence IDs of
                the rows that are not english (IDs only, at most 20 per group).

Refusal codes (exit 1, "FAIL <code> ..." lines, nothing written):
  XL01 no mode given, or --no-rules together with --rules
  XL02 the S-Labs rows cannot be loaded (data_lists.ListError messages)
  XL03 the rules file is missing, unreadable or not UTF-8 JSON (also: nested too deeply), or has no
       language.deepset block
  XL04 the rule ID, the main or supplementary language set, a word-list count/SHA-256, the known
       overlaps or a self-check count differs from the rules file

Run from Git Bash at the repo root:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/external_lang.py --self-check
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import build_lists
import data_lists

ROOT = build_lists.ROOT
DEFAULT_RULES = ROOT / "data" / "external" / "external_rules.json"
RULE_ID = "external_lang_v1"
LANGS = ("english", "german", "undetermined")
MAIN_LANGS = ("english", "undetermined")  # rows of the main evaluation (design D7, lead decision, option b)
SUPPLEMENTARY_LANGS = ("german",)         # rows identified as non-English: supplementary results
SPLITS = ("train", "validation")
ID_LIMIT = 20
LENGTH_BUCKETS = (("0", 0, 0), ("1-3", 1, 3), ("4-9", 4, 9), ("10+", 10, None))  # word tokens per row

EN_WORDS = tuple("""
a about above across after again against all although always and another any anyone anything are aren as at
be because been before behind being below between beyond both but by
can cannot could couldn
did didn do does doesn doing don down during
each ever every everyone everything
few for from further
had hadn has hasn have having he her here herself him himself his how
i if into is isn it its itself
just
may me might more most must my myself
never no nor not nothing now
of off on once only onto or other our ours ourselves out over own
please
same shall she should shouldn since some someone something
than that the their theirs them themselves then there these they this those though through to too toward towards
under unless until up upon
very
wasn we were weren what when where whether which while who whom whose why with within without would wouldn
yes yet you your yours yourself
""".split())

DE_WORDS = tuple("""
aber alle allen alles als andere anderen auch auf aus
bei beim bis bist bitte
da dabei dafuer dafür daher damit dann daran darauf darf darfst das dass davon dazu dein deine deinem deinen deiner
dem den denn der des deshalb dich diese diesem diesen dieser dieses doch dort du duerfen durch dürfen
ein eine einem einen einer eines einige er es etwa etwas euch euer eure
fuer für
ganz gegen gern gerne
habe haben habt hast hatte hatten hier hinter
ich ihm ihn ihnen ihr ihre ihrem ihren ihrer ins ist
ja jede jedem jeden jeder jedes jedoch jemand jetzt
kann kannst kein keine keinem keinen keiner koennen koennte koennten können könnte könnten
mal mehr mein meine meinem meinen meiner mich mir mit moechte moechten muessen muss musst möchte möchten müssen
nach neben nein nicht nichts nie niemals niemand noch nur
ob obwohl oder ohne
schon sehr seid sein seine seinem seinen seiner seit selbst sich sie sind soll sollen sollst sollte sollten sondern
sowie
trotzdem
ueber und uns unser unsere unseren unter über
viel viele vielleicht vom von vor
waehrend waere wann waren warum wegen weil welche welchen welcher welches wem wen wenig weniger wenn wer werden
weshalb wie wieder wieso willst wir wird wirst wo woher wohin wollen wuerde wuerden wurde wurden während wäre würde
würden
zu zum zur zwischen
""".split())

# Ordinary words of both languages (or of the other one): in neither list. Hand-picked, not complete.
AMBIGUOUS = tuple("in an am was so will also war die man hat bin im um dir nun falls us such".split())

# Recorded limit: list words that are also words of the other language and stay in their list
# (German "all", "her", "Not"; English "den", "MIT"). One such hit can decide a short text.
KNOWN_OVERLAPS = {"english": ("all", "her", "not"), "german": ("den", "mit")}


def word_list_errors(en=EN_WORDS, de=DE_WORDS, ambiguous=AMBIGUOUS):
    """Messages for every broken word-list property (empty when the lists are valid)."""
    errors = []
    for name, words in (("EN_WORDS", en), ("DE_WORDS", de), ("AMBIGUOUS", ambiguous)):
        if not words:
            errors.append(f"{name} is empty")
        bad = sorted(w for w in set(words) if not isinstance(w, str) or build_lists.pk(w).split() != [w])
        if bad:
            errors.append(f"{name}: {len(bad)} entries are not single punct-key words (first: {bad[0]!a})")
        if len(set(words)) != len(words):
            errors.append(f"{name}: {len(words) - len(set(words))} repeated entries")
    both = sorted(set(en) & set(de))
    if both:
        errors.append(f"EN_WORDS and DE_WORDS share {len(both)} words (first: {both[0]!a})")
    listed = sorted(set(ambiguous) & (set(en) | set(de)))
    if listed:
        errors.append(f"{len(listed)} AMBIGUOUS words are in a word list (first: {listed[0]!a})")
    return errors


def overlap_errors(overlaps=KNOWN_OVERLAPS, en=EN_WORDS, de=DE_WORDS):
    """Messages for a stale KNOWN_OVERLAPS record: every recorded word must be in the list it is recorded for."""
    errors = []
    if sorted(overlaps) != ["english", "german"]:
        return ["KNOWN_OVERLAPS does not have exactly the keys english and german"]
    for lang, words in (("english", en), ("german", de)):
        gone = sorted(set(overlaps[lang]) - set(words))
        if gone:
            errors.append(f"KNOWN_OVERLAPS.{lang}: {len(gone)} words are not in the {lang} list (first: {gone[0]!a})")
        if list(overlaps[lang]) != sorted(set(overlaps[lang])):
            errors.append(f"KNOWN_OVERLAPS.{lang} is not sorted without repeats")
    return errors


_ERRORS = word_list_errors() + overlap_errors()
if _ERRORS:
    raise ValueError("external_lang word lists are invalid: " + "; ".join(_ERRORS))
if sorted(MAIN_LANGS + SUPPLEMENTARY_LANGS) != sorted(LANGS):
    raise ValueError("external_lang: MAIN_LANGS and SUPPLEMENTARY_LANGS do not split LANGS")
_EN = frozenset(EN_WORDS)
_DE = frozenset(DE_WORDS)


# ---------------------------------------------------------------- rule

def tokens(text):
    """Word tokens of one text: the words of its punct key (rule step 1)."""
    if not isinstance(text, str):
        raise TypeError("external_lang: text must be str")
    return build_lists.pk(text).split()


def count_hits(text):
    """(en_hits, de_hits): occurrences of English and German list words among the tokens (rule step 2)."""
    en = de = 0
    for t in tokens(text):
        if t in _EN:
            en += 1
        elif t in _DE:
            de += 1
    return en, de


def decide(en_hits, de_hits):
    """Language name from the two hit counts (rule step 3)."""
    if en_hits > de_hits and en_hits >= 1:
        return "english"
    if de_hits > en_hits:
        return "german"
    return "undetermined"


def classify(text):
    """(language, en_hits, de_hits) for one text; language is one of LANGS."""
    en, de = count_hits(text)
    return decide(en, de), en, de


def evaluation_set(language):
    """"main" or "supplementary": the evaluation a row with this language name belongs to."""
    if language in MAIN_LANGS:
        return "main"
    if language in SUPPLEMENTARY_LANGS:
        return "supplementary"
    raise ValueError("external_lang: unknown language name")


def words_sha256(words):
    """SHA-256 of the sorted words, one per line (LF, UTF-8, final LF)."""
    return hashlib.sha256(("\n".join(sorted(words)) + "\n").encode("utf-8")).hexdigest()


def rule_record():
    """What external_rules.json records about this rule (compared by --self-check)."""
    return {"rule_id": RULE_ID,
            "main": list(MAIN_LANGS),
            "supplementary": list(SUPPLEMENTARY_LANGS),
            "stopwords": {"english": {"count": len(EN_WORDS), "sha256": words_sha256(EN_WORDS)},
                          "german": {"count": len(DE_WORDS), "sha256": words_sha256(DE_WORDS)}},
            "known_overlaps": {lang: list(words) for lang, words in KNOWN_OVERLAPS.items()}}


# ---------------------------------------------------------------- self-check

def _bucket(n):
    for name, lo, hi in LENGTH_BUCKETS:
        if n >= lo and (hi is None or n <= hi):
            return name
    raise ValueError(n)


def self_check(rows):
    """(counts, ids, lengths) of the rule on {split: {sid: (label, text)}}; no text is kept.

    counts[split] = rows, english, german, undetermined, undetermined_no_hit (0 and 0),
    undetermined_tie (equal counts >= 1), by_label[label][language].
    ids[split][language] = sorted sentence IDs of the german / undetermined rows.
    lengths[split][language][bucket] = rows per word-token bucket (german / undetermined).
    """
    counts, ids, lengths = {}, {}, {}
    for split in SPLITS:
        c = {"rows": 0, "english": 0, "german": 0, "undetermined": 0,
             "undetermined_no_hit": 0, "undetermined_tie": 0, "by_label": {}}
        i = {"german": [], "undetermined": []}
        ln = {lang: {b[0]: 0 for b in LENGTH_BUCKETS} for lang in ("german", "undetermined")}
        for sid in sorted(rows[split]):
            label, text = rows[split][sid]
            lang, en, de = classify(text)
            c["rows"] += 1
            c[lang] += 1
            c["by_label"].setdefault(str(label), {k: 0 for k in LANGS})[lang] += 1
            if lang == "undetermined":
                c["undetermined_no_hit" if en == 0 else "undetermined_tie"] += 1
            if lang != "english":
                i[lang].append(sid)
                ln[lang][_bucket(len(tokens(text)))] += 1
        c["by_label"] = {k: c["by_label"][k] for k in sorted(c["by_label"])}
        counts[split], ids[split], lengths[split] = c, i, ln
    return counts, ids, lengths


def load_rules_block(path):
    """(language.deepset block, errors) from the rules file (XL03)."""
    shown = build_lists.show(path)
    try:
        doc = json.loads(Path(path).read_bytes().decode("utf-8"))
    except OSError as exc:
        return None, [f"XL03 {shown}: cannot read ({build_lists.exc_name(exc)})"]
    except ValueError:
        return None, [f"XL03 {shown}: not UTF-8 JSON"]
    except RecursionError:
        return None, [f"XL03 {shown}: JSON nested too deeply"]
    try:
        block = doc["language"]["deepset"]
        if not isinstance(block, dict):
            raise TypeError
    except (KeyError, TypeError):
        return None, [f"XL03 {shown}: no language.deepset block"]
    return block, []


def compare_rules(block, counts):
    """XL04 messages for every difference between the rules block and this module / these counts."""
    errors = []
    rec = rule_record()
    if block.get("rule_id") != rec["rule_id"]:
        errors.append(f"XL04 rule_id: rules file {block.get('rule_id')!a}, module {rec['rule_id']!a}")
    for key in ("main", "supplementary"):
        if block.get(key) != rec[key]:
            errors.append(f"XL04 {key}: the rules file differs from the module (module: {', '.join(rec[key])})")
    if block.get("known_overlaps") != rec["known_overlaps"]:
        errors.append("XL04 known_overlaps: the rules file differs from the module")
    recorded = block.get("stopwords")
    for lang, want in rec["stopwords"].items():
        got = recorded.get(lang) if isinstance(recorded, dict) else None
        if got != want:
            errors.append(f"XL04 stopwords.{lang}: the rules file differs from the module "
                          f"(module: count {want['count']}, sha256 {want['sha256'][:12]})")
    check = block.get("self_check")
    recorded_counts = check.get("counts") if isinstance(check, dict) else None
    if not isinstance(recorded_counts, dict):
        errors.append("XL04 self_check.counts: missing in the rules file")
    else:
        for split in SPLITS:
            if recorded_counts.get(split) != counts[split]:
                errors.append(f"XL04 self_check.counts.{split}: the rules file differs from this run")
        extra = sorted(set(recorded_counts) - set(SPLITS))
        if extra:
            errors.append(f"XL04 self_check.counts: unknown split keys {extra!a}")
    return errors


def _refuse(errors):
    for e in errors:
        print(("FAIL " + e).encode("ascii", "backslashreplace").decode("ascii"))
    print(f"refused: {len(errors)} error(s); nothing written")
    return 1


def _pct(n, total):
    return f"{100 * n / total:.2f}%" if total else "n/a"


def _main_line(by_lang):
    """"main N (P%), supplementary M" for a {language: rows} mapping."""
    main = sum(by_lang[k] for k in MAIN_LANGS)
    supplementary = sum(by_lang[k] for k in SUPPLEMENTARY_LANGS)
    return f"main {main} ({_pct(main, main + supplementary)}), supplementary {supplementary}"


def parse_args(argv):
    p = argparse.ArgumentParser(description="English / non-English rule for external sources (see module docstring).")
    p.add_argument("--self-check", action="store_true",
                   help="apply the rule to raw S-Labs train and validation rows and print counts")
    p.add_argument("--rules", help="rules file to compare with (default: data/external/external_rules.json)")
    p.add_argument("--no-rules", action="store_true", help="do not compare with the rules file")
    p.add_argument("--ids", action="store_true", help="also print the sentence IDs of rows that are not english")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.self_check:
        return _refuse(["XL01 nothing to do: run with --self-check"])
    if args.no_rules and args.rules:
        return _refuse(["XL01 --no-rules cannot be combined with --rules"])
    data_lists.install_test_guard()
    try:
        rows = data_lists.load_rows()
    except data_lists.ListError as exc:
        return _refuse([f"XL02 S-Labs rows: {e}" for e in exc.errors])
    counts, ids, lengths = self_check(rows)
    del rows
    rec = rule_record()
    sw = rec["stopwords"]
    print(f"rule {rec['rule_id']}: english words {sw['english']['count']} (sha256 {sw['english']['sha256'][:12]}), "
          f"german words {sw['german']['count']} (sha256 {sw['german']['sha256'][:12]})")
    print("self-check on raw S-Labs train and validation rows (counts only; test is never read)")
    print(f"main evaluation = {' + '.join(MAIN_LANGS)}; supplementary = {' + '.join(SUPPLEMENTARY_LANGS)}")
    total = {k: 0 for k in ("rows",) + LANGS}
    for split in SPLITS:
        c = counts[split]
        for k in total:
            total[k] += c[k]
        print(f"{split}: rows {c['rows']}, english {c['english']} ({_pct(c['english'], c['rows'])}), "
              f"german {c['german']}, undetermined {c['undetermined']} "
              f"(no list word {c['undetermined_no_hit']}, equal counts {c['undetermined_tie']})")
        print("  " + _main_line(c))
        for label, by in c["by_label"].items():
            print(f"  label {label}: " + ", ".join(f"{k} {by[k]}" for k in LANGS) + "; " + _main_line(by))
        for lang in ("german", "undetermined"):
            print(f"  {lang} rows by word tokens: "
                  + ", ".join(f"{b[0]}: {lengths[split][lang][b[0]]}" for b in LENGTH_BUCKETS))
            if args.ids:
                print(f"  {lang} IDs: {build_lists.fmt_ids(ids[split][lang], ID_LIMIT)}")
    print(f"total: rows {total['rows']}, english {total['english']} ({_pct(total['english'], total['rows'])}), "
          f"german {total['german']}, undetermined {total['undetermined']}; {_main_line(total)}")
    if args.no_rules:
        print("rules file: not compared (--no-rules)")
        return 0
    path = Path(args.rules) if args.rules else DEFAULT_RULES
    block, errors = load_rules_block(path)
    if not errors:
        errors = compare_rules(block, counts)
    if errors:
        return _refuse(errors)
    print(f"rules file {build_lists.show(path)}: rule ID, main and supplementary sets, word lists, known "
          "overlaps and self-check counts match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
