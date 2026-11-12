"""Tokenizer verification on the train and validation splits (CPU only).

Checks:
  - row counts (train 11089, validation 2101);
  - per model and split: median/max token length and rows longer than 256;
  - BERT and RoBERTa: ids equal the standalone `tokenizers` reference (tokenizer.json), 100%;
  - DeBERTa: ids vs a raw sentencepiece reference ([CLS] + spm ids + [SEP]),
    spm.model SHA-256 must equal the recorded value. The adopted variant
    (model_specs default, fix_mistral_regex=False) must differ on exactly the
    recorded 8 row IDs, each with its recorded cause ("byte-fallback" when the
    reference uses <0x..> byte pieces, else "normalisation"); the default variant
    (no kwarg) must give ids identical to the adopted one and must emit the
    fix_mistral_regex warning (proves the warning capture works); the adopted
    load must not emit it; fix_mistral_regex=True must match 0 rows;
  - batched encoding (batches of 64, padding="longest", pads stripped by
    attention_mask) equals unpadded encoding for all three models;
  - provenance: revisions, spm.model SHA-256, HF_HUB_OFFLINE, library versions.

Must run with HF_HUB_OFFLINE=1: transformers 5.22.2 applies (and warns about)
fix_mistral_regex only for offline/local loads; online, True is a no-op.

Prints ids, counts and row IDs only, never raw text. test.csv guard: an audit
hook aborts on Python-level opens (open, io.open, os.open) of a path ending in
test.csv. It is a development-hygiene guard, not a security boundary: opens
from native code and Windows path aliases are not covered.

Run from Git Bash at the repo root:
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/check_tokenizers.py

Exit code 0 when all checks pass, 1 on any failure.
"""
import sys


def _audit(event, args):
    if event == "open" and args:
        p = args[0]
        if isinstance(p, bytes):
            p = p.decode("utf-8", "replace")
        if isinstance(p, str) and p.lower().endswith("test.csv"):
            print("FAIL: attempt to open a test.csv path was blocked")
            sys.stdout.flush()
            import os
            os._exit(1)


sys.addaudithook(_audit)

import csv
import hashlib
import logging
import os
import re
import statistics
import warnings
from pathlib import Path

import model_specs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "s-labs-prompt-injection" / "data"
EXPECTED_COUNTS = {"train": 11089, "validation": 2101}
EXPECTED_DEBERTA_MISMATCHES = {
    "train:000679": "normalisation", "train:002481": "normalisation",
    "train:003980": "byte-fallback", "train:004359": "normalisation",
    "train:004999": "normalisation", "train:006743": "normalisation",
    "train:007375": "byte-fallback", "train:007670": "normalisation",
}
EXPECTED_SPM_SHA256 = "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd"
BATCH = 64
MAXLEN = 256

from huggingface_hub import constants as hf_constants

if not hf_constants.HF_HUB_OFFLINE:
    print("FAIL: run with HF_HUB_OFFLINE=1 (transformers applies fix_mistral_regex only to offline/local loads)")
    sys.exit(1)


def fail(msg):
    print("FAIL: " + msg)
    sys.exit(1)


def check(name, ok, got):
    if not ok:
        fail(f"{name}: got {got!r}")
    print(f"ok {name}: {got}")


# ---- data (train and validation only) ----
row_ids, texts = [], []
for split in ("train", "validation"):
    with open(DATA / f"{split}.csv", encoding="utf-8", newline="") as f:
        rows = [r["text"] for r in csv.DictReader(f)]
    check(f"rows {split}", len(rows) == EXPECTED_COUNTS[split], len(rows))
    row_ids += [f"{split}:{i:06d}" for i in range(len(rows))]
    texts += rows
n_train = EXPECTED_COUNTS["train"]


def encode_unpadded(tok, batch_size=512):
    out = []
    for i in range(0, len(texts), batch_size):
        out += tok(texts[i:i + batch_size], add_special_tokens=True)["input_ids"]
    return out


def encode_batched(tok):
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding="longest", add_special_tokens=True)
        for ids, mask in zip(enc["input_ids"], enc["attention_mask"]):
            out.append([t for t, m in zip(ids, mask) if m == 1])
    return out


def length_report(name, ids_list):
    for split, sl in (("train", slice(0, n_train)), ("validation", slice(n_train, None))):
        lens = [len(x) for x in ids_list[sl]]
        print(f"len {name} {split}: n={len(lens)} median={statistics.median(lens)} "
              f"max={max(lens)} rows>{MAXLEN}={sum(l > MAXLEN for l in lens)}")


def mismatch_ids(ids_list, ref_list):
    return [row_ids[k] for k, (a, b) in enumerate(zip(ids_list, ref_list)) if a != b]


from huggingface_hub import hf_hub_download

for key in ("bert", "roberta"):
    repo, rev = model_specs.MODELS[key]
    tok = model_specs.load_tokenizer(key)
    ids = encode_unpadded(tok)
    length_report(key, ids)
    from tokenizers import Tokenizer
    ref_tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", revision=rev))
    ref = [e.ids for e in ref_tok.encode_batch(texts)]
    bad = mismatch_ids(ids, ref)
    check(f"{key} == tokenizers reference", not bad, f"{len(texts) - len(bad)}/{len(texts)} match; mismatches {bad[:10]}")
    bb = mismatch_ids(encode_batched(tok), ids)
    check(f"{key} batched == unpadded", not bb, f"{len(texts) - len(bb)}/{len(texts)} match; mismatches {bb[:10]}")

# ---- DeBERTa ----
import sentencepiece as spm
import tokenizers
import transformers

repo, rev = model_specs.MODELS["deberta"]
spm_path = hf_hub_download(repo, "spm.model", revision=rev)
spm_sha = hashlib.sha256(Path(spm_path).read_bytes()).hexdigest()
check("spm.model sha256", spm_sha == EXPECTED_SPM_SHA256, spm_sha)
sp = spm.SentencePieceProcessor(model_file=spm_path)

records = []
handler = logging.Handler()
handler.emit = lambda rec: records.append(rec.getMessage())
tlog = logging.getLogger("transformers")
tlog.addHandler(handler)
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    tok_adopted = model_specs.load_tokenizer("deberta")
tlog.removeHandler(handler)
msgs = records + [str(w.message) for w in wlist]
check("adopted load: no fix_mistral_regex warning", not any("fix_mistral_regex" in m for m in msgs),
      f"{len(msgs)} warning message(s) captured")

check("spm [CLS] id", sp.piece_to_id("[CLS]") == tok_adopted.cls_token_id, tok_adopted.cls_token_id)
check("spm [SEP] id", sp.piece_to_id("[SEP]") == tok_adopted.sep_token_id, tok_adopted.sep_token_id)
cls_id, sep_id = tok_adopted.cls_token_id, tok_adopted.sep_token_id
ref = [[cls_id] + sp.encode(t) + [sep_id] for t in texts]

ids_adopted = encode_unpadded(tok_adopted)
length_report("deberta", ids_adopted)
byte_piece = re.compile(r"^<0x[0-9A-F]{2}>$")
got = {}
for k, (a, b) in enumerate(zip(ids_adopted, ref)):
    if a != b:
        pieces = sp.encode(texts[k], out_type=str)
        got[row_ids[k]] = "byte-fallback" if any(byte_piece.match(p) for p in pieces) else "normalisation"
print(f"deberta adopted vs spm reference: {len(texts) - len(got)}/{len(texts)} match")
for rid in sorted(got)[:20]:
    print(f"  mismatch {rid}: {got[rid]}")
unexpected = sorted(set(got) - set(EXPECTED_DEBERTA_MISMATCHES))
missing = sorted(set(EXPECTED_DEBERTA_MISMATCHES) - set(got))
wrong_cause = sorted(r for r in set(got) & set(EXPECTED_DEBERTA_MISMATCHES) if got[r] != EXPECTED_DEBERTA_MISMATCHES[r])
check("deberta adopted mismatches == expected rows and causes", not (unexpected or missing or wrong_cause),
      f"unexpected {unexpected[:10]}, missing {missing}, wrong cause {wrong_cause}")
unk_id = tok_adopted.unk_token_id
print(f"deberta adopted [UNK] ids total: {sum(x.count(unk_id) for x in ids_adopted)}")

# default variant: loaded without the fix_mistral_regex kwarg
from transformers import AutoTokenizer

records.clear()
tlog.addHandler(handler)
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    tok_default = AutoTokenizer.from_pretrained(repo, revision=rev)
tlog.removeHandler(handler)
msgs = records + [str(w.message) for w in wlist]
default_warns = any("fix_mistral_regex" in m for m in msgs)
check("default load emits fix_mistral_regex warning (warning-capture control)", default_warns, default_warns)
bd = mismatch_ids(encode_unpadded(tok_default), ids_adopted)
check("deberta default ids == adopted ids", not bd, f"{len(texts) - len(bd)}/{len(texts)} match; mismatches {bd[:10]}")

tok_true = model_specs.load_tokenizer("deberta", fix_mistral_regex=True)
bt = mismatch_ids(encode_unpadded(tok_true), ref)
print(f"deberta fix_mistral_regex=True vs spm reference: {len(texts) - len(bt)}/{len(texts)} match")
check("deberta fix_mistral_regex=True matches 0 rows", len(bt) == len(texts), len(texts) - len(bt))

bb = mismatch_ids(encode_batched(tok_adopted), ids_adopted)
check("deberta batched == unpadded", not bb, f"{len(texts) - len(bb)}/{len(texts)} match; mismatches {bb[:10]}")

# ---- provenance ----
for key, (r, v) in model_specs.MODELS.items():
    print(f"revision {key}: {r}@{v}")
print(f"spm.model sha256: {spm_sha}")
print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")
print(f"versions: transformers={transformers.__version__} tokenizers={tokenizers.__version__} "
      f"sentencepiece={spm.__version__}")
print("ALL TOKENIZER CHECKS PASSED")
