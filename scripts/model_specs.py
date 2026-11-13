"""Single source of model IDs, pinned revisions, tokenizer and weight loading.

Scripts import this module as `import model_specs`: `scripts/` is sys.path[0]
when a script is run as `python scripts/x.py`. Future training code should
import it the same way, or add `scripts/` to sys.path first.

DeBERTa-v3 normaliser: transformers 5.22.2 builds DebertaV2Tokenizer with
[Replace, NFC, Strip(right)] and drops the nmt_nfkc normalisation stored in
spm.model, so full-width, math-bold and zero-width characters no longer fold
to their ASCII forms. load_tokenizer("deberta") replaces the normaliser with
the original sentencepiece one (Strip, Precompiled charsmap from spm.model,
Replace " {2,}" -> " "; the same as transformers' DebertaV2Converter).
Agreement with raw sentencepiece on train+validation goes from 13,182 to
13,188 of 13,190 rows; the 2 remaining rows differ because spm uses
byte-fallback pieces that the fast Unigram model does not have
(scripts/check_tokenizers.py).

Rules:
  - Load tokenizers only with load_tokenizer(key) and pretrained weights only
    with load_model(key). load_model requires HF_HUB_OFFLINE=1, refuses keys
    without a pinned weight hash in WEIGHTS, checks size and SHA-256 before
    loading, and loads in FP32: the default dtype="auto" would keep DeBERTa's
    stored FP16. It sets DISABLE_SAFETENSORS_CONVERSION=1 for the whole process.
  - Never call AutoTokenizer.from_pretrained on a checkpoint or run directory:
    DeBERTa reverts to the stock normaliser, and a directory holding only
    config.json silently yields a 5-7-token vocab. Do not save tokenizers into
    checkpoints.
  - Record the model key and tokenizer_fingerprint(tok) in the run config.
    check_tokenizer(tok, key) compares a tokenizer with the pinned value in
    TOKENIZER_SHA256; eval and demo code should also assert that the recorded
    value equals check_tokenizer(load_tokenizer(key), key).
  - Seed the RNG before load_model: the classification head is randomly
    initialised.
  - The pinned fingerprints hold for the exact library versions in
    requirements.txt; a version bump requires re-verification
    (scripts/check_tokenizers.py). The file hashes do not depend on versions.

transformers, torch, huggingface_hub, tokenizers and sentencepiece are
imported lazily inside functions so that importing this module is cheap.
"""
import hashlib
import json
import os

MODELS = {
    "bert": ("google-bert/bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594"),
    "roberta": ("FacebookAI/roberta-base", "e2da8e2f811d1448a5b465c236feacd80ffbac7b"),
    "deberta": ("microsoft/deberta-v3-small", "a36c739020e01763fe789b4b85e2df55d6180012"),
}

# transformers 5.22.2 emits a false-positive fix_mistral_regex warning when
# DeBERTa is loaded offline (config.json lacks transformers_version). Offline,
# passing True replaces the Metaspace pre-tokenizer and breaks tokenization
# (online it is a no-op); passing False silences the warning without changing
# the tokenizer. Model weights must be loaded at the same revisions.
TOKENIZER_KWARGS = {"deberta": {"fix_mistral_regex": False}}

SPM_SHA256 = "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd"

# Expected tokenizer_fingerprint() value per key (pinned tokenizer pipelines).
TOKENIZER_SHA256 = {
    "bert": "4d03705f0a40bce5fc03ae80e84ff8ec33eafd64ec90aded019d2be80300fc52",
    "roberta": "0f49de6c1aae4b1b86b2badcb8000545cbb34cf6399caebbe84d76ccb636cb0e",
    "deberta": "4adacb9141d658578ad4de72802c8cb37d4f9618ee7652dc6f6e7a85ac037b5b",
}

# key -> (file name at the pinned revision, sha256, size in bytes); values are
# the Hub LFS metadata of that file (HfApi().model_info(repo, revision=rev,
# files_metadata=True)). DeBERTa only for now; BERT/RoBERTa are added when
# their weights are first downloaded, and load_model refuses them until then.
WEIGHTS = {
    "deberta": ("pytorch_model.bin", "d4b0ebcc7799a2c2dde1a39199cb0ed709f2c6b63a48b8c05133b77955f271a9", 286059269),
}

# load_model overrides that cannot change which weights are read or their dtype.
MODEL_OVERRIDES = {"output_loading_info", "attn_implementation", "id2label", "label2id"}
# attn_implementation also accepts Hub kernel repo IDs (remote code); allow only built-ins.
ATTN_IMPLEMENTATIONS = {"eager", "sdpa"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _deberta_spm_normalizer(repo, rev):
    """Build DeBERTa-v3's original sentencepiece normaliser from the pinned spm.model.

    Runs before the tokenizer is loaded, so spm.model is SHA-checked before
    transformers parses it; the charsmap is parsed from the hashed bytes.
    """
    from huggingface_hub import hf_hub_download
    # Never import transformers' sentencepiece_model_pb2_new: it registers the
    # same proto file as sentencepiece.sentencepiece_model_pb2, and having both
    # in one process raises TypeError.
    from sentencepiece import sentencepiece_model_pb2
    from tokenizers import Regex, normalizers

    with open(hf_hub_download(repo, "spm.model", revision=rev), "rb") as f:
        data = f.read()
    got = hashlib.sha256(data).hexdigest()
    if got != SPM_SHA256:
        raise RuntimeError(f"{repo} spm.model sha256 {got[:12]} != pinned {SPM_SHA256[:12]}")
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(data)
    charsmap = proto.normalizer_spec.precompiled_charsmap
    if not charsmap:
        raise RuntimeError(f"{repo} spm.model has an empty precompiled_charsmap")
    return normalizers.Sequence(
        [normalizers.Strip(), normalizers.Precompiled(charsmap), normalizers.Replace(Regex(" {2,}"), " ")])


# key -> function(repo, rev) returning the normaliser that replaces the loaded one.
TOKENIZER_NORMALIZERS = {"deberta": _deberta_spm_normalizer}


def normalizer_types(tok):
    n = json.loads(tok.backend_tokenizer.to_str())["normalizer"]
    if n is None:
        return []
    if n["type"] == "Sequence":
        return [x["type"] for x in n["normalizers"]]
    return [n["type"]]


def tokenizer_fingerprint(tok):
    # Byte-exact: the pinned TOKENIZER_SHA256 values depend on this encoding.
    # Covers added tokens, normaliser, pre-tokenizer, post-processor, decoder and
    # vocabulary. Unigram scores are dropped (float round-trip drift; spm.model
    # is SHA-checked separately); truncation/padding state and Python-side
    # attributes are not included.
    j = json.loads(tok.backend_tokenizer.to_str())
    m = j["model"]
    if m.get("type") == "Unigram":
        m["vocab"] = [p for p, _ in m["vocab"]]
    keep = {k: j[k] for k in ("added_tokens", "normalizer", "pre_tokenizer", "post_processor", "decoder", "model")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, ensure_ascii=True).encode("ascii")).hexdigest()


def check_tokenizer(tok, key):
    got = tokenizer_fingerprint(tok)
    want = TOKENIZER_SHA256[key]
    if got != want:
        raise RuntimeError(
            f"tokenizer fingerprint mismatch for {key!r}: got {got[:12]}, expected {want[:12]} "
            f"(len={len(tok)}, normalizer={normalizer_types(tok)}); "
            f"load tokenizers only via model_specs.load_tokenizer({key!r})")
    return got


def load_tokenizer(key, *, _verify=True, **overrides):
    """_verify=False skips the fingerprint guard; only scripts/check_tokenizers.py uses it for controls."""
    from transformers import AutoTokenizer

    repo, rev = MODELS[key]
    build = TOKENIZER_NORMALIZERS.get(key)
    normalizer = build(repo, rev) if build else None
    tok = AutoTokenizer.from_pretrained(repo, revision=rev, **{**TOKENIZER_KWARGS.get(key, {}), **overrides})
    if normalizer is not None:
        if getattr(tok, "do_lower_case", False):
            raise RuntimeError(f"{key}: do_lower_case=True would be lost when the normaliser is replaced")
        tok.backend_tokenizer.normalizer = normalizer
    if _verify:
        check_tokenizer(tok, key)
    return tok


def load_model(key, num_labels=2, **overrides):
    bad = sorted(set(overrides) - MODEL_OVERRIDES)
    if bad:
        raise RuntimeError(f"load_model: override(s) not allowed: {bad}; allowed: {sorted(MODEL_OVERRIDES)}")
    if overrides.get("attn_implementation", "eager") not in ATTN_IMPLEMENTATIONS:
        raise RuntimeError(f"load_model: attn_implementation must be one of {sorted(ATTN_IMPLEMENTATIONS)}")
    if key not in WEIGHTS:
        raise RuntimeError(f"load_model: no pinned weight hash for {key!r}; add it to model_specs.WEIGHTS "
                           "and run `scripts/check_weights.py --download`")
    from huggingface_hub import constants

    if not constants.HF_HUB_OFFLINE:
        raise RuntimeError("load_model requires HF_HUB_OFFLINE=1 (set before Python starts)")
    # Read by transformers at call time; offline mode already prevents the
    # background safetensors conversion request.
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"

    import torch
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from transformers import AutoModelForSequenceClassification

    repo, rev = MODELS[key]
    fname, sha, size = WEIGHTS[key]
    try:
        path = hf_hub_download(repo, fname, revision=rev)
        config_path = hf_hub_download(repo, "config.json", revision=rev)
    except LocalEntryNotFoundError as e:
        raise RuntimeError("weights not in cache: run `scripts/check_weights.py --download`") from e
    got_size = os.path.getsize(path)
    if got_size != size:
        raise RuntimeError(f"{repo} {fname}: size {got_size} != pinned {size}")
    got_sha = sha256_file(path)
    if got_sha != sha:
        raise RuntimeError(f"{repo} {fname}: sha256 {got_sha[:12]} != pinned {sha[:12]}")
    with open(config_path, encoding="utf-8") as f:
        if "transformers_weights" in json.load(f):
            # would make transformers load a different, unhashed file
            raise RuntimeError(f"{repo} config.json redirects weights (transformers_weights)")
    return AutoModelForSequenceClassification.from_pretrained(
        repo, revision=rev, num_labels=num_labels, dtype=torch.float32,
        use_safetensors=fname.endswith(".safetensors"), **overrides)
