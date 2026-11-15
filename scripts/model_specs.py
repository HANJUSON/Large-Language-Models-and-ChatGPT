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
  - Fine-tuned checkpoints (README 7.1 runs/): save only with
    save_checkpoint(model, key, folder, run_id, step): FP32 state_dict as
    safetensors, streamed one tensor at a time to .tmp, then os.replace,
    metadata model_key, repo, revision, run_id, step (sorted in the header, so
    equal tensors and metadata give equal bytes; the bytes equal
    safetensors.torch.save with the header's __metadata__ keys sorted); returns
    the file SHA-256, size and a tensor_digest per tensor. The folder holds
    only model.safetensors (no config, no tokenizer). Load only with
    load_finetuned(key, path, sha256):
    HF_HUB_OFFLINE=1, size and SHA-256 checked before the bytes are parsed,
    metadata key/repo/revision checked, the model built with from_config from
    the pinned config (num_labels=2, FP32, eager) under fork_rng, identical
    key set, all FP32, load_state_dict strict. Never from_pretrained on a run
    or checkpoint folder; the tokenizer comes from load_tokenizer(key) and its
    fingerprint must equal the run record.
  - The pinned fingerprints hold for the exact library versions in
    requirements.txt; a version bump requires re-verification
    (scripts/check_tokenizers.py). The file hashes do not depend on versions.

transformers, torch, huggingface_hub, tokenizers and sentencepiece are
imported lazily inside functions so that importing this module is cheap.
"""
import hashlib
import json
import os
import sys

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
# files_metadata=True)), read 2026-11-13. BERT and RoBERTa ship safetensors at
# their pinned revisions; DeBERTa-v3-small only a pickle .bin (README 9).
WEIGHTS = {
    "bert": ("model.safetensors", "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3", 440449768),
    "roberta": ("model.safetensors", "5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81", 498818054),
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


CHECKPOINT_FILE = "model.safetensors"


def _cpu_bytes(t):
    """(CPU tensor, its raw native-order bytes as a zero-copy uint8 numpy view)."""
    import torch

    c = t.detach().to("cpu").contiguous()
    return c, c.reshape(-1).view(torch.uint8).numpy()


def _digest(c, raw):
    return [str(c.dtype), list(c.shape), hashlib.sha256(raw).hexdigest()]


def tensor_digest(t):
    """[dtype, shape, SHA-256 of the raw bytes] of one tensor; equal digests mean bitwise-equal tensors."""
    return _digest(*_cpu_bytes(t))


def _plain_json_string(s):
    """True when json.dumps(s) is s in quotes: then Python and safetensors (serde_json) encode it alike."""
    return json.dumps(s, ensure_ascii=True) == f'"{s}"'


def save_checkpoint(model, key, folder, run_id, step):
    """Save model as FP32 safetensors in folder/model.safetensors; see the module rules.

    The file is streamed one tensor at a time (one CPU tensor in host RAM at a
    time, hashed while written) in the layout safetensors.torch.save gives with
    the __metadata__ keys sorted: 8-byte little-endian header length, compact
    JSON header {"__metadata__": sorted metadata, tensors by name: dtype, shape,
    data_offsets} padded with spaces to a multiple of 8, then the tensor bytes
    in name order (safetensors orders by dtype, then name; all are F32 here).
    Returns {"sha256", "size", "tensors"}; tensors maps each name to
    tensor_digest() of the bytes written (the caller's reload check compares it).
    """
    import torch

    if sys.byteorder != "little":
        raise RuntimeError("save_checkpoint: safetensors stores little-endian bytes; this host is big-endian")
    repo, rev = MODELS[key]
    os.makedirs(folder, exist_ok=True)
    final = os.path.join(folder, CHECKPOINT_FILE)
    tmp = final + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    other = sorted(set(os.listdir(folder)) - {CHECKPOINT_FILE})
    if other:
        raise RuntimeError(f"save_checkpoint: the checkpoint folder holds other files: {other}")
    sd = model.state_dict()
    for name, t in sd.items():
        if t.dtype != torch.float32:
            raise RuntimeError(f"save_checkpoint: {name} is {t.dtype}, not float32")
    meta = {"model_key": key, "repo": repo, "revision": rev, "run_id": str(run_id), "step": str(step)}
    if not all(_plain_json_string(s) for s in list(sd) + list(meta.values())):
        raise RuntimeError("save_checkpoint: tensor names and metadata must be plain ASCII (no JSON escapes)")
    names, entries, offset = sorted(sd), {}, 0
    for name in names:
        n = sd[name].numel() * 4
        entries[name] = {"dtype": "F32", "shape": list(sd[name].shape), "data_offsets": [offset, offset + n]}
        offset += n
    header = json.dumps({"__metadata__": dict(sorted(meta.items())), **entries}, separators=(",", ":"),
                        ensure_ascii=True).encode("ascii")
    header += b" " * (-len(header) % 8)
    h, digests, size = hashlib.sha256(), {}, 0
    try:
        with open(tmp, "xb") as f:
            for chunk in (len(header).to_bytes(8, "little"), header):
                f.write(chunk)
                h.update(chunk)
                size += len(chunk)
            for name in names:
                c, raw = _cpu_bytes(sd[name])
                f.write(raw)
                h.update(raw)
                size += raw.nbytes
                digests[name] = _digest(c, raw)
                del c, raw
        os.replace(tmp, final)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {"sha256": h.hexdigest(), "size": size, "tensors": digests}


def _header(data):
    n = int.from_bytes(data[:8], "little")
    if len(data) < 8 + n:
        raise RuntimeError("checkpoint: truncated safetensors header")
    return n, json.loads(data[8:8 + n].decode("utf-8"))


def checkpoint_metadata(data):
    """The __metadata__ dict of safetensors bytes (header parsed without loading tensors)."""
    return _header(data)[1].get("__metadata__") or {}


def load_finetuned(key, path, sha256, *, size=None, attn_implementation="eager"):
    """Load a save_checkpoint file of model key; see the module rules. Returns the model (CPU, eval mode)."""
    if attn_implementation not in ATTN_IMPLEMENTATIONS:
        raise RuntimeError(f"load_finetuned: attn_implementation must be one of {sorted(ATTN_IMPLEMENTATIONS)}")
    from huggingface_hub import constants

    if not constants.HF_HUB_OFFLINE:
        raise RuntimeError("load_finetuned requires HF_HUB_OFFLINE=1 (set before Python starts)")
    repo, rev = MODELS[key]
    got_size = os.path.getsize(path)
    if size is not None and got_size != size:
        raise RuntimeError(f"checkpoint: size {got_size} != recorded {size}")
    with open(path, "rb") as f:
        data = f.read()
    got_sha = hashlib.sha256(data).hexdigest()
    if got_sha != sha256:
        raise RuntimeError(f"checkpoint: sha256 {got_sha[:12]} != recorded {sha256[:12]}")
    meta = checkpoint_metadata(data)
    for field, want in (("model_key", key), ("repo", repo), ("revision", rev)):
        if meta.get(field) != want:
            raise RuntimeError(f"checkpoint: metadata {field} {meta.get(field)!r} != {want!r}")

    import torch
    from safetensors.torch import load
    from transformers import AutoConfig, AutoModelForSequenceClassification

    state = load(data)
    del data  # freed before the model is built: host RAM peaks at two copies, not three
    config = AutoConfig.from_pretrained(repo, revision=rev, num_labels=2)
    with torch.random.fork_rng(devices=[]):
        model = AutoModelForSequenceClassification.from_config(
            config, dtype=torch.float32, attn_implementation=attn_implementation)
    want_keys, got_keys = set(model.state_dict()), set(state)
    if want_keys != got_keys:
        missing, extra = sorted(want_keys - got_keys), sorted(got_keys - want_keys)
        raise RuntimeError(f"checkpoint: key set differs from the model: missing {len(missing)} {missing[:3]}, "
                           f"unexpected {len(extra)} {extra[:3]}")
    bad = sorted(k for k, t in state.items() if t.dtype != torch.float32)
    if bad:
        raise RuntimeError(f"checkpoint: non-float32 tensors {bad[:3]}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
