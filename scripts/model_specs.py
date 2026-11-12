"""Single source of model IDs, pinned revisions and tokenizer load kwargs.

Scripts import this module as `import model_specs`: `scripts/` is sys.path[0]
when a script is run as `python scripts/x.py`. Future training code should
import it the same way, or add `scripts/` to sys.path first.

transformers is imported lazily inside load_tokenizer so that importing this
module is cheap and has no side effects.
"""

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


def load_tokenizer(key, **overrides):
    from transformers import AutoTokenizer

    repo, rev = MODELS[key]
    kwargs = {**TOKENIZER_KWARGS.get(key, {}), **overrides}
    return AutoTokenizer.from_pretrained(repo, revision=rev, **kwargs)
