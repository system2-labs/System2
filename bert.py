"""bert.py - lazy, cached ModernBERT loader.

Every module in the repo (heads, system, encoding, load_system2, sys2.engine,
routing.domain_router) goes through get_bert(), so one backbone name == one
resident 395M model. Nothing loads at import time: a load costs ~10-16s and used
to run three times over (bert.py, engine, router) on every `import`.
"""

import os
import threading

_cached_models = {}
_cache_lock = threading.Lock()

# Instruct is what every models/*.pt checkpoint records in `bert_model` and what
# routing/domain_centroids.pt was computed against (routing/compute_centroids.py).
# The old "ModernBERT-large" default silently disagreed with both.
DEFAULT_BACKBONE = os.environ.get("BERT_MODEL", "answerdotai/ModernBERT-Large-Instruct")


def get_bert(model_name=None):
    """Fetch cached (tokenizer, model) pair for the given backbone name (default: DEFAULT_BACKBONE)."""
    model_name = model_name or DEFAULT_BACKBONE
    cached = _cached_models.get(model_name)
    if cached is not None:
        return cached
    # Serialize cold-cache initialization so concurrent callers never load two
    # 395M copies for the same backbone name.
    with _cache_lock:
        if model_name not in _cached_models:
            # Deferred import: `import transformers` + AutoModel/AutoTokenizer resolution
            # costs seconds on its own, so pay it only when a backbone is really needed.
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
            mod = AutoModel.from_pretrained(model_name)
            _cached_models[model_name] = (tok, mod)
        return _cached_models[model_name]


def __getattr__(name):
    # PEP 562 compat shim: legacy scripts (eval.py, train.py, interact.py,
    # test_mote_canary.py) do `from bert import tokenizer, model`. Resolving those
    # names lazily on first access keeps them working while `import bert` and
    # `from bert import get_bert` stay free. Only fires for attributes that are
    # missing from the module dict, so load_system2's `bert.model = ...` override
    # still wins once set.
    if name in ("tokenizer", "model"):
        tok, mod = get_bert()
        return tok if name == "tokenizer" else mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def encode(text, model_name=None):
    """text -> mean-pooled (1, 1024) tensor."""
    tok, mod = get_bert(model_name)
    # The shared backbone may live on MPS/CUDA after System2Engine moves it;
    # place inputs on the model's device to prevent CPU/accelerator mismatch.
    model_device = next(mod.parameters()).device
    inputs = tok(text, return_tensors="pt").to(model_device)
    last_hidden = mod(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


if __name__ == "__main__":
    print(encode("hello world").shape)  # (1, 1024)
