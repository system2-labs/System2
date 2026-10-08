import os
import torch
from transformers import AutoModel, AutoTokenizer

_cached_models = {}


def get_bert(model_name="answerdotai/ModernBERT-large"):
    """Fetch cached (tokenizer, model) pair for the given backbone name."""
    if model_name not in _cached_models:
        tok = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
        mod = AutoModel.from_pretrained(model_name)
        _cached_models[model_name] = (tok, mod)
    return _cached_models[model_name]


# Module-level default instance
DEFAULT_BACKBONE = os.environ.get("BERT_MODEL", "answerdotai/ModernBERT-large")
tokenizer, model = get_bert(DEFAULT_BACKBONE)


def encode(text, model_name=None):
    """text -> mean-pooled (1, 1024) tensor."""
    tok, mod = get_bert(model_name) if model_name else (tokenizer, model)
    inputs = tok(text, return_tensors="pt")
    last_hidden = mod(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)


if __name__ == "__main__":
    print(encode("hello world").shape)  # (1, 1024)

