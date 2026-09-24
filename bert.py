import torch
from transformers import AutoModel, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-large")


def encode(text):
    """text -> mean-pooled (1, 1024) tensor."""
    inputs = tokenizer(text, return_tensors="pt")
    last_hidden = model(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)


if __name__ == "__main__":
    print(encode("hello world").shape)  # (1, 1024)
