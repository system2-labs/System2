import json
from datasets import load_from_disk


def load_ag_news(path="./data/ag_news", train_size=None, test_size=7600):
    """Load AG News. Returns (train_dataset, test_dataset) as raw dict lists.

    train_size: cap on training rows (keeps CPU training tractable).
    """
    ds = load_from_disk(path)
    train = [{"text": ex["text"], "label": ex["label"]} for ex in ds["train"]]
    if train_size:
        train = train[:train_size]
    test = [{"text": ex["text"], "label": ex["label"]} for ex in ds["test"]]
    return train, test


def load_arc(path="./data/arc", train_size=None):
    """Load AI2 ARC scientific reasoning dataset. Returns (train, test)."""
    ds = load_from_disk(path)
    train = [
        {"question": ex.get("question"), "choices": ex.get("choices"), "text": ex["text"], "label": ex["label"]}
        for ex in ds["train"]
    ]
    if train_size:
        train = train[:train_size]
    test = [
        {"question": ex.get("question"), "choices": ex.get("choices"), "text": ex["text"], "label": ex["label"]}
        for ex in ds["test"]
    ]
    return train, test


def load_science_mix(path="./data/science_mix", train_size=None):
    """Load combined scientific reasoning mix (ARC + OpenBookQA + SciQ)."""
    ds = load_from_disk(path)
    train = [
        {"question": ex.get("question"), "choices": ex.get("choices"), "text": ex["text"], "label": ex["label"]}
        for ex in ds["train"]
    ]
    if train_size:
        train = train[:train_size]
    test = [
        {"question": ex.get("question"), "choices": ex.get("choices"), "text": ex["text"], "label": ex["label"]}
        for ex in ds["test"]
    ]
    return train, test


def load_arc_mmlu_science(path="./data/arc_mmlu_science", train_size=None):
    """Load combined ARC + MMLU pure STEM science reasoning dataset."""
    ds = load_from_disk(path)
    train = [
        {"question": ex.get("question"), "choices": ex.get("choices"), "text": ex["text"], "label": ex["label"], "source": ex.get("source")}
        for ex in ds["train"]
    ]
    if train_size:
        train = train[:train_size]
    test = [
        {"question": ex.get("question"), "choices": ex.get("choices"), "text": ex["text"], "label": ex["label"], "source": ex.get("source")}
        for ex in ds["test"]
    ]
    return train, test


if __name__ == "__main__":
    train, test = load_arc(train_size=5)
    print(f"ARC train: {len(train)}, test: {len(test)}")
    print("sample:\n", train[0]["text"])
    print("label:", train[0]["label"])
