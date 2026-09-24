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


if __name__ == "__main__":
    train, test = load_ag_news(train_size=5)
    print(f"train: {len(train)}, test: {len(test)}")
    print("sample:", train[0])
