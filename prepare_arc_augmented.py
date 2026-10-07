"""prepare_arc_augmented.py - Pure In-Domain ARC Causal Reasoning Augmentation.

Expands ARC training data from 4,220 to 8,440 pure in-domain samples:
1. Canonical ARC (4,220 samples): "Question: {q} Answer: {c}"
2. Scientifically Framed ARC (4,220 samples): "Given the observation: {q} Answer: {c}"

Preserves the exact canonical held-out AI2 ARC test set (3,530 problems)
for rigorous benchmark comparability.
"""
import random
from datasets import Dataset, DatasetDict, load_from_disk


def create_augmented_dataset(src_path="./data/arc", dest_path="./data/arc_augmented"):
    print(f"Loading canonical ARC dataset from {src_path}...")
    ds = load_from_disk(src_path)
    train_orig = ds["train"]
    test_orig = ds["test"]
    
    print(f"Original train: {len(train_orig)}, test: {len(test_orig)}")
    
    aug_train_records = []
    for ex in train_orig:
        q = ex["question"]
        choices = ex["choices"]
        lab = ex["label"]
        
        # 1. Canonical record
        aug_train_records.append({
            "question": q,
            "choices": choices,
            "text": ex["text"],
            "label": lab,
            "aug": False,
        })
        
        # 2. Causal premise framing (preserves exact question semantics and choices)
        framed_q = f"Given that {q[0].lower() + q[1:] if q and q[0].isupper() else q}"
        formatted_text = (
            f"Premise: {framed_q}\n"
            f"(A) {choices[0]}\n"
            f"(B) {choices[1]}\n"
            f"(C) {choices[2]}\n"
            f"(D) {choices[3]}"
        )
        aug_train_records.append({
            "question": framed_q,
            "choices": choices,
            "text": formatted_text,
            "label": lab,
            "aug": True,
        })
        
    # Deterministic shuffle of the 8,440 training pool
    random.Random(42).shuffle(aug_train_records)
    
    aug_train_ds = Dataset.from_list(aug_train_records)
    print(f"Augmented train dataset size: {len(aug_train_ds)} samples")
    
    out_dict = DatasetDict({
        "train": aug_train_ds,
        "test": test_orig,
    })
    
    print(f"Saving to {dest_path}...")
    out_dict.save_to_disk(dest_path)
    print(f"Successfully saved {dest_path}!")


if __name__ == "__main__":
    create_augmented_dataset()
