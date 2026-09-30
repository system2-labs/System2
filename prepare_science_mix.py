"""prepare_science_mix.py - Scaled scientific reasoning dataset.

Combines:
1. AI2 ARC (Challenge + Easy): ~3,358 train samples
2. OpenBookQA:                 ~4,957 train samples
3. SciQ:                      ~11,679 train samples
Total Train:                  ~19,994 reasoning problems (~6x expansion)

Evaluation Set:
- Preserves the full held-out AI2 ARC test set (3,530 problems)
  to ensure strict comparability against Phase 2 Baselines (Runs 0 and 1).
"""
import random
from datasets import Dataset, DatasetDict, load_dataset


def prepare_arc_train():
    challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")["train"]
    easy = load_dataset("allenai/ai2_arc", "ARC-Easy")["train"]
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
    records = []
    
    for split in [challenge, easy]:
        for ex in split:
            key = ex.get("answerKey")
            choices = ex.get("choices", {})
            texts = choices.get("text", [])
            if key not in label_map or len(texts) != 4:
                continue
            q = ex["question"].strip()
            c_texts = [t.strip() for t in texts]
            formatted_text = f"Question: {q}\n(A) {c_texts[0]}\n(B) {c_texts[1]}\n(C) {c_texts[2]}\n(D) {c_texts[3]}"
            records.append({
                "question": q,
                "choices": c_texts,
                "text": formatted_text,
                "label": label_map[key],
                "source": "arc",
            })
    return records


def prepare_openbookqa_train():
    ds = load_dataset("allenai/openbookqa", "main")["train"]
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    records = []
    
    for ex in ds:
        key = ex.get("answerKey")
        choices = ex.get("choices", {})
        texts = choices.get("text", [])
        if key not in label_map or len(texts) != 4:
            continue
        q = ex["question_stem"].strip()
        c_texts = [t.strip() for t in texts]
        formatted_text = f"Question: {q}\n(A) {c_texts[0]}\n(B) {c_texts[1]}\n(C) {c_texts[2]}\n(D) {c_texts[3]}"
        records.append({
            "question": q,
            "choices": c_texts,
            "text": formatted_text,
            "label": label_map[key],
            "source": "openbookqa",
        })
    return records


def prepare_sciq_train(seed=42):
    ds = load_dataset("allenai/sciq")["train"]
    rng = random.Random(seed)
    records = []
    
    for ex in ds:
        q = ex["question"].strip()
        correct = ex["correct_answer"].strip()
        d1 = ex["distractor1"].strip()
        d2 = ex["distractor2"].strip()
        d3 = ex["distractor3"].strip()
        
        if not (q and correct and d1 and d2 and d3):
            continue
        
        c_texts = [correct, d1, d2, d3]
        rng.shuffle(c_texts)
        label = c_texts.index(correct)
        
        formatted_text = f"Question: {q}\n(A) {c_texts[0]}\n(B) {c_texts[1]}\n(C) {c_texts[2]}\n(D) {c_texts[3]}"
        records.append({
            "question": q,
            "choices": c_texts,
            "text": formatted_text,
            "label": label,
            "source": "sciq",
        })
    return records


def prepare_arc_test():
    challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    easy = load_dataset("allenai/ai2_arc", "ARC-Easy")["test"]
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
    records = []
    
    for split in [challenge, easy]:
        for ex in split:
            key = ex.get("answerKey")
            choices = ex.get("choices", {})
            texts = choices.get("text", [])
            if key not in label_map or len(texts) != 4:
                continue
            q = ex["question"].strip()
            c_texts = [t.strip() for t in texts]
            formatted_text = f"Question: {q}\n(A) {c_texts[0]}\n(B) {c_texts[1]}\n(C) {c_texts[2]}\n(D) {c_texts[3]}"
            records.append({
                "question": q,
                "choices": c_texts,
                "text": formatted_text,
                "label": label_map[key],
                "source": "arc",
            })
    return records


def main():
    print("Preparing ARC train records...")
    arc_records = prepare_arc_train()
    print(f"ARC train: {len(arc_records)}")

    print("Preparing OpenBookQA train records...")
    obqa_records = prepare_openbookqa_train()
    print(f"OpenBookQA train: {len(obqa_records)}")

    print("Preparing SciQ train records...")
    sciq_records = prepare_sciq_train()
    print(f"SciQ train: {len(sciq_records)}")

    all_train = arc_records + obqa_records + sciq_records
    # Shuffle combined dataset
    random.Random(42).shuffle(all_train)
    print(f"Total Combined Train: {len(all_train)}")

    print("Preparing held-out ARC test records...")
    arc_test = prepare_arc_test()
    print(f"Held-out ARC test: {len(arc_test)}")

    dataset_dict = DatasetDict({
        "train": Dataset.from_list(all_train),
        "test": Dataset.from_list(arc_test),
    })

    out_dir = "./data/science_mix"
    print(f"Saving combined dataset to {out_dir}...")
    dataset_dict.save_to_disk(out_dir)
    print(f"Successfully saved {out_dir}!")


if __name__ == "__main__":
    main()
