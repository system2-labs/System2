"""prepare_arc_mmlu_science.py - Prepare ARC + MMLU Pure High-School/Conceptual Science.

Combines canonical ARC train/val (4,220 samples) with 1,016 pure 4-choice natural science
questions from MMLU (conceptual physics, high school physics, biology, and chemistry).
Preserves the canonical 3,530 held-out ARC test problems 100% untouched.
"""
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import os


def format_arc_split(*splits):
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
    records = []
    
    for split_data in splits:
        for ex in split_data:
            key = ex.get("answerKey")
            choices = ex.get("choices", {})
            texts = choices.get("text", [])
            
            if key not in label_map or len(texts) != 4:
                continue
            
            q = ex["question"].strip()
            formatted_text = (
                f"Question: {q}\n"
                f"(A) {texts[0]}\n"
                f"(B) {texts[1]}\n"
                f"(C) {texts[2]}\n"
                f"(D) {texts[3]}"
            )
            records.append({
                "question": q,
                "choices": [texts[0], texts[1], texts[2], texts[3]],
                "text": formatted_text,
                "label": label_map[key]
            })
    return records


def format_mmlu_science():
    sub_configs = [
        "conceptual_physics",
        "high_school_physics",
        "high_school_biology",
        "high_school_chemistry",
    ]
    records = []
    for sub in sub_configs:
        for split in ["test", "validation", "dev"]:
            ds = load_dataset("cais/mmlu", sub, split=split)
            for ex in ds:
                choices = ex["choices"]
                if len(choices) != 4:
                    continue
                q = ex["question"].strip()
                label = ex["answer"]
                if label not in [0, 1, 2, 3]:
                    continue
                formatted_text = (
                    f"Question: {q}\n"
                    f"(A) {choices[0]}\n"
                    f"(B) {choices[1]}\n"
                    f"(C) {choices[2]}\n"
                    f"(D) {choices[3]}"
                )
                records.append({
                    "question": q,
                    "choices": [choices[0], choices[1], choices[2], choices[3]],
                    "text": formatted_text,
                    "label": label
                })
    return records


def main():
    print("Loading allenai/ai2_arc...")
    challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    easy = load_dataset("allenai/ai2_arc", "ARC-Easy")
    
    arc_train_records = format_arc_split(
        challenge["train"],
        easy["train"],
        challenge["validation"],
        easy["validation"],
    )
    print(f"Loaded {len(arc_train_records)} ARC train records.")
    
    mmlu_science_records = format_mmlu_science()
    print(f"Loaded {len(mmlu_science_records)} MMLU pure science records.")
    
    # Format canonical test set
    arc_test_records = format_arc_split(challenge["test"], easy["test"])
    test_questions = set(r["question"].strip().lower() for r in arc_test_records)
    print(f"Canonical ARC test set has {len(arc_test_records)} problems.")
    
    # Filter MMLU records to guarantee 0% contamination with ARC test
    clean_mmlu = []
    for r in mmlu_science_records:
        if r["question"].strip().lower() not in test_questions:
            clean_mmlu.append(r)
        else:
            print("Filtered overlapping question:", r["question"])
    print(f"Clean MMLU science records: {len(clean_mmlu)}")
    
    combined_train = arc_train_records + clean_mmlu
    print(f"Combined train set size: {len(combined_train)} samples (+{len(clean_mmlu)/len(arc_train_records)*100:.1f}%)")
    
    train_ds = Dataset.from_list(combined_train)
    test_ds = Dataset.from_list(arc_test_records)
    
    output_dir = "./data/arc_mmlu_science"
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict = DatasetDict({
        "train": train_ds,
        "test": test_ds
    })
    dataset_dict.save_to_disk(output_dir)
    print(f"Successfully saved clean dataset to {output_dir}!")


if __name__ == "__main__":
    main()
