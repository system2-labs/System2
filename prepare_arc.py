"""prepare_arc.py - Prepare AI2 ARC scientific reasoning dataset.

Downloads allenai/ai2_arc (Challenge + Easy), standardizes questions
and 4 choices (A, B, C, D) into clean {text, label} records, and
saves to ./data/arc on disk.
"""
from datasets import Dataset, DatasetDict, load_dataset


def format_split(*splits):
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
    return Dataset.from_list(records)


def main():
    print("Loading allenai/ai2_arc from Hugging Face...")
    challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    easy = load_dataset("allenai/ai2_arc", "ARC-Easy")
    
    print("Formatting train split (Challenge Train/Val + Easy Train/Val)...")
    train_ds = format_split(
        challenge["train"],
        easy["train"],
        challenge["validation"],
        easy["validation"],
    )
    
    print("Formatting test split (Challenge + Easy)...")
    test_ds = format_split(challenge["test"], easy["test"])
    
    dataset_dict = DatasetDict({
        "train": train_ds,
        "test": test_ds
    })
    
    print(f"Prepared ARC dataset: {len(train_ds)} train samples, {len(test_ds)} test samples.")
    output_path = "./data/arc"
    dataset_dict.save_to_disk(output_path)
    print(f"Saved to {output_path} successfully!")


if __name__ == "__main__":
    main()
