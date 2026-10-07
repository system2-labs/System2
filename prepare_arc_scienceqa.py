"""prepare_arc_scienceqa.py - Pure Curated Elementary & Middle School Science Dataset.

Combines:
1. AI2 ARC (Challenge Train/Val + Easy Train/Val): 4,220 in-domain problems
2. ScienceQA (derek-thomas/ScienceQA): ~696 text-only 4-choice natural science problems
   (Middle school physics, chemistry, biology, earth science)

Evaluation Set:
- Preserves the canonical held-out AI2 ARC test set (3,530 problems)
  to ensure strict comparability against all Phase 2 Baselines and SOTA (.best_metric = 0.6711).
"""
from datasets import Dataset, DatasetDict, load_dataset


def prepare_arc_records():
    challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    easy = load_dataset("allenai/ai2_arc", "ARC-Easy")
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
    records = []

    for split_data in [challenge["train"], easy["train"], challenge["validation"], easy["validation"]]:
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
                "label": label_map[key],
            })
    return records


def prepare_scienceqa_records():
    records = []
    for split_name in ["train", "validation"]:
        ds = load_dataset("derek-thomas/ScienceQA", split=split_name)
        for ex in ds:
            if ex.get("image") is not None:
                continue
            if ex.get("subject") != "natural science":
                continue
            choices = ex.get("choices", [])
            if len(choices) != 4:
                continue
            ans = ex.get("answer")
            if ans not in [0, 1, 2, 3]:
                continue

            q = ex["question"].strip()
            # If hint exists, include it cleanly in question context
            hint = ex.get("hint", "").strip()
            if hint:
                full_q = f"{q} {hint}"
            else:
                full_q = q

            c_texts = [str(c).strip() for c in choices]
            formatted_text = (
                f"Question: {full_q}\n"
                f"(A) {c_texts[0]}\n"
                f"(B) {c_texts[1]}\n"
                f"(C) {c_texts[2]}\n"
                f"(D) {c_texts[3]}"
            )
            records.append({
                "question": full_q,
                "choices": c_texts,
                "text": formatted_text,
                "label": int(ans),
            })
    return records


def main():
    print("Loading in-domain ARC train & validation...")
    arc_records = prepare_arc_records()
    print(f"ARC records: {len(arc_records)}")

    print("Loading ScienceQA natural science records...")
    scienceqa_records = prepare_scienceqa_records()
    print(f"ScienceQA natural science records: {len(scienceqa_records)}")

    combined_train = arc_records + scienceqa_records
    print(f"Combined train size: {len(combined_train)}")

    print("Loading canonical held-out ARC test set (3,530 questions)...")
    from datasets import load_from_disk
    canonical_arc = load_from_disk("./data/arc")
    test_ds = canonical_arc["test"]
    print(f"Canonical test set verified: {len(test_ds)} questions.")

    train_ds = Dataset.from_list(combined_train)

    dataset_dict = DatasetDict({
        "train": train_ds,
        "test": test_ds,
    })

    output_path = "./data/arc_scienceqa"
    dataset_dict.save_to_disk(output_path)
    print(f"Successfully saved combined dataset to {output_path}!")


if __name__ == "__main__":
    main()
