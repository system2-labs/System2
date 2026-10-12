"""prepare_medical.py - Prepare MedQA USMLE 4-options dataset for System2 Track C.

Ingests GBaker/MedQA-USMLE-4-options:
- Train: 10,178 real USMLE licensing exam clinical vignettes
- Test:  1,273 held-out board examination diagnostic problems

Standardizes into the System2 feedforward deliberation schema:
- question: str
- choices: list[str] (4 candidate diagnostic/treatment options)
- text: str (formatted prompt with choices A-D)
- label: int (0..3)
- source: str ("medqa_usmle")

Saves to ./data/medical as a DatasetDict on disk.
"""

from datasets import Dataset, DatasetDict, load_dataset


def format_split(split_data):
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    records = []

    for ex in split_data:
        ans_idx = ex.get("answer_idx")
        opts = ex.get("options", {})
        if ans_idx not in label_map or len(opts) != 4:
            continue

        q = ex["question"].strip()
        c_a = opts.get("A", "").strip()
        c_b = opts.get("B", "").strip()
        c_c = opts.get("C", "").strip()
        c_d = opts.get("D", "").strip()

        if not (c_a and c_b and c_c and c_d):
            continue

        choices = [c_a, c_b, c_c, c_d]
        formatted_text = (
            f"Question: {q}\n"
            f"(A) {choices[0]}\n"
            f"(B) {choices[1]}\n"
            f"(C) {choices[2]}\n"
            f"(D) {choices[3]}"
        )
        records.append({
            "question": q,
            "choices": choices,
            "text": formatted_text,
            "label": label_map[ans_idx],
            "source": "medqa_usmle"
        })

    return Dataset.from_list(records)


def main():
    print("================================================================")
    print("PREPARING SYSTEM2 MEDICAL REASONING DATASET (TRACK C)")
    print("================================================================")
    print("Loading GBaker/MedQA-USMLE-4-options from Hugging Face...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")

    print(f"Formatting train split ({len(ds['train'])} raw records)...")
    train_ds = format_split(ds["train"])

    print(f"Formatting held-out test split ({len(ds['test'])} raw records)...")
    test_ds = format_split(ds["test"])

    out_dict = DatasetDict({
        "train": train_ds,
        "test": test_ds
    })

    output_dir = "./data/medical"
    print("\n----------------------------------------------------------------")
    print(f"Prepared Track C Medical Dataset:")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Test samples:  {len(test_ds)}")
    print(f"Saving to {output_dir}...")
    out_dict.save_to_disk(output_dir)
    print(f"Successfully saved {output_dir}!")
    print("================================================================")


if __name__ == "__main__":
    main()
