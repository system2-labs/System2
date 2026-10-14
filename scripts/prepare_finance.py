#!/usr/bin/env python3
"""prepare_finance.py - Prepare Financial Quantitative & Economic Reasoning Dataset for Track I.

Ingests cais/mmlu financial & economic subjects:
- professional_accounting (balance sheets, GAAP, taxation, depreciation)
- econometrics (time series, regression, hypothesis testing)
- high_school_macroeconomics & microeconomics

Formats into canonical System2 4-choice deliberation schema:
- question: str
- choices: list[str] (4 candidate options)
- text: str (formatted prompt with choices A-D)
- label: int (0..3)
- source: str ("mmlu_finance")

Saves to ./data/finance as a DatasetDict on disk.
"""

import os
import random
from datasets import Dataset, DatasetDict, load_dataset

FINANCE_SUBJECTS = [
    "professional_accounting",
    "econometrics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "management",
]
OUTPUT_DIR = "./data/finance"


def main():
    print("================================================================")
    print(" PREPARING SYSTEM2 FINANCIAL REASONING DATASET (TRACK I)")
    print("================================================================")

    all_samples = []

    for subj in FINANCE_SUBJECTS:
        print(f"Loading cais/mmlu '{subj}'...")
        ds = load_dataset("cais/mmlu", subj)

        for split in ["test", "validation", "dev"]:
            if split not in ds:
                continue
            for ex in ds[split]:
                q = ex.get("question", "").strip()
                choices = ex.get("choices", [])
                ans = ex.get("answer")

                if not q or len(choices) != 4 or ans not in (0, 1, 2, 3):
                    continue

                formatted_text = (
                    f"Question: {q}\n"
                    f"(A) {choices[0]}\n"
                    f"(B) {choices[1]}\n"
                    f"(C) {choices[2]}\n"
                    f"(D) {choices[3]}"
                )

                all_samples.append({
                    "question": q,
                    "choices": choices,
                    "text": formatted_text,
                    "label": int(ans),
                    "source": f"mmlu_{subj}",
                })

    print(f"\nTotal financial reasoning samples collected: {len(all_samples)}")

    random.seed(42)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]

    train_ds = Dataset.from_list(train_samples)
    test_ds = Dataset.from_list(test_samples)

    out_dict = DatasetDict({
        "train": train_ds,
        "test": test_ds,
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_dict.save_to_disk(OUTPUT_DIR)
    print(f"Saved Track I dataset to {OUTPUT_DIR}:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    print("================================================================")


if __name__ == "__main__":
    main()
