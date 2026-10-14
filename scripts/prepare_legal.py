#!/usr/bin/env python3
"""prepare_legal.py - Prepare Legal Reasoning & Statutory Logic Dataset for Track H.

Ingests cais/mmlu legal subjects:
- professional_law (Bar exam, torts, contracts, civil procedure)
- international_law
- jurisprudence

Formats into canonical System2 4-choice deliberation schema:
- question: str
- choices: list[str] (4 candidate legal interpretations)
- text: str (formatted prompt with choices A-D)
- label: int (0..3)
- source: str ("mmlu_law")

Saves to ./data/legal as a DatasetDict on disk.
"""

import os
import random
from datasets import Dataset, DatasetDict, load_dataset

LEGAL_SUBJECTS = ["professional_law", "international_law", "jurisprudence"]
OUTPUT_DIR = "./data/legal"


def main():
    print("================================================================")
    print(" PREPARING SYSTEM2 LEGAL REASONING DATASET (TRACK H)")
    print("================================================================")

    all_samples = []

    for subj in LEGAL_SUBJECTS:
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

    print(f"\nTotal legal reasoning samples collected: {len(all_samples)}")

    # Deterministic split: 80% train, 20% test
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
    print(f"Saved Track H dataset to {OUTPUT_DIR}:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    print("================================================================")


if __name__ == "__main__":
    main()
