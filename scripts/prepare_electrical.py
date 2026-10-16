#!/usr/bin/env python3
"""prepare_electrical.py - Prepare Augmented Electrical & Power Systems Dataset for Track D.

Combines EEReasonBench with cais/mmlu subjects:
- electrical_engineering (circuit theory, three-phase, transformers, machines)
- college_physics & conceptual_physics (electromagnetism, circuits, AC power)
- high_school_physics (KVL/KCL, basic electronics, Ohm's law)

Formats into canonical System2 4-choice deliberation schema:
- question: str
- choices: list[str] (4 candidate electrical engineering answers)
- text: str (formatted prompt with choices A-D)
- label: int (0..3)
- source: str ("eereason" or "mmlu_ee")

Saves to ./data/electrical as a DatasetDict on disk.
"""

import os
import random
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset

MMLU_EE_SUBJECTS = [
    "electrical_engineering",
    "college_physics",
    "conceptual_physics",
    "high_school_physics",
]
OUTPUT_DIR = "./data/electrical"


def main():
    print("================================================================")
    print(" PREPARING SYSTEM2 AUGMENTED ELECTRICAL DATASET (TRACK D)")
    print("================================================================")

    all_samples = []

    # 1. Load existing base electrical dataset if present
    if os.path.exists(OUTPUT_DIR):
        try:
            base_ds = load_from_disk(OUTPUT_DIR)
            for split in ["train", "test"]:
                if split in base_ds:
                    for ex in base_ds[split]:
                        all_samples.append({
                            "question": ex["question"],
                            "choices": ex["choices"],
                            "text": ex["text"],
                            "label": ex["label"],
                            "source": ex.get("source", "eereason"),
                        })
            print(f"Loaded {len(all_samples)} existing samples from {OUTPUT_DIR}")
        except Exception as e:
            print(f"Warning: Could not read existing {OUTPUT_DIR}: {e}")

    # 2. Ingest MMLU electrical engineering and physics datasets
    mmlu_added = 0
    for subj in MMLU_EE_SUBJECTS:
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
                mmlu_added += 1

    print(f"Ingested {mmlu_added} MMLU electrical/physics questions.")
    print(f"Total raw dataset size: {len(all_samples)}")

    # Deduplicate by question text
    seen_q = set()
    deduped = []
    for s in all_samples:
        clean_q = s["question"].strip().lower()
        if clean_q not in seen_q:
            seen_q.add(clean_q)
            deduped.append(s)

    print(f"Unique questions after deduplication: {len(deduped)}")

    # Deterministic split 80% train, 20% test
    random.seed(42)
    random.shuffle(deduped)

    split_idx = int(len(deduped) * 0.8)
    train_data = deduped[:split_idx]
    test_data = deduped[split_idx:]

    print(f"Train split: {len(train_data)} samples")
    print(f"Test split:  {len(test_data)} samples")

    # Build HF DatasetDict
    train_dict = {
        "question": [d["question"] for d in train_data],
        "choices": [d["choices"] for d in train_data],
        "text": [d["text"] for d in train_data],
        "label": [d["label"] for d in train_data],
        "source": [d["source"] for d in train_data],
    }
    test_dict = {
        "question": [d["question"] for d in test_data],
        "choices": [d["choices"] for d in test_data],
        "text": [d["text"] for d in test_data],
        "label": [d["label"] for d in test_data],
        "source": [d["source"] for d in test_data],
    }

    ds_dict = DatasetDict({
        "train": Dataset.from_dict(train_dict),
        "test": Dataset.from_dict(test_dict),
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds_dict.save_to_disk(OUTPUT_DIR)
    print(f"\nSuccessfully saved augmented electrical dataset to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
