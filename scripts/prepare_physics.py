#!/usr/bin/env python3
"""prepare_physics.py - Prepare Pure Physics & Astrophysics Dataset for Track J (Physics).

Following the In-Domain Causal Purity Law, Track J isolates formal and conceptual
physics (previously evicted from Track A ARC and Track D Vocational Electrical) into
its own dedicated manifold:
- cais/mmlu 'college_physics' (classical mechanics, electromagnetism, thermodynamics, quantum)
- cais/mmlu 'high_school_physics' (kinematics, dynamics, energy, waves, circuits)
- cais/mmlu 'conceptual_physics' (physical principles, conservation laws, relativity)
- cais/mmlu 'astronomy' (orbital mechanics, astrophysics, cosmology, spectroscopy)

Formats into canonical System2 4-choice deliberation schema:
- question: str
- choices: list[str] (4 candidate answers)
- text: str (formatted prompt with choices A-D)
- label: int (0..3)
- source: str ("mmlu_college_physics", etc.)

Saves to ./data/physics as a DatasetDict (80% train / 20% test) on disk.
"""

import os
import random
from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset

PHYSICS_SUBJECTS = [
    "college_physics",
    "high_school_physics",
    "conceptual_physics",
    "astronomy",
]
OUTPUT_DIR = "./data/physics"


def main():
    print("================================================================")
    print(" PREPARING SYSTEM2 PURE PHYSICS DATASET (TRACK J)")
    print("================================================================")

    all_samples = []

    for subj in PHYSICS_SUBJECTS:
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

                cleaned_choices = [str(c).strip() for c in choices]
                formatted_text = (
                    f"Question: {q}\n"
                    f"(A) {cleaned_choices[0]}\n"
                    f"(B) {cleaned_choices[1]}\n"
                    f"(C) {cleaned_choices[2]}\n"
                    f"(D) {cleaned_choices[3]}"
                )

                all_samples.append({
                    "question": q,
                    "choices": cleaned_choices,
                    "text": formatted_text,
                    "label": int(ans),
                    "source": f"mmlu_{subj}",
                })

    print(f"Total raw physics questions collected: {len(all_samples)}")

    # Deduplicate by normalized question text
    seen_q = set()
    deduped = []
    for s in all_samples:
        clean_q = s["question"].strip().lower()
        if clean_q not in seen_q:
            seen_q.add(clean_q)
            deduped.append(s)

    print(f"Unique questions after deduplication: {len(deduped)}")
    print("Source breakdown:", dict(Counter(s["source"] for s in deduped)))

    # Deterministic 80/20 train/test split
    random.seed(42)
    random.shuffle(deduped)

    split_idx = int(len(deduped) * 0.8)
    train_data = deduped[:split_idx]
    test_data = deduped[split_idx:]

    print(f"Train split: {len(train_data)} samples | sources: {dict(Counter(s['source'] for s in train_data))}")
    print(f"Test split:  {len(test_data)} samples | sources: {dict(Counter(s['source'] for s in test_data))}")

    ds_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data),
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds_dict.save_to_disk(OUTPUT_DIR)
    print(f"\nSuccessfully saved pure Physics dataset (Track J) to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
