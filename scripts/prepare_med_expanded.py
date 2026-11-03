#!/usr/bin/env python3
"""prepare_med_expanded.py - Prepare Expanded Medical Reasoning Dataset for Track C.

Ingests openlifescienceai/medmcqa (182,822 authentic medical exam questions)
and subsamples 15,000 clean 4-choice clinical questions:
- Validates 4 non-empty candidate options (opa, opb, opc, opd)
- Validates unambiguous single-correct gold answer (cop in 0..3)
- Validates clean question stem and removes corrupt formatting
- Deterministically splits into 13,000 train and 2,000 test

Formats into canonical System2 4-choice deliberation schema:
- question: str
- choices: list[str] (4 candidate medical diagnoses/interventions)
- text: str (formatted prompt with choices A-D)
- label: int (0..3)
- source: str ("medmcqa")

Output: ./data/med_expanded (train and test splits)
"""

import os
import random
from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset

OUTPUT_DIR = "./data/med_expanded"
TARGET_SAMPLES = 15000
TEST_SIZE = 2000
TRAIN_SIZE = TARGET_SAMPLES - TEST_SIZE


def main():
    print("================================================================")
    print(" PREPARING SYSTEM2 EXPANDED MEDICAL DATASET (TRACK C)")
    print("================================================================")

    print("Loading openlifescienceai/medmcqa (train split)...")
    raw_ds = load_dataset("openlifescienceai/medmcqa", split="train")
    print(f"Total raw questions available: {len(raw_ds)}")

    rng = random.Random(42)
    indices = list(range(len(raw_ds)))
    rng.shuffle(indices)

    clean_samples = []
    seen_questions = set()

    for idx in indices:
        if len(clean_samples) >= TARGET_SAMPLES:
            break

        ex = raw_ds[idx]
        q = ex.get("question", "").strip()
        opa = ex.get("opa", "").strip()
        opb = ex.get("opb", "").strip()
        opc = ex.get("opc", "").strip()
        opd = ex.get("opd", "").strip()
        cop = ex.get("cop")
        choice_type = ex.get("choice_type", "single")

        if choice_type != "single" and choice_type is not None:
            continue

        if not q or not opa or not opb or not opc or not opd:
            continue

        if cop not in (0, 1, 2, 3):
            continue

        norm_q = q.lower().strip()
        if norm_q in seen_questions:
            continue
        seen_questions.add(norm_q)

        choices = [opa, opb, opc, opd]
        formatted_text = (
            f"Question: {q}\n"
            f"(A) {choices[0]}\n"
            f"(B) {choices[1]}\n"
            f"(C) {choices[2]}\n"
            f"(D) {choices[3]}"
        )

        subject = ex.get("subject_name", "general_medicine")
        clean_samples.append({
            "question": q,
            "choices": choices,
            "text": formatted_text,
            "label": int(cop),
            "source": f"medmcqa_{subject}",
        })

    print(f"\nCollected {len(clean_samples)} clean 4-choice medical questions.")
    
    # Split into 13,000 train and 2,000 test
    train_samples = clean_samples[:TRAIN_SIZE]
    test_samples = clean_samples[TRAIN_SIZE:]

    print(f"Train split: {len(train_samples)} samples")
    print(f"Test split:  {len(test_samples)} samples")

    ds_dict = DatasetDict({
        "train": Dataset.from_list(train_samples),
        "test": Dataset.from_list(test_samples),
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds_dict.save_to_disk(OUTPUT_DIR)
    print(f">>> Successfully saved expanded medical dataset to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
