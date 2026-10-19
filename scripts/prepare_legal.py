#!/usr/bin/env python3
"""prepare_legal.py - Prepare Legal Reasoning & Statutory Logic Dataset for Track H.

Ingests:
1. Baseline Bar Exam & Statutory Logic:
   - cais/mmlu: professional_law, international_law, jurisprudence (1,577 train, 395 test)
2. Constitutional Law & Judicial Review:
   - cais/mmlu: high_school_government_and_politics (219 samples)
3. Statutory Construction & Legal Reasoning:
   - nguha/legalbench: scalr (571 samples)

Invariants:
- The held-out test split (395 samples) is preserved 100% byte-for-byte untouched,
  maintaining the Metric Invariant for all AutoResearch trials.
- Candidate choices strictly adhere to the canonical System2 4-choice deliberation schema.
- For 5-choice statutory interpretation (SCALR), distractor length-parity selection is
  applied to prevent token length bias artifacts.

Saves to ./data/legal as a DatasetDict on disk.
"""

import argparse
import os
import random
from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

OUTPUT_DIR = "./data/legal"
BASELINE_DIR = "/Users/thomjames-worrall/System2/data/legal"


def format_qa_text(question: str, choices: list[str]) -> str:
    """Format into canonical System2 4-choice deliberation prompt."""
    return (
        f"Question: {question.strip()}\n"
        f"(A) {choices[0].strip()}\n"
        f"(B) {choices[1].strip()}\n"
        f"(C) {choices[2].strip()}\n"
        f"(D) {choices[3].strip()}"
    )


def load_mmlu_gov_politics() -> list[dict]:
    """Ingest cais/mmlu high_school_government_and_politics (pure constitutional law)."""
    print("Loading cais/mmlu 'high_school_government_and_politics'...")
    ds = load_dataset("cais/mmlu", "high_school_government_and_politics")
    samples = []

    for split in ["test", "validation", "dev"]:
        if split not in ds:
            continue
        for ex in ds[split]:
            q = ex.get("question", "").strip()
            choices = ex.get("choices", [])
            ans = ex.get("answer")

            if not q or len(choices) != 4 or ans not in (0, 1, 2, 3):
                continue

            cleaned_choices = [c.strip() for c in choices]
            samples.append({
                "question": q,
                "choices": cleaned_choices,
                "text": format_qa_text(q, cleaned_choices),
                "label": int(ans),
                "source": "mmlu_high_school_government_and_politics",
            })

    print(f"  Ingested {len(samples)} constitutional law samples.")
    return samples


def load_legalbench_scalr() -> list[dict]:
    """Ingest nguha/legalbench scalr (statutory construction & legal reasoning).

    SCALR questions have 5 choices (1 correct holding + 4 distractors).
    We select the 3 distractors closest in length to the ground-truth holding
    to enforce distractor length parity and category homogeneity.
    """
    print("Loading nguha/legalbench 'scalr'...")
    ds = load_dataset("nguha/legalbench", "scalr", split="test")
    samples = []

    for ex in ds:
        q = ex.get("question", "").strip()
        choices = [ex.get(f"choice_{i}", "").strip() for i in range(5)]
        ans = ex.get("answer")

        if not q or any(not c for c in choices) or ans not in range(5):
            continue

        correct_len = len(choices[ans])
        wrong_indices = [i for i in range(5) if i != ans]

        # Select 3 distractors closest in character length to ground-truth choice
        selected_wrong = sorted(wrong_indices, key=lambda w: abs(len(choices[w]) - correct_len))[:3]
        kept_indices = sorted(selected_wrong + [ans])

        new_choices = [choices[i] for i in kept_indices]
        new_label = kept_indices.index(ans)

        samples.append({
            "question": q,
            "choices": new_choices,
            "text": format_qa_text(q, new_choices),
            "label": int(new_label),
            "source": "legalbench_scalr",
        })

    print(f"  Ingested {len(samples)} statutory interpretation (SCALR) samples.")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare System2 Legal Reasoning Dataset")
    parser.add_argument("--baseline-dir", default=BASELINE_DIR, help="Path to baseline legal dataset")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Path to write output dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train shuffle")
    args = parser.parse_args()

    print("================================================================")
    print(" PREPARING SYSTEM2 LEGAL REASONING DATASET (TRACK H)")
    print("================================================================")

    # 1. Load baseline dataset to preserve held-out test split
    print(f"Loading baseline dataset from {args.baseline_dir}...")
    baseline_ds = load_from_disk(args.baseline_dir)
    baseline_train = list(baseline_ds["train"])
    held_out_test = list(baseline_ds["test"])
    print(f"  Baseline Train: {len(baseline_train)} samples")
    print(f"  Held-Out Test:  {len(held_out_test)} samples (strictly preserved)")

    # 2. Ingest pure constitutional law & statutory interpretation
    gov_samples = load_mmlu_gov_politics()
    scalr_samples = load_legalbench_scalr()

    # 3. Check for any leakage into held-out test split
    test_questions = {ex["question"].strip().lower() for ex in held_out_test}
    gov_filtered = [s for s in gov_samples if s["question"].strip().lower() not in test_questions]
    scalr_filtered = [s for s in scalr_samples if s["question"].strip().lower() not in test_questions]

    if len(gov_filtered) < len(gov_samples):
        print(f"  Filtered {len(gov_samples) - len(gov_filtered)} gov samples overlapping with test set.")
    if len(scalr_filtered) < len(scalr_samples):
        print(f"  Filtered {len(scalr_samples) - len(scalr_filtered)} scalr samples overlapping with test set.")

    # 4. Merge all training samples
    all_train = baseline_train + gov_filtered + scalr_filtered
    print(f"\nTotal merged training samples: {len(all_train)}")
    print("Training source breakdown:", Counter(s["source"] for s in all_train))

    # Deterministic shuffle of training set
    random.seed(args.seed)
    random.shuffle(all_train)

    train_ds = Dataset.from_list(all_train)
    test_ds = Dataset.from_list(held_out_test)

    out_dict = DatasetDict({
        "train": train_ds,
        "test": test_ds,
    })

    # 5. Handle output path: if output-dir is a symlink, unlink it so main repo is untouched
    if os.path.islink(args.output_dir):
        print(f"\nTarget {args.output_dir} is a symlink pointing to {os.readlink(args.output_dir)}.")
        print("Unlinking symlink to keep main repo completely untouched...")
        os.unlink(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    out_dict.save_to_disk(args.output_dir)

    # 6. Verify written dataset
    verify_ds = load_from_disk(args.output_dir)
    print(f"\nVerified Track H dataset on disk at {args.output_dir}:")
    print(f"  Train: {len(verify_ds['train'])} samples")
    print(f"  Test:  {len(verify_ds['test'])} samples")
    assert len(verify_ds["test"]) == len(held_out_test), "Metric Invariant violation: test set size changed!"
    assert len(verify_ds["train"]) == len(all_train), "Train set size mismatch!"

    print("================================================================")
    print(" PREPARATION COMPLETE - HELD-OUT TEST SET UNTOUCHED")
    print("================================================================")


if __name__ == "__main__":
    main()
