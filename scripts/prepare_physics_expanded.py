#!/usr/bin/env python3
"""prepare_physics_expanded.py - Prepare Expanded Pure Physics & Astrophysics Dataset for Track J.

Combines:
1. TIGER-Lab/MMLU-Pro (category == 'physics', 1,299 questions)
   - Downsampled from up to 10 choices to canonical 4 choices with length parity
2. daman1209arora/jeebench (MCQ single-correct physics questions)
   - Clean LaTeX options parsed to canonical 4 choices
3. Existing cais/mmlu physics datasets from data/physics (692 unique questions)

Enforces:
- Canonical 4-choice deliberation schema:
  - question: str
  - choices: list[str] (exactly 4 candidate answers)
  - text: str (formatted prompt with choices A-D)
  - label: int (0..3)
  - source: str
- Length parity: distractors selected by character length proximity to the gold answer
- Deduplication by normalized question text

Output: ./data/physics_expanded (train and test splits)
"""

import os
import re
import random
from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

OUTPUT_DIR = "./data/physics_expanded"


def select_length_balanced_choices(options, answer_index, rng):
    """Downsample candidate options to 4 choices with length parity."""
    correct_opt = options[answer_index].strip()
    other_opts = [opt.strip() for i, opt in enumerate(options) if i != answer_index and opt.strip()]

    if len(other_opts) < 3:
        return None

    # Sort other options by proximity to correct option length
    target_len = len(correct_opt)
    other_opts.sort(key=lambda x: abs(len(x) - target_len))

    # Take the top distractors (closest in length)
    pool_size = min(len(other_opts), 6)
    selected_distractors = rng.sample(other_opts[:pool_size], 3)

    cands = [correct_opt] + selected_distractors
    rng.shuffle(cands)
    label = cands.index(correct_opt)

    return cands, label


def load_mmlu_pro_physics(rng):
    """Load TIGER-Lab/MMLU-Pro physics subset and format to 4 choices."""
    print("Loading TIGER-Lab/MMLU-Pro physics...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    physics_items = [x for x in ds if x.get("category") == "physics"]
    print(f"Found {len(physics_items)} raw MMLU-Pro physics questions.")

    samples = []
    for item in physics_items:
        q = item.get("question", "").strip()
        options = item.get("options", [])
        ans_idx = item.get("answer_index")

        if not q or not options or ans_idx is None or ans_idx >= len(options):
            continue

        res = select_length_balanced_choices(options, ans_idx, rng)
        if res is None:
            continue

        cands, label = res
        text = (
            f"Question: {q}\n"
            f"(A) {cands[0]}\n"
            f"(B) {cands[1]}\n"
            f"(C) {cands[2]}\n"
            f"(D) {cands[3]}"
        )
        samples.append({
            "question": q,
            "choices": cands,
            "text": text,
            "label": label,
            "source": "mmlu_pro_physics",
        })

    print(f"Processed {len(samples)} clean 4-choice MMLU-Pro physics samples.")
    return samples


def load_jeebench_physics():
    """Load daman1209arora/jeebench single-choice physics MCQs."""
    print("Loading daman1209arora/jeebench physics...")
    try:
        ds = load_dataset("daman1209arora/jeebench", split="test")
    except Exception as e:
        print(f"Warning: could not load jeebench: {e}")
        return []

    phy_mcqs = [x for x in ds if x.get("subject") == "phy" and x.get("type") == "MCQ"]
    print(f"Found {len(phy_mcqs)} single-choice physics MCQs in JEEBench.")

    samples = []
    option_pattern = re.compile(r"\((A|B|C|D)\)\s*(.*?)(?=\((?:A|B|C|D)\)|$)", re.DOTALL)

    for item in phy_mcqs:
        raw_q = item.get("question", "").strip()
        gold = item.get("gold", "").strip().upper()

        if gold not in ("A", "B", "C", "D"):
            continue

        matches = option_pattern.findall(raw_q)
        if len(matches) != 4:
            continue

        labels_found = [m[0] for m in matches]
        if labels_found != ["A", "B", "C", "D"]:
            continue

        # Extract question stem (everything before (A))
        q_stem = raw_q.split("(A)")[0].strip()
        choices = [m[1].strip() for m in matches]

        if not q_stem or any(len(c) == 0 for c in choices):
            continue

        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        label = label_map[gold]

        text = (
            f"Question: {q_stem}\n"
            f"(A) {choices[0]}\n"
            f"(B) {choices[1]}\n"
            f"(C) {choices[2]}\n"
            f"(D) {choices[3]}"
        )
        samples.append({
            "question": q_stem,
            "choices": choices,
            "text": text,
            "label": label,
            "source": "jeebench_physics",
        })

    print(f"Processed {len(samples)} clean JEEBench physics samples.")
    return samples


def load_existing_physics():
    """Load existing data/physics dataset if available."""
    samples = []
    if os.path.exists("./data/physics"):
        print("Loading existing ./data/physics...")
        try:
            ds = load_from_disk("./data/physics")
            for split in ["train", "test"]:
                if split in ds:
                    for ex in ds[split]:
                        samples.append({
                            "question": ex["question"].strip(),
                            "choices": [str(c).strip() for c in ex["choices"]],
                            "text": ex["text"].strip(),
                            "label": int(ex["label"]),
                            "source": ex.get("source", "mmlu_physics_base"),
                        })
            print(f"Loaded {len(samples)} samples from existing ./data/physics.")
        except Exception as e:
            print(f"Warning: could not load existing ./data/physics: {e}")
    return samples


def main():
    print("================================================================")
    print(" PREPARING SYSTEM2 EXPANDED PHYSICS DATASET (TRACK J)")
    print("================================================================")

    rng = random.Random(42)

    pro_samples = load_mmlu_pro_physics(rng)
    jee_samples = load_jeebench_physics()
    base_samples = load_existing_physics()

    combined = pro_samples + jee_samples + base_samples
    print(f"\nTotal raw physics questions before deduplication: {len(combined)}")

    # Deduplicate by normalized question
    seen = set()
    deduped = []
    for s in combined:
        norm_q = re.sub(r"\s+", " ", s["question"].strip().lower())
        if norm_q not in seen:
            seen.add(norm_q)
            deduped.append(s)

    print(f"Unique physics questions after deduplication: {len(deduped)}")
    print("Source distribution:", dict(Counter(s["source"] for s in deduped)))

    # Deterministic 80/20 train/test split
    rng.shuffle(deduped)
    split_idx = int(len(deduped) * 0.8)
    train_data = deduped[:split_idx]
    test_data = deduped[split_idx:]

    print(f"\nSplit: {len(train_data)} train | {len(test_data)} test")

    ds_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data),
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds_dict.save_to_disk(OUTPUT_DIR)
    print(f">>> Successfully saved expanded physics dataset to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
