#!/usr/bin/env python3
"""prepare_legal_expanded.py - Prepare Expanded Legal Reasoning Dataset for Track H.

Combines:
1. TIGER-Lab/MMLU-Pro (category == 'law', 1,101 questions)
   - Downsampled from up to 10 choices to canonical 4 choices with length parity
2. nguyenkhanh87/CaseHOLD_DeepSeek_R1_14B_Reasoning (1,000 judicial holdings)
   - Downsampled from 5 choices to canonical 4 choices with length parity
3. Existing cais/mmlu legal datasets from data/legal (1,577 samples)
   - professional_law, international_law, jurisprudence

Enforces:
- Canonical 4-choice deliberation schema:
  - question: str
  - choices: list[str] (exactly 4 candidate legal interpretations)
  - text: str (formatted prompt with choices A-D)
  - label: int (0..3)
  - source: str
- Length parity: distractors selected by character length proximity to the gold answer
- Deduplication by normalized question text

Output: ./data/legal_expanded (train and test splits)
"""

import os
import re
import random
from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

OUTPUT_DIR = "./data/legal_expanded"


def select_length_balanced_choices(options, answer_index, rng):
    """Downsample candidate options to 4 choices with length parity."""
    correct_opt = str(options[answer_index]).strip()
    other_opts = [str(opt).strip() for i, opt in enumerate(options) if i != answer_index and str(opt).strip()]

    if len(other_opts) < 3:
        return None

    # Sort other options by proximity to correct option length
    target_len = len(correct_opt)
    other_opts.sort(key=lambda x: abs(len(x) - target_len))

    # Take the top distractors (closest in length)
    pool_size = min(len(other_opts), 5)
    selected_distractors = rng.sample(other_opts[:pool_size], 3)

    cands = [correct_opt] + selected_distractors
    rng.shuffle(cands)
    label = cands.index(correct_opt)

    return cands, label


def load_mmlu_pro_law(rng):
    """Load TIGER-Lab/MMLU-Pro law subset and format to 4 choices."""
    print("Loading TIGER-Lab/MMLU-Pro law...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    law_items = [x for x in ds if x.get("category") == "law"]
    print(f"Found {len(law_items)} raw MMLU-Pro law questions.")

    samples = []
    for item in law_items:
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
            "source": "mmlu_pro_law",
        })

    print(f"Processed {len(samples)} clean 4-choice MMLU-Pro law samples.")
    return samples


def load_casehold_reasoning(rng):
    """Load nguyenkhanh87/CaseHOLD_DeepSeek_R1_14B_Reasoning and format to 4 choices."""
    print("Loading CaseHOLD Reasoning...")
    try:
        ds = load_dataset("nguyenkhanh87/CaseHOLD_DeepSeek_R1_14B_Reasoning", split="train")
    except Exception as e:
        print(f"Warning: could not load CaseHOLD: {e}")
        return []

    print(f"Found {len(ds)} raw CaseHOLD items.")
    samples = []

    for item in ds:
        context = item.get("context", "").strip()
        endings = item.get("endings", [])
        gold_lbl = item.get("label")

        if not context or len(endings) != 5 or gold_lbl not in range(5):
            continue

        q = f"Which judicial holding best completes the following legal citation context?\n{context}"
        res = select_length_balanced_choices(endings, gold_lbl, rng)
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
            "source": "casehold_holding",
        })

    print(f"Processed {len(samples)} clean 4-choice CaseHOLD samples.")
    return samples


def load_existing_legal():
    """Load existing data/legal dataset if available."""
    samples = []
    if os.path.exists("./data/legal"):
        print("Loading existing ./data/legal...")
        try:
            ds = load_from_disk("./data/legal")
            for split in ["train", "test"]:
                if split in ds:
                    for ex in ds[split]:
                        samples.append({
                            "question": ex["question"].strip(),
                            "choices": [str(c).strip() for c in ex["choices"]],
                            "text": ex["text"].strip(),
                            "label": int(ex["label"]),
                            "source": ex.get("source", "mmlu_law_base"),
                        })
            print(f"Loaded {len(samples)} samples from existing ./data/legal.")
        except Exception as e:
            print(f"Warning: could not load existing ./data/legal: {e}")
    return samples


def main():
    print("================================================================")
    print(" PREPARING SYSTEM2 EXPANDED LEGAL DATASET (TRACK H)")
    print("================================================================")

    rng = random.Random(42)

    pro_samples = load_mmlu_pro_law(rng)
    case_samples = load_casehold_reasoning(rng)
    base_samples = load_existing_legal()

    combined = pro_samples + case_samples + base_samples
    print(f"\nTotal raw legal questions before deduplication: {len(combined)}")

    # Deduplicate by normalized question
    seen = set()
    deduped = []
    for s in combined:
        norm_q = re.sub(r"\s+", " ", s["question"].strip().lower())
        if norm_q not in seen:
            seen.add(norm_q)
            deduped.append(s)

    print(f"Unique legal questions after deduplication: {len(deduped)}")
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
    print(f">>> Successfully saved expanded legal dataset to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
