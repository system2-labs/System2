#!/usr/bin/env python3
"""prepare_math_dataset.py - Build in-domain multi-step mathematical reasoning dataset (GSM8K).

Converts raw GSM8K into System2 4-choice format:
- Ground truth: Complete step-by-step reasoning trace + exact final answer.
- Distractor 1: Premature stopping / intermediate step truncation.
- Distractor 2: Arithmetic calculation slip (perturbed final calculation).
- Distractor 3: Operational / magnitude error.

Saves: data/math (train: 7,473, test: 1,319).
"""

import os
import re
import sys
import random
from typing import Dict, List, Any, Tuple
from datasets import load_dataset, Dataset, DatasetDict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "math")


def clean_trace(raw_answer: str) -> Tuple[str, str]:
    """Clean GSM8K raw trace: remove internal <<...>> tags and extract final answer."""
    # Extract final answer after ####
    final_match = re.search(r"####\s*(-?[0-9,.]+)", raw_answer)
    final_ans = final_match.group(1).replace(",", "").strip() if final_match else ""

    # Remove <<calc>> tags, leaving the evaluated result
    cleaned = re.sub(r"<<.*?=([0-9,.]+)>>", r"\1", raw_answer)
    cleaned = re.sub(r"####\s*.*", f"Therefore, the final answer is {final_ans}.", cleaned).strip()
    return cleaned, final_ans


def extract_intermediates(raw_answer: str) -> List[str]:
    """Extract intermediate values from calculation tags."""
    matches = re.findall(r"<<.*?=([0-9,.]+)>>", raw_answer)
    return [m.replace(",", "").strip() for m in matches]


def generate_distractors(cleaned_trace: str, final_ans: str, intermediates: List[str], rng: random.Random) -> List[str]:
    """Generate 3 plausible, mathematically grounded distractors."""
    distractors = set()

    # Parse numeric value
    is_float = "." in final_ans
    try:
        val = float(final_ans) if is_float else int(final_ans)
    except ValueError:
        val = 100

    # Distractor 1: Intermediate step conclusion
    valid_intermediates = [im for im in intermediates if im != final_ans]
    if valid_intermediates:
        im_val = valid_intermediates[0]
        # Replace the final answer in the last sentence with the intermediate value
        d1 = re.sub(rf"Therefore, the final answer is {re.escape(final_ans)}\.", f"Therefore, the final answer is {im_val}.", cleaned_trace)
        if d1 != cleaned_trace:
            distractors.add(d1)

    # Distractor 2: Arithmetic slip (+/- 1, 2, 5, 10)
    deltas = [1, -1, 2, -2, 5, -5, 10, -10]
    rng.shuffle(deltas)
    for d in deltas:
        slip_val = val + d
        slip_str = f"{slip_val:.1f}" if is_float else str(slip_val)
        if slip_str != final_ans:
            d2 = re.sub(rf"Therefore, the final answer is {re.escape(final_ans)}\.", f"Therefore, the final answer is {slip_str}.", cleaned_trace)
            if d2 != cleaned_trace:
                distractors.add(d2)
                break

    # Distractor 3: Factor or scaling slip (e.g. 2x, half, or +20%)
    scale_factors = [2, 0.5, 1.2, 0.8, 10]
    rng.shuffle(scale_factors)
    for factor in scale_factors:
        scale_val = round(val * factor) if not is_float else round(val * factor, 2)
        scale_str = str(scale_val)
        if scale_str != final_ans and len(distractors) < 3:
            d3 = re.sub(rf"Therefore, the final answer is {re.escape(final_ans)}\.", f"Therefore, the final answer is {scale_str}.", cleaned_trace)
            if d3 != cleaned_trace:
                distractors.add(d3)

    # Fill if needed with distinct offsets
    offset = 15
    while len(distractors) < 3:
        alt_val = val + offset
        alt_str = str(alt_val)
        d_fill = re.sub(rf"Therefore, the final answer is {re.escape(final_ans)}\.", f"Therefore, the final answer is {alt_str}.", cleaned_trace)
        distractors.add(d_fill)
        offset += 7

    return list(distractors)[:3]


def process_split(split_data, rng: random.Random) -> List[Dict[str, Any]]:
    rows = []
    letters = ["(A)", "(B)", "(C)", "(D)"]

    for item in split_data:
        question = item["question"].strip()
        raw_answer = item["answer"].strip()

        cleaned_trace, final_ans = clean_trace(raw_answer)
        intermediates = extract_intermediates(raw_answer)

        distractors = generate_distractors(cleaned_trace, final_ans, intermediates, rng)

        # 4 choices total: 1 correct + 3 distractors
        all_choices = [cleaned_trace] + distractors[:3]
        rng.shuffle(all_choices)

        label = all_choices.index(cleaned_trace)

        # Build formatted premise text
        choices_text = " ".join([f"{letters[i]} {all_choices[i]}" for i in range(4)])
        formatted_text = f"Question: {question} {choices_text}"

        rows.append({
            "question": question,
            "choices": all_choices,
            "text": formatted_text,
            "label": label,
            "source": "gsm8k",
        })

    return rows


def main():
    print("========================================================================")
    print(" PREPARING PHASE 3 DATASET: TRACK G (MATHEMATICAL REASONING - GSM8K)")
    print("========================================================================")

    print("Loading raw openai/gsm8k from HuggingFace...")
    raw_ds = load_dataset("openai/gsm8k", "main")

    rng = random.Random(42)

    print(f"Processing train split ({len(raw_ds['train'])} samples)...")
    train_rows = process_split(raw_ds["train"], rng)

    print(f"Processing test split ({len(raw_ds['test'])} samples)...")
    test_rows = process_split(raw_ds["test"], rng)

    out_ds = DatasetDict({
        "train": Dataset.from_list(train_rows),
        "test": Dataset.from_list(test_rows),
    })

    print(f"\nSaving processed math dataset to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_ds.save_to_disk(OUTPUT_DIR)

    print(f"\nTrack G dataset ready!")
    print(f"Train samples: {len(out_ds['train'])}")
    print(f"Test samples:  {len(out_ds['test'])}")
    print("Sample 0 verification:")
    s0 = out_ds["train"][0]
    print(f"  Question: {s0['question'][:100]}...")
    print(f"  Label: {s0['label']}")
    print(f"  Correct Choice: {s0['choices'][s0['label']][:120]}...")


if __name__ == "__main__":
    main()
