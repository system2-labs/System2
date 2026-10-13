"""prepare_auto.py - Ingest and prepare Automotive & HVAC Mechanical Diagnostics dataset for System2 Track E.

In-Domain Causal Purity Law: Strictly isolates automotive powertrain, transmission,
OBD-II diagnostic trouble codes (DTCs), sensor fault trees, and HVAC refrigeration
mechanics from other unrelated domains.

Source:
- build-small-hackathon/nemotron-car-diagnostics-datasets (15,048 samples):
  OBD-II DTC fault isolation, STFT/LTFT fuel trims, MAF/MAP sensor cross-correlation,
  O2 sensor oscillations, catalytic converter diagnostics, and mechanical troubleshooting.

Standardizes all examples into System2 feedforward 4-choice deliberation schema:
- question: str
- choices: list[str] (4 candidate options: 1 ground truth + 3 authentic vehicle diagnostic distractors)
- text: str (formatted prompt with choices A-D)
- label: int (0..3)
- source: str ("nemotron_car_diagnostics")

Saves to ./data/auto as a DatasetDict on disk.
"""

import os
import random
from datasets import Dataset, DatasetDict, load_dataset


def prepare_auto_dataset():
    print("Loading build-small-hackathon/nemotron-car-diagnostics-datasets...")
    ds = load_dataset("build-small-hackathon/nemotron-car-diagnostics-datasets", split="train")

    print(f"Total raw samples loaded: {len(ds)}")
    random.seed(42)

    # 1. Filter valid and concise diagnostic QA pairs
    valid_pairs = []
    for ex in ds:
        q = ex.get("question", "").strip()
        a = ex.get("answer", "").strip()

        # Clean thinking tags if present
        if "<think>" in a and "</think>" in a:
            a = a.split("</think>")[-1].strip()

        # Quality filters
        if not q or not a:
            continue
        if len(q) < 15 or len(q) > 600:
            continue
        if len(a) < 5 or len(a) > 250:
            continue
        if a.lower().startswith("here's a thinking process") or "1. analyze" in a.lower():
            continue

        valid_pairs.append({"question": q, "answer": a})

    print(f"Filtered clean diagnostic pairs: {len(valid_pairs)}")

    # 2. Extract answer pool for authentic in-domain distractor sampling
    all_answers = list({p["answer"] for p in valid_pairs})
    print(f"Unique diagnostic answer pool size: {len(all_answers)}")

    samples = []
    for pair in valid_pairs:
        q = pair["question"]
        correct_a = pair["answer"]

        # Sample 3 distinct distractors from answer pool
        distractors = []
        attempts = 0
        while len(distractors) < 3 and attempts < 50:
            attempts += 1
            cand = random.choice(all_answers)
            if cand != correct_a and cand not in distractors:
                distractors.append(cand)

        if len(distractors) != 3:
            continue

        choices = [correct_a] + distractors
        random.shuffle(choices)
        label = choices.index(correct_a)

        formatted_text = (
            f"Question: {q}\n"
            f"(A) {choices[0]}\n"
            f"(B) {choices[1]}\n"
            f"(C) {choices[2]}\n"
            f"(D) {choices[3]}"
        )

        samples.append({
            "question": q,
            "choices": choices,
            "text": formatted_text,
            "label": label,
            "source": "nemotron_car_diagnostics",
        })

    random.shuffle(samples)

    # Limit to 10,000 high-quality samples: 8,000 train, 2,000 held-out test
    total_to_keep = min(len(samples), 10000)
    samples = samples[:total_to_keep]

    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    train_ds = Dataset.from_list(train_samples)
    test_ds = Dataset.from_list(test_samples)

    out_dict = DatasetDict({
        "train": train_ds,
        "test": test_ds,
    })

    output_dir = "./data/auto"
    os.makedirs("./data", exist_ok=True)
    print("\n----------------------------------------------------------------")
    print(f"Total Prepared Track E Automotive Dataset:")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Test samples:  {len(test_ds)}")
    print(f"Saving to {output_dir}...")
    out_dict.save_to_disk(output_dir)
    print(f"Successfully saved {output_dir}!")
    print("================================================================")


if __name__ == "__main__":
    prepare_auto_dataset()
