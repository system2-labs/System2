"""prepare_electrical.py - Ingest and prepare pure Electrical, Electronics & Power Systems dataset for System2 Track D.

In-Domain Causal Purity Law: Strictly isolates electrical engineering, AC/DC circuit analysis,
machines, transformers, and power networks from unrelated mechanical or culinary data.

Sources:
1. Noru1/EEReasonBench (503 samples):
   - AC/DC network theorems, polyphase systems, transformers, machines, component fault isolation.
2. cais/mmlu (electrical_engineering: 166 samples):
   - Circuit analysis, signals, control systems, electromagnetics, semiconductor devices.

Standardizes all examples into System2 feedforward 4-choice deliberation schema:
- question: str
- choices: list[str] (4 candidate options)
- text: str (formatted prompt with choices A-D)
- label: int (0..3)
- source: str (dataset identifier)

Saves to ./data/electrical as a DatasetDict on disk.
"""

import random
import re
from datasets import Dataset, DatasetDict, load_dataset


def parse_ee_reason_bench():
    print("Loading Noru1/EEReasonBench...")
    ds = load_dataset("Noru1/EEReasonBench", split="train")

    samples = []
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    for ex in ds:
        q = ex["problem"].strip()
        opts = ex.get("choices", {})
        ans_raw = ex.get("answer", "").strip()

        # Extract letter answer e.g. "B. $$30^\\circ$$" -> "B"
        match = re.match(r"^([A-D])\b", ans_raw)
        if not match or len(opts) != 4:
            continue

        ans_letter = match.group(1)
        c_a = opts.get("A", "").strip()
        c_b = opts.get("B", "").strip()
        c_c = opts.get("C", "").strip()
        c_d = opts.get("D", "").strip()

        if not (c_a and c_b and c_c and c_d):
            continue

        choices = [c_a, c_b, c_c, c_d]
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
            "label": label_map[ans_letter],
            "source": "ee_reason_bench",
        })

    print(f"EEReasonBench parsed: {len(samples)} valid samples.")
    return samples


def parse_mmlu_electrical():
    print("Loading cais/mmlu (electrical_engineering)...")
    ds = load_dataset("cais/mmlu", "electrical_engineering")

    samples = []
    for split_name in ds.keys():
        for ex in ds[split_name]:
            q = ex["question"].strip()
            choices = [c.strip() for c in ex["choices"]]
            ans = ex["answer"]

            if len(choices) != 4 or ans not in [0, 1, 2, 3]:
                continue

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
                "label": ans,
                "source": "mmlu_electrical_engineering",
            })

    print(f"MMLU (electrical_engineering) parsed: {len(samples)} valid samples.")
    return samples


def main():
    print("================================================================")
    print("PREPARING SYSTEM2 ELECTRICAL & POWER SYSTEMS DATASET (TRACK D)")
    print("================================================================")

    random.seed(42)

    ee_samples = parse_ee_reason_bench()
    mmlu_ee_samples = parse_mmlu_electrical()

    all_samples = ee_samples + mmlu_ee_samples
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

    output_dir = "./data/electrical"
    print("\n----------------------------------------------------------------")
    print(f"Total Prepared Track D Electrical Dataset:")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Test samples:  {len(test_ds)}")
    print(f"Saving to {output_dir}...")
    out_dict.save_to_disk(output_dir)
    print(f"Successfully saved {output_dir}!")
    print("================================================================")


if __name__ == "__main__":
    main()
