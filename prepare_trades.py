"""prepare_trades.py - Ingest and prepare Technical Trades & Operational Diagnostics for System2 Track D.

Domains covered:
1. Electrical Engineering & Trades:
   - Noru1/EEReasonBench: AC/DC circuits, polyphase networks, transformers, component fault isolation.
   - cais/mmlu (electrical_engineering): Circuit fundamentals, electrical systems, load balance.
2. Culinary Chemistry & Nutrition Science:
   - cais/mmlu (nutrition): Food chemistry, preservation, nutrient stability, reaction thresholds.

Standardizes all examples into System2 feedforward 4-choice deliberation schema:
- question: str
- choices: list[str] (4 candidate options)
- text: str (formatted prompt with choices A-D)
- label: int (0..3)
- source: str (dataset identifier)

Saves to ./data/trades as a DatasetDict on disk.
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
            "source": "ee_reason_bench"
        })
        
    print(f"EEReasonBench parsed: {len(samples)} valid samples.")
    return samples


def parse_mmlu_sub(subset_name):
    print(f"Loading cais/mmlu ({subset_name})...")
    ds = load_dataset("cais/mmlu", subset_name)
    
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
                "source": f"mmlu_{subset_name}"
            })
            
    print(f"MMLU ({subset_name}) parsed: {len(samples)} valid samples.")
    return samples


def main():
    print("================================================================")
    print("PREPARING SYSTEM2 TECHNICAL TRADES DATASET (TRACK D)")
    print("================================================================")
    
    random.seed(42)
    
    ee_samples = parse_ee_reason_bench()
    mmlu_ee_samples = parse_mmlu_sub("electrical_engineering")
    mmlu_nut_samples = parse_mmlu_sub("nutrition")
    
    all_samples = ee_samples + mmlu_ee_samples + mmlu_nut_samples
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]
    
    train_ds = Dataset.from_list(train_samples)
    test_ds = Dataset.from_list(test_samples)
    
    out_dict = DatasetDict({
        "train": train_ds,
        "test": test_ds
    })
    
    output_dir = "./data/trades"
    print("\n----------------------------------------------------------------")
    print(f"Total Prepared Track D Trades Dataset:")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Test samples:  {len(test_ds)}")
    print(f"Saving to {output_dir}...")
    out_dict.save_to_disk(output_dir)
    print(f"Successfully saved {output_dir}!")
    print("================================================================")


if __name__ == "__main__":
    main()
