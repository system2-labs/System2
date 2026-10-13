"""prepare_culinary.py - Ingest and prepare Culinary Chemistry & Food Science dataset for System2 Track F.

In-Domain Causal Purity Law: Isolates food biochemistry, fermentation, leavening reactions,
protein coagulation, and culinary procedure from unrelated mechanical or electrical trades.

Sources:
1. cais/mmlu (nutrition: 306 samples):
   - Food chemistry, nutrient stability, lipid oxidation, vitamin degradation, enzymatic browning.
2. Eitanli/recipe_qa_dataset (procedural cooking & preparation):
   - Cooking failure troubleshooting, temperature thresholds, emulsification, and ingredient ratios.

Standardizes all examples into System2 feedforward 4-choice deliberation schema:
- question: str
- choices: list[str] (4 candidate options)
- text: str (formatted prompt with choices A-D)
- label: int (0..3)
- source: str ("mmlu_nutrition" or "recipe_qa")

Saves to ./data/culinary as a DatasetDict on disk.
"""

import os
import random
from datasets import Dataset, DatasetDict, load_dataset


def parse_mmlu_nutrition():
    print("Loading cais/mmlu (nutrition)...")
    ds = load_dataset("cais/mmlu", "nutrition")

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
                "source": "mmlu_nutrition",
            })

    print(f"MMLU (nutrition) parsed: {len(samples)} valid samples.")
    return samples


def parse_recipe_qa():
    print("Loading Eitanli/recipe_qa_dataset...")
    ds = load_dataset("Eitanli/recipe_qa_dataset", split="train")

    valid_pairs = []
    for ex in ds:
        q = ex.get("question", "").strip()
        ans_data = ex.get("answers", {})
        ans_texts = ans_data.get("text", [])
        if not q or not ans_texts:
            continue
        a = ans_texts[0].strip()
        if len(q) < 15 or len(q) > 400 or len(a) < 3 or len(a) > 100:
            continue
        valid_pairs.append({"question": q, "answer": a})

    print(f"RecipeQA clean pairs: {len(valid_pairs)}")
    all_answers = list({p["answer"] for p in valid_pairs})

    samples = []
    for pair in valid_pairs[:3000]:  # Take top 3,000 for balanced mix
        q = pair["question"]
        correct_a = pair["answer"]

        distractors = []
        attempts = 0
        while len(distractors) < 3 and attempts < 30:
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
            "source": "recipe_qa",
        })

    print(f"RecipeQA formatted: {len(samples)} valid 4-choice samples.")
    return samples


def main():
    print("================================================================")
    print("PREPARING SYSTEM2 CULINARY CHEMISTRY DATASET (TRACK F)")
    print("================================================================")

    random.seed(42)

    nut_samples = parse_mmlu_nutrition()
    recipe_samples = parse_recipe_qa()

    all_samples = nut_samples + recipe_samples
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

    output_dir = "./data/culinary"
    os.makedirs("./data", exist_ok=True)
    print("\n----------------------------------------------------------------")
    print(f"Total Prepared Track F Culinary Dataset:")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Test samples:  {len(test_ds)}")
    print(f"Saving to {output_dir}...")
    out_dict.save_to_disk(output_dir)
    print(f"Successfully saved {output_dir}!")
    print("================================================================")


if __name__ == "__main__":
    main()
