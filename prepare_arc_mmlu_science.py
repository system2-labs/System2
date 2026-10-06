"""prepare_arc_mmlu_science.py - Curated Multi-Source Scientific Reasoning Dataset.

Combines:
1. AI2 ARC (Challenge Train/Val + Easy Train/Val): 4,300 in-domain problems
2. MMLU Pure Science Disciplines (Physics, Chemistry, Biology, Astronomy, Genetics):
   - conceptual_physics
   - high_school_physics
   - college_physics
   - high_school_chemistry
   - college_chemistry
   - high_school_biology
   - college_biology
   - astronomy
   - medical_genetics
   - anatomy
   - virology
   Total External: ~1,986 high-purity causal reasoning problems

Evaluation Set:
- Preserves the full canonical held-out AI2 ARC test set (3,530 problems)
  to ensure strict comparability against all Phase 2 Baselines and SOTA (.best_metric = 0.6625).
"""
import random
from datasets import Dataset, DatasetDict, load_dataset


def prepare_arc_train():
    challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    easy = load_dataset("allenai/ai2_arc", "ARC-Easy")
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
    records = []

    for split in [challenge["train"], easy["train"], challenge["validation"], easy["validation"]]:
        for ex in split:
            key = ex.get("answerKey")
            choices = ex.get("choices", {})
            texts = choices.get("text", [])
            if key not in label_map or len(texts) != 4:
                continue
            q = ex["question"].strip()
            c_texts = [t.strip() for t in texts]
            formatted_text = f"Question: {q}\n(A) {c_texts[0]}\n(B) {c_texts[1]}\n(C) {c_texts[2]}\n(D) {c_texts[3]}"
            records.append({
                "question": q,
                "choices": c_texts,
                "text": formatted_text,
                "label": label_map[key],
                "source": "arc",
            })
    return records


def prepare_mmlu_science_train():
    subjects = [
        "conceptual_physics",
        "high_school_physics",
        "college_physics",
        "high_school_chemistry",
        "college_chemistry",
        "high_school_biology",
        "college_biology",
        "astronomy",
        "medical_genetics",
        "anatomy",
        "virology",
    ]
    records = []
    for sub in subjects:
        try:
            ds = load_dataset("cais/mmlu", sub)
            for split_name in ds:
                for ex in ds[split_name]:
                    q = ex.get("question", "").strip()
                    choices = ex.get("choices", [])
                    ans = ex.get("answer")
                    if not q or len(choices) != 4 or ans not in [0, 1, 2, 3]:
                        continue
                    c_texts = [str(c).strip() for c in choices]
                    formatted_text = f"Question: {q}\n(A) {c_texts[0]}\n(B) {c_texts[1]}\n(C) {c_texts[2]}\n(D) {c_texts[3]}"
                    records.append({
                        "question": q,
                        "choices": c_texts,
                        "text": formatted_text,
                        "label": ans,
                        "source": f"mmlu_{sub}",
                    })
        except Exception as e:
            print(f"Warning: failed to load cais/mmlu/{sub}: {e}")
    return records


def prepare_arc_test():
    challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    easy = load_dataset("allenai/ai2_arc", "ARC-Easy")["test"]
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
    records = []

    for split in [challenge, easy]:
        for ex in split:
            key = ex.get("answerKey")
            choices = ex.get("choices", {})
            texts = choices.get("text", [])
            if key not in label_map or len(texts) != 4:
                continue
            q = ex["question"].strip()
            c_texts = [t.strip() for t in texts]
            formatted_text = f"Question: {q}\n(A) {c_texts[0]}\n(B) {c_texts[1]}\n(C) {c_texts[2]}\n(D) {c_texts[3]}"
            records.append({
                "question": q,
                "choices": c_texts,
                "text": formatted_text,
                "label": label_map[key],
                "source": "arc",
            })
    return records


def main():
    print("Preparing ARC train records...")
    arc_train = prepare_arc_train()
    print(f"ARC train: {len(arc_train)} samples")

    print("Preparing MMLU pure STEM science records...")
    mmlu_train = prepare_mmlu_science_train()
    print(f"MMLU science train: {len(mmlu_train)} samples")

    all_train = arc_train + mmlu_train
    # Deterministic shuffle
    random.Random(42).shuffle(all_train)
    print(f"Total Combined Training Pool: {len(all_train)} samples")

    print("Preparing held-out ARC test records...")
    arc_test = prepare_arc_test()
    print(f"Canonical held-out ARC test: {len(arc_test)} samples")

    dataset_dict = DatasetDict({
        "train": Dataset.from_list(all_train),
        "test": Dataset.from_list(arc_test),
    })

    out_dir = "./data/arc_mmlu_science"
    print(f"Saving curated dataset to {out_dir}...")
    dataset_dict.save_to_disk(out_dir)
    print(f"Successfully saved {out_dir}!")


if __name__ == "__main__":
    main()
