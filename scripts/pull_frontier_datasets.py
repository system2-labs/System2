"""pull_frontier_datasets.py - Unified Ingestion Pipeline for Frontier Enterprise Datasets.

Pulls, standardizes, and stores target datasets locally in `data/frontier/`:
1. Cyber Threat Intelligence: SecQA (v1, v2)
2. Supply Chain & Operations: RetailOpt-190
3. Tax Compliance & Accounting: Australian Tax Guidance & LegalBench Tax
4. AI Safety & Sycophancy: Anthropic Model-Written Evals
5. Multi-Step Math Reasoning: OpenAI GSM8K

Note: The `data/` directory is strictly gitignored to preserve repository performance
and prevent Git history bloat.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import datasets

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "frontier")


def pull_secqa(output_dir: str):
    """Pull SecQA (v1 & v2) for cyber threat intelligence."""
    print("\n[1/5] Pulling Cyber Threat Intelligence (SecQA)...")
    sec_dir = os.path.join(output_dir, "cyber")
    os.makedirs(sec_dir, exist_ok=True)

    for config in ["secqa_v1", "secqa_v2"]:
        for split in ["val", "test"]:
            try:
                ds = datasets.load_dataset("zefang-liu/secqa", config, split=split)
                out_path = os.path.join(sec_dir, f"{config}_{split}.jsonl")
                with open(out_path, "w", encoding="utf-8") as f:
                    for row in ds:
                        record = {
                            "question": row["Question"],
                            "choices": [row["A"], row["B"], row["C"], row["D"]],
                            "answer": row["Answer"],
                            "explanation": row.get("Explanation", ""),
                        }
                        f.write(json.dumps(record) + "\n")
                print(f"  * Saved {len(ds)} rows to {out_path}")
            except Exception as e:
                print(f"  ! Warning: Failed to pull SecQA {config} {split}: {e}")


def pull_anthropic_evals(output_dir: str):
    """Pull Anthropic Model-Written Evals for cognitive bias and sycophancy."""
    print("\n[2/5] Pulling Anthropic Model-Written Evals...")
    eval_dir = os.path.join(output_dir, "anthropic_evals")
    os.makedirs(eval_dir, exist_ok=True)

    try:
        ds = datasets.load_dataset("Anthropic/model-written-evals", split="train")
        out_path = os.path.join(eval_dir, "model_written_evals_train.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(dict(row)) + "\n")
        print(f"  * Saved {len(ds)} rows to {out_path}")
    except Exception as e:
        print(f"  ! Warning: Failed to pull Anthropic evals: {e}")


def pull_openai_gsm8k(output_dir: str):
    """Pull OpenAI GSM8K multi-step math problems."""
    print("\n[3/5] Pulling OpenAI GSM8K...")
    math_dir = os.path.join(output_dir, "openai_gsm8k")
    os.makedirs(math_dir, exist_ok=True)

    for split in ["train", "test"]:
        try:
            ds = datasets.load_dataset("openai/gsm8k", "main", split=split)
            out_path = os.path.join(math_dir, f"gsm8k_{split}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for row in ds:
                    f.write(json.dumps(dict(row)) + "\n")
            print(f"  * Saved {len(ds)} rows to {out_path}")
        except Exception as e:
            print(f"  ! Warning: Failed to pull GSM8K {split}: {e}")


def pull_tax_compliance(output_dir: str):
    """Pull Australian Tax Guidance dataset."""
    print("\n[4/5] Pulling Tax Compliance (Australian Tax Guidance)...")
    tax_dir = os.path.join(output_dir, "tax")
    os.makedirs(tax_dir, exist_ok=True)

    try:
        ds = datasets.load_dataset("isaacus/australian-tax-guidance-retrieval", split="test")
        out_path = os.path.join(tax_dir, "ato_guidance_test.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(dict(row)) + "\n")
        print(f"  * Saved {len(ds)} tax guidance questions to {out_path}")
    except Exception as e:
        print(f"  ! Warning: Failed to pull tax corpus: {e}")


def pull_supply_chain(output_dir: str):
    """Pull RetailOpt supply chain optimization benchmark."""
    print("\n[5/5] Pulling Supply Chain Operations (RetailOpt-190)...")
    sc_dir = os.path.join(output_dir, "supply_chain")
    os.makedirs(sc_dir, exist_ok=True)

    try:
        ds = datasets.load_dataset("Jacoblian/RetailOpt-190", split="test")
        out_path = os.path.join(sc_dir, "retailopt_190_test.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(dict(row)) + "\n")
        print(f"  * Saved {len(ds)} supply chain optimization problems to {out_path}")
    except Exception as e:
        print(f"  ! Warning: Failed to pull supply chain data: {e}")


def main():
    parser = argparse.ArgumentParser(description="Pull and standardize frontier datasets locally.")
    parser.add_argument("--output-dir", default=DATA_DIR, help="Local directory for cached datasets")
    parser.add_argument("--target", choices=["all", "cyber", "anthropic", "gsm8k", "tax", "supply_chain"], default="all")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Pulling Frontier Datasets to {args.output_dir} ===")

    if args.target in ["all", "cyber"]:
        pull_secqa(args.output_dir)
    if args.target in ["all", "anthropic"]:
        pull_anthropic_evals(args.output_dir)
    if args.target in ["all", "gsm8k"]:
        pull_openai_gsm8k(args.output_dir)
    if args.target in ["all", "tax"]:
        pull_tax_compliance(args.output_dir)
    if args.target in ["all", "supply_chain"]:
        pull_supply_chain(args.output_dir)

    print("\n=== Dataset Ingestion Complete ===")


if __name__ == "__main__":
    main()
