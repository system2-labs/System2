#!/usr/bin/env python3
"""train_safety.py - Dedicated Training Pipeline for System2 Track K (Safety).

Trains a 4-choice System 2 feedforward deliberation model on the augmented safety dataset:
- 5 parallel ThoughtHeads across asymmetric span formulations
- Arithmetic centroid contrast [x_i || d_i]
- Scale-invariant z-score logit normalization
- LoRA rank-32, alpha-32 adaptation on top 8 layers (20-27) of ModernBERT-Large-Instruct
- Coupled decision-level calibration gradient with Cosine Annealing learning rate schedule

Outputs:
  Candidate checkpoint (default: ./models/system2_safety_candidate.pt)
  Emits evaluation JSON: {"metric": <float>, "accuracy": <float>, "brier": <float>, "ece": <float>}
"""

import argparse
import json
import os
import subprocess
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def train_safety(
    data_dir: str = "./data/safety",
    save_path: str = "./models/system2_safety_candidate.pt",
    epochs: int = 5,
    batch_size: int = 8,
    train_size: int = 1600,
    max_len: int = 384,
    device: str = "mps",
) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Set Apple Silicon MPS memory ratios to prevent thrashing
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.5"

    train_cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "train.py"),
        "--data", data_dir,
        "--save", save_path,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--train-size", str(train_size),
        "--max-len", str(max_len),
        "--dynamic-len",
        "--device", device,
        "--lora",
    ]

    print(f"[train_safety] Starting training: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, cwd=ROOT_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    # Evaluate using scripts/evaluate_safety.py
    eval_cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "scripts", "evaluate_safety.py"),
        save_path,
        "--data", data_dir,
        "--max-len", str(max_len),
        "--device", device,
    ]

    print(f"[train_safety] Evaluating candidate: {' '.join(eval_cmd)}")
    eval_proc = subprocess.run(eval_cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    if eval_proc.returncode != 0:
        raise RuntimeError(f"Evaluation failed: {eval_proc.stderr}")

    metrics = json.loads(eval_proc.stdout.strip())
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Track K Safety Deliberation Adapter")
    parser.add_argument("--data", default="./data/safety", help="Path to safety dataset")
    parser.add_argument("--save", default="./models/system2_safety_candidate.pt", help="Checkpoint save destination")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per forward step")
    parser.add_argument("--train-size", type=int, default=1600, help="Number of training samples")
    parser.add_argument("--max-len", type=int, default=384, help="Sequence context ceiling")
    parser.add_argument("--device", default="mps", help="Compute device (mps, cpu, cuda)")

    args = parser.parse_args()

    metrics = train_safety(
        data_dir=args.data,
        save_path=args.save,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_size=args.train_size,
        max_len=args.max_len,
        device=args.device,
    )

    print("\n--- Final Safety Deliberation Metrics ---")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
