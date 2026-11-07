#!/usr/bin/env python3
"""evaluate_logic.py - Metric-Guarded Evaluation Oracle for System2 Track M (Formal & Deductive Logic).

Loads a System2 checkpoint (default: models/system2_logic.pt), runs the full
held-out logic test split, and outputs standardized calibration and accuracy JSON:
  {"metric": <float>, "accuracy": <float>, "brier": <float>, "ece": <float>}

Formula:
  metric = 0.5 * accuracy + 0.5 * (1 - Brier)
"""

import argparse
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from eval import evaluate

GUARDED_THRESHOLD = 0.85


def main():
    parser = argparse.ArgumentParser(description="Evaluate Track M System2 Formal Logic Model")
    parser.add_argument("checkpoint", nargs="?", default="./models/system2_logic.pt", help="Path to logic checkpoint")
    parser.add_argument("--data", default="./data/logic", help="Path to logic dataset directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples to evaluate")
    parser.add_argument("--device", default=None, help="Device (mps, cpu, cuda)")
    parser.add_argument("--max-len", type=int, default=384, help="Max sequence length")
    parser.add_argument("--check-threshold", action="store_true", help=f"Fail with exit code 1 if metric < {GUARDED_THRESHOLD}")
    parser.add_argument("--threshold", type=float, default=GUARDED_THRESHOLD, help="Custom metric threshold to check")

    args = parser.parse_args()

    try:
        m = evaluate(
            args.checkpoint,
            batch=args.batch_size,
            max_samples=args.max_samples,
            device=args.device,
            data_dir=args.data,
            max_len=args.max_len,
            dynamic_len=True,
        )

        acc = m["accuracy"]
        brier = m["brier"]
        ece = m.get("ece", 0.0)
        metric = 0.5 * acc + 0.5 * (1.0 - brier)

        results = {
            "metric": round(metric, 4),
            "accuracy": round(acc, 4),
            "brier": round(brier, 4),
            "ece": round(ece, 4),
        }
        print(json.dumps(results))

        if args.check_threshold and results["metric"] < args.threshold:
            print(
                f"[GUARD REJECTED] Metric {results['metric']} is below required threshold {args.threshold}",
                file=sys.stderr,
            )
            sys.exit(1)

        sys.exit(0)

    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
