#!/usr/bin/env python3
"""run_experiments.py - Runs 3 sequential AutoResearch experiments back-to-back.

Path 1: Supervised Domain-Gated MoTE (ARC experts 0&1, OBQA experts 2&3, 8.3k dataset)
Path 2: Logit-Ensemble MoE (Loss on routed ensemble, lambda_aux=0.10, 8.3k dataset)
Path 3: Pure In-Domain ARC with Dynamic SoftCentroid MoTE (4 heads, pure ARC 3.5k dataset)
"""
import json
import os
import re
import subprocess
import sys
import time


def set_hyperparam(key, value):
    with open("train.py", "r") as f:
        content = f.read()
    if isinstance(value, str):
        val_str = f'"{value}"'
    elif isinstance(value, bool):
        val_str = "True" if value else "False"
    else:
        val_str = str(value)
    new_content = re.sub(rf'"{key}":\s*[^,\n]+,', f'"{key}": {val_str},', content)
    with open("train.py", "w") as f:
        f.write(new_content)


def get_best_metric():
    with open(".best_metric", "r") as f:
        return float(f.read().strip())


def run_experiment(name, hyperparams, description):
    print(f"\n{'='*75}")
    print(f"STARTING EXPERIMENT: {name}")
    print(f"Description: {description}")
    print(f"Hyperparams to apply: {hyperparams}")
    print(f"Current best metric threshold: {get_best_metric()}")
    print(f"{'='*75}\n", flush=True)

    # Apply hyperparams
    for k, v in hyperparams.items():
        set_hyperparam(k, v)

    start_time = time.time()
    # Execute runner.sh
    proc = subprocess.run(["./runner.sh", description], capture_output=True, text=True)
    duration = time.time() - start_time

    stdout = proc.stdout
    stderr = proc.stderr
    print(stdout)
    if stderr:
        print("STDERR:", stderr, file=sys.stderr)

    # Parse outcome
    committed = "BETTER:" in stdout
    reverted = "WORSE or equal:" in stdout

    # Extract metric line if available
    metric_match = re.search(r'\{"metric":\s*([0-9.]+),\s*"accuracy":\s*([0-9.]+),\s*"brier":\s*([0-9.]+)\}', stdout)
    if metric_match:
        metric = float(metric_match.group(1))
        acc = float(metric_match.group(2))
        brier = float(metric_match.group(3))
    else:
        metric, acc, brier = None, None, None

    result = {
        "name": name,
        "description": description,
        "hyperparams": hyperparams,
        "metric": metric,
        "accuracy": acc,
        "brier": brier,
        "committed": committed,
        "duration_sec": round(duration, 1),
        "new_best_metric": get_best_metric(),
    }
    print(f"\n>>> FINISHED {name}: Metric={metric}, Acc={acc}, Brier={brier}, Committed={committed} ({duration:.1f}s)\n", flush=True)
    return result


def main():
    results = []

    # Path 1: Supervised Domain-Gated MoTE (ARC experts 0&1, OBQA experts 2&3)
    r1 = run_experiment(
        name="Path 1: Supervised Domain-Gated MoTE",
        hyperparams={
            "use_mote": True,
            "mote_mode": "supervised",
            "data_dir": "./data/arc_obqa",
            "train_size": 8500,
            "epochs": 2,
            "num_experts": 4,
            "mote_top_k": 2,
            "soft_centroid": True,
            "lr_router": 5e-4,
            "lr_heads": 1e-3,
            "lr_judger": 3e-4,
            "lr_lora": 2e-4,
            "lora_r": 32,
            "lora_alpha": 32,
            "lora_layers": 8,
            "lambda_aux": 0.10,
            "task_prefixes": False,
        },
        description="Phase 2: Supervised Domain-Gated MoTE (ARC experts 0&1, OBQA experts 2&3, 8.3k)",
    )
    results.append(r1)

    # Path 2: Logit-Ensemble MoE with Load Balancing (Loss on routed ensemble)
    r2 = run_experiment(
        name="Path 2: Logit-Ensemble MoE",
        hyperparams={
            "use_mote": True,
            "mote_mode": "ensemble",
            "data_dir": "./data/arc_obqa",
            "train_size": 8500,
            "epochs": 2,
            "num_experts": 4,
            "mote_top_k": 2,
            "soft_centroid": True,
            "lr_router": 5e-4,
            "lr_heads": 1e-3,
            "lr_judger": 3e-4,
            "lr_lora": 2e-4,
            "lora_r": 32,
            "lora_alpha": 32,
            "lora_layers": 8,
            "lambda_aux": 0.10,
            "task_prefixes": False,
        },
        description="Phase 2: Logit-Ensemble MoE (Loss on routed ensemble, lambda_aux=0.10, 8.3k)",
    )
    results.append(r2)

    # Path 3: Pure In-Domain ARC with Dynamic SoftCentroid MoTE
    r3 = run_experiment(
        name="Path 3: Pure ARC with SoftCentroid MoTE",
        hyperparams={
            "use_mote": True,
            "mote_mode": "arc_soft",
            "data_dir": "./data/arc",
            "train_size": 3500,
            "epochs": 3,
            "num_experts": 4,
            "soft_centroid": True,
            "lr_heads": 1e-3,
            "lr_judger": 3e-4,
            "lr_lora": 2e-4,
            "lora_r": 32,
            "lora_alpha": 32,
            "lora_layers": 8,
            "task_prefixes": False,
        },
        description="Phase 2: Pure ARC with SoftCentroid MoTE (4 heads, dynamic candidate attention)",
    )
    results.append(r3)

    print("\n" + "=" * 80)
    print("ALL 3 EXPERIMENTS COMPLETED - SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Experiment':<44} | {'Metric':<8} | {'Accuracy':<10} | {'Brier':<8} | {'Status'}")
    print("-" * 80)
    for r in results:
        status = "COMMITTED (NEW SOTA 🏆)" if r["committed"] else "REVERTED"
        m_str = f"{r['metric']:.4f}" if r["metric"] is not None else "N/A"
        a_str = f"{r['accuracy']*100:.2f}%" if r["accuracy"] is not None else "N/A"
        b_str = f"{r['brier']:.4f}" if r["brier"] is not None else "N/A"
        print(f"{r['name']:<44} | {m_str:<8} | {a_str:<10} | {b_str:<8} | {status}")
    print("=" * 80)
    print(f"Final .best_metric: {get_best_metric()}\n")

    with open("experiments_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
