#!/usr/bin/env python3
"""run_experiments.py - Runs 3 sequential AutoResearch experiments back-to-back.

Option 1: Parallel Head Count Scaling (N=8 dense parallel heads on pure ARC)
Option 2: LoRA Rank Expansion (r=64, alpha=64 on top 8 ModernBERT layers)
Option 3: Deeper Intra-Choice Depth (head_layers=3 Transformer layers per ThoughtHead)
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
    print(f"Current best metric threshold: {get_best_metric():.4f}")
    print(f"{'='*75}\n", flush=True)

    # Apply hyperparams
    for k, v in hyperparams.items():
        set_hyperparam(k, v)

    start_time = time.time()
    # Execute runner.sh with real-time streaming
    proc = subprocess.Popen(
        ["./runner.sh", description],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        stdout_lines.append(line)
    proc.wait()
    duration = time.time() - start_time
    stdout = "".join(stdout_lines)

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

    # Active baseline state
    active_num_heads = 5
    active_lora_r = 32
    active_lora_alpha = 32
    active_head_layers = 2

    # Option 1: Parallel Head Count Scaling (N = 5 -> 8)
    opt1_params = {
        "use_mote": False,
        "data_dir": "./data/arc",
        "train_size": 3500,
        "epochs": 3,
        "batch_size": 16,
        "max_len": 128,
        "num_heads": 8,
        "head_layers": active_head_layers,
        "use_lora": True,
        "lora_r": active_lora_r,
        "lora_alpha": active_lora_alpha,
        "lora_layers": 8,
        "lr_heads": 1e-3,
        "lr_judger": 3e-4,
        "lr_lora": 2e-4,
        "lambda_div": 0.01,
        "pairwise": True,
        "diff_features": True,
        "task_prefixes": False,
        "learnable_temp": False,
    }
    r1 = run_experiment(
        name="Option 1: Parallel Head Count Scaling (N=8)",
        hyperparams=opt1_params,
        description="Phase 2: Parallel Head Count Scaling (N=8 dense parallel heads on pure ARC)",
    )
    results.append(r1)

    if r1["committed"]:
        active_num_heads = 8
        print(f">>> Option 1 committed as new SOTA! Adopting num_heads={active_num_heads} for subsequent tests.")
    else:
        print(f">>> Option 1 reverted. Retaining num_heads={active_num_heads}.")

    # Option 2: LoRA Rank Expansion (r = 32 -> 64)
    opt2_params = {
        "use_mote": False,
        "data_dir": "./data/arc",
        "train_size": 3500,
        "epochs": 3,
        "batch_size": 16,
        "max_len": 128,
        "num_heads": active_num_heads,
        "head_layers": active_head_layers,
        "use_lora": True,
        "lora_r": 64,
        "lora_alpha": 64,
        "lora_layers": 8,
        "lr_heads": 1e-3,
        "lr_judger": 3e-4,
        "lr_lora": 2e-4,
        "lambda_div": 0.01,
        "pairwise": True,
        "diff_features": True,
        "task_prefixes": False,
        "learnable_temp": False,
    }
    r2 = run_experiment(
        name="Option 2: LoRA Rank Expansion (r=64, alpha=64)",
        hyperparams=opt2_params,
        description="Phase 2: LoRA Rank Expansion (r=64, alpha=64 on top 8 ModernBERT layers)",
    )
    results.append(r2)

    if r2["committed"]:
        active_lora_r = 64
        active_lora_alpha = 64
        print(f">>> Option 2 committed as new SOTA! Adopting lora_r={active_lora_r} for subsequent tests.")
    else:
        print(f">>> Option 2 reverted. Retaining lora_r={active_lora_r}.")

    # Option 3: Deeper Intra-Choice Transformer Depth (Depth 2 -> 3)
    opt3_params = {
        "use_mote": False,
        "data_dir": "./data/arc",
        "train_size": 3500,
        "epochs": 3,
        "batch_size": 16,
        "max_len": 128,
        "num_heads": active_num_heads,
        "head_layers": 3,
        "use_lora": True,
        "lora_r": active_lora_r,
        "lora_alpha": active_lora_alpha,
        "lora_layers": 8,
        "lr_heads": 1e-3,
        "lr_judger": 3e-4,
        "lr_lora": 2e-4,
        "lambda_div": 0.01,
        "pairwise": True,
        "diff_features": True,
        "task_prefixes": False,
        "learnable_temp": False,
    }
    r3 = run_experiment(
        name="Option 3: Deeper Intra-Choice Depth (head_layers=3)",
        hyperparams=opt3_params,
        description="Phase 2: Deeper Intra-Choice Depth (head_layers=3 Transformer layers per ThoughtHead)",
    )
    results.append(r3)

    print("\n" + "=" * 80)
    print("ALL 3 EXPERIMENTS COMPLETED - SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Experiment':<48} | {'Metric':<8} | {'Accuracy':<10} | {'Brier':<8} | {'Status'}")
    print("-" * 80)
    for r in results:
        status = "COMMITTED (NEW SOTA 🏆)" if r["committed"] else "REVERTED"
        m_str = f"{r['metric']:.4f}" if r["metric"] is not None else "N/A"
        a_str = f"{r['accuracy']*100:.2f}%" if r["accuracy"] is not None else "N/A"
        b_str = f"{r['brier']:.4f}" if r["brier"] is not None else "N/A"
        print(f"{r['name']:<48} | {m_str:<8} | {a_str:<10} | {b_str:<8} | {status}")
    print("=" * 80)
    print(f"Final .best_metric: {get_best_metric():.4f}\n")

    with open("experiments_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
