#!/usr/bin/env bash
# run_chain_256_then_512.sh - Chained execution: runs max_len = 256, then automatically runs max_len = 512 overnight.
set -euo pipefail

echo "========================================================================"
echo "STAGE 1: RUNNING MAX_LEN = 256 CONTEXT EXPANSION"
echo "Start Time: $(date)"
echo "Current Guarded Metric: $(cat .best_metric_code)"
echo "========================================================================"

export MAX_LEN=256
./runner_code.sh "context window expansion: max_len = 256"

echo "========================================================================"
echo "STAGE 1 COMPLETE: $(date)"
echo "Guarded Metric after Stage 1: $(cat .best_metric_code)"
echo "========================================================================"

echo "========================================================================"
echo "STAGE 2: RUNNING MAX_LEN = 512 OVERNIGHT EXECUTION"
echo "Start Time: $(date)"
echo "========================================================================"

export MAX_LEN=512
./runner_code.sh "context window expansion: max_len = 512 (overnight run)"

echo "========================================================================"
echo "ALL RUNS FINISHED: $(date)"
echo "Final Guarded Metric: $(cat .best_metric_code)"
echo "========================================================================"
