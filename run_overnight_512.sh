#!/usr/bin/env bash
# run_overnight_512.sh - Autonomous Overnight Execution for System2 Track B at max_len = 512.
# 
# Protocol: Rebuilds model with full 512 context window across 5 epochs on MPS,
# judges against .best_metric_code on all 4,067 held-out tests, commits and pushes on improvement,
# or reverts on regression.
#
# Usage: ./run_overnight_512.sh
set -euo pipefail

echo "========================================================================"
echo "STARTING SYSTEM2-CODE OVERNIGHT 512 CONTEXT RUN"
echo "Start Time: $(date)"
echo "Context Window: max_len = 512 (99.5% dataset coverage)"
echo "Target Guarded Metric: $(cat .best_metric_code 2>/dev/null || echo 'None')"
echo "========================================================================"

export MAX_LEN=512
./runner_code.sh "context window expansion: max_len = 512 (overnight run)"

echo "========================================================================"
echo "OVERNIGHT 512 RUN CONCLUDED"
echo "End Time: $(date)"
echo "Final Guarded Metric: $(cat .best_metric_code)"
echo "========================================================================"
