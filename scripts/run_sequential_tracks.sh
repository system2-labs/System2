#!/usr/bin/env bash
# run_sequential_tracks.sh - Sequential AutoResearch execution across Track B and Track C.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "========================================================================"
echo " [AutoResearch Pipeline] STARTING MULTI-TRACK EXECUTION"
echo " Date: $(date)"
echo " Root: $ROOT_DIR"
echo "========================================================================"

# STEP 1: Track B (Multi-Language Code & Query Traces)
echo ""
echo "========================================================================"
echo " STEP 1: Running Track B (Code Traces) with Peak Epoch Adjudication"
echo " Target metric to beat: $(cat "$ROOT_DIR/scratch/worktrees/track-b/.best_metric_code")"
echo "========================================================================"
cd "$ROOT_DIR/scratch/worktrees/track-b"
./runner_code.sh "scale(track-b): 2-epoch step-calibrated training with intermediate checkpoint selection"

echo ""
echo " Track B run finished."
echo " Current Track B .best_metric_code: $(cat "$ROOT_DIR/scratch/worktrees/track-b/.best_metric_code")"

# STEP 2: Track C (Medical & Clinical Reasoning - USMLE)
echo ""
echo "========================================================================"
echo " STEP 2: Running Track C (MedQA-USMLE) Baseline Execution"
echo " Dataset: 10,178 train vignettes, 1,273 test problems"
echo "========================================================================"
cd "$ROOT_DIR/scratch/worktrees/track-c"
./runner_med.sh "baseline: MedQA-USMLE clinical reasoning"

echo ""
echo " Track C run finished."
if [[ -f "$ROOT_DIR/scratch/worktrees/track-c/.best_metric_med" ]]; then
  echo " Current Track C .best_metric_med: $(cat "$ROOT_DIR/scratch/worktrees/track-c/.best_metric_med")"
fi

echo ""
echo "========================================================================"
echo " [AutoResearch Pipeline] ALL SCHEDULED RUNS COMPLETED"
echo " Date: $(date)"
echo "========================================================================"
