#!/usr/bin/env bash
# run_sequential_tracks_e_f.sh - Sequential AutoResearch execution across Track E and Track F.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "========================================================================"
echo " [AutoResearch Pipeline] STARTING SEQUENTIAL EXECUTION: TRACKS E & F"
echo " Date: $(date)"
echo " Root: $ROOT_DIR"
echo "========================================================================"

# STEP 1: Track E (Automotive & HVAC Mechanical Diagnostics)
echo ""
echo "========================================================================"
echo " STEP 1: Running Track E (Automotive Diagnostics) Baseline"
echo " Dataset: 7,993 train, 1,999 held-out test (OBD-II, STFT, fault trees)"
echo " Worktree: $ROOT_DIR/scratch/worktrees/track-e"
echo "========================================================================"
cd "$ROOT_DIR/scratch/worktrees/track-e"
mkdir -p models data
ln -sfn "$ROOT_DIR/data/auto" data/auto
./runner_auto.sh "baseline: automotive and HVAC diagnostic fault trees"

echo ""
echo " Track E run finished."
if [[ -f "$ROOT_DIR/scratch/worktrees/track-e/.best_metric_auto" ]]; then
  echo " Current Track E .best_metric_auto: $(cat "$ROOT_DIR/scratch/worktrees/track-e/.best_metric_auto")"
  # Copy checkpoint to main models directory
  cp "$ROOT_DIR/scratch/worktrees/track-e/models/system2_auto.pt" "$ROOT_DIR/models/system2_auto.pt" || true
fi

# STEP 2: Track F (Culinary Chemistry & Food Science)
echo ""
echo "========================================================================"
echo " STEP 2: Running Track F (Culinary Chemistry) Baseline"
echo " Dataset: 2,675 train, 669 held-out test (MMLU nutrition + RecipeQA)"
echo " Worktree: $ROOT_DIR/scratch/worktrees/track-f"
echo "========================================================================"
cd "$ROOT_DIR/scratch/worktrees/track-f"
mkdir -p models data
ln -sfn "$ROOT_DIR/data/culinary" data/culinary
./runner_culinary.sh "baseline: culinary chemistry and reaction mechanics"

echo ""
echo " Track F run finished."
if [[ -f "$ROOT_DIR/scratch/worktrees/track-f/.best_metric_culinary" ]]; then
  echo " Current Track F .best_metric_culinary: $(cat "$ROOT_DIR/scratch/worktrees/track-f/.best_metric_culinary")"
  # Copy checkpoint to main models directory
  cp "$ROOT_DIR/scratch/worktrees/track-f/models/system2_culinary.pt" "$ROOT_DIR/models/system2_culinary.pt" || true
fi

echo ""
echo "========================================================================"
echo " [AutoResearch Pipeline] BOTH TRACKS E & F COMPLETED SUCCESSFULLY"
echo " Date: $(date)"
echo "========================================================================"
