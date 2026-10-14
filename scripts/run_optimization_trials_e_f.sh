#!/usr/bin/env bash
# run_optimization_trials_e_f.sh - AutoResearch Hyperparameter Optimization Trials for Tracks E & F.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE_E="$ROOT_DIR/scratch/worktrees/track-e"
WORKTREE_F="$ROOT_DIR/scratch/worktrees/track-f"

echo "========================================================================"
echo " [AutoResearch Loop] STARTING HYPERPARAMETER OPTIMIZATION TRIALS: TRACKS E & F"
echo " Date: $(date)"
echo " Root: $ROOT_DIR"
echo "========================================================================"

# ==============================================================================
# SECTION 1: Track F (Culinary Chemistry & Food Science)
# Current Baseline: 0.7475 (Accuracy: 67.71%, Brier: 0.1822)
# ==============================================================================
echo ""
echo "========================================================================"
echo " SECTION 1: Track F (Culinary Chemistry) Optimization Trials"
echo " Worktree: $WORKTREE_F"
echo " Current Baseline: $(cat "$WORKTREE_F/.best_metric_culinary")"
echo "========================================================================"
cd "$WORKTREE_F"

# --- Trial F-1: Extended Convergence Horizon (7 Epochs + Warmup) ---
echo ""
echo ">>> [Trial F-1] Testing 7 Epochs + 5% Warmup (Allowing deep loss convergence)..."
python3 -c '
with open("train.py", "r") as f:
    text = f.read()
text = text.replace("\"epochs\": 5,", "\"epochs\": 7,")
text = text.replace("\"warmup_ratio\": 0.0,", "\"warmup_ratio\": 0.05,")
with open("train.py", "w") as f:
    f.write(text)
'
./runner_culinary.sh "trial F-1: 7 epochs + 5% warmup"
echo ">>> [Trial F-1 Finished] Current Metric: $(cat "$WORKTREE_F/.best_metric_culinary")"

# --- Trial F-2: Judger Calibration & Smoothing Sharpening ---
echo ""
echo ">>> [Trial F-2] Testing Calibration Sharpening (smoothing 0.12, ensemble_temp 0.70)..."
python3 -c '
with open("train.py", "r") as f:
    text = f.read()
text = text.replace("\"judge_cal_smoothing\": 0.17,", "\"judge_cal_smoothing\": 0.12,")
text = text.replace("\"ensemble_temp\": 0.74,", "\"ensemble_temp\": 0.70,")
with open("train.py", "w") as f:
    f.write(text)
'
./runner_culinary.sh "trial F-2: calibration smoothing 0.12, temp 0.70"
echo ">>> [Trial F-2 Finished] Current Metric: $(cat "$WORKTREE_F/.best_metric_culinary")"

# If improved, sync weights to main repository
if [[ -f "$WORKTREE_F/models/system2_culinary.pt" ]]; then
    echo "Syncing updated Track F checkpoint to $ROOT_DIR/models/system2_culinary.pt..."
    cp "$WORKTREE_F/models/system2_culinary.pt" "$ROOT_DIR/models/system2_culinary.pt"
fi

# ==============================================================================
# SECTION 2: Track E (Automotive & HVAC Diagnostics)
# Current Baseline: 0.9809 (Accuracy: 99.50%, Brier: 0.0332)
# ==============================================================================
echo ""
echo "========================================================================"
echo " SECTION 2: Track E (Automotive Diagnostics) Optimization Trials"
echo " Worktree: $WORKTREE_E"
echo " Current Baseline: $(cat "$WORKTREE_E/.best_metric_auto")"
echo "========================================================================"
cd "$WORKTREE_E"

# --- Trial E-1: Brier Calibration Sharpening ---
echo ""
echo ">>> [Trial E-1] Testing Brier Calibration Sharpening (smoothing 0.12, ensemble_temp 0.68)..."
python3 -c '
with open("train.py", "r") as f:
    text = f.read()
text = text.replace("\"judge_cal_smoothing\": 0.17,", "\"judge_cal_smoothing\": 0.12,")
text = text.replace("\"ensemble_temp\": 0.74,", "\"ensemble_temp\": 0.68,")
with open("train.py", "w") as f:
    f.write(text)
'
./runner_auto.sh "trial E-1: calibration smoothing 0.12, temp 0.68"
echo ">>> [Trial E-1 Finished] Current Metric: $(cat "$WORKTREE_E/.best_metric_auto")"

# --- Trial E-2: Decision Margin Scaling ---
echo ""
echo ">>> [Trial E-2] Testing Margin Scaling (label_smoothing 0.10, lambda_judge_cal 0.08)..."
python3 -c '
with open("train.py", "r") as f:
    text = f.read()
text = text.replace("\"label_smoothing\": 0.15,", "\"label_smoothing\": 0.10,")
text = text.replace("\"lambda_judge_cal\": 0.06,", "\"lambda_judge_cal\": 0.08,")
with open("train.py", "w") as f:
    f.write(text)
'
./runner_auto.sh "trial E-2: label smoothing 0.10, lambda_judge_cal 0.08"
echo ">>> [Trial E-2 Finished] Current Metric: $(cat "$WORKTREE_E/.best_metric_auto")"

# If improved, sync weights to main repository
if [[ -f "$WORKTREE_E/models/system2_auto.pt" ]]; then
    echo "Syncing updated Track E checkpoint to $ROOT_DIR/models/system2_auto.pt..."
    cp "$WORKTREE_E/models/system2_auto.pt" "$ROOT_DIR/models/system2_auto.pt"
fi

# ==============================================================================
# SECTION 3: Update Manifest and Map
# ==============================================================================
echo ""
echo "========================================================================"
echo " SECTION 3: Introspecting Subspaces & Updating Weight Manifest"
echo "========================================================================"
cd "$ROOT_DIR"
python3 scripts/map_specialized_weights.py

echo ""
echo "========================================================================"
echo " [AutoResearch Loop] ALL OPTIMIZATION TRIALS COMPLETED SUCCESSFULLY"
echo " Date: $(date)"
echo " Final Track E Metric: $(cat "$WORKTREE_E/.best_metric_auto")"
echo " Final Track F Metric: $(cat "$WORKTREE_F/.best_metric_culinary")"
echo "========================================================================"
