#!/usr/bin/env bash
# run_back_to_back.sh - Sequential back-to-back training execution across Track I, Track H, Track J
set -euo pipefail

echo "========================================================================"
echo " SYSTEM 2 SEQUENTIAL BACK-TO-BACK TRAINING PIPELINE"
echo " Date: $(date)"
echo "========================================================================"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Stage 1: Track I (Finance)
echo ""
echo ">>> [STAGE 1/3] Track I: Retrain on Expanded Finance Dataset"
echo "------------------------------------------------------------------------"
cd "$ROOT_DIR/scratch/worktrees/track-i"
./runner_finance.sh "retrain on expanded finance dataset" || echo "[STAGE 1] Runner finished with exit code $?"

# Stage 2: Track H (Legal)
echo ""
echo ">>> [STAGE 2/3] Track H: Trial H-3 (ensemble_temp 0.50, lambda_cand_margin 0.08)"
echo "------------------------------------------------------------------------"
cd "$ROOT_DIR/scratch/worktrees/track-h"
python3 -c '
import re
with open("train.py", "r") as f:
    c = f.read()
c = re.sub(r"(\"ensemble_temp\"\s*:\s*)([^,\n]+)", r"\g<1>0.50", c)
c = re.sub(r"(\"lambda_cand_margin\"\s*:\s*)([^,\n]+)", r"\g<1>0.08", c)
with open("train.py", "w") as f:
    f.write(c)
'
./runner_legal.sh "trial H-3: ensemble_temp 0.50 with lambda_cand_margin 0.08" || echo "[STAGE 2] Runner finished with exit code $?"

# Stage 3: Track J (Physics)
echo ""
echo ">>> [STAGE 3/3] Track J: Retrain on Grounded Expanded Physics Dataset"
echo "------------------------------------------------------------------------"
cd "$ROOT_DIR"
./runner_physics.sh "retrain on grounded physics dataset" || echo "[STAGE 3] Runner finished with exit code $?"

echo ""
echo "========================================================================"
echo " BACK-TO-BACK TRAINING COMPLETE"
echo " Date: $(date)"
echo "========================================================================"
