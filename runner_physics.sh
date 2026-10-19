#!/usr/bin/env bash
# runner_physics.sh - AutoResearch loop runner for System2 Track J (Pure Physics & Astrophysics).
# Protocol: better -> commit, worse -> revert.
#
# Usage: ./runner_physics.sh "<summary of the change just made>"
set -euo pipefail

SUMMARY="${1:-tuning}"
BEST_FILE=".best_metric_physics"
CHECKPOINT="./models/system2_physics.pt"
CANDIDATE="./models/candidate_physics.pt"

mkdir -p ./models

# Guardrail: Prevent Apple Silicon Metal/MPS unified memory runaway and swap thrashing
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.7}"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.5}"

# Train on Track J Physics dataset (553 train samples, capped at max-len 1024)
python3 train.py --save "$CANDIDATE" --data "./data/physics" --train-size 553 --max-len 1024 --dynamic-len

# Judge on held-out physics & astrophysics test set (139 samples)
if ! OUT="$(python3 eval.py "$CANDIDATE" --data "./data/physics" --max-len 1024 --dynamic-len)"; then
  echo "[runner_physics] experiment INVALID (eval failed) -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

METRIC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["metric"])')"
ACC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["accuracy"])')"
BRIER="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["brier"])')"
ECE="$(echo "$OUT" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("ece", 0.0))')"

echo "[runner_physics] Evaluation: Metric=$METRIC | Acc=$ACC | Brier=$BRIER | ECE=$ECE"

if [[ ! -f "$BEST_FILE" ]]; then
  echo "$METRIC" > "$BEST_FILE"
  echo "[runner_physics] Track J baseline recorded"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(track-j): baseline physics reasoning, metric ${METRIC} (acc ${ACC}, brier ${BRIER})" || true
  git push || echo "[runner_physics] push failed (offline?)"
  exit 0
fi

BEST="$(cat "$BEST_FILE")"
IS_BETTER="$(python3 -c "print(1 if $METRIC > $BEST else 0)")"

if [[ "$IS_BETTER" -eq 1 ]]; then
  echo "[runner_physics] BETTER: $METRIC > $BEST -> commit"
  echo "$METRIC" > "$BEST_FILE"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(track-j): $SUMMARY, metric ${METRIC} (acc ${ACC}, brier ${BRIER})"
  git push || echo "[runner_physics] push failed (offline?)"
else
  echo "[runner_physics] WORSE or equal: $METRIC <= $BEST -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
fi
