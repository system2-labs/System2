#!/usr/bin/env bash
# runner_logic.sh - AutoResearch loop runner for System2 Track M (Formal & Deductive Logic).
# Protocol: better -> commit, worse -> revert.
#
# Usage: ./runner_logic.sh "<summary of the change just made>"
set -euo pipefail

SUMMARY="${1:-tuning}"
BEST_FILE=".best_metric_logic"
CHECKPOINT="./models/system2_logic.pt"
CANDIDATE="./models/candidate_logic.pt"

mkdir -p ./models

# Guardrail: Prevent Apple Silicon Metal/MPS unified memory runaway and swap thrashing
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.7}"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.5}"

# Train on Track M Formal Logic dataset (1,600 train samples, max-len 384)
python3 train.py --save "$CANDIDATE" --data "./data/logic" --train-size 1600 --max-len 384 --dynamic-len --epochs 5

# Judge on held-out logic test set (400 samples)
if ! OUT="$(python3 scripts/evaluate_logic.py "$CANDIDATE" --data "./data/logic" --max-len 384)"; then
  echo "[runner_logic] experiment INVALID (eval failed) -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

METRIC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["metric"])')"
ACC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["accuracy"])')"
BRIER="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["brier"])')"
ECE="$(echo "$OUT" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("ece", 0.0))')"

echo "[runner_logic] Evaluation: Metric=$METRIC | Acc=$ACC | Brier=$BRIER | ECE=$ECE"

if [[ ! -f "$BEST_FILE" ]]; then
  echo "$METRIC" > "$BEST_FILE"
  echo "[runner_logic] Track M baseline recorded"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add "$BEST_FILE" && git commit -m "feat(track-m): baseline formal logic deliberation, metric ${METRIC} (acc ${ACC}, brier ${BRIER})" || true
  exit 0
fi

BEST="$(cat "$BEST_FILE")"
IS_BETTER="$(python3 -c "print(1 if $METRIC > $BEST else 0)")"

if [[ "$IS_BETTER" -eq 1 ]]; then
  echo "[runner_logic] BETTER: $METRIC > $BEST -> commit"
  echo "$METRIC" > "$BEST_FILE"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add "$BEST_FILE" && git commit -m "feat(track-m): $SUMMARY, metric ${METRIC} (acc ${ACC}, brier ${BRIER})"
else
  echo "[runner_logic] WORSE or equal: $METRIC <= $BEST -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
fi
