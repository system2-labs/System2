#!/usr/bin/env bash
# runner_safety.sh - AutoResearch loop runner for System2 Track K (Agent Gatekeeper & Command Safety).
# Protocol: better -> commit, worse -> revert.
#
# Usage: ./runner_safety.sh "<summary of the change just made>"
set -euo pipefail

SUMMARY="${1:-tuning}"
BEST_FILE=".best_metric_safety"
CHECKPOINT="./models/system2_safety.pt"
CANDIDATE="./models/candidate_safety.pt"

mkdir -p ./models

# Guardrail: Prevent Apple Silicon Metal/MPS unified memory runaway and swap thrashing
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.7}"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.5}"

# Train on Track K Augmented Safety dataset (1,600 train samples, max-len 384)
python3 train.py --save "$CANDIDATE" --data "./data/safety" --train-size 1600 --max-len 384 --dynamic-len --epochs 5

# Judge on held-out safety test set (400 samples)
if ! OUT="$(python3 scripts/evaluate_safety.py "$CANDIDATE" --data "./data/safety" --max-len 384)"; then
  echo "[runner_safety] experiment INVALID (eval failed) -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

METRIC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["metric"])')"
ACC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["accuracy"])')"
BRIER="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["brier"])')"
ECE="$(echo "$OUT" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("ece", 0.0))')"

echo "[runner_safety] Evaluation: Metric=$METRIC | Acc=$ACC | Brier=$BRIER | ECE=$ECE"

if [[ ! -f "$BEST_FILE" ]]; then
  echo "$METRIC" > "$BEST_FILE"
  echo "[runner_safety] Track K baseline recorded"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(track-k): baseline safety deliberation, metric ${METRIC} (acc ${ACC}, brier ${BRIER})" || true
  git push origin track-k || echo "[runner_safety] push failed (offline?)"
  exit 0
fi

BEST="$(cat "$BEST_FILE")"
IS_BETTER="$(python3 -c "print(1 if $METRIC > $BEST else 0)")"

if [[ "$IS_BETTER" -eq 1 ]]; then
  echo "[runner_safety] BETTER: $METRIC > $BEST -> commit"
  echo "$METRIC" > "$BEST_FILE"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(track-k): $SUMMARY, metric ${METRIC} (acc ${ACC}, brier ${BRIER})"
  git push origin track-k || echo "[runner_safety] push failed (offline?)"
else
  echo "[runner_safety] WORSE or equal: $METRIC <= $BEST -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
fi
