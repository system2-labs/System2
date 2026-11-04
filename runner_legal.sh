#!/usr/bin/env bash
# runner_legal.sh - AutoResearch loop runner for System2 Track H (Legal Reasoning & Compliance).
# Protocol: better -> commit, worse -> revert.
#
# Usage: ./runner_legal.sh "<summary of the change just made>"
set -euo pipefail

SUMMARY="${1:-tuning}"
BEST_FILE=".best_metric_legal"
CHECKPOINT="./models/system2_legal.pt"
CANDIDATE="./models/candidate_legal.pt"

mkdir -p ./models

# Guardrail: Prevent Apple Silicon Metal/MPS unified memory runaway and swap thrashing
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.7}"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.5}"

# Train on Track H Multi-Jurisdiction Legal dataset (5,089 train samples, max-len 384, 5 epochs)
python3 train.py --save "$CANDIDATE" --data "./data/legal_multi_expanded" --train-size 5089 --epochs 5 --max-len 384 --dynamic-len

# Judge on held-out multi-jurisdiction legal test set (899 samples)
if ! OUT="$(python3 eval.py "$CANDIDATE" --data "./data/legal_multi_expanded" --max-len 384 --dynamic-len --ensemble-temp 0.50 --head-weights 1.0,1.0,1.1,1.0,0.5)"; then
  echo "[runner_legal] experiment INVALID (eval failed) -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

METRIC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["metric"])')"
ACC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["accuracy"])')"
BRIER="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["brier"])')"
ECE="$(echo "$OUT" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("ece", 0.0))')"

echo "[runner_legal] Evaluation: Metric=$METRIC | Acc=$ACC | Brier=$BRIER | ECE=$ECE"

if [[ ! -f "$BEST_FILE" ]]; then
  echo "$METRIC" > "$BEST_FILE"
  echo "[runner_legal] Track H baseline recorded"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(track-h): baseline legal reasoning, metric ${METRIC} (acc ${ACC}, brier ${BRIER})" || true
  git push origin track-h || echo "[runner_legal] push failed (offline?)"
  exit 0
fi

BEST="$(cat "$BEST_FILE")"
IS_BETTER="$(python3 -c "print(1 if $METRIC > $BEST else 0)")"

if [[ "$IS_BETTER" -eq 1 ]]; then
  echo "[runner_legal] BETTER: $METRIC > $BEST -> commit"
  echo "$METRIC" > "$BEST_FILE"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(track-h): $SUMMARY, metric ${METRIC} (acc ${ACC}, brier ${BRIER})"
  git push origin track-h || echo "[runner_legal] push failed (offline?)"
else
  echo "[runner_legal] WORSE or equal: $METRIC <= $BEST -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
fi
