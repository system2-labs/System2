#!/usr/bin/env bash
# runner_eval.sh - AutoResearch loop runner for System2 Track L (Rubric, Faithfulness & Hallucination Meta-Judge).
# Protocol: better -> commit, worse -> revert.
#
# Usage: ./runner_eval.sh "<summary of the change just made>"
set -euo pipefail

SUMMARY="${1:-tuning}"
BEST_FILE=".best_metric_eval"
CHECKPOINT="./models/system2_eval.pt"
CANDIDATE="./models/candidate_eval.pt"

mkdir -p ./models

# Guardrail: Prevent Apple Silicon Metal/MPS unified memory runaway and swap thrashing
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.7}"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.5}"

# Train on Track L Eval Meta-Judge dataset (1,600 train samples, max-len 384)
python3 train.py --save "$CANDIDATE" --data "./data/eval" --train-size 1600 --max-len 384 --dynamic-len --epochs 5

# Judge on held-out eval test set (400 samples)
if ! OUT="$(python3 scripts/evaluate_eval.py "$CANDIDATE" --data "./data/eval" --max-len 384)"; then
  echo "[runner_eval] experiment INVALID (eval failed) -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

METRIC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["metric"])')"
ACC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["accuracy"])')"
BRIER="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["brier"])')"
ECE="$(echo "$OUT" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("ece", 0.0))')"

echo "[runner_eval] Evaluation: Metric=$METRIC | Acc=$ACC | Brier=$BRIER | ECE=$ECE"

if [[ ! -f "$BEST_FILE" ]]; then
  echo "$METRIC" > "$BEST_FILE"
  echo "[runner_eval] Track L baseline recorded"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add "$BEST_FILE" && git commit -m "feat(track-l): baseline eval meta-judge deliberation, metric ${METRIC} (acc ${ACC}, brier ${BRIER})" || true
  exit 0
fi

BEST="$(cat "$BEST_FILE")"
IS_BETTER="$(python3 -c "print(1 if $METRIC > $BEST else 0)")"

if [[ "$IS_BETTER" -eq 1 ]]; then
  echo "[runner_eval] BETTER: $METRIC > $BEST -> commit"
  echo "$METRIC" > "$BEST_FILE"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add "$BEST_FILE" && git commit -m "feat(track-l): $SUMMARY, metric ${METRIC} (acc ${ACC}, brier ${BRIER})"
else
  echo "[runner_eval] WORSE or equal: $METRIC <= $BEST -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
fi
