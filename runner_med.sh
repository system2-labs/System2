#!/usr/bin/env bash
# runner_med.sh - AutoResearch loop runner for System2 Track C (Medical & Clinical Reasoning).
# Protocol: better -> commit, worse -> revert.
#
# Usage: ./runner_med.sh "<summary of the change just made>"
set -euo pipefail

SUMMARY="${1:-tuning}"
BEST_FILE=".best_metric_med"
CHECKPOINT="./models/system2_med.pt"
CANDIDATE="./models/candidate_med.pt"

mkdir -p ./models

# Guardrail: Prevent Apple Silicon Metal/MPS unified memory runaway and swap thrashing
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.7}"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.5}"

# Rebuild the model on Track C Expanded Med data (13,000 train samples, max-len 256 covers 100% of MedMCQA)
python3 train.py --save "$CANDIDATE" --data "./data/med_expanded" --train-size 13000 --epochs 3 --max-len 256 --dynamic-len

# Judge on held-out MedMCQA clinical test set (2,000 samples)
if ! OUT="$(python3 eval.py "$CANDIDATE" --data "./data/med_expanded" --max-len 256 --dynamic-len)"; then
  echo "[runner_med] experiment INVALID (eval failed) -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

METRIC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["metric"])')"
ACC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["accuracy"])')"
BRIER="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["brier"])')"
ECE="$(echo "$OUT" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("ece", 0.0))')"

echo "[runner_med] Evaluation: Metric=$METRIC | Acc=$ACC | Brier=$BRIER | ECE=$ECE"

if [[ ! -f "$BEST_FILE" ]]; then
  echo "$METRIC" > "$BEST_FILE"
  echo "[runner_med] Track C baseline recorded"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(system2-med): baseline, metric ${METRIC} (acc ${ACC}, brier ${BRIER})" || true
  git push origin track-c || echo "[runner_med] push failed (offline?)"
  exit 0
fi

BEST="$(cat "$BEST_FILE")"
IS_BETTER="$(python3 -c "print(1 if $METRIC > $BEST else 0)")"

if [[ "$IS_BETTER" -eq 1 ]]; then
  echo "[runner_med] BETTER: $METRIC > $BEST -> commit"
  echo "$METRIC" > "$BEST_FILE"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(system2-med): $SUMMARY, metric ${METRIC} (acc ${ACC}, brier ${BRIER})"
  git push origin track-c || echo "[runner_med] push failed (offline?)"
else
  echo "[runner_med] WORSE or equal: $METRIC <= $BEST -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
fi
