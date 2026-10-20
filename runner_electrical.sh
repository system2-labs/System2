#!/usr/bin/env bash
# runner_electrical.sh - AutoResearch loop runner for System2 Track D (Electrical, Electronics & Power Systems).
# Protocol: better -> commit, worse -> revert.
#
# Usage: ./runner_electrical.sh "<summary of the change just made>"
set -euo pipefail

SUMMARY="${1:-tuning}"
BEST_FILE=".best_metric_electrical"
CHECKPOINT="./models/system2_electrical.pt"
CANDIDATE="./models/candidate_electrical.pt"

mkdir -p ./models

# Guardrail: Prevent Apple Silicon Metal/MPS unified memory runaway and swap thrashing
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.7}"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.5}"

# Rebuild the model on Track D Pure Electrical data (In-Domain Causal Purity Law, capped at max-len 1024)
python3 train.py --save "$CANDIDATE" --data "./data/electrical_pure" --train-size 522 --max-len 1024 --dynamic-len

# Judge on held-out pure Electrical test set (134 samples)
if ! OUT="$(python3 eval.py "$CANDIDATE" --data "./data/electrical_pure" --max-len 1024 --dynamic-len)"; then
  echo "[runner_electrical] experiment INVALID (eval failed) -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

METRIC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["metric"])')"
ACC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["accuracy"])')"
BRIER="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["brier"])')"
ECE="$(echo "$OUT" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("ece", 0.0))')"

echo "[runner_electrical] Evaluation: Metric=$METRIC | Acc=$ACC | Brier=$BRIER | ECE=$ECE"

if [[ ! -f "$BEST_FILE" ]]; then
  echo "$METRIC" > "$BEST_FILE"
  echo "[runner_electrical] Track D baseline recorded"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(system2-electrical): baseline, metric ${METRIC} (acc ${ACC}, brier ${BRIER})" || true
  git push origin track-d || echo "[runner_electrical] push failed (offline?)"
  exit 0
fi

BEST="$(cat "$BEST_FILE")"
IS_BETTER="$(python3 -c "print(1 if $METRIC > $BEST else 0)")"

if [[ "$IS_BETTER" -eq 1 ]]; then
  echo "[runner_electrical] BETTER: $METRIC > $BEST -> commit"
  echo "$METRIC" > "$BEST_FILE"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(system2-electrical): $SUMMARY, metric ${METRIC} (acc ${ACC}, brier ${BRIER})"
  git push origin track-d || echo "[runner_electrical] push failed (offline?)"
else
  echo "[runner_electrical] WORSE or equal: $METRIC <= $BEST -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
fi
