#!/usr/bin/env bash
# runner_math.sh - AutoResearch loop runner for System2 Track G (Multi-Step Mathematical Reasoning / GSM8K).
# Protocol: better -> commit, worse -> revert.
#
# Usage: ./runner_math.sh "<summary of the change just made>"
set -euo pipefail

SUMMARY="${1:-tuning}"
BEST_FILE=".best_metric_math"
CHECKPOINT="./models/system2_math.pt"
CANDIDATE="./models/candidate_math.pt"

# Rebuild the model on Track G Math data with dynamic length up to 8192
python3 -u train.py --save "$CANDIDATE" --data "./data/math" --dynamic-len

# Judge on held-out Math test set
if ! OUT="$(python3 -u eval.py "$CANDIDATE" --data "./data/math" --dynamic-len)"; then
  echo "[runner_math] experiment INVALID (eval failed) -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

METRIC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["metric"])')"
echo "[runner_math] new metric = $METRIC  ($OUT)"

if [[ ! -f "$BEST_FILE" ]]; then
  echo "$METRIC" > "$BEST_FILE"
  echo "[runner_math] Track G baseline recorded"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(system2-math): baseline, ${METRIC}" || true
  git push -u origin track-g || echo "[runner_math] push failed (offline?)"
  exit 0
fi

BEST="$(cat "$BEST_FILE")"
IS_BETTER="$(python3 -c "print(1 if $METRIC > $BEST else 0)")"

if [[ "$IS_BETTER" -eq 1 ]]; then
  echo "[runner_math] BETTER: $METRIC > $BEST -> commit"
  echo "$METRIC" > "$BEST_FILE"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(system2-math): $SUMMARY, ${METRIC}"
  git push origin track-g || echo "[runner_math] push failed (offline?)"
else
  echo "[runner_math] WORSE or equal: $METRIC <= $BEST -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
fi
