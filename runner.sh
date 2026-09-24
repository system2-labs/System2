#!/usr/bin/env bash
# runner.sh - AutoResearch loop runner for the System2 AG-News experiment.
# Protocol (mirrors test-training): better -> commit, worse -> revert.
#
# Usage: ./runner.sh "<summary of the change just made>"
set -euo pipefail

SUMMARY="${1:-tuning}"
BEST_FILE=".best_metric"
CHECKPOINT="./models/system2.pt"

# Rebuild the model with current hyperparams
python3 train.py --save "$CHECKPOINT"

# Judge
if ! OUT="$(python3 eval.py "$CHECKPOINT")"; then
  echo "[runner] experiment INVALID (eval failed) -> revert"
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

METRIC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["metric"])')"
echo "[runner] new metric = $METRIC  ($OUT)"

if [[ ! -f "$BEST_FILE" ]]; then
  echo "$METRIC" > "$BEST_FILE"
  echo "[runner] baseline recorded"
  git add -A && git commit -m "feat(system2): baseline, ${METRIC}" || true
  exit 0
fi

BEST="$(cat "$BEST_FILE")"
IS_BETTER="$(python3 -c "print(1 if $METRIC > $BEST else 0)")"

if [[ "$IS_BETTER" -eq 1 ]]; then
  echo "[runner] BETTER: $METRIC > $BEST -> commit"
  echo "$METRIC" > "$BEST_FILE"
  git add -A && git commit -m "feat(system2): $SUMMARY, ${METRIC}"
  git push || echo "[runner] push failed (offline?)"
else
  echo "[runner] WORSE or equal: $METRIC <= $BEST -> revert"
  git reset --hard HEAD 2>/dev/null || true
fi
