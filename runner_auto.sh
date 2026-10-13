#!/usr/bin/env bash
# runner_auto.sh - AutoResearch loop runner for System2 Track E (Automotive & HVAC Mechanical Diagnostics).
# Protocol: better -> commit, worse -> revert.
#
# Usage: ./runner_auto.sh "<summary of the change just made>"
set -euo pipefail

SUMMARY="${1:-tuning}"
BEST_FILE=".best_metric_auto"
CHECKPOINT="./models/system2_auto.pt"
CANDIDATE="./models/candidate_auto.pt"

# Rebuild the model on Track E Automotive data with dynamic length up to 8192
python3 train.py --save "$CANDIDATE" --data "./data/auto" --dynamic-len

# Judge on held-out Automotive test set
if ! OUT="$(python3 eval.py "$CANDIDATE" --data "./data/auto" --dynamic-len)"; then
  echo "[runner_auto] experiment INVALID (eval failed) -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

METRIC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["metric"])')"
echo "[runner_auto] new metric = $METRIC  ($OUT)"

if [[ ! -f "$BEST_FILE" ]]; then
  echo "$METRIC" > "$BEST_FILE"
  echo "[runner_auto] Track E baseline recorded"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(system2-auto): baseline, ${METRIC}" || true
  git push -u origin track-e || echo "[runner_auto] push failed (offline?)"
  exit 0
fi

BEST="$(cat "$BEST_FILE")"
IS_BETTER="$(python3 -c "print(1 if $METRIC > $BEST else 0)")"

if [[ "$IS_BETTER" -eq 1 ]]; then
  echo "[runner_auto] BETTER: $METRIC > $BEST -> commit"
  echo "$METRIC" > "$BEST_FILE"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(system2-auto): $SUMMARY, ${METRIC}"
  git push origin track-e || echo "[runner_auto] push failed (offline?)"
else
  echo "[runner_auto] WORSE or equal: $METRIC <= $BEST -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
fi
