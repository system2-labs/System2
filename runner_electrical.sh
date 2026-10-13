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

# Rebuild the model on Track D Electrical data with dynamic length up to 8192
python3 train.py --save "$CANDIDATE" --data "./data/electrical" --dynamic-len

# Judge on held-out Electrical test set
if ! OUT="$(python3 eval.py "$CANDIDATE" --data "./data/electrical" --dynamic-len)"; then
  echo "[runner_electrical] experiment INVALID (eval failed) -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

METRIC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["metric"])')"
echo "[runner_electrical] new metric = $METRIC  ($OUT)"

if [[ ! -f "$BEST_FILE" ]]; then
  echo "$METRIC" > "$BEST_FILE"
  echo "[runner_electrical] Track D baseline recorded"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(system2-electrical): baseline, ${METRIC}" || true
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
  git add -A && git commit -m "feat(system2-electrical): $SUMMARY, ${METRIC}"
  git push origin track-d || echo "[runner_electrical] push failed (offline?)"
else
  echo "[runner_electrical] WORSE or equal: $METRIC <= $BEST -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
fi
