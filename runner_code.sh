#!/usr/bin/env bash
# runner_code.sh - AutoResearch loop runner for System2 Track B (Code Execution).
# Protocol: better -> commit, worse -> revert.
#
# Usage: ./runner_code.sh "<summary of the change just made>"
set -euo pipefail

SUMMARY="${1:-tuning}"
BEST_FILE=".best_metric_code"
CHECKPOINT="./models/system2_code.pt"
CANDIDATE="./models/candidate_code.pt"

MAX_LEN="${MAX_LEN:-256}"

# Rebuild the model with current hyperparams on Track B data
python3 train.py --save "$CANDIDATE" --data "./data/code_traces" --max-len "$MAX_LEN"

# Judge on held-out code execution test set (4,067 samples)
if ! OUT="$(python3 eval.py "$CANDIDATE" --data "./data/code_traces")"; then
  echo "[runner_code] experiment INVALID (eval failed) -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

METRIC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["metric"])')"
echo "[runner_code] new metric = $METRIC  ($OUT)"

if [[ ! -f "$BEST_FILE" ]]; then
  echo "$METRIC" > "$BEST_FILE"
  echo "[runner_code] Track B baseline recorded"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(system2-code): baseline, ${METRIC}" || true
  exit 0
fi

BEST="$(cat "$BEST_FILE")"
IS_BETTER="$(python3 -c "print(1 if $METRIC > $BEST else 0)")"

if [[ "$IS_BETTER" -eq 1 ]]; then
  echo "[runner_code] BETTER: $METRIC > $BEST -> commit"
  echo "$METRIC" > "$BEST_FILE"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(system2-code): $SUMMARY, ${METRIC}"
  git push origin track-b || echo "[runner_code] push failed (offline?)"
else
  echo "[runner_code] WORSE or equal: $METRIC <= $BEST -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
fi
