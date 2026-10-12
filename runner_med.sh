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

# Rebuild the model on Track C data with dynamic length up to 8192
python3 train.py --save "$CANDIDATE" --data "./data/medical" --train-size 10178 --dynamic-len

# Judge on held-out MedQA USMLE test set (1,273 samples)
if ! OUT="$(python3 eval.py "$CANDIDATE" --data "./data/medical" --dynamic-len)"; then
  echo "[runner_med] experiment INVALID (eval failed) -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

METRIC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["metric"])')"
echo "[runner_med] new metric = $METRIC  ($OUT)"

if [[ ! -f "$BEST_FILE" ]]; then
  echo "$METRIC" > "$BEST_FILE"
  echo "[runner_med] Track C baseline recorded"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(system2-med): baseline, ${METRIC}" || true
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
  git add -A && git commit -m "feat(system2-med): $SUMMARY, ${METRIC}"
  git push origin track-c || echo "[runner_med] push failed (offline?)"
else
  echo "[runner_med] WORSE or equal: $METRIC <= $BEST -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
fi
