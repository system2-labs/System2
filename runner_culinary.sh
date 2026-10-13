#!/usr/bin/env bash
# runner_culinary.sh - AutoResearch loop runner for System2 Track F (Culinary Chemistry & Food Science).
# Protocol: better -> commit, worse -> revert.
#
# Usage: ./runner_culinary.sh "<summary of the change just made>"
set -euo pipefail

SUMMARY="${1:-tuning}"
BEST_FILE=".best_metric_culinary"
CHECKPOINT="./models/system2_culinary.pt"
CANDIDATE="./models/candidate_culinary.pt"

# Rebuild the model on Track F Culinary data with dynamic length up to 8192
python3 train.py --save "$CANDIDATE" --data "./data/culinary" --dynamic-len

# Judge on held-out Culinary test set
if ! OUT="$(python3 eval.py "$CANDIDATE" --data "./data/culinary" --dynamic-len)"; then
  echo "[runner_culinary] experiment INVALID (eval failed) -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

METRIC="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["metric"])')"
echo "[runner_culinary] new metric = $METRIC  ($OUT)"

if [[ ! -f "$BEST_FILE" ]]; then
  echo "$METRIC" > "$BEST_FILE"
  echo "[runner_culinary] Track F baseline recorded"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(system2-culinary): baseline, ${METRIC}" || true
  git push -u origin track-f || echo "[runner_culinary] push failed (offline?)"
  exit 0
fi

BEST="$(cat "$BEST_FILE")"
IS_BETTER="$(python3 -c "print(1 if $METRIC > $BEST else 0)")"

if [[ "$IS_BETTER" -eq 1 ]]; then
  echo "[runner_culinary] BETTER: $METRIC > $BEST -> commit"
  echo "$METRIC" > "$BEST_FILE"
  cp "$CANDIDATE" "$CHECKPOINT"
  rm -f "$CANDIDATE"
  git add -A && git commit -m "feat(system2-culinary): $SUMMARY, ${METRIC}"
  git push origin track-f || echo "[runner_culinary] push failed (offline?)"
else
  echo "[runner_culinary] WORSE or equal: $METRIC <= $BEST -> revert"
  rm -f "$CANDIDATE"
  git reset --hard HEAD 2>/dev/null || true
fi
