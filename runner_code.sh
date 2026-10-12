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

# Rebuild the model with current hyperparams on Track B data
python3 train.py --save "$CANDIDATE" --data "./data/code_traces" --dynamic-len --max-len 512

# Evaluate all candidate checkpoints (per-epoch + final) to capture peak convergence
BEST_CANDIDATE_METRIC="-1.0"
WINNING_CKPT=""
WINNING_OUT=""

shopt -s nullglob
CKPTS=("$CANDIDATE" "$CANDIDATE".ckpt_epoch_*.pt)
shopt -u nullglob

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "[runner_code] experiment INVALID (no candidate checkpoints generated) -> revert"
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

for CKPT in "${CKPTS[@]}"; do
  echo "[runner_code] Evaluating candidate checkpoint: $CKPT"
  if OUT="$(python3 eval.py "$CKPT" --data "./data/code_traces" --dynamic-len)"; then
    M="$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["metric"])')"
    echo "[runner_code] $CKPT scored metric = $M ($OUT)"
    IS_HIGHER="$(python3 -c "print(1 if $M > $BEST_CANDIDATE_METRIC else 0)")"
    if [[ "$IS_HIGHER" -eq 1 ]]; then
      BEST_CANDIDATE_METRIC="$M"
      WINNING_CKPT="$CKPT"
      WINNING_OUT="$OUT"
    fi
  else
    echo "[runner_code] Warning: eval failed for $CKPT"
  fi
done

if [[ -z "$WINNING_CKPT" ]]; then
  echo "[runner_code] experiment INVALID (all candidate evals failed) -> revert"
  rm -f "$CANDIDATE"*
  git reset --hard HEAD 2>/dev/null || true
  exit 1
fi

echo "[runner_code] Winning candidate is $WINNING_CKPT with metric $BEST_CANDIDATE_METRIC"
if [[ "$WINNING_CKPT" != "$CANDIDATE" ]]; then
  cp "$WINNING_CKPT" "$CANDIDATE"
fi
rm -f "$CANDIDATE".ckpt_epoch_*.pt

METRIC="$BEST_CANDIDATE_METRIC"
OUT="$WINNING_OUT"
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
