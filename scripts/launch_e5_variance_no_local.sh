#!/usr/bin/env bash
# E5 variance launcher — GPT-4o patterns only (skips P10 because P12 RL holds the GPU).
# Run scripts/launch_e5_variance.sh after P12 finishes to fill in P10 cells.
# Uses --resume so re-runs only fill in missing cells.

set +e  # do NOT abort on first failure (let other patterns continue)
cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate

if [ ! -f data/variance_stratified.json ]; then
  echo "ERROR: data/variance_stratified.json missing — run launch_e5_variance.sh once first to generate it"
  exit 1
fi

PATTERNS=(p0 p1 p4 p7)   # P10 deferred — GPU held by P12 RL training
TAGS=(v1 v2 v3)
for tag in "${TAGS[@]}"; do
  for p in "${PATTERNS[@]}"; do
    echo "=== variance ${tag}: ${p} ==="
    python scripts/run_all_experiments.py \
      --pattern "${p}" \
      --run-tag "${tag}" \
      --query-ids-file data/variance_stratified.json \
      --base-only \
      --resume || echo "  ${tag}/${p} FAILED — continuing"
  done
done
echo "=== E5 GPT-4o variance wave complete; P10 cells deferred until P12 RL finishes ==="
