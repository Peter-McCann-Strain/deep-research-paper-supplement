#!/usr/bin/env bash
# Driver: run Protocol A Tavily wave across the 6 high-impact patterns
# (P0/P1 = low-placeholder controls; P3/P4/P5/P8 = high-placeholder culprits).
# 29 queries × 6 patterns = 174 Tavily cells.
set -e
cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate

PATTERNS=(p0 p1 p3 p4 p5 p8)
for p in "${PATTERNS[@]}"; do
  echo "=== Tavily wave: ${p} ==="
  python scripts/run_all_experiments.py \
    --pattern "${p}" \
    --retriever tavily \
    --query-ids-file data/protocol_a_stratified_v2.json \
    --base-only \
    --resume
done
echo "=== Protocol A Tavily wave complete ==="
