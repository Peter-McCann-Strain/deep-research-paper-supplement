#!/usr/bin/env bash
# Wave 0 — Oracle Tier-1 (pooled-existing frozen corpus) generation.
# All GPT-4o patterns x the 30-query variance-stratified subset, on free PTU.
# Ordered fast/decisive-first (P0/P1/P4 cluster leaders) so an early read-out is available;
# slowest patterns (P6/P7) last. Resumable: re-running skips completed cells.
set -e
cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate

export SEARCH_BACKEND=oracle
export ORACLE_CORPUS_PATH=data/oracle_corpus_t1.json

PATTERNS=(p0 p1 p4 p5 p2 p3 p8 p6 p7)
for p in "${PATTERNS[@]}"; do
  echo "=================== oracle T1: ${p} ($(date '+%H:%M:%S')) ==================="
  python scripts/run_all_experiments.py \
    --pattern "${p}" \
    --retriever oracle \
    --run-tag t1 \
    --query-ids-file data/variance_stratified.json \
    --base-only \
    --resume
done
echo "=================== oracle T1 generation COMPLETE ($(date '+%H:%M:%S')) ==================="
