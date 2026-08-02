#!/usr/bin/env bash
# Driver: E5 run-to-run variance experiment.
# 5 patterns × 30 stratified queries × 3 replicates = 450 cells.
# Uses the canonical run_all_experiments.py with the new --run-tag flag.
# Outputs land in canonical results/experiments/base_p{N}_v{1,2,3}/{qid}.md.
set -e
cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate

# Stratified queries for variance (independent of Protocol A's stratification)
if [ ! -f data/variance_stratified.json ]; then
  python -c "
import json, random
random.seed(7)
qs = json.load(open('data/eval_queries_v2.json'))['queries']
by_src = {}
for q in qs:
    by_src.setdefault(q['source'], []).append(q)
selected = []
quota = 30 // len(by_src) + 1
for src, items in by_src.items():
    n = min(quota, len(items))
    selected.extend(random.sample(items, n))
selected = selected[:30]
ids = [q['id'] for q in selected]
print(f'Stratified {len(ids)} queries for variance')
with open('data/variance_stratified.json', 'w') as f:
    json.dump({'query_ids': ids, 'seed': 7}, f, indent=2)
print('Saved → data/variance_stratified.json')
"
fi

PATTERNS=(p0 p1 p4 p7 p10)
TAGS=(v1 v2 v3)
for tag in "${TAGS[@]}"; do
  for p in "${PATTERNS[@]}"; do
    echo "=== variance ${tag}: ${p} ==="
    python scripts/run_all_experiments.py \
      --pattern "${p}" \
      --run-tag "${tag}" \
      --query-ids-file data/variance_stratified.json \
      --base-only \
      --resume
  done
done
echo "=== E5 variance wave complete ==="
