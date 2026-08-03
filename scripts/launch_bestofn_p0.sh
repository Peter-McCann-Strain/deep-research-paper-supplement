#!/usr/bin/env bash
# Extend the best-of-N(P0) compute-matched control: generate 8 more independent P0
# samples (v4..v11) on the 30 variance queries via live retrieval on the free PTU,
# then GPT-5.2-judge them. Combined with base_p0 + v1/v2/v3 this gives N=12 samples,
# enough for best-of-12 and a realizable (self-consistency) selector.
set -e
cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate

echo "=================== best-of-N P0 generation start ($(date '+%m-%d %H:%M')) ==================="
for v in v4 v5 v6 v7 v8 v9 v10 v11; do
  echo "--- P0 sample $v ($(date '+%H:%M')) ---"
  python scripts/run_all_experiments.py --pattern p0 --run-tag "$v" \
    --query-ids-file data/variance_stratified.json --base-only --resume
done
echo "=================== generation done; judging with GPT-5.2 ($(date '+%H:%M')) ==================="
PATS="base_p0_v4,base_p0_v5,base_p0_v6,base_p0_v7,base_p0_v8,base_p0_v9,base_p0_v10,base_p0_v11"
for i in $(seq 1 10); do
  python scripts/run_gpt52_judge.py --patterns-raw "$PATS" --resume 2>&1 | grep -iE "judged|pending|complete|Total" | tail -4
  nj=$(find results/judge_gpt52 -path '*base_p0_v*' -name '*.json' 2>/dev/null | wc -l)
  echo "  judged base_p0_v* total: ${nj}"
  ng=$(find results/experiments -path '*base_p0_v[4-9]*' -o -path '*base_p0_v1[01]*' 2>/dev/null | grep -c '\.md$' || true)
  if [ "$nj" -ge 240 ]; then echo "all v4-v11 judged"; break; fi
  sleep 60
done
echo "=================== best-of-N P0 COMPLETE ($(date '+%m-%d %H:%M')) ==================="
