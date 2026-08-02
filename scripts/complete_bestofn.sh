#!/usr/bin/env bash
# Wait for the v4-v8 (and any further v9-v11) P0 judging to finish, then rebuild dataframes
# and recompute best-of-N with all available samples. Writes the result to logs/bestofn_result.log.
cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate
for i in $(seq 1 60); do
  nj=$(find results/judge_gpt52/base_p0_v[4-8] -name '*.json' 2>/dev/null | wc -l)
  echo "[$(date '+%H:%M')] v4-v8 judged: ${nj}/150"
  if [ "$nj" -ge 150 ]; then break; fi
  sleep 90
done
echo "=== rebuild dataframes + best-of-N ==="
python scripts/build_analysis_dataframes.py >/dev/null 2>&1
python papers/paper_a_bounded_returns/analysis/build_bestofn.py 2>&1 | tee logs/bestofn_result.log
echo "DONE"
