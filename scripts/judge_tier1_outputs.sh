#!/usr/bin/env bash
# Judge the Tier-1 outputs (Protocol A Tavily wave + P11 ReAct baseline)
# using the existing canonical run_gpt52_judge.py. Routes through Azure
# non-PTU endpoint, doesn't compete with running PTU runs.
set -e
cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate

echo "=== Judging E2 P11 ReAct (base_p11) ==="
python scripts/run_gpt52_judge.py --patterns 11 --resume

echo ""
echo "=== Judging E1 Protocol A Tavily wave ==="
python scripts/run_gpt52_judge.py \
  --patterns-raw "protocol_a_tavily_p0,protocol_a_tavily_p1,protocol_a_tavily_p3,protocol_a_tavily_p4,protocol_a_tavily_p5,protocol_a_tavily_p8" \
  --resume

echo ""
echo "=== Rebuilding analysis dataframes ==="
python scripts/build_analysis_dataframes.py
echo ""
echo "=== Tier 1 judging + parquet rebuild done ==="
