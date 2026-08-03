#!/usr/bin/env bash
cd .
[ -f venv/bin/activate ] && source venv/bin/activate
export LD_LIBRARY_PATH="./.cudatk/lib:$(find venv/lib/python3.12/site-packages/nvidia -name lib -type d|tr '\n' ':')$LD_LIBRARY_PATH"
LOG=artifacts/phase_reports/programme/logs/frozen_vintage_chain.log
echo "===== CHAIN START $(date -u) ====="
# 1) wait for the freeze to finish
while pgrep -f freeze_vintage_sources >/dev/null; do sleep 60; done
n=$(ls data/frozen_corpus_vintage/*.json 2>/dev/null | grep -v MANIFEST | wc -l)
echo "[1] FREEZE done: $n/90 queries  $(date -u)"
# 2) regenerate all 4 arms on the frozen sources (sequential GPU, $0)
echo "[2] REGENERATE all arms $(date -u)"
python scripts/run_frozen_vintage.py 2>&1 | tail -40
# 3) GPT-5.2 judge (cloud, independent of all Qwen-family arms)
echo "[3] JUDGE (GPT-5.2) $(date -u)"
JUDGE_RESULTS_BASE=results/experiments_frozen_vintage python scripts/run_gpt52_judge_namespaced.py \
  --judge-out results/judge_gpt52_frozen_vintage \
  --patterns-raw base_p9,base_p14_vintage_deepseek_qwen7b,base_p13_vintage_qwen3_8b,base_p17_scale_qwen25_14b \
  --resume 2>&1 | tail -20
echo "===== CHAIN DONE rc=$? $(date -u) ====="
echo "NOTE: length-controlled scoring + per-query CIs + canonical frozen_vintage key are the final analysis step (run after verdicts land)."
