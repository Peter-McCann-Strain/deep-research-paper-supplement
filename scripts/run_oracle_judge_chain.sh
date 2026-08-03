#!/usr/bin/env bash
# Judges the E14 oracle cells (oracle_t1_p9 + oracle_t1_p10) with GPT-5.2 once E14 generation
# finishes, matching how the existing oracle arm (oracle_t1_p0..p8) was judged
# (results/judge_gpt52/). GPT-5.2 = the real judge. ~$2-4 for 60 reports. Launch:
#   nohup bash scripts/run_oracle_judge_chain.sh &
set -u
cd .
[ -f venv/bin/activate ] && source venv/bin/activate
LOG=reports/phase_reports/logs/oracle_judge.log
mkdir -p reports/phase_reports/logs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "oracle judge chain started; waiting for E14 oracle generation (queue3) to complete..."
until grep -aq "GPU QUEUE3 COMPLETE" reports/phase_reports/logs/gpu_queue3.log 2>/dev/null; do sleep 120; done
say "E14 generation done; GPT-5.2-judging oracle_t1_p9 + oracle_t1_p10..."
python scripts/run_gpt52_judge.py --patterns-raw oracle_t1_p9,oracle_t1_p10 --resume >>"$LOG" 2>&1
say "ORACLE JUDGING COMPLETE (oracle_t1_p9/p10 scored by GPT-5.2 -> results/judge_gpt52/)."
