#!/usr/bin/env bash
# GPT-5.2 judge loop — judges the new GPU-generated reports with the REAL judge (GPT-5.2), the
# same judge that scored the 248k corpus, so the vintage/replicate arms are comparable to it.
# GPT-4o is NOT used (user directive 2026-06-12: 5.2 is the judge). Default config = 2 GPT-5.2
# reads/report (gpt52_primary + gpt52_diverse), 1 pass each => ~$0.072/report, ~$58 for ~810
# reports (well under the $300 ceiling). Writes corpus-safe verdicts to reports/eval_v2/verdicts/.
# Runs in parallel with the GPU generation queues. Launch:  nohup bash scripts/run_ptu_judge_loop.sh &
set -u
cd .
[ -f venv/bin/activate ] && source venv/bin/activate
LOG=reports/phase_reports/logs/ptu_judge_loop.log
mkdir -p reports/phase_reports/logs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
PATTERNS="p14_vintage_deepseek_qwen7b,p17_scale_qwen25_14b"

say "GPT-5.2 judge loop started (judging new GPU reports with the real judge; ~\$58 projected for all)."
caught_up=0
while true; do
  # resume is DEFAULT (skip already-judged). 1 pass per judge to control GPT-5.2 spend.
  python scripts/run_eval_v2.py --phase judge \
      --patterns "$PATTERNS" --passes-per-judge 1 >>"$LOG" 2>&1
  if grep -aq "GPU QUEUE2 COMPLETE" reports/phase_reports/logs/gpu_queue2.log 2>/dev/null; then
    caught_up=$((caught_up+1)); say "GPU generation complete; GPT-5.2 catch-up pass $caught_up/2"
    [ "$caught_up" -ge 2 ] && break
  fi
  sleep 300
done
say "GPT-5.2 JUDGE LOOP COMPLETE — all new GPU reports judged by GPT-5.2."
