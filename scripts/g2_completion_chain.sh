#!/usr/bin/env bash
# G2 self-driving finish: wait for the (now hang-proof) gen process to EXIT, then
# GPT-5.2 judge -> build_second_backbone dry-run. No restart-loop: the fix agent's
# SIGALRM backstop guarantees the runner exits on its own; we just wait for it.
cd .
[ -f venv/bin/activate ] && source venv/bin/activate
export LD_LIBRARY_PATH="./.cudatk/lib:$(find venv/lib/python3.12/site-packages/nvidia -name lib -type d|tr '\n' ':')$LD_LIBRARY_PATH"
RV=results/experiments_gpt41_backbone
echo "===== G2 FINISH CHAIN START $(date -u) ; waiting for gen to exit ====="
while pgrep -f "run_gpt41_backbone.py --run" >/dev/null 2>&1; do
  echo "[wait $(date -u +%H:%M)] gen alive, reports=$(ls $RV/*/*.md 2>/dev/null|wc -l)/120"
  sleep 600
done
echo "[1] gen EXITED at $(ls $RV/*/*.md 2>/dev/null|wc -l)/120  $(date -u)"
echo "[2] JUDGE (GPT-5.2)  $(date -u)"
JUDGE_RESULTS_BASE=$RV python scripts/run_gpt52_judge_namespaced.py \
  --judge-out results/judge_gpt52_gpt41_backbone --patterns-raw p0_base,p4_base,p4_oracle \
  --resume --concurrency 3 2>&1 | tail -8
echo "[3] build_second_backbone DRY-RUN  $(date -u)"
python scripts/build_second_backbone.py --judge-out results/judge_gpt52_gpt41_backbone 2>&1 | tail -22
echo "===== G2 FINISH CHAIN DONE — review second_backbone, then --write  $(date -u) ====="
