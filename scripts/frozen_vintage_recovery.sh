#!/usr/bin/env bash
# Recovery: finish p17 (14B GGUF) + p13 (8B GGUF, OOM workaround) on the accepted 89-query
# frozen set, re-judge all arms, then relaunch the master GPU queue (E9 -> E10-full).
cd .
[ -f venv/bin/activate ] && source venv/bin/activate
export LD_LIBRARY_PATH="./.cudatk/lib:$(find venv/lib/python3.12/site-packages/nvidia -name lib -type d|tr '\n' ':')$LD_LIBRARY_PATH"
LOGDIR=artifacts/phase_reports/programme/logs
DONEDIR=artifacts/phase_reports/programme/logs/gpu_queue/done
RV=results/experiments_frozen_vintage
echo "===== FROZEN-VINTAGE RECOVERY START $(date -u) ====="

# 1) wait for the running p17 (14B) regen to finish (frees the GPU)
while pgrep -f "run_frozen_vintage.py --arm base_p17" >/dev/null; do sleep 60; done
p17n=$(ls $RV/base_p17_scale_qwen25_14b/*.md 2>/dev/null | wc -l)
echo "[1] p17 (14B) regen done: ${p17n}/89  $(date -u)"

# 2) regen p13 (Qwen3-8B via GGUF — OOM workaround); tolerant of failure
echo "[2] p13 (8B GGUF) regen $(date -u)"
python scripts/run_frozen_vintage.py --arm base_p13_vintage_qwen3_8b > $LOGDIR/regen_p13_gguf.log 2>&1 || true
p13n=$(ls $RV/base_p13_vintage_qwen3_8b/*.md 2>/dev/null | wc -l)
echo "[2] p13 (8B) done: ${p13n}/89  (if low/0, GGUF also failed -> document 3-point curve)"

# 3) re-judge all arms on the shared frozen set (GPT-5.2, cloud JUDGE endpoint)
echo "[3] re-judge all arms $(date -u)"
JUDGE_RESULTS_BASE=$RV python scripts/run_gpt52_judge_namespaced.py \
  --judge-out results/judge_gpt52_frozen_vintage \
  --patterns-raw base_p9,base_p14_vintage_deepseek_qwen7b,base_p13_vintage_qwen3_8b,base_p17_scale_qwen25_14b \
  --resume 2>&1 | tail -15

# 4) mark frozen-vintage complete (consciously accepting the 89/90 shared set; query #90
#    dropped for a benign flaky-fetch reason, unbiased). This sentinel makes the master
#    queue SKIP STEP0's strict 90/90 check and proceed to E9 -> E10.
mkdir -p "$DONEDIR"; touch "$DONEDIR/frozen_vintage_chain.done"
echo "[4] frozen-vintage sentinel set (accepted shared set; p9/p14=89, p17=${p17n}, p13=${p13n})"

# 5) relaunch the master GPU queue -> E9 14B point -> E10-full noise-RL
nohup bash scripts/run_gpu_queue_all.sh > $LOGDIR/gpu_queue_all2.out 2>&1 &
echo "[5] master GPU queue relaunched (E9 -> E10-full) PID $!  $(date -u)"
echo "===== FROZEN-VINTAGE RECOVERY DONE $(date -u) ====="
