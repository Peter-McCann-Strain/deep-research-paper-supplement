#!/usr/bin/env bash
cd .
[ -f venv/bin/activate ] && source venv/bin/activate
export LD_LIBRARY_PATH="./.cudatk/lib:$(find venv/lib/python3.12/site-packages/nvidia -name lib -type d|tr '\n' ':')$LD_LIBRARY_PATH"
# wait for the running master GPU queue (B/C/D) to finish
while pgrep -f "run_gpu_queue_all|train_e10_noise_rl|run_gpu_experiments.sh" >/dev/null; do sleep 120; done
echo "[A-followon] queue idle $(date -u) — re-running E10 (skips B/C/D sentinels, trains A_clean full 300 steps)"
bash scripts/run_gpu_experiments.sh --only e10 --e10-full --e10-gpu-hours 192
echo "[A-followon] A_clean full run done $(date -u)"
