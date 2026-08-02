#!/usr/bin/env bash
# GPU queue #3 — E14: the missing P9/P10 ORACLE cells (local 7B models on the ideal-source corpus).
# Only oracle_t1_p0..p8 (GPT-4o) exist; this adds oracle_t1_p9 (Qwen2.5-7B) + oracle_t1_p10
# (DeepResearcher-7b) on the 30 variance queries, completing the oracle dual-ceiling analysis for the
# local arms. Proven-to-fit 7B models. Uses the existing oracle mechanism (SEARCH_BACKEND=oracle +
# run_all_experiments.py) — no new code. Launch:  nohup bash scripts/run_gpu_queue3.sh &
set -u
cd .
[ -f venv/bin/activate ] && source venv/bin/activate
export SEARCH_BACKEND=oracle
export ORACLE_CORPUS_PATH=data/oracle_corpus_t1.json
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG=reports/phase_reports/logs/gpu_queue3.log
mkdir -p reports/phase_reports/logs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# wait until the GPU is free (any lingering smoke-test model unloaded) to avoid contention/OOM
say "queue3 (E14 oracle P9/P10) started; waiting for GPU to be free..."
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')" -lt 3000 ]; do sleep 30; done
say "GPU free; generating oracle cells (30 variance queries each)."

for p in p9 p10; do
  say "=== oracle $p : generate (30 variance queries) ==="
  python scripts/run_all_experiments.py --pattern "$p" --retriever oracle --run-tag t1 \
    --query-ids-file data/variance_stratified.json --base-only --resume >>"$LOG" 2>&1
  say "=== oracle $p : DONE ==="
done
say "GPU QUEUE3 COMPLETE (E14 oracle_t1_p9 + oracle_t1_p10). These need GPT-5.2 judging (separate pass)."
say "Next GPU candidates: E7 selectors (DR-Judge inference); E13' detector panel (needs perturbation set)."
