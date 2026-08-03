#!/usr/bin/env bash
# GPU queue #2 — chains behind the vintage queue so the RTX 5080 stays saturated.
# Runs the E9 local SCALE arm (Qwen2.5-14B in the frozen P9 scaffold) once P14 finishes.
# 1 pass over 90 queries (judged once/query by GPT-5.2 — no replicate-judging issue). Smoke-test
# gates on OOM (a 14B + extraction is tight on 16GB; if it OOMs it's skipped, honest hardware limit).
# Corpus-safe (own pattern dir). Launch:  nohup bash scripts/run_gpu_queue2.sh &
set -u
cd .
[ -f venv/bin/activate ] && source venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG=reports/phase_reports/logs/gpu_queue2.log
mkdir -p reports/phase_reports/logs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "queue2 started; waiting for the vintage queue (P14) to finish before using the GPU..."
until grep -aq "GPU QUEUE COMPLETE" reports/phase_reports/logs/gpu_queue.log 2>/dev/null; do sleep 60; done
say "P14 done; GPU free. Waiting for Qwen2.5-14B download..."
D="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct"
until [ -d "$D" ] && [ "$(find "$D" -name '*.incomplete' 2>/dev/null | wc -l)" -eq 0 ] \
      && [ -n "$(find "$D" -name '*.safetensors' 2>/dev/null | head -1)" ]; do sleep 30; done
say "Qwen2.5-14B ready."

P=p17_scale_qwen25_14b
say "=== $P : SMOKE TEST (1 query) ==="
if python scripts/run_eval_v2.py --phase generate --patterns "$P" --max-queries 1 --n-repeats 1 --max-concurrent 1 >>"$LOG" 2>&1 \
   && ! tail -60 "$LOG" | grep -qiE "OutOfMemory|CUDA out of memory"; then
  say "=== $P : FULL RUN (90 queries x 1, max-concurrent 1) ==="
  python scripts/run_eval_v2.py --phase generate --patterns "$P" --n-repeats 1 --max-concurrent 1 >>"$LOG" 2>&1
  say "=== $P : DONE ==="
else
  say "!! $P OOM/failed in smoke test on 16GB — skipped (14B+extraction over budget). Honest hardware limit."
fi

say "GPU QUEUE2 COMPLETE (E9 14B scale arm)."
say "Next GPU candidates to append: E14 P9/P10 oracle cells (oracle_corpus_t1.json on disk); E7 selectors; E13' detector panel."
