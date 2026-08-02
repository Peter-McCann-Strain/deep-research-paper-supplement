#!/usr/bin/env bash
# Self-driving GPU queue — keeps the RTX 5080 saturated with the E8 vintage curve.
# Per pattern: wait for its model to finish downloading, smoke-test ONE query (catch OOM/load
# issues before committing), then run the full 90 queries x 3 repeats. Local patterns run
# --max-concurrent 1 (one model on 16GB). Each vintage pattern writes to its OWN results dir
# (corpus-safe; nothing existing is touched). Launch:  nohup bash scripts/run_gpu_queue.sh &
set -u
cd .
[ -f venv/bin/activate ] && source venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True   # reduce CUDA fragmentation (OOM hint)
LOG=reports/phase_reports/logs/gpu_queue.log
mkdir -p reports/phase_reports/logs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

wait_model () {  # $1 = HF hub dir name
  local d="$HOME/.cache/huggingface/hub/$1"
  say "waiting for model $1 ..."
  until [ -d "$d" ] && [ "$(find "$d" -name '*.incomplete' 2>/dev/null | wc -l)" -eq 0 ] \
        && [ -n "$(find "$d" -name '*.safetensors' 2>/dev/null | head -1)" ]; do sleep 20; done
  say "model $1 ready."
}

run_pattern () {  # $1 = pattern name, $2 = hf hub dir
  wait_model "$2"
  say "=== $1 : SMOKE TEST (1 query) ==="
  if ! python scripts/run_eval_v2.py --phase generate --patterns "$1" \
        --max-queries 1 --n-repeats 1 --max-concurrent 1 >>"$LOG" 2>&1; then
    say "!! $1 smoke test FAILED (non-zero exit) — skipping full run"; return 1; fi
  if tail -50 "$LOG" | grep -qiE "OutOfMemory|CUDA out of memory"; then
    say "!! $1 OOM in smoke test — skipping full run"; return 1; fi
  say "=== $1 : FULL RUN (90 queries x 3 repeats, max-concurrent 1) ==="
  python scripts/run_eval_v2.py --phase generate --patterns "$1" \
        --n-repeats 3 --max-concurrent 1 >>"$LOG" 2>&1
  say "=== $1 : DONE ==="
}

# E8 vintage curve (Qwen-family, frozen P9 scaffold) — study subjects, not judges
run_pattern p13_vintage_qwen3_8b            models--Qwen--Qwen3-8B
run_pattern p14_vintage_deepseek_qwen7b     models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B

say "GPU QUEUE COMPLETE (vintage curve p13+p14). Reports under results*/reports/p13_*/ + p14_*/."
say "Append next GPU jobs here when set up: E2 P10/P4 replicate top-up, E14 P9/P10 oracle cells, E13' detector panel."
