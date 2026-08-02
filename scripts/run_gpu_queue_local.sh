#!/usr/bin/env bash
# Queue + launch the full LOCAL 7B benchmark generation run on the RTX 5080.
#
# Behaviour:
#   1. Wait until the GPU has > MIN_FREE_MB MiB free (default 3072 = 3 GB), so we
#      do not collide with any other GPU job (another judging lane may be busy).
#   2. Run the local benchmark harness at full budget (--limit 15 => 150 reports).
#   3. Tee all output to reports/phase_reports/logs/local_benchmark_gen.log.
#
# Intended to be nohup-launched so it survives the parent shell:
#   nohup bash scripts/run_gpu_queue_local.sh > /dev/null 2>&1 &
#
# Only one 7B model is ever resident at a time (the harness unloads between
# patterns), and the harness writes ONLY under results/local_benchmark/.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MIN_FREE_MB="${MIN_FREE_MB:-3072}"     # require > 3 GB free before starting
POLL_SECS="${POLL_SECS:-30}"
LIMIT="${LIMIT:-15}"

LOG_DIR="$REPO_ROOT/reports/phase_reports/logs"
LOG_FILE="$LOG_DIR/local_benchmark_gen.log"
mkdir -p "$LOG_DIR"

{
  echo "==== run_gpu_queue_local.sh starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  echo "Waiting for GPU to have > ${MIN_FREE_MB} MiB free (poll ${POLL_SECS}s)..."

  while true; do
    FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    if [ -z "$FREE_MB" ]; then
      echo "  nvidia-smi unavailable; retrying in ${POLL_SECS}s"
      sleep "$POLL_SECS"
      continue
    fi
    echo "  GPU free: ${FREE_MB} MiB"
    if [ "$FREE_MB" -gt "$MIN_FREE_MB" ]; then
      echo "  GPU has enough free VRAM (${FREE_MB} > ${MIN_FREE_MB} MiB). Launching."
      break
    fi
    sleep "$POLL_SECS"
  done

  # Activate venv (Python deps live here).
  if [ -f "$REPO_ROOT/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/venv/bin/activate"
  fi

  echo "==== launching run_local_benchmark_gen.py --limit ${LIMIT} ===="
  python -u scripts/run_local_benchmark_gen.py --limit "$LIMIT"
  RC=$?
  echo "==== run_local_benchmark_gen.py exited rc=${RC} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
} 2>&1 | tee -a "$LOG_FILE"
