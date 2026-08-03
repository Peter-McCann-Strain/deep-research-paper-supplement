#!/usr/bin/env bash
# =============================================================================
# run_gpu_queue_all.sh — MASTER serial GPU queue (single RTX 5080, 16 GB)
# =============================================================================
# WHAT THIS IS
#   The top-level, self-driving, resumable driver that chains the remaining
#   GPU-touching programme work in dependency + priority order on the ONE card.
#   GPU jobs are STRICTLY SEQUENTIAL — only ONE model is ever resident in the
#   16 GB VRAM (a wait_gpu_free 12 GB gate sits before every model load).
#
#   It REUSES the proven scripts/run_gpu_experiments.sh patterns verbatim:
#   set -u -o pipefail, venv activation, PYTORCH_ALLOC_CONF, the llama.cpp sm_120
#   LD_LIBRARY_PATH prepend, a fail-closed numpy 1.x guard, a wait_gpu_free
#   polling gate, and sentinel-based step() resume (STOP-on-failure).
#
# FIRING ORDER (one GPU job at a time):
#   STEP 0  WAIT for the running frozen-vintage chain to finish     [$0, gate]
#   STEP 1  E9 SCALE-CURVE 14B LOCAL TIER (GGUF gen)                 [~1.2 GPU-day]
#   STEP 2  E10-FULL NOISE-RL (8-adapter multi-seed GRPO)           [~6-10 GPU-days]
#
# RATIONALE: the cheap 14B capability point lands BEFORE the multi-day RL block
# monopolises the card. The Azure E9 tiers (gpt-4o-mini / gpt-4.1 / gpt-4o) touch
# NO GPU and run on the PTU queue (scripts/run_ptu_queue_all.sh) in parallel.
#
# NOTHING here mutates the canonical store, judges, trains during --plan/--dry,
# or relaunches the frozen-vintage chain. GENERATION/TRAINING is $0 (all local);
# the paid GPT-5.2 JUDGING of every arm is a SEPARATE human-launched step (the
# exact command shape is PRINTED per arm).
#
# USAGE
#   nohup bash scripts/run_gpu_queue_all.sh > /dev/null 2>&1 &   # full queue (E9-14B + E10-full)
#   bash scripts/run_gpu_queue_all.sh --plan                      # print firing plan, run NOTHING
#   bash scripts/run_gpu_queue_all.sh --dry                       # cheap smoke/dry per step
#   bash scripts/run_gpu_queue_all.sh --self-test                 # offline wiring checks, exit 0/1
#   bash scripts/run_gpu_queue_all.sh --only e9                   # one step (step0|e9|e10)
#   bash scripts/run_gpu_queue_all.sh --skip e10                  # skip the multi-day arm
#   bash scripts/run_gpu_queue_all.sh --skip-wait                 # bypass STEP 0 only if frozen chain already sentinel'd
#   bash scripts/run_gpu_queue_all.sh --e10-gpu-hours 192         # assert the E10 GPU-block reservation gate
#
# Resume: re-running picks up where it left off (per-step sentinels under
# artifacts/phase_reports/programme/logs/gpu_queue/done/). Delete a sentinel to redo.
# =============================================================================
set -u -o pipefail

REPO_ROOT="."
cd "$REPO_ROOT" || { echo "FATAL: cannot cd $REPO_ROOT"; exit 2; }

# ── flags ────────────────────────────────────────────────────────────────────
MODE="run"            # run | plan | dry
ONLY=""               # step0 | e9 | e10
SKIP=""               # comma list
SKIP_WAIT=0           # bypass STEP 0 (only if frozen chain already sentinel'd)
E10_GPU_HOURS="192"   # default >=144 so E10-full launches; lower => E10 parks
SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --plan) MODE="plan" ;;
    --dry)  MODE="dry"  ;;
    --self-test) SELF_TEST=1 ;;
    --only) ONLY="${2:-}"; shift ;;
    --skip) SKIP="${2:-}"; shift ;;
    --skip-wait) SKIP_WAIT=1 ;;
    --e10-gpu-hours) E10_GPU_HOURS="${2:-}"; shift ;;
    -h|--help) sed -n '2,55p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1"; exit 2 ;;
  esac
  shift
done

want(){  # want <step> -> 0 if it should run
  local e="$1"
  [ -n "$ONLY" ] && { [ "$ONLY" = "$e" ] && return 0 || return 1; }
  case ",$SKIP," in *",$e,"*) return 1 ;; esac
  return 0
}

# ── venv ─────────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
if [ -f venv/bin/activate ]; then [ -f venv/bin/activate ] && source venv/bin/activate; else
  echo "FATAL: venv/bin/activate missing — run from a configured checkout"; exit 2; fi

# ── CUDA fragmentation hint ──────────────────────────────────────────────────
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ── llama.cpp sm_120 CUDA libs on LD_LIBRARY_PATH (BEFORE any native load) ────
_ld_parts=""
[ -d "$REPO_ROOT/.cudatk/lib" ] && _ld_parts="$REPO_ROOT/.cudatk/lib"
if [ -d "$REPO_ROOT/venv/lib/python3.12/site-packages/nvidia" ]; then
  while IFS= read -r d; do _ld_parts="${_ld_parts:+$_ld_parts:}$d"; done < <(
    find "$REPO_ROOT/venv/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib | sort)
fi
[ -n "$_ld_parts" ] && export LD_LIBRARY_PATH="${_ld_parts}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── logging / sentinels ──────────────────────────────────────────────────────
LOGDIR="artifacts/phase_reports/programme/logs/gpu_queue"
DONEDIR="$LOGDIR/done"
mkdir -p "$LOGDIR" "$DONEDIR"
LOG="$LOGDIR/run_gpu_queue_all.log"
say(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
hr(){  echo "==============================================================" | tee -a "$LOG"; }

# ── numpy guard (HARD, fail-closed) — same contract as run_gpu_experiments.sh ─
numpy_guard(){
  python - <<'PY'
import sys
try:
    import numpy as np
except Exception as e:
    print(f"NUMPY-GUARD: numpy import FAILED: {type(e).__name__}: {e}", file=sys.stderr); sys.exit(3)
v = tuple(int(x) for x in np.__version__.split(".")[:2])
if v[0] != 1:
    print(f"NUMPY-GUARD: numpy {np.__version__} — this stack is pinned to 1.x "
          f"(llama.cpp + transformers/bnb ABI). Refusing to launch GPU jobs.", file=sys.stderr)
    sys.exit(3)
print(f"NUMPY-GUARD ok: numpy {np.__version__}")
PY
}

# ── GPU-free wait (sequential discipline) ────────────────────────────────────
MIN_FREE_MIB=12000
wait_gpu_free(){
  local tag="$1"
  [ "$MODE" = "plan" ] && { say "[$tag] (plan) would wait for >= ${MIN_FREE_MIB} MiB free"; return 0; }
  command -v nvidia-smi >/dev/null 2>&1 || { say "[$tag] nvidia-smi absent — proceeding (cannot poll VRAM)"; return 0; }
  say "[$tag] waiting for GPU >= ${MIN_FREE_MIB} MiB free (sequential gate) ..."
  while :; do
    local free
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    [ -n "$free" ] && [ "$free" -ge "$MIN_FREE_MIB" ] 2>/dev/null && { say "[$tag] GPU free=${free} MiB — go"; return 0; }
    sleep 20
  done
}

# ── step runner: gated, logged, resumable, STOP-on-failure ───────────────────
# step <sentinel> <label> -- <command...>
step(){
  local sentinel="$DONEDIR/$1"; shift
  local label="$1"; shift
  [ "$1" = "--" ] && shift
  if [ -f "$sentinel" ]; then say "SKIP  [$label] (sentinel $(basename "$sentinel") present)"; return 0; fi
  hr; say "STEP  [$label]"; say "  cmd: $*"
  if [ "$MODE" = "plan" ]; then return 0; fi
  if "$@" >>"$LOG" 2>&1; then
    say "OK    [$label]"; : > "$sentinel"; return 0
  else
    say "FAIL  [$label] (non-zero exit) — STOPPING queue so it can be inspected/resumed"; return 1
  fi
}

# =============================================================================
# SELF-TEST (offline; no GPU, no API, no model). Verifies wiring + presence.
# =============================================================================
if [ "$SELF_TEST" = "1" ]; then
  rc=0
  check(){ if eval "$2"; then echo "  OK   $1"; else echo "  FAIL $1"; rc=1; fi; }
  check "E9 14B runner present"        "[ -f scripts/run_e8_vintage_14b_gen.py ]"
  check "E9 scale-curve harness present" "[ -f scripts/run_e9_scale_curve.py ]"
  check "run_gpu_experiments.sh present" "[ -f scripts/run_gpu_experiments.sh ]"
  check "frozen chain script present"  "[ -f scripts/run_frozen_vintage_chain.sh ]"
  check "e10 readiness gate present"   "[ -f scripts/e10_noise_rl_readiness.py ]"
  check "E9 harness dry-run is clean"  "python scripts/run_e9_scale_curve.py --dry-run >/dev/null 2>&1"
  check "E9 harness self-test passes"  "python scripts/run_e9_scale_curve.py --self-test >/dev/null 2>&1"
  check "logdir creatable"             "[ -d '$LOGDIR' ]"
  echo; [ "$rc" = "0" ] && echo "  SELF-TEST PASS" || echo "  SELF-TEST FAIL"
  exit "$rc"
fi

# =============================================================================
# BANNER
# =============================================================================
hr
say "MASTER GPU QUEUE — mode=$MODE only='${ONLY:-all}' skip='${SKIP:-none}' e10_gpu_hours=$E10_GPU_HOURS"
say "Firing order: STEP0 wait-frozen-vintage -> E9 14B local gen -> E10-full noise-RL"
say "Single RTX 5080; jobs strictly SEQUENTIAL; LD_LIBRARY_PATH exported; numpy 1.x guarded."
hr

if [ "$MODE" != "plan" ]; then
  numpy_guard >>"$LOG" 2>&1 || { say "NUMPY GUARD FAILED — aborting (see log)"; exit 3; }
  say "numpy guard passed."
fi

# =============================================================================
# STEP 0 — WAIT FOR THE RUNNING FROZEN-VINTAGE CHAIN (precondition gate, $0)
# =============================================================================
# Block until run_frozen_vintage_chain.sh (and its freeze/regen children)
# finishes; then VERIFY 90/90 frozen sources AND verdicts for all 4 arms. Do
# NOT relaunch the chain — it is owned by its own nohup. If incomplete, STOP.
FROZEN_SENTINEL="$DONEDIR/frozen_vintage_chain.done"
FV_VERDICTS="results/judge_gpt52_frozen_vintage"
FV_ARMS="base_p9 base_p14_vintage_deepseek_qwen7b base_p13_vintage_qwen3_8b base_p17_scale_qwen25_14b"

verify_frozen_complete(){
  # 90/90 frozen sources
  local n
  n="$(ls data/frozen_corpus_vintage/*.json 2>/dev/null | grep -v MANIFEST | wc -l | tr -d ' ')"
  say "  frozen sources: ${n}/90"
  [ "$n" -ge 90 ] 2>/dev/null || { say "  frozen sources incomplete (${n}/90)"; return 1; }
  # verdicts present for all 4 arms
  local arm missing=0
  for arm in $FV_ARMS; do
    if [ -d "$FV_VERDICTS/$arm" ] && [ -n "$(ls -A "$FV_VERDICTS/$arm" 2>/dev/null)" ]; then
      say "  verdicts OK: $arm"
    else
      say "  verdicts MISSING: $arm"; missing=1
    fi
  done
  [ "$missing" = "0" ]
}

if want step0; then
  hr; say ">>> STEP 0 — WAIT FOR FROZEN-VINTAGE CHAIN (precondition gate)"
  if [ -f "$FROZEN_SENTINEL" ]; then
    say "SKIP  [STEP0] frozen-vintage sentinel present."
  elif [ "$SKIP_WAIT" = "1" ]; then
    say "STEP0 --skip-wait: bypassing the wait. Verifying completion only ..."
    if [ "$MODE" = "plan" ]; then
      say "  (plan) would verify 90/90 sources + 4-arm verdicts."
    elif verify_frozen_complete; then
      : > "$FROZEN_SENTINEL"; say "OK    [STEP0] frozen-vintage verified (skip-wait)."
    else
      say "FAIL  [STEP0] --skip-wait but frozen-vintage NOT complete — STOPPING."; exit 1
    fi
  else
    if [ "$MODE" = "plan" ]; then
      say "  (plan) would: pgrep-wait on run_frozen_vintage_chain.sh|freeze_vintage_sources|run_frozen_vintage.py,"
      say "  then verify 90/90 frozen sources + verdicts for: $FV_ARMS"
    else
      say "  blocking until the running frozen-vintage chain finishes (NOT relaunching it) ..."
      while pgrep -f 'run_frozen_vintage_chain.sh|freeze_vintage_sources|run_frozen_vintage.py' >/dev/null 2>&1; do
        sleep 60
      done
      say "  chain process gone — verifying completion ..."
      if verify_frozen_complete; then
        : > "$FROZEN_SENTINEL"; say "OK    [STEP0] frozen-vintage complete + verified."
      else
        say "FAIL  [STEP0] chain ended but outputs incomplete — STOPPING (inspect, do not relaunch here)."; exit 1
      fi
    fi
  fi
fi

# =============================================================================
# STEP 1 — E9 SCALE-CURVE 14B LOCAL TIER (GPU; $0 local gen)
# =============================================================================
# The ONLY GPU part of E9: the local 14B capability point, generated via the
# EXISTING GGUF runner (pattern p17_scale_qwen25_14b, llama.cpp Q4_K_M,
# strict-greedy, n=90), reusing E8b's frozen scaffold. The Azure tiers run on
# the PTU queue. Smoke first, then full. (This is the prereg-required 14B point
# the original .pyc had to skip; the GGUF path resolves the OOM.)
if want e9; then
  hr; say ">>> STEP 1 — E9 14B LOCAL TIER (GGUF; \$0 local generation)"
  wait_gpu_free "E9.14B.smoke"
  step e9_14b_smoke "E9 14B GGUF SMOKE (load + 1 query)" -- \
    python scripts/run_e8_vintage_14b_gen.py --smoke \
    || exit 1
  if [ "$MODE" != "dry" ]; then
    wait_gpu_free "E9.14B.full"
    step e9_14b_full "E9 14B GGUF FULL (90 queries, strict-greedy)" -- \
      python scripts/run_e8_vintage_14b_gen.py \
      || exit 1
    say "E9 14B judge (separate, paid, GPT-5.2): JUDGE_RESULTS_BASE=results/experiments_e8_14b \\"
    say "  python scripts/run_gpt52_judge_namespaced.py --judge-out results/judge_gpt52_e8_14b --patterns-raw p17_scale_qwen25_14b --resume"
    say "E9 build (after the Azure tiers + this 14B point are judged): python scripts/build_e9_scale.py"
  fi
  say "<<< STEP 1 (E9 14B) done"
fi

# =============================================================================
# STEP 2 — E10-FULL NOISE-RL (GPU; multi-day, MOST EXPENSIVE GPU job, gated)
# =============================================================================
# Delegates to the proven chain. Gate G2 (E7 youden_j) is already PASSED; Gate
# G0 (prereg split) + Gate G1 (GPU-block reservation >=144h) are enforced INSIDE
# run_gpu_experiments.sh. We pass --e10-gpu-hours explicitly so E10-full LAUNCHES;
# without a reserved window (>=144) E10 PARKS by design (exit 0, not a failure).
# 8 adapters strictly sequential with wait_gpu_free between every arm; each arm
# sentinel'd => resumable mid-multi-day. Canonical UNTOUCHED by training.
if want e10; then
  hr; say ">>> STEP 2 — E10-FULL NOISE-RL (multi-day; gated; --e10-gpu-hours $E10_GPU_HOURS)"
  E10_ARGS="--only e10 --e10-full --e10-gpu-hours $E10_GPU_HOURS"
  [ "$MODE" = "plan" ] && E10_ARGS="$E10_ARGS --plan"
  [ "$MODE" = "dry"  ] && E10_ARGS="$E10_ARGS --dry"
  # shellcheck disable=SC2086
  step e10_full_delegate "E10-full via run_gpu_experiments.sh" -- \
    bash scripts/run_gpu_experiments.sh $E10_ARGS \
    || exit 1
  say "E10 held-out GPT-5.2 judging + post-hoc canonical build are a SEPARATE human step."
  say "<<< STEP 2 (E10-full) done"
fi

hr
say "MASTER GPU QUEUE COMPLETE (mode=$MODE). Sentinels: $DONEDIR/  Log: $LOG"
say "Generation/training was \$0 (local). PAID GPT-5.2 judging of each arm is the separate human-launched step."
hr
