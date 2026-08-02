#!/usr/bin/env bash
# =============================================================================
# run_gpu_experiments.sh — single-GPU experiment queue for E8 / E11 / E10
# =============================================================================
# WHAT THIS IS
#   One self-driving, resumable runner for the three GPU-touching experiments,
#   fired in dependency + priority order on the single RTX 5080 (16 GB). GPU
#   jobs are STRICTLY SEQUENTIAL — only one model is ever resident. Each step is
#   gated (cheap test first, then the real run), logged, and resumable (a step
#   whose sentinel exists is skipped). NOTHING here mutates the canonical store,
#   calls a paid API, or trains during a --plan/--dry pass.
#
# FIRING ORDER (dependency + priority; one GPU job at a time):
#   1. E8  VINTAGE/CAPACITY  [priority 7.5, launch-now, QUICK]
#        E8a  arm2  DeepSeek-R1-Distill-Qwen-7B  (transformers 4-bit, frozen P9)
#        E8b  14B   Qwen2.5-14B GGUF capacity anchor (llama.cpp, ~76 tok/s)
#   2. E11 P14-VTR           [priority 7,   launch-now]
#        E11 vtr_drjudge GPU verifier arm (local DR-Judge-7B; PTU refiner).
#        (The PTU-only vtr_gpt4o/control arms touch no GPU and are NOT gated by
#         this queue — see GPU_EXPERIMENTS_PLAN.md; this runner does the GPU arm.)
#   3. E10 NOISE-RL          [priority 7,   MULTI-DAY, gated on E7]
#        Trimmed variant is the DEFAULT (A+B+C single-seed, ~3 GPU-days).
#        Full multi-seed (8 adapters, ~6-10 GPU-days) is opt-in via --e10-full.
#
# GENERATION/TRAINING here is $0 (all local). The paid GPT-5.2 JUDGING of every
# arm is a SEPARATE, human-launched step (run_gpt52_judge_namespaced.py, JUDGE
# Azure endpoint, never PTU) — this runner NEVER judges and prints the exact
# judge command shape for each arm instead.
#
# USAGE
#   nohup bash scripts/run_gpu_experiments.sh            > /dev/null 2>&1 &   # default queue (E8+E11+E10-trim)
#   bash scripts/run_gpu_experiments.sh --plan                                # print firing plan, run NOTHING
#   bash scripts/run_gpu_experiments.sh --dry                                 # run each cheap TEST only (no real runs)
#   bash scripts/run_gpu_experiments.sh --only e8                             # one experiment (e8|e11|e10)
#   bash scripts/run_gpu_experiments.sh --skip e10                            # skip the multi-day arm
#   bash scripts/run_gpu_experiments.sh --e10-full                            # E10 FULL 8-adapter multi-seed
#   bash scripts/run_gpu_experiments.sh --e10-gpu-hours 192                   # assert the E10 GPU-block reservation gate
#
# Resume: re-running picks up where it left off (per-arm sentinels under
# reports/phase_reports/logs/gpu_experiments/done/). Delete a sentinel to redo it.
# =============================================================================
set -u -o pipefail

REPO_ROOT="."
cd "$REPO_ROOT" || { echo "FATAL: cannot cd $REPO_ROOT"; exit 2; }

# ── venv ─────────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
if [ -f venv/bin/activate ]; then [ -f venv/bin/activate ] && source venv/bin/activate; else
  echo "FATAL: venv/bin/activate missing — run from a configured checkout"; exit 2; fi

# ── CUDA fragmentation hint (matches run_gpu_queue.sh) ───────────────────────
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ── llama.cpp sm_120 CUDA libs on LD_LIBRARY_PATH (E8b needs this BEFORE the
#    native llama.cpp import). The Python caller also self-prepends, but we
#    export here too so the env is correct for the whole queue. ──────────────
_ld_parts=""
[ -d "$REPO_ROOT/.cudatk/lib" ] && _ld_parts="$REPO_ROOT/.cudatk/lib"
if [ -d "$REPO_ROOT/venv/lib/python3.12/site-packages/nvidia" ]; then
  while IFS= read -r d; do _ld_parts="${_ld_parts:+$_ld_parts:}$d"; done < <(
    find "$REPO_ROOT/venv/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib | sort)
fi
if [ -n "$_ld_parts" ]; then
  export LD_LIBRARY_PATH="${_ld_parts}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ── logging ──────────────────────────────────────────────────────────────────
LOGDIR="reports/phase_reports/logs/gpu_experiments"
DONEDIR="$LOGDIR/done"
mkdir -p "$LOGDIR" "$DONEDIR"
LOG="$LOGDIR/run_gpu_experiments.log"
say(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
hr(){  echo "==============================================================" | tee -a "$LOG"; }

# ── flags ────────────────────────────────────────────────────────────────────
MODE="run"            # run | plan | dry
ONLY=""               # e8 | e11 | e10  (empty = all)
SKIP=""               # comma list to skip
E10_FULL=0            # 0 = trimmed default, 1 = full 8-adapter multi-seed
E10_GPU_HOURS=""      # assert the E10 GPU-block reservation gate (>=144 for full)
while [ $# -gt 0 ]; do
  case "$1" in
    --plan) MODE="plan" ;;
    --dry)  MODE="dry"  ;;
    --only) ONLY="${2:-}"; shift ;;
    --skip) SKIP="${2:-}"; shift ;;
    --e10-full) E10_FULL=1 ;;
    --e10-gpu-hours) E10_GPU_HOURS="${2:-}"; shift ;;
    -h|--help) sed -n '2,55p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1"; exit 2 ;;
  esac
  shift
done
want(){  # want <exp> -> 0 if it should run
  local e="$1"
  [ -n "$ONLY" ] && { [ "$ONLY" = "$e" ] && return 0 || return 1; }
  case ",$SKIP," in *",$e,"*) return 1 ;; esac
  return 0
}

# ── numpy guard (HARD) ───────────────────────────────────────────────────────
# llama.cpp (E8b) + transformers/bnb (E8a/E10/E11-drjudge) are built against the
# pinned numpy 1.26.x ABI on this box. A silent numpy>=2 (or absent numpy) is a
# classic source of native-extension import crashes mid-queue. Fail CLOSED here
# before any model loads rather than 3 GPU-days in.
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
# Block until the card has >= MIN_FREE_MIB free, so we never start a model while
# the previous job is still releasing VRAM. Polls; never runs two models at once.
MIN_FREE_MIB=12000
wait_gpu_free(){
  local tag="$1"
  command -v nvidia-smi >/dev/null 2>&1 || { say "[$tag] nvidia-smi absent — proceeding (cannot poll VRAM)"; return 0; }
  say "[$tag] waiting for GPU >= ${MIN_FREE_MIB} MiB free (sequential gate) ..."
  while :; do
    local free
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    [ -n "$free" ] && [ "$free" -ge "$MIN_FREE_MIB" ] 2>/dev/null && { say "[$tag] GPU free=${free} MiB — go"; return 0; }
    sleep 20
  done
}

# ── step runner: gated, logged, resumable ────────────────────────────────────
# step <sentinel> <human-label> -- <command...>
#   - skips if the sentinel exists (resume)
#   - in --plan: prints the command, runs nothing
#   - in --dry : the caller is responsible for passing the *test* command
#   - on success: writes the sentinel; on failure: STOPS the queue (set -e-like)
step(){
  local sentinel="$DONEDIR/$1"; shift
  local label="$1"; shift
  [ "$1" = "--" ] && shift
  if [ -f "$sentinel" ]; then say "SKIP  [$label] (sentinel $1.done present)"; return 0; fi
  hr; say "STEP  [$label]"; say "  cmd: $*"
  if [ "$MODE" = "plan" ]; then return 0; fi
  if "$@" >>"$LOG" 2>&1; then
    say "OK    [$label]"
    : > "$sentinel"
    return 0
  else
    say "FAIL  [$label] (non-zero exit) — STOPPING queue so it can be inspected/resumed"
    return 1
  fi
}

# =============================================================================
# FIRING PLAN BANNER
# =============================================================================
hr
say "GPU EXPERIMENT QUEUE — mode=$MODE only='${ONLY:-all}' skip='${SKIP:-none}' e10_full=$E10_FULL"
say "Firing order: E8 (vintage+capacity) -> E11 (drjudge GPU arm) -> E10 (noise-RL)"
say "Single RTX 5080; jobs strictly SEQUENTIAL; LD_LIBRARY_PATH exported; numpy 1.x guarded."
hr

if [ "$MODE" != "plan" ]; then
  numpy_guard >>"$LOG" 2>&1 || { say "NUMPY GUARD FAILED — aborting (see log)"; exit 3; }
  say "numpy guard passed."
fi

# =============================================================================
# 1) E8 VINTAGE / CAPACITY  [priority 7.5 — launch-now, QUICK]
# =============================================================================
if want e8; then
  hr; say ">>> E8 VINTAGE / CAPACITY (frozen P9 scaffold; \$0 local generation)"

  # ---- E8a: DeepSeek-R1-Distill-Qwen-7B vintage arm2 (transformers 4-bit) ----
  # 2025-01 vintage point on the gap-vs-date curve. Generated via the standard
  # eval harness pattern (same as run_gpu_queue.sh), --max-concurrent 1 (one 7B).
  wait_gpu_free "E8a"
  step e8a_test "E8a DeepSeek-7B SMOKE (1 query)" -- \
    python scripts/run_eval_v2.py --phase generate \
      --patterns p14_vintage_deepseek_qwen7b --max-queries 1 --n-repeats 1 --max-concurrent 1 \
    || exit 1
  if [ "$MODE" != "dry" ]; then
    wait_gpu_free "E8a"
    step e8a_full "E8a DeepSeek-7B FULL (90 queries)" -- \
      python scripts/run_eval_v2.py --phase generate \
        --patterns p14_vintage_deepseek_qwen7b --n-repeats 1 --max-concurrent 1 \
      || exit 1
    say "E8a judge (separate, paid): JUDGE_RESULTS_BASE=reports/eval_v2/reports \\"
    say "  python scripts/run_gpt52_judge_namespaced.py --judge-out results/judge_gpt52 --patterns-raw p14_vintage_deepseek_qwen7b"
  fi

  # ---- E8b: Qwen2.5-14B GGUF capacity anchor (llama.cpp, ~76 tok/s) ----------
  # Same-vintage (2024-09) larger-capacity point; strict-greedy GGUF Q4_K_M.
  wait_gpu_free "E8b"
  step e8b_test "E8b 14B GGUF SMOKE (load + 1 query)" -- \
    python scripts/run_e8_vintage_14b_gen.py --smoke \
    || exit 1
  if [ "$MODE" != "dry" ]; then
    wait_gpu_free "E8b"
    step e8b_full "E8b 14B GGUF FULL (90 queries, strict-greedy)" -- \
      python scripts/run_e8_vintage_14b_gen.py \
      || exit 1
    say "E8b judge (separate, paid): JUDGE_RESULTS_BASE=results/experiments_e8_14b \\"
    say "  python scripts/run_gpt52_judge_namespaced.py --judge-out results/judge_gpt52_e8_14b --patterns-raw p17_scale_qwen25_14b"
    say "E8 build (after BOTH arms judged): python scripts/build_e8_vintage.py"
  fi
  say "<<< E8 done"
fi

# =============================================================================
# 2) E11 P14-VTR — DR-Judge-7B GPU verifier arm  [priority 7 — launch-now]
# =============================================================================
# Only the LOCAL-GPU verifier arm (vtr_drjudge) is sequenced here; it forces
# concurrency=1 (DR-Judge 7B 4-bit, peak ~14.6 GiB). The PTU-only arms
# (vtr_gpt4o / control) touch no GPU and are launched independently (see plan).
# Generation is $0 (PTU refiner + local verifier); GPT-5.2 judging is separate.
if want e11; then
  hr; say ">>> E11 P14-VTR — DR-Judge-7B GPU verifier arm (concurrency forced to 1)"
  # cheap test: zero-API dry-run (no LLMCaller, no GPU load) on 2 items
  step e11_test "E11 vtr_drjudge DRY-RUN (zero API, 2 items)" -- \
    python scripts/run_e11_vtr.py --dry-run --limit 2 --bases p0,p4 --arms vtr_drjudge \
    || exit 1
  if [ "$MODE" != "dry" ]; then
    wait_gpu_free "E11"
    step e11_full "E11 vtr_drjudge RUN (p0,p4 x 30 variance queries)" -- \
      python scripts/run_e11_vtr.py --run --bases p0,p4 --arms vtr_drjudge --rounds 2 \
        --concurrency 1 --resume \
      || exit 1
    say "E11 judge (separate, paid): wire RESULTS_BASE->results/experiments_e11_vtr then"
    say "  python scripts/run_gpt52_judge_namespaced.py --judge-out results/judge_gpt52_e11 --patterns-raw e11_vtr_p0_drjudge,e11_vtr_p4_drjudge"
    say "E11 build (after judging): python scripts/build_e11_vtr.py"
  fi
  say "<<< E11 done"
fi

# =============================================================================
# 3) E10 NOISE-RL — GRPO arms  [priority 7 — MULTI-DAY, gated on E7]
# =============================================================================
# Trimmed variant DEFAULT (A+B+C single-seed, ~3 GPU-days). FULL (8 adapters,
# ~6-10 GPU-days) is opt-in via --e10-full. Every arm is local-GPU only; NO paid
# API, NO Opus; calibration read from a pinned read-only canonical snapshot;
# canonical UNTOUCHED by training. GPT-5.2 held-out judging is a separate pass.
if want e10; then
  hr; say ">>> E10 NOISE-RL (GRPO; multi-day; gated on E7) — full=$E10_FULL"
  SCALE_FLAG="--trim"; [ "$E10_FULL" = "1" ] && SCALE_FLAG="--full"

  # ---- Gate 0: prereg split must exist + hash-match; readiness gate must pass.
  step e10_split "E10 prereg split (idempotent; hash-anchored)" -- \
    python scripts/e10_prereg_split.py \
    || exit 1

  # GPU-block reservation gate (G1). If --e10-gpu-hours given, assert it; else
  # readiness FAILS CLOSED on G1 and the queue stops here (by design — E10 needs
  # an owner-reserved contiguous window). Other gates (G2/G3) must still be green.
  if [ "$MODE" != "plan" ]; then
    if [ -n "$E10_GPU_HOURS" ]; then
      step e10_ready "E10 readiness gate (--gpu-block-hours $E10_GPU_HOURS)" -- \
        python scripts/e10_noise_rl_readiness.py --gpu-block-hours "$E10_GPU_HOURS" \
        || { say "E10 readiness RED — parking E10 (reserve a GPU block + re-run with --e10-gpu-hours)"; exit 1; }
    else
      say "E10: no --e10-gpu-hours asserted. Running readiness in REPORT mode (G1 will be red)."
      python scripts/e10_noise_rl_readiness.py >>"$LOG" 2>&1 || true
      say "E10 PARKED: pass --e10-gpu-hours N (>=144 for full / >=72 for trim) to launch GRPO."
      say "<<< E10 parked (not a failure; gated)"; exit 0
    fi
  fi

  # ---- Gate 1: CPU wiring dry-run (validates noise layer + calibration, no GPU)
  step e10_dryrun "E10 wiring dry-run (CPU; noise arm + calibration)" -- \
    python scripts/train_e10_noise_rl.py --arm A_clean --trim --max-steps 1 --dry-run \
    || exit 1

  if [ "$MODE" = "dry" ]; then
    say "E10 --dry: split + readiness-report + CPU wiring dry-run done; no GRPO launched."
    say "<<< E10 (dry) done"
  else
    # ---- Real GRPO arms, STRICTLY SEQUENTIAL (one 7B + 2 LoRA resident at a time).
    # Trimmed default = A,B(s1),C(s1). Full = A, B{1,2,3}, C{1,2,3}, D.
    wait_gpu_free "E10.A"
    step "e10_A_${SCALE_FLAG#--}" "E10 arm A_clean ($SCALE_FLAG)" -- \
      python scripts/train_e10_noise_rl.py --arm A_clean $SCALE_FLAG || exit 1

    if [ "$E10_FULL" = "1" ]; then
      for s in 1 2 3; do
        wait_gpu_free "E10.B.s$s"
        step "e10_B_full_s$s" "E10 arm B_struct full noise-seed $s" -- \
          python scripts/train_e10_noise_rl.py --arm B_struct --full --noise-seed "$s" || exit 1
      done
      for s in 1 2 3; do
        wait_gpu_free "E10.C.s$s"
        step "e10_C_full_s$s" "E10 arm C_random full noise-seed $s" -- \
          python scripts/train_e10_noise_rl.py --arm C_random --full --noise-seed "$s" || exit 1
      done
      wait_gpu_free "E10.D"
      step "e10_D_full" "E10 arm D_corrected full" -- \
        python scripts/train_e10_noise_rl.py --arm D_corrected --full || exit 1
    else
      wait_gpu_free "E10.B.s1"
      step "e10_B_trim_s1" "E10 arm B_struct trim noise-seed 1" -- \
        python scripts/train_e10_noise_rl.py --arm B_struct --trim --noise-seed 1 || exit 1
      wait_gpu_free "E10.C.s1"
      step "e10_C_trim_s1" "E10 arm C_random trim noise-seed 1" -- \
        python scripts/train_e10_noise_rl.py --arm C_random --trim --noise-seed 1 || exit 1
    fi
    say "E10 held-out judging + analysis (separate, paid, post-hoc): GPT-5.2 via"
    say "  run_gpt52_judge_namespaced.py (JUDGE endpoint, never PTU); then the e10_noise_rl post-hoc builder appends canonical['e10_noise_rl']."
    say "<<< E10 done"
  fi
fi

hr
say "QUEUE COMPLETE (mode=$MODE). Sentinels: $DONEDIR/  Log: $LOG"
say "Generation/training was \$0 (local). PAID GPT-5.2 judging is the separate human-launched step per arm above."
hr
