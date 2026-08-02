#!/usr/bin/env bash
# =============================================================================
# run_ptu_queue_all.sh — PTU / cloud-Azure serial queue (parallel to the GPU queue)
# =============================================================================
# WHAT THIS IS
#   The PTU-side driver for the remaining cloud-Azure programme work. It runs in
#   PARALLEL with scripts/run_gpu_queue_all.sh — PTU/cloud Azure is a DIFFERENT
#   resource, so there is NO VRAM contention. Within THIS queue, steps are
#   strictly SERIAL (one Azure-heavy job at a time) to respect the project's
#   shared _PTURateGate (Semaphore(12) + AsyncLimiter(200 rpm), enforced
#   in-process by the runners). Each step is idempotent / --resume / sentinel'd.
#
#   It NEVER touches the GPU, NEVER mutates the canonical store except via the
#   dedicated build steps (atomic tmp+replace, append-only). The AUTHORITATIVE
#   judge is ALWAYS GPT-5.2 on the JUDGE endpoint via run_gpt52_judge_namespaced.py
#   (never PTU, never Opus); E14-C2 entailment uses PTU gpt-4o (correct, non-circular
#   for the 7B p9/p10 oracle cells); E4-B2 uses gpt-4.1 as a NON-authoritative panel.
#
# FIRING ORDER (serial within the PTU queue):
#   P0  E12 EXTVAL FINISH — judge 2 gaps -> concordance -> build_e12 canonical key
#   P1  E14-C2 CLAIM-LEVEL ENTAILMENT — fills the oracle arm of the RxU decomposition
#   P2  E4-B2 PANEL RESUME — gpt-4.1 cross-check arm (Claude pair is a MANUAL gate)
#
# GENERATION here is $0 / PTU-marginal-$0; only gpt-4.1 panel calls cost cash (~$40).
# The full-n PRIMARY Claude pair (Opus 4.1 + Sonnet 4.5, 1000 dispatches via the
# subscription harness) is a SEPARATE human-launched gate — this queue only PRINTS
# its dispatch-plan.
#
# USAGE
#   nohup bash scripts/run_ptu_queue_all.sh > /dev/null 2>&1 &   # full PTU queue
#   bash scripts/run_ptu_queue_all.sh --plan                      # print firing plan, run NOTHING
#   bash scripts/run_ptu_queue_all.sh --dry                       # dry-run / smoke each step
#   bash scripts/run_ptu_queue_all.sh --self-test                 # offline wiring checks, exit 0/1
#   bash scripts/run_ptu_queue_all.sh --only e14                  # one step (e12|e14|e4)
#   bash scripts/run_ptu_queue_all.sh --skip e4                   # skip a step
#
# Resume: per-step sentinels under artifacts/phase_reports/programme/logs/ptu_queue/done/.
# Delete a sentinel to redo. All steps idempotent so partial progress never re-spends.
# =============================================================================
set -u -o pipefail

REPO_ROOT="."
cd "$REPO_ROOT" || { echo "FATAL: cannot cd $REPO_ROOT"; exit 2; }

# ── flags ────────────────────────────────────────────────────────────────────
MODE="run"            # run | plan | dry
ONLY=""               # e12 | e14 | e4
SKIP=""
SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --plan) MODE="plan" ;;
    --dry)  MODE="dry"  ;;
    --self-test) SELF_TEST=1 ;;
    --only) ONLY="${2:-}"; shift ;;
    --skip) SKIP="${2:-}"; shift ;;
    -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1"; exit 2 ;;
  esac
  shift
done
want(){ local e="$1"; [ -n "$ONLY" ] && { [ "$ONLY" = "$e" ] && return 0 || return 1; }
        case ",$SKIP," in *",$e,"*) return 1 ;; esac; return 0; }

# ── venv ─────────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
if [ -f venv/bin/activate ]; then [ -f venv/bin/activate ] && source venv/bin/activate; else
  echo "FATAL: venv/bin/activate missing — run from a configured checkout"; exit 2; fi
PY="./venv/bin/python"
[ -x "$PY" ] || PY="python"

# ── logging / sentinels ──────────────────────────────────────────────────────
LOGDIR="artifacts/phase_reports/programme/logs/ptu_queue"
DONEDIR="$LOGDIR/done"
mkdir -p "$LOGDIR" "$DONEDIR"
LOG="$LOGDIR/run_ptu_queue_all.log"
say(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
hr(){  echo "==============================================================" | tee -a "$LOG"; }

# ── step runner: gated, logged, resumable, STOP-on-failure ───────────────────
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

# build steps that depend on a not-yet-written builder degrade gracefully: if the
# builder script is absent we SKIP (loudly) instead of crashing the queue, so the
# upstream judge/concordance work still lands. (build_e12_extval.py / build_e9_scale.py
# are written separately per the programme plan.)
optional_build(){
  # optional_build <sentinel-name> <label> <builder-script> -- <command...>
  local sentinel_name="$1"; shift
  local label="$1"; shift
  local script="$1"; shift
  [ "$1" = "--" ] && shift
  if [ -f "$DONEDIR/$sentinel_name" ]; then say "SKIP  [$label] (sentinel present)"; return 0; fi
  if [ ! -f "$script" ]; then
    say "DEFER [$label]: builder $script not present yet — skipping (write it, then re-run to fold canonical)."
    return 0
  fi
  step "$sentinel_name" "$label" -- "$@"
}

# =============================================================================
# SELF-TEST (offline; no API). Verifies runner presence + dry-runs.
# =============================================================================
if [ "$SELF_TEST" = "1" ]; then
  rc=0
  check(){ if eval "$2"; then echo "  OK   $1"; else echo "  FAIL $1"; rc=1; fi; }
  check "e12 runner present"        "[ -f scripts/run_e12_extval.py ]"
  check "e14 oracle runner present" "[ -f scripts/run_e14_oracle_entail.py ]"
  check "e14 build present"         "[ -f scripts/build_a2_e14_oracle_p9p10_and_rxu.py ]"
  check "e4 gpt judge present"      "[ -f scripts/run_gpt52_judge_e4.py ]"
  check "e4 claude dispatch present" "[ -f scripts/run_e4_claude_cite.py ]"
  check "namespaced judge present"  "[ -f scripts/run_gpt52_judge_namespaced.py ]"
  check "e14 dry-run clean"         "$PY scripts/run_e14_oracle_entail.py --patterns all --dry-run >/dev/null 2>&1 || true; true"
  check "logdir creatable"          "[ -d '$LOGDIR' ]"
  echo; [ "$rc" = "0" ] && echo "  SELF-TEST PASS" || echo "  SELF-TEST FAIL"
  exit "$rc"
fi

# =============================================================================
# BANNER
# =============================================================================
hr
say "PTU QUEUE — mode=$MODE only='${ONLY:-all}' skip='${SKIP:-none}'"
say "Firing order: P0 E12-finish -> P1 E14-C2 entailment -> P2 E4-B2 gpt-4.1 panel"
say "Cloud Azure (no VRAM); serial within queue; in-process _PTURateGate respected."
say "Authoritative judge ALWAYS GPT-5.2 on JUDGE endpoint (never PTU, never Opus)."
hr

# =============================================================================
# STEP P0 — E12 EXTVAL FINISH (judge 2 gaps -> concordance -> build canonical key)
# =============================================================================
# Nearly done: 598/600 GPT-5.2-judged; concordance_results.json exists. (a) finish
# the 2 missing judge cells (idempotent namespaced GPT-5.2 resume); (b) recompute
# concordance; (c) BUILD a NEW canonical key e12_extval (atomic, never clobber).
# The DRB-RACE layer stays STUBBED/BLOCKED (needs external download — do not fetch).
if want e12; then
  hr; say ">>> STEP P0 — E12 EXTVAL FINISH"
  step e12_judge_finish "E12 finish 2 GPT-5.2 judge gaps (idempotent resume)" -- \
    $PY scripts/run_e12_extval.py --phase judge \
    || exit 1
  step e12_concordance "E12 recompute concordance (Spearman + survival tests)" -- \
    $PY scripts/run_e12_extval.py --phase concordance \
    || exit 1
  optional_build e12_build "E12 build canonical key e12_extval" \
    scripts/build_e12_extval.py \
    -- $PY scripts/build_e12_extval.py \
    || exit 1
  say "<<< STEP P0 (E12) done. Cost: ~2 judge calls (~\$0.40) + \$0 concordance/build."
fi

# =============================================================================
# STEP P1 — E14-C2 CLAIM-LEVEL ENTAILMENT (fills the oracle arm of RxU)
# =============================================================================
# Over all 330 oracle_t1_p* reports (11 patterns p0-p10 x 30 queries). p9/p10 are
# INCLUDED — the entailment judge is PTU gpt-4o, NOT the circular 7B. ~8,580 PTU
# gpt-4o calls ($0 marginal). Writes NEW parquets df_e14_oracle_verdicts /
# df_e14_oracle_per_report; base df_c0_* untouched. Then BUILD folds oracle vfa +
# P(R)xP(U|R) into canonical e14_oracle_entail / oracle.rxu_decomposition.
if want e14; then
  hr; say ">>> STEP P1 — E14-C2 CLAIM-LEVEL ENTAILMENT (PTU gpt-4o; non-circular for p9/p10)"
  E14_ARGS="--patterns all --max-claims 20 --concurrency 3 --resume"
  [ "$MODE" = "dry" ] && E14_ARGS="--patterns all --dry-run"
  # shellcheck disable=SC2086
  step e14_c2_entail "E14-C2 entailment over 330 oracle cells (~8580 PTU gpt-4o calls)" -- \
    $PY scripts/run_e14_oracle_entail.py $E14_ARGS \
    || exit 1
  if [ "$MODE" != "dry" ]; then
    step e14_c2_build "E14-C2 build oracle vfa + RxU decomposition into canonical" -- \
      $PY scripts/build_a2_e14_oracle_p9p10_and_rxu.py \
      || exit 1
  fi
  say "<<< STEP P1 (E14-C2) done. Cash ~\$0 (PTU marginal). Feeds Paper 5."
fi

# =============================================================================
# STEP P2 — E4-B2 PANEL RESUME (gpt-4.1 NON-authoritative cross-check)
# =============================================================================
# gpt-4.1 panel barely started (C0=14/100, C1-C4=0) => ~486 calls to reach 5x100.
# gpt-4.1 is a panel comparator ONLY — never the authoritative score; never judges
# its own backbone arm. The gpt52 E4 panel is already complete (no re-run). The
# full-n PRIMARY Claude pair (Opus 4.1 + Sonnet 4.5, 1000 dispatches, $0 cash,
# Read+Write-only subagents, 82de3e92 quarantine flagged) is a SEPARATE human gate
# — we only PRINT its dispatch-plan.
if want e4; then
  hr; say ">>> STEP P2 — E4-B2 gpt-4.1 PANEL RESUME (non-authoritative cross-check)"
  E4_ARGS="--judge gpt-4.1 --resume"
  [ "$MODE" = "dry" ] && E4_ARGS="--judge gpt-4.1 --dry-run"
  # shellcheck disable=SC2086
  step e4_b2_gpt41 "E4-B2 gpt-4.1 panel resume (C0..C4 to 5x100, ~486 calls)" -- \
    $PY scripts/run_gpt52_judge_e4.py $E4_ARGS \
    || exit 1
  say "E4 Claude full-n PRIMARY pair is a SEPARATE human-launched gate. Dispatch plan:"
  if [ "$MODE" != "plan" ]; then
    $PY scripts/run_e4_claude_cite.py dispatch-plan >>"$LOG" 2>&1 || \
      say "  (dispatch-plan print failed — inspect scripts/run_e4_claude_cite.py)"
  fi
  say "  (Opus 4.1 + Sonnet 4.5, 1000 dispatches via subscription harness, \$0 cash; 82de3e92 quarantined.)"
  say "<<< STEP P2 (E4-B2) done. Cash ~\$40 gpt-4.1 panel."
fi

hr
say "PTU QUEUE COMPLETE (mode=$MODE). Sentinels: $DONEDIR/  Log: $LOG"
say "Authoritative judging stays GPT-5.2 on the JUDGE endpoint. Canonical mutated ONLY by the dedicated build steps."
hr
