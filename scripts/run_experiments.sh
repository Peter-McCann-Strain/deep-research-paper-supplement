#!/usr/bin/env bash
# =============================================================================
# Deep Research Experiment Campaign Runner
# =============================================================================
#
# Runs all experiments sequentially to avoid rate-limit conflicts.
# Checkpointed — safe to interrupt and resume.
#
# Usage:
#   ./scripts/run_experiments.sh              # Run everything (recommended)
#   ./scripts/run_experiments.sh base          # Only base patterns (11 × 90 queries)
#   ./scripts/run_experiments.sh ablations     # Only ablation sweeps (36 × 5 queries)
#   ./scripts/run_experiments.sh fast          # Quick test: 1 query per experiment
#   ./scripts/run_experiments.sh pattern p6    # Single pattern, all queries
#   ./scripts/run_experiments.sh dry           # Show what would run
#
# Estimated times (sequential):
#   Base patterns (990 runs):  ~50-80 hours
#   Ablation sweeps (180 runs): ~20-30 hours
#   Total: ~70-110 hours
#
# All GPT-4o calls are FREE (PTU). No cost for base runs or GPT-4o ablations.
# P9/P10 local model runs: FREE (local GPU).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Activate virtual environment
[ -f venv/bin/activate ] && source venv/bin/activate

# Logging
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/experiments_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=============================================="
log "Deep Research Experiment Campaign"
log "=============================================="
log "Project: $PROJECT_DIR"
log "Log: $LOG_FILE"
log ""

MODE="${1:-all}"
EXTRA="${2:-}"

case "$MODE" in
    base)
        log "Mode: BASE PATTERNS ONLY (11 patterns × 90 queries = 990 runs)"
        python scripts/run_all_experiments.py --base-only 2>&1 | tee -a "$LOG_FILE"
        ;;

    ablations)
        log "Mode: ABLATION SWEEPS ONLY (36 configs × 5 representative queries = 180 runs)"
        python scripts/run_all_experiments.py --ablations-only --ablation-queries 5 2>&1 | tee -a "$LOG_FILE"
        ;;

    all)
        log "Mode: FULL CAMPAIGN"
        log ""
        log "Phase 1: Base patterns (11 × 90 = 990 runs)"
        log "Phase 2: Ablation sweeps (36 × 5 representative = 180 runs)"
        log "Total: 1170 runs"
        log ""

        # Phase 1: All base patterns on all 90 queries
        log "=== PHASE 1: BASE PATTERNS ==="
        python scripts/run_all_experiments.py --base-only 2>&1 | tee -a "$LOG_FILE"

        log ""
        log "=== PHASE 2: ABLATION SWEEPS ==="
        # Phase 2: All ablations on 5 representative queries
        python scripts/run_all_experiments.py --ablations-only --ablation-queries 5 2>&1 | tee -a "$LOG_FILE"
        ;;

    fast)
        log "Mode: FAST TEST (all experiments × 1 query)"
        python scripts/run_all_experiments.py --query q1_bert_vs_gpt 2>&1 | tee -a "$LOG_FILE"
        ;;

    pattern)
        if [ -z "$EXTRA" ]; then
            echo "Usage: $0 pattern <pattern_key>  (e.g. p6, p7, p8)"
            exit 1
        fi
        log "Mode: SINGLE PATTERN $EXTRA (90 queries)"
        python scripts/run_all_experiments.py --base-only --pattern "$EXTRA" 2>&1 | tee -a "$LOG_FILE"
        ;;

    ablation)
        if [ -z "$EXTRA" ]; then
            echo "Usage: $0 ablation <ablation_id>  (e.g. p7_deep_graph)"
            exit 1
        fi
        log "Mode: SINGLE ABLATION $EXTRA (5 representative queries)"
        python scripts/run_all_experiments.py --ablations-only --ablation "$EXTRA" --ablation-queries 5 2>&1 | tee -a "$LOG_FILE"
        ;;

    dry)
        log "Mode: DRY RUN"
        python scripts/run_all_experiments.py --dry-run --ablation-queries 5 2>&1 | tee -a "$LOG_FILE"
        ;;

    *)
        echo "Usage: $0 {all|base|ablations|fast|pattern <key>|ablation <id>|dry}"
        echo ""
        echo "Modes:"
        echo "  all        Full campaign: base (990) + ablations (180) = 1170 runs"
        echo "  base       Base patterns only: 11 × 90 = 990 runs"
        echo "  ablations  Ablation sweeps only: 36 × 5 = 180 runs"
        echo "  fast       Quick test: all experiments × 1 query = 47 runs"
        echo "  pattern    Single base pattern, all 90 queries"
        echo "  ablation   Single ablation config, 5 representative queries"
        echo "  dry        Show what would run without executing"
        exit 1
        ;;
esac

log ""
log "Experiment run complete. Log saved to $LOG_FILE"
