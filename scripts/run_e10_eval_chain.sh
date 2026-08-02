#!/usr/bin/env bash
cd .
[ -f venv/bin/activate ] && source venv/bin/activate
LOG=artifacts/phase_reports/programme/logs
echo "===== E10 EVAL CHAIN START $(date -u) ====="
# 1) generate held-out reports for all 8 arms (GPU, transformers QLoRA, $0)
echo "[1] GENERATE $(date -u)"
python scripts/run_e10_eval.py 2>&1 | tail -25
# 2) hard leak check
if ls results/experiments_e10/e10_*/82de3e92*.md >/dev/null 2>&1; then echo "[ABORT] quarantine leak detected"; exit 1; fi
echo "[2] no quarantine leak confirmed"
# 3) GPT-5.2 judge (the only paid step; ~$24)
echo "[3] JUDGE (GPT-5.2) $(date -u)"
JUDGE_RESULTS_BASE=results/experiments_e10 python scripts/run_gpt52_judge_namespaced.py \
  --judge-out results/judge_gpt52_e10 --experiment-tag e10 \
  --patterns-raw e10_A,e10_B_s1,e10_B_s2,e10_B_s3,e10_C_s1,e10_C_s2,e10_C_s3,e10_D \
  --resume --concurrency 5 2>&1 | tail -15
# 4) build DRY-RUN only (print B-vs-C/D-A; human reviews before --write)
echo "[4] BUILD --dry-run (NO canonical write yet) $(date -u)"
python scripts/build_e10_noise_rl.py --self-test 2>&1 | tail -2
python scripts/build_e10_noise_rl.py --dry-run 2>&1 | tail -30
echo "===== E10 EVAL CHAIN DONE — review deltas, then build_e10_noise_rl.py --write $(date -u) ====="
