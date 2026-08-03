#!/usr/bin/env bash
# Frozen-vintage decode-backend de-confound: regenerate p9 + p14 via GGUF (so all 4 arms
# share the llama.cpp backend), re-judge, rebuild the frozen_vintage canonical key.
cd .
[ -f venv/bin/activate ] && source venv/bin/activate
export LD_LIBRARY_PATH="./.cudatk/lib:$(find venv/lib/python3.12/site-packages/nvidia -name lib -type d|tr '\n' ':')$LD_LIBRARY_PATH"
RV=results/experiments_frozen_vintage
echo "===== DE-CONFOUND START $(date -u) ====="
# 1) regenerate p9 + p14 via GGUF on the same 89 frozen sources
for arm in base_p9 base_p14_vintage_deepseek_qwen7b; do
  echo "[1] regen $arm (GGUF) $(date -u)"
  python scripts/run_frozen_vintage.py --arm $arm 2>&1 | tail -3
  echo "    $arm: $(ls $RV/$arm/*.md 2>/dev/null|wc -l)/89"
done
# 2) clear the stale (transformers) verdicts for p9+p14 so they re-judge the new GGUF reports
echo "[2] clearing stale p9+p14 verdicts"
find results/judge_gpt52_frozen_vintage -path '*base_p9*' -name '*.json' -delete 2>/dev/null
find results/judge_gpt52_frozen_vintage -path '*base_p14*' -name '*.json' -delete 2>/dev/null
# 3) re-judge all 4 arms (p9+p14 fresh, p13+p17 cached/idempotent)
echo "[3] re-judge (GPT-5.2) $(date -u)"
JUDGE_RESULTS_BASE=$RV python scripts/run_gpt52_judge_namespaced.py \
  --judge-out results/judge_gpt52_frozen_vintage \
  --patterns-raw base_p9,base_p14_vintage_deepseek_qwen7b,base_p13_vintage_qwen3_8b,base_p17_scale_qwen25_14b \
  --resume 2>&1 | tail -8
# 4) rebuild frozen_vintage (dry-run; human lands it in the guarded Phase-5 rebuild)
echo "[4] frozen_vintage DRY-RUN (de-confounded; all 4 arms now GGUF)"
python scripts/build_frozen_vintage.py --dry-run 2>&1 | grep -iE "raw_overall_mean|length_adjusted|p17_minus_p9|p14_minus_p9|p13" | head -12
echo "===== DE-CONFOUND DONE $(date -u) — review then --write in Phase-5 rebuild ====="
