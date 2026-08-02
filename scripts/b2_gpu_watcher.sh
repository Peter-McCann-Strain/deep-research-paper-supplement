#!/usr/bin/env bash
# B2 external-validity watcher: the orchestration-on-7B run needs ~13GB free VRAM, but the
# card is currently shared with another project's 5GB process. This watcher polls every 10
# minutes and, as soon as enough VRAM frees up, runs P1 (and then P4) on the local 7B over a
# 12-query subset, then GPT-5.2-judges them. P1_7B vs P9 (P0-arch on 7B) isolates the
# orchestration premium at 7B, for comparison with P1-vs-P0 at GPT-4o scale.
set -e
cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate
python -c "import json; json.dump({'query_ids':['q1_bert_vs_gpt','q2_rag_vs_finetuning','q3_single_vs_multi_agent','q4_paperqa_storm_autosurvey','q5_lost_in_middle','dsqa_0712','dsqa_0683','dsqa_0650','174539434801411914-s20','174539435106414397-s6','67dfb8ea-cc84-4fb9-abc2-4794aa20eb44','b02ede76-2353-40f3-9da9-d319c617ab0d'],'seed':42}, open('data/b2_subset.json','w'))"
NEED=13000
for i in $(seq 1 144); do  # up to 24h
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "[$(date '+%m-%d %H:%M')] GPU free=${FREE}MiB (need ${NEED})"
  if [ "${FREE:-0}" -ge "$NEED" ]; then
    echo "=== enough VRAM; running B2 (P1 then P4 on 7B) ==="
    for p in p1 p4; do
      echo "--- ${p} on 7B ($(date '+%H:%M')) ---"
      DR_LOCAL_LLM=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python scripts/run_all_experiments.py --pattern $p --run-tag 7b \
        --query-ids-file data/b2_subset.json --base-only --resume || echo "  ${p} run errored (likely 7B flooring on hard queries; partial reports still usable)"
    done
    echo "=== GPT-5.2 judging the 7B orchestration reports ==="
    python scripts/run_gpt52_judge.py --patterns-raw base_p1_7b,base_p4_7b --resume 2>&1 | tail -4
    echo "=== B2 COMPLETE ($(date '+%m-%d %H:%M')) ==="
    break
  fi
  sleep 600
done
