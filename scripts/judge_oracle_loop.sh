#!/usr/bin/env bash
# Autonomous GPT-5.2 judging loop for the oracle Tier-1 arm.
# Re-judges (resume-safe) every ~8 min, picking up newly-generated oracle reports,
# until all 270 cells (9 patterns x 30 queries) are judged AND generation has stopped.
# Runs on the GPT-5.2 judge endpoint — separate from the PTU, no generation contention.
cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate
PATS="oracle_t1_p0,oracle_t1_p1,oracle_t1_p4,oracle_t1_p5,oracle_t1_p2,oracle_t1_p3,oracle_t1_p8,oracle_t1_p6,oracle_t1_p7"
for i in $(seq 1 90); do
  echo "=================== GPT-5.2 oracle judge pass $i ($(date '+%H:%M:%S')) ==================="
  python scripts/run_gpt52_judge.py --patterns-raw "$PATS" --resume 2>&1 | grep -iE "judged|pending|Estimated|done|complete|Total" | tail -6
  njudged=$(find results/judge_gpt52 -path '*oracle_t1*' -name '*.json' 2>/dev/null | wc -l)
  ngen=$(find results/experiments -path '*oracle_t1*' -name '*.md' 2>/dev/null | wc -l)
  genalive=$(ps aux | grep -c '[l]aunch_oracle_t1')
  echo "  judged ${njudged} / generated ${ngen} (target 270) | gen alive: ${genalive}"
  if [ "$njudged" -ge "$ngen" ] && [ "$genalive" -eq 0 ] && [ "$ngen" -ge 270 ]; then
    echo "=================== ALL ORACLE T1 GENERATED + JUDGED ($(date '+%H:%M:%S')) ==================="
    break
  fi
  sleep 480
done
