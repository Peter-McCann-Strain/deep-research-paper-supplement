#!/usr/bin/env bash
# Finisher: generation is complete (270/270), so just drain the GPT-5.2 judge
# queue for all oracle patterns. Resume-safe; stops when nothing is pending.
cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate
PATS="oracle_t1_p2,oracle_t1_p3,oracle_t1_p8,oracle_t1_p6,oracle_t1_p7"
for i in $(seq 1 15); do
  echo "=================== finish pass $i ($(date '+%m-%d %H:%M:%S')) ==================="
  python scripts/run_gpt52_judge.py --patterns-raw "$PATS" --resume 2>&1 | grep -iE "judged|pending|done|complete|Total" | tail -6
  njudged=$(find results/judge_gpt52 -path '*oracle_t1*' -name '*.json' 2>/dev/null | wc -l)
  echo "  judged ${njudged}/270"
  if [ "$njudged" -ge 270 ]; then
    echo "=================== ALL 270 ORACLE CELLS JUDGED ($(date '+%m-%d %H:%M:%S')) ==================="
    break
  fi
  sleep 30
done
