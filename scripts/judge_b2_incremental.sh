#!/usr/bin/env bash
# Incrementally GPT-5.2-judge the B2 7B reports (base_p1_7b, base_p4_7b) as they generate,
# so judging overlaps with the slow 7B generation. Resume-safe; stops when generation is
# done and all generated reports are judged.
cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate
for i in $(seq 1 60); do
  python scripts/run_gpt52_judge.py --patterns-raw base_p1_7b,base_p4_7b --resume 2>&1 | grep -iE 'judged|pending|complete' | tail -2
  genalive=$(ps aux | grep -cE '[r]un_all_experiments.*7b')
  gen=$(find results/experiments/base_p1_7b results/experiments/base_p4_7b -name '*.md' 2>/dev/null | wc -l)
  jud=$(find results/judge_gpt52/base_p1_7b results/judge_gpt52/base_p4_7b -name '*.json' 2>/dev/null | wc -l)
  echo "  [$(date +%H:%M)] gen ${gen}/24 | judged ${jud} | gen-alive ${genalive}"
  if [ "$genalive" -eq 0 ] && [ "$jud" -ge "$gen" ] && [ "$gen" -ge 1 ]; then echo "B2 JUDGING DONE"; break; fi
  sleep 180
done
