#!/usr/bin/env bash
# GPU queue #4 — E13' local DETECTOR panel. Chains after the oracle queue (queue3) AND the
# perturbation set (PTU). Runs phi-4 / Mistral-7B / DeepSeek-Distill-8B as constructed-truth
# DETECTORS (not judges) on the injected-defect corpus, computing per-family detection ROC with a
# pre-specified floor. GPU, one model at a time. Launch:  nohup bash scripts/run_gpu_queue4.sh &
set -u
cd .
[ -f venv/bin/activate ] && source venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG=reports/phase_reports/logs/gpu_queue4.log
mkdir -p reports/phase_reports/logs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "queue4 (E13' detector panel) started."
say "  waiting for oracle queue3 to finish (GPU free)..."
until grep -aq "GPU QUEUE3 COMPLETE" reports/phase_reports/logs/gpu_queue3.log 2>/dev/null; do sleep 120; done
say "  waiting for the perturbation set (ground_truth.jsonl) to be populated..."
until [ -s reports/perturbation_set/ground_truth.jsonl ]; do sleep 120; done
say "  waiting for GPU to be free..."
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')" -lt 3000 ]; do sleep 30; done
say "prereqs met; running the local detector panel on GPU (phi-4, Mistral-7B, DeepSeek-Distill-8B)."
python scripts/run_detector_panel.py >>"$LOG" 2>&1
say "GPU QUEUE4 COMPLETE (E13' detector panel -> reports/perturbation_set/detector_results.json)."
say "GPU pipeline drained: vintage (p14) + oracle (p9/p10) + detector panel done. Next heavy GPU arm = E10 RL (gated on E7, multi-week)."
