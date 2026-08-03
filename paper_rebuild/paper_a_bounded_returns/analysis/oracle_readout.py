#!/usr/bin/env python
"""Oracle Tier-1 read-out: does better retrieval move the cluster?

Paired comparison (same 30 variance queries, GPT-5.2 judge) of each pattern's
oracle-corpus score vs its own baseline score. Answers:
  (1) do scores rise under ideal sources?  (2) does the gap from P0 to the cluster close?
  (3) does the cluster re-order?
Re-run anytime; uses whatever oracle cells are judged so far.
"""
import json, glob, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = "."
JG = f"{ROOT}/results/judge_gpt52"
VARQ = set(json.load(open(f"{ROOT}/data/variance_stratified.json"))["query_ids"])
NAME = {f"p{i}": n for i, n in zip(range(11),
  ["P0 Single-pass","P1 Iterative","P2 Supervisor","P3 MERIDIAN","P4 STORM",
   "P5 Hier.W&D","P6 Reactive","P7 Graph","P8 Beam","P9 Qwen7B","P10 DeepRes"])}
CLUSTER = {"p1","p4","p5","p6","p7","p8"}

def scores(judge_dir):
    out = {}
    for f in glob.glob(f"{JG}/{judge_dir}/*.json"):
        qid = f.split("/")[-1][:-5]
        if qid in VARQ:
            out[qid] = json.load(open(f)).get("overall_score")
    return out

rows = []
for p in [f"p{i}" for i in range(9)]:
    base = scores(f"base_{p}")
    orac = scores(f"oracle_t1_{p}")
    common = [q for q in orac if q in base and orac[q] is not None and base[q] is not None]
    if not orac:
        continue
    bmean = np.mean([base[q] for q in base]) if base else float("nan")
    omean = np.mean([orac[q] for q in orac])
    pmean_b = np.mean([base[q] for q in common]) if common else float("nan")
    pmean_o = np.mean([orac[q] for q in common]) if common else float("nan")
    rows.append((p, NAME[p], len(base), len(orac), len(common), bmean, omean,
                 pmean_o - pmean_b if common else float("nan")))

print(f"{'pat':4s} {'name':14s} {'nBase':>5s} {'nOrac':>5s} {'nPair':>5s} "
      f"{'base':>6s} {'oracle':>6s} {'pairedΔ':>8s}")
for p, nm, nb, no, nc, bm, om, d in sorted(rows, key=lambda r: -(r[6])):
    star = " *cluster" if p in CLUSTER else (" baseline" if p == "p0" else "")
    print(f"{p:4s} {nm:14s} {nb:5d} {no:5d} {nc:5d} {bm:6.3f} {om:6.3f} {d:+8.3f}{star}")

# headline reads
orac_means = {p: om for p, nm, nb, no, nc, bm, om, d in rows}
base_means = {p: bm for p, nm, nb, no, nc, bm, om, d in rows}
if "p0" in orac_means:
    cl = [orac_means[p] for p in CLUSTER if p in orac_means]
    print("\n--- READ-OUT (GPT-5.2, variance-30 subset) ---")
    print(f"oracle P0 = {orac_means['p0']:.3f} vs baseline cluster mean(gpt52) ~0.46 "
          f"-> {'P0 now MEETS/EXCEEDS the baseline cluster' if orac_means['p0']>=0.46 else 'P0 still below cluster'}")
    if len(cl) >= 3:
        print(f"oracle cluster mean = {np.mean(cl):.3f} (baseline ~0.45); "
              f"P0-to-cluster gap: baseline {0.46-base_means['p0']:.3f} -> oracle {np.mean(cl)-orac_means['p0']:.3f}")
    print("patterns with oracle data:", sorted(orac_means.keys()))
