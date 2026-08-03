#!/usr/bin/env python
"""LOSO source-jackknife LaTeX table for Paper A (composition-artefact defence).

READ-ONLY on canonical_numbers.json. Writes tables/tab_loso_paperA.tex.
One row per leave-one-source-out refit plus the full-panel baseline: dropped source,
queries remaining, ICC_query (Gate-1), Gate-3 judge-robust separations (of 55),
top-1 pattern, and max rank displacement vs the full 3-judge rank table.
"""
import json, os, warnings
warnings.filterwarnings("ignore")
ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
TAB = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/tables"
os.makedirs(TAB, exist_ok=True)
L = json.load(open(f"{ANA}/canonical_numbers.json"))["loso_robustness"]

SLAB = {"deepsearch_qa": "DeepSearch-QA", "draco": "DRACO", "litqa2": "LitQA2",
        "custom": "Custom", "research_qa": "Research-QA"}
PLAB = lambda p: p.replace("base_p", "P")

base = L["full_panel_baseline"]
loso = L["leave_one_source_out"]

lines = [r"\begin{tabular}{lrrrlr}", r"\toprule",
         r"Drop & $n_q$ & ICC$_{\mathrm{query}}$ & Gate-3 & Top-1 & $\Delta$rank \\",
         r"\midrule"]
# full-panel baseline row
lines.append(
    f"\\textit{{(full panel)}} & {base['n_queries']} & {base['gate1_mixed']['icc_query']:.3f} & "
    f"{base['gate3_judge_robust_of_55']}/55 & {PLAB(base['rank_table']['rank_desc'][0])} & --- \\\\")
lines.append(r"\midrule")
for s, v in loso.items():
    top1 = PLAB(v["top1_pattern"])
    held = r"$^{\checkmark}$" if v["top1_matches_full"] else r"$^{\times}$"
    lines.append(
        f"$-$\\,{SLAB[s]} & {v['n_queries_remaining']} & {v['gate1_mixed']['icc_query']:.3f} & "
        f"{v['gate3_judge_robust_of_55']}/55 & {top1}{held} & {v['rank_max_displacement_vs_full']} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{TAB}/tab_loso_paperA.tex", "w").write("\n".join(lines) + "\n")
print("wrote tab_loso_paperA.tex;",
      f"full ICC_q={base['gate1_mixed']['icc_query']:.3f} Gate-3={base['gate3_judge_robust_of_55']}/55;",
      "drops top1 held:", all(v["top1_matches_full"] for v in loso.values()),
      "max disp:", max(v["rank_max_displacement_vs_full"] for v in loso.values()))
