#!/usr/bin/env python
"""Three-way (run / query / judge) REML variance-decomposition table for Paper A.

READ-ONLY on canonical_numbers.json['variance_decomposition']['three_way']. Writes
tables/tab_three_way.tex. Reports the crossed-RE variance components on the balanced
two-judge intersection (gpt52 x sonnet48) over the replicate matrix: the run facet
(re-running the SAME architecture) carries the largest variance share, query next,
judge smallest -- so most score variance is sampling/query noise, not architecture, which
is exactly why the architecture cluster is statistically flat. The judge facet has only
two levels, so a cross-judge anchor (per-judge means + paired-cell correlation) is reported
as an identification-free companion to the low-precision sigma2_judge.
"""
import json, os, warnings
warnings.filterwarnings("ignore")
ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
TAB = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/tables"
os.makedirs(TAB, exist_ok=True)
TW = json.load(open(f"{ANA}/canonical_numbers.json"))["variance_decomposition"]["three_way"]
R = TW["reml_3way"]
A = TW["cross_judge_anchor"]
vf = R["var_fraction"]

rows = [("Run (re-run same arch.)", R["sigma2_run"], vf["run"]),
        ("Query", R["sigma2_query"], vf["query"]),
        ("Judge", R["sigma2_judge"], vf["judge"]),
        ("Residual", R["sigma2_resid"], vf["resid"])]

lines = [r"\begin{tabular}{lrr}", r"\toprule",
         r"Variance component & $\sigma^2$ & share \\", r"\midrule"]
for lab, s2, fr in rows:
    lines.append(f"{lab} & {s2:.5f} & {fr*100:.1f}\\% \\\\")
lines += [r"\midrule",
          f"\\multicolumn{{3}}{{l}}{{\\footnotesize $n_{{\\mathrm{{obs}}}}={R['n_obs']}$, REML crossed, "
          f"grand mean {R['grand_mean']:.3f}; ICC$_{{\\mathrm{{query}}}}={R['icc_query']:.3f}$}} \\\\",
          f"\\multicolumn{{3}}{{l}}{{\\footnotesize Judge anchor (2 levels): GPT-5.2 "
          f"{A['per_judge_mean_overall']['gpt52']:.3f} vs Sonnet "
          f"{A['per_judge_mean_overall']['claude_sonnet48']:.3f},}} \\\\",
          f"\\multicolumn{{3}}{{l}}{{\\footnotesize paired-cell $r={A['paired_cell_corr_pearson']:.3f}$ "
          f"($n={A['n_paired_cells']}$ cells)}} \\\\",
          r"\bottomrule", r"\end{tabular}"]
open(f"{TAB}/tab_three_way.tex", "w").write("\n".join(lines) + "\n")
print("wrote tab_three_way.tex; run/query/judge/resid shares =",
      f"{vf['run']:.3f}/{vf['query']:.3f}/{vf['judge']:.3f}/{vf['resid']:.3f}; n_obs={R['n_obs']}")
