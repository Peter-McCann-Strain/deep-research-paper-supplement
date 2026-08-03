#!/usr/bin/env python
"""Paper 4 (judge-reward RL fails) LaTeX tables, booktabs, read-only on canonical.

Reads canonical_numbers.json and writes:
  tab_drjudge_confusion.tex   per-dimension FPR/FNR/J for DR-Judge-7B (error structure)
  tab_drjudge_youden.tex      overall signed Youden's J per judge + phase + gap-bootstrap
  tab_drjudge_structured.tex  structured-vs-random error decomposition (degenerate flag)
  tab_e7_selector.tex         best-of-N selector ladder (gain over single run) + Gate G2

This script DOES NOT mutate canonical_numbers.json.
Run: ./venv/bin/python paper_rebuild/paper_a_bounded_returns/analysis/make_paper4_tables.py
"""
import json, os, warnings
warnings.filterwarnings("ignore")
ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
TAB = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/tables"
os.makedirs(TAB, exist_ok=True)
R = json.load(open(f"{ANA}/canonical_numbers.json"))

DIM_SHORT = {"information_recall": "Info.\\ recall", "factual_accuracy": "Factual",
             "coverage": "Coverage", "analytical_depth": "Depth",
             "citation_quality": "Citation", "logical_coherence": "Coherence",
             "organization": "Org.", "instruction_following": "Instr.",
             "attribution_quality": "Attrib."}
JLAB = {"DR-Judge-7B": "DR-Judge-7B (RL)", "claude_opus": "Opus (panel)",
        "claude_sonnet": "Sonnet (panel)", "gpt52": "GPT-5.2 (panel)"}


def w(name, s):
    open(f"{TAB}/{name}.tex", "w").write(s)
    print("wrote", name)


# ---------- Table A: DR-Judge per-dimension confusion ----------
es = R["drjudge_error_structure"]
c = es["confusion"]
pdim = es["per_dimension"]
order = sorted(pdim, key=lambda d: -pdim[d]["fnr"])
rows = []
for d in order:
    v = pdim[d]
    rows.append(f"{DIM_SHORT[d]} & {v['n']} & {v['fpr']:.3f} & {v['fnr']:.3f} & {v['error_rate']:.3f} \\\\")
tA = (r"""\begin{tabular}{lrrrr}
\toprule
Dimension & $n$ & FPR & FNR & Err.\\
\midrule
""" + "\n".join(rows) + f"""
\\midrule
\\textbf{{Overall}} & {c['n']:,} & {c['fpr']:.3f} & {c['fnr']:.3f} & {c['error_rate']:.3f} \\\\
\\bottomrule
\\end{{tabular}}""")
w("tab_drjudge_confusion", tA)

# ---------- Table B: overall signed Youden's J per judge ----------
yj = R["drjudge_youden_j"]
gap = yj["gap_bootstrap_drjudge_minus_best_panel"]
jorder = ["DR-Judge-7B", "gpt52", "claude_sonnet", "claude_opus"]
rows = []
for jn in jorder:
    ov = yj["judges"][jn]["overall"]
    rows.append(f"{JLAB[jn]} & {ov['n']:,} & {ov.get('kappa', float('nan')):.3f} & "
                f"{ov['youden_j_signed']:.3f} & {ov['phase'].capitalize()} \\\\")
tB = (r"""\begin{tabular}{lrrrl}
\toprule
Judge & $n$ & $\kappa$ & signed $J$ & Phase\\
\midrule
""" + "\n".join(rows) + f"""
\\bottomrule
\\end{{tabular}}
% gap DR-Judge minus best panel (Opus): {gap['obs_gap_overall_J']:.3f}, 95% CI [{gap['ci95'][0]:.3f}, {gap['ci95'][1]:.3f}], excludes 0 = {gap['excludes_0']}""")
w("tab_drjudge_youden", tB)

# ---------- Table C: structured vs random error decomposition ----------
sr = yj["structured_vs_random"]["per_judge"]
rows = []
for jn in jorder:
    v = sr[jn]
    deg = v["degenerate_self_referential_axis"]
    if deg:
        # panel members: split is mechanically 100% structured -> not substantive
        rows.append(f"{JLAB[jn]} & {v['n']:,} & -- & -- & -- & \\textit{{degenerate}}$^\\dagger$ \\\\")
    else:
        rows.append(f"{JLAB[jn]} & {v['n']:,} & {v['err_rate_undisputed']:.3f} & "
                    f"{v['err_rate_disputed']:.3f} & {v['structured_error_fraction']:.3f} & substantive \\\\")
tC = (r"""\begin{tabular}{lrrrrl}
\toprule
Judge & $n$ & Err.\ undisp. & Err.\ disp. & Struct.\ frac. & Axis\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
w("tab_drjudge_structured", tC)

# ---------- Table D: E7 best-of-N selector ladder + Gate G2 ----------
e7 = R["e7_selector_kappa"]
lad = e7["selector_ladder"]
single = lad["single_run_mean"]
RUNGS = [("oracle_upper_bound", "Oracle (argmax true score)"),
         ("gpt52_noise", "GPT-5.2-quality selector"),
         ("gpt4o_noise", "GPT-4o-quality selector"),
         ("random_lower_bound", "Random pick")]
rows = []
for k, lab in RUNGS:
    r = lad[k]
    ci = f"[{r['gain_ci95'][0]:+.4f}, {r['gain_ci95'][1]:+.4f}]"
    rows.append(f"{lab} & {r['selected_mean']:.3f} & {r['gain']:+.4f} & {ci} \\\\")
g2 = e7["gate_g2"]
tD = (r"""\begin{tabular}{lrrc}
\toprule
Selector & Sel.\ mean & Gain & 95\% CI\\
\midrule
""" + f"Single run (baseline) & {single:.3f} & --- & --- \\\\\n\\midrule\n"
      + "\n".join(rows) + f"""
\\bottomrule
\\end{{tabular}}
% Gate G2: max|structured-random|={g2['max_abs_structured_minus_random']:.4f} < {g2['equivalence_threshold']} -> b~=e ({g2['gate_fires_b_approx_e']})""")
w("tab_e7_selector", tD)

print("\nPaper-4 tables written to", TAB)
print(f"DR-Judge overall FPR={c['fpr']:.4f} FNR={c['fnr']:.4f}; "
      f"overall signed J={yj['judges']['DR-Judge-7B']['overall']['youden_j_signed']:.4f}; "
      f"oracle selector gain={lad['oracle_upper_bound']['gain']:+.4f}")
