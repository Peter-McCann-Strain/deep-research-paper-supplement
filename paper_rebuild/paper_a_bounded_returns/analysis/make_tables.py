#!/usr/bin/env python
"""Generate booktabs LaTeX tables from canonical_numbers.json (+ refit variance).
Out: paper_rebuild/paper_a_bounded_returns/tables/*.tex
Run:  ./venv/bin/python paper_rebuild/paper_a_bounded_returns/analysis/make_tables.py
"""
import json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
TAB = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/tables"
os.makedirs(TAB, exist_ok=True)
R = json.load(open(f"{ANA}/canonical_numbers.json"))

PRETTY = {  # pattern -> display name
 "base_p0":"P0 Single-pass","base_p1":"P1 Iterative RAG","base_p2":"P2 Supervisor",
 "base_p3":"P3 MERIDIAN","base_p4":"P4 STORM","base_p5":"P5 Hier.\\ W\\&D",
 "base_p6":"P6 Reactive","base_p7":"P7 Graph","base_p8":"P8 Beam",
 "base_p9":"P9 Qwen2.5-7B","base_p10":"P10 DeepResearcher","base_p11":"P11 ReAct","base_p12":"P12 RL (ours)"}
DIM_SHORT = {"information_recall":"Info.\\ recall","factual_accuracy":"Factual","coverage":"Coverage",
 "analytical_depth":"Depth","citation_quality":"Citation","logical_coherence":"Coherence",
 "organization":"Org.","instruction_following":"Instr.","attribution_quality":"Attrib."}

def w(name, s):
    open(f"{TAB}/{name}.tex","w").write(s); print("wrote", name)

# variance_components is owned by build_numbers.py (single declared REML/lbfgs estimator,
# no optimizer shopping); this script used to carry a second, competing implementation that
# ran last in rebuild_all.sh and silently overwrote build_numbers.py's value on every rebuild
# -- including reintroducing a base_p11/base_p12 pattern-contamination bug after it had
# already been fixed upstream (adversarial review 2026-07-28, round 13). Removed rather than
# fixed a second time: one writer per canonical key.

# ---------- Table 1: headline means ----------
h = R["headline"]["per_pattern"]
order3 = [p for p in R["headline"]["rank_desc"] if p in ("base_p1","base_p4","base_p6","base_p7","base_p8","base_p5","base_p2","base_p3","base_p0","base_p10","base_p9")]
rows = []
for p in order3:
    r = h[p]
    rows.append(f"{PRETTY[p]} & {r['n_cells']} & {r['mean_3judge']:.3f} & {r['std_3judge']:.3f} & "
                f"{r['mean_gpt52']:.3f} & {r['mean_opus']:.3f} & {r['mean_sonnet_corrected']:.3f} \\\\")
t1 = r"""\begin{tabular}{lrrrrrr}
\toprule
Pattern & $N$ & Mean & SD & GPT-5.2 & Opus & Sonnet\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}"""
w("tab_headline_means", t1)

# ---------- Table 2: per-dimension means ----------
pd_ = R["per_dimension"]
dims = list(DIM_SHORT.keys())
hdr = "Pattern & " + " & ".join(DIM_SHORT[d] for d in dims) + r"\\"
rows=[]
for p in order3:
    vals = pd_.get(p,{})
    rows.append(PRETTY[p] + " & " + " & ".join(f"{vals.get(d,float('nan')):.2f}" for d in dims) + r"\\")
t2 = "\\begin{tabular}{l" + "r"*len(dims) + "}\n\\toprule\n" + hdr + "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"
w("tab_per_dimension", t2)

# ---------- Table 3: IRR ----------
irr = R["irr"]
pdrows = "\n".join(f"{DIM_SHORT[d]} & {v:.3f} \\\\" for d,v in irr["per_dimension_alpha"].items())
t3 = f"""\\begin{{tabular}}{{lr}}
\\toprule
Statistic & Value\\\\
\\midrule
Krippendorff $\\alpha$ (overall) & {irr['krippendorff_alpha_overall']:.3f}\\\\
ICC(A,1) & {irr['icc_a1']:.3f}\\\\
ICC(A,$k{{=}}3$) & {irr['icc_ak3']:.3f}\\\\
\\midrule
\\multicolumn{{2}}{{l}}{{\\emph{{Per-dimension }} $\\alpha$:}}\\\\
{pdrows}
\\bottomrule
\\end{{tabular}}"""
w("tab_irr", t3)

# ---------- Table 4: verdict reconciliation (rows sum to total via by_family) ----------
vd = R["verdicts"]
fam = vd["by_family"]
FAMLAB = {"ablation":"Ablations (7 interventions)",
          "variance":"Variance experiment (reruns)","protocol_a":"Bing-vs-Tavily intervention",
          "oracle":"Oracle-retrieval intervention","disentanglement":"Tool-layer disentanglement probe"}
# "base" is split rather than labelled as one "P0--P12" row: pooling the eleven
# canonical patterns with the post-hoc single-judge probes under one "base"
# label contradicted this paper's own "eleven base patterns" framing (fixed
# 2026-07-28, adversarial review round 33).
base_rows = (f"Base patterns (P0--P10) & {vd['base_rows_main11']:,}\\\\\n"
             f"Post-hoc single-judge probes (P11, P12, P11-16-turn, 7B replicates) & "
             f"{vd['base_rows_posthoc_probes']:,}\\\\")
famrows = base_rows + "\n" + "\n".join(f"{FAMLAB.get(k,k)} & {fam[k]:,}\\\\"
                    for k in ["ablation","variance","protocol_a","oracle","disentanglement"]
                    if k in fam)
t4 = f"""\\begin{{tabular}}{{lr}}
\\toprule
Population & Verdicts\\\\
\\midrule
{famrows}
\\midrule
\\textbf{{Total released verdicts}} & \\textbf{{{vd['total_rows']:,}}}\\\\
$\\geq$2-judge consensus triples & {vd['triples_ge2_judges']:,}\\\\
3-judge complete triples & {vd['triples_eq3_judges']:,}\\\\
\\bottomrule
\\end{{tabular}}"""
w("tab_verdicts", t4)

# ---------- Table 5: citation provenance ----------
cp = R["citations"]["per_pattern"]
rows=[]
for p in order3:
    if p in cp:
        r=cp[p]
        rows.append(f"{PRETTY[p]} & {r['cites_per_report']:.1f} & {100*r['placeholder_rate']:.1f} & {100*r['academic_rate']:.1f} & {100*r['real_url_rate']:.1f} \\\\")
t5 = r"""\begin{tabular}{lrrrr}
\toprule
Pattern & Cites/rep & Placeholder \% & Academic \% & Real-URL \%\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}"""
w("tab_citations", t5)

# ---------- Table 6: DR-Judge ----------
dj = R["drjudge"]
t6 = f"""\\begin{{tabular}}{{lrr}}
\\toprule
Subset & $\\kappa$ & $n$\\\\
\\midrule
All test verdicts & {dj['kappa_overall']:.3f} & {dj['n_test']:,}\\\\
Undisputed (panel unanimous) & {dj['kappa_undisputed']:.3f} & {dj['n_undisputed']:,}\\\\
Disputed (panel split) & {dj['kappa_disputed']:.3f} & {dj['n_disputed']:,}\\\\
\\bottomrule
\\end{{tabular}}"""
w("tab_drjudge", t6)

# ---------- Table 7: ablations ----------
ab = R["ablations"]
ABN = {"ablation_p3_no_quality_eval":"P3 $-$ quality-eval","ablation_p3_no_topic_mining":"P3 $-$ topic-mining",
 "ablation_p4_fixed_perspectives":"P4 $-$ adaptive persp.","ablation_p4_no_conversations":"P4 $-$ conversations",
 "ablation_p4_no_triangulation":"P4 $-$ triangulation","ablation_p5_fixed_width":"P5 $-$ adaptive width",
 "ablation_p5_no_meta_eval":"P5 $-$ meta-eval"}
def _fmt_p(p, holm_bound=False):
    # Unified math notation for a p-value column: renders a\times10^{b}, matching the
    # Holm-p style so the Wilcoxon-p and Holm-p columns are typographically consistent.
    # holm_bound=True reports exact-zero underflow (<=1e-4) as the "<10^{-4}" bound the
    # Holm column already used; the Wilcoxon column keeps its real mantissa/exponent.
    if p is None:
        return "--"
    if holm_bound and p <= 1e-4:
        return r"$<10^{-4}$"
    s = f"{p:.1e}"            # e.g. "1.0e+00", "6.9e-02", "9.6e-10"
    mant, exp = s.split("e")
    exp = int(exp)
    if exp == 0:
        return f"${mant}$"
    return rf"${mant}\times10^{{{exp}}}$"
rows=[]
for k in ABN:
    if k in ab and "delta" in ab[k]:
        r=ab[k]
        ci=f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
        wp = _fmt_p(r.get('wilcoxon_p'))
        hp = _fmt_p(r.get('p_holm'), holm_bound=True)
        sig = r"\textsuperscript{$\dagger$}" if r.get('holm_sig') else ""
        rows.append(f"{ABN[k]} & {r['n']} & {r['delta']:+.3f} & {ci} & {wp} & {hp}{sig} \\\\")
t7 = r"""\begin{tabular}{lrrcrr}
\toprule
Ablation & $N$ & $\Delta$ & 95\% CI & Wilcoxon $p$ & Holm $p$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}"""
w("tab_ablations", t7)

# ---------- Table 8: single-judge (P11/P12) ----------
sj = R["single_judge_gpt52"]["per_pattern"]
rows=[]
for p in R["single_judge_gpt52"]["rank_desc"]:
    if p in PRETTY:
        r=sj[p]; rows.append(f"{PRETTY[p]} & {r['mean']:.3f} & {r['std']:.3f} & {r['n']} \\\\")
t8 = r"""\begin{tabular}{lrrr}
\toprule
Pattern & GPT-5.2 mean & SD & $N$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}"""
w("tab_single_judge", t8)
print("\nAll tables written to", TAB)
