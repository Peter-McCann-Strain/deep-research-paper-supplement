#!/usr/bin/env python
"""Paper 3 (Randomness / variance) -- LaTeX tables from canonical_numbers.json.

Reads canonical['variance_decomposition'] live (never hardcodes a produced number).
Emits bare booktabs tabular blocks (same convention as the other tables/*.tex):
  tab_mde_grid.tex        : MDE80 by (n queries x r replicates) given run noise.
  tab_three_way.tex       : crossed run x query x judge REML variance components.
  tab_run_bootstrap.tex   : per-architecture sigma2_query / sigma2_run / ICC_query
                            with parametric-bootstrap 95% CIs.
  tab_var_bayes.tex       : Bayesian beta-binomial cross-check of the small-n claims.
"""
import json, os, math
ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
TAB = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/tables"
os.makedirs(TAB, exist_ok=True)
V = json.load(open(f"{ANA}/canonical_numbers.json"))["variance_decomposition"]

PRETTY = {"base_p0": "P0 Single-pass", "base_p1": "P1 Iterative RAG", "base_p4": "P4 STORM",
          "base_p5": "P5 Hier.\\ W\\&D", "base_p6": "P6 Reactive", "base_p7": "P7 Graph",
          "base_p8": "P8 Beam", "base_p10": "P10 DeepResearcher"}


def w(name, s):
    open(f"{TAB}/{name}.tex", "w").write(s)
    print("wrote", name)


# ---------- tab_mde_grid ----------
def mde_grid():
    g = V["mde"]["grid"]
    # keys like n30_r1 -> (30,1)
    cells = {}
    ns, rs = set(), set()
    for k, v in g.items():
        n = int(k.split("_")[0][1:]); r = int(k.split("_")[1][1:])
        cells[(n, r)] = v; ns.add(n); rs.add(r)
    ns, rs = sorted(ns), sorted(rs)
    head = " & ".join(["$n$ \\textbackslash\\ $r$"] + [f"$r{{=}}{r}$" for r in rs])
    lines = [f"\\begin{{tabular}}{{l{'r'*len(rs)}}}", "\\toprule", head + "\\\\", "\\midrule"]
    for n in ns:
        row = [f"$n{{=}}{n}$"] + [f"{cells[(n,r)]:.4f}" if (n, r) in cells else "--" for r in rs]
        lines.append(" & ".join(row) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    w("tab_mde_grid", "\n".join(lines) + "\n")


# ---------- tab_three_way ----------
def three_way():
    t = V["three_way"]["reml_3way"]
    vf = t["var_fraction"]
    order = [("run", "sigma2_run", "Run (re-execution)"),
             ("query", "sigma2_query", "Query difficulty"),
             ("judge", "sigma2_judge", "Judge stringency"),
             ("resid", "sigma2_resid", "Residual")]
    lines = ["\\begin{tabular}{lrr}", "\\toprule",
             "Component & $\\sigma^2$ & \\% of total\\\\", "\\midrule"]
    for fk, sk, lab in order:
        lines.append(f"{lab} & {t[sk]:.4f} & {vf[fk]*100:.1f}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    w("tab_three_way", "\n".join(lines) + "\n")


# ---------- tab_run_bootstrap ----------
def run_bootstrap():
    bc = V["bootstrap_ci"]["bootstrap_ci"]
    order = [a for a in ["base_p0", "base_p1", "base_p4", "base_p7", "base_p10",
                         "base_p5", "base_p6", "base_p8"] if a in bc]
    lines = ["\\begin{tabular}{lrrr}", "\\toprule",
             "Architecture & $\\sigma^2_{\\mathrm{query}}$ & $\\sigma^2_{\\mathrm{run}}$ "
             "& ICC$_{\\mathrm{query}}$\\\\",
             "\\multicolumn{4}{l}{\\footnotesize point [95\\% parametric-bootstrap CI]}\\\\",
             "\\midrule"]
    for a in order:
        d = bc[a]; p = d["point"]; ci = d["ci"]
        rag = "$^{\\dagger}$" if d["coverage"] == "ragged" else ""
        sq = f"{p['sigma2_query']:.4f}\\,[{ci['sigma2_query']['lo']:.4f},{ci['sigma2_query']['hi']:.4f}]"
        sr = f"{p['sigma2_run']:.4f}\\,[{ci['sigma2_run']['lo']:.4f},{ci['sigma2_run']['hi']:.4f}]"
        iq = f"{p['icc_query']:.3f}\\,[{ci['icc_query']['lo']:.3f},{ci['icc_query']['hi']:.3f}]"
        lines.append(f"{PRETTY[a]}{rag} & {sq} & {sr} & {iq}\\\\")
    lines += ["\\bottomrule",
              "\\multicolumn{4}{l}{\\footnotesize $^{\\dagger}$ragged replicate coverage "
              "($<30$ queries); CIs wide, near the $\\sigma^2{\\geq}0$ boundary.}\\\\",
              "\\end{tabular}"]
    w("tab_run_bootstrap", "\n".join(lines) + "\n")


# ---------- tab_var_bayes ----------
def var_bayes():
    b = V["bayes_crosscheck"]
    of = b["bayes_oracle_factual_tost"]
    mde = b["clt_vs_bayes_mde"]
    lines = ["\\begin{tabular}{lrr}", "\\toprule",
             "Claim (Bayesian beta-binomial) & Estimate & 94\\% interval\\\\", "\\midrule"]
    # oracle factual TOST
    ci = of["cluster_delta_hdi94"]
    lines.append("Oracle$-$base factual $\\Delta$ (cluster) "
                 f"& {of['posterior_mean_cluster_delta']:+.4f} & "
                 f"$[{ci[0]:+.4f},{ci[1]:+.4f}]$\\\\")
    lines.append("\\quad $P(|\\Delta|<0.05)$ & "
                 f"{of['bayes_tost_0.05']['p_practical_equivalence']:.3f} & ---\\\\")
    lines.append("\\quad $P(|\\Delta|<0.02)$ & "
                 f"{of['bayes_tost_0.02']['p_practical_equivalence']:.3f} & ---\\\\")
    lines.append("\\midrule")
    # flip rate CLT vs bayes
    lines.append("Run-noise flip rate $p$ & "
                 f"{mde['p_disagree']:.4f} & "
                 f"$[{mde['beta_binom_ci94'][0]:.4f},{mde['beta_binom_ci94'][1]:.4f}]$\\\\")
    lines.append("\\quad CLT Normal 94\\% width / Bayes width & "
                 f"{1.0/mde['width_ratio_bayes_over_clt']:.3f} & ---\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    w("tab_var_bayes", "\n".join(lines) + "\n")


mde_grid()
three_way()
run_bootstrap()
var_bayes()
print("done variance tables")
