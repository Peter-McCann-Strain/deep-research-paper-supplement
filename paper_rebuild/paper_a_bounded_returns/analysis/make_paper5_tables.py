#!/usr/bin/env python
"""Paper-5 (Dose ceiling) LaTeX tables, all read live from canonical_numbers.json.
Out: paper_rebuild/paper_a_bounded_returns/tables/tab_e5_equivalence.tex
                                            tab_e5_gold_consumption.tex
                                            tab_rxu.tex
                                            tab_citation_faithfulness.tex
Run: ./venv/bin/python paper_rebuild/paper_a_bounded_returns/analysis/make_paper5_tables.py
Mirrors make_tables.py: raw booktabs `tabular`, no float wrapper, \\input-ed by main.tex.
DOES NOT mutate canonical_numbers.json.
"""
import json, os
ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
TAB = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/tables"
os.makedirs(TAB, exist_ok=True)
C = json.load(open(f"{ANA}/canonical_numbers.json"))


def w(name, s):
    open(f"{TAB}/{name}.tex", "w").write(s)
    print("wrote", name)


PRETTY = {"base_p0": "P0 Single-pass", "base_p1": "P1 Iterative RAG",
          "base_p4": "P4 STORM", "base_p5": "P5 Hier.\\ W\\&D",
          "base_p7": "P7 Graph", "base_p8": "P8 Beam",
          "base_p9": "P9 Qwen2.5-7B", "base_p10": "P10 DeepResearcher",
          "base_p11": "P11 ReAct",
          "p0": "P0 Single-pass", "p1": "P1 Iterative RAG", "p4": "P4 STORM",
          "p5": "P5 Hier.\\ W\\&D", "p8": "P8 Beam", "p9": "P9 Qwen2.5-7B",
          "p10": "P10 DeepResearcher", "p11": "P11 ReAct"}

# ---------------- 1. E5 equivalence (TOST) ----------------
EQ = C["e5_equivalence"]
rows = [("Factual flat (g100 $-$ g000)", EQ["factual_flat"]),
        ("Interleaving null (interl. $-$ g100)", EQ["interleaving_null"])]
lines = [r"\begin{tabular}{lrrrrc}", r"\toprule",
         r"Contrast & $n$ & $\Delta$ & 90\% CI & $p_{\mathrm{TOST}}$ & Equiv.\\",
         r"\midrule"]
for lab, blk in rows:
    t = blk["tost"]
    ci = t["ci90_inside_bound"]
    eq = r"\checkmark" if t["equivalent_at_05_alpha"] else r"$\times$"
    lines.append(
        f"{lab} & {t['n']} & ${t['mean_diff']:+.4f}$ & "
        f"$[{ci[0]:+.4f},\\,{ci[1]:+.4f}]$ & ${t['p_tost']:.3f}$ & {eq}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
bound = EQ["factual_flat"]["tost"]["bound"]
anchor = EQ["e2_mde80_anchor"]
ratio = EQ["factual_flat"]["margin_vs_e2_mde80"]
hdr = (f"% E5 equivalence (TOST), margin $\\pm{bound:.2f}$ "
       f"($\\approx{ratio:.1f}\\times$ E2 MDE80 $={anchor:.4f}$); pooled P0/P1/P4, GPT-5.2; "
       f"from canonical e5_equivalence.\n")
w("tab_e5_equivalence", hdr + "\n".join(lines) + "\n")

# ---------------- 2. E5 gold consumption ----------------
GC = C["e5_gold_consumption"]
pool = GC["pooled_over_architectures"]
order = ["g000", "g025", "g050", "g075", "g100", "interleaved"]
lab = {"g000": "0.00", "g025": "0.25", "g050": "0.50", "g075": "0.75",
       "g100": "1.00", "interleaved": "interl."}
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         (r"Gold frac. & Cites emit. & Gold avail. & Gold cited & "
          r"Resol.\ rate & Lex.\ recall\\"),
         r"\midrule"]
for c in order:
    v = pool[c]
    rr = v["gold_resolution_rate_mean"]
    lr = v["gold_content_lexical_recall_mean"]
    rr_s = "--" if rr is None else f"{rr:.2f}"
    lr_s = "--" if lr is None else f"{lr:.3f}"
    lines.append(
        f"{lab[c]} & {v['citations_emitted_total']} & {v['gold_available_total']} & "
        f"{v['gold_cited_total']} & {rr_s} & {lr_s}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
fs = GC["carried_from_e5_dose_response"]["factual_accuracy_slope"]
cs = GC["carried_from_e5_dose_response"]["citation_quality_slope"]
hdr = (f"% E5 gold-consumption manipulation check (q3, pooled P0/P1/P4, $0 CPU regex/URL join). "
       f"Gold cited tracks gold available $\\approx$1:1, content enters the body (lex.\\ recall "
       f"$\\sim$0.4), yet the carried judge factual slope is flat "
       f"(${fs:+.4f}$ factual, ${cs:+.4f}$ citation). From canonical e5_gold_consumption.\n")
w("tab_e5_gold_consumption", hdr + "\n".join(lines) + "\n")

# ---------------- 3. R x U|R conditional utilisation ----------------
RX = C["oracle"]["rxu_conditional"]
pp = RX["per_pattern"]
# sort by U|R ascending so the utilisation ceiling reads top-down
items = sorted(pp.items(), key=lambda kv: kv[1]["mean_U_given_R"])
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         (r"Pattern & $n_{\mathrm{claims}}$ & $\overline{R}$ & $\overline{U\mid R}$ & "
          r"$\overline{U\mid R}$ 95\% CI & Support\\"),
         r"\midrule"]
for key, v in items:
    ci = v["u_given_r_ci95"]
    lines.append(
        f"{PRETTY.get(key, key)} & {v['n_claims']} & {v['mean_R']:.3f} & "
        f"{v['mean_U_given_R']:.3f} & $[{ci[0]:.3f},\\,{ci[1]:.3f}]$ & "
        f"{v['support_rate']:.3f}\\\\")
cms = RX["cluster_mean_support_rate"]; cci = RX["cluster_mean_support_rate_ci95"]
lines.append(r"\midrule")
lines.append(
    f"Cluster mean support & & & & & {cms:.3f} $[{cci[0]:.3f},\\,{cci[1]:.3f}]$\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
ncov = RX["coverage"]["n_patterns_covered"]
hdr = (f"% R x U|R conditional utilisation (DeepResearch-Slice, arXiv:2601.03261), "
       f"{ncov} patterns with saved C0 NLI snapshots. Support = $\\overline{{R}}\\cdot\\overline{{U|R}}$. "
       f"High $\\overline{{R}}$ with low $\\overline{{U|R}}$ = utilisation-bound. "
       f"From canonical oracle.rxu_conditional.\n")
w("tab_rxu", hdr + "\n".join(lines) + "\n")

# ---------------- 4. Citation faithfulness (LOSO proxy lower bound) ----------------
CF = C["citation_faithfulness"]
strict = CF["strict_loso"]["status"]
pl = CF["proxy_lowerbound"]
per = pl["per_pattern"]
items = sorted(per.items(), key=lambda kv: kv[1]["post_rationalisation_rate_lowerbound"])
lines = [r"\begin{tabular}{lrrr}", r"\toprule",
         (r"Pattern & $n_{\mathrm{cited}}$ & Post-rat.\ rate (LB) & "
          r"Faithful (UB)\\"),
         r"\midrule"]
for key, v in items:
    lines.append(
        f"{PRETTY.get(key, key)} & {v['n_cited']} & "
        f"{v['post_rationalisation_rate_lowerbound']:.3f} & "
        f"{v['faithfulness_rate_upperbound']:.3f}\\\\")
lines.append(r"\midrule")
lines.append(
    f"Pooled & {pl['n_cited_total']} & "
    f"{pl['post_rationalisation_rate_lowerbound_overall']:.3f} & "
    f"{1 - pl['post_rationalisation_rate_lowerbound_overall']:.3f}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
hdr = (f"% Citation faithfulness via LOSO entailment replay (cf.\\ arXiv:2412.18004, "
       f"$\\sim$57\\% post-hoc). Strict LOSO is {strict} on disk; reported here is the "
       f"on-disk own-chunk non-entailment PROXY: a LOWER BOUND on post-rationalisation "
       f"(UPPER BOUND on faithfulness), NOT the headline metric. "
       f"From canonical citation_faithfulness.proxy_lowerbound.\n")
w("tab_citation_faithfulness", hdr + "\n".join(lines) + "\n")

print("done: 4 Paper-5 tables")
