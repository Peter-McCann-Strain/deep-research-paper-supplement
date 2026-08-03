#!/usr/bin/env python
"""Emit tables/tab_b2.tex and tables/tab_bestofn_decoupled.tex from canonical_numbers.json."""
import json
ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
TAB = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/tables"
cn = json.load(open(f"{ANA}/canonical_numbers.json"))

# ---- B2 table ----
b2 = cn["b2_7b_premium"]
rows = [
    ("Single pass (P0 architecture)", b2["p9"]["mean"], b2["p0_gpt4o"]["mean"]),
    ("P1 architecture (iterative RAG)", b2["p1_7b"]["mean"], b2["p1_gpt4o"]["mean"]),
    ("P4 architecture (STORM)", b2["p4_7b"]["mean"], b2["p4_gpt4o"]["mean"]),
]
def fmt_test(t):
    return f"${t['delta']:+.3f}$ [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}], $p{{=}}{t['wilcoxon_p']:.2f}$"
prem = [
    ("P1 premium over single pass", b2["premium_7b_p1_test"], b2["premium_gpt4o_p1_test"]),
    ("P4 premium over single pass", b2["premium_7b_p4_test"], b2["premium_gpt4o_p4_test"]),
]
with open(f"{TAB}/tab_b2.tex", "w") as f:
    f.write("\\begin{tabular}{lcc}\n\\toprule\n")
    f.write("Architecture & Qwen2.5-7B backbone & GPT-4o backbone\\\\\n\\midrule\n")
    for name, a, b in rows:
        f.write(f"{name} & {a:.3f} & {b:.3f}\\\\\n")
    f.write("\\midrule\n")
    for name, a, b in prem:
        f.write(f"{name} & {fmt_test(a)} & {fmt_test(b)}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---- best-of-N decoupled table ----
bo = cn["best_of_n"]
naive = bo["curve"]
pure = bo["pure_noise"]["prediction"]
dec = bo["decoupled"]["curve"]
cl_full = bo["cluster_mean"]
cl_B = bo["decoupled"]["cluster_mean_half_B"]
ks = [1, 2, 3, 4, 5, 7, 9, 12]
with open(f"{TAB}/tab_bestofn_decoupled.tex", "w") as f:
    f.write("\\begin{tabular}{lcccccccc}\n\\toprule\n")
    f.write("$k$ & " + " & ".join(str(k) for k in ks) + "\\\\\n\\midrule\n")
    f.write("Naive best-of-$k$ (same-judge selection) & " +
            " & ".join(f"{naive[str(k)]['best_of_k']:.3f}" for k in ks) + "\\\\\n")
    f.write("Pure-noise prediction ($\\sigma{=}" +
            f"{bo['pure_noise']['sigma_within_query']:.3f}" + "$) & " +
            " & ".join(f"{pure[str(k)]['predicted_best_of_k']:.3f}" for k in ks) + "\\\\\n")
    f.write("Decoupled best-of-$k$ (held-out scoring) & " +
            " & ".join(f"{dec[str(k)]['best_of_k_decoupled']:.3f}" for k in ks) + "\\\\\n")
    # No explicit size command: this row sits inside main.tex's
    # \resizebox{\textwidth}{!}{...} for this table, so an explicit \scriptsize
    # here compounded with that uniform scale-down and rendered below the
    # paper's 9pt floor (found by direct visual inspection of the compiled PDF,
    # adversarial review 2026-07-28, round 28). Matching the surrounding rows'
    # size guarantees it scales identically to the rest of the table instead.
    f.write("\\quad gap to cluster (95\\% CI) & " +
            " & ".join("$[" + f"{dec[str(k)]['gap_ci95'][0]:+.2f}" + "," +
                       f"{dec[str(k)]['gap_ci95'][1]:+.2f}" + "]$"
                       for k in ks) + "\\\\\n")
    f.write("\\midrule\n")
    f.write(f"Cluster reference (full / held-out basis) & \\multicolumn{{8}}{{c}}{{{cl_full:.3f} / {cl_B:.3f}}}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("wrote tab_b2.tex and tab_bestofn_decoupled.tex")
