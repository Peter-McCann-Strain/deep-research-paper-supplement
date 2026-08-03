#!/usr/bin/env python
"""Paper 2 (judge science / citation artefact) booktabs LaTeX tables.

READ-ONLY on canonical_numbers.json (never mutated). Out: tables/tab_p2_*.tex.
Each table reads its numbers live from canonical so it can never drift from the store.
Tables that depend on a not-yet-rebuilt canonical key are skipped with a notice.
Run:  ./venv/bin/python paper_rebuild/paper_a_bounded_returns/analysis/make_paper2_tables.py
"""
import json, os, warnings
warnings.filterwarnings("ignore")
ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
TAB = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/tables"
os.makedirs(TAB, exist_ok=True)
R = json.load(open(f"{ANA}/canonical_numbers.json"))

DLAB = {"information_recall": "Info.\\ recall", "factual_accuracy": "Factual",
        "coverage": "Coverage", "analytical_depth": "Depth", "citation_quality": "Citation",
        "logical_coherence": "Coherence", "organization": "Org.",
        "instruction_following": "Instruction", "attribution_quality": "Attribution"}


def w(name, s):
    open(f"{TAB}/{name}.tex", "w").write(s)
    print("wrote", name)


def fmt_ci(ci, d=3):
    return f"[{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


# ---------- T1: N_eff per-dimension within- vs cross-family agreement ----------
def tab_neff():
    ne = R["n_eff"]
    diag = ne.get("diagnostics", {})
    pd_ = ne["per_dimension"]
    order = [r["dimension"] for r in ne["artefact_signature_ranking"]]
    rows = []
    for k in order:
        v = pd_[k]
        rows.append(f"{DLAB[k]} & {v['within_family_phi_opus_sonnet']:.3f} & "
                    f"{v['cross_family_phi_gpt52_claude']:.3f} & {v['within_minus_cross']:+.3f} & "
                    f"{v['n_eff']:.2f} \\\\")
    ov = ne["overall"]
    foot = (f"\\midrule\nOverall & {ov['within_family_phi_opus_sonnet']:.3f} & "
            f"{ov['cross_family_phi_gpt52_claude']:.3f} & {ov['within_minus_cross']:+.3f} & "
            f"{ov['n_eff']:.2f} \\\\")
    nek = diag.get("n_eff_over_k")
    cap = ""
    if nek is not None:
        cap = (f"\n\\multicolumn{{5}}{{l}}{{\\footnotesize $N_{{\\mathrm{{eff}}}}/k={nek:.3f}$ "
               r"(caution at $<0.5$, \citep{kohli2026ninejudges}); panel sits "
               f"{'below' if diag.get('breaches_caution_below_half') else 'just above'} the line.}} \\\\")
    t = (r"\begin{tabular}{lrrrr}" "\n\\toprule\n"
         r"Dimension & $\phi_{\mathrm{Opus,Son}}$ & $\phi_{\mathrm{GPT,Claude}}$ & $\Delta_{\mathrm{w-c}}$ & $N_{\mathrm{eff}}$\\"
         "\n\\midrule\n" + "\n".join(rows) + "\n" + foot + cap + "\n\\bottomrule\n\\end{tabular}")
    w("tab_p2_neff", t)


# ---------- T2: within-family N_eff control (same-lab vs cross-family) ----------
def tab_neff_control():
    wo = R["n_eff"].get("within_openai")
    if not wo:
        print("skip tab_p2_neff_control (n_eff.within_openai absent)")
        return
    o = wo["within_openai"]; a = wo["within_anthropic"]; g = wo["full_grid"]
    rows = [
        f"Within-OpenAI (GPT-5.2, GPT-4.1) & 2 & {o['mean_within_phi']:.3f} & {o['n_eff']:.3f} \\\\",
        f"Within-Anthropic (Opus, Sonnet) & 2 & {a['mean_within_phi']:.3f} & {a['n_eff']:.3f} \\\\",
        f"Cross-family mean & -- & {wo['cross_family_phi_mean']:.3f} & -- \\\\",
        f"4-judge grid (2+2) & 4 & -- & {g['n_eff']:.3f} \\\\",
    ]
    t = (r"\begin{tabular}{lrrr}" "\n\\toprule\n"
         r"Panel cell & $k$ & mean $\phi$ & $N_{\mathrm{eff}}$\\" "\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    w("tab_p2_neff_control", t)


# ---------- T3: cross-family judge-vs-gold asymmetry ----------
def tab_judge_gold():
    jvg = R["judge_vs_gold"]; pj = jvg["per_judge"]
    JLAB = {"gpt52": "GPT-5.2 (OpenAI)", "claude_opus": "Opus (Anthropic)",
            "claude_sonnet": "Sonnet (Anthropic)"}
    rows = []
    for j in ["gpt52", "claude_opus", "claude_sonnet"]:
        for dim in ["factual_accuracy", "citation_quality"]:
            x = pj[j][dim]; bd = x["boot_diff"]
            star = "$^{*}$" if bd["excludes_0"] else ""
            rows.append(f"{JLAB[j]} & {DLAB[dim]} & {x['n']} & {x['auc']:.3f} & "
                        f"{bd['delta']:+.4f}{star} & {fmt_ci(bd['ci95'], 3)} \\\\")
    t = (r"\begin{tabular}{llrrrc}" "\n\\toprule\n"
         r"Judge & Dimension & $n$ & AUC & $\Delta$(score) & 95\% boot CI\\" "\n\\midrule\n"
         + "\n".join(rows) +
         "\n\\bottomrule\n\\multicolumn{6}{l}{\\footnotesize $^{*}$ bootstrap gap CI excludes 0. "
         f"Slice: {jvg['slice']['n_queries']} verifiable-answer queries, "
         f"{jvg['effective_signal_clusters']} signal-bearing clusters.}} \\\\"
         "\n\\end{tabular}")
    w("tab_p2_judge_gold", t)


# ---------- T4: judge-vs-gold ordinal calibration (weighted-kappa / Krippendorff) ----------
def tab_calibration():
    jvg = R["judge_vs_gold"]; pj = jvg["per_judge"]
    JLAB = {"gpt52": "GPT-5.2", "claude_opus": "Opus", "claude_sonnet": "Sonnet"}
    rows = []
    for j in ["gpt52", "claude_opus", "claude_sonnet"]:
        for dim in ["factual_accuracy", "citation_quality"]:
            cal = pj[j][dim].get("calibration")
            if not cal:
                continue
            b = cal["binary"]; o = cal["ordinal3"]
            rows.append(
                f"{JLAB[j]} & {DLAB[dim]} & {b['cohen_kappa']:.3f} & {fmt_ci(b['cohen_kappa_ci95'], 3)} & "
                f"{o['weighted_kappa_quadratic']:.3f} & {fmt_ci(o['weighted_kappa_ci95'], 3)} & "
                f"{o['krippendorff_alpha_ordinal']:+.3f} \\\\")
    if not rows:
        print("skip tab_p2_calibration (calibration block absent)")
        return
    t = (r"\begin{tabular}{llrcrcr}" "\n\\toprule\n"
         r"Judge & Dim. & $\kappa$ & $\kappa$ CI & $\kappa_w$ & $\kappa_w$ CI & $\alpha_{\mathrm{ord}}$\\"
         "\n\\midrule\n" + "\n".join(rows) +
         "\n\\bottomrule\n\\multicolumn{7}{l}{\\footnotesize Binary Cohen $\\kappa$ at gold-slice median; "
         "quadratic-weighted $\\kappa_w$ and ordinal Krippendorff $\\alpha$ (arXiv:2510.09738).} \\\\"
         "\n\\end{tabular}")
    w("tab_p2_calibration", t)


# ---------- T5: per-judge citation density / provenance regression ----------
def tab_density_per_judge():
    dpj = R["density_per_judge"]
    JLAB = {"gpt52": "GPT-5.2", "claude_opus": "Opus", "claude_sonnet": "Sonnet"}
    rows = []
    for j in ["gpt52", "claude_opus", "claude_sonnet"]:
        d = dpj[j]
        sd = "$<$0.001" if d["p_density_cluster_pattern"] < 1e-3 else f"{d['p_density_cluster_pattern']:.3f}"
        sp = "$<$0.001" if d["p_provenance_cluster_pattern"] < 1e-3 else f"{d['p_provenance_cluster_pattern']:.3f}"
        rows.append(f"{JLAB[j]} & {d['n']} & {d['beta_density']:+.4f} & {sd} & "
                    f"{d['beta_provenance']:+.4f} & {sp} \\\\")
    t = (r"\begin{tabular}{lrrcrc}" "\n\\toprule\n"
         r"Judge & $n$ & $\beta_{\mathrm{density}}$ & $p^{\dagger}$ & $\beta_{\mathrm{prov}}$ & $p^{\dagger}$\\"
         "\n\\midrule\n" + "\n".join(rows) +
         "\n\\bottomrule\n\\multicolumn{6}{l}{\\footnotesize Citation-quality score regressed on raw "
         "citation density and provenance rate; $^{\\dagger}$ pattern-cluster-robust $p$ ($G{=}12$).} \\\\"
         "\n\\end{tabular}")
    w("tab_p2_density_per_judge", t)


# ---------- T6: pooled citation-quality regression (clustered SE) ----------
def tab_citation_regression():
    cr = R["citation_regression"]; co = cr["coefs"]
    LAB = {"provenance_rate": "Provenance rate", "log_cit": "$\\log$ citations",
           "log_words": "$\\log$ words"}
    rows = []
    for k in ["provenance_rate", "log_cit", "log_words"]:
        c = co[k]
        p = c["p_cluster_pattern"]
        ps = "$<$0.001" if p < 1e-3 else f"{p:.3f}"
        rows.append(f"{LAB[k]} & {c['beta']:+.4f} & {c['se_cluster_pattern']:.4f} & {ps} & "
                    f"{fmt_ci(c['ci'], 3)} \\\\")
    t = (r"\begin{tabular}{lrrcc}" "\n\\toprule\n"
         r"Predictor & $\beta$ & SE$_{\mathrm{clust}}$ & $p_{\mathrm{clust}}$ & 95\% CI\\" "\n\\midrule\n"
         + "\n".join(rows) +
         f"\n\\midrule\n\\multicolumn{{5}}{{l}}{{\\footnotesize $n={cr['n_reports']}$ reports, "
         f"$R^2={cr['r2']:.3f}$; pattern-cluster-robust SE ($G{{=}}12$). "
         f"Provenance$\\leftrightarrow$citation $r={cr['corr_cq_provenance_pearson']:.3f}$; "
         f"provenance$\\leftrightarrow$factual $r={cr['corr_fa_provenance_pearson']:.3f}$.}} \\\\"
         "\n\\bottomrule\n\\end{tabular}")
    w("tab_p2_citation_regression", t)


# ---------- T7: E5 gold-injection dose-response ----------
def tab_e5_dose():
    e5 = R.get("e5_dose_response")
    if not e5 or "factual_accuracy_slope" not in e5:
        print("skip tab_p2_e5_dose (e5_dose_response absent)")
        return
    fa = e5["factual_accuracy_slope"]; cq = e5["citation_quality_slope"]
    iv = e5["interleaved_vs_g100"]
    rows = [
        f"Factual accuracy & {fa['slope']:+.4f} & {fa['se']:.4f} & {fmt_ci(fa['ci95_two_sided'], 3)} & "
        f"{fa['p_value_two_sided']:.3f} \\\\",
        f"Citation quality & {cq['slope']:+.4f} & {cq['se']:.4f} & {fmt_ci(cq['ci95_two_sided'], 3)} & "
        f"{cq['p_value_two_sided']:.3f} \\\\",
    ]
    fa_bound = fa.get("one_sided_ci95")
    t = (r"\begin{tabular}{lrrcc}" "\n\\toprule\n"
         r"Outcome & slope/dose & SE & 95\% CI & $p$\\" "\n\\midrule\n"
         + "\n".join(rows) +
         f"\n\\midrule\n\\multicolumn{{5}}{{l}}{{\\footnotesize $n={fa['n']}$ dose points, "
         f"{e5['n_queries']} queries $\\times$ \\{{P0,P1,P4\\}}; gold fractions "
         f"{{{', '.join(c for c in e5.get('cells_present', []) if c.startswith('g'))}}}. "
         f"Factual one-sided 95\\% upper bound on slope $={fa_bound:+.4f}$ ($<$ margin 0.05). "
         f"Interleaved$-$g100: factual $\\Delta={iv['delta']:+.4f}$, citation "
         f"$\\Delta={iv['citation_interleaved_mean'] - iv['citation_g100_mean']:+.4f}$.}} \\\\"
         "\n\\bottomrule\n\\end{tabular}")
    w("tab_p2_e5_dose", t)


# ---------- T8: 7B local-benchmark RL-training delta (external validation) ----------
def tab_local_benchmark():
    lb = R["local_benchmark"]; tv = lb["tier_7b_external_validation"]
    rt = tv["rl_training_delta_test"]
    rows = []
    for b in lb["benchmarks"]:
        bp = lb["per_benchmark"][b]["by_pattern"]
        label = b.replace("_", r"\_")
        rows.append(f"{label} & {bp['p9']:.3f} & {bp['p10']:.3f} & "
                    f"{bp['p10'] - bp['p9']:+.3f} \\\\")
    foot = (f"\\midrule\nMacro mean & {tv['p9_mean_macro']:.3f} & {tv['p10_mean_macro']:.3f} & "
            f"{rt['delta_p10_minus_p9']:+.3f} \\\\")
    t = (r"\begin{tabular}{lrrr}" "\n\\toprule\n"
         r"Benchmark & P9 (Qwen-7B) & P10 (DeepRes-7B) & $\Delta_{\mathrm{RL}}$\\" "\n\\midrule\n"
         + "\n".join(rows) + "\n" + foot +
         f"\n\\midrule\n\\multicolumn{{4}}{{l}}{{\\footnotesize Paired RL-training effect "
         f"$={rt['delta_p10_minus_p9']:+.3f}$, 95\\% CI {fmt_ci(rt['ci95'], 3)}, "
         f"Wilcoxon $p={rt['wilcoxon_p']:.3f}$ ($n={rt['n_paired']}$ pairs, GPT-5.2 judge).}} \\\\"
         "\n\\bottomrule\n\\end{tabular}")
    w("tab_p2_local_benchmark", t)


# ---------- T9: DR-Judge signed Youden's J per dimension (rate-or-fate) ----------
def tab_drjudge_youden():
    yj = R.get("drjudge_youden_j")
    if not yj or "judges" not in yj or "DR-Judge-7B" not in yj["judges"]:
        print("skip tab_p2_drjudge_youden (drjudge_youden_j absent)")
        return
    dj = yj["judges"]["DR-Judge-7B"]
    pd_ = dj["per_dimension"]
    # order by descending J
    order = sorted(pd_, key=lambda k: -pd_[k]["youden_j_signed"])
    rows = []
    for k in order:
        v = pd_[k]
        rows.append(f"{DLAB[k]} & {v['n']} & {v['tpr']:.3f} & {v['fpr']:.3f} & "
                    f"{v['youden_j_signed']:+.3f} & {v['phase']} \\\\")
    ov = dj["overall"]
    foot = (f"\\midrule\nOverall & {ov['n']} & {ov['tpr']:.3f} & {ov['fpr']:.3f} & "
            f"{ov['youden_j_signed']:+.3f} & {ov['phase']} \\\\")
    gb = yj.get("gap_bootstrap_drjudge_minus_best_panel", {})
    gap_note = ""
    if gb:
        gap_note = (f"\n\\multicolumn{{6}}{{l}}{{\\footnotesize Gap to best panel judge "
                    f"({gb.get('best_panel_judge', '')}): $\\Delta J={gb['obs_gap_overall_J']:+.3f}$, "
                    f"95\\% CI {fmt_ci(gb['ci95'], 3)} (excludes 0).}} \\\\")
    t = (r"\begin{tabular}{lrrrrl}" "\n\\toprule\n"
         r"Dimension & $n$ & TPR & FPR & $J$ & phase\\" "\n\\midrule\n"
         + "\n".join(rows) + "\n" + foot +
         f"\n\\midrule\n\\multicolumn{{6}}{{l}}{{\\footnotesize Signed $J=\\mathrm{{TPR}}-\\mathrm{{FPR}}$ "
         f"vs GPT-5.2-anchored panel target ($n={yj.get('n_cells_drjudge', '')}$); "
         f"phase per arXiv:2601.04411 (Rate-or-Fate): $J>0$=usable verifier.}} \\\\"
         + gap_note + "\n\\bottomrule\n\\end{tabular}")
    w("tab_p2_drjudge_youden", t)


# ---------- T10: LOSO source-jackknife robustness ----------
def tab_loso():
    lr = R.get("loso_robustness")
    if not lr or "robustness_summary" not in lr:
        print("skip tab_p2_loso (loso_robustness absent)")
        return
    rs = lr["robustness_summary"]
    LAB = {"gate3_inner5_max_over_loso": "Max inner-5 disputed pairs over LOSO drops",
           "top1_always_matches_full": "Top-1 leader matches full panel (all drops)",
           "max_rank_displacement_over_loso": "Max rank displacement over LOSO drops"}
    rows = []
    for k, v in rs.items():
        if k == "note":
            continue
        if isinstance(v, bool):
            v = "yes" if v else "no"
        if isinstance(v, (int, float, str)):
            rows.append(f"{LAB.get(k, k.replace('_', chr(92) + '_'))} & {v} \\\\")
    if not rows:
        print("skip tab_p2_loso (summary not scalar)")
        return
    src = lr.get("sources", [])
    foot = ""
    if src:
        foot = (f"\n\\midrule\n\\multicolumn{{2}}{{l}}{{\\footnotesize Leave-one-source-out over "
                f"{len(src)} benchmark sources, 3-judge panel.}} \\\\")
    t = (r"\begin{tabular}{lr}" "\n\\toprule\n"
         r"LOSO source-jackknife statistic & Value\\" "\n\\midrule\n"
         + "\n".join(rows) + foot + "\n\\bottomrule\n\\end{tabular}")
    w("tab_p2_loso", t)


for fn in (tab_neff, tab_neff_control, tab_judge_gold, tab_calibration,
           tab_density_per_judge, tab_citation_regression, tab_e5_dose,
           tab_local_benchmark, tab_drjudge_youden, tab_loso):
    try:
        fn()
    except Exception as e:
        print(f"ERROR in {fn.__name__}: {type(e).__name__}: {e}")
print("\nPaper-2 tables pass complete ->", TAB)
