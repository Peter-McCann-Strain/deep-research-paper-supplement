#!/usr/bin/env python
"""LaTeX tables for Paper 1 -- Routability. Reads canonical_numbers.json (read-only).

Produces:
  tab_routability_equivalence.tex  -- TOST of each realizable router's LOOCV headroom
                                      against the +-0.02 Gate-G1 ROPE (GPT-5.2).
  tab_routability_judgerobust.tex  -- multi-judge Gate-G1 robustness (raw oracle vs best
                                      realizable router LOOCV headroom; does G1 fire).
"""
import json, os, warnings
warnings.filterwarnings("ignore")
ROOT = "."
TAB = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/tables"
os.makedirs(TAB, exist_ok=True)
R = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))["routability"]


def w(name, s):
    open(f"{TAB}/{name}.tex", "w").write(s)
    print("wrote", name)


# ---------- Table: TOST equivalence vs the +-0.02 gate (GPT-5.2 Stage B) ----------
# routability.equivalence may be transiently absent from canonical (a concurrent builder
# clobbered/rebuilt the file); emit the table only when the key is live. Never hardcode.
eq = R.get("equivalence")
if eq is not None:
    gate = eq["gate_threshold"]
    RLAB = {"source_router": "Source router", "knn_router_k7": "kNN router ($k{=}7$)",
            "gbm_router": "GBM router", "logreg_router": "Logistic router",
            "candidate_ridge_router": "Restricted ridge (strongest)"}
    order = ["source_router", "knn_router_k7", "gbm_router", "logreg_router", "candidate_ridge_router"]
    sym = eq["symmetric_tost_pm_gate"]
    hl = eq["routers_loocv_headroom"]
    rows = []
    for k in order:
        s = sym[k]
        hd = hl[k]
        ci = s["ci90"]
        equiv = r"\checkmark" if s["equivalent_at_05_alpha"] else r"--"
        star = r"$^{\dagger}$" if k == eq["primary_router"] else ""
        rows.append(f"{RLAB[k]}{star} & {hd:+.4f} & $[{ci[0]:+.4f},\\,{ci[1]:+.4f}]$ & "
                    f"{s['p_tost']:.4f} & {equiv} \\\\")
    tost = (
        "\\begin{tabular}{lrrrc}\n\\toprule\n"
        "Realizable router & LOOCV headroom & 90\\% CI & $p_{\\mathrm{TOST}}$ & Equiv.\\ $\\pm" + f"{gate:.02f}" + "$\\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"
    )
    w("tab_routability_equivalence", tost)
else:
    print("SKIP tab_routability_equivalence: routability.equivalence absent from canonical")

# ---------- Table: multi-judge Gate-G1 robustness ----------
JUD = ["gpt52", "opus", "sonnet", "panel_mean"]
JLAB = {"gpt52": "GPT-5.2", "opus": "Opus", "sonnet": "Sonnet", "panel_mean": "Panel mean"}
ALAB = {"base_p1": "P1", "base_p4": "P4", "base_p5": "P5", "base_p6": "P6"}
pj = R["judge_robustness"]["per_judge"]
fires = R["judge_robustness"]["g1_fires_by_judge"]
rows = []
for j in JUD:
    sa = pj[j]["stage_a"]["raw"]
    sb = pj[j]["stage_b"]
    bf = sa["best_fixed_by_mean"]
    raw = sa["raw_gain_over_best_fixed"]
    best_router = sb["best_router_loocv_headroom"]
    fire = r"\checkmark" if fires[j] else r"$\times$"
    rows.append(f"{JLAB[j]} & {ALAB.get(bf, bf)} & {sb['n_queries']} & {raw:+.4f} & "
                f"{best_router:+.4f} & {fire} \\\\")
jr = (
    "\\begin{tabular}{lrrrrc}\n\\toprule\n"
    "Judge & Best-fixed & $N$ & Raw oracle & Best router (LOOCV) & G1 fires\\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"
)
w("tab_routability_judgerobust", jr)
print("G1 fires on all judges:", all(fires.values()))
