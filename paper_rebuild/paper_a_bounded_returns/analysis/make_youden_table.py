#!/usr/bin/env python
"""DR-Judge Youden's J table for Paper A (informedness vs prevalence-confounded kappa).

READ-ONLY on canonical_numbers.json['drjudge_youden_j']. Writes tables/tab_youden_paperA.tex.
Reports per-judge overall signed Youden's J (= TPR - FPR vs the adjudicated panel target)
next to Cohen's kappa for continuity. J is the primary statistic because kappa confounds
prevalence with informedness; the gap between J and kappa is the prevalence correction.
Correctly reads the nested judges[name].overall.{youden_j_signed, kappa} schema.
"""
import json, os, warnings
warnings.filterwarnings("ignore")
ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
TAB = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/tables"
os.makedirs(TAB, exist_ok=True)
YJ = json.load(open(f"{ANA}/canonical_numbers.json"))["drjudge_youden_j"]

JLAB = {"DR-Judge-7B": "DR-Judge-7B", "gpt52": "GPT-5.2", "claude_opus": "Claude Opus",
        "claude_sonnet": "Claude Sonnet"}
order = ["DR-Judge-7B", "gpt52", "claude_sonnet", "claude_opus"]

rows = []
for j in order:
    jd = YJ["judges"].get(j)
    if not jd or "overall" not in jd:
        continue
    ov = jd["overall"]
    Jval = ov.get("youden_j_signed", ov.get("youden_j"))
    kap = ov.get("kappa")
    rows.append(f"{JLAB.get(j, j)} & {Jval:+.3f} & {kap:+.3f} \\\\")

lines = [r"\begin{tabular}{lrr}", r"\toprule",
         r"Judge & Youden's $J$ & Cohen's $\kappa$ \\", r"\midrule"]
lines += rows
lines += [r"\bottomrule",
          f"\\multicolumn{{3}}{{l}}{{\\footnotesize Signed $J=\\mathrm{{TPR}}-\\mathrm{{FPR}}$ vs the adjudicated "
          f"panel target ($n={YJ.get('n_cells_drjudge', '')}$ cells).}} \\\\",
          r"\multicolumn{3}{l}{\footnotesize $J$ is primary; $\kappa$ confounds prevalence with informedness.} \\",
          r"\end{tabular}"]
open(f"{TAB}/tab_youden_paperA.tex", "w").write("\n".join(lines) + "\n")
print("wrote tab_youden_paperA.tex; rows:", len(rows),
      "| DR-Judge J=%.3f kappa=%.3f | gpt52 J=%.3f kappa=%.3f"
      % (YJ["judges"]["DR-Judge-7B"]["overall"]["youden_j_signed"],
         YJ["judges"]["DR-Judge-7B"]["overall"]["kappa"],
         YJ["judges"]["gpt52"]["overall"]["youden_j"],
         YJ["judges"]["gpt52"]["overall"]["kappa"]))
