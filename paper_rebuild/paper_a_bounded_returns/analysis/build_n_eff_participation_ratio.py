#!/usr/bin/env python
"""Spectral participation ratio on the panel's overall-score correlation matrix.

Corroborating check for n_eff.overall.n_eff (criterion-verdict level, N_eff=1.65): computes
the participation ratio PR = (sum lambda_i)^2 / sum(lambda_i^2) of the eigenvalues of the
3x3 Pearson correlation matrix already stored at irr.judge_pearson_cells (overall-score level,
983-cell complete-case panel). Referenced in the prose (main.tex Results and Discussion,
"a spectral participation ratio on overall scores agrees, ~1.4") but was previously a
one-off unpersisted computation; this script lands it under n_eff so the reconciliation
gate can verify it like every other inline statistic.
"""
import json
import os
import numpy as np

ANA = os.path.dirname(os.path.abspath(__file__))
cn = json.load(open(f"{ANA}/canonical_numbers.json"))

judges = ["gpt52", "claude_opus", "claude_sonnet"]
cells = cn["irr"]["judge_pearson_cells"]
R = np.array([[cells[a][b] for b in judges] for a in judges])
eig = np.linalg.eigvalsh(R)
pr = float((eig.sum() ** 2) / (eig ** 2).sum())

out = {
    "judges": judges,
    "source_key": "irr.judge_pearson_cells",
    "eigenvalues": [round(float(x), 4) for x in eig],
    "participation_ratio": round(pr, 4),
    "note": "Corroborating spectral check on overall-score correlations (983-cell complete-case "
            "panel), distinct from n_eff.overall.n_eff (1.6547) which is computed at the "
            "criterion-verdict level over 36,113 crossed cells. The two agree in direction "
            "(~1.4 vs ~1.65) but are not the same statistic and are not expected to match exactly.",
}

DRY_RUN = ("--dry-run" in __import__("sys").argv) or ("--write" not in __import__("sys").argv)
if DRY_RUN:
    print("[dry-run] computed participation_ratio_overall_scores; NOT writing (pass --write to land).")
else:
    cn["n_eff"]["participation_ratio_overall_scores"] = out
    _tmp = f"{ANA}/canonical_numbers.json.tmp"
    open(_tmp, "w").write(json.dumps(cn, indent=1))
    os.replace(_tmp, f"{ANA}/canonical_numbers.json")

print(f"participation ratio (overall scores) = {pr:.4f}  eigenvalues={list(eig)}")
