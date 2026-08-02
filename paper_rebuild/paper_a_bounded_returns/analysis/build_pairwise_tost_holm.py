#!/usr/bin/env python
"""Holm multiplicity correction for the six-pattern-cluster TOST family + Wilcoxon-TOST
at the +/-0.02 margin.

The store's pairwise_verified reports 15 pairwise TOST equivalence declarations on the
six-pattern cluster {P1,P4,P5,P6,P7,P8} at +/-0.05 (9/15 Wilcoxon-TOST, 6/15 t-TOST)
UNCORRECTED for multiplicity, and only the t-TOST count at +/-0.02 (0/15), although the
paper claims 0/15 'under both' tests at +/-0.02. This builder:
  1. Recomputes the per-pair TOST p-values (p_tost = max of the two one-sided p-values;
     identical machinery/data prep to build_pairwise.py: 3-judge mean of ovc, sonnet ->
     overall_score_recomputed) for BOTH the t-TOST and Wilcoxon-TOST at +/-0.05 and
     +/-0.02.
  2. Applies Holm step-down across the 15 equivalence declarations (the declarations are
     the discoveries) per method x margin, and counts survivors.
  3. Lands pairwise_verified['tost6_holm'] (both margins, both methods, per-pair p and
     Holm-adjusted p) and pairwise_verified['tost6_w_pm02'] (the Wilcoxon count at
     +/-0.02 so the paper's 'under both' is backed by the store).
APPEND-ONLY: adds two NEW subkeys inside pairwise_verified, refuses to overwrite,
atomic tempfile+replace, never touches sibling keys. Deterministic (no randomness).

Usage: python build_pairwise_tost_holm.py [--write] [--force]
"""
import itertools
import json
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
CANON = f"{ANA}/canonical_numbers.json"

ov = pd.read_parquet(f"{ROOT}/data/analysis/df_overall_scores.parquet")
ov["ovc"] = ov["overall_score"].where(~ov.judge.eq("claude_sonnet"),
                                      ov.get("overall_score_recomputed"))
base = ov[ov.pattern.str.match(r"^base_p\d+$")
          & ov.judge.isin(["gpt52", "claude_opus", "claude_sonnet"])]
avg = base.groupby(["pattern", "query_id"], observed=True)["ovc"].mean().unstack(0)

C6 = ["base_p1", "base_p4", "base_p5", "base_p6", "base_p7", "base_p8"]
PAIRS = list(itertools.combinations(C6, 2))


def tost_t_p(a, b, m):
    d = avg[[a, b]].dropna()
    d = (d[a] - d[b]).values
    n = len(d)
    mu = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    p_lower = 1 - tdist.cdf((mu + m) / se, n - 1)   # H0: mu <= -m
    p_upper = tdist.cdf((mu - m) / se, n - 1)       # H0: mu >= +m
    return float(max(p_lower, p_upper)), n


def tost_w_p(a, b, m):
    d = avg[[a, b]].dropna()
    d = (d[a] - d[b]).values
    try:
        p_lower = wilcoxon(d + m, alternative="greater").pvalue
        p_upper = wilcoxon(d - m, alternative="less").pvalue
        return float(max(p_lower, p_upper)), len(d)
    except Exception:
        return 1.0, len(d)


def holm_adjust(pv):
    pv = np.asarray(pv, float)
    idx = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(idx):
        run = max(run, (m - r) * pv[i])
        adj[i] = min(run, 1.0)
    return adj


def family(method_fn, margin):
    ps, ns = [], []
    for a, b in PAIRS:
        p, n = method_fn(a, b, margin)
        ps.append(p)
        ns.append(n)
    adj = holm_adjust(ps)
    per_pair = {f"{a.replace('base_','')}|{b.replace('base_','')}": {
                    "n": int(ns[i]),
                    "p_tost": round(ps[i], 4),
                    "p_holm": round(float(adj[i]), 4),
                    "equiv_raw": bool(ps[i] < 0.05),
                    "equiv_holm": bool(adj[i] < 0.05)}
                for i, (a, b) in enumerate(PAIRS)}
    return {"n_pairs": len(PAIRS),
            "n_equiv_raw": int(sum(1 for p in ps if p < 0.05)),
            "n_equiv_holm": int(sum(1 for p in adj if p < 0.05)),
            "per_pair": per_pair}


t05 = family(tost_t_p, 0.05)
w05 = family(tost_w_p, 0.05)
t02 = family(tost_t_p, 0.02)
w02 = family(tost_w_p, 0.02)

# ---- consistency checks vs the landed raw counts ----
cn = json.load(open(CANON))
pw = cn["pairwise_verified"]
checks = {
    "tost6_t_pm05": (pw.get("tost6_t_pm05"), t05["n_equiv_raw"]),
    "tost6_wilcoxon_pm05": (pw.get("tost6_wilcoxon_pm05"), w05["n_equiv_raw"]),
    "tost6_t_pm02": (pw.get("tost6_t_pm02"), t02["n_equiv_raw"]),
}
for k, (landed, mine) in checks.items():
    tag = "OK" if landed == mine else "MISMATCH"
    print(f"[{tag}] {k}: landed={landed} recomputed={mine}")

tost6_holm = {
    "_note": ("Holm step-down across the 15 pairwise TOST equivalence declarations of the "
              "six-pattern cluster (equivalence declarations = discoveries; family per "
              "method x margin). p_tost = max of the two one-sided TOST p-values; data prep "
              "identical to build_pairwise.py (3-judge mean, sonnet recomputed). Deterministic."),
    "family_size": 15,
    "alpha": 0.05,
    "wilcoxon_pm05": w05,
    "t_pm05": t05,
    "wilcoxon_pm02": {k: w02[k] for k in ("n_pairs", "n_equiv_raw", "n_equiv_holm")},
    "t_pm02": {k: t02[k] for k in ("n_pairs", "n_equiv_raw", "n_equiv_holm")},
    "summary": (f"+/-0.05: Wilcoxon-TOST {w05['n_equiv_raw']}/15 raw -> "
                f"{w05['n_equiv_holm']}/15 Holm; t-TOST {t05['n_equiv_raw']}/15 raw -> "
                f"{t05['n_equiv_holm']}/15 Holm. +/-0.02: 0/15 under both, raw and Holm "
                f"(Wilcoxon raw={w02['n_equiv_raw']}, t raw={t02['n_equiv_raw']})."),
}

tost6_w_pm02 = {
    "_note": ("Wilcoxon-TOST at the +/-0.02 margin (previously only the t-TOST count was "
              "landed although the paper says 0/15 'under both'). Same machinery as "
              "tost6_wilcoxon_pm05."),
    "n_equiv_raw": w02["n_equiv_raw"],
    "n_equiv_holm": w02["n_equiv_holm"],
    "min_p_tost": round(min(v["p_tost"] for v in w02["per_pair"].values()), 4),
    "per_pair": w02["per_pair"],
}

print()
print(f"+/-0.05 Wilcoxon: raw {w05['n_equiv_raw']}/15 -> Holm {w05['n_equiv_holm']}/15")
print(f"+/-0.05 t-TOST : raw {t05['n_equiv_raw']}/15 -> Holm {t05['n_equiv_holm']}/15")
print(f"+/-0.02 Wilcoxon: raw {w02['n_equiv_raw']}/15 -> Holm {w02['n_equiv_holm']}/15 "
      f"(min p_tost={tost6_w_pm02['min_p_tost']})")
print(f"+/-0.02 t-TOST : raw {t02['n_equiv_raw']}/15 -> Holm {t02['n_equiv_holm']}/15")
print("\nper-pair (Wilcoxon +/-0.05):")
for k, v in sorted(w05["per_pair"].items(), key=lambda kv: kv[1]["p_tost"]):
    print(f"  {k:10s} p={v['p_tost']:.4f} holm={v['p_holm']:.4f} "
          f"raw={'Y' if v['equiv_raw'] else '.'} holm_sig={'Y' if v['equiv_holm'] else '.'}")

if "--write" in sys.argv:
    cn = json.load(open(CANON))  # fresh read: keep the read-modify-write window short
    pw = cn["pairwise_verified"]
    if ("tost6_holm" in pw or "tost6_w_pm02" in pw) and "--force" not in sys.argv:
        print("[REFUSING] tost6_holm/tost6_w_pm02 already in pairwise_verified (use --force)")
        sys.exit(1)
    pw["tost6_holm"] = tost6_holm
    pw["tost6_w_pm02"] = tost6_w_pm02
    fd, tmp = tempfile.mkstemp(dir=ANA, prefix="canonical_numbers.", suffix=".json.tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(cn, f, indent=1)
    os.replace(tmp, CANON)
    print(f"[WROTE pairwise_verified.tost6_holm + pairwise_verified.tost6_w_pm02 "
          f"(store {len(cn)} top-level keys preserved)]")
else:
    print("[DRY-RUN: no write; pass --write to land the keys]")
