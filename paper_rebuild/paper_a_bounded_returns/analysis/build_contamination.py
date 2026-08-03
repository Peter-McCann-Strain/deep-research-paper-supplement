#!/usr/bin/env python
"""E6 STC-AUDIT — STEP 4: fit the contamination RATE regression, write canonical key `contamination`.

Part of E6 (prereg docs/publication/prereg/prereg_E6.md). Robustness appendix, NOT a headline
(2606.05241 owns the framing). Modelled on build_judge_vs_gold.py (atomic tmp+os.replace
write into canonical_numbers.json; seeded generator on SORTED inputs per the set-hash-drift
lesson).

STEP-0 COVERAGE DECISION (authoritative basis, recorded here per the prereg requirement)
----------------------------------------------------------------------------------------
Prereg requires snippet classification across ALL 11 architectures, but full pre-citation
snippet text (search.json extractions) is on disk only for P0/P1/P9/P12. Resolution — a
DUAL-BASIS design:

  PRIMARY  (basis='citation') : df_citations cited-URL/domain is the ONLY uniform 11/12-
            architecture logged-snippet signal (995 pattern x query cells). The per-snippet
            rate regression P(contaminated) ~ search_count + architecture is fit on THIS
            basis. It is the CITED subset, so it UNDER-COUNTS retrieved-but-uncited
            contaminated snippets — a conservative bound on the leakage rate, stated as such.
  SENSITIVITY (basis='search'): the P0/P1/P9/P12 full-snippet classifier pass is reported as
            a higher-recall robustness check, never as the headline coefficient.

The architecture coefficient in the PRIMARY rate regression is the primary endpoint, together
with the four headline effects recomputed on the decontaminated query set (wired by
rebuild_all.sh, STEP 5). The human picks/confirms the authoritative basis before the paid
classifier pass; this script honours --basis.

Inputs (READ-ONLY): results/contamination_e6/{telemetry.parquet, snippets_<basis>.parquet,
regex_flags_<basis>.parquet (STEP 2), classifier/labels_<basis>.json (STEP 3, human-supplied)}.
The contamination signal per snippet = regex_contaminated OR classifier-contaminated (union),
so the regex gate alone yields a usable PRIMARY result even before the GPT-4o pass exists.

GRACEFUL DEGRADATION (build/verify safety): if the classifier output is ABSENT or a STUB
(empty labels), the script still imports and runs end-to-end on whatever signal exists
(regex-only, or nothing), prints a plan, and — unless --finalize is passed AND a real
classifier pass is present — writes to a SIDE-CAR file (contamination_provisional.json under
the E6 dir) rather than mutating canonical_numbers.json. The real `contamination` canonical
key is written ONLY by an explicit --finalize run after the human supplies the classifier pass.

Usage:
    [ -f venv/bin/activate ] && source venv/bin/activate
    python paper_rebuild/paper_a_bounded_returns/analysis/build_contamination.py --dry-run     # plan only
    python paper_rebuild/paper_a_bounded_returns/analysis/build_contamination.py               # provisional side-car
    python paper_rebuild/paper_a_bounded_returns/analysis/build_contamination.py --finalize    # write canonical key (human, later)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(".")
ANA = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
E6 = ROOT / "results" / "contamination_e6"
CANONICAL = ANA / "canonical_numbers.json"
SEED = 20260611  # E6 prereg date; seeded generator on SORTED inputs -> deterministic.

PROTECTED_DATA = ROOT / "data" / "analysis"  # never written


def _load_classifier_labels(basis: str) -> Dict:
    p = E6 / "classifier" / f"labels_{basis}.json"
    if not p.exists():
        return {"present": False, "n": 0, "labels": []}
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {"present": False, "n": 0, "labels": []}
    labels = d.get("labels", [])
    return {"present": bool(labels), "n": len(labels), "labels": labels, "meta": d}


def _cluster_bootstrap_ci(df, coef_fn, n_boot=500, rng=None):
    """Resample QUERIES (clusters) with replacement; refit; percentile CI on a coefficient.
    Seeded generator on the SORTED cluster list -> deterministic (set-hash-drift lesson).

    n_boot is configurable (--n-boot): each draw refits the logit on the full snippet table,
    so on the ~21k-row citation basis this dominates runtime. Default 500 (precise enough for
    a robustness-appendix CI); --quick drops it to 50 for the build/verify smoke path."""
    import pandas as pd
    rng = rng or np.random.default_rng(SEED)
    qids = sorted(df["cluster_id"].dropna().unique().tolist())
    if len(qids) < 3:
        return None
    by_q = {q: df[df["cluster_id"] == q] for q in qids}
    obs = coef_fn(df)
    draws: List[float] = []
    for _ in range(n_boot):
        pick = rng.choice(len(qids), size=len(qids), replace=True)
        samp = pd.concat([by_q[qids[i]] for i in pick], ignore_index=True)
        c = coef_fn(samp)
        if c is not None and np.isfinite(c):
            draws.append(c)
    if len(draws) < 50:
        return {"coef": _safe(obs), "ci95": None, "n_boot_ok": len(draws)}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"coef": _safe(obs), "ci95": [_safe(lo), _safe(hi)],
            "excludes_0": bool(lo > 0 or hi < 0), "n_boot_ok": len(draws)}


def _safe(x):
    try:
        return round(float(x), 5)
    except (TypeError, ValueError):
        return None


def _fit_rate_regression(snips, ref_pattern: str):
    """Fit P(contaminated) ~ search_count + C(architecture). Returns a closure giving the
    POOLED architecture effect (LR-test-style) plus the per-architecture coefficients.

    Per the binding amendment, we NEVER report raw totals — only the modelled per-snippet
    rate and the architecture coefficient holding search_count constant.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    d = snips.dropna(subset=["contaminated", "search_count", "architecture"]).copy()
    d["contaminated"] = d["contaminated"].astype(int)
    if d["contaminated"].nunique() < 2 or d["architecture"].nunique() < 2:
        return None, "degenerate (single-class outcome or single architecture)"

    # reference category = ref_pattern if present, else the most frequent architecture
    arch = pd.Categorical(d["architecture"])
    if ref_pattern in set(arch.categories):
        d["architecture"] = pd.Categorical(
            d["architecture"],
            categories=[ref_pattern] + [a for a in arch.categories if a != ref_pattern])

    def _coef_search(df):
        try:
            m = smf.logit("contaminated ~ search_count + C(architecture)", data=df).fit(
                disp=0, maxiter=200)
            return float(m.params.get("search_count", np.nan))
        except Exception:
            return None

    try:
        model = smf.logit("contaminated ~ search_count + C(architecture)", data=d).fit(
            disp=0, maxiter=200)
    except Exception as e:
        return None, f"fit failed: {type(e).__name__}: {str(e)[:120]}"

    arch_params = {k: _safe(v) for k, v in model.params.items()
                   if k.startswith("C(architecture)")}
    out = {
        "n_snippets": int(len(d)),
        "n_contaminated": int(d["contaminated"].sum()),
        "reference_architecture": ref_pattern if ref_pattern in set(arch.categories)
        else str(d["architecture"].cat.categories[0]),
        "search_count_coef": _safe(model.params.get("search_count")),
        "search_count_p": _safe(model.pvalues.get("search_count")),
        "architecture_coefs": arch_params,
        "architecture_pvalues": {k: _safe(model.pvalues.get(k)) for k in arch_params},
        "pseudo_r2": _safe(getattr(model, "prsquared", np.nan)),
        "coef_search_fn": _coef_search,  # for the bootstrap CI of the search_count slope
        "fit_ok": True,
    }
    return out, None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basis", choices=["citation", "search"], default="citation",
                    help="PRIMARY basis (default citation = uniform 11/12-architecture)")
    ap.add_argument("--ref", default="base_p0",
                    help="reference architecture for the regression contrasts")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the signal availability; write NOTHING")
    ap.add_argument("--finalize", action="store_true",
                    help="write the `contamination` key into canonical_numbers.json. Refuses "
                         "unless a REAL (non-stub) classifier pass is present, so a build/"
                         "verify run can never seed canonical with placeholder numbers.")
    ap.add_argument("--n-boot", type=int, default=500,
                    help="cluster-bootstrap draws for the search_count slope CI (default 500)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke mode: n_boot=50 (build/verify path; CI is indicative only)")
    args = ap.parse_args(argv)
    n_boot = 50 if args.quick else args.n_boot

    import pandas as pd

    snip_path = E6 / f"snippets_{args.basis}.parquet"
    regex_path = E6 / f"regex_flags_{args.basis}.parquet"
    cls = _load_classifier_labels(args.basis)

    have_snip = snip_path.exists()
    have_regex = regex_path.exists()

    print("=" * 70)
    print("E6 STC-AUDIT — STEP 4 contamination rate regression")
    print(f"  basis (PRIMARY)    : {args.basis}")
    print(f"  reference arch     : {args.ref}")
    print(f"  snippets table     : {snip_path.name} -> {'present' if have_snip else 'ABSENT'}")
    print(f"  regex flags        : {regex_path.name} -> {'present' if have_regex else 'ABSENT'}")
    print(f"  classifier labels  : {'present (' + str(cls['n']) + ')' if cls['present'] else 'ABSENT/STUB'}")
    print(f"  canonical target   : {CANONICAL}")
    print(f"  finalize           : {args.finalize}")

    # Assemble the per-snippet contamination signal = regex OR classifier (union).
    result_block: Dict = {
        "_note": (
            "E6 contamination rate regression. PRIMARY basis = df_citations cited-URL/domain "
            "(uniform 11/12-architecture, CITED subset -> conservative under-count of the "
            "true retrieved-snippet leakage rate). Binding amendment: we report the modelled "
            "per-snippet contamination RATE via logistic regression P(contaminated) ~ "
            "search_count + C(architecture), NEVER raw totals (tautological in search count). "
            "GPT-4o is a deterministic classifier TOOL here, never a judge; GPT-5.2 is the "
            "only authoritative judge and is untouched by E6."),
        "prereg": "docs/publication/prereg/prereg_E6.md",
        "basis": args.basis,
        "reference_architecture": args.ref,
        "signal_definition": "regex_contaminated OR classifier_contaminated (union)",
        "classifier_present": bool(cls["present"]),
        "regex_present": bool(have_regex),
    }

    regression = None
    contaminated_query_set: List[str] = []

    if have_snip and (have_regex or cls["present"]):
        snips = pd.read_parquet(snip_path).reset_index(drop=True)
        snips["snip_row"] = np.arange(len(snips))
        snips["regex_flag"] = 0
        if have_regex:
            rf = pd.read_parquet(regex_path)
            if "regex_contaminated" in rf.columns and len(rf) == len(snips):
                snips["regex_flag"] = rf["regex_contaminated"].astype(int).values
        snips["cls_flag"] = 0
        if cls["present"]:
            cls_by_row = {int(l.get("row_index", -1)): int(l.get("contaminated", 0))
                          for l in cls["labels"]}
            snips["cls_flag"] = snips["snip_row"].map(cls_by_row).fillna(0).astype(int)
        snips["contaminated"] = ((snips["regex_flag"] == 1) | (snips["cls_flag"] == 1)).astype(int)
        snips["architecture"] = snips["pattern"].astype(str)
        snips["cluster_id"] = snips.get("query_id")

        # search_count regressor: join telemetry by (pattern, query_id) where canonical;
        # snippets lacking a canonical query_id contribute via architecture only (search_count
        # imputed to the per-architecture median, flagged in the manifest).
        tele_path = E6 / "telemetry.parquet"
        if tele_path.exists():
            tele = pd.read_parquet(tele_path)
            tk = tele.dropna(subset=["canonical_query_id"])[
                ["pattern", "canonical_query_id", "search_count"]].drop_duplicates()
            snips = snips.merge(
                tk, left_on=["pattern", "query_id"],
                right_on=["pattern", "canonical_query_id"], how="left").drop(
                columns=["canonical_query_id"], errors="ignore")
        if "search_count" not in snips.columns:
            snips["search_count"] = np.nan
        med = snips.groupby("architecture")["search_count"].transform("median")
        snips["search_count_imputed"] = snips["search_count"].isna()
        snips["search_count"] = snips["search_count"].fillna(med).fillna(
            snips["search_count"].median()).fillna(0.0)

        # public-benchmark partition only (drop custom / non-public) for the endpoint
        if "is_public" in snips.columns:
            snips_pub = snips[snips["is_public"].fillna(False)].copy()
        else:
            snips_pub = snips
        if len(snips_pub) < 5:
            snips_pub = snips  # fall back so the smoke path still exercises the fit

        reg, err = _fit_rate_regression(snips_pub, args.ref)
        if reg is not None:
            coef_fn = reg.pop("coef_search_fn", None)
            if coef_fn is not None:
                reg["search_count_coef_cluster_ci"] = _cluster_bootstrap_ci(
                    snips_pub.assign(cluster_id=snips_pub["cluster_id"]),
                    coef_fn, n_boot=n_boot, rng=np.random.default_rng(SEED))
                reg["n_boot"] = n_boot
            regression = reg
            # per-architecture modelled rate (mean predicted contamination), descriptive
            reg["per_architecture_observed_rate"] = {
                str(p): _safe(g["contaminated"].mean())
                for p, g in snips_pub.groupby("architecture")
            }
            reg["share_search_count_imputed"] = _safe(
                snips_pub["search_count_imputed"].mean())
        else:
            result_block["regression_error"] = err

        # contaminated query set = queries with >=1 contaminated snippet (for decontamination)
        contaminated_query_set = sorted(
            str(q) for q in snips_pub.loc[snips_pub["contaminated"] == 1, "query_id"]
            .dropna().unique() if str(q))
        result_block["n_snippets_scored"] = int(len(snips_pub))
        result_block["n_snippets_contaminated"] = int(snips_pub["contaminated"].sum())
        result_block["n_contaminated_queries"] = len(contaminated_query_set)
    else:
        result_block["status"] = (
            "no signal yet — run build_contamination_telemetry.py (STEP 1) + "
            "contamination_regex_gate.py (STEP 2); the classifier pass (STEP 3) is optional "
            "for a regex-only PRIMARY result.")

    result_block["regression"] = regression
    result_block["contaminated_query_set"] = contaminated_query_set
    # decontamination wiring target (consumed by rebuild_all.sh STEP 5)
    contam_list_path = E6 / "contaminated_queries.json"

    print(f"  signal             : "
          f"regex={'on' if have_regex else 'off'} / classifier={'on' if cls['present'] else 'off'}")
    if regression:
        print(f"  rate regression    : n={regression['n_snippets']} "
              f"contam={regression['n_contaminated']} "
              f"search_count_coef={regression['search_count_coef']} "
              f"(p={regression['search_count_p']})")
        print(f"  architecture coefs : {regression['architecture_coefs']}")
        print(f"  contaminated qset  : {len(contaminated_query_set)} queries")
    else:
        print(f"  rate regression    : not fit ({result_block.get('regression_error', 'no signal')})")
    print("=" * 70)

    if args.dry_run:
        print("[dry-run] nothing written.")
        return 0

    # Always write the human-readable side-car + the contaminated-query list under the E6 dir.
    E6.mkdir(parents=True, exist_ok=True)
    (E6 / "contamination_provisional.json").write_text(json.dumps(result_block, indent=2))
    contam_list_path.write_text(json.dumps(
        {"basis": args.basis, "contaminated_query_set": contaminated_query_set}, indent=2))
    print(f"wrote {E6/'contamination_provisional.json'} and {contam_list_path}")

    # Mutate canonical ONLY on an explicit --finalize AND a REAL classifier pass present.
    # This guard is what keeps a build/verify run from seeding canonical with placeholders.
    if not args.finalize:
        print("[provisional] canonical_numbers.json NOT touched (pass --finalize after the "
              "human supplies the classifier pass to write the `contamination` key).")
        return 0
    if not cls["present"]:
        print("[REFUSING --finalize] no real classifier pass present yet; canonical untouched. "
              "Supply results/contamination_e6/classifier/labels_<basis>.json first.")
        return 0

    cn = json.loads(CANONICAL.read_text())
    cn["contamination"] = result_block
    _txt = json.dumps(cn, indent=1)
    _tmp = str(CANONICAL) + ".tmp"
    with open(_tmp, "w") as fh:
        fh.write(_txt)
    os.replace(_tmp, str(CANONICAL))
    print(f"wrote canonical key `contamination` into {CANONICAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
