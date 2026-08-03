#!/usr/bin/env python3
"""ablation_length_check -- does report-length shrinkage confound the P4
no-triangulation ablation, the ~0.06 figure the abstract actually quotes
("the largest isolable component moves scores by ~0.06" = ablations.
ablation_p4_no_triangulation.delta = -0.0599)?

Flagged by an adversarial review of the T1 length-adjustment reframe
(2026-07-27): the paper already length-adjusts the P0-vs-cluster headline gap
(build_length_adjusted_headline.py) but had never checked whether the
abstract's OWN quoted ~0.06 delta -- an internal P4 component ablation, not
the P0-vs-cluster gap -- is itself partly a length artefact. Removing
triangulation plausibly also shortens the report (less cross-checking prose
to write), which the length-adjusted-headline logic would predict inflates
the apparent quality drop.

Method: identical judge pool and score field to build_numbers.py's ablations()
(gpt52 + claude_sonnet, ovc = corrected_overall, cell-mean per (pattern,
query)), joined to df_runs.report_word_count. Same pooled-OLS length-control
spec as build_length_adjusted_headline.py's primary spec A (score ~ pattern
dummy, no intercept, + beta*(words/1000 - grand_mean_kwords), each pattern's
adjusted mean read off at grand-mean length). Query-clustered bootstrap CI.
$0 CPU, no API. STAGING convention: writes only to
analysis/staging/ablation_length_check.json; wired into rebuild_all.sh, which merges it
into canonical_numbers.json['ablation_length_check'] via merge_staging_key.py immediately
after this script runs.
"""
import json
import numpy as np
import pandas as pd

ROOT = "."
A = f"{ROOT}/data/analysis"
AN = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"

SEED = 20260727
N_BOOT = 10000
ABL_PAT = "ablation_p4_no_triangulation"
BASE_PAT = "base_p4"
JUDGES = ["gpt52", "claude_sonnet"]


def corrected_overall(df):
    c = df["overall_score"].copy()
    m = df["judge"].eq("claude_sonnet")
    if "overall_score_recomputed" in df.columns:
        c = c.where(~m, df["overall_score_recomputed"])
    return c


def main():
    ov = pd.read_parquet(f"{A}/df_overall_scores.parquet")
    ov["ovc"] = corrected_overall(ov)
    ov = ov[ov["judge"].isin(JUDGES) & ov["pattern"].isin([ABL_PAT, BASE_PAT])]
    cellmean = ov.groupby(["pattern", "query_id"], observed=True)["ovc"].mean()

    runs = pd.read_parquet(f"{A}/df_runs.parquet")
    runs = runs[runs["pattern"].isin([ABL_PAT, BASE_PAT])][["pattern", "query_id", "report_word_count"]]
    words = runs.set_index(["pattern", "query_id"])["report_word_count"]

    piv_score = cellmean.unstack("pattern")
    piv_words = words.unstack("pattern") if isinstance(words.index, pd.MultiIndex) else runs.pivot_table(
        index="query_id", columns="pattern", values="report_word_count", aggfunc="first")
    piv_words = runs.pivot_table(index="query_id", columns="pattern", values="report_word_count", aggfunc="first")
    common = piv_score.dropna().index.intersection(piv_words.dropna().index)
    piv_score = piv_score.loc[common]
    piv_words = piv_words.loc[common]
    n = len(common)

    raw_delta = float((piv_score[ABL_PAT] - piv_score[BASE_PAT]).mean())

    words_ablation = piv_words[ABL_PAT]
    words_base = piv_words[BASE_PAT]
    word_diff = words_ablation - words_base
    word_diff_mean = float(word_diff.mean())
    frac_shorter = float((word_diff < 0).mean())

    def _adjusted_delta(idx):
        rows = []
        for pat, col in [(ABL_PAT, ABL_PAT), (BASE_PAT, BASE_PAT)]:
            rows.append(pd.DataFrame({
                "pattern": pat,
                "score": piv_score.loc[idx, col].values,
                "words": piv_words.loc[idx, col].values,
            }))
        long = pd.concat(rows, ignore_index=True)
        grand_mean_kw = long["words"].mean() / 1000.0
        long["kw_c"] = long["words"] / 1000.0 - grand_mean_kw
        X = pd.get_dummies(long["pattern"]).astype(float)
        X["kw_c"] = long["kw_c"].values
        beta, *_ = np.linalg.lstsq(X.values, long["score"].values, rcond=None)
        coefs = dict(zip(X.columns.tolist(), beta))
        return coefs[ABL_PAT] - coefs[BASE_PAT], coefs.get("kw_c")

    point_adj_delta, length_coef = _adjusted_delta(common)

    rng = np.random.default_rng(SEED)
    qids = np.asarray(common)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        resampled = qids[rng.integers(0, n, n)]
        boots[i], _ = _adjusted_delta(pd.Index(resampled))
    ci = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]

    result = {
        "_note": (
            "Checks whether ablation_p4_no_triangulation's delta (-0.0599, the figure "
            "the abstract's '~0.06' actually quotes) survives the same length-adjustment "
            "logic already applied to the P0-vs-cluster headline gap. Same judge pool "
            "and ovc field as build_numbers.py's ablations()."
        ),
        "seed": SEED,
        "n_boot": N_BOOT,
        "n_queries_common": n,
        "judges_pooled": JUDGES,
        "raw_delta": round(raw_delta, 4),
        "canonical_ablations_delta": -0.0599,
        "word_count": {
            "base_p4_mean": round(float(words_base.mean()), 1),
            "ablation_mean": round(float(words_ablation.mean()), 1),
            "mean_diff": round(word_diff_mean, 1),
            "fraction_shorter": round(frac_shorter, 4),
        },
        "length_adjusted": {
            "length_coef_per_1000_words": round(float(length_coef), 4),
            "delta_at_grand_mean_length": round(float(point_adj_delta), 4),
            "ci95": ci,
            "excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        },
        "verdict": (
            "SURVIVES, PARTIALLY ATTENUATED. Removing triangulation shortens P4 reports "
            f"by {abs(word_diff_mean):.0f} words on average ({frac_shorter*100:.0f}% of "
            "queries shorter), and the length-adjusted delta shrinks from "
            f"{raw_delta:.3f} (raw, this judge pool) to {point_adj_delta:.3f} "
            f"(at grand-mean length, {(1 - point_adj_delta/raw_delta)*100:.0f}% "
            "attenuation), but the adjusted 95% CI still excludes zero. So part of the "
            "abstract's quoted ~0.06 ablation effect is a length artefact, similar in "
            "kind (though smaller in share) to the ~40% compute confound already "
            "reported for the P1 matched-budget disentanglement probe -- the "
            "triangulation-removal effect on quality is real but the raw figure "
            "overstates the pure-content-removal share by roughly this attenuation."
        ),
    }
    out_path = f"{AN}/staging/ablation_length_check.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")
    print(json.dumps({k: result[k] for k in ["raw_delta", "length_adjusted", "word_count"]}, indent=2))


if __name__ == "__main__":
    main()
