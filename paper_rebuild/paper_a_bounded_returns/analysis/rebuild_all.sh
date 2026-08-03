#!/usr/bin/env bash
# Archival full-corpus rebuild script.
#
# This is kept for method provenance and for maintainers who have the private raw
# generated-report forests, raw judge verdict trees, and optional local-model/GPU
# outputs. It is not the public reproduction entry point. Public users should run:
#
#   deep-research paper rebuild paper-a --check-only
#   deep-research paper rebuild paper-a --skip-compile
#   deep-research paper rebuild paper-a
#
# To run this archival script deliberately, set:
#
#   DEEP_RESEARCH_ALLOW_ARCHIVAL_REBUILD=1 bash paper_rebuild/paper_a_bounded_returns/analysis/rebuild_all.sh
set -e
cd "$(dirname "$0")/../../.."

if [ "${DEEP_RESEARCH_ALLOW_ARCHIVAL_REBUILD:-}" != "1" ]; then
  cat >&2 <<'EOF'
This archival full-corpus rebuild needs raw artifacts that are intentionally not
shipped in the public GitHub release. Use the supported public rebuild instead:

  deep-research paper rebuild paper-a --skip-compile

Maintainers with the private archival stores can opt in with:

  DEEP_RESEARCH_ALLOW_ARCHIVAL_REBUILD=1 bash paper_rebuild/paper_a_bounded_returns/analysis/rebuild_all.sh
EOF
  exit 2
fi

missing=0
for required in results/experiments data/analysis/df_runs.parquet data/analysis/df_verdicts.parquet; do
  if [ ! -e "$required" ]; then
    echo "missing archival rebuild input: $required" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "archival rebuild aborted; required raw/derived inputs are unavailable" >&2
  exit 2
fi

[ -f venv/bin/activate ] && source venv/bin/activate
A=paper_rebuild/paper_a_bounded_returns/analysis
echo "[1/9] build_numbers";            python $A/build_numbers.py            >/dev/null
# [E6] decontamination recompute — self-guards (exits 0 with a notice if E6 has not produced
# results/contamination_e6/contaminated_queries.json yet), so the chain stays green pre-E6.
echo "[1e6] build_contamination_decontaminated"; python $A/build_contamination_decontaminated.py >/dev/null
echo "[1e6b] build_contamination_clean_capability"; python $A/build_contamination_clean_capability.py >/dev/null 2>&1 || true
echo "[2/9] build_numbers_extended";   python $A/build_numbers_extended.py   >/dev/null
echo "[2b] build_bing_tavily_citation_volume"; python $A/build_bing_tavily_citation_volume.py --write >/dev/null
echo "[2c] build_bing_tavily_sonnet_crosscheck"; python $A/build_bing_tavily_sonnet_crosscheck.py --write >/dev/null
echo "[3/9] build_pairwise";           python $A/build_pairwise.py           >/dev/null
echo "[3b] build_pairwise_tost_holm"; python $A/build_pairwise_tost_holm.py --write --force >/dev/null
echo "[4/9] build_citation_regression";python $A/build_citation_regression.py --write>/dev/null
echo "[5/9] build_run_stability";      python $A/build_run_stability.py      >/dev/null
echo "[5a1] build_judge_scale_standardized"; python $A/build_judge_scale_standardized.py >/dev/null
echo "[5b] build_bestofn";          python $A/build_bestofn.py           >/dev/null
echo "[5b2] build_bestofn_decoupled"; python $A/build_bestofn_decoupled.py >/dev/null
echo "[5b2b] build_bestofn_decoupled_reversed"; python $A/build_bestofn_decoupled_reversed.py >/dev/null
echo "[5b3] build_b2";              python $A/build_b2.py                >/dev/null
echo "[5b4] build_p11_16turn";      python $A/build_p11_16turn.py        >/dev/null
echo "[5b4b] build_p12_vs_p9";      python $A/build_p12_vs_p9.py         >/dev/null
echo "[5b5] build_joint_holm";      python $A/build_joint_holm.py        >/dev/null
echo "[5b6] build_review_robustness"; python $A/build_review_robustness.py >/dev/null
echo "[5b7] build_length_adjusted_headline"; python $A/build_length_adjusted_headline.py --write --force >/dev/null
echo "[5c] build_oracle_opus";        python $A/build_oracle_opus.py        >/dev/null 2>&1 || true
echo "[5d] build_oracle_robust_ci";    python $A/build_oracle_robust_ci.py   >/dev/null
echo "[5e] build_oracle_factual_tost"; python $A/build_oracle_factual_tost.py>/dev/null
echo "[5f] build_disentanglement";     python $A/build_disentanglement.py --write >/dev/null
echo "[5g] build_carried_metrics";     python $A/build_carried_metrics.py    >/dev/null
echo "[5h] build_judge_vs_gold";       python $A/build_judge_vs_gold.py --write >/dev/null
echo "[5i] build_variance_decomposition"; python $A/build_variance_decomposition.py >/dev/null
echo "[5j] build_n_eff";               python $A/build_n_eff.py --write      >/dev/null
echo "[5j1] build_n_eff_participation_ratio"; python $A/build_n_eff_participation_ratio.py --write >/dev/null
echo "[5k] build_routability";         python $A/build_routability.py        >/dev/null
echo "[5l] build_routability_stageb";  python $A/build_routability_stageb.py >/dev/null
echo "[5m] build_e4_cite_causal_v2";      python $A/build_e4_cite_causal_v2.py     >/dev/null 2>&1 || true
echo "[5e2] build_a2_e14_oracle_p9p10_and_rxu"; ./venv/bin/python scripts/build_a2_e14_oracle_p9p10_and_rxu.py >/dev/null
echo "[5n] build_e5_dose_response";        python "$A/../../../scripts/build_e5_dose_response.py" >/dev/null
echo "[5n] build_local_benchmark";         python "$A/../../../scripts/build_local_benchmark.py" >/dev/null
echo "[5h1] build_judge_vs_human";       python $A/build_judge_vs_human.py --write >/dev/null
echo "[5h2] wire_healthbench_into_judge_vs_human"; python scripts/wire_healthbench_into_judge_vs_human.py >/dev/null
echo "[5n] build_e7_selector_kappa";       python scripts/build_e7_selector_kappa.py >/dev/null
echo "[5n] build_e8_vintage";              python "$A/../../../scripts/build_e8_vintage.py" >/dev/null 2>&1 || true
echo "[5n] build_drb_race_by_family";      python $A/build_drb_race_by_family.py --write >/dev/null 2>&1 || true
# [5n] build_e12_external_validation — SKIPPED (2026-06-30 reconcile). This analysis-dir
# builder writes the LEGACY key 'external_validation_e12', which nothing in the paper
# consumes; the live, consumed key is 'e12_extval' (owned by scripts/build_e12_extval.py).
# Running it here only pollutes the store with a stale duplicate key and was implicated in
# a prior stale-write key-drop. Left disabled so a future rebuild cannot drop/duplicate keys.
echo "[5n] build_e12_external_validation (SKIPPED: legacy key, see comment)"; true
echo "[5n] build_e13_detector_roc";        python $A/build_e13_detector_roc.py >/dev/null 2>&1 || true
echo "[5n] build_e14_oracle_entail";       python $A/build_e14_oracle_entail.py >/dev/null 2>&1 || true
echo "[5l2] build_routability_judgerobust"; python $A/build_routability_judgerobust.py >/dev/null
echo "[5o] build_loso_jackknife"; python $A/build_loso_jackknife.py >/dev/null 2>&1 || true
echo "[5i2] build_variance_3way_sonnet"; python $A/build_variance_3way_sonnet.py >/dev/null
echo "[T1_drjudge_errmatrix] drjudge_error_structure"; python scripts/build_drjudge_error_structure.py >/dev/null 2>&1
echo "[5o] build_e5_equivalence"; python "$A/../../../scripts/build_e5_equivalence.py" --write >/dev/null
echo "[5p] build_routability_equivalence"; python $A/build_routability_equivalence.py --write >/dev/null
echo "[5n] build_e5_gold_consumption"; python "$A/../../../scripts/build_e5_gold_consumption.py" >/dev/null
echo "[5j2] build_n_eff_within_openai"; python $A/build_n_eff_within_openai.py >/dev/null
echo "[p2a] build_p2_winmult"; python $A/build_p2_winmult.py >/dev/null
echo "[p2b] build_p2_neff_k"; python $A/build_p2_neff_k.py >/dev/null
echo "[p2c] build_p2_judge_kappa"; python $A/build_p2_judge_kappa.py >/dev/null
echo "[p2d] build_p2_youden_j"; python $A/build_p2_youden_j.py --write >/dev/null
echo "[p2e] build_p2_var_bootstrap"; python $A/build_p2_var_bootstrap.py >/dev/null
echo "[p2f] build_p2_bayes_crosscheck"; python $A/build_p2_bayes_crosscheck.py >/dev/null
echo "[p2g] build_p2_rxu_conditional"; python $A/build_p2_rxu_conditional.py >/dev/null 2>&1 || true
echo "[p2h] build_p2_faithfulness"; python $A/build_p2_faithfulness.py >/dev/null 2>&1 || true
# [p3] STAGING-only builders (per-script convention: never touch canonical_numbers.json
# themselves; "the main programme loop merges staging blobs into canonical separately").
# Each is followed by an explicit merge_staging_key.py call so a fresh rebuild cannot
# silently drop these keys or leave them stale relative to their staging blob.
echo "[p3a] build_frozen_defence (repo-root analysis/, not $A)"; python paper_rebuild/paper_a_bounded_returns/supporting_analysis/build_frozen_defence.py >/dev/null
echo "[p3a-merge] frozen_defence";      python $A/merge_staging_key.py paper_rebuild/paper_a_bounded_returns/supporting_analysis/staging/frozen_defence.json frozen_defence
echo "[p3b] build_bakeoff";             python $A/build_bakeoff.py            >/dev/null
echo "[p3b-merge] bakeoff";             python $A/merge_staging_key.py $A/staging/bakeoff.json bakeoff
echo "[p3b2] build_bakeoff_concentration"; python $A/build_bakeoff_concentration.py >/dev/null
echo "[p3b2-merge] bakeoff_concentration"; python $A/merge_staging_key.py $A/staging/bakeoff_concentration.json bakeoff_concentration
echo "[p3c] build_second_backbone";     python $A/build_second_backbone.py    >/dev/null
echo "[p3c-merge] second_backbone";     python $A/merge_staging_key.py $A/staging/second_backbone.json second_backbone
echo "[p3d] build_second_backbone_claude"; python $A/build_second_backbone_claude.py >/dev/null
echo "[p3d-merge] second_backbone_claude"; python $A/merge_staging_key.py $A/staging/second_backbone_claude.json second_backbone_claude
echo "[p3e] build_frozen_vintage_3family"; python $A/build_frozen_vintage_3family.py >/dev/null
echo "[p3e-merge] frozen_vintage_3family"; python $A/merge_staging_key.py $A/staging/frozen_vintage_3family.json frozen_vintage_3family
echo "[p3f] build_headline_replicates_3family"; python $A/build_headline_replicates_3family.py >/dev/null
echo "[p3f-merge] headline_replicates_3family"; python $A/merge_staging_key.py $A/staging/headline_replicates_3family.json headline_replicates_3family
echo "[p3g] build_synthablation_sonnet";  python $A/build_synthablation_sonnet.py >/dev/null
echo "[p3g-merge] synthablation_sonnet_crosscheck"; python $A/merge_staging_key.py $A/staging/synthablation_sonnet.json synthablation_sonnet_crosscheck
echo "[p3h] build_distractor_sonnet";     python $A/build_distractor_sonnet.py >/dev/null
echo "[p3h-merge] distractor_sonnet_crosscheck"; python $A/merge_staging_key.py $A/staging/distractor_sonnet.json distractor_sonnet_crosscheck
echo "[p3i] build_counterfactual_sonnet"; python $A/build_counterfactual_sonnet.py >/dev/null
echo "[p3i-merge] counterfactual_sonnet_crosscheck"; python $A/merge_staging_key.py $A/staging/counterfactual_sonnet.json counterfactual_sonnet_crosscheck
echo "[p3j] build_perturb_sonnet";        python $A/build_perturb_sonnet.py >/dev/null
echo "[p3j-merge] perturb_sonnet_crosscheck"; python $A/merge_staging_key.py $A/staging/perturb_sonnet.json perturb_sonnet_crosscheck
echo "[p3k] build_ablation_length_check"; python $A/build_ablation_length_check.py >/dev/null
echo "[p3k-merge] ablation_length_check"; python $A/merge_staging_key.py $A/staging/ablation_length_check.json ablation_length_check
echo "[p3l] build_isoquant_claimtype"; python $A/build_isoquant_claimtype.py --write >/dev/null
echo "[6/9] make_stratification_figure";python $A/make_stratification_figure.py >/dev/null
echo "[7/9] make_cost_figure";         python $A/make_cost_figure.py         >/dev/null
echo "[8/9] make_oracle_figure";       python $A/make_oracle_figure.py       >/dev/null
echo "[8b] make_money_figure";         python $A/make_money_figure.py        >/dev/null
echo "[8c] make_cd_diagram";           python $A/make_cd_diagram.py          >/dev/null
echo "[8d] make_vintage_figure";       python $A/make_vintage_figure.py      >/dev/null
echo "[8e] make_disentanglement_figure"; python $A/make_disentanglement_figure.py >/dev/null
echo "[8f] make_e5_dose_response_figure"; python $A/make_e5_dose_response.py >/dev/null
echo "[8g] make_judge_gold_figure";     python $A/make_judge_gold_figure.py  >/dev/null
echo "[9/9] make_tables";              python $A/make_tables.py              >/dev/null
echo "[9c] make_b2_bestofn_tables";   python $A/make_b2_bestofn_tables.py  >/dev/null
echo "[9c2] make_paper2_tables";      python $A/make_paper2_tables.py      >/dev/null
echo "[9b] build_irr_robust";          python $A/build_irr_robust.py
echo "[9d] build_search_robustness"; python "$A/../../../scripts/build_search_robustness.py" --write --force >/dev/null 2>&1 || true
echo "[9e] build_search_robustness_clustered"; python $A/build_search_robustness_clustered.py --write --force >/dev/null 2>&1 || true
echo "[verify] reconcile_tables";      python $A/reconcile_tables.py
echo "[5b8] build_oracle_gap_compression_ci"; python $A/build_oracle_gap_compression_ci.py
echo "[5b9] build_e14_neutral_bracket";       python $A/build_e14_neutral_bracket.py
echo "[5b10] build_disentanglement_perdim_ci"; python $A/build_disentanglement_perdim_ci.py --write >/dev/null
echo "[5b11] build_citation_regression_perjudge_inference"; python $A/build_citation_regression_perjudge_inference.py --write >/dev/null
echo "[verify2] reconcile_prose";             python $A/reconcile_prose.py
echo "DONE. canonical_numbers.json + figures/ + tables/ regenerated."
