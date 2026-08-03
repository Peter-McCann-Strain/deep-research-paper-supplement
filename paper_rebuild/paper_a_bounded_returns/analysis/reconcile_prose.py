#!/usr/bin/env python
"""Prose drift tripwire (companion to reconcile_tables.py): extract every numeric literal
from main.tex PROSE (between \\begin{abstract} and \\appendix), and check each against the
flattened value set of canonical_numbers.json at the token's PRINTED precision (a prose
"0.145" matches a store 0.1454; percentage and absolute-value transforms are admitted, as in
reconcile_tables.py). Legitimate non-store numbers (design counts, rubric weights, stats
conventions, model names, results quoted from OTHER papers, ...) are catalogued in
prose_whitelist.json and reported per category. main.tex is read FRESH on every run.

Skipped structurally (never tokenised): comments, \\label/\\ref/\\Cref args, citation keys,
\\href URLs, \\includegraphics, figure dimensions (0.92\\linewidth), date/vintage strings
(2024-09, Mar~16), years 19xx/20xx, and scientific-notation p-value magnitudes (5x10^-5),
the last two reported as auto-rule counts.

Prints: total tokens, matched, whitelisted (per category), unaccounted (with line numbers
and the nearest store key). Always exits 0.

Because the store's rounded-value grid is dense (grid membership at 2-3 decimals is weakly
selective), a second, strict pass anchors the LOAD-BEARING inline statistics of the abstract,
contributions, and results to exact store keys: each anchor is a context regex that reads the
number currently printed in the prose and compares it to its canonical value at printed
precision (half-up tolerant). A reworded sentence downgrades an anchor to NOT-FOUND (warn);
a changed number is reported as DRIFT.
"""
import json
import os
import re
import sys

ROOT = "."
PAPER = f"{ROOT}/paper_rebuild/paper_a_bounded_returns"
ANA = f"{PAPER}/analysis"
MAIN = sys.argv[1] if len(sys.argv) > 1 else f"{PAPER}/main.tex"  # optional override for self-tests
WHITELIST = f"{ANA}/prose_whitelist.json"

# ---------------------------------------------------------------- store loading
cn = json.load(open(f"{ANA}/canonical_numbers.json"))


def walk(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, path + (str(i),))
    elif isinstance(o, bool):
        return
    elif isinstance(o, (int, float)):
        v = float(o)
        if v == v and abs(v) != float("inf"):  # drop NaN/inf (e.g. mean_opus NaN for P11/P12)
            yield path, v


STORE = list(walk(cn))
RAW = [v for _, v in STORE]
# transformed spaces admitted by reconcile_tables.py: raw, x100 (percent), abs
SPACES = RAW + [v * 100 for v in RAW] + [abs(v) for v in RAW]

_round_cache = {}


def rounded_set(d):
    if d not in _round_cache:
        _round_cache[d] = {round(v, d) for v in SPACES}
    return _round_cache[d]


INT_EXACT = {round(v, 4) for v in SPACES if abs(v - round(v)) < 5e-5}


def matches_store(x, d):
    """Printed-precision match: token x printed with d decimals matches if some store value
    (raw, x100, or abs) rounds to x at d decimals. Guards: a sub-resolution zero never
    matches; integer tokens (d=0) match exact-integer store values at any size, and rounded
    store values only for x >= 30 (small counts go through the whitelist instead)."""
    if d == 0:
        if float(x) in INT_EXACT or -float(x) in INT_EXACT:
            return True
        return abs(x) >= 30 and (round(x, 0) in rounded_set(0) or round(-x, 0) in rounded_set(0))
    r = round(x, d)
    if r == 0 and x != 0:
        return False
    return r in rounded_set(d) or round(-x, d) in rounded_set(d)


def nearest_store(x):
    """Best (key-path, value) explanation for an unmatched token, over raw and percent space."""
    best, bd = None, float("inf")
    for path, v in STORE:
        for cand, tag in ((v, ""), (v * 100, " (x100)")):
            dd = abs(cand - x)
            rel = dd / max(abs(x), 1e-9)
            score = min(dd, rel)
            if score < bd:
                bd, best = score, ("/".join(path) + tag, cand)
    return best


# ---------------------------------------------------------------- prose extraction
src = open(MAIN).read()  # read FRESH; main.tex is actively edited elsewhere
lines = src.splitlines()
try:
    a = next(i for i, l in enumerate(lines) if "\\begin{abstract}" in l)
    b = next(i for i, l in enumerate(lines) if l.strip() == "\\appendix")
except StopIteration:
    raise SystemExit("[reconcile_prose] could not locate abstract/appendix bounds in main.tex")

SCI = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\{\\times\}|\\times|\\!)*\s*10\^\{?(-?\d+)\}?")
DATE = re.compile(r"\b(19|20)\d\d\$?\s*(?:--?|-)\s*\$?\d{2}\b")  # 2024-09, $2024$-$09$
MONTHDAY = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)~?\d+\b")
TOKEN = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?")

n_sci = n_year = 0


def clean(line):
    global n_sci
    l = re.sub(r"(?<!\\)%.*", "", line)                                # comments
    l = re.sub(r"\\(label|ref|Cref|cref|eqref|pageref|nameref)\{[^}]*\}", " ", l)
    l = re.sub(r"\\cite\w*(\[[^\]]*\])?\{[^}]*\}", " ", l)             # citation keys
    l = re.sub(r"\\href\{[^}]*\}\{[^}]*\}", " ", l)                    # URLs/DOIs (both args)
    l = re.sub(r"\\href\{[^}]*\}", " ", l)
    l = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", " ", l)
    l = re.sub(r"\\(begin|end)\{[^}]*\}", " ", l)
    l = re.sub(r"[\d.]+\\(linewidth|textwidth|columnwidth|linespread)", " ", l)  # 0.92\linewidth
    l = re.sub(r"\\multirow\{\d+\}\{[^}]*\}", " ", l)
    l = re.sub(r"\\(shortstack|resizebox|parbox)\b(\[[^\]]*\])?", " ", l)
    l = DATE.sub(" ", l)                                               # 2024-09 vintages
    l = MONTHDAY.sub(" ", l)                                           # Mar~16
    # scientific notation (p-value magnitudes): count and remove
    n_sci += len(SCI.findall(l))
    l = SCI.sub(" ", l)
    l = re.sub(r"(?<=\d)\{,\}(?=\d)", "", l)                           # 248{,}536 -> 248536
    l = l.replace("{:}", ":").replace("{-}", "-").replace("{+}", "+").replace("{=}", "=")
    l = re.sub(r"\^\{?-?\d+\}?", " ", l)                               # N^2, chi^2, 10^{-15} residue
    l = re.sub(r"_\{?\d+\}?", " ", l)                                  # MDE_{80}
    l = l.replace("$", " ").replace("\\%", "%").replace("~", " ")
    l = re.sub(r"\\[a-zA-Z]+", " ", l)                                 # remaining macros
    l = re.sub(r"\\[!,;:<>]", " ", l)                                  # \! \, spacing escapes
    l = l.replace("--", " to ")                                        # range dashes, not signs
    return l


cleaned_lines = [clean(lines[i]) for i in range(a, b)]
PROSE = re.sub(r"\s+", " ", " ".join(cleaned_lines))  # normalized one-string prose for anchors

tokens = []  # (lineno_1based, text, value, decimals)
for i in range(a, b):
    for m in TOKEN.finditer(clean(lines[i])):
        t = m.group(0)
        v = float(t)
        d = len(t.split(".")[1]) if "." in t else 0
        if d == 0 and 1900 <= abs(v) <= 2099:
            n_year += 1
            continue
        tokens.append((i + 1, t, v, d))

# ---------------------------------------------------------------- whitelist
wl = json.load(open(WHITELIST)) if os.path.exists(WHITELIST) else {"categories": {}}
SMALL_INT_MAX = wl.get("auto_rules", {}).get("small_integers_max", -1)
CATS = wl.get("categories", {})
cat_values = {name: set(spec.get("values", [])) for name, spec in CATS.items()}


def whitelist_category(v, d):
    if d == 0 and 0 <= v <= SMALL_INT_MAX:
        return "small-integer/notation (auto)"
    for name, vals in cat_values.items():
        if v in vals or (d == 0 and int(v) in vals):
            return name
    return None


# ---------------------------------------------------------------- reconcile
matched, whitelisted, unaccounted = 0, {}, []
for ln, t, v, d in tokens:
    if matches_store(v, d):
        matched += 1
        continue
    cat = whitelist_category(v, d)
    if cat:
        whitelisted[cat] = whitelisted.get(cat, 0) + 1
        continue
    unaccounted.append((ln, t, v))

n_wl = sum(whitelisted.values())
total = len(tokens)
print(f"[reconcile_prose] prose region: main.tex lines {a+1}-{b+1} (abstract -> \\appendix), read fresh")
print(f"[reconcile_prose] structurally skipped: {n_year} year tokens, {n_sci} sci-notation p-value magnitudes")
print(f"[reconcile_prose] {total} numeric tokens found")
print(f"[reconcile_prose]   matched to canonical store (printed precision, raw/x100/abs): {matched}")
print(f"[reconcile_prose]   whitelisted: {n_wl}")
for name in sorted(whitelisted, key=whitelisted.get, reverse=True):
    note = CATS.get(name, {}).get("note", "")
    print(f"    {whitelisted[name]:4d}  {name}" + (f"  -- {note}" if note else ""))
print(f"[reconcile_prose]   unaccounted: {len(unaccounted)}")
if unaccounted:
    print("[reconcile_prose] WARN — unaccounted numeric tokens (line: token -> nearest store key):")
    for ln, t, v in unaccounted:
        key, val = nearest_store(v)
        print(f"  L{ln}: {t}  -> nearest {key} = {val:g}")
else:
    print("[reconcile_prose] PASS — every prose numeric token is store-backed or whitelisted")

# ---------------------------------------------------------------- load-bearing anchors
# The grid pass above is tolerance-faithful to reconcile_tables.py but weakly selective (the
# store's rounded grid is dense). These anchors pin the load-bearing inline statistics of the
# abstract, contributions, and results to EXACT store keys: the regex reads the number the
# prose currently prints; it must equal the canonical value at printed precision (half-up).


def P(path):
    """Store accessor by /-path."""
    o = cn
    for seg in path.split("/"):
        o = o[int(seg)] if isinstance(o, list) else o[seg]
    return float(o)


HB = "headline_cluster_gap"
ANCHORS = [
    # --- abstract ---
    ("abstract: largest component lever",
     r"largest isolable component moves scores by\s+(0\.\d+)",
     [lambda: max(abs(v["delta"]) for v in cn["ablations"].values() if isinstance(v, dict) and "delta" in v)]),
    ("abstract: 7B-vs-GPT-4o-twin gap",
     r"scores (0\.\d+) below its GPT-4o twin",
     [f"{HB}/three_judge/p0_minus_p9/point"]),
    ("abstract: 7B-vs-best-pipeline gap",
     r"(0\.\d+)\s*\(\s*(0\.\d+)\s*\)\s*below the top-cluster pipeline P4",
     [lambda: P(f"{HB}/three_judge/per_pattern_means/base_p4") - P(f"{HB}/three_judge/per_pattern_means/base_p9"),
      "contamination/clean_residual_capability/clean_panel3/p4_minus_p9/gap"]),
    ("contrib2: decontaminated P4-vs-P9",
     r"\(\s*(0\.\d+) and (0\.\d+) on the contamination-clean residual",
     ["contamination/clean_residual_capability/clean_panel3/p0_minus_p9/gap",
      "contamination/clean_residual_capability/clean_panel3/p4_minus_p9/gap"]),
    ("abstract: 7B orchestration premiums",
     r"adds only (0\.\d+)\s*to\s*(0\.\d+)",
     ["b2_7b_premium/premium_7b_p1", "b2_7b_premium/premium_7b_p4"]),
    ("abstract: effective votes",
     r"(\d\.\d+) effective votes",
     ["n_eff/overall/n_eff"]),
    ("abstract: oracle citation lift + CI",
     r"raises the six-pattern top cluster's citation quality \( \+(0\.\d+) , 95% CI \[(0\.\d+),\s*(0\.\d+)\]",
     ["oracle/cluster_ci_robust/dims/citation_quality/point_delta",
      "oracle/cluster_ci_robust/dims/citation_quality/block_ci/0",
      "oracle/cluster_ci_robust/dims/citation_quality/block_ci/1"]),
    ("abstract: entailment shortfall closed",
     r"closes only\s+(0\.\d+) of it",
     ["e14_oracle_entail/cluster_retrieval_component"]),
    ("abstract: released verdict count",
     r"(\d{6}) criterion-level verdicts",
     ["verdicts/total_rows"]),
    # --- contributions ---
    ("contrib2: per-judge capability gap range",
     r"per-judge (0\.\d+) to (0\.\d+) against P4",
     [lambda: min(P("headline/per_pattern/base_p4/mean_gpt52") - P("headline/per_pattern/base_p9/mean_gpt52"),
                  P("headline/per_pattern/base_p4/mean_opus") - P("headline/per_pattern/base_p9/mean_opus"),
                  P("headline/per_pattern/base_p4/mean_sonnet_corrected") - P("headline/per_pattern/base_p9/mean_sonnet_corrected")),
      lambda: max(P("headline/per_pattern/base_p4/mean_gpt52") - P("headline/per_pattern/base_p9/mean_gpt52"),
                  P("headline/per_pattern/base_p4/mean_opus") - P("headline/per_pattern/base_p9/mean_opus"),
                  P("headline/per_pattern/base_p4/mean_sonnet_corrected") - P("headline/per_pattern/base_p9/mean_sonnet_corrected"))]),
    ("contrib2/scale: closure bound",
     r"closure beyond \+(0\.\d+)",
     ["b2_7b_premium/premium_7b_p4_test/ci95/1"]),
    ("contrib3: provenance beta",
     r"\{provenance\}\}=\+(0\.\d+)",
     ["citation_regression/coefs/provenance_rate/beta"]),
    ("contrib3: density beta",
     r"\{count\}\}=\+(0\.\d+)",
     ["citation_regression/coefs/log_cit/beta"]),
    ("retrieval: GPT-5.2 density null",
     r"=(-0\.\d+) , query-clustered\s*\n?\s*\$?p=0\.92",
     ["density_per_judge/gpt52/beta_density"]),
    ("contrib4/oracle: gap shrink",
     r"gap from (0\.\d+) to (0\.\d+)",
     ["oracle/gap_p0_to_cluster_base", "oracle/gap_p0_to_cluster_oracle"]),
    # --- results: judge audit ---
    ("results: Krippendorff alpha",
     r"panel attains Krippendorff =(0\.\d+)",
     ["irr/krippendorff_alpha_overall"]),
    ("results: ICC(A,1)",
     r"ICC\(A,1\)\}=(0\.\d+)",
     ["irr/icc_a1"]),
    ("results: ICC(A,k=3)",
     r"k=3\)=(0\.\d+)",
     ["irr/icc_ak3"]),
    ("results: judge-gold AUC (GPT-5.2)",
     r"factual-accuracy AUC is (0\.\d+)",
     ["judge_vs_gold/per_judge/gpt52/factual_accuracy/auc"]),
    ("results: judge-gold AUC (Claude)",
     r"\( (0\.\d+) for Opus, (0\.\d+) for Sonnet",
     ["judge_vs_gold/per_judge/claude_opus/factual_accuracy/auc",
      "judge_vs_gold/per_judge/claude_sonnet/factual_accuracy/auc"]),
    ("results: DeLong AUC gaps",
     r"over Sonnet \+(0\.\d+)[^)]*\) and over Opus \+(0\.\d+)",
     ["judge_vs_gold/cross_family_test/factual_accuracy/comparisons/gpt52_vs_claude_sonnet/delong/diff",
      "judge_vs_gold/cross_family_test/factual_accuracy/comparisons/gpt52_vs_claude_opus/delong/diff"]),
    # --- results: cluster ---
    ("results: headline span",
     r"pipelines span (0\.\d+) \(P0\) to (0\.\d+) \(P1\)",
     ["headline/per_pattern/base_p0/mean_3judge", "headline/per_pattern/base_p1/mean_3judge"]),
    ("results: judge-robust pair count",
     r"Of the (\d+) pattern pairs, (\d+) are",
     [lambda: 55.0, "pairwise_verified/judge_robust_of_55"]),
    ("results: within-judge Holm counts",
     r"within-judge counts (\d+), (\d+), and (\d+)",
     ["pairwise_verified/within_judge_holm_sig/gpt52",
      "pairwise_verified/within_judge_holm_sig/claude_opus",
      "pairwise_verified/within_judge_holm_sig/claude_sonnet"]),
    ("results: panel cluster gap",
     r"panel reports is (0\.\d+)",
     [f"{HB}/three_judge/cluster_minus_p0/point"]),
    ("results: GPT-5.2 cluster gap",
     r"judge alone puts it at (0\.\d+)",
     [f"{HB}/gpt52/cluster_minus_p0/point"]),
    ("results: Cohen's d per judge",
     r"Cohen's d=(0\.\d+) for Opus and (0\.\d+) for Sonnet against (0\.\d+) for GPT-5\.2",
     ["judge_scale_standardized_gaps/per_judge/claude_opus/cohen_d_paired",
      "judge_scale_standardized_gaps/per_judge/claude_sonnet/cohen_d_paired",
      "judge_scale_standardized_gaps/per_judge/gpt52/cohen_d_paired"]),
    # --- results: capability ---
    ("results: GPT-4o range",
     r"GPT-4o range \( (0\.\d+) \)",
     [lambda: P("headline/per_pattern/base_p1/mean_3judge") - P("headline/per_pattern/base_p0/mean_3judge")]),
    ("results: intra-cluster range",
     r"intra-cluster range \( (0\.\d+) \)",
     [lambda: (lambda ms: max(ms) - min(ms))([P(f"{HB}/three_judge/per_pattern_means/base_p{i}") for i in (1, 4, 5, 6, 7, 8)])]),
    ("limitations: decontaminated P0-vs-P9 (table row, prose only pointers here)",
     r"P0-vs-P9 falls from 0\.231 to (0\.\d+)\s*,\s*and cluster-vs-P9 \(the six-pattern",
     ["contamination/clean_residual_capability/clean_panel3/p0_minus_p9/gap"]),
    ("limitations: decontaminated P4-vs-P9 (table row, full-set + clean-slice)",
     r"P4-vs-P9 & (0\.\d+) & (0\.\d+)",
     ["contamination/clean_residual_capability/full90_panel3_reference/p4_minus_p9/gap",
      "contamination/clean_residual_capability/clean_panel3/p4_minus_p9/gap"]),
    ("results: local capacity effect + CI",
     r"lifts the overall score by \+(0\.\d+) \( 95% CI \[(0\.\d+), (0\.\d+)\]",
     ["frozen_vintage/capacity_axis/paired_diffs/p17_minus_p9/point",
      "frozen_vintage/capacity_axis/paired_diffs/p17_minus_p9/ci95/0",
      "frozen_vintage/capacity_axis/paired_diffs/p17_minus_p9/ci95/1"]),
    # --- results: RL ---
    ("results: RL delta",
     r"by =(0\.\d+) with posterior",
     [lambda: P("headline/per_pattern/base_p10/mean_3judge") - P("headline/per_pattern/base_p9/mean_3judge")]),
    ("results: RL external-benchmark delta",
     r"P10 beats P9 by \+(0\.\d+)",
     ["local_benchmark/tier_7b_external_validation/rl_training_delta_test/delta_p10_minus_p9"]),
    # --- results: best-of-N / disentanglement / length ---
    ("results: naive best-of-N vs cluster",
     r"naive curve reaches (0\.\d+) at four samples against the orchestrated cluster's (0\.\d+)",
     ["best_of_n/curve/4/best_of_k", "best_of_n/cluster_mean"]),
    ("results: pure-noise prediction",
     r"predicts (0\.\d+) at k=4 and (0\.\d+) at k=5",
     ["best_of_n/pure_noise/prediction/4/predicted_best_of_k",
      "best_of_n/pure_noise/prediction/5/predicted_best_of_k"]),
    ("results: decoupled curve k4/k5/cluster",
     r"\( (0\.\d+) at k=4 , (0\.\d+) at k=5 , against the cluster's (0\.\d+) \)",
     ["best_of_n/decoupled/curve/4/best_of_k_decoupled",
      "best_of_n/decoupled/curve/5/best_of_k_decoupled",
      "best_of_n/decoupled/cluster_mean_half_B"]),
    ("results: decoupled level at k=7",
     r"roughly seven samples \( (0\.\d+) \)",
     ["best_of_n/decoupled/curve/7/best_of_k_decoupled"]),
    ("results: disentangle full-budget gap",
     r"beats P0 by \+(0\.\d+) on the overall score \(Wilcoxon p=(0\.\d+)",
     ["disentanglement/p1_arm/unmatched/delta", "disentanglement/p1_arm/unmatched/wilcoxon_p"]),
    ("results: disentangle matched gap",
     r"falls to \+(0\.\d+) and is no longer",
     ["disentanglement/p1_arm/matched/delta"]),
    ("results: clamp effect",
     r"paired by query\) is (-0\.\d+)",
     ["disentanglement/p1_arm/clamp_effect/delta"]),
    ("results: clamp Wilcoxon p",
     r"Wilcoxon p=(0\.\d+) ; sign-flip",
     ["disentanglement/p1_arm/clamp_effect/wilcoxon_p"]),
    ("results: length-adjusted cluster gap",
     r"cluster-vs-P0 gap of (0\.\d+) falls to (-0\.\d+)",
     [f"{HB}/three_judge/cluster_minus_p0/point",
      "length_adjusted_headline/specs/pooled_ols_kwords_vintage_method/three_judge/gaps/cluster_minus_p0/point"]),
    ("results: length-adjusted P0-vs-P9",
     r"P0-vs-P9 falls from (0\.\d+) to (0\.\d+)",
     [f"{HB}/three_judge/p0_minus_p9/point",
      "length_adjusted_headline/specs/pooled_ols_kwords_vintage_method/three_judge/gaps/p0_minus_p9/point"]),
    ("results: length-adjusted cluster-vs-P9",
     r"cluster-vs-P9 from (0\.\d+) to (0\.\d+)",
     [f"{HB}/three_judge/cluster_minus_p9/point",
      "length_adjusted_headline/specs/pooled_ols_kwords_vintage_method/three_judge/gaps/cluster_minus_p9/point"]),
    ("limitations: decontaminated cluster-vs-P9 (distinct pipeline from the length-adjusted one above)",
     r"cluster-vs-P9 \(the six-pattern cluster of[^)]*\) falls from (0\.\d+) to (0\.\d+)",
     ["contamination/clean_residual_capability/full90_panel3_reference/cluster_minus_p9/gap",
      "contamination/clean_residual_capability/clean_panel3/cluster_minus_p9/gap"]),
    ("results: length-adjusted GPT-5.2 gap",
     r"adjusted cluster gap at (-0\.\d+)",
     ["length_adjusted_headline/specs/pooled_ols_kwords_vintage_method/per_judge/gpt52/pooled_ols_kwords/cluster_minus_p0/point"]),
    # --- results: replications / oracle / search ---
    ("results: second-backbone orchestration gain + CI",
     r"bounded but positive \( \+(0\.\d+) , 95% CI \[(0\.\d+),(0\.\d+)\]",
     ["second_backbone_gpt41/contrast_a_bounded_gain/delta",
      "second_backbone_gpt41/contrast_a_bounded_gain/ci95/0",
      "second_backbone_gpt41/contrast_a_bounded_gain/ci95/1"]),
    ("results: second-backbone oracle lift + CI",
     r"\( \+(0\.\d+) , 95% CI \[(0\.\d+),(0\.\d+)\] , n=17",
     ["second_backbone_gpt41/contrast_b_oracle_bottleneck/delta",
      "second_backbone_gpt41/contrast_b_oracle_bottleneck/ci95/0",
      "second_backbone_gpt41/contrast_b_oracle_bottleneck/ci95/1"]),
    ("oracle: panel citation cross-check",
     r"citation quality rises \(Opus \+(0\.\d+)[^)]*Sonnet \+(0\.\d+)",
     ["oracle/panel_cross_check/opus_cluster_dims/citation_quality/delta",
      "oracle/panel_cross_check/sonnet_cluster_dims/citation_quality/delta"]),
    ("oracle: panel factual cross-check",
     r"factual accuracy stays flat \(Opus \+(0\.\d+)[^)]*Sonnet \+(0\.\d+)",
     ["oracle/panel_cross_check/opus_cluster_dims/factual_accuracy/delta",
      "oracle/panel_cross_check/sonnet_cluster_dims/factual_accuracy/delta"]),
    ("oracle: dose factual slope",
     r"factual-accuracy slope of (-0\.\d+)",
     ["e5_dose_response/factual_accuracy_slope/slope"]),
    ("retrieval: search-conditioned cluster coef",
     r"cluster coefficient at \+(0\.\d+)",
     ["search_robustness/cluster_gap_conditioned_on_search/coef"]),
    ("retrieval: per-query search coef",
     r"per-query level gives\s*\$?\+(0\.\d+)",
     ["search_robustness/per_query_clustered_inference/coef"]),
]

n_ok = n_drift = n_missing = 0
drift_lines, missing_lines = [], []
for label, rx, resolvers in ANCHORS:
    m = re.search(rx, PROSE)
    if not m:
        n_missing += 1
        missing_lines.append(f"  NOT FOUND: {label}  (regex: {rx})")
        continue
    for g, res in zip(m.groups(), resolvers):
        printed = float(g)
        d = len(g.split(".")[1]) if "." in g else 0
        try:
            expect = res() if callable(res) else P(res)
        except (KeyError, IndexError, TypeError):
            n_missing += 1
            missing_lines.append(f"  STORE PATH MISSING: {label} ({res})")
            continue
        tol = 0.5 * 10 ** (-d) + 1e-9
        if abs(printed - expect) <= tol:
            n_ok += 1
        else:
            n_drift += 1
            where = res if isinstance(res, str) else "derived"
            drift_lines.append(f"  DRIFT: {label}: prose prints {g}, store {where} = {expect:.4f} (prints as {round(expect, d):g})")

print(f"[reconcile_prose] load-bearing anchors (abstract/contributions/results): "
      f"{n_ok} values OK, {n_drift} DRIFT, {n_missing} not located")
for s in drift_lines:
    print(s)
for s in missing_lines:
    print(s)
if n_drift == 0 and n_missing == 0:
    print("[reconcile_prose] PASS — all anchored load-bearing statistics equal their canonical values at printed precision")
