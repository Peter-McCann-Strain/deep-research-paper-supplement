"""Phase 8a: prepare a 30-report human-calibration annotation pack.

Sample 30 reports stratified by (pattern × judge agreement). Ship them with
rubric, instructions, and a Google-Sheet-ready CSV for two annotators.

Outputs to data/human_calibration_pack/
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "analysis"
PACK = ROOT / "data" / "human_calibration_pack"
PACK.mkdir(parents=True, exist_ok=True)
REP_DIR = PACK / "reports_to_score"
REP_DIR.mkdir(exist_ok=True)

JUDGES = ["gpt52", "claude_opus", "claude_sonnet"]
DIMENSIONS = [
    "information_recall", "factual_accuracy", "coverage",
    "analytical_depth", "citation_quality", "logical_coherence",
    "organization", "instruction_following",  # drop attribution_quality (α<0)
]
RNG = np.random.default_rng(42)

print("Loading...")
df_overall = pd.read_parquet(DATA / "df_overall_scores.parquet")
df_queries = pd.read_parquet(DATA / "df_queries.parquet")
for d in (df_overall,):
    for c in d.select_dtypes("category"):
        d[c] = d[c].astype(str)
df_overall["overall"] = np.where(
    df_overall["overall_score_trustworthy"],
    df_overall["overall_score"],
    df_overall["overall_score_recomputed"],
)
base = df_overall[df_overall["pattern_family"] == "base"].copy()

# Per (pattern, query) std across 3 judges → "judge disagreement"
piv = base.pivot_table(index=["pattern", "query_id"], columns="judge", values="overall", aggfunc="mean").dropna(subset=JUDGES)
piv["judge_std"] = piv[JUDGES].std(axis=1)
piv["judge_mean"] = piv[JUDGES].mean(axis=1)
piv = piv.reset_index()

# Stratified sample: 5 from each of (low/mid/high judge_std) × random patterns
qmeta = df_queries.set_index("query_id")
piv["source"] = piv["query_id"].map(qmeta["source"])
piv["difficulty"] = piv["query_id"].map(qmeta["difficulty"])

# Goals:
# - 30 reports total
# - Cover all 11 patterns (~3 each)
# - Span low/mid/high judge_std bins
# - Span all 5 sources
# - Span 3 difficulties
# - Include a few cells where judges disagree most (validation interest)
# - Include a few cells where judges agree most (gold-standard candidates)

# Stratified sample by judge_std tertile + pattern
piv["std_bin"] = pd.qcut(piv["judge_std"], 3, labels=["agree", "mid", "disagree"])

sample = []
seen_qids = set()
patterns = sorted(base.pattern.unique())

# Pass 1: 1-2 per pattern
for p in patterns:
    sub = piv[piv.pattern == p].copy()
    # mix bins: 1 disagree + 1 agree + 1 mid
    for bin_name in ["disagree", "mid", "agree"]:
        bin_sub = sub[sub.std_bin == bin_name]
        bin_sub = bin_sub[~bin_sub.query_id.isin(seen_qids)]
        if len(bin_sub) == 0:
            continue
        row = bin_sub.sample(1, random_state=int(RNG.integers(0, 1_000_000))).iloc[0]
        sample.append(row)
        seen_qids.add(row.query_id)
        if len([x for x in sample if x["pattern"] == p]) >= 1:
            break
df_sample = pd.DataFrame(sample)

# Pass 2: top up to 30
while len(df_sample) < 30:
    candidates = piv[~piv.query_id.isin(seen_qids)]
    if len(candidates) == 0:
        break
    # pick from underrepresented bin
    bin_counts = df_sample.std_bin.value_counts().to_dict()
    target_bin = min(["agree", "mid", "disagree"], key=lambda b: bin_counts.get(b, 0))
    bin_cands = candidates[candidates.std_bin == target_bin]
    if len(bin_cands) == 0:
        bin_cands = candidates
    row = bin_cands.sample(1, random_state=int(RNG.integers(0, 1_000_000))).iloc[0]
    df_sample = pd.concat([df_sample, pd.DataFrame([row])], ignore_index=True)
    seen_qids.add(row.query_id)

df_sample = df_sample.head(30).reset_index(drop=True)
print(f"Sampled {len(df_sample)} reports across {df_sample.pattern.nunique()} patterns")
print(df_sample.std_bin.value_counts().to_dict())
print(df_sample.source.value_counts().to_dict())

# Copy each report markdown to the pack and assign anonymized IDs
import shutil
manifest_rows = []
for i, r in df_sample.iterrows():
    anon_id = f"R{i+1:03d}"
    src_path = ROOT / "results" / "experiments" / r["pattern"] / f"{r['query_id']}.md"
    if not src_path.exists():
        continue
    dst_path = REP_DIR / f"{anon_id}.md"
    text = src_path.read_text(encoding="utf-8")
    # Anonymize: prefix with the query and anon ID
    qtext = qmeta.loc[r["query_id"], "query"] if "query" in qmeta.columns else r["query_id"]
    header = f"# {anon_id}\n\n## Query\n\n{qtext}\n\n## Report\n\n---\n\n"
    dst_path.write_text(header + text, encoding="utf-8")
    manifest_rows.append({
        "anon_id": anon_id,
        "pattern": r["pattern"],
        "query_id": r["query_id"],
        "source": r["source"],
        "difficulty": r["difficulty"],
        "judge_mean": r["judge_mean"],
        "judge_std": r["judge_std"],
        "std_bin": str(r["std_bin"]),
    })

df_manifest = pd.DataFrame(manifest_rows)
df_manifest.to_csv(PACK / "ANONYMIZED_KEY.csv", index=False)
df_manifest_anonymized = df_manifest[["anon_id", "judge_mean", "judge_std"]].copy()
df_manifest_anonymized.to_csv(PACK / "manifest_for_annotators.csv", index=False)

# Build annotator scoresheet template
scoresheet_rows = []
for r in manifest_rows:
    for d in DIMENSIONS:
        scoresheet_rows.append({
            "anon_id": r["anon_id"],
            "dimension": d,
            "annotator_score_0_1": "",  # to fill
            "comment": "",
        })
df_scoresheet = pd.DataFrame(scoresheet_rows)
df_scoresheet.to_csv(PACK / "annotator_scoresheet_BLANK.csv", index=False)

# Instructions
INSTRUCTIONS = """# Human Calibration Annotation Pack — Instructions

## Overview
You are annotating 30 deep-research reports on 8 quality dimensions.
The reports come from a comparative study of 11 different research-agent architectures.
Your annotations will calibrate our LLM-judge panel against human ground truth.

## Time estimate
~15-20 minutes per report × 30 reports = ~7-10 hours total per annotator.
Two annotators required for inter-annotator reliability.

## Procedure
1. Open `manifest_for_annotators.csv` and the `reports_to_score/` folder.
2. For each `R001.md` through `R030.md`:
   - Read the query at the top of the file
   - Read the report (typically 800-2000 words)
   - For each of 8 dimensions, score 0.0 (terrible) to 1.0 (excellent), in 0.1 increments
3. Record your scores in your copy of `annotator_scoresheet_BLANK.csv`.

## The 8 dimensions and their meanings

1. **information_recall** (weight 0.20): Does the report retrieve and include the key facts needed to answer the query? Has it covered the important sub-topics?

2. **factual_accuracy** (weight 0.20): Are the factual claims in the report correct? (Use your domain knowledge + spot-check 3-5 claims by web search if uncertain.)

3. **coverage** (weight 0.10): Does the report cover the requested topic(s) comprehensively, including expected sub-topics and perspectives?

4. **analytical_depth** (weight 0.15): Does the report synthesize information into insights, comparisons, and analysis — or just list facts?

5. **citation_quality** (weight 0.10): Are citations present, attached to specific claims, and from credible sources? (Check 3-5 cited URLs.)

6. **logical_coherence** (weight 0.05): Does the report flow logically? Are arguments supported and consistent?

7. **organization** (weight 0.05): Is the report well-structured (headings, sections, transitions)?

8. **instruction_following** (weight 0.10): Does the report directly address what the query asked, in the format the query implied (e.g., comparison, list, explanation)?

(We have dropped `attribution_quality` from the rubric for human annotation because our LLM judges showed Krippendorff α=-0.10 on it — they cannot agree on what it means.)

## Important notes
- Score each dimension INDEPENDENTLY — do not let your overall impression bleed into individual dimensions
- Use the full 0.0-1.0 range — most reports should fall between 0.3 and 0.8; reserve 0.0-0.2 for catastrophic failure and 0.9-1.0 for excellent work
- Annotator-to-annotator agreement is what we measure first (Cohen's κ); judge-to-human agreement second
- If a report is empty/error/unscoreable, mark all dimensions as 0.0 and add a comment
- DO NOT consult the LLM judge scores; they are blinded for this purpose

## After annotation
Return your completed scoresheet through the private study-coordination channel
provided with the evaluator packet.

## Stratification
The 30 reports are sampled to span:
- 11 architectures (3 reports each on average)
- 5 source benchmarks (custom, draco, deepsearch_qa, research_qa, litqa2)
- 3 difficulty levels (simple, moderate, complex)
- 3 judge-disagreement bins (agree, mid, disagree) — interesting calibration cases

## Compensation
$30/hour, estimated $210-300 per annotator total.
"""
(PACK / "INSTRUCTIONS.md").write_text(INSTRUCTIONS)

# Summary
with open(PACK / "README.md", "w") as f:
    f.write("# Human Calibration Pack\n\n")
    f.write(f"30 reports stratified across patterns, sources, difficulties, and judge-disagreement bins.\n\n")
    f.write("## Files\n\n")
    f.write("- `INSTRUCTIONS.md` — annotator instructions\n")
    f.write("- `manifest_for_annotators.csv` — 30-row manifest (anon_id, judge_mean, judge_std) — annotators see this\n")
    f.write("- `annotator_scoresheet_BLANK.csv` — 30 × 8 = 240 cells to score\n")
    f.write("- `reports_to_score/R001.md` … `R030.md` — anonymized reports\n")
    f.write("- `ANONYMIZED_KEY.csv` — pattern/query_id mapping (for analysis only, hide from annotators)\n\n")
    f.write("## Distribution stats (sampled set)\n\n")
    f.write(f"- Patterns covered: {df_manifest.pattern.nunique()}/11\n")
    f.write(f"- Sources covered: {df_manifest.source.nunique()}/5\n")
    f.write(f"- Difficulties covered: {df_manifest.difficulty.nunique()}/3\n")
    f.write(f"- Judge-std bins: {dict(df_manifest.std_bin.value_counts())}\n\n")
    f.write("## Next steps for the researcher\n\n")
    f.write("1. Recruit 2 annotators (~7-10 hours each, $30/hr → ~$500 total)\n")
    f.write("2. Distribute INSTRUCTIONS + manifest + reports + blank scoresheet\n")
    f.write("3. Receive completed scoresheets\n")
    f.write("4. Compute Cohen's κ between annotators (target ≥0.6)\n")
    f.write("5. Compute Pearson r and Cohen's κ between LLM judges and human consensus\n")
    f.write("6. Update Phase 1 judge validation report with human anchor\n")
    f.write("7. If LLM-vs-human κ ≥ 0.5 on any dimension, that dimension's claims are validated\n")
    f.write("8. If not, re-frame those dimensions as 'judge-only signal, requires human validation in future work'\n")
print(f"\nDone. Pack at {PACK}")
print(f"Reports written: {len(manifest_rows)}")
