#!/usr/bin/env python3
"""E7 step 1: Prepare DR-Judge-7B training data from df_verdicts + reports.

Reads the 158k criterion verdicts from `data/analysis/df_verdicts.parquet`
and the corresponding reports from `results/experiments/{pattern}/{qid}.md`,
then produces SFT examples in JSONL format:

    {
      "messages": [
        {"role": "system", "content": "<rubric criterion + scoring instructions>"},
        {"role": "user", "content": "<query + report excerpt>"},
        {"role": "assistant", "content": "{\"satisfied\": true, \"reasoning\": \"...\"}"}
      ],
      "metadata": {"pattern": "...", "query_id": "...", "dimension": "..."}
    }

Strategy:
- Aggregate to majority-vote consensus across 3 judges (target = mode of `satisfied`).
- When judges disagree (split 1-2 in favor of satisfied), keep both polarities so
  the model learns disagreement-aware verdicts. We tag these as `is_disputed=True`.
- Train/val/test split at the QUERY level to avoid leakage (90 queries → 72/9/9).
- Cap report excerpts at 6,000 tokens-equivalent (~24,000 chars) to fit Qwen2.5
  context comfortably.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("data/dr_judge_training")
SOURCE_VERDICTS = Path("data/analysis/df_verdicts.parquet")
REPORTS_BASE = Path("results/experiments")
QUERIES_PATH = Path("data/eval_queries_v2.json")

REPORT_CHAR_CAP = 24_000
RANDOM_SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.80, 0.10  # rest is test


JUDGE_INSTR = """You are a research-report evaluator. Given a research query, a report, and a single rubric criterion, output ONLY a JSON object with two fields:
  "satisfied": true | false  (true if the report clearly satisfies the criterion)
  "reasoning": "<one-sentence justification>"

Be strict: partial or vague coverage is NOT_SATISFIED."""


def load_query_text(queries_path: Path = QUERIES_PATH) -> dict[str, str]:
    data = json.loads(queries_path.read_text())
    return {q["id"]: q["query"] for q in data["queries"]}


def load_report(pattern: str, qid: str) -> str | None:
    p = REPORTS_BASE / pattern / f"{qid}.md"
    if not p.exists():
        return None
    text = p.read_text()
    if len(text) > REPORT_CHAR_CAP:
        text = text[:REPORT_CHAR_CAP] + "\n\n[... report truncated for evaluation ...]"
    return text


def aggregate_verdicts(df: pd.DataFrame) -> pd.DataFrame:
    """Roll up per-judge verdicts to consensus per (pattern, qid, criterion_id).

    Adds columns:
      `consensus`: True if >50% of judges marked satisfied
      `is_disputed`: True if judges disagree (1/3 or 2/3 split)
      `n_judges`: number of judges that scored this criterion
    """
    df = df.dropna(subset=["satisfied"]).copy()
    df = df[df["satisfied_is_known"]]
    grouped = df.groupby(["pattern", "query_id", "criterion_id"], as_index=False)
    rows = []
    for (pat, qid, crit_id), sub in grouped:
        n = len(sub)
        n_sat = sub["satisfied"].sum()
        consensus = (n_sat / n) > 0.5
        is_disputed = 0 < n_sat < n
        # Pick the "best" reasoning - prefer GPT-5.2's, fallback to longest
        if "gpt52" in sub["judge"].values:
            reasoning = sub.loc[sub["judge"] == "gpt52", "reasoning"].iloc[0]
            evidence = sub.loc[sub["judge"] == "gpt52", "evidence"].iloc[0]
        else:
            reasoning = sub["reasoning"].astype(str).iloc[sub["reasoning"].str.len().argmax()]
            evidence = sub["evidence"].astype(str).iloc[sub["evidence"].str.len().argmax()]
        rows.append({
            "pattern": pat, "query_id": qid, "criterion_id": crit_id,
            "criterion": sub["criterion"].iloc[0],
            "dimension": sub["dimension"].iloc[0],
            "consensus": bool(consensus),
            "is_disputed": bool(is_disputed),
            "n_judges": int(n),
            "n_satisfied": int(n_sat),
            "reasoning": reasoning,
            "evidence": evidence,
        })
    return pd.DataFrame(rows)


def make_example(query: str, report: str, criterion: str, dimension: str,
                 consensus: bool, reasoning: str) -> dict:
    """Build one SFT example."""
    user_msg = f"""Research query:
{query}

Report:
{report}

Rubric criterion (dimension: {dimension}):
{criterion}

Output JSON only."""
    assistant_msg = json.dumps({
        "satisfied": consensus,
        "reasoning": reasoning if reasoning else (
            "The report directly addresses this criterion." if consensus
            else "The report does not adequately address this criterion."
        ),
    })
    return {
        "messages": [
            {"role": "system", "content": JUDGE_INSTR},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    df = pd.read_parquet(SOURCE_VERDICTS)
    print(f"Loaded {len(df):,} verdict rows from {SOURCE_VERDICTS}")
    df = df[df["pattern_family"] == "base"]
    print(f"  -> {len(df):,} base-pattern verdict rows")

    consensus = aggregate_verdicts(df)
    print(f"  -> {len(consensus):,} consensus rows")
    print(f"     disputed: {consensus['is_disputed'].sum():,} ({100*consensus['is_disputed'].mean():.1f}%)")
    print(f"     satisfied rate: {100*consensus['consensus'].mean():.1f}%")

    queries = load_query_text()
    all_qids = sorted(set(consensus["query_id"]))
    rng.shuffle(all_qids)
    n_train = int(len(all_qids) * TRAIN_FRAC)
    n_val = int(len(all_qids) * VAL_FRAC)
    train_qids = set(all_qids[:n_train])
    val_qids = set(all_qids[n_train:n_train + n_val])
    test_qids = set(all_qids[n_train + n_val:])
    print(f"  Query split: train={len(train_qids)} val={len(val_qids)} test={len(test_qids)}")

    examples_by_split = {"train": [], "val": [], "test": []}
    skipped = 0
    for _, row in consensus.iterrows():
        report = load_report(row["pattern"], row["query_id"])
        if not report or len(report) < 200:
            skipped += 1
            continue
        query = queries.get(row["query_id"])
        if not query:
            skipped += 1
            continue
        ex = make_example(query, report, row["criterion"], row["dimension"],
                          row["consensus"], row["reasoning"])
        ex["metadata"] = {
            "pattern": row["pattern"], "query_id": row["query_id"],
            "criterion_id": row["criterion_id"], "dimension": row["dimension"],
            "is_disputed": row["is_disputed"], "n_judges": row["n_judges"],
        }
        if row["query_id"] in train_qids:
            examples_by_split["train"].append(ex)
        elif row["query_id"] in val_qids:
            examples_by_split["val"].append(ex)
        else:
            examples_by_split["test"].append(ex)

    print(f"  Skipped (missing report or query): {skipped:,}")
    for split, exs in examples_by_split.items():
        out_path = OUT_DIR / f"{split}.jsonl"
        with out_path.open("w") as f:
            for ex in exs:
                f.write(json.dumps(ex) + "\n")
        print(f"  Wrote {len(exs):,} {split} examples -> {out_path}")

    # Save split metadata
    (OUT_DIR / "split_manifest.json").write_text(json.dumps({
        "train_qids": sorted(train_qids),
        "val_qids": sorted(val_qids),
        "test_qids": sorted(test_qids),
        "random_seed": RANDOM_SEED,
        "report_char_cap": REPORT_CHAR_CAP,
    }, indent=2))


if __name__ == "__main__":
    main()
