"""Public documentation and redaction contract tests."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from deep_research.evaluation.multi_judge import _stable_seed as stable_judge_seed
from deep_research.evaluation.pairwise_arena import _stable_seed as stable_pairwise_seed


def test_public_query_manifest_records_redacted_rows_without_private_sentinels():
    payload = json.loads(Path("data/eval_queries_v2.json").read_text())
    redacted = [
        row
        for row in payload["queries"]
        if row.get("metadata", {}).get("public_redaction")
    ]

    assert len(redacted) == 2
    assert all("anonymized" in row["metadata"]["public_redaction"].lower() for row in redacted)
    assert "two draco-derived" in Path("NOTICE").read_text().lower()
    assert "Two DRACO-derived rows" in Path("DATA_LICENSES.md").read_text()
    assert "public_redaction" in Path("data/README.md").read_text()
    assert "Two DRACO-derived rows" in Path("data/analysis/DATA_DICTIONARY.md").read_text()
    assert "two DRACO-derived rows" in Path("data/analysis/build_manifest.json").read_text()


def test_public_redacted_query_identifiers_removed_from_shipped_data():
    import pandas as pd

    forbidden = ("Evert", "Calderon", "Mesa Group", "Miro board", "Creditily", "creditily")
    text_paths = [
        Path("data/eval_queries_v2.json"),
        Path("data/all_90_queries.json"),
        Path("repro/reference/REFERENCE_MANIFEST.json"),
        Path("data/analysis/build_manifest.json"),
    ]
    for path in text_paths:
        text = path.read_text(errors="ignore")
        assert not any(marker in text for marker in forbidden), path

    for path in (
        Path("data/analysis/df_queries.parquet"),
        Path("data/analysis/df_citations.parquet"),
        Path("data/analysis/df_verdicts.parquet"),
    ):
        df = pd.read_parquet(path)
        for column in df.columns:
            values = df[column].astype("string")
            assert not any(
                values.str.contains(marker, regex=False, na=False).any()
                for marker in forbidden
            ), f"{path}:{column}"


def test_protocol_doc_describes_expanded_public_scope_without_raw_artifacts():
    doc = Path("docs/evaluation_protocol.md").read_text()
    human_doc = Path("docs/human_evaluation_protocol.md").read_text()

    assert "Public Export Note" in doc
    assert "reusable pattern and" in doc and "evaluation modules" in doc
    assert "Paper A rebuild package" in doc
    assert "Final 9-dimension rubric-v2 scoring" in doc
    assert "Earlier planning notes used a 7-dimension shorthand" in doc
    assert "raw generated report forests" in doc
    assert "raw judge-verdict packet" in doc
    assert "not raw evaluator packets" in human_doc


def test_script_catalog_covers_every_top_level_public_script():
    catalog_path = Path("repro/SCRIPT_CATALOG.csv")
    rows = list(csv.DictReader(catalog_path.read_text().splitlines()))
    expected = {
        path.name
        for path in Path("scripts").iterdir()
        if path.is_file() and path.suffix in {".py", ".sh", ".js"}
    }
    actual = {row["script"] for row in rows}

    assert actual == expected
    assert "compile_portfolio_pdf.py" not in actual

    allowed_statuses = {
        "prefer public CLI wrapper",
        "supported public helper",
        "optional non-default workflow",
        "optional external download",
        "optional GPU/local-model workflow",
        "requires non-public raw artifacts",
        "historical analysis helper",
        "internal worker helper",
    }
    for row in rows:
        assert row["public_status"] in allowed_statuses
        assert row["family"].strip()
        assert row["required_inputs_or_services"].strip()
        assert row["expected_outputs"].strip()
        assert row["summary"].strip()


def test_script_docs_point_to_maintained_catalog():
    script_readme = Path("scripts/README.md").read_text()
    repro_map = Path("repro/PAPER_A_REPRO_MAP.md").read_text()

    assert "repro/SCRIPT_CATALOG.csv" in script_readme
    assert "Direct script entry points intended for public use" in script_readme
    assert "verify_headline_numbers.py" in script_readme
    assert "publish_huggingface.py" in script_readme
    assert "repro/SCRIPT_CATALOG.csv" in repro_map
    assert "one-row-per-script" in repro_map


def test_huggingface_publish_path_is_documented_without_token_arguments():
    guide = Path("docs/huggingface_release.md").read_text()
    card = Path("repro/HUGGINGFACE_DATASET_CARD.md").read_text()
    script = Path("scripts/publish_huggingface.py").read_text()
    manifest = json.loads(Path("PUBLIC_MANIFEST.json").read_text())

    assert "HF_TOKEN" in guide
    assert "--dry-run" in guide
    assert "--allow-dirty" in guide
    assert "Deep Research Paper Supplement" in card
    assert "license: other" in card
    assert "mixed-license public data" in card
    assert 'add_argument("--token' not in script
    assert 'os.environ.get("HF_TOKEN")' in script
    assert "docs/huggingface_release.md" in manifest["required_paths"]
    assert "repro/HUGGINGFACE_DATASET_CARD.md" in manifest["required_paths"]
    assert "scripts/publish_huggingface.py" in manifest["required_paths"]


def test_check_api_help_is_safe_and_does_not_run_provider_calls():
    result = subprocess.run(
        [sys.executable, "scripts/check_api.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "API Connectivity Check" not in result.stdout
    assert "Testing" not in result.stdout


def test_check_api_skips_local_models_by_default_without_provider_calls():
    result = subprocess.run(
        [sys.executable, "scripts/check_api.py", "--model", "Qwen2.5-7B-Instruct"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "No API-backed models selected." in result.stdout
    assert "Testing" not in result.stdout


def test_paper_artifact_index_links_manuscript_assets_and_coverage():
    index = Path("repro/PAPER_A_ARTIFACT_INDEX.md").read_text()
    paper_readme = Path("paper_rebuild/paper_a_bounded_returns/README.md").read_text()
    readme = Path("README.md").read_text()
    coverage = Path("data/analysis/coverage_report.md").read_text()

    for required in (
        "paper_rebuild/paper_a_bounded_returns/main.tex",
        "figures/fig1_money.pdf",
        "figures/fig_oracle.pdf",
        "tables/tab_headline_means.tex",
        "tables/tab_bestofn_decoupled.tex",
        "data/analysis/coverage_report.md",
        "analysis/build_isoquant_claimtype.py",
        "analysis/staging/isoquant_claimtype.json",
    ):
        assert required in index
    assert "Bounded Returns to Orchestration" in readme
    assert "papers/paper_a_bounded_returns/main.pdf" in readme
    assert "Exclusion reason" in readme
    assert "Executive summary" in coverage
    assert "Missing-file sections" in coverage
    assert "PAPER_A_ARTIFACT_INDEX.md" in paper_readme
    assert "PAPER_A_ARTIFACT_INDEX.md" in readme


def test_isoquant_claim_block_has_public_builder_not_excluded_scratchpad():
    canonical = json.loads(
        Path("paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json").read_text()
    )
    manifest = json.loads(Path("PUBLIC_MANIFEST.json").read_text())
    block = canonical["capability_isoquant_and_claimtype"]
    meta = block["_meta"]
    public_script = "paper_rebuild/paper_a_bounded_returns/analysis/build_isoquant_claimtype.py"
    staging = "paper_rebuild/paper_a_bounded_returns/analysis/staging/isoquant_claimtype.json"
    text = json.dumps(block)

    assert meta["script"] == public_script
    assert public_script in manifest["required_paths"]
    assert staging in manifest["required_paths"]
    assert "scratchpad" not in text
    assert "manual-rerun" not in text

    result = subprocess.run(
        [sys.executable, public_script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["canonical_matches_public_staging"] is True


def test_data_docs_describe_included_analysis_tables_without_absolute_paths():
    top_dictionary = Path("data/DATA_DICTIONARY.md").read_text()
    analysis_dictionary = Path("data/analysis/DATA_DICTIONARY.md").read_text()
    data_readme = Path("data/README.md").read_text()
    data_licenses = Path("DATA_LICENSES.md").read_text()
    references_bib = Path("paper_rebuild/paper_a_bounded_returns/references.bib").read_text()

    assert "Compact derived parquet/metadata tables" in top_dictionary
    assert "not absolute local paths" in analysis_dictionary
    assert "df_citations.parquet" in analysis_dictionary
    assert "df_e14_oracle_verdicts.parquet" in analysis_dictionary
    assert "simple`, `moderate`, or `complex" in analysis_dictionary
    assert "Public redaction pass" in analysis_dictionary
    assert "Two DRACO-derived rows" in analysis_dictionary
    assert "retained only" in analysis_dictionary
    assert "Anthropic API directly" in analysis_dictionary
    assert "No Downloads Required" in data_readme
    assert "does not regenerate the selected 90-query Paper A manifest" in data_readme
    assert "data/analysis/*.parquet" in data_licenses
    assert "final scrubbed manuscript" in data_licenses
    assert "paper_rebuild/" in data_licenses
    assert "Absolute path to the `.md` report" not in analysis_dictionary
    assert "COLM/EMNLP" not in analysis_dictionary
    assert "not yet web-verified" not in references_bib
    assert "PHASE3_FIX_LIST" not in references_bib


def test_archival_exclusion_reconciler_fails_politely_without_private_packets():
    result = subprocess.run(
        [sys.executable, "scripts/reconcile_exclusions.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "requires non-public archival Claude-Code quarantine packet files" in result.stderr
    assert "Traceback" not in result.stderr


def test_public_evaluation_seeds_are_process_stable():
    assert stable_judge_seed("judge", 1, "q1") == 1300634655
    assert stable_pairwise_seed("q1", "system_a", "system_b") == 1325890070


def test_public_config_and_types_restore_reusable_source_contract():
    from deep_research.config import DEFAULT_MODEL, JUDGE_MODEL, MODELS, PROJECT_ROOT
    from deep_research.types import Document, ResearchReport, SourceType

    assert PROJECT_ROOT == Path.cwd()
    assert DEFAULT_MODEL in MODELS
    assert JUDGE_MODEL in MODELS
    assert Document(title="Example", source_type=SourceType.WEB).model_dump()["title"] == "Example"
    assert ResearchReport(query="Q", title="T").full_text().startswith("# T")
