"""Public documentation and redaction contract tests."""

from __future__ import annotations

import json
from pathlib import Path


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


def test_protocol_doc_describes_expanded_public_scope_without_raw_artifacts():
    doc = Path("docs/evaluation_protocol.md").read_text()
    human_doc = Path("docs/human_evaluation_protocol.md").read_text()

    assert "Public Export Note" in doc
    assert "reusable pattern and" in doc and "evaluation modules" in doc
    assert "Paper A rebuild package" in doc
    assert "raw generated report forests" in doc
    assert "raw judge-verdict packet" in doc
    assert "not raw evaluator packets" in human_doc
