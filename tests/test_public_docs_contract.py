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


def test_protocol_doc_does_not_claim_missing_historical_modules_are_shipped():
    doc = Path("docs/evaluation_protocol.md").read_text()
    human_doc = Path("docs/human_evaluation_protocol.md").read_text()

    assert "Public Export Note" in doc
    assert "public" in doc and "does not ship the" in doc and "archived execution engine" in doc
    forbidden_claims = [
        "Implementation is " + "provided by",
        "`deep_research/" + "evaluation/`",
        "`deep_research/" + "ablation/framework.py`",
        "`execution_" + "pipeline.py`",
        "`judge_" + "pipeline.py`",
        "`retrieval_" + "eval.py`",
        "`rubric_" + "v2.py`",
        "`multi_" + "judge.py`",
        "`citation_" + "verifier.py`",
    ]
    for claim in forbidden_claims:
        assert claim not in doc
        assert claim not in human_doc
