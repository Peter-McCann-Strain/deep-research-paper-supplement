"""FreshWiki / STORM benchmark integration.

Evaluates Wikipedia-style article generation quality.
100 high-quality Wikipedia articles, chosen to avoid training data contamination.

Dataset: https://huggingface.co/datasets/EchoShao8899/FreshWiki
Paper: Shao et al., NAACL 2024 (arXiv:2402.14207)

Scoring dimensions:
- Interest: Is the article engaging?
- Organization: Is it well-structured?
- Relevance: Is content relevant to the topic?
- Coverage: Does it cover all aspects?
- Verifiability: Are claims verifiable with citations?
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from deep_research.benchmarks.base import (
    BenchmarkDataset,
    BenchmarkLoadError,
    BenchmarkQuery,
    BenchmarkResult,
)
from deep_research.config import DATA_DIR
from deep_research.types import ResearchReport

log = structlog.get_logger()

_CACHE_DIR = DATA_DIR / "benchmarks" / "freshwiki"


class FreshWikiBenchmark(BenchmarkDataset):
    """FreshWiki benchmark: Wikipedia article generation quality."""

    @property
    def name(self) -> str:
        return "FreshWiki"

    @property
    def description(self) -> str:
        return "Wikipedia-style article generation quality (STORM/NAACL 2024)"

    async def load(self, max_queries: int = 0) -> List[BenchmarkQuery]:
        """Load FreshWiki dataset."""
        cache_path = _CACHE_DIR / "freshwiki_queries.json"
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            log.info("freshwiki_cache_hit")
            data = json.loads(cache_path.read_text())
            queries = [BenchmarkQuery(**q) for q in data]
        else:
            queries = await self._download()
            if queries:
                cache_path.write_text(json.dumps([_query_to_dict(q) for q in queries], indent=2))

        if max_queries > 0:
            queries = queries[:max_queries]

        log.info("freshwiki_loaded", queries=len(queries))
        return queries

    async def _download(self) -> List[BenchmarkQuery]:
        """Download FreshWiki from HuggingFace.

        The dataset has inconsistent column schemas across JSON files.
        Most files: {title, url, summary, content(list of sections), references, ...}
        Some files: {text} only.

        content is a list: [{section_title, section_content: [{sentence, refs}]}]
        We reconstruct article text from this structured format.
        """
        try:
            from huggingface_hub import HfApi, hf_hub_download
            import json as _json

            api = HfApi()
            repo_files = api.list_repo_files("EchoShao8899/FreshWiki", repo_type="dataset")
            json_files = [f for f in repo_files if f.startswith("json/") and f.endswith(".json")]

            queries = []
            for i, filepath in enumerate(json_files):
                try:
                    local_path = hf_hub_download(
                        "EchoShao8899/FreshWiki",
                        filename=filepath,
                        repo_type="dataset",
                    )
                    with open(local_path) as fh:
                        data = _json.load(fh)

                    # Handle structured format (most files)
                    if "title" in data and "content" in data:
                        title = data["title"].replace("_", " ")
                        content_sections = data.get("content", [])
                        headings = []
                        article_text = data.get("summary", "") + "\n\n"

                        for section in content_sections:
                            sec_title = section.get("section_title", "")
                            if sec_title:
                                headings.append(sec_title)
                                article_text += f"## {sec_title}\n\n"
                            for item in section.get("section_content", []):
                                article_text += item.get("sentence", "") + " "
                            article_text += "\n\n"

                    # Handle text-only format
                    elif "text" in data:
                        text = data["text"]
                        title = filepath.split("/")[-1].replace(".json", "").replace("_", " ")
                        article_text = text
                        headings = re.findall(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)

                    else:
                        continue

                    queries.append(
                        BenchmarkQuery(
                            id=f"fw_{i:04d}",
                            query=f"Write a comprehensive article about: {title}",
                            domain="wikipedia",
                            difficulty="moderate",
                            reference_answer=article_text[:50000],
                            rubric={
                                "reference_headings": headings,
                                "reference_word_count": len(article_text.split()),
                            },
                            metadata={
                                "title": title,
                                "section_count": len(headings),
                                "url": data.get("url", ""),
                            },
                        )
                    )

                except Exception as e:
                    log.warning("freshwiki_file_failed", file=filepath, error=str(e))
                    continue

            if not queries:
                raise BenchmarkLoadError("FreshWiki download returned no usable JSON records")
            return queries

        except ImportError as exc:
            log.error("datasets_or_hf_hub_not_installed")
            raise BenchmarkLoadError(
                "FreshWiki requires huggingface_hub; use the public API workflow if benchmark downloads are not needed"
            ) from exc
        except Exception as e:
            log.error("freshwiki_download_failed", error=str(e))
            raise BenchmarkLoadError(
                "FreshWiki download failed; check network access and dataset availability"
            ) from e

    async def score(
        self,
        query: BenchmarkQuery,
        report: ResearchReport,
    ) -> BenchmarkResult:
        """Score using STORM's evaluation dimensions (1-5 scale, normalized to 0-1)."""
        report_text = report.full_text()
        scores: Dict[str, float] = {}

        # 1. Coverage: How much of the reference content is covered
        if query.reference_answer:
            scores["coverage"] = self._score_coverage(report_text, query.reference_answer)

        # 2. Organization: Structure quality
        scores["organization"] = self._score_organization(report, query)

        # 3. Relevance: Topic relevance
        scores["relevance"] = self._score_relevance(report_text, query)

        # 4. Verifiability: Citation presence and quality
        scores["verifiability"] = self._score_verifiability(report)

        # 5. Interest: Depth and detail
        scores["interest"] = self._score_interest(report)

        # 6. Heading recall: How many reference headings are covered
        scores["heading_recall"] = self._score_heading_recall(report_text, query)

        # Overall: equal weights across STORM dimensions
        weights = {
            "coverage": 0.25,
            "organization": 0.15,
            "relevance": 0.15,
            "verifiability": 0.20,
            "interest": 0.10,
            "heading_recall": 0.15,
        }
        overall = sum(scores.get(k, 0) * w for k, w in weights.items())

        return BenchmarkResult(
            benchmark_name=self.name,
            pattern_name=report.pattern_name,
            query_id=query.id,
            scores=scores,
            overall_score=overall,
        )

    def _score_coverage(self, report_text: str, reference: str) -> float:
        """Score how much reference content is covered."""
        ref_lower = reference.lower()
        report_lower = report_text.lower()

        # Extract meaningful sentences from reference
        sentences = [s.strip() for s in ref_lower.split(".") if len(s.strip()) > 30][
            :50
        ]  # Cap at 50 sentences
        if not sentences:
            return 0.0

        covered = 0
        for sentence in sentences:
            terms = [t for t in sentence.split() if len(t) > 4][:6]
            if terms:
                found = sum(1 for t in terms if t in report_lower)
                if found >= max(1, len(terms) // 3):
                    covered += 1

        return covered / len(sentences)

    def _score_organization(self, report: ResearchReport, query: BenchmarkQuery) -> float:
        """Score structural organization."""
        score = 0.0

        # Has title
        if report.title and report.title != report.query:
            score += 0.15

        # Has abstract
        if report.abstract:
            score += 0.15

        # Has multiple sections
        n_sections = len(report.sections)
        if n_sections >= 5:
            score += 0.30
        elif n_sections >= 3:
            score += 0.20
        elif n_sections >= 1:
            score += 0.10

        # Sections have content
        if report.sections:
            avg_len = sum(len(s.content) for s in report.sections) / n_sections
            if avg_len > 500:
                score += 0.20
            elif avg_len > 200:
                score += 0.10

        # Section variety (not all same length)
        if n_sections >= 3:
            lengths = [len(s.content) for s in report.sections]
            if max(lengths) > 2 * min(lengths):
                score += 0.10

        # Has references section
        text = report.full_text()
        if "references" in text.lower() or "## References" in text:
            score += 0.10

        return min(1.0, score)

    def _score_relevance(self, report_text: str, query: BenchmarkQuery) -> float:
        """Score topical relevance."""
        title = query.metadata.get("title", query.query)
        title_terms = [t.lower() for t in title.split() if len(t) > 3]
        if not title_terms:
            return 0.5

        report_lower = report_text.lower()
        found = sum(1 for t in title_terms if t in report_lower)
        return found / len(title_terms)

    def _score_verifiability(self, report: ResearchReport) -> float:
        """Score citation presence and quality."""
        text = report.full_text()
        inline_refs = set(re.findall(r"\[\d+\]", text))

        score = 0.0
        # Has inline citations
        if inline_refs:
            score += 0.3
        # Good number of citations
        n_refs = len(inline_refs)
        if n_refs >= 10:
            score += 0.3
        elif n_refs >= 5:
            score += 0.2
        elif n_refs >= 1:
            score += 0.1

        # Citations have URLs
        if report.citations:
            with_url = sum(1 for c in report.citations if c.source_url)
            score += 0.2 * (with_url / len(report.citations))

        # Citations are diverse sources
        if report.citations:
            unique_domains = set()
            for c in report.citations:
                if c.source_url:
                    parts = c.source_url.split("/")
                    if len(parts) > 2:
                        unique_domains.add(parts[2])
            score += min(0.2, len(unique_domains) * 0.04)

        return min(1.0, score)

    def _score_interest(self, report: ResearchReport) -> float:
        """Score depth and engagement."""
        text = report.full_text()
        words = len(text.split())

        score = 0.0
        # Minimum length
        if words >= 2000:
            score += 0.3
        elif words >= 1000:
            score += 0.2

        # Has specific data (numbers, dates, names)
        numbers = len(re.findall(r"\b\d+\.?\d*%?\b", text))
        score += min(0.3, numbers * 0.01)

        # Has examples or case studies
        for marker in ["for example", "such as", "case study", "specifically"]:
            if marker in text.lower():
                score += 0.1
                break

        # Multiple sections with depth
        if report.sections:
            deep_sections = sum(1 for s in report.sections if len(s.content) > 300)
            score += min(0.3, deep_sections * 0.06)

        return min(1.0, score)

    def _score_heading_recall(self, report_text: str, query: BenchmarkQuery) -> float:
        """Score how many reference article headings are covered."""
        ref_headings = query.rubric.get("reference_headings", [])
        if not ref_headings:
            return 0.5

        report_lower = report_text.lower()
        covered = 0
        for heading in ref_headings:
            # Check if heading topic appears in report
            h_terms = [t.lower() for t in heading.split() if len(t) > 3]
            if h_terms:
                found = sum(1 for t in h_terms if t in report_lower)
                if found >= max(1, len(h_terms) // 2):
                    covered += 1

        return covered / len(ref_headings)


def _query_to_dict(q: BenchmarkQuery) -> Dict[str, Any]:
    return {
        "id": q.id,
        "query": q.query,
        "domain": q.domain,
        "difficulty": q.difficulty,
        "rubric": q.rubric,
        "reference_answer": q.reference_answer,
        "expected_citations": q.expected_citations,
        "metadata": q.metadata,
    }
