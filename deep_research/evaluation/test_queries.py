"""Test queries with expected answer elements for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TestQuery:
    """A test query with expected answer elements for scoring."""
    id: str
    query: str
    difficulty: str  # "simple", "moderate", "complex"
    expected_elements: List[str] = field(default_factory=list)
    expected_sources: List[str] = field(default_factory=list)
    description: str = ""


TEST_QUERIES: List[TestQuery] = [
    TestQuery(
        id="q1_bert_vs_gpt",
        query="Compare and contrast the BERT and GPT architectures, including their training objectives, strengths, weaknesses, and typical use cases.",
        difficulty="simple",
        description="Simple factual comparison. Any system should get this right.",
        expected_elements=[
            "BERT uses bidirectional/masked language modeling",
            "GPT uses autoregressive/causal language modeling",
            "BERT encoder-only architecture",
            "GPT decoder-only architecture",
            "BERT better for classification/NLU tasks",
            "GPT better for generation tasks",
            "Attention mechanism in both",
            "Pre-training and fine-tuning paradigm",
            "BERT: Devlin et al. 2018/2019",
            "GPT: Radford et al. / OpenAI",
            "Transformer architecture as foundation",
            "Transfer learning in NLP",
        ],
        expected_sources=["arxiv.org", "aclanthology.org", "papers.nips.cc"],
    ),
    TestQuery(
        id="q2_rag_vs_finetuning",
        query="What are the tradeoffs between retrieval-augmented generation (RAG) and fine-tuning for reducing hallucination in large language models? Include specific benchmark results where available.",
        difficulty="moderate",
        description="Moderate. Requires multi-source synthesis with specific benchmark numbers.",
        expected_elements=[
            "RAG retrieves external documents at inference time",
            "Fine-tuning adjusts model weights on domain data",
            "RAG provides attribution/citation capability",
            "Fine-tuning can embed knowledge in parameters",
            "RAG requires vector database infrastructure",
            "Fine-tuning requires labeled training data",
            "Hallucination rates comparison",
            "Cost comparison (inference vs training)",
            "Latency tradeoffs",
            "Hybrid approaches (RAG + fine-tuning)",
            "Specific benchmarks (TruthfulQA, HaluEval, etc.)",
            "Knowledge freshness/update frequency",
        ],
        expected_sources=["arxiv.org", "semantic scholar"],
    ),
    TestQuery(
        id="q3_single_vs_multi_agent",
        query="What are the tradeoffs between single-agent and multi-agent approaches for automated deep research? Compare specific systems and their performance metrics.",
        difficulty="complex",
        description="Complex, contested. Must present balanced view with specific metrics.",
        expected_elements=[
            "Single-agent: simpler architecture, lower coordination overhead",
            "Multi-agent: specialization, parallel processing",
            "Coordination challenges in multi-agent systems",
            "Cost implications of multiple LLM calls",
            "Quality vs efficiency tradeoffs",
            "Specific systems (STORM, AutoSurvey, PaperQA2, etc.)",
            "Evaluation metrics (coverage, accuracy, coherence)",
            "Error propagation in pipelines",
            "Scalability considerations",
            "Human-in-the-loop vs fully automated",
        ],
        expected_sources=["arxiv.org"],
    ),
    TestQuery(
        id="q4_paperqa_storm_autosurvey",
        query="Compare PaperQA2, STORM, and AutoSurvey as automated research systems. What are their architectures, strengths, limitations, and performance on benchmarks?",
        difficulty="complex",
        description="Breadth. Three systems with specific metrics.",
        expected_elements=[
            "PaperQA2: retrieval-focused, citation verification",
            "STORM: perspective-driven, Wikipedia-style articles",
            "AutoSurvey: survey paper generation",
            "PaperQA2 performance on LitQA benchmark",
            "STORM article quality metrics",
            "Architecture differences",
            "Source handling approaches",
            "Citation accuracy comparison",
            "Cost and efficiency differences",
            "Limitations of each system",
            "Use case recommendations",
            "Evaluation methodology differences",
        ],
        expected_sources=["arxiv.org"],
    ),
    TestQuery(
        id="q5_lost_in_middle",
        query="Explain the 'lost in the middle' effect in large language models. What causes it, what are the specific performance numbers, and what mitigation strategies have been proposed?",
        difficulty="complex",
        description="Depth. Technical topic requiring specific numbers and mitigation strategies.",
        expected_elements=[
            "U-shaped attention pattern",
            "Performance degrades for information in middle of context",
            "Liu et al. 2023 original paper",
            "Specific performance numbers (accuracy drops)",
            "Position bias in attention mechanism",
            "Context window length effects",
            "Mitigation: instruction tuning",
            "Mitigation: retrieval ordering strategies",
            "Mitigation: chunking approaches",
            "Impact on RAG systems",
            "Benchmark setup (multi-document QA)",
            "Model-specific variations (GPT, Claude, etc.)",
        ],
        expected_sources=["arxiv.org"],
    ),
]


def get_query(query_id: str) -> TestQuery:
    """Get a test query by ID."""
    for q in TEST_QUERIES:
        if q.id == query_id:
            return q
    raise ValueError(f"Unknown query ID: {query_id}")


def get_all_queries() -> List[TestQuery]:
    return TEST_QUERIES
