"""Stage 2: Simulate expert conversations between perspective pairs.

Conversations use two-step source extractions as context instead of
retrieved/reranked chunks.  Each participant sees the relevant extractions
formatted as numbered evidence blocks with key findings and data points.
"""

from __future__ import annotations

import asyncio
import itertools
from typing import Any, Dict, List, Tuple

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.tools.source_extractor import format_extractions_as_evidence, SourceExtraction
from deep_research.types import Perspective
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

# Number of back-and-forth turns per conversation
CONVERSATION_TURNS = 3

INTERVIEWER_SYSTEM = """You are a skilled research interviewer with expertise in {interviewer_name}.
Your focus areas: {interviewer_focus}.

You are interviewing an expert ({expert_name}) about the research topic below.
Ask probing, specific questions that draw out the expert's unique perspective.
Challenge assumptions, ask for evidence, and explore nuances.
Reference the provided source material when relevant.

Research topic: {query}

Background source material:
{context}
"""

EXPERT_SYSTEM = """You are {expert_name}: {expert_description}
Your focus areas: {expert_focus}.

You are being interviewed about the research topic below.
Draw on your specialized knowledge and the provided source material to give detailed,
evidence-based answers. Be specific — cite data, methods, studies, and concrete examples.
When you are uncertain, say so and explain what evidence would be needed.
Acknowledge when you agree or disagree with established views.

Research topic: {query}

Background source material:
{context}
"""

INTERVIEWER_OPENING_PROMPT = """Based on your perspective as {interviewer_name} and the background
source material provided in your instructions above, formulate your first probing question
for {expert_name}.

Ask a question that explores how the expert's perspective on this topic differs from or
complements your own. Be specific and reference the background material."""

INTERVIEWER_FOLLOWUP_PROMPT = """Based on the expert's response, ask a follow-up question that:
- Probes deeper into specific claims they made
- Asks for evidence or methodology details
- Explores implications or limitations
- Connects to your own perspective's focus areas

Use the evidence provided in your instructions above."""

EXPERT_RESPONSE_PROMPT = """Answer the interviewer's question based on your expertise and the
source material provided in your instructions above. Be detailed and specific.

Provide a thorough response with concrete evidence, examples, and nuances."""


def _format_context(extractions: List[SourceExtraction]) -> str:
    """Format source extractions as context for the conversation."""
    if not extractions:
        return "No specific source material available."
    return format_extractions_as_evidence(extractions)


def _build_perspective_pairs(
    perspectives: List[Perspective],
) -> List[Tuple[Perspective, Perspective]]:
    """Generate all unique perspective pairs for conversations.

    Each pair has (interviewer, expert). We create pairs such that every
    perspective gets to be interviewed by at least one other perspective.
    For N perspectives, this yields N*(N-1)/2 pairs.
    To keep costs manageable, we cap at 10 pairs.
    """
    all_pairs = list(itertools.combinations(perspectives, 2))
    # For each pair (A, B), A interviews B. We already get all combos.
    if len(all_pairs) > 10:
        # Prioritize: ensure each perspective appears at least once as expert
        selected: List[Tuple[Perspective, Perspective]] = []
        expert_seen: set = set()
        for a, b in all_pairs:
            if b.name not in expert_seen:
                selected.append((a, b))
                expert_seen.add(b.name)
            if a.name not in expert_seen:
                selected.append((b, a))
                expert_seen.add(a.name)
            if len(selected) >= 10:
                break
        # Fill remaining slots
        for pair in all_pairs:
            if pair not in selected and len(selected) < 10:
                selected.append(pair)
        return selected
    return all_pairs


async def _run_single_conversation(
    interviewer: Perspective,
    expert: Perspective,
    query: str,
    context_extractions: List[SourceExtraction],
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
    n_turns: int = CONVERSATION_TURNS,
) -> Dict[str, Any]:
    """Run a multi-turn conversation between an interviewer and an expert.

    Args:
        interviewer: The perspective acting as interviewer.
        expert: The perspective acting as the interviewed expert.
        query: The research query.
        context_extractions: Source extractions relevant to these perspectives.
        llm: LLM caller instance.
        model: Model to use for conversation simulation.
        n_turns: Number of question-answer turns.

    Returns:
        Dict containing the conversation transcript and metadata.
    """
    context_text = _format_context(context_extractions)

    # Build system prompts (evidence injected once here, not repeated each turn)
    interviewer_sys = INTERVIEWER_SYSTEM.format(
        interviewer_name=interviewer.name,
        interviewer_focus=", ".join(interviewer.focus_areas),
        expert_name=expert.name,
        query=query,
        context=context_text,
    )
    expert_sys = EXPERT_SYSTEM.format(
        expert_name=expert.name,
        expert_description=expert.description,
        expert_focus=", ".join(expert.focus_areas),
        query=query,
        context=context_text,
    )

    transcript: List[Dict[str, str]] = []
    interviewer_messages: List[Dict[str, str]] = [
        {"role": "system", "content": interviewer_sys},
    ]
    expert_messages: List[Dict[str, str]] = [
        {"role": "system", "content": expert_sys},
    ]

    for turn in range(n_turns):
        # Interviewer asks a question
        if turn == 0:
            interviewer_messages.append({
                "role": "user",
                "content": INTERVIEWER_OPENING_PROMPT.format(
                    interviewer_name=interviewer.name,
                    expert_name=expert.name,
                ),
            })
        else:
            interviewer_messages.append({
                "role": "user",
                "content": INTERVIEWER_FOLLOWUP_PROMPT,
            })

        question = await llm.complete_messages(
            interviewer_messages,
            model=model,
            temperature=0.5,
            max_tokens=1024,
        )

        transcript.append({
            "role": "interviewer",
            "perspective": interviewer.name,
            "turn": turn,
            "content": question,
        })

        # Add question to both message histories
        interviewer_messages.append({"role": "assistant", "content": question})
        expert_messages.append({"role": "user", "content": question})

        # Expert responds
        expert_messages.append({
            "role": "user",
            "content": EXPERT_RESPONSE_PROMPT,
        })

        answer = await llm.complete_messages(
            expert_messages,
            model=model,
            temperature=0.4,
            max_tokens=1536,
        )

        transcript.append({
            "role": "expert",
            "perspective": expert.name,
            "turn": turn,
            "content": answer,
        })

        # Add answer to both message histories
        expert_messages.append({"role": "assistant", "content": answer})
        interviewer_messages.append({"role": "user", "content": f"Expert's response:\n{answer}"})

        log.debug("conversation_turn", interviewer=interviewer.name,
                  expert=expert.name, turn=turn)

    log.info("conversation_complete", interviewer=interviewer.name,
             expert=expert.name, turns=n_turns)

    return {
        "interviewer": interviewer.name,
        "expert": expert.name,
        "transcript": transcript,
        "turn_count": n_turns,
    }


async def run_all_conversations(
    perspectives: List[Perspective],
    query: str,
    perspective_extractions: Dict[str, List[SourceExtraction]],
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
    n_turns: int = CONVERSATION_TURNS,
) -> List[Dict[str, Any]]:
    """Run simulated conversations for all perspective pairs in parallel.

    Args:
        perspectives: All discovered perspectives.
        query: The research query.
        perspective_extractions: Mapping of perspective name to its source
            extractions (SourceExtraction objects with structured fields).
        llm: LLM caller instance.
        model: Model for conversation simulation.
        n_turns: Turns per conversation.

    Returns:
        List of conversation dicts, each containing the transcript and metadata.
    """
    pairs = _build_perspective_pairs(perspectives)
    log.info("conversation_pairs", count=len(pairs),
             pairs=[(a.name, b.name) for a, b in pairs])

    async def _run_pair(interviewer: Perspective, expert: Perspective) -> Dict[str, Any]:
        # Combine extractions from both perspectives
        combined: List[SourceExtraction] = []
        for name in [interviewer.name, expert.name]:
            combined.extend(perspective_extractions.get(name, []))

        # Deduplicate by doc_id, keeping the first occurrence
        seen_ids: set = set()
        deduped: List[SourceExtraction] = []
        for e in combined:
            eid = e.doc_id or e.url
            if eid not in seen_ids:
                seen_ids.add(eid)
                deduped.append(e)

        return await _run_single_conversation(
            interviewer=interviewer,
            expert=expert,
            query=query,
            context_extractions=deduped,
            llm=llm,
            model=model,
            n_turns=n_turns,
        )

    # Run conversations with limited concurrency to avoid PTU saturation.
    # Each conversation makes 2*n_turns LLM calls, so 3 concurrent conversations
    # = ~6 concurrent LLM calls, well within the global rate limiter's capacity.
    max_concurrent_conversations = 3
    sem = asyncio.Semaphore(max_concurrent_conversations)

    async def _run_pair_with_limit(interviewer: Perspective, expert: Perspective) -> Dict[str, Any]:
        async with sem:
            return await _run_pair(interviewer, expert)

    tasks = [_run_pair_with_limit(interviewer, expert) for interviewer, expert in pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    conversations: List[Dict[str, Any]] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            interviewer, expert = pairs[i]
            log.warning("conversation_failed",
                        interviewer=interviewer.name,
                        expert=expert.name,
                        error=str(result))
            continue
        conversations.append(result)

    log.info("all_conversations_complete", total=len(conversations),
             failed=len(results) - len(conversations))
    return conversations


def extract_all_conversation_text(conversations: List[Dict[str, Any]]) -> str:
    """Concatenate all conversation transcripts into a single text for downstream processing."""
    parts = []
    for conv in conversations:
        parts.append(f"\n=== Conversation: {conv['interviewer']} interviewing {conv['expert']} ===\n")
        for turn in conv.get("transcript", []):
            role_label = f"[{turn['perspective']} ({turn['role']})]"
            parts.append(f"{role_label}\n{turn['content']}\n")
    return "\n".join(parts)
