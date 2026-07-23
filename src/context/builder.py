"""Context builder — merges retrieved chunks into a numbered, citation-ready prompt."""

from dataclasses import dataclass

from config.settings import get_settings
from constants.logger import setup_logger

LOGGER = setup_logger(__name__)
settings = get_settings()


@dataclass
class Citation:
    ref_id: int
    source_id: str
    chunk_id: str
    text_preview: str


def build_context(
    chunks: list,
    graph_facts: str = "",
    max_tokens: int | None = None,
) -> tuple[str, list[Citation]]:
    """Pack retrieved ``chunks`` (+ optional ``graph_facts``) into a numbered,
    citation-ready context string under an approximate token budget.

    ``max_tokens`` defaults to ``CONTEXT__MAX_TOKENS`` (see ``ContextSettings``)
    so the budget is configured centrally like every other tunable in the
    project. Token counts are estimated from word counts via
    ``CONTEXT__TOKEN_RATIO`` (no tokenizer dependency at this layer).
    """
    max_tokens = max_tokens if max_tokens is not None else settings.context.max_tokens
    token_ratio = settings.context.token_ratio

    citations: list[Citation] = []
    context_parts: list[str] = []
    token_count = 0.0

    # Reserve budget for the graph-knowledge block so the total stays under cap.
    graph_tokens = len(graph_facts.split()) * token_ratio if graph_facts else 0.0
    # Guarantee chunks get at least 30% of the budget even if graph_facts are large;
    # otherwise a huge knowledge block can squeeze out all document evidence.
    min_chunk_budget = max_tokens * 0.3
    chunk_budget = max(min_chunk_budget, max_tokens - graph_tokens)

    for i, chunk in enumerate(chunks, start=1):
        approx_tokens = len(chunk.text.split()) * token_ratio
        if token_count + approx_tokens > chunk_budget:
            break

        context_parts.append(f"[{i}] (source: {chunk.source_id})\n{chunk.text}")
        citations.append(Citation(
            ref_id=i,
            source_id=chunk.source_id,
            chunk_id=chunk.chunk_id,
            text_preview=chunk.text[:100],
        ))
        token_count += approx_tokens

    if graph_facts:
        context_parts.append(f"\n[Graph Knowledge]\n{graph_facts}")
        token_count += graph_tokens

    context = "\n\n".join(context_parts)
    LOGGER.info(
        f"Built context: {len(citations)} chunks, ~{int(token_count)}/{max_tokens} tokens."
    )
    return context, citations
