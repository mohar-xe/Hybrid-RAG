"""Context builder — merges retrieved chunks into a numbered, citation-ready prompt."""

from dataclasses import dataclass

from constants.logger import setup_logger

LOGGER = setup_logger(__name__)


@dataclass
class Citation:
    ref_id: int
    source_id: str
    chunk_id: str
    text_preview: str


def build_context(
    chunks: list,
    graph_facts: str = "",
    max_tokens: int = 3000,
) -> tuple[str, list[Citation]]:
    citations: list[Citation] = []
    context_parts: list[str] = []
    token_count = 0

    for i, chunk in enumerate(chunks, start=1):
        approx_tokens = len(chunk.text.split()) * 1.3
        if token_count + approx_tokens > max_tokens:
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

    context = "\n\n".join(context_parts)
    LOGGER.info(f"Built context: {len(citations)} chunks, ~{int(token_count)} tokens.")
    return context, citations