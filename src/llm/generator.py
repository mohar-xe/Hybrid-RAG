"""LLM generation via any OpenAI-compatible endpoint."""

from typing import Generator

import httpx

from config.settings import get_settings
from constants.logger import setup_logger

LOGGER = setup_logger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a precise research assistant. Answer the question using ONLY the provided context.

Rules:
- Cite sources using bracket notation [1], [2], etc.
- If the context doesn't contain enough information, say so explicitly.
- Do not invent information not present in the context.
- Be concise but thorough."""

# Closed-book: no retrieval context is supplied. Used by the evaluation's
# `direct` baseline to measure the model's parametric knowledge alone.
CLOSED_BOOK_SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer the question directly from your own knowledge.

Rules:
- Give the most likely answer even if you are uncertain.
- Be concise: answer in as few words as the question allows.
- Do not ask for clarification and do not mention missing context."""


def generate(
    query: str,
    context: str = "",
    *,
    stream: bool = False,
    closed_book: bool = False,
) -> str | Generator[str, None, None]:
    """Generate an answer from an OpenAI-compatible endpoint.

    When ``closed_book`` is True the retrieval ``context`` is ignored and a
    closed-book system prompt is used (the evaluation's ``direct`` baseline).
    """
    if closed_book:
        messages = [
            {"role": "system", "content": CLOSED_BOOK_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

    if stream:
        return _stream_generate(messages)

    response = httpx.post(
        f"{settings.generator.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.generator.api_key.get_secret_value()}"},
        json={
            "model": settings.generator.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1024,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]
    LOGGER.info(f"Generated {len(answer.split())} words.")
    return answer


def _stream_generate(messages: list[dict]) -> Generator[str, None, None]:
    import json

    with httpx.stream(
        "POST",
        f"{settings.generator.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.generator.api_key.get_secret_value()}"},
        json={
            "model": settings.generator.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1024,
            "stream": True,
        },
        timeout=120.0,
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0].get("delta", {})
                if content := delta.get("content"):
                    yield content