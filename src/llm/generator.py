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


def generate(query: str, context: str, stream: bool = False) -> str | Generator[str, None, None]:
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