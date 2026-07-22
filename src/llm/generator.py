"""LLM generation via any OpenAI-compatible endpoint, with Ollama fallback."""

from typing import Generator

from config.settings import get_settings
from constants.logger import setup_logger
from models.client import ApiClient, OllamaClient
from models.fallback import with_fallback, BackendTag

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


def _build_messages(
    query: str, context: str = "", *, closed_book: bool = False
) -> list[dict]:
    if closed_book:
        return [
            {"role": "system", "content": CLOSED_BOOK_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]


def _api_client() -> ApiClient:
    return ApiClient(
        base_url=settings.generator.base_url,
        api_key=settings.generator.api_key.get_secret_value(),
        timeout=120.0,
    )


def _ollama_client() -> OllamaClient:
    return OllamaClient(base_url=settings.generator.ollama_base_url)


def _api_generate(messages: list[dict]) -> str:
    kwargs = {"temperature": 0.1}
    if settings.generator.max_tokens:
        kwargs["max_tokens"] = settings.generator.max_tokens
    return _api_client().chat(
        messages=messages,
        model=settings.generator.model,
        **kwargs,
    )


def _ollama_generate(messages: list[dict]) -> str:
    return _ollama_client().chat(
        messages=messages,
        model=settings.generator.ollama_model,
        options={"temperature": 0.1},
    )


def _api_stream_generate(messages: list[dict]) -> Generator[str, None, None]:
    kwargs = {"temperature": 0.1}
    if settings.generator.max_tokens:
        kwargs["max_tokens"] = settings.generator.max_tokens
    yield from _api_client().chat_stream(
        messages=messages,
        model=settings.generator.model,
        **kwargs,
    )


def _single_yield(value: str) -> Generator[str, None, None]:
    yield value


def generate(
    query: str,
    context: str = "",
    *,
    stream: bool = True,
    closed_book: bool = False,
) -> str | Generator[str, None, None]:
    """Generate an answer, trying the remote API first then falling back to Ollama.

    Returns a generator (streaming) or a string (non-streaming). The backend
    tag is logged internally.
    """
    messages = _build_messages(query, context, closed_book=closed_book)

    if stream:
        try:
            return _api_stream_generate(messages)
        except Exception as exc:
            LOGGER.warning(
                "Generator API stream failed (%s), falling back to Ollama...", exc
            )
            return _single_yield(_ollama_generate(messages))

    result, _backend = with_fallback(
        _api_generate,
        _ollama_generate,
        "generator",
        fallback_enabled=settings.generator.fallback_enabled,
        messages=messages,
    )
    LOGGER.info(
        "Generated %d words (backend=%s).",
        len(result.split()),
        _backend,
    )
    return result
