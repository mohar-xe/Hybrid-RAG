"""LLM generation via any OpenAI-compatible endpoint, with Ollama fallback."""

import json
from typing import Generator

from config.settings import get_settings
from constants.logger import setup_logger
from models.client import ApiClient, OllamaClient
from models.fallback import with_fallback, BackendTag
from models.rate_limiter import RateLimiter

LOGGER = setup_logger(__name__)
settings = get_settings()

# Paces remote generation requests (``GENERATOR__API_RATE_LIMIT``). The eval
# harness bundles many (question, context) pairs per call; this limiter keeps
# per-minute token flow under the provider's TPM ceiling.
_GENERATOR_LIMITER = RateLimiter(calls_per_minute=settings.generator.api_rate_limit)

SYSTEM_PROMPT = """You are a precise research assistant. Answer the question using ONLY the provided context.

Rules:
- Cite sources using bracket notation [1], [2], etc.
- If the context doesn't contain enough information, say so explicitly.
- Do not invent information not present in the context.
- Be concise but thorough.
- Output ONLY the final answer. Do NOT show any reasoning, chain-of-thought, or internal thinking — just the answer itself with citations."""

# Short-answer prompt for evaluation (HotpotQA / SQuAD style). The model must
# output the exact answer span — typically a name, number, or short phrase —
# without explanation, citation markers, or full-sentence framing. This is
# critical for exact-match and F1 scoring that evaluates against gold spans.
EVAL_SYSTEM_PROMPT = """You are a precise extractive QA system. Answer the question using ONLY the provided context.

Rules:
- Output ONLY the exact answer span — one or a few words, never a full sentence.
- Do NOT add explanations, citations, or any other text.
- If the context does not contain the answer, output "None".
- Never use your own knowledge."""

# Closed-book: no retrieval context is supplied. Used by the evaluation's
# `direct` baseline to measure the model's parametric knowledge alone.
CLOSED_BOOK_SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer the question directly from your own knowledge.

Rules:
- Output ONLY the exact answer span — one or a few words.
- Do NOT add any explanations, citations, or full sentences.
- If you do not know the answer, output "None"."""

# Bundled prompt for the eval harness's batched generation: many
# (question, context) pairs per call, mixed RAG + closed-book, answered in one
# JSON object. Answers stay extractive (exact spans) for exact-match/F1 scoring.
BATCH_SYSTEM_PROMPT = """You are a precise extractive QA system. Answer EACH numbered question below.

Each question may be followed by a "Context:" block. If a context is present, answer using ONLY it. If no context is given, answer from your own knowledge.

Rules:
- Output ONLY a JSON object mapping question indices (as strings: "0", "1", ...) to answers.
- Each answer must be an exact span — one or a few words. Never full sentences, explanations, or citations.
- For a question WITH a context: if the context does not contain the answer, output "None" (never use your own knowledge).
- For a question WITHOUT a context: if you do not know the answer, output "None"."""


def _build_messages(
    query: str, context: str = "", *, closed_book: bool = False, eval_mode: bool = False
) -> list[dict]:
    if closed_book:
        return [
            {"role": "system", "content": CLOSED_BOOK_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
    prompt = EVAL_SYSTEM_PROMPT if eval_mode else SYSTEM_PROMPT
    return [
        {"role": "system", "content": prompt},
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
    eval_mode: bool = False,
) -> str | Generator[str, None, None]:
    """Generate an answer, trying the remote API first then falling back to Ollama.

    Returns a generator (streaming) or a string (non-streaming). The backend
    tag is logged internally. Pass ``eval_mode=True`` for extractive short-answer
    evaluation (HotpotQA/SQuAD) — the model outputs only the answer span.
    """
    messages = _build_messages(query, context, closed_book=closed_book, eval_mode=eval_mode)

    if stream:
        try:
            return _api_stream_generate(messages)
        except Exception as exc:
            if not settings.generator.fallback_enabled:
                raise
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


def _parse_batch_answers(content: str, count: int) -> list[str]:
    """Parse a bundled generation response into one answer per item.

    Expects ``{"0": "ans", "1": "ans", ...}``. Tolerates code fences; missing
    entries become ``"None"`` (never raises — a garbled batch degrades to
    ``None`` answers rather than aborting the eval).
    """
    content = (content or "").strip()
    if content.startswith("```"):
        content = (
            content.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
    answers: list[str] = ["None"] * count
    if not content:
        return answers
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        LOGGER.warning("Bundled generation response unparseable: %s", exc)
        return answers
    if not isinstance(raw, dict):
        LOGGER.warning(
            "Bundled generation response not an object (%s); marking batch None.",
            type(raw).__name__,
        )
        return answers
    for key, value in raw.items():
        try:
            idx = int(key)
        except (ValueError, TypeError):
            continue
        if 0 <= idx < count and isinstance(value, str):
            answers[idx] = value.strip() or "None"
    return answers


def generate_batch(
    items: list[tuple[str, str]],
    *,
    batch_size: int = 40,
) -> list[str]:
    """Answer many (question, context) pairs with one bundled API call per batch.

    ``items`` is a list of ``(question, context)`` tuples; an empty/blank
    context means closed-book (answer from parametric knowledge). Returns one
    answer per item, in input order. Extractive answers (exact spans) suitable
    for exact-match/F1 scoring. Paced by ``GENERATOR__API_RATE_LIMIT``; a failed
    batch degrades to ``"None"`` answers without aborting the run.
    """
    if not items:
        return []
    count = len(items)
    results: list[str | None] = [None] * count

    for start in range(0, count, batch_size):
        batch = items[start : start + batch_size]
        blocks = []
        for i, (question, context) in enumerate(batch):
            if context.strip():
                blocks.append(
                    f"---Q{start + i}---\nContext:\n{context}\n\nQuestion: {question}"
                )
            else:
                blocks.append(f"---Q{start + i}---\nQuestion: {question} (no context)")
        user_content = "\n\n".join(blocks)

        kwargs: dict = {"temperature": 0.0}
        if settings.generator.max_tokens:
            kwargs["max_tokens"] = settings.generator.max_tokens
        else:
            kwargs["max_tokens"] = max(2048, len(batch) * 96)
        if "gemini" in settings.generator.model.lower() or "deepseek" in settings.generator.model.lower():
            kwargs["thinking"] = {"type": "disabled"}

        _GENERATOR_LIMITER.acquire()
        try:
            content = _api_client().chat(
                messages=[
                    {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                model=settings.generator.model,
                **kwargs,
            )
        except Exception as exc:
            LOGGER.error("Bundled generation batch %d failed: %s", start, exc)
            for j in range(len(batch)):
                results[start + j] = "None"
            continue

        parsed = _parse_batch_answers(content, len(batch))
        for j, answer in enumerate(parsed):
            results[start + j] = answer
        LOGGER.info(
            "Generated batch %d-%d (%d answers).",
            start,
            start + len(batch) - 1,
            len(batch),
        )

    return [r if r is not None else "None" for r in results]
