"""Generation adapter for fast_eval + an LLM-call tally (rerank excluded).

fast_eval treats the provider key as paid, so it generates **per query**
(interactive, real latency), not bundled. This module wraps the project's
``llm.generator`` generation primitives and threads a small ``CallTracker`` through
them so the run reports how many paid LLM calls it made.

Tally convention (matches the user's request):
  * **LLM calls** = generation + entity extraction (DeepSeek v4-flash, the paid key).
  * **Rerank** is recorded separately — it hits the Jina reranker, not the paid LLM —
    and is *not* added to the LLM total.
  * **Embeddings** (Mistral) are recorded separately too.
"""

from __future__ import annotations

import random
import threading
import time
from collections import defaultdict

# Exhaustively caught on a single call? No — a paid key must ride out transient
# rate-limit bursts instead of degrading to `[]` / Ollama. Retry with jittered
# exponential backoff on transient HTTP errors (429 Too Many Requests, 5xx, and
# network/connect errors). The pool's interleaving gives natural spacing; the
# backoff handles a genuine sustained throttle.
_TRANSIENT = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "Connection",
    "ConnectError",
    "Timeout",
    "timed out",
)


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(token in text for token in _TRANSIENT)


def _retry_call(fn, *, attempts: int = 6, base_delay: float = 1.5):
    """Call ``fn`` with jittered exponential backoff on transient failures.

    Raises the last exception if every attempt is exhausted. Non-transient
    errors (bad request, auth, schema) raise immediately — retrying those is
    pointless and would mask a real misconfig.
    """
    delay = base_delay
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if not _is_transient(exc) or attempt == attempts - 1:
                raise
            time.sleep(delay * (0.5 + random.random()))
            delay *= 2


class CallTracker:
    """Thread-safe counter + timings grouped by call *kind*.

    Kinds: ``generation``, ``entity``, ``embedding``, ``rerank``. ``rerank`` and
    ``embedding`` are tracked for visibility but excluded from the ``llm_total``.
    """

    LLM_KINDS = ("generation", "entity")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[str, int] = defaultdict(int)
        self._times: dict[str, list[float]] = defaultdict(list)

    def record(self, kind: str, ms: float = 0.0) -> None:
        with self._lock:
            self._calls[kind] += 1
            self._times[kind].append(ms)

    def calls(self, kind: str) -> int:
        with self._lock:
            return self._calls[kind]

    def llm_total(self) -> int:
        return sum(self._calls[k] for k in self.LLM_KINDS)

    def total_ms(self, kind: str) -> float:
        with self._lock:
            return sum(self._times[kind])

    def summary(self) -> dict:
        with self._lock:
            calls = {k: v for k, v in self._calls.items()}
            return {
                "generation_calls": calls.get("generation", 0),
                "entity_calls": calls.get("entity", 0),
                "embedding_calls": calls.get("embedding", 0),
                "rerank_calls": calls.get("rerank", 0),
                "llm_calls_total": self.llm_total(),
            }


def generate_answer(
    question: str,
    context: str = "",
    *,
    closed_book: bool = False,
    tracker: CallTracker | None = None,
) -> tuple[str, float]:
    """Generate one answer (interactive per query) and record a generation call.

    Returns ``(answer, generation_ms)``. Uses the project generator, which reads the
    provider/model/key from the live environment (set by ``fast_eval.run`` from
    ``fast_eval.config`` before this module is imported).
    """
    from llm.generator import generate

    t0 = time.perf_counter()

    def _call():
        return generate(
            question,
            context,
            stream=False,
            closed_book=closed_book,
            eval_mode=True,
        )

    try:
        answer = _retry_call(_call)
    except Exception:
        answer = "None"  # exhausted retries: degrade, never crash the cell
    gen_ms = (time.perf_counter() - t0) * 1000.0
    if tracker is not None:
        tracker.record("generation", gen_ms)
    return (str(answer) if answer else "None"), gen_ms


def extract_query_entities(
    question: str, *, tracker: CallTracker | None = None
) -> list[str]:
    """Extract query entities once per query (graph seeding), recorded as an entity call."""
    from graph.entity_extraction import extract_query_entities as _extract

    t0 = time.perf_counter()
    try:
        entities = _retry_call(lambda: _extract(question))
    except Exception:
        entities = []  # exhausted retries: degrade to no graph seeding, never crash
    ms = (time.perf_counter() - t0) * 1000.0
    if tracker is not None:
        tracker.record("entity", ms)
    return entities
