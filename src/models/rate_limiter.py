"""Simple thread-safe rate limiter (token-bucket style)."""

import time
import threading


class RateLimiter:
    """Rate limiter that enforces a maximum number of calls per minute."""

    def __init__(self, calls_per_minute: float = 60.0):
        self._min_interval = (
            60.0 / max(calls_per_minute, 0.1) if calls_per_minute > 0 else 0.0
        )
        self._last_call = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until the next call is allowed."""
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.time()


# Module-level shared instances used across the codebase.
GEMINI_LIMITER = RateLimiter(calls_per_minute=5)  # Gemini KG extraction: 5 RPM
MISTRAL_LIMITER = RateLimiter(calls_per_minute=1)  # Mistral embedding/rerank: 1 RPM
