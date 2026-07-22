"""Generic fallback logic: try primary, then fallback on failure."""

from typing import Any, Callable, Literal

from constants.logger import setup_logger

LOGGER = setup_logger(__name__)

BackendTag = Literal["api", "ollama", "hf"]


def with_fallback(
    primary_fn: Callable[..., Any],
    fallback_fn: Callable[..., Any],
    component: str,
    *,
    fallback_enabled: bool = True,
    primary_tag: BackendTag = "api",
    fallback_tag: BackendTag = "ollama",
    **kwargs: Any,
) -> tuple[Any, BackendTag]:
    """Try ``primary_fn(**kwargs)``; on failure log and try ``fallback_fn(**kwargs)``.

    Returns:
        ``(result, tag)`` where ``tag`` is ``primary_tag`` or ``fallback_tag``.
    """
    try:
        result = primary_fn(**kwargs)
        return result, primary_tag
    except Exception as exc:
        if not fallback_enabled:
            raise
        LOGGER.warning(
            "%s: primary backend (%s) failed: %s. Trying fallback (%s)...",
            component,
            primary_tag,
            exc,
            fallback_tag,
        )
        try:
            result = fallback_fn(**kwargs)
            return result, fallback_tag
        except Exception as exc2:
            LOGGER.error(
                "%s: fallback backend (%s) also failed: %s",
                component,
                fallback_tag,
                exc2,
            )
            raise
