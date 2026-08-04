"""Small, fail-open Langfuse observability helpers."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from langfuse import get_client

logger = logging.getLogger(__name__)

_initialised = False


def tracing_enabled() -> bool:
    """Return whether Langfuse tracing is configured and enabled."""
    configured = bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    )

    return configured


def initialise_observability() -> None:
    """Initialise Langfuse when tracing is configured."""
    global _initialised
    if _initialised or not tracing_enabled():
        return

    try:
        get_client()
        _initialised = True
        logger.info("Langfuse observability enabled")
    except Exception:
        logger.exception("Langfuse initialisation failed; tracing is disabled")


@contextmanager
def observation(
    *,
    name: str,
    as_type: str = "span",
    input: Any | None = None,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[Any | None, None, None]:
    """Create a current Langfuse observation, or yield ``None`` when disabled."""
    if not tracing_enabled():
        yield None
        return

    try:
        context = get_client().start_as_current_observation(
            name=name,
            as_type=as_type,
            input=input,
            model=model,
            metadata=metadata,
        )
    except Exception:
        logger.exception("Could not start Langfuse observation '%s'", name)
        yield None
        return

    with context as current:
        yield current


def current_trace_id() -> str | None:
    """Return the active trace ID when tracing is enabled."""
    if not tracing_enabled():
        return None
    try:
        return get_client().get_current_trace_id()
    except Exception:
        logger.exception("Could not read the current Langfuse trace ID")
        return None


def openai_usage_details(usage: Any) -> dict[str, int] | None:
    """Normalise token usage returned by OpenAI Responses or Embeddings APIs."""
    if usage is None:
        return None

    input_tokens = getattr(usage, "input_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    details = {
        key: int(value)
        for key, value in (
            ("input", input_tokens),
            ("output", output_tokens),
            ("total", total_tokens),
        )
        if value is not None
    }
    return details or None


def flush_observability() -> None:
    """Flush queued observations without affecting application success."""
    if not tracing_enabled():
        return
    try:
        get_client().flush()
    except Exception:
        logger.exception("Could not flush Langfuse observations")


def shutdown_observability() -> None:
    """Flush and stop the Langfuse exporter during process shutdown."""
    if not tracing_enabled():
        return
    try:
        get_client().shutdown()
    except Exception:
        logger.exception("Could not shut down Langfuse observability")
