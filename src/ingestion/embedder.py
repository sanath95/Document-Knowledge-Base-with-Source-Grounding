"""
Embedder
────────
Async-batched text embedding using the OpenAI Embeddings API.
Uses asyncio.gather for concurrent batch requests, which meaningfully
reduces wall-clock time when embedding large corpora.
"""

from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI

from config.settings import EmbeddingConfig
from utils.observability import observation, openai_usage_details

logger = logging.getLogger(__name__)


class Embedder:
    """
    Embed lists of text strings using the OpenAI Embeddings API.

    Args:
        api_key:  OpenAI API key.
        config:   Embedding configuration (model name, batch size).
    """

    def __init__(self, api_key: str, config: EmbeddingConfig) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._config = config

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """
        Embed *texts* in parallel batches.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of embedding vectors in the same order as *texts*.

        Raises:
            ValueError: If *texts* is empty.
            RuntimeError: If any OpenAI API call fails.
        """
        if not texts:
            raise ValueError("embed_many() received an empty text list.")

        batches = self._make_batches(texts)
        logger.info(
            "Embedding %d texts in %d batch(es) with model '%s'",
            len(texts),
            len(batches),
            self._config.model,
        )

        tasks = [self._embed_batch(batch, idx) for idx, batch in enumerate(batches)]
        batch_results: list[list[list[float]]] = await asyncio.gather(*tasks)

        embeddings: list[list[float]] = [
            vec for batch in batch_results for vec in batch
        ]
        logger.info("  → %d embeddings received", len(embeddings))
        return embeddings

    async def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for a single string."""
        results = await self.embed_many([text])
        return results[0]

    # ── private ───────────────────────────────────────────────────────────────

    def _make_batches(self, texts: list[str]) -> list[list[str]]:
        size = self._config.batch_size
        return [texts[i : i + size] for i in range(0, len(texts), size)]

    async def _embed_batch(
        self, batch: list[str], batch_index: int
    ) -> list[list[float]]:
        with observation(
            name="openai.embedding",
            as_type="embedding",
            input={
                "batch_index": batch_index,
                "text_count": len(batch),
                "character_count": sum(len(text) for text in batch),
            },
            model=self._config.model,
        ) as embedding_observation:
            try:
                response = await self._client.embeddings.create(
                    input=batch, model=self._config.model
                )
            except Exception as exc:
                raise RuntimeError(
                    f"OpenAI embedding request failed (batch {batch_index}): {exc}"
                ) from exc

            # response.data is ordered by index field — sort defensively
            sorted_data = sorted(response.data, key=lambda item: item.index)
            embeddings = [item.embedding for item in sorted_data]
            if embedding_observation is not None:
                embedding_observation.update(
                    output={
                        "embedding_count": len(embeddings),
                        "dimensions": len(embeddings[0]) if embeddings else 0,
                    },
                    usage_details=openai_usage_details(response.usage),
                )
            return embeddings
