"""
Reranker
────────
Cross-encoder reranking of candidate chunks using sentence-transformers.
CPU-bound inference is offloaded to a thread pool so it does not block
the async event loop.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import CrossEncoder

from config.settings import RerankerConfig
from utils.logging import get_logger
from utils.models import RetrievedChunk

logger = get_logger(__name__)


class Reranker:
    """
    Rerank a list of RetrievedChunk objects using a cross-encoder model.

    Args:
        config: Reranker configuration (model name, cache directory, threshold).
    """

    def __init__(self, config: RerankerConfig) -> None:
        self._config = config
        self._model = CrossEncoder(
            config.model_name,
            model_kwargs={"cache_dir": config.cache_dir},
        )
        # Single-worker executor — cross-encoder is not thread-safe with >1 thread
        self._executor = ThreadPoolExecutor(max_workers=1)
        logger.info("Reranker loaded: %s", config.model_name)

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Score and re-order *chunks* by relevance to *query*.

        Inference is run in a background thread to avoid blocking the event loop.

        Args:
            query:  The user's search query.
            chunks: Candidates returned by the vector store.

        Returns:
            Filtered (score > threshold) and sorted (descending) list of chunks.
        """
        if not chunks:
            return []

        pairs = [(query, chunk.document) for chunk in chunks]

        loop = asyncio.get_running_loop()
        scores: list[float] = await loop.run_in_executor(
            self._executor,
            lambda: [float(s) for s in self._model.predict(pairs)],
        )

        reranked = [
            RetrievedChunk(
                document=chunk.document,
                metadata=chunk.metadata,
                score=score,
            )
            for chunk, score in zip(chunks, scores)
            if score > self._config.score_threshold
        ]

        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked
