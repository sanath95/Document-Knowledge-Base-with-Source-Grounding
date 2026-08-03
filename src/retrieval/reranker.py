"""
Reranker
────────
Cross-encoder reranking of candidate chunks using Hugging Face Transformers.
CPU-bound inference is offloaded to a thread pool so it does not block
the async event loop.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
        )
        self._model.to("cpu")
        self._model.eval()
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
            Filtered and sorted (descending) list with normalized scores.
        """
        if not chunks:
            return []

        loop = asyncio.get_running_loop()
        scores: list[float] = await loop.run_in_executor(
            self._executor,
            self._predict_scores,
            query,
            chunks,
        )

        reranked = [
            replace(
                chunk,
                score=score,
                reranker_score=score,
            )
            for chunk, score in zip(chunks, scores)
            if score >= self._config.score_threshold
        ]

        reranked.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
        return [
            replace(chunk, reranker_rank=rank)
            for rank, chunk in enumerate(reranked, start=1)
        ]

    def _predict_scores(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[float]:
        """Run batched CPU inference and normalize logits with sigmoid."""
        scores: list[float] = []
        batch_size = max(1, self._config.batch_size)

        with torch.inference_mode():
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                inputs = self._tokenizer(
                    [(query, chunk.document) for chunk in batch],
                    padding=True,
                    truncation=True,
                    max_length=self._config.max_length,
                    return_tensors="pt",
                )
                logits = self._model(**inputs).logits.view(-1).float()
                scores.extend(torch.sigmoid(logits).cpu().tolist())

        return scores
