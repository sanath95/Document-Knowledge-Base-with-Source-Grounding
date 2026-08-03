"""Multilingual dense + BM25 retrieval, RRF fusion, and reranking."""

from __future__ import annotations

import asyncio
import re
from dataclasses import replace

import bm25s
from bm25s.tokenization import Tokenizer

from config.settings import HybridRetrievalConfig
from ingestion.embedder import Embedder
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore
from utils.logging import get_logger
from utils.models import RetrievalFilter, RetrievedChunk

logger = get_logger(__name__)

_UNICODE_WORD = re.compile(r"(?u)\b\w+\b")


class HybridRetriever:
    """Fuse semantic and lexical retrieval before multilingual reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
        config: HybridRetrievalConfig,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker
        self._config = config
        self._chunks: list[RetrievedChunk] = []
        self._tokenizer: Tokenizer | None = None
        self._bm25: bm25s.BM25 | None = None
        self.refresh_sparse_index()

    @property
    def indexed_chunk_count(self) -> int:
        """Number of Chroma chunks represented in the disposable BM25 index."""
        return len(self._chunks)

    def refresh_sparse_index(self) -> None:
        """Rebuild the in-memory BM25 index from the current Chroma collection."""
        self._chunks = self._vector_store.export_chunks()
        self._tokenizer = None
        self._bm25 = None

        if not self._chunks:
            logger.warning("BM25 index not built because Chroma contains no chunks.")
            return

        self._tokenizer = Tokenizer(
            stemmer=None,
            stopwords=[],
            splitter=lambda text: text.split(),
        )
        corpus = [self._normalise_text(chunk.document) for chunk in self._chunks]
        corpus_tokens = self._tokenizer.tokenize(corpus, show_progress=False)
        self._bm25 = bm25s.BM25(method="lucene")
        self._bm25.index(corpus_tokens, show_progress=False)
        logger.info("Built in-memory multilingual BM25 index over %d chunks.", len(corpus))

    async def retrieve(
        self,
        query: str,
        filters: RetrievalFilter | None = None,
    ) -> list[RetrievedChunk]:
        """Run both retrieval legs, fuse candidates, and return reranked chunks."""
        dense_results, sparse_results = await asyncio.gather(
            self._dense_retrieve(query, filters),
            asyncio.to_thread(self._sparse_retrieve, query, filters),
        )
        fused = self._fuse(dense_results, sparse_results)
        candidates = fused[: self._config.rerank_candidates]
        reranked = await self._reranker.rerank(query=query, chunks=candidates)
        return reranked[: self._config.final_top_k]

    async def _dense_retrieve(
        self,
        query: str,
        filters: RetrievalFilter | None,
    ) -> list[RetrievedChunk]:
        query_embedding = await self._embedder.embed_one(query)
        return self._vector_store.query(
            query_embedding=query_embedding,
            top_k=self._config.dense_top_k,
            filters=filters,
        )

    def _sparse_retrieve(
        self,
        query: str,
        filters: RetrievalFilter | None,
    ) -> list[RetrievedChunk]:
        if not self._bm25 or not self._tokenizer or not self._chunks:
            return []

        normalised_query = self._normalise_text(query)
        if not normalised_query:
            return []

        query_tokens = self._tokenizer.tokenize(
            [normalised_query],
            update_vocab=False,
            show_progress=False,
        )
        result_ids, result_scores = self._bm25.retrieve(
            query_tokens,
            k=len(self._chunks),
            show_progress=False,
        )

        matches: list[RetrievedChunk] = []
        for position, raw_score in zip(result_ids[0], result_scores[0]):
            score = float(raw_score)
            if score <= 0.0:
                break

            chunk = self._chunks[int(position)]
            if filters and not filters.matches(chunk.metadata):
                continue

            rank = len(matches) + 1
            matches.append(
                replace(
                    chunk,
                    score=score,
                    sparse_score=score,
                    sparse_rank=rank,
                )
            )
            if len(matches) >= self._config.sparse_top_k:
                break

        return matches

    def _fuse(
        self,
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Deduplicate by Chroma ID and combine ranks with reciprocal rank fusion."""
        chunks: dict[str, RetrievedChunk] = {}
        fused_scores: dict[str, float] = {}

        for rank, chunk in enumerate(dense_results, start=1):
            chunks[chunk.chunk_id] = replace(chunk, dense_rank=rank)
            fused_scores[chunk.chunk_id] = 1.0 / (self._config.rrf_k + rank)

        for rank, sparse_chunk in enumerate(sparse_results, start=1):
            chunk_id = sparse_chunk.chunk_id
            existing = chunks.get(chunk_id)
            if existing:
                chunks[chunk_id] = replace(
                    existing,
                    sparse_score=sparse_chunk.sparse_score,
                    sparse_rank=rank,
                )
            else:
                chunks[chunk_id] = replace(sparse_chunk, sparse_rank=rank)
                fused_scores[chunk_id] = 0.0
            fused_scores[chunk_id] += 1.0 / (self._config.rrf_k + rank)

        ordered = sorted(
            chunks.values(),
            key=lambda chunk: (-fused_scores[chunk.chunk_id], chunk.chunk_id),
        )
        return [
            replace(
                chunk,
                score=fused_scores[chunk.chunk_id],
                fused_score=fused_scores[chunk.chunk_id],
                fused_rank=rank,
            )
            for rank, chunk in enumerate(ordered, start=1)
        ]

    @staticmethod
    def _normalise_text(text: str) -> str:
        """Unicode-aware lowercase word tokenization without linguistic rules."""
        return " ".join(_UNICODE_WORD.findall(text.lower()))
