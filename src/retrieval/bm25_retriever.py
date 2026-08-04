"""In-memory BM25 retrieval over chunks already persisted in ChromaDB."""

from __future__ import annotations

import chromadb
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

from utils.logging import get_logger
from utils.models import RetrievedChunk

logger = get_logger(__name__)


class BM25Retriever:
    """Build and query a BM25 index using an independently owned tokenizer."""

    def __init__(
        self,
        collection: chromadb.Collection,
        tokenizer_model_name: str,
        tokenizer_cache_dir: str,
    ) -> None:
        self._collection = collection
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_name,
            cache_dir=tokenizer_cache_dir,
        )
        self._chunks: list[RetrievedChunk] = []
        self._token_sets: list[set[str]] = []
        self._index: BM25Okapi | None = None
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the lexical index from the current Chroma collection."""
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        self._chunks = [
            RetrievedChunk(
                chunk_id=chunk_id,
                document=document,
                metadata=metadata or {},
                score=0.0,
            )
            for chunk_id, document, metadata in zip(ids, documents, metadatas)
        ]

        tokenized_corpus = [
            self._tokenizer.tokenize(chunk.document) for chunk in self._chunks
        ]
        self._token_sets = [set(tokens) for tokens in tokenized_corpus]
        self._index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        logger.info("BM25 index loaded: %d chunks", len(self._chunks))

    def query(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Return the highest-ranked lexical matches for *query*."""
        if self._index is None or top_k <= 0:
            return []

        query_tokens = self._tokenizer.tokenize(query)
        if not query_tokens:
            return []

        allowed_ids: set[str] | None = None
        if filters:
            filtered = self._collection.get(where=filters, include=[])
            allowed_ids = set(filtered.get("ids") or [])

        scores = self._index.get_scores(query_tokens)
        query_token_set = set(query_tokens)
        candidate_indexes = [
            index
            for index, chunk in enumerate(self._chunks)
            if (allowed_ids is None or chunk.chunk_id in allowed_ids)
            and not query_token_set.isdisjoint(self._token_sets[index])
        ]
        candidate_indexes.sort(key=lambda index: float(scores[index]), reverse=True)

        return [
            RetrievedChunk(
                chunk_id=self._chunks[index].chunk_id,
                document=self._chunks[index].document,
                metadata=self._chunks[index].metadata,
                score=float(scores[index]),
            )
            for index in candidate_indexes[:top_k]
        ]
