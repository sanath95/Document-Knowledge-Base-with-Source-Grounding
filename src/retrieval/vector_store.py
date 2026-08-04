"""
VectorStore
───────────
Thin, typed wrapper around ChromaDB.
Handles collection lifecycle, upsert, and vector query.
"""

from __future__ import annotations

import logging

import chromadb

from config.settings import ChromaConfig
from utils.models import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manage a ChromaDB collection for document chunks.

    Args:
        config: ChromaDB configuration (collection name, persist dir, metric).
    """

    def __init__(self, config: ChromaConfig) -> None:
        self._config = config
        self._client: chromadb.ClientAPI = self._build_client()
        self._collection: chromadb.Collection = self._get_or_create_collection()

    # ── Public API ────────────────────────────────────────────────────────────

    def upsert(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Insert or update chunks in the collection.

        Args:
            chunks:     DocumentChunk objects to store.
            embeddings: Embedding vectors (must match length of *chunks*).

        Raises:
            ValueError: If lengths of *chunks* and *embeddings* differ.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )
        if not chunks:
            logger.warning("upsert() called with empty chunk list — skipped.")
            return

        self._collection.upsert(
            ids=[chunk.chunk_id(i) for i, chunk in enumerate(chunks)],
            documents=[chunk.content for chunk in chunks],
            embeddings=embeddings,
            metadatas=[chunk.to_metadata() for chunk in chunks],
        )
        logger.info(
            "Upserted %d chunks → collection now has %d documents.",
            len(chunks),
            self._collection.count(),
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the *top_k* most similar chunks.

        Args:
            query_embedding: Embedding of the search query.
            top_k:           Maximum number of results.

        Returns:
            List of RetrievedChunk objects (unranked; score = cosine similarity).
        """
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        ids: list[str] = result["ids"][0]
        docs: list[str] = result["documents"][0]
        metadatas: list[dict] = result["metadatas"][0]

        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                document=doc,
                metadata=meta,
                score=0.0,
            )
            for chunk_id, doc, meta in zip(ids, docs, metadatas)
        ]

    @property
    def collection(self) -> chromadb.Collection:
        """Expose underlying collection for agent tool injection."""
        return self._collection

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_client(self) -> chromadb.ClientAPI:
        if self._config.persist_dir:
            logger.info("ChromaDB — persisting to '%s'", self._config.persist_dir)
            return chromadb.PersistentClient(path=self._config.persist_dir)
        logger.info("ChromaDB — using in-memory client")
        return chromadb.Client()

    def _get_or_create_collection(self) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=self._config.collection_name,
            metadata={"hnsw:space": self._config.distance_metric},
        )
