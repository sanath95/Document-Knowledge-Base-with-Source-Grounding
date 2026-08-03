"""Rank-fusion utilities for hybrid retrieval."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from utils.models import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse ranked chunk lists using reciprocal rank fusion."""
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    scores: dict[str, float] = defaultdict(float)
    chunks_by_id: dict[str, RetrievedChunk] = {}

    for ranking in rankings:
        seen_ids: set[str] = set()
        for rank, chunk in enumerate(ranking, start=1):
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            scores[chunk.chunk_id] += 1.0 / (rrf_k + rank)
            chunks_by_id.setdefault(chunk.chunk_id, chunk)

    fused = [
        replace(chunk, score=scores[chunk_id])
        for chunk_id, chunk in chunks_by_id.items()
    ]
    fused.sort(key=lambda chunk: chunk.score, reverse=True)
    return fused
