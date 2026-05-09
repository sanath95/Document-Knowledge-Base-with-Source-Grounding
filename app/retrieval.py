from __future__ import annotations

import math
import os
import re
from collections import OrderedDict

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import Settings
from app.models import Evidence
from app.openai_client import embed_texts, make_client
from app.vector_store import fetch_all_evidence, get_collection


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\-]+", text.lower())


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if math.isclose(low, high):
        return [1.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def _vector_search(settings: Settings, query: str) -> list[Evidence]:
    client = make_client(settings)
    collection = get_collection(settings)
    query_embedding = embed_texts(client, [query], settings.embedding_model)[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=settings.vector_results,
        include=["documents", "metadatas", "distances"],
    )
    evidence: list[Evidence] = []
    ids = result["ids"][0]
    docs = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    for item_id, text, metadata, distance in zip(ids, docs, metadatas, distances, strict=True):
        similarity = max(0.0, 1.0 - float(distance))
        evidence.append(
            Evidence(
                id=item_id,
                document=str(metadata["document"]),
                page=int(metadata["page"]),
                text=str(text),
                score=similarity,
                source="vector",
            )
        )
    return evidence


def _bm25_search(settings: Settings, query: str) -> list[Evidence]:
    collection = get_collection(settings)
    corpus = fetch_all_evidence(collection)
    if not corpus:
        return []
    tokenized = [tokenize(item.text) for item in corpus]
    bm25 = BM25Okapi(tokenized)
    raw_scores = list(bm25.get_scores(tokenize(query)))
    normalized = _normalize_scores(raw_scores)
    ranked = sorted(zip(corpus, normalized, strict=True), key=lambda item: item[1], reverse=True)
    return [
        Evidence(
            id=item.id,
            document=item.document,
            page=item.page,
            text=item.text,
            score=float(score),
            source="bm25",
        )
        for item, score in ranked[: settings.bm25_results]
        if score > 0
    ]


def _merge(candidates: list[Evidence]) -> list[Evidence]:
    merged: OrderedDict[str, Evidence] = OrderedDict()
    for item in sorted(candidates, key=lambda evidence: evidence.score, reverse=True):
        existing = merged.get(item.id)
        if existing is None:
            merged[item.id] = item
            continue
        merged[item.id] = Evidence(
            id=item.id,
            document=item.document,
            page=item.page,
            text=item.text,
            score=max(existing.score, item.score),
            source=f"{existing.source}+{item.source}",
        )
    return list(merged.values())


def _rerank_if_available(settings: Settings, query: str, candidates: list[Evidence]) -> list[Evidence]:
    if not settings.enable_reranker or not candidates:
        return candidates
    try:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(settings.reranker_model)
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = model.predict(pairs)
        normalized = _normalize_scores([float(score) for score in np.asarray(scores).tolist()])
        reranked = [
            Evidence(
                id=candidate.id,
                document=candidate.document,
                page=candidate.page,
                text=candidate.text,
                score=float(score),
                source=f"{candidate.source}+rerank",
            )
            for candidate, score in zip(candidates, normalized, strict=True)
        ]
        return sorted(reranked, key=lambda evidence: evidence.score, reverse=True)
    except Exception:
        return candidates


def retrieve(settings: Settings, query: str) -> list[Evidence]:
    vector_results = _vector_search(settings, query)
    bm25_results = _bm25_search(settings, query)
    merged = _merge(vector_results + bm25_results)
    reranked = _rerank_if_available(settings, query, merged)
    filtered = [item for item in reranked if item.score >= settings.min_relevance]
    return filtered[: settings.final_results]

