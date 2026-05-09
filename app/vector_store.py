from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import Settings
from app.models import Chunk, Evidence


def get_client(path: Path) -> chromadb.PersistentClient:
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def recreate_collection(settings: Settings) -> Collection:
    client = get_client(settings.chroma_dir)
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    return client.create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def get_collection(settings: Settings) -> Collection:
    client = get_client(settings.chroma_dir)
    return client.get_collection(name=settings.collection_name, embedding_function=None)


def collection_exists(settings: Settings) -> bool:
    client = get_client(settings.chroma_dir)
    return any(collection.name == settings.collection_name for collection in client.list_collections())


def add_chunks(collection: Collection, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    collection.add(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {
                "document": chunk.document,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ],
        embeddings=embeddings,
    )


def fetch_all_evidence(collection: Collection) -> list[Evidence]:
    data = collection.get(include=["documents", "metadatas"])
    evidence: list[Evidence] = []
    for item_id, document, metadata in zip(data["ids"], data["documents"], data["metadatas"], strict=True):
        evidence.append(
            Evidence(
                id=item_id,
                document=str(metadata["document"]),
                page=int(metadata["page"]),
                text=str(document),
                score=0.0,
                source="index",
            )
        )
    return evidence

