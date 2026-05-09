from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / "Technical Interview Docs"
CHROMA_DIR = ROOT_DIR / ".chroma"
COLLECTION_NAME = "document_pages"


@dataclass(frozen=True)
class Settings:
    docs_dir: Path
    chroma_dir: Path
    collection_name: str
    openai_api_key: str | None
    openai_model: str
    embedding_model: str
    enable_reranker: bool
    reranker_model: str
    chunk_size: int
    chunk_overlap: int
    vector_results: int
    bm25_results: int
    final_results: int
    min_relevance: float


def load_settings() -> Settings:
    load_dotenv(ROOT_DIR / ".env")
    return Settings(
        docs_dir=Path(os.getenv("DOCS_DIR", str(DOCS_DIR))).resolve(),
        chroma_dir=Path(os.getenv("CHROMA_DIR", str(CHROMA_DIR))).resolve(),
        collection_name=os.getenv("CHROMA_COLLECTION", COLLECTION_NAME),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        enable_reranker=os.getenv("ENABLE_RERANKER", "false").lower() in {"1", "true", "yes"},
        reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "180")),
        vector_results=int(os.getenv("VECTOR_RESULTS", "12")),
        bm25_results=int(os.getenv("BM25_RESULTS", "12")),
        final_results=int(os.getenv("FINAL_RESULTS", "6")),
        min_relevance=float(os.getenv("MIN_RELEVANCE", "0.18")),
    )

