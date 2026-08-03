"""
Centralised configuration — loaded once at import time.
All values come from environment variables; defaults are only for
development convenience and must be overridden in production.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _require_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Check your .env file or shell environment."
        )
    return value


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    batch_size: int = 500


@dataclass(frozen=True)
class ChromaConfig:
    collection_name: str = "knowledge_base"
    persist_dir: str = "./knowledge_base"
    distance_metric: str = "cosine"


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    cache_dir: str = "./hf_models"
    score_threshold: float = 0.0


@dataclass(frozen=True)
class AgentConfig:
    llm_model: str = "openai:gpt-4o-mini"
    temperature: float = 0.0
    parallel_tool_calls: bool = True
    retrieval_top_k: int = 10


@dataclass(frozen=True)
class IngestionConfig:
    pdf_folder: Path = field(default_factory=lambda: Path("./data"))
    supported_extensions: tuple[str, ...] = (".pdf",)
    markdown_headers: tuple[tuple[str, str], ...] = (
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    )


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)


def load_settings() -> Settings:
    """Build and return validated Settings from the environment."""
    return Settings(
        openai_api_key=_require_env("OPENAI_API_KEY"),
        embedding=EmbeddingConfig(
            model=os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
            batch_size=int(os.environ.get("EMBED_BATCH_SIZE", "500")),
        ),
        chroma=ChromaConfig(
            collection_name=os.environ.get("CHROMA_COLLECTION", "knowledge_base"),
            persist_dir=os.environ.get("CHROMA_PERSIST_DIR", "./knowledge_base"),
        ),
        reranker=RerankerConfig(
            model_name=os.environ.get(
                "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
            ),
            cache_dir=os.environ.get("RERANKER_CACHE_DIR", "./hf_models"),
            score_threshold=float(os.environ.get("RERANKER_THRESHOLD", "0.0")),
        ),
        agent=AgentConfig(
            llm_model=os.environ.get("LLM_MODEL", "openai:gpt-4o-mini"),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
        ),
        ingestion=IngestionConfig(
            pdf_folder=Path(os.environ.get("PDF_FOLDER", "./data")),
        ),
    )
