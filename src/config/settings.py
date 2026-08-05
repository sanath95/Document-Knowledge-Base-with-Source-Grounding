"""Application configuration loaded from environment variables and defaults."""

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
    model: str = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
    batch_size: int = int(os.environ.get("EMBED_BATCH_SIZE", "500"))


@dataclass(frozen=True)
class ChromaConfig:
    collection_name: str = os.environ.get("CHROMA_COLLECTION", "knowledge_base")
    persist_dir: str = os.environ.get("CHROMA_PERSIST_DIR", "./knowledge_base")
    distance_metric: str = "cosine"


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = os.environ.get(
        "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    cache_dir: str = os.environ.get("RERANKER_CACHE_DIR", "./hf_models")


@dataclass(frozen=True)
class ClassifierConfig:
    model: str = os.environ.get("CLASSIFIER_MODEL", "gpt-5.4-nano")


@dataclass(frozen=True)
class ValidatorConfig:
    model: str = os.environ.get("VALIDATOR_MODEL", "gpt-5.4-nano")


@dataclass(frozen=True)
class AgentConfig:
    llm_model: str = os.environ.get("LLM_MODEL", "openai:gpt-5.4")
    temperature: float = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
    parallel_tool_calls: bool = True
    dense_top_k: int = int(os.environ.get("DENSE_TOP_K", "25"))
    bm25_top_k: int = int(os.environ.get("BM25_TOP_K", "25"))
    fusion_top_k: int = int(os.environ.get("FUSION_TOP_K", "25"))
    final_top_k: int = int(os.environ.get("FINAL_TOP_K", "10"))
    rrf_k: int = int(os.environ.get("RRF_K", "60"))


@dataclass(frozen=True)
class IngestionConfig:
    pdf_folder: Path = Path(os.environ.get("PDF_FOLDER", "./data"))
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
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)


def load_settings() -> Settings:
    """Build settings from the values loaded above."""
    return Settings(openai_api_key=_require_env("OPENAI_API_KEY"))
