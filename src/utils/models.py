"""
Shared domain models used across ingestion, retrieval, and agent layers.
All models are immutable dataclasses (or frozen Pydantic models where
external serialisation is needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Ingestion models ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PageContent:
    """Raw Markdown extracted from a single PDF page."""
    page_number: int          # 1-based
    markdown: str
    source_pdf: str           # normalised filename, e.g. "report_2024.pdf"


@dataclass(frozen=True)
class DocumentChunk:
    """A semantically bounded section of a PDF page, ready for embedding."""
    content: str
    source_pdf: str           # normalised filename
    page_number: int          # 1-based
    h1: Optional[str] = None
    h2: Optional[str] = None
    h3: Optional[str] = None

    # ── Derived helpers ───────────────────────────────────────────────────────

    def chunk_id(self, index: int) -> str:
        """Stable, collision-free identifier for ChromaDB upsert."""
        safe = self.source_pdf.replace(" ", "_").replace("/", "_")
        return f"{safe}__p{self.page_number:04d}__c{index:04d}"

    def to_metadata(self) -> dict[str, str | int]:
        """Flat dict suitable for ChromaDB metadatas."""
        meta: dict[str, str | int] = {
            "pdf_name": self.source_pdf,
            "page_number": self.page_number,
        }
        for header_key in ("h1", "h2", "h3"):
            value = getattr(self, header_key)
            if value:
                meta[header_key] = value
        return meta

    @staticmethod
    def normalise_source(path: Path | str) -> str:
        """
        Return a canonical source name from any path variant.
        Always lower-cased filename only — no directory prefix.
        """
        return Path(path).name.strip().lower()


# ── Retrieval models ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by the vector store, optionally reranked."""
    chunk_id: str
    document: str
    metadata: dict[str, str | int]
    score: float

    @property
    def source_pdf(self) -> str:
        return str(self.metadata.get("pdf_name", "<unknown>"))

    @property
    def page_number(self) -> int:
        return int(self.metadata.get("page_number", 0))

    def citation(self) -> str:
        return f"[{self.source_pdf}, page {self.page_number}]"
