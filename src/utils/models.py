"""
Shared domain models used across ingestion, retrieval, and agent layers.
All models are immutable dataclasses (or frozen Pydantic models where
external serialisation is needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

class RetrievalFilter(BaseModel):
    """Typed, shared eligibility filter for dense and sparse retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pdf_names: list[str] | None = Field(
        default=None,
        description="PDF filenames to search (case-insensitive).",
    )
    page_from: int | None = Field(default=None, ge=1)
    page_to: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_page_range(self) -> "RetrievalFilter":
        if (
            self.page_from is not None
            and self.page_to is not None
            and self.page_from > self.page_to
        ):
            raise ValueError("page_from must be less than or equal to page_to")
        return self

    @property
    def normalised_pdf_names(self) -> list[str] | None:
        if not self.pdf_names:
            return None
        return sorted({DocumentChunk.normalise_source(name) for name in self.pdf_names})

    def to_chroma_where(self) -> dict | None:
        """Translate this filter to an equivalent Chroma metadata filter."""
        clauses: list[dict] = []
        names = self.normalised_pdf_names
        if names:
            clauses.append(
                {"pdf_name": names[0] if len(names) == 1 else {"$in": names}}
            )
        if self.page_from is not None:
            clauses.append({"page_number": {"$gte": self.page_from}})
        if self.page_to is not None:
            clauses.append({"page_number": {"$lte": self.page_to}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def matches(self, metadata: dict[str, str | int]) -> bool:
        """Apply the same filter semantics to an exported Chroma chunk."""
        names = self.normalised_pdf_names
        source = DocumentChunk.normalise_source(str(metadata.get("pdf_name", "")))
        page = int(metadata.get("page_number", 0))
        return (
            (not names or source in names)
            and (self.page_from is None or page >= self.page_from)
            and (self.page_to is None or page <= self.page_to)
        )


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by the vector store, optionally reranked."""
    document: str
    metadata: dict[str, str | int]
    score: float
    chunk_id: str = ""
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    reranker_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_rank: int | None = None
    reranker_rank: int | None = None

    @property
    def source_pdf(self) -> str:
        return str(self.metadata.get("pdf_name", "<unknown>"))

    @property
    def page_number(self) -> int:
        return int(self.metadata.get("page_number", 0))

    def citation(self) -> str:
        return f"[{self.source_pdf}, page {self.page_number}]"


@dataclass
class DocumentIndex:
    """Summary of all chunks belonging to a single PDF in the store."""
    pdf_name: str
    chunk_count: int
