"""
IngestionPipeline
─────────────────
Orchestrates the full PDF → chunk → embed → store pipeline.
Async where it benefits (batched embedding), sync otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.settings import Settings
from ingestion.chunker import MarkdownChunker
from ingestion.embedder import Embedder
from ingestion.pdf_parser import PDFParser
from retrieval.vector_store import VectorStore
from utils.logging import get_logger
from utils.models import DocumentChunk

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestionSummary:
    """Small observable summary of a completed folder ingestion run."""

    document_count: int
    successful_document_count: int
    chunk_count: int
    failed_files: tuple[str, ...]


class IngestionPipeline:
    """
    End-to-end ingestion of a PDF folder into ChromaDB.

    Args:
        settings:     Global application settings.
        vector_store: Pre-initialised VectorStore instance.
    """

    def __init__(self, settings: Settings, vector_store: VectorStore) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._parser = PDFParser()
        self._chunker = MarkdownChunker(settings.ingestion)
        self._embedder = Embedder(settings.openai_api_key, settings.embedding)

    async def run(self) -> IngestionSummary:
        """
        Discover PDFs in the configured folder and ingest each one.

        Raises:
            NotADirectoryError: If the PDF folder does not exist.
            FileNotFoundError:  If no PDFs are found.

        Returns:
            Counts and failed filenames for logging and observability.
        """
        folder = self._settings.ingestion.pdf_folder
        if not folder.is_dir():
            raise NotADirectoryError(f"PDF folder not found: {folder}")

        pdf_files = sorted(
            p
            for ext in self._settings.ingestion.supported_extensions
            for p in folder.glob(f"*{ext}")
        )
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in: {folder}")

        logger.info(
            "Found %d PDF(s) in '%s'", len(pdf_files), folder
        )

        total_chunks = 0
        failed: list[str] = []

        for idx, pdf_path in enumerate(pdf_files, 1):
            logger.info("─── [%d/%d] %s", idx, len(pdf_files), pdf_path.name)
            try:
                count = await self._ingest_pdf(pdf_path)
                total_chunks += count
            except Exception as exc:
                logger.error("Skipped '%s' — %s", pdf_path.name, exc)
                failed.append(pdf_path.name)

        logger.info(
            "Ingestion complete: %d chunks from %d PDF(s). Failed: %s",
            total_chunks,
            len(pdf_files) - len(failed),
            failed or "none",
        )
        return IngestionSummary(
            document_count=len(pdf_files),
            successful_document_count=len(pdf_files) - len(failed),
            chunk_count=total_chunks,
            failed_files=tuple(failed),
        )

    async def ingest_file(self, pdf_path: Path) -> int:
        """
        Ingest a single PDF file.

        Args:
            pdf_path: Path to the PDF.

        Returns:
            Number of chunks stored.
        """
        return await self._ingest_pdf(pdf_path)

    # ── Private ───────────────────────────────────────────────────────────────

    async def _ingest_pdf(self, pdf_path: Path) -> int:
        pages = self._parser.parse(pdf_path)
        chunks: list[DocumentChunk] = self._chunker.chunk(pages)

        if not chunks:
            logger.warning("No chunks produced for '%s' — skipping.", pdf_path.name)
            return 0

        texts = [c.content for c in chunks]
        embeddings = await self._embedder.embed_many(texts)

        self._vector_store.upsert(chunks, embeddings)
        return len(chunks)
