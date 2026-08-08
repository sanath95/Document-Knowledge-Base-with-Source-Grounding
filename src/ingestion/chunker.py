"""
MarkdownChunker
───────────────
Splits per-page Markdown into header-bounded DocumentChunk objects
using LangChain's MarkdownHeaderTextSplitter.
"""

from __future__ import annotations

import logging

from langchain_text_splitters import MarkdownHeaderTextSplitter

from config.settings import IngestionConfig
from utils.models import DocumentChunk, PageContent

logger = logging.getLogger(__name__)


class MarkdownChunker:
    """
    Split Markdown pages into semantically bounded chunks.

    Args:
        config: Ingestion configuration (provides header definitions).
    """

    def __init__(self, config: IngestionConfig) -> None:
        self._splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=list(config.markdown_headers),
            strip_headers=False,
            return_each_line=False,
        )

    def chunk(self, pages: list[PageContent]) -> list[DocumentChunk]:
        """
        Convert a list of PageContent objects into DocumentChunks.

        Args:
            pages: Pages from a single PDF (as returned by PDFParser).

        Returns:
            Flat, ordered list of DocumentChunk objects.
        """
        chunks: list[DocumentChunk] = []

        for page in pages:
            page_chunks = self._chunk_page(page)
            chunks.extend(page_chunks)

        logger.info("  → %d chunks produced from %d pages", len(chunks), len(pages))
        return chunks

    # ── private ───────────────────────────────────────────────────────────────

    def _chunk_page(self, page: PageContent) -> list[DocumentChunk]:
        splits = self._splitter.split_text(page.markdown)
        result: list[DocumentChunk] = []

        for split in splits:
            content: str = split.page_content.strip()
            if not content:
                continue

            header_meta: dict[str, str] = split.metadata  # {"h1": ..., "h2": ...}
            result.append(
                DocumentChunk(
                    content=content,
                    source_pdf=page.source_pdf,
                    page_number=page.page_number,
                    h1=header_meta.get("h1"),
                    h2=header_meta.get("h2"),
                    h3=header_meta.get("h3"),
                )
            )

        return result
