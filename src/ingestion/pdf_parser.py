"""
PDFParser
─────────
Converts a PDF file into a list of PageContent objects using pymupdf4llm.
Stateless — create once, call parse() as many times as needed.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf4llm

from utils.logging import get_logger
from utils.models import DocumentChunk, PageContent

logger = get_logger(__name__)


class PDFParser:
    """Extract per-page Markdown from a PDF file."""

    def parse(self, pdf_path: Path) -> list[PageContent]:
        """
        Convert every page of *pdf_path* to Markdown.

        Args:
            pdf_path: Absolute or relative path to a PDF file.

        Returns:
            Ordered list of PageContent (one per page, empty pages excluded).

        Raises:
            FileNotFoundError: If *pdf_path* does not exist.
            RuntimeError: If pymupdf4llm fails to process the file.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        source_name = DocumentChunk.normalise_source(pdf_path)
        logger.info("Parsing PDF: %s", pdf_path.name)

        try:
            page_chunks: list[dict] = pymupdf4llm.to_markdown(
                str(pdf_path), page_chunks=True
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to parse '{pdf_path.name}': {exc}") from exc

        pages: list[PageContent] = []
        for chunk in page_chunks:
            text: str = (chunk.get("text") or "").strip()
            if not text:
                continue
            pages.append(
                PageContent(
                    page_number=chunk["metadata"]["page_number"],
                    markdown=text,
                    source_pdf=source_name,
                )
            )

        logger.info("  → %d non-empty pages extracted", len(pages))
        return pages
