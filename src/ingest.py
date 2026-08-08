"""
ingest.py
─────────
CLI entry point.  Run this once (or whenever new PDFs are added) to
populate the ChromaDB vector store.

Usage:
    python ingest.py
    PDF_FOLDER=./my_docs python ingest.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

from config.settings import load_settings
from ingestion.pipeline import IngestionPipeline
from retrieval.vector_store import VectorStore
from utils.logging import configure_logging
from utils.observability import (
    flush_observability,
    initialise_observability,
    observation,
)

configure_logging()
logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info("Loading settings …")
    settings = load_settings()
    initialise_observability()

    try:
        with observation(
            name="kb.ingestion",
            input={
                "pdf_folder": str(settings.ingestion.pdf_folder),
                "collection": settings.chroma.collection_name,
            },
        ) as ingestion_observation:
            vector_store = VectorStore(settings.chroma)
            pipeline = IngestionPipeline(
                settings=settings,
                vector_store=vector_store,
            )
            summary = await pipeline.run()
            if ingestion_observation is not None:
                ingestion_observation.update(
                    output={
                        "document_count": summary.document_count,
                        "successful_document_count": (
                            summary.successful_document_count
                        ),
                        "chunk_count": summary.chunk_count,
                        "failed_files": list(summary.failed_files),
                    },
                    level="WARNING" if summary.failed_files else "DEFAULT",
                    status_message=(
                        "some documents failed" if summary.failed_files else None
                    ),
                )
    finally:
        flush_observability()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (NotADirectoryError, FileNotFoundError) as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(0)
