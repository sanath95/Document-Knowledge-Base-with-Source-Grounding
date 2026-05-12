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
import sys

from config.settings import load_settings
from ingestion.pipeline import IngestionPipeline
from retrieval.vector_store import VectorStore
from utils.logging import get_logger

logger = get_logger(__name__)


async def main() -> None:
    logger.info("Loading settings …")
    settings = load_settings()

    vector_store = VectorStore(settings.chroma)
    pipeline = IngestionPipeline(settings=settings, vector_store=vector_store)

    await pipeline.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (NotADirectoryError, FileNotFoundError) as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(0)
