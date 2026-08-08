"""Centralised logging configuration for the project."""

from __future__ import annotations

import logging
import sys


def configure_logging() -> None:
    """Configure application logging once at the process entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        stream=sys.stdout,
    )
