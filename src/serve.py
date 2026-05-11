"""
serve.py
────────
Start the pydantic-ai web agent.

Usage:
    python serve.py
"""

from __future__ import annotations

import sys

from agent.qa_agent import QAAgent
from config.settings import load_settings
from utils.logging import get_logger

logger = get_logger(__name__)


logger.info("Loading settings …")
settings = load_settings()

logger.info("Initialising QA agent …")
qa_agent = QAAgent(settings)

logger.info("Starting web server …")
app = qa_agent.to_web()


if __name__ == "__main__":
    try:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=7932)
    except EnvironmentError as exc:
        logger.error("Environment error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server stopped.")
        sys.exit(0)
