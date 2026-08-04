"""Start the FastAPI document question-answering service.

Usage:
    python serve.py
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from agent.qa_agent import QAAgent
from config.settings import load_settings
from orchestration.query_classifier import QueryClassifier
from orchestration.query_graph import build_query_graph
from utils.logging import get_logger

logger = get_logger(__name__)


class QueryRequest(BaseModel):
    """Request body accepted by the question-answering endpoint."""

    query: str = Field(min_length=1, max_length=10_000)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must not be blank")
        return query


logger.info("Loading settings...")
settings = load_settings()

logger.info("Initialising QA agent...")
qa_agent = QAAgent(settings)

logger.info("Initialising query classifier...")
query_classifier = QueryClassifier(settings.openai_api_key, settings.classifier)

logger.info("Compiling query graph...")
query_graph = build_query_graph(qa_agent, query_classifier)

app = FastAPI(title="Document Knowledge Base API")


async def _safe_answer_stream(query: str) -> AsyncGenerator[str, None]:
    """Stream graph output and sanitize stream-time errors."""
    try:
        async for event in query_graph.astream(
            {"query": query},
            stream_mode="custom",
            version="v2",
        ):
            if event["type"] == "custom":
                yield event["data"]
    except Exception:
        logger.exception("Answer stream failed")
        yield "\n\n[The answer service is temporarily unavailable.]"


@app.post("/query", response_class=StreamingResponse)
async def query_documents(request: QueryRequest) -> StreamingResponse:
    """Stream the grounded final answer for a document question."""
    return StreamingResponse(
        _safe_answer_stream(request.query),
        media_type="text/plain; charset=utf-8",
        headers={"X-Content-Type-Options": "nosniff"},
    )


if __name__ == "__main__":
    try:
        import uvicorn

        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "8000"))
        logger.info("Starting API server on %s:%s...", host, port)
        uvicorn.run(app, host=host, port=port)
    except EnvironmentError as exc:
        logger.error("Environment error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server stopped.")
        sys.exit(0)
