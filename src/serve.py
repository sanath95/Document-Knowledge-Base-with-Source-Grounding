"""Start the FastAPI document question-answering service.

Usage:
    python serve.py
"""

from __future__ import annotations

import os
import sys

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from agent.qa_agent import QAAgent
from config.settings import load_settings
from orchestration.answer_validator import AnswerValidator
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

logger.info("Initialising answer validator...")
answer_validator = AnswerValidator(settings.openai_api_key, settings.validator)

logger.info("Compiling query graph...")
query_graph = build_query_graph(qa_agent, query_classifier, answer_validator)

app = FastAPI(title="Document Knowledge Base API")


async def _safe_answer(query: str) -> PlainTextResponse:
    """Run the buffered graph and return only a complete validated response."""
    try:
        result = await query_graph.ainvoke({"query": query})
        response = result.get("response")
        if not response:
            raise RuntimeError("Query graph returned no response")
        return PlainTextResponse(
            response,
            headers={"X-Content-Type-Options": "nosniff"},
        )
    except Exception:
        logger.exception("Answer request failed")
        return PlainTextResponse(
            "[The answer service is temporarily unavailable.]",
            status_code=503,
            headers={"X-Content-Type-Options": "nosniff"},
        )


@app.post("/query", response_class=PlainTextResponse)
async def query_documents(request: QueryRequest) -> PlainTextResponse:
    """Return the complete grounded answer and its validation results."""
    return await _safe_answer(request.query)


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
