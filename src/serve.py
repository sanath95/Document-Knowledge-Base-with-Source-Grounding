"""Start the FastAPI document question-answering service.

Usage:
    python serve.py
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from agent.qa_agent import QAAgent
from config.settings import load_settings
from orchestration.answer_validator import AnswerValidator
from orchestration.query_classifier import QueryClassifier
from orchestration.query_graph import build_query_graph
from utils.logging import get_logger
from utils.observability import (
    current_trace_id,
    initialise_observability,
    observation,
    shutdown_observability,
)

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

initialise_observability(instrument_pydantic_ai=True)

logger.info("Initialising QA agent...")
qa_agent = QAAgent(settings)

logger.info("Initialising query safety classifier...")
query_classifier = QueryClassifier(settings.openai_api_key, settings.classifier)

logger.info("Initialising answer validator...")
answer_validator = AnswerValidator(settings.openai_api_key, settings.validator)

logger.info("Compiling query graph...")
query_graph = build_query_graph(qa_agent, query_classifier, answer_validator)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Flush queued observations when the API process shuts down."""
    yield
    shutdown_observability()


app = FastAPI(title="Document Knowledge Base API", lifespan=lifespan)


async def _safe_answer(query: str) -> PlainTextResponse:
    """Run the buffered graph and return only a complete validated response."""
    with observation(
        name="kb.query",
        input={"query": query},
        metadata={"endpoint": "/query"},
    ) as root_observation:
        trace_id = current_trace_id()
        headers = {"X-Content-Type-Options": "nosniff"}
        if trace_id:
            headers["X-Langfuse-Trace-Id"] = trace_id

        try:
            result = await query_graph.ainvoke({"query": query})
            response = result.get("response")
            if not response:
                raise RuntimeError("Query graph returned no response")

            if result.get("classification_failed"):
                outcome = "classification_failed"
            else:
                safety_assessment = result.get("safety_assessment")
                if (
                    safety_assessment is not None
                    and safety_assessment.safety == "unsafe"
                ):
                    outcome = "guardrailed"
                elif result.get("validation_failed"):
                    outcome = "validation_unavailable"
                else:
                    outcome = "success"

            validation = result.get("validation")
            if root_observation is not None:
                root_observation.update(
                    output={"response": response, "outcome": outcome}
                )
                if outcome in {"classification_failed", "validation_unavailable"}:
                    root_observation.update(
                        level="WARNING",
                        status_message=outcome,
                    )
                if validation is not None and not result.get("validation_failed"):
                    root_observation.score_trace(
                        name="faithfulness",
                        value=1 if validation.faithfulness else 0,
                        data_type="BOOLEAN",
                        comment=validation.faithfulness_reason,
                    )
                    root_observation.score_trace(
                        name="context_sufficiency",
                        value=1 if validation.context_sufficiency else 0,
                        data_type="BOOLEAN",
                        comment=validation.context_sufficiency_reason,
                    )

            return PlainTextResponse(response, headers=headers)
        except Exception as exc:
            logger.exception("Answer request failed")
            if root_observation is not None:
                root_observation.update(
                    output={"outcome": "failed"},
                    level="ERROR",
                    status_message=type(exc).__name__,
                )
            return PlainTextResponse(
                "[The answer service is temporarily unavailable.]",
                status_code=503,
                headers=headers,
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
