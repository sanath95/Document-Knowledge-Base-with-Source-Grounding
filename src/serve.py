"""Start the FastAPI document question-answering service.

Usage:
    python serve.py
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from utils.logging import configure_logging
from utils.observability import (
    current_trace_id,
    initialise_observability,
    observation,
    shutdown_observability,
)

configure_logging()
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Request body accepted by the question-answering endpoint."""

    conversation_id: UUID
    query: str = Field(min_length=1, max_length=10_000)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must not be blank")
        return query


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Initialise service dependencies and release them on shutdown."""
    from agent.qa_agent import QAAgent
    from config.settings import load_settings
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from orchestration.answer_validator import AnswerValidator
    from orchestration.query_classifier import QueryClassifier
    from orchestration.query_contextualizer import QueryContextualizer
    from orchestration.query_graph import build_query_graph

    logger.info("Loading settings...")
    settings = load_settings()
    initialise_observability()
    qa_agent: QAAgent | None = None

    try:
        logger.info("Initialising QA agent...")
        qa_agent = QAAgent(settings)

        logger.info("Initialising query safety classifier...")
        query_classifier = QueryClassifier(
            settings.openai_api_key,
            settings.classifier,
            settings.conversation.history_max_turns,
        )

        logger.info("Initialising conversation query contextualizer...")
        query_contextualizer = QueryContextualizer(
            settings.openai_api_key,
            settings.contextualizer,
            settings.conversation.history_max_turns,
        )

        logger.info("Initialising answer validator...")
        answer_validator = AnswerValidator(
            settings.openai_api_key,
            settings.validator,
        )

        logger.info("Compiling query graph...")
        checkpoint_path = Path(settings.conversation.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(
            str(checkpoint_path)
        ) as checkpointer:
            app.state.query_graph = build_query_graph(
                qa_agent,
                query_contextualizer,
                query_classifier,
                answer_validator,
                checkpointer,
            )
            yield
    finally:
        if qa_agent is not None:
            qa_agent.close()
        shutdown_observability()


app = FastAPI(title="Document Knowledge Base API", lifespan=lifespan)


async def _safe_answer(
    query: str,
    conversation_id: UUID,
    app: FastAPI,
) -> PlainTextResponse:
    """Run the buffered graph and return only a complete validated response."""
    with observation(
        name="kb.query",
        input={"query": query},
        metadata={
            "endpoint": "/query",
            "conversation_id": str(conversation_id),
        },
    ) as root_observation:
        trace_id = current_trace_id()
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Conversation-Id": str(conversation_id),
        }
        if trace_id:
            headers["X-Langfuse-Trace-Id"] = trace_id

        try:
            result = await app.state.query_graph.ainvoke(
                {"query": query},
                config={
                    "configurable": {
                        "thread_id": str(conversation_id),
                    }
                },
            )
            response = result.get("response")
            if not response:
                raise RuntimeError("Query graph returned no response")

            if result.get("contextualization_failed"):
                outcome = "contextualization_failed"
            elif result.get("classification_failed"):
                outcome = "classification_failed"
            else:
                if result.get("safety") == "unsafe":
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
                if outcome in {
                    "contextualization_failed",
                    "classification_failed",
                    "validation_unavailable",
                }:
                    root_observation.update(
                        level="WARNING",
                        status_message=outcome,
                    )
                if validation is not None and not result.get("validation_failed"):
                    root_observation.score_trace(
                        name="faithfulness",
                        value=1 if validation["faithfulness"] else 0,
                        data_type="BOOLEAN",
                        comment=validation["faithfulness_reason"],
                    )
                    root_observation.score_trace(
                        name="context_sufficiency",
                        value=1 if validation["context_sufficiency"] else 0,
                        data_type="BOOLEAN",
                        comment=validation["context_sufficiency_reason"],
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
async def query_documents(
    request: Request,
    payload: QueryRequest,
) -> PlainTextResponse:
    """Return the complete grounded answer and its validation results."""
    return await _safe_answer(payload.query, payload.conversation_id, request.app)


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
