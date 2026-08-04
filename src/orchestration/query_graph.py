"""LangGraph workflow that guards and routes document questions."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Literal, NotRequired, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter

from orchestration.query_classifier import QueryAssessment
from utils.logging import get_logger

logger = get_logger(__name__)

_UNSAFE_RESPONSE = (
    "I can't help with that request. You can ask a question about the indexed "
    "documents."
)
_OUT_OF_SCOPE_RESPONSE = (
    "I can answer questions about the indexed documents. Please ask about their "
    "content, findings, or sources."
)
_AMBIGUOUS_RESPONSE = (
    "Please clarify what you would like to know from the indexed documents."
)
_CLASSIFICATION_FAILURE_RESPONSE = (
    "I couldn't safely classify that request. Please rephrase it as a question "
    "about the indexed documents."
)


class AnswerStreamer(Protocol):
    """Interface implemented by the existing Pydantic-AI wrapper."""

    def stream_answer(self, query: str) -> AsyncGenerator[str, None]: ...


class QueryClassifierProtocol(Protocol):
    """Interface implemented by the tool-less input classifier."""

    async def classify(self, query: str) -> QueryAssessment: ...


class QueryState(TypedDict):
    """State carried by the query workflow."""

    query: str
    assessment: NotRequired[QueryAssessment]
    classification_failed: NotRequired[bool]
    completed: NotRequired[bool]


def build_query_graph(
    answer_streamer: AnswerStreamer,
    query_classifier: QueryClassifierProtocol,
):
    """Compile a graph that admits only safe, in-scope queries to the agent."""

    async def classify_query(state: QueryState) -> dict:
        try:
            assessment = await query_classifier.classify(state["query"])
        except Exception:
            logger.exception("Query classification failed")
            return {"classification_failed": True}
        return {"assessment": assessment, "classification_failed": False}

    def route_query(
        state: QueryState,
    ) -> Literal["run_qa_agent", "write_guardrail_response"]:
        if state.get("classification_failed"):
            return "write_guardrail_response"

        assessment = state.get("assessment")
        if (
            assessment is not None
            and assessment.safety == "safe"
            and assessment.scope == "in_scope"
        ):
            return "run_qa_agent"
        return "write_guardrail_response"

    async def write_guardrail_response(
        state: QueryState,
        writer: StreamWriter,
    ) -> dict[str, bool]:
        if state.get("classification_failed"):
            response = _CLASSIFICATION_FAILURE_RESPONSE
        else:
            assessment = state.get("assessment")
            if assessment is None:
                response = _CLASSIFICATION_FAILURE_RESPONSE
            elif assessment.safety == "unsafe":
                response = _UNSAFE_RESPONSE
            elif assessment.scope == "ambiguous":
                response = _AMBIGUOUS_RESPONSE
            else:
                response = _OUT_OF_SCOPE_RESPONSE

        writer(response)
        return {"completed": True}

    async def run_qa_agent(
        state: QueryState,
        writer: StreamWriter,
    ) -> dict[str, bool]:
        async for chunk in answer_streamer.stream_answer(state["query"]):
            writer(chunk)
        return {"completed": True}

    builder = StateGraph(QueryState)
    builder.add_node("classify_query", classify_query)
    builder.add_node("run_qa_agent", run_qa_agent)
    builder.add_node("write_guardrail_response", write_guardrail_response)
    builder.add_edge(START, "classify_query")
    builder.add_conditional_edges("classify_query", route_query)
    builder.add_edge("run_qa_agent", END)
    builder.add_edge("write_guardrail_response", END)
    return builder.compile()
