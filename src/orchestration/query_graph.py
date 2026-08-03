"""One-node LangGraph workflow for document questions."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import NotRequired, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter


class AnswerStreamer(Protocol):
    """Interface implemented by the existing Pydantic-AI wrapper."""

    def stream_answer(self, query: str) -> AsyncGenerator[str, None]: ...


class QueryState(TypedDict):
    """State carried by the query workflow."""

    query: str
    completed: NotRequired[bool]


def build_query_graph(answer_streamer: AnswerStreamer):
    """Compile a graph that streams one Pydantic-AI answer."""

    async def run_qa_agent(
        state: QueryState,
        writer: StreamWriter,
    ) -> dict[str, bool]:
        async for chunk in answer_streamer.stream_answer(state["query"]):
            writer(chunk)
        return {"completed": True}

    builder = StateGraph(QueryState)
    builder.add_node("run_qa_agent", run_qa_agent)
    builder.add_edge(START, "run_qa_agent")
    builder.add_edge("run_qa_agent", END)
    return builder.compile()
