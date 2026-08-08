"""LangGraph workflow that guards and routes document questions."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from operator import add
from typing import Annotated, Literal, NotRequired, Protocol, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from orchestration.answer_validator import AnswerValidation
from orchestration.query_classifier import QuerySafetyAssessment
from utils.models import ConversationTurn, EvidenceChunk, QAResult

logger = logging.getLogger(__name__)

_UNSAFE_RESPONSE = (
    "I can't help with that request. You can ask a question about the indexed "
    "documents."
)
_CLASSIFICATION_FAILURE_RESPONSE = (
    "I couldn't safely classify that request. Please rephrase it and try again."
)
_CONTEXTUALIZATION_FAILURE_RESPONSE = (
    "I couldn't resolve that follow-up from the conversation. Please restate "
    "the question with the relevant document or subject."
)


class AnswerGenerator(Protocol):
    """Interface implemented by the existing Pydantic-AI wrapper."""

    async def generate_answer(
        self,
        query: str,
        history: Sequence[ConversationTurn],
    ) -> QAResult: ...


class QueryClassifierProtocol(Protocol):
    """Interface implemented by the tool-less safety classifier."""

    async def classify(
        self,
        query: str,
        resolved_query: str | None = None,
        history: Sequence[ConversationTurn] = (),
    ) -> QuerySafetyAssessment: ...


class QueryContextualizerProtocol(Protocol):
    """Interface implemented by the tool-less follow-up contextualizer."""

    async def contextualize(
        self,
        query: str,
        history: Sequence[ConversationTurn],
    ) -> str: ...


class AnswerValidatorProtocol(Protocol):
    """Interface implemented by the schema-constrained answer validator."""

    async def validate(
        self,
        query: str,
        retrieved_chunks: tuple[EvidenceChunk, ...],
        answer: str,
    ) -> AnswerValidation: ...


class QueryState(TypedDict):
    """State carried by the query workflow."""

    query: str
    history: Annotated[list[ConversationTurn], add]
    resolved_query: NotRequired[str]
    contextualization_failed: NotRequired[bool]
    safety: NotRequired[Literal["safe", "unsafe"]]
    classification_failed: NotRequired[bool]
    answer: NotRequired[str]
    retrieved_chunks: NotRequired[list[PersistedEvidenceChunk]]
    validation: NotRequired[PersistedValidation]
    validation_failed: NotRequired[bool]
    response: NotRequired[str]


class PersistedEvidenceChunk(TypedDict):
    """JSON-compatible evidence representation stored in checkpoints."""

    chunk_id: str
    document: str
    metadata: dict[str, str | int]


class PersistedValidation(TypedDict):
    """JSON-compatible validation representation stored in checkpoints."""

    faithfulness: bool
    faithfulness_reason: str | None
    context_sufficiency: bool
    context_sufficiency_reason: str | None


def build_query_graph(
    answer_generator: AnswerGenerator,
    query_contextualizer: QueryContextualizerProtocol,
    query_classifier: QueryClassifierProtocol,
    answer_validator: AnswerValidatorProtocol,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Compile a graph that guards, answers, validates, and formats queries."""

    async def contextualize_query(state: QueryState) -> dict:
        history = state.get("history", ())
        if not history:
            return {
                "resolved_query": state["query"],
                "contextualization_failed": False,
            }

        try:
            resolved_query = await query_contextualizer.contextualize(
                state["query"],
                history,
            )
        except Exception:
            logger.exception("Query contextualization failed")
            return {"contextualization_failed": True}
        return {
            "resolved_query": resolved_query,
            "contextualization_failed": False,
        }

    def route_contextualization(
        state: QueryState,
    ) -> Literal["classify_query", "write_contextualization_failure"]:
        if state.get("contextualization_failed"):
            return "write_contextualization_failure"
        return "classify_query"

    async def write_contextualization_failure(_state: QueryState) -> dict:
        return {"response": _CONTEXTUALIZATION_FAILURE_RESPONSE}

    async def classify_query(state: QueryState) -> dict:
        try:
            safety_assessment = await query_classifier.classify(
                state["query"],
                state.get("resolved_query", state["query"]),
                state.get("history", ()),
            )
        except Exception:
            logger.exception("Query classification failed")
            return {"classification_failed": True}
        return {
            "safety": safety_assessment.safety,
            "classification_failed": False,
        }

    def route_query(
        state: QueryState,
    ) -> Literal["run_qa_agent", "write_guardrail_response"]:
        if state.get("classification_failed"):
            return "write_guardrail_response"

        if state.get("safety") == "safe":
            return "run_qa_agent"
        return "write_guardrail_response"

    async def write_guardrail_response(state: QueryState) -> dict:
        if state.get("classification_failed"):
            response = _CLASSIFICATION_FAILURE_RESPONSE
        else:
            safety = state.get("safety")
            if safety is None:
                response = _CLASSIFICATION_FAILURE_RESPONSE
            elif safety == "unsafe":
                response = _UNSAFE_RESPONSE
            else:
                raise RuntimeError("Safe query was routed to the guardrail response")

        return {"response": response}

    async def run_qa_agent(state: QueryState) -> dict:
        result = await answer_generator.generate_answer(
            state.get("resolved_query", state["query"]),
            state.get("history", ()),
        )
        return {
            "answer": result.answer,
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "document": chunk.document,
                    "metadata": dict(chunk.metadata),
                }
                for chunk in result.retrieved_chunks
            ],
        }

    async def validate_answer(state: QueryState) -> dict:
        try:
            validation = await answer_validator.validate(
                query=state.get("resolved_query", state["query"]),
                retrieved_chunks=tuple(
                    EvidenceChunk(
                        chunk_id=chunk["chunk_id"],
                        document=chunk["document"],
                        metadata=dict(chunk["metadata"]),
                    )
                    for chunk in state.get("retrieved_chunks", ())
                ),
                answer=state.get("answer", ""),
            )
        except Exception:
            logger.exception("Answer validation failed")
            return {"validation_failed": True}
        return {
            "validation": validation.model_dump(),
            "validation_failed": False,
        }

    async def write_validated_response(state: QueryState) -> dict:
        answer = state.get("answer", "").rstrip()
        if state.get("validation_failed"):
            validation_text = (
                "Validation\n"
                "- Faithful to retrieved documents: unavailable\n"
                "- Context sufficient to fully answer the query: unavailable\n"
                "- Reason: Validation could not be completed."
            )
        else:
            validation = state.get("validation")
            if validation is None:
                raise RuntimeError("Validation result is missing")

            lines = [
                "Validation",
                "- Faithful to retrieved documents: "
                f"{str(validation['faithfulness']).lower()}",
            ]
            if not validation["faithfulness"]:
                lines.append(
                    f"- Faithfulness reason: {validation['faithfulness_reason']}"
                )

            lines.append(
                "- Context sufficient to fully answer the query: "
                f"{str(validation['context_sufficiency']).lower()}"
            )
            if not validation["context_sufficiency"]:
                lines.append(
                    "- Context sufficiency reason: "
                    f"{validation['context_sufficiency_reason']}"
                )
            validation_text = "\n".join(lines)

        return {
            "response": f"{answer}\n\n{validation_text}",
            "history": [
                {
                    "user": state["query"],
                    "assistant": answer,
                }
            ],
        }

    builder = StateGraph(QueryState)
    builder.add_node("contextualize_query", contextualize_query)
    builder.add_node("classify_query", classify_query)
    builder.add_node("run_qa_agent", run_qa_agent)
    builder.add_node("validate_answer", validate_answer)
    builder.add_node("write_validated_response", write_validated_response)
    builder.add_node("write_guardrail_response", write_guardrail_response)
    builder.add_node(
        "write_contextualization_failure",
        write_contextualization_failure,
    )
    builder.add_edge(START, "contextualize_query")
    builder.add_conditional_edges("contextualize_query", route_contextualization)
    builder.add_conditional_edges("classify_query", route_query)
    builder.add_edge("run_qa_agent", "validate_answer")
    builder.add_edge("validate_answer", "write_validated_response")
    builder.add_edge("write_validated_response", END)
    builder.add_edge("write_guardrail_response", END)
    builder.add_edge("write_contextualization_failure", END)
    return builder.compile(checkpointer=checkpointer)
