"""LangGraph workflow that guards and routes document questions."""

from __future__ import annotations

from typing import Literal, NotRequired, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from orchestration.answer_validator import AnswerValidation
from orchestration.query_classifier import QueryAssessment
from utils.logging import get_logger
from utils.models import EvidenceChunk, QAResult

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


class AnswerGenerator(Protocol):
    """Interface implemented by the existing Pydantic-AI wrapper."""

    async def generate_answer(self, query: str) -> QAResult: ...


class QueryClassifierProtocol(Protocol):
    """Interface implemented by the tool-less input classifier."""

    async def classify(self, query: str) -> QueryAssessment: ...


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
    assessment: NotRequired[QueryAssessment]
    classification_failed: NotRequired[bool]
    answer: NotRequired[str]
    retrieved_chunks: NotRequired[tuple[EvidenceChunk, ...]]
    validation: NotRequired[AnswerValidation]
    validation_failed: NotRequired[bool]
    response: NotRequired[str]
    completed: NotRequired[bool]


def build_query_graph(
    answer_generator: AnswerGenerator,
    query_classifier: QueryClassifierProtocol,
    answer_validator: AnswerValidatorProtocol,
):
    """Compile a graph that guards, answers, validates, and formats queries."""

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

    async def write_guardrail_response(state: QueryState) -> dict:
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

        return {"response": response, "completed": True}

    async def run_qa_agent(state: QueryState) -> dict:
        result = await answer_generator.generate_answer(state["query"])
        return {
            "answer": result.answer,
            "retrieved_chunks": result.retrieved_chunks,
        }

    async def validate_answer(state: QueryState) -> dict:
        try:
            validation = await answer_validator.validate(
                query=state["query"],
                retrieved_chunks=state.get("retrieved_chunks", ()),
                answer=state.get("answer", ""),
            )
        except Exception:
            logger.exception("Answer validation failed")
            return {"validation_failed": True}
        return {"validation": validation, "validation_failed": False}

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
                f"{str(validation.faithfulness).lower()}",
            ]
            if not validation.faithfulness:
                lines.append(
                    f"- Faithfulness reason: {validation.faithfulness_reason}"
                )

            lines.append(
                "- Context sufficient to fully answer the query: "
                f"{str(validation.context_sufficiency).lower()}"
            )
            if not validation.context_sufficiency:
                lines.append(
                    "- Context sufficiency reason: "
                    f"{validation.context_sufficiency_reason}"
                )
            validation_text = "\n".join(lines)

        return {
            "response": f"{answer}\n\n{validation_text}",
            "completed": True,
        }

    builder = StateGraph(QueryState)
    builder.add_node("classify_query", classify_query)
    builder.add_node("run_qa_agent", run_qa_agent)
    builder.add_node("validate_answer", validate_answer)
    builder.add_node("write_validated_response", write_validated_response)
    builder.add_node("write_guardrail_response", write_guardrail_response)
    builder.add_edge(START, "classify_query")
    builder.add_conditional_edges("classify_query", route_query)
    builder.add_edge("run_qa_agent", "validate_answer")
    builder.add_edge("validate_answer", "write_validated_response")
    builder.add_edge("write_validated_response", END)
    builder.add_edge("write_guardrail_response", END)
    return builder.compile()
