"""Structured validation of an answer against its retrieved evidence."""

from __future__ import annotations

import json

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, model_validator

from config.settings import ValidatorConfig
from utils.models import EvidenceChunk
from utils.observability import observation, openai_usage_details


_VALIDATOR_INSTRUCTIONS = """
You are an evidence validator for a retrieval-augmented document QA system.
Do not answer the query.

Evaluate exactly two properties:

1. faithfulness: true only when every material factual claim in the answer is
   supported by, or is a reasonable direct inference from, the retrieved chunks.
   Set it to false if a material claim is unsupported or contradicts the chunks.

2. context_sufficiency: true only when the retrieved chunks contain enough
   information to fully answer every material part of the query. This property
   concerns the evidence, not the quality or wording of the answer.

An answer that correctly says the documents are insufficient can be faithful
while context_sufficiency is false.

For each false property, provide a concise, evidence-specific reason. For each
true property, its reason must be null. Do not use outside knowledge.
""".strip()


class AnswerValidation(BaseModel):
    """Boolean evidence checks returned by the validator model."""

    model_config = ConfigDict(extra="forbid")

    faithfulness: bool
    faithfulness_reason: str | None
    context_sufficiency: bool
    context_sufficiency_reason: str | None

    @model_validator(mode="after")
    def reasons_match_results(self) -> "AnswerValidation":
        pairs = (
            ("faithfulness", self.faithfulness, self.faithfulness_reason),
            (
                "context_sufficiency",
                self.context_sufficiency,
                self.context_sufficiency_reason,
            ),
        )
        for name, passed, reason in pairs:
            if passed and reason is not None:
                raise ValueError(f"{name}_reason must be null when {name} is true")
            if not passed and (reason is None or not reason.strip()):
                raise ValueError(f"{name}_reason is required when {name} is false")
        return self


class AnswerValidator:
    """Validate a generated answer using only its retrieved chunks."""

    def __init__(
        self,
        api_key: str,
        config: ValidatorConfig,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI(api_key=api_key)
        self._config = config

    async def validate(
        self,
        query: str,
        retrieved_chunks: tuple[EvidenceChunk, ...],
        answer: str,
    ) -> AnswerValidation:
        """Return schema-constrained boolean validation results."""
        payload = {
            "query": query,
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_pdf": chunk.source_pdf,
                    "page_number": chunk.page_number,
                    "document": chunk.document,
                    "section_headers": {
                        key: chunk.metadata[key]
                        for key in ("h1", "h2", "h3")
                        if key in chunk.metadata
                    },
                }
                for chunk in retrieved_chunks
            ],
            "answer": answer,
        }
        observation_input = {
            "query": query,
            "answer": answer,
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_pdf": chunk.source_pdf,
                    "page_number": chunk.page_number,
                }
                for chunk in retrieved_chunks
            ],
        }
        with observation(
            name="answer.validate",
            as_type="generation",
            input=observation_input,
            model=self._config.model,
        ) as generation:
            response = await self._client.responses.parse(
                model=self._config.model,
                instructions=_VALIDATOR_INSTRUCTIONS,
                input=json.dumps(payload, ensure_ascii=False),
                text_format=AnswerValidation,
                max_output_tokens=512,
                store=False,
            )
            validation = response.output_parsed
            if validation is None:
                raise RuntimeError("Answer validator returned no structured assessment")
            if generation is not None:
                generation.update(
                    output=validation.model_dump(),
                    usage_details=openai_usage_details(response.usage),
                )
            return validation
