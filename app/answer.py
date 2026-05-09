from __future__ import annotations

from app.config import Settings
from app.models import Answer, Evidence
from app.openai_client import make_client


ABSTENTION = (
    "I do not have enough evidence in the indexed documents to answer that question confidently."
)


def _format_evidence(evidence: list[Evidence]) -> str:
    blocks = []
    for index, item in enumerate(evidence, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{index}] Document: {item.document}",
                    f"Page: {item.page}",
                    f"Text: {item.text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _source_markdown(evidence: list[Evidence]) -> str:
    if not evidence:
        return ""
    parts = ["\n\n### Sources"]
    for item in evidence:
        snippet = item.text.replace("\n", " ")
        if len(snippet) > 550:
            snippet = snippet[:550].rsplit(" ", 1)[0] + "..."
        parts.append(f"- **{item.document}, page {item.page}**  \n  {snippet}")
    return "\n".join(parts)


def answer_question(settings: Settings, question: str, evidence: list[Evidence]) -> Answer:
    if not evidence:
        return Answer(text=ABSTENTION, evidence=[], abstained=True)

    client = make_client(settings)
    instructions = """You answer questions about a small PDF knowledge base.
Use only the provided evidence. Do not use outside knowledge.
If the evidence is insufficient or ambiguous, say that you do not have enough evidence.
Answer in the same language as the user's question when practical.
Every factual claim must be supported by the evidence.
Include source references in prose using the format (Document, page N)."""
    prompt = f"""Question:
{question}

Evidence:
{_format_evidence(evidence)}

Write a concise answer grounded only in the evidence."""
    response = client.responses.create(
        model=settings.openai_model,
        instructions=instructions,
        input=prompt,
    )
    text = response.output_text.strip()
    abstained = "not have enough evidence" in text.lower() or "insufficient" in text.lower()
    return Answer(text=text + _source_markdown(evidence), evidence=evidence, abstained=abstained)

