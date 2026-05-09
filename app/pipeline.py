from __future__ import annotations

from app.answer import answer_question
from app.config import Settings
from app.models import Answer
from app.retrieval import retrieve


def ask(settings: Settings, question: str) -> Answer:
    evidence = retrieve(settings, question)
    return answer_question(settings, question, evidence)

