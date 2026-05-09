from __future__ import annotations

from collections.abc import Iterable

from openai import OpenAI

from app.config import Settings


def make_client(settings: Settings) -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or your shell environment.")
    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(client: OpenAI, texts: Iterable[str], model: str, batch_size: int = 64) -> list[list[float]]:
    items = list(texts)
    embeddings: list[list[float]] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend([item.embedding for item in response.data])
    return embeddings

