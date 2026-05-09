from __future__ import annotations

import hashlib
import re

from app.models import Chunk, PageText


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        part = " ".join(words[start : start + chunk_size])
        chunks.append(part)
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def build_chunks(pages: list[PageText], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for index, text in enumerate(_chunk_text(page.text, chunk_size, overlap)):
            digest = hashlib.sha1(f"{page.document}:{page.page}:{index}:{text[:80]}".encode()).hexdigest()
            chunks.append(
                Chunk(
                    id=f"{page.document}:{page.page}:{index}:{digest[:10]}",
                    document=page.document,
                    page=page.page,
                    chunk_index=index,
                    text=text,
                )
            )
    return chunks

