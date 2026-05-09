from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    document: str
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document: str
    page: int
    chunk_index: int
    text: str


@dataclass(frozen=True)
class Evidence:
    id: str
    document: str
    page: int
    text: str
    score: float
    source: str


@dataclass(frozen=True)
class Answer:
    text: str
    evidence: list[Evidence]
    abstained: bool

