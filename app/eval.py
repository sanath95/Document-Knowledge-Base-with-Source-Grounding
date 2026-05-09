from __future__ import annotations

from dataclasses import dataclass

from app.config import load_settings
from app.pipeline import ask
from app.vector_store import collection_exists


@dataclass(frozen=True)
class SmokeCase:
    name: str
    question: str
    expect_sources: bool
    expect_abstention: bool = False


CASES = [
    SmokeCase("basic-test1", "What is Test1.pdf about?", True),
    SmokeCase("test1-reporting", "What does Test1.pdf say about reporting?", True),
    SmokeCase("basic-test2", "What is the dissertation in Test2.pdf about?", True),
    SmokeCase("exact-term-bm25", "Which document mentions layout optimization?", True),
    SmokeCase("abstention", "What does the material say about password rotation?", False, True),
]


def main() -> None:
    settings = load_settings()
    if not collection_exists(settings):
        raise SystemExit("No index found. Run `uv run python -m app.ingest` first.")

    failures = 0
    for case in CASES:
        print(f"\n=== {case.name} ===")
        answer = ask(settings, case.question)
        print(answer.text[:1200])
        has_sources = bool(answer.evidence)
        if case.expect_sources and not has_sources:
            print("FAIL: expected at least one source")
            failures += 1
        if case.expect_sources and not any(item.document and item.page for item in answer.evidence):
            print("FAIL: expected document/page citation metadata")
            failures += 1
        if case.expect_abstention and not answer.abstained:
            print("FAIL: expected strict abstention")
            failures += 1
        if case.expect_sources and not answer.text.strip():
            print("FAIL: expected non-empty answer")
            failures += 1

    if failures:
        raise SystemExit(f"{failures} smoke-test check(s) failed")
    print("\nAll smoke-test checks passed.")


if __name__ == "__main__":
    main()

