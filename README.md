# Document Knowledge Base with Source Grounding

Semorai AI Engineer take-home implementation for Task B: source-grounded Q&A over PDFs.

The app indexes every PDF in `Technical Interview Docs` except the task brief, then lets an interviewer ask natural-language questions in a Gradio web UI. Answers are generated from retrieved evidence and include document/page citations plus supporting snippets.

## Setup

Install dependencies with `uv`:

```powershell
uv sync
```

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set:

```text
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-5.2
```

`OPENAI_MODEL` is configurable in case model access differs on interview day.

## Build The Index

Run ingestion before the demo:

```powershell
uv run python -m app.ingest
```

This extracts PDF text page-by-page, chunks it, embeds it with OpenAI, and writes a persistent ChromaDB index to `.chroma`.

## Run The Web App

```powershell
uv run python -m app.ui
```

The app will show a clear message if `.env` is missing `OPENAI_API_KEY` or if the Chroma index has not been built yet.

## Run Smoke Tests

```powershell
uv run python -m app.eval
```

The smoke tests ask representative questions, verify that grounded answers include sources, and check that unsupported questions abstain instead of hallucinating.

## Optional Reranking

The core retrieval pipeline combines Chroma vector search with BM25 keyword search. A local cross-encoder reranker can be enabled if the model is already installed or cached:

```powershell
uv sync --extra rerank
```

Then set:

```text
ENABLE_RERANKER=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

If the reranker cannot load, the app falls back to vector plus BM25 retrieval.

## Interview Demo Checklist

1. Run `uv sync`.
2. Add `OPENAI_API_KEY` to `.env`.
3. Run `uv run python -m app.ingest`.
4. Run `uv run python -m app.ui`.
5. Ask multiple questions in one session.
6. Confirm answers include source document names, page numbers, and snippets.
7. Ask an unsupported question and confirm the app abstains.

## Design

See [DESIGN.md](DESIGN.md) for architecture, tradeoffs, and failure modes.
