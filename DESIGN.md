# Design Note

## Architecture

This project implements a deterministic retrieval-augmented generation pipeline for the Semorai Task B take-home. It is intentionally not an autonomous agent. The application follows a fixed flow: extract PDF text, chunk it with source metadata, embed chunks, retrieve evidence for a user question, and ask an OpenAI model to synthesize an answer only from that evidence.

The source corpus is every PDF in `Technical Interview Docs`, except the task brief itself. Ingestion is explicit so the live Gradio app can start quickly and predictably during the interview.

## Retrieval And Grounding

Each chunk stores its document name, page number, chunk index, and text in a persistent ChromaDB collection. The system uses OpenAI embeddings for vector search and a BM25 keyword retriever for exact terms, titles, acronyms, and wording that semantic retrieval may miss. Candidates from both retrievers are merged and deduplicated.

A local cross-encoder reranker is optional. It is disabled by default because it adds a large model dependency and may require a cached model. If enabled but unavailable, retrieval falls back to vector plus BM25 without failing the demo.

## Answer Synthesis

The OpenAI Responses API receives the user question and only the retrieved evidence. The system prompt requires the model to answer from evidence, cite document/page references, and abstain when evidence is insufficient or ambiguous. The UI also displays source snippets below the generated answer so interviewers can inspect the grounding directly.

## Tradeoffs

- ChromaDB was chosen over pgvector, Qdrant, JSON, or SQLite because it provides local persistent vector search with minimal setup.
- Gradio `Blocks` was chosen over Streamlit because the task is chat-oriented while still needing source cards and status messages.
- Explicit pre-indexing was chosen over auto-indexing on app launch to make demo startup reliable.
- Confidence scores were intentionally left out of v1 to avoid implying calibration that the system does not actually prove.

## Failure Modes

- Poor PDF text extraction can reduce retrieval quality. The current provided PDFs are mostly text-extractable.
- The system may retrieve plausible but insufficient evidence for broad or vague questions. The answer prompt and eval script check for abstention behavior.
- Reranking is optional and local-only by default, so the core app does not depend on a large model download.
- The app requires network/API access for OpenAI embeddings during ingestion and answer generation during Q&A.

