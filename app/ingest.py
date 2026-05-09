from __future__ import annotations

from app.chunking import build_chunks
from app.config import load_settings
from app.openai_client import embed_texts, make_client
from app.pdf_loader import iter_pdf_pages
from app.vector_store import add_chunks, recreate_collection


def main() -> None:
    settings = load_settings()
    if not settings.docs_dir.exists():
        raise SystemExit(f"Documents folder not found: {settings.docs_dir}")
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Add it to .env or your shell environment.")
    client = make_client(settings)

    pages = iter_pdf_pages(settings.docs_dir)
    if not pages:
        raise SystemExit(f"No extractable PDF text found in: {settings.docs_dir}")

    chunks = build_chunks(pages, settings.chunk_size, settings.chunk_overlap)
    print(f"Loaded {len(pages)} pages from {settings.docs_dir}")
    print(f"Built {len(chunks)} chunks")

    embeddings = embed_texts(client, [chunk.text for chunk in chunks], settings.embedding_model)
    collection = recreate_collection(settings)
    add_chunks(collection, chunks, embeddings)
    print(f"Indexed {collection.count()} chunks in {settings.chroma_dir}")


if __name__ == "__main__":
    main()
