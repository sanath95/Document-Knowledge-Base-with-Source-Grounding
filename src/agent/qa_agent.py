"""
QAAgent
───────
Pydantic-AI agent wired with the retrieve_and_rerank tool.
All external dependencies are injected via AgentDeps — no module-level singletons.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Optional

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModelSettings

from config.settings import AgentConfig, Settings
from ingestion.embedder import Embedder
from retrieval.bm25_retriever import BM25Retriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore
from utils.logging import get_logger
from utils.models import RetrievedChunk

logger = get_logger(__name__)

_AGENT_INSTRUCTIONS = """
You are a retrieval-augmented document QA agent.

Your job:
1. Understand the user's question.
2. Formulate effective semantic search queries.
3. Use retrieve_and_rerank to gather evidence.
4. Answer ONLY from retrieved evidence.

Retrieval strategy:
- Rewrite vague questions into precise retrieval queries while preserving exact
  identifiers, names, and domain terms that are useful for keyword search.
- Break multi-part questions into separate focused searches.
- For comparisons retrieve evidence for each item independently.
- Prefer multiple targeted searches over one broad search.
- Retry retrieval with reformulated queries only if evidence is weak or irrelevant.
- Stop retrieving once sufficient evidence is collected.

Grounding rules:
- Never use outside knowledge.
- Every factual statement must be supported by retrieved evidence.
- Cite sources inline using: [pdf_name, page X]
- If evidence is incomplete, uncertain, or conflicting: explicitly say so.
- If the documents do not contain the answer: say that clearly.

Answer style:
- Be concise but complete.
- Prefer bullet points for multi-part answers.
- Summarise repeated evidence instead of quoting excessively.
"""


@dataclass
class AgentDeps:
    """All runtime dependencies required by the agent's tools."""
    vector_store: VectorStore
    embedder: Embedder
    reranker: Reranker
    bm25_retriever: BM25Retriever
    dense_top_k: int = 25
    bm25_top_k: int = 25
    fusion_top_k: int = 25
    final_top_k: int = 10
    rrf_k: int = 60


def build_agent(config: AgentConfig) -> Agent[AgentDeps]:
    """
    Construct and return the pydantic-ai Agent.

    Tools are registered here as closures so no global state is required.

    Args:
        config: Agent configuration (model, temperature, etc.).

    Returns:
        A fully configured Agent[AgentDeps].
    """
    agent: Agent[AgentDeps] = Agent(
        config.llm_model,
        deps_type=AgentDeps,
        instructions=_AGENT_INSTRUCTIONS,
        model_settings=OpenAIModelSettings(
            temperature=config.temperature,
            parallel_tool_calls=config.parallel_tool_calls,
        ),
    )

    @agent.tool
    async def retrieve_and_rerank(
        ctx: RunContext[AgentDeps],
        search_query: str,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Retrieve, fuse, and rerank document chunks relevant to *search_query*.

        Args:
            search_query:
                Optimised semantic retrieval query.
            filters:
                Optional ChromaDB metadata filter.
                Examples:
                  {"pdf_name": "report.pdf"}
                  {"pdf_name": {"$in": ["a.pdf", "b.pdf"]}}
                  {"page_number": {"$gte": 10}}

        Returns a hybrid-reranked list of chunks with 'document', 'metadata',
        and cross-encoder 'score'.
        """
        query_embedding = await ctx.deps.embedder.embed_one(search_query)

        dense_candidates: list[RetrievedChunk] = ctx.deps.vector_store.query(
            query_embedding=query_embedding,
            top_k=ctx.deps.dense_top_k,
            filters=filters,
        )

        sparse_candidates: list[RetrievedChunk] = ctx.deps.bm25_retriever.query(
            query=search_query,
            top_k=ctx.deps.bm25_top_k,
            filters=filters,
        )

        fused_candidates: list[RetrievedChunk] = reciprocal_rank_fusion(
            [dense_candidates, sparse_candidates],
            rrf_k=ctx.deps.rrf_k,
        )[: ctx.deps.fusion_top_k]

        reranked: list[RetrievedChunk] = await ctx.deps.reranker.rerank(
            query=search_query,
            chunks=fused_candidates,
        )

        return [
            {
                "document": chunk.document,
                "metadata": chunk.metadata,
                "score": chunk.score,
            }
            for chunk in reranked[: ctx.deps.final_top_k]
        ]

    return agent


class QAAgent:
    """
    High-level wrapper around the pydantic-ai Agent.
    Owns the AgentDeps and exposes a clean run interface.

    Args:
        settings: Global application settings.
    """

    def __init__(self, settings: Settings) -> None:
        self._vector_store = VectorStore(settings.chroma)
        self._embedder = Embedder(settings.openai_api_key, settings.embedding)
        self._reranker = Reranker(settings.reranker)
        self._bm25_retriever = BM25Retriever(
            collection=self._vector_store.collection,
            tokenizer_model_name=settings.reranker.model_name,
            tokenizer_cache_dir=settings.reranker.cache_dir,
        )
        self._agent = build_agent(settings.agent)
        self._deps = AgentDeps(
            vector_store=self._vector_store,
            embedder=self._embedder,
            reranker=self._reranker,
            bm25_retriever=self._bm25_retriever,
            dense_top_k=settings.agent.dense_top_k,
            bm25_top_k=settings.agent.bm25_top_k,
            fusion_top_k=settings.agent.fusion_top_k,
            final_top_k=settings.agent.final_top_k,
            rrf_k=settings.agent.rrf_k,
        )

    async def stream_answer(self, query: str) -> AsyncGenerator[str, None]:
        """Stream only the final answer text after the agent runs its tools."""
        async with self._agent.run_stream(query, deps=self._deps) as result:
            async for chunk in result.stream_text(delta=True):
                yield chunk

    @property
    def vector_store(self) -> VectorStore:
        """Expose VectorStore so the ingestion pipeline can share it."""
        return self._vector_store
