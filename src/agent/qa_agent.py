"""
QAAgent
───────
Pydantic-AI agent wired with retrieve_and_rerank and list_documents tools.
All external dependencies are injected via AgentDeps — no module-level singletons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModelSettings

from config.settings import AgentConfig, Settings
from ingestion.embedder import Embedder
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore
from utils.logging import get_logger
from utils.models import DocumentIndex, RetrievalFilter, RetrievedChunk

logger = get_logger(__name__)

_AGENT_INSTRUCTIONS = """
You are a retrieval-augmented document QA agent.

Your job:
1. Understand the user's question.
2. Formulate effective semantic search queries.
3. Use retrieve_and_rerank to gather evidence.
4. Answer ONLY from retrieved evidence.

Retrieval strategy:
- Rewrite vague questions into precise retrieval queries.
- Break multi-part questions into separate focused searches.
- For comparisons retrieve evidence for each item independently.
- Prefer multiple targeted searches over one broad search.
- Retry retrieval with reformulated queries only if evidence is weak or irrelevant.
- Stop retrieving once sufficient evidence is collected.

Document selection:
- Use list_documents when:
  - the user asks what documents are available,
  - the document reference is ambiguous,
  - the user asks about a specific document that may not exist,
  - or the user asks for comparisons between documents.

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
    hybrid_retriever: HybridRetriever


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
    def list_documents(ctx: RunContext[AgentDeps]) -> list[dict]:
        """
        List all indexed PDF documents in the knowledge base.

        Use this tool when:
        - the user asks what documents are available,
        - a document name is ambiguous,
        - retrieval should be scoped to a specific document,
        - or the user asks for comparisons between documents.

        Returns a list of dicts with 'pdf_name' and 'chunks'.
        """
        documents: list[DocumentIndex] = ctx.deps.vector_store.list_documents()
        return [
            {"pdf_name": doc.pdf_name, "chunks": doc.chunk_count}
            for doc in documents
        ]

    @agent.tool
    async def retrieve_and_rerank(
        ctx: RunContext[AgentDeps],
        search_query: str,
        filters: RetrievalFilter | None = None,
    ) -> list[dict]:
        """
        Retrieve and rerank document chunks relevant to *search_query*.

        Args:
            search_query:
                Optimised semantic retrieval query.
            filters:
                Optional document/page filter shared by dense and sparse search.
                Examples:
                  {"pdf_names": ["report.pdf"]}
                  {"pdf_names": ["a.pdf", "b.pdf"], "page_from": 10}
                  {"page_from": 10, "page_to": 20}

        Returns chunks with the existing response fields and retrieval diagnostics.
        """
        reranked: list[RetrievedChunk] = await ctx.deps.hybrid_retriever.retrieve(
            query=search_query,
            filters=filters,
        )

        return [
            {
                "document": chunk.document,
                "metadata": chunk.metadata,
                "score": chunk.score,
                "chunk_id": chunk.chunk_id,
                "dense_score": chunk.dense_score,
                "dense_rank": chunk.dense_rank,
                "sparse_score": chunk.sparse_score,
                "sparse_rank": chunk.sparse_rank,
                "fused_score": chunk.fused_score,
                "fused_rank": chunk.fused_rank,
                "reranker_score": chunk.reranker_score,
                "reranker_rank": chunk.reranker_rank,
            }
            for chunk in reranked
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
        self._hybrid_retriever = HybridRetriever(
            vector_store=self._vector_store,
            embedder=self._embedder,
            reranker=self._reranker,
            config=settings.retrieval,
        )
        self._agent = build_agent(settings.agent)
        self._deps = AgentDeps(
            vector_store=self._vector_store,
            hybrid_retriever=self._hybrid_retriever,
        )

    def to_web(self) -> object:
        """Expose the agent as a web app via pydantic-ai's built-in server."""
        return self._agent.to_web(deps=self._deps)

    @property
    def vector_store(self) -> VectorStore:
        """Expose VectorStore so the ingestion pipeline can share it."""
        return self._vector_store
