# src/generation/rag_chain.py
"""
RAG Chain — End-to-End Query Pipeline
======================================

Full pipeline per query:

  1. HyDE expansion  (optional)
     Generate a fake ideal answer → embed THAT instead of raw query.
     Closes the distribution gap between short queries and verbose docs.

  2. Hybrid retrieval
     Dense ANN (FAISS) + BM25 sparse → RRF fusion → top-20 candidates.

  3. Cross-encoder reranking  (optional)
     Re-score top-20 with Cohere / BGE → top-5.
     Relevance gate: if all scores < threshold → return "no info" immediately.

  4. Streaming generation
     Inject parent chunks into GPT-4o prompt → stream tokens via SSE.
     Emit __SOURCES__ JSON suffix for the frontend citation renderer.

Usage:
    chain = RAGChain.from_config(retriever, embedder)
    async for token in chain.query("What is the ETL schema?"):
        print(token, end="", flush=True)
"""

from __future__ import annotations
import json
import logging
import time
from typing import AsyncGenerator, List, Optional

from openai import AsyncOpenAI

from src.config import get_config, get_env
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from src.generation.prompt_templates import build_messages, no_results_message

logger = logging.getLogger(__name__)


class RAGChain:
    """
    Orchestrates retrieval → optional reranking → streaming LLM generation.

    Args:
        retriever:     HybridRetriever instance
        embedder:      any embedder with embed_query() method
        reranker:      optional CohereReranker / CrossEncoderReranker
        hyde:          optional HyDEGenerator
        model:         OpenAI model for final generation
        top_k_retrieval: initial retrieval count (pre-reranking)
        top_k_final:   chunks injected into LLM prompt (post-reranking)
    """

    def __init__(
        self,
        retriever:       HybridRetriever,
        embedder,
        reranker=None,
        hyde=None,
        model:           str = "gpt-4o",
        top_k_retrieval: int = 20,
        top_k_final:     int = 5,
    ):
        self.client          = AsyncOpenAI(api_key=get_env().openai_api_key)
        self.retriever       = retriever
        self.embedder        = embedder
        self.reranker        = reranker
        self.hyde            = hyde
        self.model           = model
        self.top_k_retrieval = top_k_retrieval
        self.top_k_final     = top_k_final

    @classmethod
    def from_config(cls, retriever: HybridRetriever, embedder) -> "RAGChain":
        """Build a RAGChain wired from YAML config."""
        cfg = get_config()

        reranker = None
        if cfg.retrieval.reranker.enabled:
            if cfg.retrieval.reranker.provider == "cohere":
                from src.retrieval.hyde import CohereReranker
                reranker = CohereReranker(
                    relevance_threshold=cfg.retrieval.reranker.relevance_threshold
                )
            elif cfg.retrieval.reranker.provider == "cross-encoder":
                from src.retrieval.hyde import CrossEncoderReranker
                reranker = CrossEncoderReranker(threshold=cfg.retrieval.reranker.relevance_threshold)

        hyde = None
        if cfg.retrieval.hyde.enabled:
            from src.retrieval.hyde import HyDEGenerator
            hyde = HyDEGenerator(
                model=cfg.retrieval.hyde.model,
                max_tokens=cfg.retrieval.hyde.max_tokens,
                temperature=cfg.retrieval.hyde.temperature,
            )

        return cls(
            retriever=retriever,
            embedder=embedder,
            reranker=reranker,
            hyde=hyde,
            model=cfg.generation.llm.model,
            top_k_retrieval=cfg.retrieval.hybrid.top_k_dense,
            top_k_final=cfg.retrieval.hybrid.top_k_final,
        )

    # ── Internal helpers ───────

    async def _query_embedding(self, query: str) -> List[float]:
        """Embed the query, with optional HyDE expansion."""
        if self.hyde:
            try:
                hyde_doc = await self.hyde.generate(query)
                return self.embedder.embed_query(hyde_doc)
            except Exception as exc:
                logger.warning(f"HyDE failed, using raw query: {exc}")
        return self.embedder.embed_query(query)

    async def retrieve_and_rerank(
        self,
        query:           str,
        metadata_filter: Optional[dict] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve + optionally rerank.  Returns empty list if nothing passes threshold."""
        q_emb  = await self._query_embedding(query)
        chunks = self.retriever.retrieve(
            query_embedding=q_emb,
            query_text=query,
            top_k_dense=self.top_k_retrieval,
            top_k_sparse=self.top_k_retrieval,
            top_k_final=self.top_k_retrieval,
            metadata_filter=metadata_filter,
        )
        if not chunks:
            return []
        if self.reranker:
            chunks = self.reranker.rerank(query=query, chunks=chunks, top_n=self.top_k_final)
        else:
            chunks = chunks[: self.top_k_final]
        return chunks

    # ── Public API ───────

    async def query(
        self,
        query:           str,
        chat_history:    Optional[List[dict]] = None,
        metadata_filter: Optional[dict]       = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming RAG query.

        Yields:
            Token strings as they arrive from GPT-4o.
            Final token: "__SOURCES__:<json_array>" (for frontend citation rendering).
        """
        t0     = time.perf_counter()
        chunks = await self.retrieve_and_rerank(query, metadata_filter)

        if not chunks:
            yield no_results_message()
            return

        logger.info(
            f"Retrieved {len(chunks)} chunks in {time.perf_counter()-t0:.2f}s  "
            f"query='{query[:60]}'"
        )

        messages = build_messages(query, chunks, chat_history)
        cfg      = get_config()

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=cfg.generation.llm.max_tokens,
            temperature=cfg.generation.llm.temperature,
            stream=True,
        )

        async for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

        # Emit source metadata for frontend
        sources = [
            {
                "index":        i + 1,
                "display_source": c.display_source,
                "source_path":  c.source_path,
                "score":        round(c.score, 3),
            }
            for i, c in enumerate(chunks)
        ]
        yield f"\n__SOURCES__:{json.dumps(sources)}"

        logger.info(f"RAG complete  latency={time.perf_counter()-t0:.2f}s")

    async def query_sync(
        self,
        query:        str,
        chat_history: Optional[List[dict]] = None,
    ) -> dict:
        """
        Non-streaming version — collects the full response.
        Used by the evaluation pipeline (RAGAS needs full strings).

        Returns:
            {"answer": str, "sources": list}
        """
        parts, sources_json = [], None
        async for token in self.query(query, chat_history):
            if token.startswith("\n__SOURCES__:"):
                sources_json = token.removeprefix("\n__SOURCES__:")
            else:
                parts.append(token)
        return {
            "answer":  "".join(parts),
            "sources": json.loads(sources_json) if sources_json else [],
        }
