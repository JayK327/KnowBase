# tests/integration/test_pipeline.py
"""
Integration tests — full pipeline with real chunker + FAISS, mock embedder + LLM.
No API calls. Covers: ingest → chunk → embed → index → retrieve → generate.
"""

from __future__ import annotations
import asyncio
import hashlib
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from src.ingestion.chunkers.hierarchical_chunker import HierarchicalChunker
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.models import ExtractedDocument, DocumentFormat, RawDocument
from src.retrieval.vector_store.faiss_vs import FAISSVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from src.generation.rag_chain import RAGChain

DIM = 64   # We keep dim small for fast tests


def det_emb(text: str, dim: int = DIM) -> list:
    """Deterministic pseudo-embedding based on MD5 hash of text."""
    h = int(hashlib.md5(text.encode()).hexdigest(), 16)
    np.random.seed(h % (2**32))
    v = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    return (v / norm if norm > 1e-8 else v).tolist()


DOCS = [
    ExtractedDocument(
        doc_id="d1", source_path="s3://t/users.pdf",
        text=(
            "The users table stores user_id (UUID primary key), email (VARCHAR 255 UNIQUE), "
            "created_at (TIMESTAMP), updated_at (TIMESTAMP), is_active (BOOLEAN). "
            "Partitioned by created_at. Updated by user_registration Kafka topic."
        ),
        format=DocumentFormat.PDF,
    ),
    ExtractedDocument(
        doc_id="d2", source_path="s3://t/etl.md",
        text=(
            "The ETL pipeline runs every 15 minutes via scheduled jobs. "
            "It reads from S3 raw zone, transforms with PySpark, writes to Delta Lake silver. "
            "Checkpointing at s3://checkpoint/etl. Retries 3 times on failure."
        ),
        format=DocumentFormat.MD,
    ),
    ExtractedDocument(
        doc_id="d3", source_path="s3://t/auth.html",
        text=(
            "JWT tokens use RS256 signing algorithm. "
            "Access token TTL: 24 hours. Refresh token TTL: 30 days. "
            "Tokens invalidated on password change or logout. "
            "Validation endpoint: /auth/verify."
        ),
        format=DocumentFormat.HTML,
    ),
]


# ── FAISS fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def populated_store():
    """Build real FAISS index from DOCS using deterministic embeddings."""
    chunker = HierarchicalChunker(parent_chunk_size=80, child_chunk_size=25)
    store   = FAISSVectorStore(embedding_dim=DIM)
    all_chunks = []
    for doc in DOCS:
        all_chunks.extend(chunker.chunk_document(doc))

    records = []
    for c in all_chunks:
        rec = {
            "chunk_id": c.chunk_id, "doc_id": c.doc_id,
            "parent_chunk_id": c.parent_chunk_id or "",
            "text": c.text, "chunk_type": c.chunk_type.value,
            "source_path": c.source_path, "metadata": c.metadata,
        }
        if c.is_child:
            rec["embedding"] = det_emb(c.text)
        records.append(rec)

    store.upsert(records)
    return store, all_chunks


@pytest.fixture(scope="module")
def bm25(populated_store):
    _, chunks = populated_store
    corpus = [{"chunk_id": c.chunk_id, "text": c.text} for c in chunks if c.is_child]
    return BM25Retriever(corpus=corpus)


@pytest.fixture(scope="module")
def retriever(populated_store, bm25):
    store, _ = populated_store
    return HybridRetriever(vector_store=store, bm25_retriever=bm25)


# ── FAISS tests ───────────────────────────────────────────────────────────────

class TestFAISSStore:
    def test_has_vectors(self, populated_store):
        store, _ = populated_store
        assert store.total_vectors > 0

    def test_similarity_search_returns_results(self, populated_store):
        store, _ = populated_store
        results = store.similarity_search(det_emb("users table UUID"), k=5)
        assert len(results) > 0

    def test_get_by_id(self, populated_store):
        store, chunks = populated_store
        cid = next(c.chunk_id for c in chunks if c.is_child)
        rec = store.get_by_id(cid)
        assert rec is not None
        assert rec["chunk_id"] == cid

    def test_get_unknown_id_returns_none(self, populated_store):
        store, _ = populated_store
        assert store.get_by_id("does-not-exist") is None

    def test_save_and_load(self, populated_store, tmp_path):
        store, _ = populated_store
        path = str(tmp_path / "test_idx")
        store.save(path)
        loaded = FAISSVectorStore(embedding_dim=DIM)
        loaded.load(path)
        assert loaded.total_vectors == store.total_vectors


# ── Retrieval tests ───────────────────────────────────────────────────────────

class TestHybridRetrieverIntegration:
    def test_returns_chunks(self, retriever):
        results = retriever.retrieve(det_emb("users table email"), "users table email", top_k_final=3)
        assert len(results) > 0

    def test_bm25_finds_exact_keyword(self, retriever):
        """BM25 should surface ETL doc even with a zero dense embedding."""
        results = retriever.retrieve([0.0]*DIM, "ETL checkpointing S3", top_k_final=3)
        sources = [r.source_path for r in results]
        assert any("etl" in s.lower() for s in sources)

    def test_no_duplicate_parent_ids(self, retriever):
        results = retriever.retrieve(det_emb("auth JWT"), "JWT authentication", top_k_final=5)
        pids = [r.parent_chunk_id for r in results]
        assert len(pids) == len(set(pids))

    def test_text_field_is_parent_text(self, retriever):
        """Retrieved chunk text should come from the parent (longer context)."""
        results = retriever.retrieve(det_emb("token TTL"), "token TTL hours", top_k_final=2)
        for r in results:
            assert len(r.text) >= len(r.child_text)


# ── Ingestion pipeline tests ──────────────────────────────────────────────────

class TestIngestionPipeline:
    @pytest.fixture
    def pipeline(self):
        store   = FAISSVectorStore(embedding_dim=DIM)
        embedder= MagicMock()
        embedder.embed_documents.side_effect = lambda texts: [det_emb(t) for t in texts]
        return IngestionPipeline(vector_store=store, embedder=embedder)

    def test_chunk_produces_parent_and_child(self, pipeline):
        from src.ingestion.models import ChunkType
        chunks = pipeline.chunk(DOCS[0])
        types  = {c.chunk_type for c in chunks}
        assert ChunkType.PARENT in types and ChunkType.CHILD in types

    def test_embed_returns_only_children(self, pipeline):
        from src.ingestion.models import ChunkType
        chunks   = pipeline.chunk(DOCS[0])
        embedded = pipeline.embed(chunks)
        assert all(ec.chunk.chunk_type == ChunkType.CHILD for ec in embedded)

    def test_process_doc_populates_store(self, pipeline):
        result = pipeline.process_document(
            RawDocument(
                doc_id="test-ingest", source_path="s3://t/x.txt",
                format=DocumentFormat.TXT,
                raw_bytes=DOCS[0].text.encode(),
            )
        )
        assert result is not None
        assert result["total_chunks"] > 0
        assert result["embeddings"] > 0
        assert pipeline.vector_store.total_vectors > 0

    def test_empty_text_produces_none(self, pipeline):
        result = pipeline.process_document(
            RawDocument(doc_id="empty", source_path="s3://t/e.txt",
                        format=DocumentFormat.TXT, raw_bytes=b"   ")
        )
        assert result is None

    def test_dry_run_does_not_write(self):
        store   = MagicMock()
        emb     = MagicMock()
        emb.embed_documents.return_value = [[0.1]*DIM] * 10
        dry     = IngestionPipeline(vector_store=store, embedder=emb, dry_run=True)
        dry.process_document(
            RawDocument(doc_id="d", source_path="s3://t/d.txt",
                        format=DocumentFormat.TXT, raw_bytes=DOCS[0].text.encode())
        )
        store.upsert.assert_not_called()


# ── RAGChain tests (mocked LLM) ───────────────────────────────────────────────

class TestRAGChain:
    @pytest.fixture
    def chain(self, retriever):
        embedder = MagicMock()
        embedder.embed_query.side_effect = lambda t: det_emb(t)
        return RAGChain(
            retriever=retriever,
            embedder=embedder,
            reranker=None,
            hyde=None,
            model="gpt-4o",
            top_k_retrieval=5,
            top_k_final=3,
        )

    @pytest.mark.asyncio
    async def test_retrieve_and_rerank_returns_chunks(self, chain):
        chunks = await chain.retrieve_and_rerank("JWT authentication TTL")
        assert isinstance(chunks, list)
        assert all(isinstance(c, RetrievedChunk) for c in chunks)

    @pytest.mark.asyncio
    async def test_empty_store_returns_no_info(self):
        store   = FAISSVectorStore(embedding_dim=DIM)
        bm25_e  = BM25Retriever(corpus=[])
        ret     = HybridRetriever(vector_store=store, bm25_retriever=bm25_e)
        emb     = MagicMock()
        emb.embed_query.return_value = [0.0]*DIM
        c = RAGChain(retriever=ret, embedder=emb, top_k_retrieval=5, top_k_final=3)

        tokens = []
        async for t in c.query("anything?"):
            tokens.append(t)
        assert "don't have enough information" in "".join(tokens).lower()
