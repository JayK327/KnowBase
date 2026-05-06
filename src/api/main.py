# src/api/main.py
"""
FastAPI Application
====================
Startup sequence:
  1. Load config (OmegaConf)
  2. Init FAISS vector store (load from disk if index exists)
  3. Build BM25 index from stored child-chunk metadata
  4. Init RAGChain (embedder + hybrid retriever + reranker + HyDE)
  5. Init IngestionPipeline

All heavy objects live on app.state, created once at startup, reused per request.

Routes:
  POST /chat          — streaming or JSON RAG query
  POST /ingest/upload — ingest a single uploaded file
  POST /ingest/dir    — ingest all files in a local directory
  GET  /health        — liveness probe
  GET  /ready         — readiness probe (checks chain init)
"""

from __future__ import annotations
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_config
from src.api.middleware import RequestLoggingMiddleware, RateLimitMiddleware, APIKeyMiddleware
from src.api.routes import chat, ingest, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    cfg = get_config()
    logger.info(f"Starting {cfg.app.name} v{cfg.app.version}")

    # ── Vector store ──────────────────────────────────────────────────────
    from src.retrieval.vector_store.faiss_vs import FAISSVectorStore
    from src.ingestion.embedders.openai_embedder import OpenAIEmbedder

    vs = FAISSVectorStore(embedding_dim=cfg.ingestion.embedding.embedding_dim)
    index_path = cfg.ingestion.vector_store.index_path
    if Path(index_path).exists():
        vs.load(index_path)
        logger.info(f"FAISS loaded  vectors={vs.total_vectors}")
    else:
        logger.warning("No FAISS index found — vector store empty")

    app.state.vector_store = vs

    # ── Embedder ──────────────────────────────────────────────────────────
    embedder = OpenAIEmbedder(model=cfg.ingestion.embedding.model)
    app.state.embedder = embedder

    # ── BM25 ──────────────────────────────────────────────────────────────
    from src.retrieval.bm25_retriever import BM25Retriever
    bm25_path = os.getenv("BM25_INDEX_PATH", "data/bm25_index")
    if Path(f"{bm25_path}/bm25.pkl").exists():
        bm25 = BM25Retriever.load(bm25_path)
    else:
        logger.warning("BM25 index not found — sparse retrieval disabled")
        bm25 = BM25Retriever(corpus=[])
    app.state.bm25 = bm25

    # ── Retriever ─────────────────────────────────────────────────────────
    from src.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(
        vector_store=vs,
        bm25_retriever=bm25,
        rrf_k=cfg.retrieval.hybrid.rrf_k,
        dense_weight=cfg.retrieval.hybrid.dense_weight,
    )

    # ── RAG chain ─────────────────────────────────────────────────────────
    from src.generation.rag_chain import RAGChain
    rag_chain = RAGChain.from_config(retriever=retriever, embedder=embedder)
    app.state.rag_chain = rag_chain
    logger.info(f"RAG chain ready  model={cfg.generation.llm.model}")

    # ── Ingestion pipeline ────────────────────────────────────────────────
    from src.ingestion.pipeline import IngestionPipeline
    app.state.ingestion_pipeline = IngestionPipeline(
        vector_store=vs, embedder=embedder
    )

    logger.info("Application ready")
    yield   # ← runs here until shutdown

    logger.info("Shutting down")


def create_app() -> FastAPI:
    cfg = get_config()

    app = FastAPI(
        title="RAG Chatbot Platform",
        description="Production RAG pipeline — OpenAI + FAISS + BM25",
        version=cfg.app.version,
        lifespan=lifespan,
        docs_url="/docs",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.api.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Custom middleware (outermost first)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=cfg.api.rate_limit.requests_per_minute)
    app.add_middleware(RequestLoggingMiddleware)
    if cfg.api.auth.enabled:
        app.add_middleware(APIKeyMiddleware, api_key=os.getenv("API_KEY", ""), enabled=True)

    # Routes
    app.include_router(chat.router)
    app.include_router(ingest.router)
    app.include_router(health.router)

    return app


app = create_app()
