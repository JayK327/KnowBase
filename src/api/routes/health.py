# src/api/routes/health.py
"""Liveness / readiness / metrics probes."""

from __future__ import annotations
import time
from fastapi import APIRouter, Request
from src.config import get_config

router  = APIRouter(tags=["ops"])
_START  = time.time()


@router.get("/health", summary="Liveness probe")
async def health(request: Request):
    cfg = get_config()
    return {
        "status":                "ok",
        "version":               cfg.app.version,
        "llm_model":             cfg.generation.llm.model,
        "embedding_model":       cfg.ingestion.embedding.model,
        "vector_store_provider": cfg.ingestion.vector_store.provider,
        "vector_store_ready":    request.app.state.vector_store is not None,
    }


@router.get("/ready", summary="Readiness probe")
async def ready(request: Request):
    if getattr(request.app.state, "rag_chain", None) is None:
        return {"status": "not_ready", "reason": "RAG chain not initialised"}
    return {"status": "ready"}


@router.get("/metrics", summary="Basic runtime metrics")
async def metrics():
    return {"uptime_seconds": round(time.time() - _START, 1)}
