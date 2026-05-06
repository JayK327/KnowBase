# src/api/routes/ingest.py
"""
Ingestion Routes
================
POST /ingest/upload   — ingest a single uploaded file (multipart)
POST /ingest/dir      — ingest all supported files in a local directory
GET  /ingest/status   — index stats (vector count, BM25 doc count)
"""

from __future__ import annotations
import logging
import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from src.api.schemas import IngestDirRequest, IngestResponse
from src.ingestion.models import RawDocument, DocumentFormat

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])

_EXT_MAP = {
    "pdf": DocumentFormat.PDF, "docx": DocumentFormat.DOCX,
    "doc": DocumentFormat.DOCX, "html": DocumentFormat.HTML,
    "htm": DocumentFormat.HTML, "txt": DocumentFormat.TXT, "md": DocumentFormat.MD,
}


@router.post("/upload", response_model=IngestResponse, summary="Ingest uploaded file")
async def ingest_upload(request: Request, file: UploadFile = File(...)):
    pipeline = request.app.state.ingestion_pipeline
    if not pipeline:
        raise HTTPException(503, "Ingestion pipeline not initialised")

    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else ""
    fmt = _EXT_MAP.get(ext)
    if not fmt:
        raise HTTPException(400, f"Unsupported format: .{ext}  Supported: {list(_EXT_MAP)}")

    raw = RawDocument(
        source_path=f"upload://{file.filename}",
        format=fmt,
        raw_bytes=await file.read(),
        metadata={"filename": file.filename},
    )
    t0 = time.perf_counter()
    result = pipeline.process_document(raw)
    if result is None:
        raise HTTPException(422, "Failed to extract text from file")

    return IngestResponse(
        status="completed",
        total_documents=1, successful_documents=1, failed_documents=0,
        total_chunks=result["total_chunks"], total_embeddings=result["embeddings"],
        duration_seconds=round(time.perf_counter() - t0, 2), dry_run=False,
    )


@router.post("/dir", response_model=IngestResponse, summary="Ingest local directory")
async def ingest_directory(body: IngestDirRequest, request: Request):
    pipeline = request.app.state.ingestion_pipeline
    if not pipeline:
        raise HTTPException(503, "Ingestion pipeline not initialised")

    pipeline.dry_run = body.dry_run
    try:
        stats = pipeline.run_from_directory(body.directory)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        logger.error(f"Ingestion error: {exc}")
        raise HTTPException(500, str(exc))

    return IngestResponse(
        status="completed" if stats.failed_documents == 0 else "completed_with_errors",
        total_documents=stats.total_documents,
        successful_documents=stats.successful_documents,
        failed_documents=stats.failed_documents,
        total_chunks=stats.total_chunks,
        total_embeddings=stats.total_embeddings,
        duration_seconds=round(stats.duration_seconds, 2),
        dry_run=body.dry_run,
    )


@router.get("/status", summary="Index stats")
async def ingest_status(request: Request):
    vs   = request.app.state.vector_store
    bm25 = request.app.state.bm25
    return {
        "vector_store_vectors": getattr(vs, "total_vectors", "n/a"),
        "bm25_documents":       len(getattr(bm25, "corpus", [])),
    }


# ---------------------------------------------------------------------------


# src/api/routes/health.py
"""Health and readiness probes."""

from __future__ import annotations
import time
from fastapi import APIRouter, Request
from src.config import get_config

router = APIRouter(tags=["ops"])
_START = time.time()


@router.get("/health", summary="Liveness probe")
async def health(request: Request):
    cfg = get_config()
    return {
        "status":                "ok",
        "version":               cfg.app.version,
        "llm_model":             cfg.generation.llm.model,
        "vector_store_provider": cfg.ingestion.vector_store.provider,
        "vector_store_ready":    request.app.state.vector_store is not None,
    }


@router.get("/ready", summary="Readiness probe")
async def ready(request: Request):
    if request.app.state.rag_chain is None:
        return {"status": "not_ready", "reason": "RAG chain not initialised"}
    return {"status": "ready"}


@router.get("/metrics", summary="Basic metrics")
async def metrics():
    return {
        "uptime_seconds": round(time.time() - _START, 1),
    }
