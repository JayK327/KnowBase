# src/api/routes/chat.py
"""
POST /chat
===========
Supports both streaming (SSE) and non-streaming JSON responses.

Streaming (default, stream=true):
  Each token is wrapped as a JSON SSE event:
    data: {"token": "The"}\n\n
    data: {"token": " users"}\n\n
    ...
    data: {"sources": [{"index":1, ...}]}\n\n
    data: [DONE]\n\n

Non-streaming (stream=false):
  Returns a complete ChatResponse JSON object including
  citations and quality signals.
"""

from __future__ import annotations
import json
import logging
import time

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatRequest, ChatResponse, SourceRef, QualitySignals

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", summary="RAG query (streaming or JSON)")
async def chat(body: ChatRequest, request: Request):
    chain = request.app.state.rag_chain
    if not chain:
        raise HTTPException(503, "RAG chain not initialised")

    history = (
        [{"role": m.role, "content": m.content} for m in body.chat_history]
        if body.chat_history else None
    )

    # ── Streaming ────────────────────────────────────────────────────────
    if body.stream:
        async def event_stream():
            try:
                async for token in chain.query(
                    query=body.query,
                    chat_history=history,
                    metadata_filter=body.metadata_filter,
                ):
                    if token.startswith("\n__SOURCES__:"):
                        sources = json.loads(token.removeprefix("\n__SOURCES__:"))
                        yield f"data: {json.dumps({'sources': sources})}\n\n"
                    else:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error(f"Stream error: {exc}")
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Non-streaming ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        result = await chain.query_sync(query=body.query, chat_history=history)
    except Exception as exc:
        logger.error(f"RAG error: {exc}")
        raise HTTPException(500, str(exc))

    latency_ms = (time.perf_counter() - t0) * 1000
    sources = [
        SourceRef(
            index=s["index"],
            display_source=s.get("display_source", ""),
            source_path=s.get("source_path", ""),
            relevance_score=s.get("score", 0.0),
        )
        for s in result.get("sources", [])
    ]
    return ChatResponse(
        session_id=body.session_id,
        answer=result["answer"],
        sources=sources,
        quality_signals=QualitySignals(
            citation_count=len(sources),
            hallucinated_citations=[],
            faithfulness_warning=False,
            uncited_source_count=0,
        ),
        latency_ms=round(latency_ms, 2),
    )
