# src/api/middleware.py
"""
FastAPI Middleware Stack
========================
Three middleware layers (outermost → innermost):
  1. RateLimitMiddleware  — sliding-window IP-based rate limiting
  2. RequestLoggingMiddleware — request ID, latency logging
  3. APIKeyMiddleware      — Bearer token auth (disabled in dev)
"""

from __future__ import annotations
import logging
import time
import uuid
from collections import defaultdict, deque
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ── Request logging ──────────────────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Adds X-Request-ID and X-Latency-Ms to every response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        rid = str(uuid.uuid4())[:8]
        request.state.request_id = rid
        t0   = time.perf_counter()
        resp = await call_next(request)
        ms   = (time.perf_counter() - t0) * 1000
        resp.headers["X-Request-ID"] = rid
        resp.headers["X-Latency-Ms"] = str(round(ms, 1))
        logger.info(
            f"[{rid}] {request.method} {request.url.path} "
            f"→ {resp.status_code} ({ms:.1f}ms)"
        )
        return resp


# ── API key auth ─────────────────────────────────────────────────────────────

class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validates Bearer token in Authorization header.
    Skips /health, /ready, /docs, /openapi.json.
    In production replace with JWT / OAuth2.
    """

    SKIP = {"/health", "/ready", "/metrics", "/docs", "/openapi.json"}

    def __init__(self, app, api_key: str, enabled: bool = True):
        super().__init__(app)
        self.api_key = api_key
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled or request.url.path in self.SKIP:
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing Authorization header"})
        if auth.removeprefix("Bearer ").strip() != self.api_key:
            return JSONResponse(status_code=403, content={"detail": "Invalid API key"})
        return await call_next(request)


# ── Rate limiting ─────────────────────────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter keyed by client IP.
    In-process only — resets on restart.
    For multi-replica deployments use Redis-backed slowapi instead.
    """

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.rpm     = requests_per_minute
        self._windows: dict = defaultdict(lambda: deque())

    def _ip(self, req: Request) -> str:
        fwd = req.headers.get("X-Forwarded-For")
        return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "unknown")

    def _limited(self, ip: str) -> bool:
        now    = time.time()
        window = self._windows[ip]
        while window and window[0] < now - 60:
            window.popleft()
        if len(window) >= self.rpm:
            return True
        window.append(now)
        return False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path.startswith("/chat"):
            if self._limited(self._ip(request)):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Retry after 60s."},
                    headers={"Retry-After": "60"},
                )
        return await call_next(request)
