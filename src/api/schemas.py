# src/api/schemas.py
"""Pydantic v2 request/response models."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import uuid


class ChatMessage(BaseModel):
    role:    str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=32_000)


class ChatRequest(BaseModel):
    query:           str              = Field(..., min_length=1, max_length=4_000)
    session_id:      str              = Field(default_factory=lambda: str(uuid.uuid4()))
    top_k:           int              = Field(default=5, ge=1, le=20)
    metadata_filter: Optional[Dict]   = None
    chat_history:    Optional[List[ChatMessage]] = None
    stream:          bool             = True

    @field_validator("query")
    @classmethod
    def strip(cls, v: str) -> str:
        return v.strip()


class SourceRef(BaseModel):
    index:           int
    display_source:  str
    source_path:     str
    relevance_score: float


class QualitySignals(BaseModel):
    citation_count:          int
    hallucinated_citations:  List[int]
    faithfulness_warning:    bool
    uncited_source_count:    int


class ChatResponse(BaseModel):
    session_id:      str
    answer:          str
    sources:         List[SourceRef]
    quality_signals: QualitySignals
    latency_ms:      float


class IngestDirRequest(BaseModel):
    directory: str = Field(..., description="Local directory path to ingest")
    dry_run:   bool = False


class IngestResponse(BaseModel):
    status:              str
    total_documents:     int
    successful_documents:int
    failed_documents:    int
    total_chunks:        int
    total_embeddings:    int
    duration_seconds:    float
    dry_run:             bool
