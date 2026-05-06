# src/ingestion/models.py
"""
Core data models shared across ingestion and retrieval.

Data flow:
  RawDocument → ExtractedDocument → List[Chunk] → List[EmbeddedChunk]

Chunk design:
  Parent chunk (512 tokens): stored for LLM context injection.
  Child chunk  (128 tokens): embedded for vector similarity search.
  Each child carries parent_chunk_id to enable parent promotion at query time.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import uuid


class DocumentFormat(str, Enum):
    PDF  = "pdf"
    DOCX = "docx"
    HTML = "html"
    TXT  = "txt"
    MD   = "md"


class ChunkType(str, Enum):
    PARENT = "parent"
    CHILD  = "child"


@dataclass
class RawDocument:
    """Binary document as read from disk / upload."""
    doc_id:      str            = field(default_factory=lambda: str(uuid.uuid4()))
    source_path: str            = ""
    format:      DocumentFormat = DocumentFormat.TXT
    raw_bytes:   Optional[bytes]= None
    metadata:    dict           = field(default_factory=dict)


@dataclass
class ExtractedDocument:
    """Plain text extracted from a RawDocument."""
    doc_id:      str
    source_path: str
    text:        str
    format:      DocumentFormat
    num_pages:   Optional[int]  = None
    title:       Optional[str]  = None
    author:      Optional[str]  = None
    language:    str            = "en"
    metadata:    dict           = field(default_factory=dict)

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = str(uuid.uuid4())


@dataclass
class Chunk:
    """Single text chunk — parent (LLM context) or child (embedding)."""
    chunk_id:        str             = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id:          str             = ""
    parent_chunk_id: Optional[str]   = None   # None for parent chunks
    text:            str             = ""
    token_count:     int             = 0
    chunk_type:      ChunkType       = ChunkType.CHILD
    chunk_index:     int             = 0
    child_index:     int             = 0
    source_path:     str             = ""
    metadata:        dict            = field(default_factory=dict)

    @property
    def is_parent(self) -> bool:
        return self.chunk_type == ChunkType.PARENT

    @property
    def is_child(self) -> bool:
        return self.chunk_type == ChunkType.CHILD


@dataclass
class EmbeddedChunk:
    """A Chunk with its dense vector embedding attached."""
    chunk:           Chunk
    embedding:       List[float]
    embedding_model: str  = ""
    embedding_dim:   int  = 0
