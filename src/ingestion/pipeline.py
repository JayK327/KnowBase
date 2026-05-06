# src/ingestion/pipeline.py
"""
Ingestion Pipeline Orchestrator
================================
Wires: file → extract → chunk → embed → upsert to vector store.

Supports:
  - Local file directory ingestion
  - Single-file ingestion (used by /ingest/upload API endpoint)
  - Dry-run mode (validate without writing)

Stats tracked per run: doc count, chunk count, embedding count, duration, throughput.

Usage:
    pipeline = IngestionPipeline.from_config()
    stats = pipeline.run_from_directory("data/raw/")
    print(stats.summary())
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from src.config import get_config
from src.ingestion.models import (
    RawDocument, ExtractedDocument, Chunk, EmbeddedChunk, DocumentFormat,
)
from src.ingestion.extractors.pdf_extractor import PDFExtractor
from src.ingestion.extractors.docx_extractor import DocxExtractor, HtmlExtractor
from src.ingestion.chunkers.hierarchical_chunker import HierarchicalChunker
from src.ingestion.embedders.openai_embedder import OpenAIEmbedder

logger = logging.getLogger(__name__)

EXT_TO_FORMAT = {
    ".pdf":  DocumentFormat.PDF,
    ".docx": DocumentFormat.DOCX,
    ".doc":  DocumentFormat.DOCX,
    ".html": DocumentFormat.HTML,
    ".htm":  DocumentFormat.HTML,
    ".txt":  DocumentFormat.TXT,
    ".md":   DocumentFormat.MD,
}


@dataclass
class IngestionStats:
    total_documents:    int   = 0
    successful_documents: int = 0
    failed_documents:   int   = 0
    total_chunks:       int   = 0
    total_parent_chunks:int   = 0
    total_child_chunks: int   = 0
    total_embeddings:   int   = 0
    _start: float             = field(default_factory=time.time)
    _end:   Optional[float]   = None

    @property
    def duration_seconds(self) -> float:
        return (self._end or time.time()) - self._start

    def summary(self) -> str:
        return (
            f"Ingestion complete | "
            f"docs={self.successful_documents}/{self.total_documents} | "
            f"chunks={self.total_chunks} (p={self.total_parent_chunks}, "
            f"c={self.total_child_chunks}) | "
            f"embeddings={self.total_embeddings} | "
            f"duration={self.duration_seconds:.1f}s"
        )


class IngestionPipeline:
    """
    Full ingestion pipeline: file → text → chunks → embeddings → vector store.
    """

    def __init__(
        self,
        vector_store,
        embedder=None,
        chunker: Optional[HierarchicalChunker] = None,
        dry_run: bool = False,
    ):
        cfg = get_config()
        self.vector_store = vector_store
        self.dry_run      = dry_run

        self.embedder = embedder or OpenAIEmbedder(
            model=cfg.ingestion.embedding.model,
            batch_size=cfg.ingestion.embedding.batch_size,
        )
        self.chunker = chunker or HierarchicalChunker(
            parent_chunk_size=cfg.ingestion.chunking.parent_chunk_size,
            child_chunk_size=cfg.ingestion.chunking.child_chunk_size,
            parent_overlap=cfg.ingestion.chunking.parent_overlap,
            child_overlap=cfg.ingestion.chunking.child_overlap,
        )
        self._extractors = {
            DocumentFormat.PDF:  PDFExtractor(),
            DocumentFormat.DOCX: DocxExtractor(),
            DocumentFormat.HTML: HtmlExtractor(),
        }

    @classmethod
    def from_config(cls, dry_run: bool = False) -> "IngestionPipeline":
        """Build pipeline from YAML config (uses FAISS vector store by default)."""
        from src.retrieval.vector_store.faiss_vs import FAISSVectorStore
        cfg = get_config()
        vs  = FAISSVectorStore()
        index_path = cfg.ingestion.vector_store.index_path
        if Path(index_path).exists():
            vs.load(index_path)
        return cls(vector_store=vs, dry_run=dry_run)

    # ── Stage: Extract ────────────────────────────────────────────────────

    def extract(self, doc: RawDocument) -> Optional[ExtractedDocument]:
        extractor = self._extractors.get(doc.format)
        if not extractor:
            # Treat TXT / MD as plain text
            if doc.format in (DocumentFormat.TXT, DocumentFormat.MD):
                text = doc.raw_bytes.decode("utf-8", errors="replace") if doc.raw_bytes else ""
                return ExtractedDocument(
                    doc_id=doc.doc_id, source_path=doc.source_path,
                    text=text, format=doc.format, metadata=doc.metadata,
                )
            logger.warning(f"No extractor for format={doc.format}")
            return None
        try:
            return extractor.extract(doc)
        except Exception as exc:
            logger.error(f"Extraction failed [{doc.source_path}]: {exc}")
            return None

    # ── Stage: Chunk ──────────────────────────────────────────────────────

    def chunk(self, doc: ExtractedDocument) -> List[Chunk]:
        if not doc.text.strip():
            logger.warning(f"Empty text, skipping [{doc.source_path}]")
            return []
        return self.chunker.chunk_document(doc)

    # ── Stage: Embed ──────────────────────────────────────────────────────

    def embed(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        """Only CHILD chunks are embedded; parents are stored for context only."""
        children = [c for c in chunks if c.is_child]
        if not children:
            return []
        embeddings = self.embedder.embed_documents([c.text for c in children])
        return [
            EmbeddedChunk(
                chunk=c, embedding=e,
                embedding_model=getattr(self.embedder, "model", self.embedder.MODEL),
                embedding_dim=len(e),
            )
            for c, e in zip(children, embeddings)
        ]

    # ── Stage: Upsert ─────────────────────────────────────────────────────

    def upsert(self, chunks: List[Chunk], embedded: List[EmbeddedChunk]) -> None:
        if self.dry_run:
            logger.info(f"[DRY RUN] Would upsert {len(embedded)} embeddings")
            return

        emb_map = {ec.chunk.chunk_id: ec.embedding for ec in embedded}
        records = []
        for c in chunks:
            rec = {
                "chunk_id":        c.chunk_id,
                "doc_id":          c.doc_id,
                "parent_chunk_id": c.parent_chunk_id or "",
                "text":            c.text,
                "token_count":     c.token_count,
                "chunk_type":      c.chunk_type.value,
                "source_path":     c.source_path,
                "metadata":        c.metadata,
            }
            if c.chunk_id in emb_map:
                rec["embedding"] = emb_map[c.chunk_id]
            records.append(rec)

        self.vector_store.upsert(records)

    # ── Orchestration ─────────────────────────────────────────────────────

    def process_document(self, doc: RawDocument) -> Optional[dict]:
        """Process one document end-to-end. Returns stats dict or None on failure."""
        extracted = self.extract(doc)
        if not extracted:
            return None
        chunks = self.chunk(extracted)
        if not chunks:
            return None
        embedded = self.embed(chunks)
        self.upsert(chunks, embedded)
        return {
            "doc_id":         extracted.doc_id,
            "total_chunks":   len(chunks),
            "parent_chunks":  sum(1 for c in chunks if c.is_parent),
            "child_chunks":   sum(1 for c in chunks if c.is_child),
            "embeddings":     len(embedded),
        }

    def run_from_directory(self, directory: str) -> IngestionStats:
        """Ingest all supported documents in a local directory (recursive)."""
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        def _iter():
            for file_path in sorted(path.rglob("*")):
                if not file_path.is_file():
                    continue
                fmt = EXT_TO_FORMAT.get(file_path.suffix.lower())
                if not fmt:
                    continue
                yield RawDocument(
                    source_path=str(file_path),
                    format=fmt,
                    raw_bytes=file_path.read_bytes(),
                    metadata={"filename": file_path.name},
                )

        return self._run(_iter())

    def _run(self, docs: Iterator[RawDocument]) -> IngestionStats:
        stats = IngestionStats()
        for doc in docs:
            stats.total_documents += 1
            try:
                result = self.process_document(doc)
                if result:
                    stats.successful_documents  += 1
                    stats.total_chunks          += result["total_chunks"]
                    stats.total_parent_chunks   += result["parent_chunks"]
                    stats.total_child_chunks    += result["child_chunks"]
                    stats.total_embeddings      += result["embeddings"]
                else:
                    stats.failed_documents += 1
            except Exception as exc:
                logger.error(f"Pipeline error [{doc.source_path}]: {exc}")
                stats.failed_documents += 1

        stats._end = time.time()
        logger.info(stats.summary())
        return stats
