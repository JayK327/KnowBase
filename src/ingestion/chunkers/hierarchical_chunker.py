# src/ingestion/chunkers/hierarchical_chunker.py
"""
Hierarchical Chunker
====================

This is the core idea behind the system.

The tradeoff is simple:
- Small chunks → great for retrieval, but too little context for the LLM
- Large chunks → great for the LLM, but embeddings become less precise

So instead of picking one, we do both.

We split documents in two passes:
- Parent chunks (~512 tokens) → used as context for the LLM
- Child chunks (~128 tokens) → used for retrieval

Each parent is broken into multiple children:

    ┌─────────────────────────────────────────────┐
    │ Parent chunk (~512 tokens)                  │  ← sent to LLM
    │  ┌────────────┐ ┌────────────┐ ┌──────────┐ │
    │  │ Child 1    │ │ Child 2    │ │ Child 3  │ │  ← embedded + retrieved
    │  │ (~128 tok) │ │ (~128 tok) │ │(~128 tok)│ │
    │  └────────────┘ └────────────┘ └──────────┘ │
    └─────────────────────────────────────────────┘

At query time:
1. We retrieve relevant *child* chunks (high precision)
2. Then map them back to their *parent* chunks
3. Only the parent chunks are sent to the LLM

This gives you precise retrieval *and* enough context for reasoning.

Each child stores a `parent_chunk_id`, so the lookup is just a quick O(1) step.

In practice, this significantly improves answer quality compared to
fixed-size chunking, especially on technical documents.

Output format:
    Flat list like:
    [parent_0, child_0a, child_0b, parent_1, child_1a, ...]

Parents always come before their children.
"""

from __future__ import annotations
import uuid
import logging
from typing import List

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.ingestion.models import ExtractedDocument, Chunk, ChunkType

logger = logging.getLogger(__name__)


class HierarchicalChunker:
    """
    Two-pass splitter:
      Pass 1 → parent chunks  (default 512 tokens, overlap 50)
      Pass 2 → child chunks   (default 128 tokens, overlap 20)

    Args:
        parent_chunk_size: token budget per parent chunk
        child_chunk_size:  token budget per child chunk
        parent_overlap:    token overlap between adjacent parents
        child_overlap:     token overlap between adjacent children
        encoding_model:    tiktoken model (cl100k_base for GPT-4 family)
    """

    def __init__(
        self,
        parent_chunk_size: int  = 512,
        child_chunk_size:  int  = 128,
        parent_overlap:    int  = 50,
        child_overlap:     int  = 20,
        encoding_model:    str  = "cl100k_base",
    ):
        self.enc = tiktoken.get_encoding(encoding_model)

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
            length_function=self._tok,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            length_function=self._tok,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _tok(self, text: str) -> int:
        """Count tokens without raising on special tokens."""
        return len(self.enc.encode(text, disallowed_special=()))

    def chunk_document(self, doc: ExtractedDocument) -> List[Chunk]:
        """
        Produce parent + child chunks from an ExtractedDocument.

        Returns:
            Flat list of Chunk objects.  Parents precede their children.
            Empty list if doc.text is blank.
        """
        if not doc.text.strip():
            return []

        base_meta = {
            "doc_id":      doc.doc_id,
            "source_path": doc.source_path,
            "title":       doc.title or "",
            "format":      doc.format.value if doc.format else "",
        }

        chunks: List[Chunk] = []
        parent_texts = self.parent_splitter.split_text(doc.text)

        for chunk_idx, parent_text in enumerate(parent_texts):
            parent_id = str(uuid.uuid4())

            # ── Parent chunk ─────
            chunks.append(Chunk(
                chunk_id=parent_id,
                doc_id=doc.doc_id,
                parent_chunk_id=None,
                text=parent_text,
                token_count=self._tok(parent_text),
                chunk_type=ChunkType.PARENT,
                chunk_index=chunk_idx,
                source_path=doc.source_path,
                metadata={**base_meta, "chunk_type": "parent", "chunk_index": chunk_idx},
            ))

            # ── Child chunks (what gets embedded) ─────
            for child_idx, child_text in enumerate(
                self.child_splitter.split_text(parent_text)
            ):
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc.doc_id,
                    parent_chunk_id=parent_id,   # ← key linkage
                    text=child_text,
                    token_count=self._tok(child_text),
                    chunk_type=ChunkType.CHILD,
                    chunk_index=chunk_idx,
                    child_index=child_idx,
                    source_path=doc.source_path,
                    metadata={
                        **base_meta,
                        "chunk_type":     "child",
                        "chunk_index":    chunk_idx,
                        "child_index":    child_idx,
                        "parent_chunk_id": parent_id,
                    },
                ))

        n_parents  = sum(1 for c in chunks if c.is_parent)
        n_children = sum(1 for c in chunks if c.is_child)
        logger.debug(
            f"[{doc.doc_id}] chunked → {n_parents} parents, {n_children} children"
        )
        return chunks
