# src/retrieval/vector_store/faiss_vs.py
"""
FAISS Vector Store
==================

Index type: IndexFlatIP (exact inner-product / cosine on normalised vectors).
For >1M vectors, swap to IndexIVFFlat or HNSW:
    index = faiss.IndexHNSWFlat(dim, 32)   # M=32 neighbours

Design:
  - Embeddings are L2-normalised before insertion → inner product = cosine similarity
  - All chunk metadata (text, source, type) stored in a dict keyed by chunk_id
  - chunk_id ↔ faiss_row_id mapping for O(1) lookups
  - save() / load() for persistence between restarts

Interface is identical to the (optional) Pinecone/Databricks wrappers
so the retriever never knows which backend is active.
"""

from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Local FAISS vector store with cosine similarity.

    Args:
        embedding_dim: dimension of embedding vectors (default 3072 for text-embedding-3-large)
    """

    def __init__(self, embedding_dim: int = 3072):
        self.embedding_dim = embedding_dim
        self._meta: Dict[str, Dict]   = {}   # chunk_id → record (no embedding)
        self._id_to_row: Dict[str, int] = {} # chunk_id → faiss row index
        self._row_to_id: Dict[int, str] = {} # faiss row index → chunk_id
        self._next_row = 0
        self._index = None
        self._init_index()

    def _init_index(self):
        try:
            import faiss
            self._faiss = faiss
            self._index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info(f"FAISS IndexFlatIP initialised  dim={self.embedding_dim}")
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """L2-normalise a 1-D vector in place; safe for zero vectors."""
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-10 else v

    # ── Write ─────────────────────────────────────────────────────────────

    def upsert(self, records: List[Dict[str, Any]]) -> None:
        """
        Add records to the index.  Only records with an 'embedding' key are
        indexed for similarity search; all records are stored in metadata.

        Args:
            records: list of dicts.  Required keys: 'chunk_id'.
                     Optional key: 'embedding' (list of floats).
        """
        vectors, ids_to_add = [], []

        for rec in records:
            cid = rec.get("chunk_id", "")
            if not cid:
                continue
            # Store metadata without the embedding vector
            self._meta[cid] = {k: v for k, v in rec.items() if k != "embedding"}

            emb = rec.get("embedding")
            if emb and cid not in self._id_to_row:
                v = self._normalize(np.array(emb, dtype=np.float32))
                vectors.append(v)
                ids_to_add.append(cid)

        if vectors:
            matrix = np.stack(vectors, axis=0)
            self._index.add(matrix)
            for cid in ids_to_add:
                row = self._next_row
                self._id_to_row[cid] = row
                self._row_to_id[row]  = cid
                self._next_row += 1
            logger.debug(f"Upserted {len(vectors)} vectors  total={self._index.ntotal}")

    # ── Read ──────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        embedding: List[float],
        k: int = 20,
        filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Approximate nearest-neighbour search.

        Args:
            embedding: query vector
            k:         number of results
            filter:    optional metadata filter (applied post-search)

        Returns:
            List of metadata dicts sorted by cosine similarity descending,
            each augmented with a 'score' field.
        """
        if self._index.ntotal == 0:
            return []

        q = self._normalize(np.array(embedding, dtype=np.float32)).reshape(1, -1)
        k_actual = min(k * 3 if filter else k, self._index.ntotal)
        scores, rows = self._index.search(q, k_actual)

        results = []
        for score, row in zip(scores[0], rows[0]):
            if row < 0:
                continue
            cid = self._row_to_id.get(int(row))
            if not cid:
                continue
            rec = dict(self._meta.get(cid, {}))
            rec["score"] = float(score)

            # Apply optional metadata filter
            if filter:
                if not all(rec.get(fk) == fv for fk, fv in filter.items()):
                    continue

            results.append(rec)
            if len(results) >= k:
                break

        return results

    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Point lookup by chunk_id.  Used for parent promotion."""
        return self._meta.get(chunk_id)

    def delete(self, chunk_ids: List[str]) -> None:
        """Remove from metadata (FAISS flat index doesn't support deletion)."""
        for cid in chunk_ids:
            self._meta.pop(cid, None)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save index + metadata to disk."""
        import faiss
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p / "index.faiss"))
        with open(p / "state.pkl", "wb") as f:
            pickle.dump({
                "meta":        self._meta,
                "id_to_row":   self._id_to_row,
                "row_to_id":   self._row_to_id,
                "next_row":    self._next_row,
                "embedding_dim": self.embedding_dim,
            }, f)
        logger.info(f"FAISS index saved to {path}  ({self._index.ntotal} vectors)")

    def load(self, path: str) -> None:
        """Load a previously saved index."""
        import faiss
        p = Path(path)
        self._index = faiss.read_index(str(p / "index.faiss"))
        with open(p / "state.pkl", "rb") as f:
            state = pickle.load(f)
        self._meta         = state["meta"]
        self._id_to_row    = state["id_to_row"]
        self._row_to_id    = state["row_to_id"]
        self._next_row     = state["next_row"]
        self.embedding_dim = state["embedding_dim"]
        logger.info(f"FAISS index loaded from {path}  ({self._index.ntotal} vectors)")

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0
