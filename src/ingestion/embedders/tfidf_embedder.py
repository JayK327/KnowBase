# src/ingestion/embedders/tfidf_embedder.py
"""
TF-IDF Embedder — zero API key, drop-in for OpenAIEmbedder.
Identical interface: embed_documents(), embed_query(), embed_single(), embed_batch().
Vectors are L2-normalised so inner product = cosine similarity in FAISSVectorStore.
"""
from __future__ import annotations
import logging, pickle
from pathlib import Path
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class TFIDFEmbedder:
    MODEL = "tfidf-local"  # mirrors OpenAIEmbedder.MODEL

    def __init__(self, max_features: int = 8_000, ngram_range: tuple = (1, 2),
                 sublinear_tf: bool = True):
        self._vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                                    sublinear_tf=sublinear_tf, min_df=1)
        self._fitted = False
        self.embedding_dim: Optional[int] = None

    def fit(self, texts: List[str]) -> "TFIDFEmbedder":
        self._vec.fit(texts)
        self._fitted = True
        self.embedding_dim = len(self._vec.vocabulary_)
        logger.info(f"TFIDFEmbedder fitted: vocab={self.embedding_dim}")
        return self

    def _check(self):
        if not self._fitted:
            raise RuntimeError("Call embed_documents() first to fit the vocabulary.")

    def _encode(self, texts: List[str]) -> List[List[float]]:
        self._check()
        m = self._vec.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        return (m / norms).tolist()

    # ── Public API (identical to OpenAIEmbedder) ──────────────────────────
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Fit on corpus if not already fitted, then embed. Used by IngestionPipeline."""
        if not self._fitted:
            logger.info(f"Auto-fitting on {len(texts)} texts...")
            self.fit(texts)
        return self._encode(texts)

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query at inference time."""
        return self._encode([query])[0]

    def embed_single(self, text: str) -> List[float]:
        return self._encode([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._check()
        return self._encode(texts)

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        with open(p / "tfidf.pkl", "wb") as f:
            pickle.dump(self._vec, f)

    @classmethod
    def load(cls, path: str) -> "TFIDFEmbedder":
        with open(Path(path) / "tfidf.pkl", "rb") as f:
            v = pickle.load(f)
        o = cls.__new__(cls)
        o._vec = v; o._fitted = True
        o.embedding_dim = len(v.vocabulary_)
        return o
