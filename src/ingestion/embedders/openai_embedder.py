# src/ingestion/embedders/openai_embedder.py
"""
OpenAI Embedder — text-embedding-3-large
=========================================
Produces 3072-dimensional embeddings. Supports:
  - Dimension reduction (e.g. 1536-d via `dimensions` param)
  - Batch embedding with automatic page-sizing
  - Exponential backoff on RateLimitError
  - Separate embed_query() vs embed_documents() (different normalization intent)

Why text-embedding-3-large?
  - Best OpenAI MTEB score → better retrieval quality
  - Larger context window → better HyDE query expansion
  - Supports shrinking dims with no retraining
  - Trivially swappable for Cohere Embed v3 via the CohereEmbedder class below
"""

from __future__ import annotations
import logging
import time
from typing import List, Optional

from openai import OpenAI
from openai import RateLimitError, APIError
from src.config import get_env

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """
    Batch embedding via OpenAI Embeddings API.

    Args:
        model:      OpenAI embedding model
        dimensions: optional output dimensions (Matryoshka truncation)
        batch_size: max texts per API call (API cap: 2048)
        max_retries: retry budget for rate-limit errors
    """

    MODEL      = "text-embedding-3-large"
    BATCH_SIZE = 100
    MAX_RETRIES = 4

    def __init__(
        self,
        model:      str           = "text-embedding-3-large",
        dimensions: Optional[int] = None,   # None → full 3072-d
        batch_size: int           = 100,
    ):
        self.client     = OpenAI(api_key=get_env().openai_api_key)
        self.model      = model
        self.dimensions = dimensions
        self.batch_size = batch_size

    def _call(self, texts: List[str]) -> List[List[float]]:
        """Single API call — one batch of up to batch_size texts."""
        kwargs = dict(model=self.model, input=texts)
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self.client.embeddings.create(**kwargs)
                # API returns embeddings in the same order as input
                return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
            except RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"OpenAI rate-limited. Retry {attempt+1}/{self.MAX_RETRIES} in {wait}s")
                time.sleep(wait)
            except APIError as e:
                logger.error(f"OpenAI API error: {e}")
                raise
        raise RuntimeError(f"Embedding failed after {self.MAX_RETRIES} retries")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts in batches.
        Returns embeddings in the same order as input.
        """
        results: List[List[float]] = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            bn = i // self.batch_size + 1
            logger.info(f"Embedding batch {bn}/{n_batches}  ({len(batch)} texts)")
            results.extend(self._call(batch))

        return results

    def embed_single(self, text: str) -> List[float]:
        """Embed a single string.  Used for query embedding at inference time."""
        return self._call([text])[0]

    def embed_query(self, query: str) -> List[float]:
        """Alias for clarity — embeds a search query."""
        return self.embed_single(query)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Alias for clarity — embeds document chunks at index time."""
        return self.embed_batch(texts)


# ---------------------------------------------------------------------------
# Optional: Cohere embedder as a drop-in replacement
# ---------------------------------------------------------------------------

class CohereEmbedder:
    """
    Drop-in alternative using Cohere Embed v3.
    The key difference: Cohere requires input_type differentiation
    ('search_document' at index time, 'search_query' at query time).
    Using the wrong type places query/doc in different embedding spaces
    and silently degrades retrieval quality.
    """

    MODEL      = "embed-english-v3.0"
    BATCH_SIZE = 96   # Cohere API max

    def __init__(self):
        import cohere
        self.client = cohere.Client(api_key=get_env().cohere_api_key)

    def embed_batch(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        all_embs: List[List[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            resp = self.client.embed(texts=batch, model=self.MODEL, input_type=input_type, embedding_types=["float"])
            all_embs.extend(resp.embeddings.float_)
        return all_embs

    def embed_single(self, text: str) -> List[float]:
        return self.embed_batch([text], input_type="search_document")[0]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_batch([query], input_type="search_query")[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_batch(texts, input_type="search_document")
