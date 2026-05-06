# tests/conftest.py
"""Shared fixtures for unit and integration tests."""

import hashlib
import pytest
import numpy as np
from unittest.mock import MagicMock

DIM = 64


def det_emb(text: str, dim: int = DIM) -> list:
    h = int(hashlib.md5(text.encode()).hexdigest(), 16)
    np.random.seed(h % (2**32))
    v = np.random.randn(dim).astype(np.float32)
    return (v / (np.linalg.norm(v) + 1e-9)).tolist()


@pytest.fixture
def fake_embedder():
    e = MagicMock()
    e.embed_query.side_effect      = lambda t: det_emb(t)
    e.embed_single.side_effect     = lambda t: det_emb(t)
    e.embed_documents.side_effect  = lambda ts: [det_emb(t) for t in ts]
    e.embed_batch.side_effect      = lambda ts: [det_emb(t) for t in ts]
    return e


@pytest.fixture
def sample_corpus():
    return [
        {"chunk_id": f"c{i}", "text": text}
        for i, text in enumerate([
            "users table user_id UUID primary key email VARCHAR",
            "ETL pipeline Databricks PySpark Delta Lake S3 checkpointing",
            "JWT token RS256 authentication 24 hour TTL access refresh",
            "Kafka user_events topic consumer registration service",
            "Delta Lake Z-ordering event_date partition pruning performance",
        ])
    ]
