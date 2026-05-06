# tests/unit/test_chunker.py
"""
Unit tests for HierarchicalChunker.
Tests: parent-child linking, token limits, empty doc, metadata propagation.
"""

import pytest
from src.ingestion.chunkers.hierarchical_chunker import HierarchicalChunker
from src.ingestion.models import ExtractedDocument, DocumentFormat, ChunkType


@pytest.fixture
def chunker():
    return HierarchicalChunker(
        parent_chunk_size=80,
        child_chunk_size=25,
        parent_overlap=8,
        child_overlap=4,
    )


@pytest.fixture
def doc():
    return ExtractedDocument(
        doc_id="test-001",
        source_path="s3://test/schema.pdf",
        text=(
            "The users table stores user_id (UUID primary key), email (VARCHAR 255 UNIQUE), "
            "created_at (TIMESTAMP), updated_at (TIMESTAMP), and is_active (BOOLEAN). "
            "The table is partitioned by created_at for query performance. "
            "It is updated by the user_registration_consumer Kafka topic. "
            "Foreign key: user_id references accounts.account_id. "
            "Retention policy: 7 years per GDPR compliance requirements."
        ),
        format=DocumentFormat.PDF,
    )


def test_produces_both_types(chunker, doc):
    chunks = chunker.chunk_document(doc)
    types  = {c.chunk_type for c in chunks}
    assert ChunkType.PARENT in types
    assert ChunkType.CHILD  in types


def test_every_child_links_to_a_parent(chunker, doc):
    chunks     = chunker.chunk_document(doc)
    parent_ids = {c.chunk_id for c in chunks if c.is_parent}
    for c in chunks:
        if c.is_child:
            assert c.parent_chunk_id in parent_ids, (
                f"Child {c.chunk_id} points to unknown parent {c.parent_chunk_id}"
            )


def test_parents_have_no_parent_id(chunker, doc):
    for c in chunker.chunk_document(doc):
        if c.is_parent:
            assert c.parent_chunk_id is None


def test_child_token_counts_within_limit(chunker, doc):
    for c in chunker.chunk_document(doc):
        if c.is_child:
            assert c.token_count <= chunker.child_splitter._chunk_size + 10


def test_empty_text_returns_empty_list(chunker):
    doc = ExtractedDocument(doc_id="e", source_path="x", text="", format=DocumentFormat.TXT)
    assert chunker.chunk_document(doc) == []


def test_whitespace_only_returns_empty_list(chunker):
    doc = ExtractedDocument(doc_id="w", source_path="x", text="  \n\n  ", format=DocumentFormat.TXT)
    assert chunker.chunk_document(doc) == []


def test_chunk_ids_are_unique(chunker, doc):
    ids = [c.chunk_id for c in chunker.chunk_document(doc)]
    assert len(ids) == len(set(ids))


def test_metadata_contains_source_path(chunker, doc):
    for c in chunker.chunk_document(doc):
        assert c.metadata["source_path"] == doc.source_path


def test_parent_indices_are_sequential(chunker, doc):
    parents = [c for c in chunker.chunk_document(doc) if c.is_parent]
    assert [c.chunk_index for c in parents] == list(range(len(parents)))


# tests/unit/test_bm25_retriever.py
"""Unit tests for BM25Retriever."""

import os, tempfile, pytest
from src.retrieval.bm25_retriever import BM25Retriever


@pytest.fixture
def corpus():
    return [
        {"chunk_id": "c1", "text": "The ETL_JOB_ID column is a UUID primary key in the jobs table"},
        {"chunk_id": "c2", "text": "User authentication uses JWT tokens with 24-hour expiry"},
        {"chunk_id": "c3", "text": "Kafka consumer reads from user_events topic at offset 0"},
        {"chunk_id": "c4", "text": "Delta Lake tables use Z-ordering on timestamp for pruning"},
        {"chunk_id": "c5", "text": "ETL pipeline processes 10000 records per batch with checkpointing"},
    ]


@pytest.fixture
def bm25(corpus):
    return BM25Retriever(corpus=corpus)


def test_returns_results(bm25):
    assert len(bm25.retrieve("ETL pipeline job")) > 0


def test_relevant_result_is_first(bm25):
    results = bm25.retrieve("ETL_JOB_ID UUID", k=5)
    assert "etl" in results[0]["text"].lower() or "uuid" in results[0]["text"].lower()


def test_scores_are_descending(bm25):
    results = bm25.retrieve("Kafka consumer events", k=5)
    scores  = [r["bm25_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_empty(bm25):
    assert bm25.retrieve("") == []


def test_empty_corpus_returns_empty():
    assert BM25Retriever(corpus=[]).retrieve("anything") == []


def test_update_adds_docs(bm25):
    n = len(bm25.corpus)
    bm25.update([{"chunk_id": "c99", "text": "New Spark streaming document"}])
    assert len(bm25.corpus) == n + 1
    assert any(r["chunk_id"] == "c99" for r in bm25.retrieve("Spark streaming"))


def test_update_avoids_duplicates(bm25):
    n = len(bm25.corpus)
    bm25.update([{"chunk_id": "c1", "text": "duplicate entry"}])
    assert len(bm25.corpus) == n


def test_save_and_load_roundtrip(bm25):
    with tempfile.TemporaryDirectory() as tmp:
        bm25.save(tmp)
        loaded  = BM25Retriever.load(tmp)
        orig    = bm25.retrieve("ETL job", k=3)
        restored= loaded.retrieve("ETL job", k=3)
        assert [r["chunk_id"] for r in orig] == [r["chunk_id"] for r in restored]
