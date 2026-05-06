# tests/unit/test_retriever.py
"""Unit tests for HybridRetriever — uses mock vector store and real BM25."""

import pytest
from unittest.mock import MagicMock
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from src.retrieval.bm25_retriever import BM25Retriever


def make_record(cid, pid, text, source="doc.pdf"):
    return {"chunk_id": cid, "parent_chunk_id": pid, "text": text,
            "chunk_type": "child", "source_path": f"s3://{source}",
            "doc_id": "d1", "metadata": {}}


def make_parent(pid, text, source="doc.pdf"):
    return {"chunk_id": pid, "parent_chunk_id": "", "text": text,
            "chunk_type": "parent", "source_path": f"s3://{source}",
            "doc_id": "d1", "metadata": {}}


@pytest.fixture
def corpus():
    return [
        {"chunk_id": "c1", "text": "users table user_id UUID primary key email"},
        {"chunk_id": "c2", "text": "ETL pipeline Databricks checkpointing S3"},
        {"chunk_id": "c3", "text": "JWT token RS256 authentication 24 hours"},
        {"chunk_id": "c4", "text": "Delta Lake Z-ordering event_date pruning"},
        {"chunk_id": "c5", "text": "Kafka user_events topic registration service"},
    ]


@pytest.fixture
def vs(corpus):
    store = MagicMock()
    child_records = {
        c["chunk_id"]: {**c, "parent_chunk_id": f"p{c['chunk_id'][1:]}"}
        for c in corpus
    }
    parent_records = {
        f"p{c['chunk_id'][1:]}": make_parent(
            f"p{c['chunk_id'][1:]}", f"PARENT: {c['text']}"
        )
        for c in corpus
    }
    all_records = {**child_records, **parent_records}

    store.similarity_search.return_value = [
        {**child_records[c["chunk_id"]], "score": 0.9 - i * 0.1}
        for i, c in enumerate(corpus)
    ]
    store.get_by_id.side_effect = lambda cid: all_records.get(cid)
    return store


@pytest.fixture
def retriever(vs, corpus):
    bm25 = BM25Retriever(corpus=corpus)
    return HybridRetriever(vector_store=vs, bm25_retriever=bm25)


def test_returns_retrieved_chunks(retriever):
    results = retriever.retrieve([0.1]*3072, "users table UUID", top_k_final=3)
    assert len(results) > 0
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_no_duplicate_parents(retriever):
    results = retriever.retrieve([0.1]*3072, "table schema", top_k_final=5)
    pids = [r.parent_chunk_id for r in results]
    assert len(pids) == len(set(pids)), "Duplicate parent IDs"


def test_parent_text_injected(retriever):
    results = retriever.retrieve([0.1]*3072, "ETL pipeline", top_k_final=2)
    for r in results:
        assert "PARENT:" in r.text


def test_rrf_score_formula():
    r = HybridRetriever(vector_store=MagicMock(), bm25_retriever=MagicMock(), rrf_k=60)
    assert abs(r._rrf(0) - 1/60)  < 1e-9
    assert abs(r._rrf(1) - 1/61)  < 1e-9
    assert r._rrf(0) > r._rrf(10)


def test_overlap_candidate_scores_higher():
    r = HybridRetriever(MagicMock(), MagicMock(), rrf_k=60, dense_weight=0.6)
    dense  = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    sparse = [{"chunk_id": "a"}]
    fused  = r._fuse(dense, sparse, top_k=5)
    ids    = [f["chunk_id"] for f in fused]
    assert ids.index("a") < ids.index("b")


def test_display_source_strips_path():
    c = RetrievedChunk("c","p","t","ct",0.9,"s3://bucket/docs/schema.pdf","d",{})
    assert c.display_source == "schema.pdf"


# tests/unit/test_citation_extractor.py -------------------------------------------------------------

from src.generation.citation_extractor import CitationExtractor, CitationResult


def make_chunk(cid, score=0.9):
    return RetrievedChunk(cid, f"p{cid}", f"text of {cid}", "child", score, f"s3://{cid}.pdf", "d1", {})


ext = CitationExtractor()

def test_parses_citation():
    res = ext.extract("See the schema [Source 1].", [make_chunk("c1")])
    assert 1 in res.cited_indices
    assert res.cited_sources[0].chunk_id == "c1"


def test_parses_multiple():
    chunks = [make_chunk("c1"), make_chunk("c2")]
    res = ext.extract("Schema [Source 1] and pipeline [Source 2].", chunks)
    assert res.cited_indices == {1, 2}


def test_out_of_range_is_hallucinated():
    res = ext.extract("See [Source 9].", [make_chunk("c1")])
    assert 9 in res.hallucinated_indices
    assert 9 not in res.cited_indices


def test_faithfulness_warning_on_long_uncited_answer():
    res = ext.extract("X" * 200, [make_chunk("c1")])
    assert res.faithfulness_warning is True


def test_no_warning_on_no_info_response():
    res = ext.extract("I don't have enough information to answer.", [make_chunk("c1")])
    assert res.faithfulness_warning is False


def test_strip_sources_suffix():
    raw = "Answer here.\n__SOURCES__:[{\"index\":1}]"
    assert "__SOURCES__" not in ext._strip_sources(raw)


# tests/unit/test_metrics.py-------------------------------------------------------------

import math
from src.evaluation.metrics import (
    reciprocal_rank, ndcg_at_k, evaluate_retrieval
)


def test_rr_rank_1():
    assert abs(reciprocal_rank(["a","b","c"], {"a"}) - 1.0) < 1e-9

def test_rr_rank_2():
    assert abs(reciprocal_rank(["b","a","c"], {"a"}) - 0.5) < 1e-9

def test_rr_not_found():
    assert reciprocal_rank(["b","c"], {"a"}) == 0.0

def test_ndcg_perfect():
    assert abs(ndcg_at_k(["a","b","c"], {"a","b","c"}, 3) - 1.0) < 1e-6

def test_ndcg_zero():
    assert ndcg_at_k(["x","y"], {"a"}, 2) == 0.0

def test_evaluate_perfect():
    res = evaluate_retrieval(["q1","q2"], [["a"],["b"]], [{"a"},{"b"}], k=1)
    assert abs(res.mrr - 1.0)      < 1e-6
    assert abs(res.hit_rate - 1.0) < 1e-6

def test_evaluate_zero():
    res = evaluate_retrieval(["q1"], [["x","y"]], [{"a"}], k=5)
    assert res.mrr == 0.0
    assert res.hit_rate == 0.0
