# src/retrieval/reranker.py
"""
Cross-Encoder Reranker
======================
Takes top-K retrieved chunks and re-scores them using a cross-encoder.

Bi-encoder (retrieval):   embeds query and document INDEPENDENTLY → O(1) query
Cross-encoder (reranking): encodes (query, document) JOINTLY → O(K) inference

Cross-encoders are ~10× more accurate at relevance scoring but too slow
for full-corpus search.  Applying them to only the top-K from retrieval
gives the best of both worlds.

Pipeline:
  retrieve top-20 (fast bi-encoder) → rerank to top-5 (accurate cross-encoder)

Two backends:
  1. CohereReranker  — managed API, ~50ms/call, easy
  2. CrossEncoderReranker — local HuggingFace, free, better on tech docs,
                            requires transformers + torch
"""

from __future__ import annotations
import logging
from typing import List
from src.retrieval.hybrid_retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class CohereReranker:
    """
    Reranker using Cohere rerank-english-v3.0.

    Args:
        relevance_threshold: chunks below this score are dropped.
                             If ALL chunks are below threshold, returns [].
                             Caller should check for empty list and return
                             "no information found" rather than hallucinate.
    """

    MODEL = "rerank-english-v3.0"

    def __init__(self, relevance_threshold: float = 0.35):
        import cohere
        from src.config import get_env
        self.client    = cohere.Client(api_key=get_env().cohere_api_key)
        self.threshold = relevance_threshold

    def rerank(
        self,
        query:  str,
        chunks: List[RetrievedChunk],
        top_n:  int = 5,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []

        resp = self.client.rerank(
            model=self.MODEL,
            query=query,
            documents=[c.text for c in chunks],
            top_n=top_n,
            return_documents=False,
        )

        output = []
        for result in resp.results:
            if result.relevance_score < self.threshold:
                logger.debug(f"Chunk {result.index} below threshold ({result.relevance_score:.3f})")
                continue
            c = chunks[result.index]
            # Replace RRF score with cross-encoder score for citation confidence
            output.append(RetrievedChunk(
                chunk_id=c.chunk_id, parent_chunk_id=c.parent_chunk_id,
                text=c.text, child_text=c.child_text,
                score=result.relevance_score,
                source_path=c.source_path, doc_id=c.doc_id, metadata=c.metadata,
            ))

        logger.info(f"Reranked {len(chunks)} → {len(output)} chunks  (threshold={self.threshold})")
        return output


class CrossEncoderReranker:
    """
    Local reranker using BAAI/bge-reranker-v2-m3 (open-source, self-hosted).
    Outperforms Cohere on technical documentation.  Requires torch.
    """

    def __init__(
        self,
        model_name:  str   = "BAAI/bge-reranker-v2-m3",
        device:      str   = "auto",
        threshold:   float = 0.35,
    ):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        self.device    = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.threshold = threshold
        logger.info(f"BGE cross-encoder loaded on {self.device}")

    def _score(self, query: str, doc: str) -> float:
        import torch
        inputs = self.tokenizer(
            [[query, doc]], padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            return torch.sigmoid(self.model(**inputs).logits[0][0]).item()

    def rerank(
        self,
        query:  str,
        chunks: List[RetrievedChunk],
        top_n:  int = 5,
    ) -> List[RetrievedChunk]:
        scored = [
            RetrievedChunk(
                chunk_id=c.chunk_id, parent_chunk_id=c.parent_chunk_id,
                text=c.text, child_text=c.child_text,
                score=self._score(query, c.text),
                source_path=c.source_path, doc_id=c.doc_id, metadata=c.metadata,
            )
            for c in chunks
        ]
        filtered = [c for c in scored if c.score >= self.threshold]
        return sorted(filtered, key=lambda x: x.score, reverse=True)[:top_n]
