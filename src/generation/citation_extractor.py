# src/generation/citation_extractor.py
"""
Citation Extractor & Faithfulness Validator
===========================================
Parses [Source N] references from LLM output and validates them
against the retrieved chunk list.

Detects two failure modes:
  1. Hallucinated citation — LLM references [Source 5] but only 3 chunks existed.
  2. Uncited answer       — LLM produces a long answer with zero [Source N] refs.
     This is a faithfulness red flag: the model is likely hallucinating facts.

Both signals are logged and returned in quality_signals so the API consumer
(monitoring dashboard / human reviewer) can act on them.
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import List, Set

from src.retrieval.hybrid_retriever import RetrievedChunk

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


@dataclass
class CitationResult:
    answer:               str
    cited_indices:        Set[int]
    cited_sources:        List[RetrievedChunk]
    uncited_chunks:       List[RetrievedChunk]
    hallucinated_indices: List[int]
    faithfulness_warning: bool


class CitationExtractor:
    """Parse and validate citations in a RAG-generated answer."""

    def extract(self, answer: str, chunks: List[RetrievedChunk]) -> CitationResult:
        """
        Parse citations and cross-reference against retrieved chunks.

        Args:
            answer: generated LLM text
            chunks: the RetrievedChunk list used as context (1-indexed in prompt)
        """
        raw = {int(m) for m in CITATION_RE.findall(answer)}

        valid    = {i for i in raw if 1 <= i <= len(chunks)}
        halluc   = sorted(raw - valid)

        if halluc:
            logger.warning(f"Hallucinated citation indices: {halluc}")

        cited    = [chunks[i - 1] for i in sorted(valid)]
        cited_ids= {c.chunk_id for c in cited}
        uncited  = [c for c in chunks if c.chunk_id not in cited_ids]

        is_no_info = "don't have enough information" in answer.lower()
        faith_warn = (
            len(raw) == 0
            and len(answer.strip()) > 80
            and not is_no_info
        )
        if faith_warn:
            logger.warning("Faithfulness warning: non-trivial answer with zero citations.")

        return CitationResult(
            answer=answer,
            cited_indices=valid,
            cited_sources=cited,
            uncited_chunks=uncited,
            hallucinated_indices=halluc,
            faithfulness_warning=faith_warn,
        )

    def to_api_dict(self, result: CitationResult) -> dict:
        """Serialize for the /chat JSON response."""
        return {
            "answer": self._strip_sources(result.answer),
            "sources": [
                {
                    "index":        i + 1,
                    "display_source": c.display_source,
                    "source_path":  c.source_path,
                    "relevance_score": round(c.score, 3),
                }
                for i, c in enumerate(result.cited_sources)
            ],
            "quality_signals": {
                "citation_count":        len(result.cited_indices),
                "hallucinated_citations": result.hallucinated_indices,
                "faithfulness_warning":  result.faithfulness_warning,
                "uncited_source_count":  len(result.uncited_chunks),
            },
        }

    @staticmethod
    def _strip_sources(text: str) -> str:
        """Remove __SOURCES__ JSON suffix from streaming responses."""
        return text.split("\n__SOURCES__:")[0] if "\n__SOURCES__:" in text else text
