# src/evaluation/metrics.py
"""
Retrieval Evaluation Metrics
=============================
MRR@K, NDCG@K, Recall@K, Precision@K, Hit Rate@K.
Used to benchmark retrieval strategies independently of generation quality.

Typical workflow:
  1. Embed a set of test queries
  2. Run each retrieval strategy (dense / sparse / hybrid)
  3. Compare against ground-truth relevant_chunk_ids
  4. Call evaluate_retrieval() to get a RetrievalEvalResult
  5. Log to MLflow alongside RAGAS scores
"""

from __future__ import annotations
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RetrievalEvalResult:
    mrr:       float
    ndcg:      float
    recall:    float
    precision: float
    hit_rate:  float
    k:         int
    num_queries: int

    def to_dict(self) -> Dict:
        return {
            f"MRR@{self.k}":       round(self.mrr,       4),
            f"NDCG@{self.k}":      round(self.ndcg,      4),
            f"Recall@{self.k}":    round(self.recall,    4),
            f"Precision@{self.k}": round(self.precision, 4),
            f"HitRate@{self.k}":   round(self.hit_rate,  4),
            "num_queries":         self.num_queries,
        }

    def print_report(self):
        print(f"\n{'═'*44}")
        print(f"  Retrieval Evaluation  K={self.k}  n={self.num_queries}")
        print(f"{'═'*44}")
        for k, v in self.to_dict().items():
            print(f"  {k:<22}: {v}")
        print(f"{'═'*44}\n")


# ── Individual metric functions ───────────────────────────────────────────────

def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    for rank, did in enumerate(retrieved, 1):
        if did in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    return sum(
        (1.0 if did in relevant else 0.0) / math.log2(rank + 1)
        for rank, did in enumerate(retrieved[:k], 1)
    )


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    actual = dcg_at_k(retrieved, relevant, k)
    ideal  = sum(
        1.0 / math.log2(i + 2)
        for i in range(min(len(relevant), k))
    )
    return actual / ideal if ideal > 0 else 0.0


# ── Aggregate evaluation ──────────────────────────────────────────────────────

def evaluate_retrieval(
    queries:         List[str],
    retrieved_lists: List[List[str]],   # ordered chunk_ids per query
    relevant_sets:   List[Set[str]],    # ground-truth chunk_ids per query
    k:               int = 10,
) -> RetrievalEvalResult:
    """
    Compute MRR, NDCG, Recall, Precision, Hit Rate over a query set.

    Args:
        queries:         query strings (for error reporting)
        retrieved_lists: for each query, ordered list of retrieved chunk IDs
        relevant_sets:   for each query, set of ground-truth relevant chunk IDs
        k:               rank cutoff

    Returns:
        RetrievalEvalResult with aggregate scores
    """
    assert len(retrieved_lists) == len(relevant_sets), "Length mismatch"

    mrr_s, ndcg_s, rec_s, prec_s, hit_s = [], [], [], [], []

    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        top_k   = retrieved[:k]
        matched = set(top_k) & relevant

        mrr_s.append(reciprocal_rank(retrieved, relevant))
        ndcg_s.append(ndcg_at_k(retrieved, relevant, k))
        rec_s.append(len(matched) / len(relevant) if relevant else 0.0)
        prec_s.append(len(matched) / len(top_k)   if top_k    else 0.0)
        hit_s.append(1.0 if matched else 0.0)

    return RetrievalEvalResult(
        mrr=float(np.mean(mrr_s)),
        ndcg=float(np.mean(ndcg_s)),
        recall=float(np.mean(rec_s)),
        precision=float(np.mean(prec_s)),
        hit_rate=float(np.mean(hit_s)),
        k=k,
        num_queries=len(queries),
    )


# ── Generation quality stats ──────────────────────────────────────────────────

def generation_stats(answers: List[str], sources_lists: List[List]) -> pd.DataFrame:
    """
    Compute per-answer quality signals (no ground truth required).

    Columns: answer_length, citation_count, has_citations,
             is_no_answer, faithfulness_warning
    """
    import re
    CITE_RE  = re.compile(r"\[Source\s+\d+\]", re.IGNORECASE)
    NO_INFO  = ["don't have enough information", "not found in", "no relevant"]

    rows = []
    for answer, sources in zip(answers, sources_lists):
        cites   = CITE_RE.findall(answer)
        no_info = any(p in answer.lower() for p in NO_INFO)
        rows.append({
            "answer_length":         len(answer),
            "citation_count":        len(cites),
            "has_citations":         len(cites) > 0,
            "is_no_answer":          no_info,
            "num_sources_retrieved": len(sources),
            "faithfulness_warning":  len(cites) == 0 and len(answer) > 100 and not no_info,
        })

    df = pd.DataFrame(rows)
    logger.info(
        f"Generation stats | "
        f"avg_len={df['answer_length'].mean():.0f} | "
        f"cited={df['has_citations'].mean()*100:.1f}% | "
        f"no_answer={df['is_no_answer'].mean()*100:.1f}% | "
        f"faith_warn={df['faithfulness_warning'].mean()*100:.1f}%"
    )
    return df


# ── RAG quality metrics added for run_eval.py ─────────────────────────────

def rouge_l_score(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1. Requires: pip install rouge-score"""
    try:
        from rouge_score import rouge_scorer as rs
        return rs.RougeScorer(["rougeL"], use_stemmer=True).score(reference, hypothesis)["rougeL"].fmeasure
    except ImportError:
        return 0.0


def context_precision(retrieved_chunks: list, ground_truth: str,
                      overlap_threshold: float = 0.08) -> float:
    """Fraction of retrieved chunks with >=8% token overlap with ground truth.
    Maps to RAGAS context_precision (which uses an LLM judge)."""
    gt_toks = set(ground_truth.lower().split())
    if not retrieved_chunks or not gt_toks:
        return 0.0
    relevant = sum(
        1 for c in retrieved_chunks
        if len(gt_toks & set(c.get("text","").lower().split())) / len(gt_toks) >= overlap_threshold
    )
    return relevant / len(retrieved_chunks)


def context_recall(retrieved_chunks: list, ground_truth: str,
                   min_token_len: int = 4) -> float:
    """Fraction of ground-truth keywords (>3 chars) covered by retrieved context.
    Maps to RAGAS context_recall (which uses an LLM judge)."""
    gt_toks = {w for w in ground_truth.lower().split() if len(w) >= min_token_len}
    if not gt_toks:
        return 0.0
    ctx = " ".join(c.get("text","") for c in retrieved_chunks).lower()
    return len(gt_toks & set(ctx.split())) / len(gt_toks)


def faithfulness_score(answer: str, retrieved_chunks: list,
                       support_threshold: float = 0.25) -> float:
    """Fraction of answer sentences where >=25% of tokens appear in context.
    Maps to RAGAS faithfulness (which uses NLI entailment)."""
    import re
    ctx_toks = set(" ".join(c.get("text","") for c in retrieved_chunks).lower().split())
    sents = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 15]
    if not sents:
        return 1.0
    supported = sum(
        1 for s in sents
        if len(set(s.lower().split()) & ctx_toks) / max(len(s.split()), 1) >= support_threshold
    )
    return supported / len(sents)


def answer_relevancy_score(question: str, answer: str) -> float:
    """BM25 overlap between answer and question, normalised to [0,1].
    Maps to RAGAS answer_relevancy (which generates reverse questions via LLM)."""
    from rank_bm25 import BM25Okapi
    q_toks = question.lower().split()
    a_toks = answer.lower().split()
    if not q_toks or not a_toks:
        return 0.5
    mini = BM25Okapi([q_toks])
    sq = mini.get_scores(q_toks)[0]
    sa = mini.get_scores(a_toks)[0]
    return min(sa / sq, 1.0) if sq > 0 else 0.5
