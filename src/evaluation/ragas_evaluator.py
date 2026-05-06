# src/evaluation/ragas_evaluator.py
"""
RAGAS Evaluation Pipeline
==========================
Evaluates the complete RAG pipeline end-to-end.
All results are logged to MLflow for tracking across pipeline versions.

RAGAS metrics (all score 0–1, higher is better):

  faithfulness       — are all claims in the answer supported by context?
                       0 = pure hallucination, 1 = fully grounded
                       THE most important metric for a production RAG system.

  answer_relevancy   — does the answer address what was actually asked?
                       Catches verbose, off-topic answers.

  context_precision  — of the chunks retrieved, how many were actually useful?
                       Low → retriever is pulling in noise.

  context_recall     — were ALL chunks needed to answer the question retrieved?
                       Low → retriever missed important documents.

  answer_correctness — factual accuracy vs ground-truth (requires labelled data).

Usage:
    evaluator = RAGASEvaluator()
    df = evaluator.evaluate_from_file("data/eval/qa_pairs.json", rag_chain)
    print(df[["question","faithfulness","answer_relevancy"]].head())
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import mlflow
import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    Orchestrates RAGAS evaluation over a QA dataset.

    Args:
        experiment_name:   MLflow experiment name
        metrics:           subset of RAGAS metrics to compute (default: all five)
        mlflow_tracking_uri: MLflow server URI  (default: sqlite:///mlflow.db)
    """

    ALL_METRICS = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]

    def __init__(
        self,
        experiment_name:    str            = "rag_evaluation",
        metrics:            Optional[List] = None,
        mlflow_tracking_uri:Optional[str]  = None,
    ):
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.metric_names = metrics or self.ALL_METRICS

    def _load_metrics(self):
        """Lazy-import RAGAS metrics (slow to import, avoid at module level)."""
        from ragas.metrics import (
            faithfulness, answer_relevancy,
            context_precision, context_recall, answer_correctness,
        )
        name_map = {
            "faithfulness":        faithfulness,
            "answer_relevancy":    answer_relevancy,
            "context_precision":   context_precision,
            "context_recall":      context_recall,
            "answer_correctness":  answer_correctness,
        }
        return [name_map[n] for n in self.metric_names if n in name_map]

    # ── Inference helpers ─────────────────────────────────────────────────

    async def _run_one(self, chain, question: str) -> dict:
        result = await chain.query_sync(question)
        return {
            "answer":   result.get("answer", ""),
            "contexts": [
                s.get("display_source", "") + ": " + s.get("text_preview", "")
                for s in result.get("sources", [])
            ],
        }

    async def _run_all(self, chain, questions: List[str], batch_size: int = 5) -> List[dict]:
        results = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            logger.info(
                f"Evaluating batch {i//batch_size+1}/"
                f"{(len(questions)+batch_size-1)//batch_size}  ({len(batch)} questions)"
            )
            batch_results = await asyncio.gather(
                *[self._run_one(chain, q) for q in batch]
            )
            results.extend(batch_results)
        return results

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate_from_file(
        self,
        qa_path:    str,
        rag_chain,
        run_name:   Optional[str] = None,
        batch_size: int           = 5,
    ) -> pd.DataFrame:
        """
        Load QA pairs from JSON, run through RAG chain, compute RAGAS metrics.

        QA file format (list of objects):
            [
              {
                "question":     "What is the primary key of the users table?",
                "ground_truth": "The primary key is user_id, a UUID type."
              },
              ...
            ]

        Args:
            qa_path:    path to QA JSON file
            rag_chain:  initialised RAGChain instance
            run_name:   MLflow run name (auto-generated if not provided)
            batch_size: async batch size for inference

        Returns:
            DataFrame with per-question RAGAS scores.
        """
        p = Path(qa_path)
        if not p.exists():
            raise FileNotFoundError(f"QA file not found: {qa_path}")

        with open(p) as f:
            qa_data = json.load(f)

        questions    = [d["question"] for d in qa_data]
        ground_truths= [d.get("ground_truth", "") for d in qa_data]
        logger.info(f"Evaluating {len(questions)} questions from {p.name}")

        # ── Run inference ─────────────────────────────────────────────────
        t0      = time.time()
        outputs = asyncio.run(self._run_all(rag_chain, questions, batch_size))
        inf_sec = time.time() - t0
        logger.info(f"Inference complete: {inf_sec:.1f}s  ({len(questions)} questions)")

        answers  = [o["answer"]   for o in outputs]
        contexts = [o["contexts"] for o in outputs]

        # ── RAGAS evaluation ──────────────────────────────────────────────
        from ragas import evaluate
        dataset = Dataset.from_dict({
            "question":    questions,
            "answer":      answers,
            "contexts":    contexts,
            "ground_truth":ground_truths,
        })
        metrics = self._load_metrics()
        rname   = run_name or f"rag_eval_{int(time.time())}"

        with mlflow.start_run(run_name=rname):
            result   = evaluate(dataset=dataset, metrics=metrics)
            scores_df= result.to_pandas()

            # Log aggregate metrics
            logger.info(f"\n{'='*50}\n  RAGAS Results — {rname}\n{'='*50}")
            for col in self.metric_names:
                if col in scores_df.columns:
                    avg = scores_df[col].mean()
                    mlflow.log_metric(col, round(avg, 4))
                    logger.info(f"  {col:<25}: {avg:.4f}")

            mlflow.log_params({
                "num_questions":   len(questions),
                "inference_time":  round(inf_sec, 1),
                "avg_latency_s":   round(inf_sec / max(len(questions), 1), 2),
                "qa_file":         p.name,
            })

            csv_path = "ragas_scores.csv"
            scores_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)

        return scores_df

    def evaluate_from_lists(
        self,
        questions:    List[str],
        answers:      List[str],
        contexts:     List[List[str]],
        ground_truths:List[str],
        run_name:     Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Evaluate pre-computed answers (avoids re-running inference).
        Useful for ablations: test different retrieval configs with same answers.
        """
        from ragas import evaluate
        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        metrics = self._load_metrics()

        with mlflow.start_run(run_name=run_name or "ragas_eval"):
            result   = evaluate(dataset=dataset, metrics=metrics)
            scores_df= result.to_pandas()
            for col in scores_df.columns:
                if col not in ("question", "answer", "contexts", "ground_truth"):
                    mlflow.log_metric(col, round(scores_df[col].mean(), 4))

        return scores_df
