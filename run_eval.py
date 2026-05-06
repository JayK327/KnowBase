"""
run_eval.py  —  DataChat RAG Evaluation
========================================

src/ modules used:
  src/ingestion/models.py                  RawDocument, DocumentFormat
  src/ingestion/embedders/tfidf_embedder   TFIDFEmbedder  (no API key)
  src/ingestion/pipeline.py                IngestionPipeline
  src/retrieval/vector_store/faiss_vs.py   FAISSVectorStore
  src/retrieval/bm25_retriever.py          BM25Retriever
  src/retrieval/hybrid_retriever.py        HybridRetriever (RRF + parent swap)
  src/evaluation/metrics.py               ALL metric functions

Only two things here have no src/ equivalent:
  build_relevant_sets()         — approximates ground-truth labels via BM25
  generate_answer_extractive()  — replaces GPT-4o for zero-cost evaluation

Usage:
  python run_eval.py
  python run_eval.py --raw-dir data/raw --k 5

Output:
  data/outputs/evaluation_results.json
  data/outputs/evaluation_report.md
"""

import argparse, json, logging, re, time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from src.ingestion.models import RawDocument, DocumentFormat
from src.ingestion.embedders.tfidf_embedder import TFIDFEmbedder
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.vector_store.faiss_vs import FAISSVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.evaluation.metrics import (
    evaluate_retrieval, reciprocal_rank,
    context_precision, context_recall,
    faithfulness_score, answer_relevancy_score, rouge_l_score,
)

logging.basicConfig(level=logging.WARNING)

DEFAULT_RAW_DIR   = Path("data/raw")
DEFAULT_EVAL_FILE = Path("data/eval/qa_pairs.json")
DEFAULT_OUT_JSON  = Path("data/outputs/evaluation_results.json")
DEFAULT_OUT_MD    = Path("data/outputs/evaluation_report.md")

EXT_FORMAT = {".md": DocumentFormat.MD, ".txt": DocumentFormat.TXT,
              ".html": DocumentFormat.HTML, ".htm": DocumentFormat.HTML}


# ── Step 1: Load documents ────────────────────────────────────────────────
# src used: src/ingestion/models.RawDocument

def load_raw_documents(raw_dir: Path) -> List[RawDocument]:
    docs = []
    for p in sorted(raw_dir.iterdir()):
        fmt = EXT_FORMAT.get(p.suffix.lower())
        if not fmt:
            continue
        docs.append(RawDocument(
            source_path=str(p), format=fmt,
            raw_bytes=p.read_bytes(),
            metadata={"filename": p.name},
        ))
    if not docs:
        raise ValueError(f"No supported documents in {raw_dir}")
    print(f"  Loaded {len(docs)} docs  ({sum(len(d.raw_bytes) for d in docs):,} chars)")
    return docs


# ── Step 2: Build indices ─────────────────────────────────────────────────
# src used: IngestionPipeline, TFIDFEmbedder, FAISSVectorStore,
#           BM25Retriever, HybridRetriever

def build_indices(raw_docs: List[RawDocument]) -> tuple:
    """
    Run IngestionPipeline with TFIDFEmbedder (zero API cost).
    TFIDFEmbedder has identical interface to OpenAIEmbedder — swap in
    OpenAIEmbedder for production-quality embeddings.
    """
    embedder     = TFIDFEmbedder(max_features=8_000, ngram_range=(1, 2))
    vector_store = FAISSVectorStore(embedding_dim=8_000)
    pipeline     = IngestionPipeline(vector_store=vector_store, embedder=embedder)

    all_chunks_map: Dict[str, dict] = {}

    for doc in raw_docs:
        extracted = pipeline.extract(doc)
        if not extracted:
            continue
        chunks = pipeline.chunk(extracted)
        if not chunks:
            continue
        pipeline.upsert(chunks, pipeline.embed(chunks))
        for c in chunks:
            all_chunks_map[c.chunk_id] = {
                "chunk_id":        c.chunk_id,
                "parent_chunk_id": c.parent_chunk_id or "",
                "text":            c.text,
                "chunk_type":      c.chunk_type.value,
                "source":          doc.metadata.get("filename", c.source_path),
            }

    children = [v for v in all_chunks_map.values() if v["chunk_type"] == "child"]
    parents  = [v for v in all_chunks_map.values() if v["chunk_type"] == "parent"]
    print(f"  Indexed  {len(parents)} parents, {len(children)} children")
    print(f"  FAISS    {vector_store.total_vectors} vectors  dim={embedder.embedding_dim}")

    bm25 = BM25Retriever(corpus=[
        {"chunk_id": c["chunk_id"], "text": c["text"]} for c in children
    ])
    print(f"  BM25     {len(children)} documents")

    retriever = HybridRetriever(
        vector_store=vector_store, bm25_retriever=bm25, rrf_k=60, dense_weight=0.6,
    )
    return embedder, vector_store, bm25, retriever, children, parents


# ── Step 3: Ground-truth relevant sets ───────────────────────────────────
# src used: BM25Retriever.retrieve()

def build_relevant_sets(qa_data: List[Dict], bm25: BM25Retriever,
                        top_n: int = 4) -> List[Set[str]]:
    """
    Proxy for human-annotated relevance labels.
    Run BM25 against each ground_truth answer — top_n results = relevant set.
    No src/ equivalent because this is specific to evaluation approximation.
    """
    return [
        {r["chunk_id"] for r in bm25.retrieve(qa["ground_truth"], k=top_n)}
        for qa in qa_data
    ]


# ── Step 4: Extractive answer (eval stub — replaces GPT-4o) ──────────────
# No src/ equivalent. In production use: asyncio.run(chain.query_sync(q))

def generate_answer_extractive(question: str, ctx_chunks: List[dict]) -> str:
    q_toks = set(question.lower().split())
    scored = [
        (len(q_toks & set(s.strip().lower().split())), s.strip())
        for c in ctx_chunks
        for s in re.split(r"[.!?\n]", c.get("text", ""))
        if len(s.strip()) > 20
    ]
    top = [s for _, s in sorted(scored, reverse=True)[:4] if s]
    return ". ".join(top) + "." if top else "Not found in retrieved context."


# ── Step 5: Per-question evaluation loop ─────────────────────────────────
# src used: FAISSVectorStore.similarity_search()
#           BM25Retriever.retrieve()
#           HybridRetriever.retrieve()  → _fuse() + _promote_to_parent() inside
#           metrics.*  all scoring functions

def evaluate_questions(qa_data, relevant_sets, embedder,
                       vector_store, bm25, retriever, K) -> tuple:
    print(f"  {'Question':<52}  {'faith':>5}  {'c_prec':>6}  {'c_rec':>5}  {'relev':>5}  {'rouge':>5}")
    print("  " + "-" * 88)

    dense_all, sparse_all, hybrid_all, per_q = [], [], [], []

    for i, (qa, rel_set) in enumerate(zip(qa_data, relevant_sets)):
        q, gt = qa["question"], qa["ground_truth"]

        # Dense — TFIDFEmbedder.embed_query() → FAISSVectorStore.similarity_search()
        q_emb     = embedder.embed_query(q)
        dense_ids = [r["chunk_id"] for r in vector_store.similarity_search(q_emb, k=K*2)
                     if isinstance(r, dict) and r.get("chunk_id")]

        # Sparse — BM25Retriever.retrieve()
        sparse_ids = [r["chunk_id"] for r in bm25.retrieve(q, k=K*2)]

        # Hybrid — HybridRetriever.retrieve(): _fuse() + _promote_to_parent() internally
        hybrid_chunks = retriever.retrieve(
            query_embedding=q_emb, query_text=q,
            top_k_dense=K*2, top_k_sparse=K*2, top_k_final=K,
        )
        hybrid_ids = [c.chunk_id for c in hybrid_chunks]

        dense_all.append(dense_ids)
        sparse_all.append(sparse_ids)
        hybrid_all.append(hybrid_ids)

        # Context: top-5 promoted parent chunks (parent text already in c.text)
        ctx = [{"text": c.text, "source": c.display_source, "chunk_id": c.chunk_id}
               for c in hybrid_chunks[:5]]

        answer = generate_answer_extractive(q, ctx)

        # All from src/evaluation/metrics.py
        rl       = rouge_l_score(answer, gt)
        ctx_prec = context_precision(ctx, gt)
        ctx_rec  = context_recall(ctx, gt)
        faith    = faithfulness_score(answer, ctx)
        relev    = answer_relevancy_score(q, answer)

        per_q.append({
            "question": q, "ground_truth": gt, "answer": answer,
            "rouge_l":           round(rl,       4),
            "context_precision": round(ctx_prec, 4),
            "context_recall":    round(ctx_rec,  4),
            "faithfulness":      round(faith,    4),
            "answer_relevancy":  round(relev,    4),
            "retrieved_sources": list({c["source"] for c in ctx}),
            "mrr_hybrid":        round(reciprocal_rank(hybrid_ids, rel_set), 4),
        })

        q_s = (q[:50] + "..") if len(q) > 52 else q
        print(f"  [{i+1:2d}] {q_s:<52}  "
              f"{faith:>5.3f}  {ctx_prec:>6.3f}  {ctx_rec:>5.3f}  {relev:>5.3f}  {rl:>5.3f}")

    return per_q, dense_all, sparse_all, hybrid_all


# ── Step 6: Aggregate ─────────────────────────────────────────────────────
# src used: evaluate_retrieval() from src/evaluation/metrics.py

def aggregate_metrics(qa_data, per_q, dense_all, sparse_all,
                      hybrid_all, relevant_sets, K) -> tuple:
    questions = [qa["question"] for qa in qa_data]
    strategies = {"Dense (TF-IDF)": dense_all,
                  "Sparse (BM25)":  sparse_all,
                  "Hybrid (RRF)":   hybrid_all}
    retrieval_results = {}
    for name, lists in strategies.items():
        retrieval_results[name] = evaluate_retrieval(
            queries=questions, retrieved_lists=lists,
            relevant_sets=relevant_sets, k=K,
        ).to_dict()

    gen_cols = ["rouge_l","context_precision","context_recall",
                "faithfulness","answer_relevancy"]
    generation_results = {}
    for col in gen_cols:
        vals = [r[col] for r in per_q]
        generation_results[col] = {
            "mean": round(float(np.mean(vals)),           4),
            "std":  round(float(np.std(vals)),            4),
            "min":  round(float(np.min(vals)),            4),
            "max":  round(float(np.max(vals)),            4),
            "p25":  round(float(np.percentile(vals, 25)), 4),
            "p75":  round(float(np.percentile(vals, 75)), 4),
        }
    return retrieval_results, generation_results


# ── Console helpers ───────────────────────────────────────────────────────

def _print_retrieval(results, K):
    cols = [f"MRR@{K}", f"NDCG@{K}", f"Recall@{K}", f"Precision@{K}", f"HitRate@{K}"]
    print(f"\n{'='*70}\n  RETRIEVAL METRICS  (K={K})\n{'='*70}")
    print(f"  {'Strategy':<22}" + "".join(f"  {c:>12}" for c in cols))
    print("  " + "-"*68)
    for name, m in results.items():
        print(f"  {name:<22}" + "".join(f"  {m.get(c,0):>12.4f}" for c in cols))

def _print_generation(results):
    print(f"\n{'='*70}\n  GENERATION METRICS\n{'='*70}")
    print(f"  {'Metric':<25}  {'Mean':>6}  {'Std':>6}  {'Min':>6}  {'Max':>6}")
    print("  " + "-"*56)
    for col, s in results.items():
        print(f"  {col:<25}  {s['mean']:>6.4f}  {s['std']:>6.4f}  {s['min']:>6.4f}  {s['max']:>6.4f}")

def _print_improvement(results, K):
    d = results.get("Dense (TF-IDF)", {})
    h = results.get("Hybrid (RRF)", {})
    print(f"\n{'='*70}\n  HYBRID vs DENSE\n{'='*70}")
    for m in [f"MRR@{K}", f"NDCG@{K}", f"HitRate@{K}"]:
        dv, hv = d.get(m, 0), h.get(m, 0)
        pct = (hv - dv) / max(dv, 1e-9) * 100
        print(f"  {m:<20}: {dv:.4f} → {hv:.4f}  ({pct:+.1f}%)")


# ── Markdown report ───────────────────────────────────────────────────────

def build_md_report(output: Dict, K: int) -> str:
    cs, rm, gm, pq = (output["corpus_stats"], output["retrieval_metrics"],
                       output["generation_metrics"], output["per_question"])
    lines = [
        "# DataChat — Evaluation Report", "",
        f"**Corpus:** {cs['num_documents']} documents · {cs['total_chars']:,} chars  ",
        f"**Chunks:** {cs['num_parent_chunks']} parents · {cs['num_child_chunks']} children  ",
        f"**QA pairs:** {len(pq)}  |  **K:** {K}  ",
        f"**Embedder:** TF-IDF local (swap to OpenAIEmbedder for production numbers)", "",
        "---", "", f"## Retrieval Metrics (K={K})", "",
        f"| Strategy | MRR@{K} | NDCG@{K} | Recall@{K} | Precision@{K} | HitRate@{K} |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in rm.items():
        b = "**" if "Hybrid" in name else ""
        lines.append(f"| {b}{name}{b} | {b}{m.get(f'MRR@{K}',0)}{b} | "
                     f"{m.get(f'NDCG@{K}',0)} | {m.get(f'Recall@{K}',0)} | "
                     f"{m.get(f'Precision@{K}',0)} | {m.get(f'HitRate@{K}',0)} |")
    d_mrr = rm.get("Dense (TF-IDF)",{}).get(f"MRR@{K}", 0)
    h_mrr = rm.get("Hybrid (RRF)",  {}).get(f"MRR@{K}", 0)
    pct   = (h_mrr - d_mrr) / max(d_mrr, 1e-9) * 100
    lines += ["", f"> Hybrid vs Dense: **{pct:+.1f}% MRR@{K}**", "", "---", "",
              "## Generation Metrics (Hybrid RRF, top-5 parent chunks)", "",
              "| Metric | Mean | Std | Min | Max | P25 | P75 |",
              "|---|---|---|---|---|---|---|"]
    for col, s in gm.items():
        lines.append(f"| **{col}** | **{s['mean']}** | {s['std']} | "
                     f"{s['min']} | {s['max']} | {s['p25']} | {s['p75']} |")
    lines += ["", "---", "", "## Per-Question Results", "",
              "| # | Question | Faith | Ctx Prec | Ctx Rec | Relevancy | ROUGE-L |",
              "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(pq, 1):
        q = (r["question"][:52]+"…") if len(r["question"]) > 55 else r["question"]
        lines.append(f"| {i} | {q} | {r['faithfulness']:.3f} | "
                     f"{r['context_precision']:.3f} | {r['context_recall']:.3f} | "
                     f"{r['answer_relevancy']:.3f} | {r['rouge_l']:.3f} |")
    lines += ["", "---", "", "## Metric → src/ mapping", "",
              "| Metric | Function | File |", "|---|---|---|",
              f"| MRR/NDCG/Recall/Precision/HitRate | `evaluate_retrieval()` | `src/evaluation/metrics.py` |",
              "| Context Precision | `context_precision()` | `src/evaluation/metrics.py` |",
              "| Context Recall | `context_recall()` | `src/evaluation/metrics.py` |",
              "| Faithfulness | `faithfulness_score()` | `src/evaluation/metrics.py` |",
              "| Answer Relevancy | `answer_relevancy_score()` | `src/evaluation/metrics.py` |",
              "| ROUGE-L | `rouge_l_score()` | `src/evaluation/metrics.py` |",
              "| Dense retrieval | `FAISSVectorStore.similarity_search()` | `src/retrieval/vector_store/faiss_vs.py` |",
              "| Sparse retrieval | `BM25Retriever.retrieve()` | `src/retrieval/bm25_retriever.py` |",
              "| RRF + parent swap | `HybridRetriever.retrieve()` | `src/retrieval/hybrid_retriever.py` |",
              "| Chunking | `HierarchicalChunker.chunk_document()` | `src/ingestion/chunkers/hierarchical_chunker.py` |",
              "| Embedding | `TFIDFEmbedder.embed_query()` | `src/ingestion/embedders/tfidf_embedder.py` |",
              "| Full ingest | `IngestionPipeline.process_document()` | `src/ingestion/pipeline.py` |"]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def run_evaluation(raw_dir=DEFAULT_RAW_DIR, eval_file=DEFAULT_EVAL_FILE,
                   out_json=DEFAULT_OUT_JSON, out_md=DEFAULT_OUT_MD, K=10) -> Dict:
    t0 = time.time()
    print("\n" + "="*70)
    print("  DataChat RAG Evaluation  (all logic from src/)")
    print("="*70)

    print("\n[1/5] Loading documents...")
    raw_docs = load_raw_documents(raw_dir)

    print("\n[2/5] Running IngestionPipeline...")
    embedder, vector_store, bm25, retriever, children, parents = build_indices(raw_docs)

    print("\n[3/5] QA pairs + relevant sets...")
    if not eval_file.exists():
        raise FileNotFoundError(f"QA file not found: {eval_file}")
    with open(eval_file) as f:
        qa_data = json.load(f)
    relevant_sets = build_relevant_sets(qa_data, bm25, top_n=4)
    print(f"  {len(qa_data)} QA pairs loaded")

    print(f"\n[4/5] Evaluating {len(qa_data)} questions  (K={K})...\n")
    per_q, dense_all, sparse_all, hybrid_all = evaluate_questions(
        qa_data, relevant_sets, embedder, vector_store, bm25, retriever, K)

    print("\n[5/5] Aggregating...")
    retrieval_results, generation_results = aggregate_metrics(
        qa_data, per_q, dense_all, sparse_all, hybrid_all, relevant_sets, K)

    _print_retrieval(retrieval_results, K)
    _print_generation(generation_results)
    _print_improvement(retrieval_results, K)

    output = {
        "corpus_stats": {
            "num_documents":    len(raw_docs),
            "num_parent_chunks":len(parents),
            "num_child_chunks": len(children),
            "total_chars":      sum(len(d.raw_bytes) for d in raw_docs),
            "documents":        [Path(d.source_path).name for d in raw_docs],
        },
        "evaluation_config": {"K": K, "rrf_k": 60, "dense_weight": 0.6},
        "retrieval_metrics":  retrieval_results,
        "generation_metrics": generation_results,
        "per_question":       per_q,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_md_report(output, K))

    print(f"\n  Saved JSON → {out_json}")
    print(f"  Saved MD   → {out_md}")
    print(f"  Time       → {time.time()-t0:.1f}s\n")
    return output


def parse_args():
    p = argparse.ArgumentParser(description="DataChat RAG Evaluation")
    p.add_argument("--raw-dir",   type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    p.add_argument("--out-json",  type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md",    type=Path, default=DEFAULT_OUT_MD)
    p.add_argument("--k",         type=int,  default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(raw_dir=args.raw_dir, eval_file=args.eval_file,
                   out_json=args.out_json, out_md=args.out_md, K=args.k)
