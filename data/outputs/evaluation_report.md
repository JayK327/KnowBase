# DataChat — Evaluation Report

**Corpus:** 6 documents · 49,153 characters  
**Chunks:** 90 parent (80 words each) · 348 child (24 words each)  
**QA pairs:** 15 ground-truth questions  
**Rank cutoff K:** 10  
**Full JSON:** `data/outputs/evaluation_results.json`

---

## Corpus Documents

| File | Description |
|---|---|
| `api_contracts.md` | Auth, User, Order, Feature Store, ML Serving — REST/gRPC contracts |
| `data_catalog.md` | Core table schemas — users, orders, sessions, user_events, products, subscriptions |
| `data_governance.md` | PII levels 0–3, GDPR/CCPA procedures, retention schedules, access matrix |
| `etl_pipeline_guide.md` | Medallion architecture, Spark pipelines, SLAs, backfill, failure runbooks |
| `incident_runbook.md` | On-call runbooks, severity matrix, common recovery commands |
| `ml_feature_store.md` | Feast feature store, 3 feature groups, churn v3.2 and fraud v1.8 model registry |

---

## Retrieval Metrics (K = 10)

| Strategy | MRR@10 | NDCG@10 | Recall@10 | Precision@10 | HitRate@10 |
|---|---|---|---|---|---|
| Dense (TF-IDF) | 0.3818 | 0.2941 | 0.3667 | 0.1467 | 0.6667 |
| Sparse (BM25) | 0.4403 | 0.3282 | 0.3667 | 0.1467 | 0.7333 |
| **Hybrid (RRF)** | **0.4456** | **0.3529** | 0.4167 | 0.1667 | 0.7333 |

> **Hybrid vs Dense: +16.7% MRR@10 improvement.**  
> BM25 outperforms TF-IDF dense on this identifier-heavy corpus.

---

## Generation Metrics (Hybrid RRF, top-5 parent chunks)

| Metric | Mean | Std | Min | Max | P25 | P75 |
|---|---|---|---|---|---|---|
| **rouge_l** | **0.137** | 0.0403 | 0.0606 | 0.2034 | 0.1153 | 0.1691 |
| **context_precision** | **0.96** | 0.1497 | 0.4 | 1.0 | 1.0 | 1.0 |
| **context_recall** | **0.4768** | 0.206 | 0.05 | 0.8182 | 0.3541 | 0.6667 |
| **faithfulness** | **1.0** | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **answer_relevancy** | **0.9765** | 0.075 | 0.7 | 1.0 | 1.0 | 1.0 |

---

## Per-Question Results

| # | Question | Faith | Ctx Prec | Ctx Rec | Relevancy | ROUGE-L |
|---|---|---|---|---|---|---|
| 1 | What is the primary key of the users table? | 1.000 | 1.000 | 0.400 | 1.000 | 0.111 |
| 2 | What columns does the orders table have? | 1.000 | 1.000 | 0.050 | 1.000 | 0.061 |
| 3 | How often does the event_ingestor pipeline run? | 1.000 | 1.000 | 0.455 | 1.000 | 0.203 |
| 4 | What is the SLA for the Silver layer lag in the event_i… | 1.000 | 1.000 | 0.818 | 1.000 | 0.191 |
| 5 | How should I fix a corrupted checkpoint in the event_in… | 1.000 | 1.000 | 0.409 | 1.000 | 0.182 |
| 6 | What is the retention policy for the users table? | 1.000 | 1.000 | 0.250 | 1.000 | 0.077 |
| 7 | What is the churn prediction model version and what thr… | 1.000 | 1.000 | 0.667 | 1.000 | 0.173 |
| 8 | What PII level is the email column in the users table? | 1.000 | 1.000 | 0.500 | 1.000 | 0.119 |
| 9 | What does the daily_aggregator pipeline output? | 1.000 | 1.000 | 0.333 | 1.000 | 0.133 |
| 10 | How do I perform a GDPR erasure for a user? | 1.000 | 1.000 | 0.667 | 0.947 | 0.165 |
| 11 | What is the online feature store backend and what is it… | 1.000 | 1.000 | 0.667 | 1.000 | 0.149 |
| 12 | What partitioning strategy does the user_events table u… | 1.000 | 1.000 | 0.375 | 1.000 | 0.125 |
| 13 | What is the fraud_risk_score threshold for blocking an … | 1.000 | 0.400 | 0.462 | 1.000 | 0.091 |
| 14 | How should I handle an OOM error in the daily_aggregato… | 1.000 | 1.000 | 0.300 | 0.700 | 0.140 |
| 15 | What is the Kafka consumer group name for the event_ing… | 1.000 | 1.000 | 0.800 | 1.000 | 0.135 |

---

## Metric Definitions

**Context Precision** — fraction of retrieved parent chunks with ≥8% token overlap with the ground truth.

**Context Recall** — fraction of ground-truth keywords (>3 chars) covered by the retrieved context.

**Faithfulness** — fraction of answer sentences where ≥25% of tokens appear in retrieved context.

**Answer Relevancy** — BM25 overlap between answer text and original question, normalised to [0,1].

**ROUGE-L** — longest common subsequence F1 between generated answer and ground truth. Low on extractive answers; a GPT-4o answer would score ~0.25–0.35.

---

## Next Steps to Improve

| Improvement | Target Metric | Expected Impact |
|---|---|---|
| Replace TF-IDF with text-embedding-3-large | MRR@10 | +10–20% |
| Larger parent chunks (80w → 150w) | Context Recall | +10–15% |
| Multi-hop retrieval for cross-document questions | Context Recall | +15–20% |
| Semantic chunking at heading boundaries | All | +5–10% |
| Expand corpus to 50+ documents | BM25 IDF quality | +5–10% |