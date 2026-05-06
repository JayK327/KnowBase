# ML Feature Store Documentation

## Overview

The DataChat Feature Store centralises reusable ML features so that data scientists
do not recompute the same signals independently in every model. Features are computed
once, stored in the feature store, and served to both online (real-time inference)
and offline (training) use cases from a single source of truth.

The feature store is built on Feast (Feature Store for Machine Learning) backed by
Redis for online serving and Delta Lake for offline training data.

---

## Architecture

### Offline Store
- **Backend:** Delta Lake on S3 (`s3://datachat-features/offline/`)
- **Use case:** Training dataset generation, batch inference, backfills
- **Access:** Spark SQL or Feast SDK `get_historical_features()`
- **Latency:** Minutes (batch retrieval)

### Online Store
- **Backend:** Redis Cluster (3-node, 16GB per node)
- **Use case:** Real-time model inference (<10ms latency requirement)
- **Access:** Feast SDK `get_online_features()` or REST API
- **Latency:** <5ms p99

### Feature Computation
Features are computed by scheduled Spark jobs (daily or hourly) and materialized
into both stores. The computation jobs run after the `daily_aggregator` pipeline
completes.

---

## Feature Groups

### user_engagement_features

**Description:** User behavioral signals derived from the last 30 days of activity.
**Computation schedule:** Daily at 08:30 UTC
**Entity:** user_id
**TTL (online store):** 48 hours

| Feature Name | Type | Description |
|---|---|---|
| days_since_last_session | INTEGER | Days since user's last recorded session |
| sessions_last_7d | INTEGER | Session count in the past 7 days |
| sessions_last_30d | INTEGER | Session count in the past 30 days |
| avg_session_duration_7d | FLOAT | Average session duration (seconds) over 7 days |
| event_count_last_7d | INTEGER | Total event count in the past 7 days |
| page_views_last_7d | INTEGER | Page view count in the past 7 days |
| feature_adoption_score | FLOAT | Fraction of key features used (0.0–1.0) |
| primary_platform | VARCHAR | Most-used platform: web, ios, android |

### user_purchase_features

**Description:** Purchase history and revenue signals per user.
**Computation schedule:** Daily at 08:45 UTC
**Entity:** user_id
**TTL (online store):** 24 hours

| Feature Name | Type | Description |
|---|---|---|
| lifetime_revenue_usd | FLOAT | Total revenue from confirmed orders |
| order_count_lifetime | INTEGER | Total confirmed order count |
| days_since_first_order | INTEGER | Days since first purchase |
| days_since_last_order | INTEGER | Days since most recent purchase |
| avg_order_value_usd | FLOAT | Mean order value across all orders |
| has_refunded | BOOLEAN | True if user has ever received a refund |
| refund_count | INTEGER | Total number of refunds |
| current_plan | VARCHAR | Current subscription plan type |
| is_paying | BOOLEAN | True if user has an active paid subscription |

### user_risk_features

**Description:** Signals used by the fraud and churn risk models.
**Computation schedule:** Hourly
**Entity:** user_id
**TTL (online store):** 2 hours

| Feature Name | Type | Description |
|---|---|---|
| login_failure_count_24h | INTEGER | Failed login attempts in the last 24 hours |
| ip_country_mismatch | BOOLEAN | Profile country differs from session IP country |
| velocity_orders_1h | INTEGER | Orders placed in the last 1 hour |
| velocity_orders_24h | INTEGER | Orders placed in the last 24 hours |
| churn_probability_score | FLOAT | Output of churn model v3.2 (0.0–1.0) |
| fraud_risk_score | FLOAT | Output of fraud model v1.8 (0.0–1.0) |

---

## Accessing Features

### Offline retrieval (training datasets)

```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_repo/")

entity_df = pd.DataFrame({
    "user_id":   ["user-001", "user-002"],
    "event_timestamp": ["2024-06-01", "2024-06-01"],
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_engagement_features:sessions_last_30d",
        "user_engagement_features:feature_adoption_score",
        "user_purchase_features:lifetime_revenue_usd",
        "user_purchase_features:is_paying",
    ],
).to_df()
```

### Online retrieval (real-time inference)

```python
feature_vector = store.get_online_features(
    features=[
        "user_engagement_features:sessions_last_7d",
        "user_risk_features:churn_probability_score",
    ],
    entity_rows=[{"user_id": "user-001"}],
).to_dict()
```

---

## Models Using This Feature Store

### Churn Prediction Model (v3.2)
- **Features used:** All of `user_engagement_features` + `user_purchase_features`
- **Framework:** XGBoost, 847 estimators
- **Training cadence:** Weekly retrain on 90-day rolling window
- **Serving latency:** <20ms (online store + model inference)
- **Output:** `churn_probability_score` (also written back to `user_risk_features`)
- **Threshold:** 0.65 triggers automated email campaign

### Fraud Detection Model (v1.8)
- **Features used:** `user_risk_features` (all), `user_purchase_features.velocity_*`
- **Framework:** LightGBM
- **Training cadence:** Daily retrain
- **Output:** `fraud_risk_score` — written to `user_risk_features` and published
  to Kafka topic `fraud_scores` for real-time order blocking
- **Threshold:** 0.85 triggers automatic order hold

### Personalised Recommendation Model
- **Features used:** `user_engagement_features`, product interaction signals
- **Framework:** Two-tower neural network (PyTorch)
- **Inference:** Batch nightly, top-10 recommendations per user stored in
  `gold.user_recommendations`

---

## Feature Governance

### Adding a new feature
1. File a ticket in the data platform Jira board with the feature definition
2. A data engineer reviews the computation SQL for correctness and performance
3. Feature is added to the Feast feature store registry
4. Computation job is scheduled and backfilled for 90 days
5. Feature is available in both online and offline stores

### Deprecating a feature
1. Check downstream model usage: `grep -r "feature_name" model_configs/`
2. Notify owning model teams with 30-day notice
3. Mark feature as `deprecated: true` in feature registry
4. Remove from computation jobs after all downstream usage is removed

### Feature drift monitoring
The platform monitors feature distributions daily using Evidently AI.
Alerts fire when:
- Mean of a numeric feature shifts by more than 2 standard deviations
- Null rate of any feature increases by more than 5 percentage points
- A feature's online store fill rate drops below 90%

---

## SLAs

| Store | Freshness SLA | Latency SLA |
|---|---|---|
| Online (Redis) | Hourly features: <90 min lag; Daily features: <3 hour lag | p99 <5ms |
| Offline (Delta Lake) | Daily features: available by 10:00 UTC | N/A (batch) |
