# ETL Pipeline Architecture and Runbooks

## Overview

DataChat's data platform runs on a medallion architecture (Bronze → Silver → Gold)
implemented in Apache Spark on a managed cluster environment. All pipelines are
orchestrated by a workflow scheduler that runs jobs on a defined cron schedule.
This document covers pipeline architecture, job descriptions, SLAs, and on-call
runbooks for common failure scenarios.

---

## Architecture: Medallion Layers

### Bronze Layer (Raw Ingestion)
- **Location:** `s3://datachat-data-lake/bronze/`
- **Format:** JSON and Avro, schema-on-read
- **Purpose:** Exact copy of source data with no transformations
- **Retention:** 90 days hot, then archived to Glacier
- **Key tables:** `bronze.raw_events`, `bronze.raw_orders`, `bronze.raw_users`

### Silver Layer (Cleaned + Structured)
- **Location:** `s3://datachat-data-lake/silver/`
- **Format:** Delta Lake (Parquet + transaction log)
- **Purpose:** Deduplicated, type-cast, schema-enforced data
- **Retention:** 3 years
- **Key tables:** `silver.events`, `silver.orders`, `silver.users`

### Gold Layer (Analytics-Ready)
- **Location:** `s3://datachat-data-lake/gold/`
- **Format:** Delta Lake, Z-ordered for common query patterns
- **Purpose:** Aggregated, business-logic-applied tables and marts
- **Retention:** 5 years
- **Key tables:** `gold.daily_active_users`, `gold.revenue_summary`, `gold.funnel_metrics`

---

## Pipeline: event_ingestor

**Description:** Reads raw event JSON from Kafka topic `user_events`, validates
against event schema registry, deduplicates on `event_id`, and writes to
`bronze.raw_events` and `silver.events`.

**Schedule:** Continuous Spark Structured Streaming (micro-batch every 2 minutes)
**SLA:** Silver layer lag must not exceed 5 minutes from Kafka produce time
**Owner:** Platform Data Engineering (pde-team@datachat.io)
**Checkpoint path:** `s3://datachat-checkpoints/event_ingestor/`

### Configuration
- Kafka brokers: `kafka-01.internal:9092,kafka-02.internal:9092`
- Consumer group: `event_ingestor_prod`
- Starting offset: latest (new deployments use earliest for backfill)
- Batch interval: 120 seconds
- Max offsets per trigger: 50,000

### Deduplication Logic
Events are deduplicated using a watermark of 24 hours on `server_timestamp`.
Within the watermark window, duplicates are dropped by keeping the first occurrence
of each `event_id`. Events outside the watermark are assumed unique and passed through.

### Schema Validation
Events failing schema validation are written to `bronze.rejected_events` with
a `rejection_reason` column. These are not retried automatically. Rejection rate
above 2% triggers a PagerDuty alert.

### Failure Runbook
**Symptom:** Silver layer lag exceeds 5 minutes (monitored via Datadog metric
`event_ingestor.lag_seconds`)

1. Check Kafka consumer lag: `kafka-consumer-groups.sh --describe --group event_ingestor_prod`
2. If lag is growing, check Spark executor logs for OOM errors
3. If OOM: increase `spark.executor.memory` from 4g to 8g and restart the job
4. If checkpoint is corrupted (common after ungraceful shutdown): delete checkpoint
   directory and restart with `--reset-checkpoint` flag. This causes reprocessing
   from the last committed Kafka offset.
5. Escalate to Platform Data Engineering if lag exceeds 30 minutes

---

## Pipeline: order_processor

**Description:** Micro-batch pipeline that reads confirmed order events from the
`order_events` Kafka topic and upserts into `silver.orders`. Handles both new
orders and status updates (e.g. pending → confirmed → shipped → delivered).

**Schedule:** Every 5 minutes (cron: `*/5 * * * *`)
**SLA:** Orders must appear in silver layer within 10 minutes of being placed
**Owner:** Commerce Data Engineering
**Dependency:** Requires `silver.users` to be available for user validation

### Upsert Logic
Uses Delta Lake MERGE operation keyed on `order_id`. On match (existing order):
updates `status`, `updated_at`, `refunded_at`, and `metadata`. On no match:
inserts the full row.

### Failure Runbook
**Symptom:** Orders not appearing in silver layer; monitoring alert `order_processor.delay`

1. Check job logs for JDBC connection errors to source database
2. Verify `silver.users` has been updated in the last 15 minutes
3. If JDBC timeout: increase `spark.sql.broadcastTimeout` from 300 to 600 seconds
4. If merge conflicts: check for concurrent writes from the admin dashboard
5. Manual backfill: `spark-submit order_processor.py --start-date=YYYY-MM-DD --end-date=YYYY-MM-DD`

---

## Pipeline: user_sync

**Description:** Full refresh of `silver.users` from the user-service PostgreSQL
database. Runs hourly. Handles soft deletes (is_active = false) and plan changes.

**Schedule:** Every hour at minute 0 (cron: `0 * * * *`)
**SLA:** Must complete within 25 minutes
**Typical duration:** 8-12 minutes
**Row count written:** ~48 million rows per run

### Implementation
Uses a full OVERWRITE strategy on non-partitioned columns. Partitions by
`created_at` month. Only partitions with changes in the last 7 days are
rewritten (Z-order applied per partition).

### Failure Runbook
**Symptom:** `silver.users` has stale data (updated_at > 2 hours ago)

1. Check PostgreSQL replica lag: if replica lag > 30 seconds, pipeline reads
   stale data. Check `pg_stat_replication` on primary.
2. If pipeline timed out: increase `spark.sql.execution.arrow.maxRecordsPerBatch`
   from 10000 to 50000 for faster JDBC reads.
3. Emergency: trigger manual run from the scheduler UI, job name `user_sync_manual`

---

## Pipeline: daily_aggregator

**Description:** Builds all Gold layer daily aggregation tables. Runs once per day.
This is the most resource-intensive pipeline. Output tables feed all BI dashboards.

**Schedule:** Daily at 06:00 UTC (cron: `0 6 * * *`)
**SLA:** Must complete by 08:00 UTC (2-hour window)
**Typical duration:** 45-70 minutes
**Owner:** Analytics Engineering
**Dependency:** Requires `silver.events`, `silver.orders`, `silver.users` to be current

### Output Tables

**gold.daily_active_users**
- Definition: Users with at least one `session.start` event on the date
- Grain: one row per user per date
- Columns: `user_id`, `date`, `session_count`, `event_count`, `platform`

**gold.revenue_summary**
- Definition: Aggregated daily revenue by product and country
- Grain: one row per product_id, country_code, date
- Columns: `date`, `product_id`, `country_code`, `order_count`, `gross_revenue_usd`,
  `net_revenue_usd` (after refunds), `refund_count`

**gold.funnel_metrics**
- Definition: Daily conversion funnel from session start to purchase
- Grain: one row per date, platform, acquisition_channel
- Columns: `date`, `platform`, `acquisition_channel`, `sessions`, `signups`,
  `trial_starts`, `conversions`, `conversion_rate`

### Failure Runbook
**Symptom:** BI dashboards show "No data" for today's date

1. Check pipeline status — common failure: a Silver dependency is not ready
2. Run dependency check: `python check_dependencies.py --pipeline daily_aggregator`
3. If Silver tables are ready but pipeline failed: check for partition discovery
   errors. Run `MSCK REPAIR TABLE silver.events` and retry.
4. If OOM on Gold aggregation: the `gold.funnel_metrics` job is the heaviest.
   Increase driver memory: `--driver-memory 16g`
5. Manual partial run: `spark-submit daily_aggregator.py --tables revenue_summary
   --date 2024-06-15`

---

## Data Quality Checks

All Silver and Gold pipelines run Great Expectations checks after each write.
Failed checks write to `monitoring.dq_results` and fire PagerDuty alerts.

### Critical checks (pipeline fails on breach)
- `users.email` unique constraint
- `orders.amount_usd` > 0 for non-refunded orders
- `events.event_timestamp` within 48 hours of `server_timestamp`
- `silver.events` row count within 20% of previous day's count

### Warning checks (alert only, pipeline continues)
- `orders.country_code` null rate < 15%
- `events.session_id` null rate < 5%
- `users.acquisition_channel` null rate < 10%

---

## Backfill Procedures

### Standard backfill (< 7 days)
```bash
spark-submit pipelines/backfill.py \
  --pipeline event_ingestor \
  --start-date 2024-06-01 \
  --end-date 2024-06-07 \
  --parallelism 4
```

### Historical backfill (> 7 days)
For large backfills, use the managed batch job to avoid overwhelming the cluster.
Submit via the scheduler UI with job type `backfill_large`. Set `max_executors=20`
and `executor_memory=8g`. Estimated throughput: 1 billion events per 2 hours.

### Checkpoint reset
If a streaming job needs to reprocess from a specific Kafka offset:
1. Stop the streaming job
2. Delete checkpoint: `aws s3 rm s3://datachat-checkpoints/<job_name>/ --recursive`
3. Set `kafka.startingOffsets` to the target timestamp offset
4. Restart the job

---

## Monitoring and Alerting

All pipelines emit metrics to Datadog. Key dashboards:
- **Pipeline Health:** https://app.datadoghq.com/dashboard/pipeline-health
- **Data Freshness:** https://app.datadoghq.com/dashboard/data-freshness
- **Error Rates:** https://app.datadoghq.com/dashboard/pipeline-errors

### On-call escalation path
1. First response: on-call data engineer (PagerDuty rotation: `data-platform-oncall`)
2. If unresolved in 30 minutes: escalate to data engineering lead
3. If data unavailable for BI dashboards: notify analytics@datachat.io

---

## Glossary

- **Micro-batch:** Spark Structured Streaming with a trigger interval (not continuous)
- **Z-ordering:** Delta Lake data layout optimization for multi-dimensional range queries
- **Watermark:** Spark Streaming threshold for late data handling
- **MERGE:** Delta Lake upsert operation combining INSERT and UPDATE
- **Medallion:** Bronze/Silver/Gold layered data architecture pattern
