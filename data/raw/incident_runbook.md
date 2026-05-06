# On-Call Incident Runbook — Data Platform

## Overview

This runbook covers the most common data platform incidents encountered during
on-call rotations. The on-call engineer is responsible for acknowledging PagerDuty
alerts within 15 minutes and resolving P1 incidents within 2 hours.

On-call schedule: https://datachat.pagerduty.com/schedules/data-platform

**Escalation:** Slack channel #data-platform-incidents for real-time coordination.
Engineering Lead on-call: check PagerDuty escalation policy.

---

## Incident Severity Levels

| Level | Description | Response Time | Example |
|---|---|---|---|
| P1 | Data completely unavailable; BI dashboards broken | 15 min ack, 2 hr resolve | daily_aggregator failed, no data for today |
| P2 | Data delayed > 2x SLA | 30 min ack, 4 hr resolve | Silver layer lag > 10 minutes |
| P3 | Data quality issue; data present but partially incorrect | 2 hr ack, next business day resolve | Null rate spike in one column |
| P4 | Informational; no immediate user impact | Next business day | Slow pipeline job, no SLA breach |

---

## Alert: Pipeline Failed — daily_aggregator

**Alert name:** `daily_aggregator.job_failed`
**Typical time:** Fires between 06:00–08:00 UTC

### Immediate triage (5 minutes)
1. Open the scheduler UI and check the failed task
2. Check which Gold tables are missing: `SELECT table_name, MAX(date) FROM gold.metadata GROUP BY 1`
3. Identify the root cause task — the aggregator runs tasks in order; the first failure
   causes all downstream tasks to be skipped

### Common causes and fixes

**Cause A: Silver dependency not ready**

Check: `SELECT MAX(event_date) FROM silver.events` — should be yesterday's date.
If not ready, the aggregator started before Silver was complete.

Fix: Wait for Silver to finish (check `event_ingestor` job status), then trigger
a manual backfill for yesterday:
`python daily_aggregator.py --date $(date -d "yesterday" +%Y-%m-%d) --force`

**Cause B: Schema evolution in source table**

Check: Look for `AnalysisException: cannot resolve column` in the job logs.

Fix: The Silver table gained a new column that the Gold job doesn't handle.
Add the column to the Gold job config and redeploy. In the meantime, trigger
a run with `--skip-column-validation` flag.

**Cause C: OOM on gold.funnel_metrics**

Check: Look for `java.lang.OutOfMemoryError: GC overhead limit exceeded` in executor logs.

Fix: The funnel query does a large cross join. Increase driver memory:
`spark-submit --driver-memory 20g daily_aggregator.py`

**Cause D: Partition not found (holiday traffic spike)**

Check: Look for `FileNotFoundException` referencing a specific date partition.

Fix: The partition exists in the table but the metastore doesn't know about it.
Run `MSCK REPAIR TABLE silver.events` and retry.

---

## Alert: Event Ingestor Lag

**Alert name:** `event_ingestor.lag_seconds > 300`
**What it means:** Silver.events is more than 5 minutes behind Kafka

### Triage
1. Check Kafka consumer lag: run `kafka_lag_check.sh` from the ops-scripts repo
2. Check Spark Streaming dashboard for batch duration — if batches are taking
   longer than the trigger interval, they are queuing up

### Common causes and fixes

**Cause A: Kafka partition count increased**

If the ops team increased Kafka partitions, Spark Streaming must restart to
pick up the new partitions. Simply restart the streaming job.

**Cause B: Executor OOM on large JSON payloads**

New event types occasionally have very large `properties` payloads. This causes
executor OOM when many large events arrive at once.

Fix: Increase `spark.executor.memory` from 4g to 8g. In the streaming job config,
set `spark.streaming.kafka.maxRatePerPartition=5000` to throttle ingestion rate.

**Cause C: Schema registry unreachable**

The event_ingestor calls the event registry API for every event type it encounters.
If the registry is down, it falls back to schema-less mode and lag may increase
as retries pile up.

Fix: Check `http://event-registry.internal:8085/health`. If down, page the
Platform team. Temporarily set `SCHEMA_VALIDATION=false` env var and restart
the job to continue ingestion without validation.

**Cause D: Checkpoint corruption**

After an ungraceful pod restart, the Spark checkpoint directory can become
corrupted. Symptoms: job fails immediately on startup with
`StreamingQueryException: Checkpoint metadata mismatch`.

Fix:
```bash
aws s3 rm s3://datachat-checkpoints/event_ingestor/ --recursive
# Restart job — it will re-read from last committed Kafka offset
```

Note: This causes a brief data gap. Any events between the last Kafka commit
and the restart will be re-ingested, creating duplicates. Run the deduplication
job afterwards: `python ops/deduplicate_events.py --hours-back 2`

---

## Alert: Feature Store Fill Rate Low

**Alert name:** `feature_store.online_fill_rate < 0.9`
**What it means:** More than 10% of users don't have feature values in Redis

### Triage
1. Check which feature group has low fill rate:
   `curl http://feature-store.internal:8083/v1/monitoring/fill_rates`
2. Check the materialisation job for that feature group in the scheduler UI

### Common causes

**Cause A: Materialisation job failed**

The daily materialisation job reads from Delta Lake and writes to Redis.
If it failed, Redis values are stale/missing.

Fix: Manually trigger the materialisation job for the affected feature group:
`python feature_store/materialise.py --feature-group user_engagement_features --full`

**Cause B: Redis memory exhaustion**

Check Redis memory usage: `redis-cli INFO memory | grep used_memory_human`

If memory > 14GB (of 16GB), Redis is evicting keys due to `allkeys-lru` policy.

Fix: Reduce TTL for low-priority features in the feature registry config,
or provision a new Redis node (contact infrastructure team).

---

## Alert: Data Quality Check Failed

**Alert name:** `great_expectations.critical_check_failed`
**What it means:** A data quality constraint was violated in Silver or Gold

### Common checks that fail

**`orders.amount_usd > 0` violated**

This means one or more refund rows have the wrong sign convention. Find the rows:
`SELECT * FROM silver.orders WHERE amount_usd <= 0 AND status != 'refunded'`

Fix: Update the affected rows in the source and trigger a reprocess of that
date's partition.

**`events row count within 20% of previous day` violated**

Could mean genuine traffic drop or data loss. Check:
1. Is it a holiday or weekend? Traffic naturally drops — widen the threshold
2. Check Kafka `user_events` topic consumer lag — are events being produced?
3. Check `bronze.rejected_events` — are events failing schema validation in bulk?

**`users.email unique constraint` violated**

This should never happen. If it does, there is a bug in the user_sync pipeline.
Immediately alert the Platform Data Engineering lead and halt the user_sync job
until investigated.

---

## Runbook: GDPR Erasure Emergency

If a user reports that their data was not erased within the 30-day legal window:

1. Immediately run the erasure script: `python gdpr/erasure.py --user-id <uuid> --force`
2. Verify completion: `python gdpr/verify_erasure.py --user-id <uuid>`
3. Log the incident in the `audit.gdpr_erasures` table with `delayed: true`
4. Notify the Legal team within 24 hours via legal@datachat.io

Failure to complete GDPR erasures within 30 days can result in regulatory fines
up to 4% of annual revenue under GDPR Article 83.

---

## Common Commands Reference

```bash
# Check Silver freshness
python ops/check_freshness.py --tables silver.events,silver.orders,silver.users

# Manual backfill a single date
python daily_aggregator.py --date 2024-06-15 --tables revenue_summary --force

# Check Kafka consumer lag
bash ops/kafka_lag_check.sh event_ingestor_prod

# Reset streaming checkpoint
aws s3 rm s3://datachat-checkpoints/<job_name>/ --recursive

# Run deduplication after checkpoint reset
python ops/deduplicate_events.py --hours-back 4

# Trigger feature materialisation
python feature_store/materialise.py --feature-group user_engagement_features

# GDPR erasure
python gdpr/erasure.py --user-id <uuid>
python gdpr/verify_erasure.py --user-id <uuid>

# Repair Hive metastore partition metadata
spark-sql -e "MSCK REPAIR TABLE silver.events"
```
