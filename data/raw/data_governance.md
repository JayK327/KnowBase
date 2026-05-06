# Data Governance, PII Classification, and Retention Policies

## Overview

This document defines DataChat's data governance policies including PII
classification levels, data retention schedules, access control requirements,
and GDPR/CCPA compliance procedures. All data engineers are required to
classify new tables before they are promoted to the Silver or Gold layer.

---

## PII Classification Levels

### Level 0 — Non-PII (Public)
Data with no personal identifiers. Can be freely shared.

**Examples:** Aggregate revenue by product, daily active user counts,
funnel conversion rates, anonymised cohort metrics.

**Access:** No restrictions. Available to all internal teams.

### Level 1 — Pseudonymous
Data linked to an internal ID (e.g. `user_id`, `session_id`, `anonymous_id`)
but not directly to a real-world identity. Can be re-identified if combined
with Level 2 data.

**Examples:** `user_events` table (user_id + behaviour), `sessions` table,
`user_engagement_features`, `user_purchase_features`.

**Access:** Data engineering and analytics teams. No external sharing without
anonymisation. Must be encrypted at rest.

### Level 2 — Directly Identifiable PII
Data containing information that directly identifies a person.

**Examples:** `users.email`, `users.username`, payment card last four digits,
billing addresses, phone numbers.

**Access:** Restricted to data engineering team leads, security team, and
authorised product engineers via role-based access control. Audit log required
for all accesses. Must be tokenised or masked in downstream tables.

### Level 3 — Sensitive PII
Highest sensitivity. Regulatory requirements apply.

**Examples:** Full payment card numbers (not stored — tokenised at point of
capture by Stripe), government ID numbers, health information.

**Policy:** Level 3 data must never be written to the data warehouse. If
discovered, immediate escalation to the security team is required.

---

## Column-Level PII Inventory

| Table | Column | PII Level | Handling |
|---|---|---|---|
| users | user_id | Level 1 | UUID pseudonym |
| users | email | Level 2 | Tokenised in downstream tables |
| users | username | Level 2 | Masked in BI exports |
| orders | user_id | Level 1 | UUID pseudonym |
| orders | amount_usd | Level 0 | No restriction |
| sessions | user_id | Level 1 | UUID pseudonym |
| sessions | anonymous_id | Level 1 | Cookie-based pseudonym |
| user_events | properties.ip_address | Level 2 | Dropped from Silver layer |
| user_events | properties.user_agent | Level 1 | Kept, no direct ID |

---

## Data Retention Schedules

| Table / Layer | Hot Retention | Cold Retention | Deletion Method |
|---|---|---|---|
| `bronze.raw_events` | 90 days | Archive to Glacier (3 years) | Partition drop |
| `silver.events` | 3 years | None (delete) | Partition drop |
| `silver.orders` | 10 years | None | Regulatory hold |
| `silver.users` | 7 years | None | Regulatory hold |
| `gold.*` | 5 years | None | Partition drop |
| `monitoring.*` | 90 days | None | TTL |
| Feature store (offline) | 2 years | None | Partition drop |
| Feature store (online/Redis) | 48 hours | None | Redis TTL |

**Note:** Financial records (`orders`) are subject to a mandatory 10-year
retention requirement under SOX compliance. These rows cannot be hard-deleted
even on GDPR erasure requests — instead, PII columns are overwritten with
null values ("right to erasure" via pseudonymisation).

---

## GDPR Compliance Procedures

### Right to Access (Article 15)
A user requests all data DataChat holds about them.

**Process:**
1. Receive request via support ticket tagged `gdpr-access`
2. Data engineer runs the erasure script: `python gdpr/access_report.py --user-id <uuid>`
3. Script queries `silver.users`, `silver.orders`, `silver.sessions` for the user
4. Output JSON report is delivered to the support team within 30 days (legal requirement)

### Right to Erasure (Article 17)
A user requests deletion of their personal data.

**Process:**
1. Receive request via support ticket tagged `gdpr-erasure`
2. Verify user identity via support team
3. Run the erasure job: `python gdpr/erasure.py --user-id <uuid>`
4. **What is deleted:**
   - `users.email` → replaced with `ERASED_<hash>`
   - `users.username` → replaced with null
   - All `user_events` rows for the user → hard deleted (Bronze and Silver)
   - All `sessions` rows → user_id set to null (anonymous)
   - Feature store online: immediate deletion
   - Feature store offline: partition drop on next daily run
5. **What is retained (legal hold):**
   - `orders` rows — financial records. PII columns nulled, amounts retained.
   - `subscriptions` rows — financial records. Same policy as orders.
6. Erasure must complete within 30 days of request
7. Confirmation logged to `audit.gdpr_erasures` table

### Right to Portability (Article 20)
User requests their data in a machine-readable format.

**Process:** Same as access report but output in JSON-LD format.
Script: `python gdpr/portability_export.py --user-id <uuid> --format json-ld`

---

## CCPA Compliance

California Consumer Privacy Act requirements apply to California residents.

- **Opt-out of data sale:** DataChat does not sell user data. CCPA data sale
  opt-out is therefore trivially satisfied.
- **Right to know:** Equivalent to GDPR Article 15. Same access report script.
- **Right to delete:** Equivalent to GDPR Article 17. Same erasure script.
- **Response SLA:** 45 days (vs 30 days for GDPR)

---

## Access Control Matrix

| Role | Level 0 | Level 1 | Level 2 | Level 3 |
|---|---|---|---|---|
| Data Analyst | Read | Read | No access | No access |
| Data Engineer | Read/Write | Read/Write | Read with audit log | No access |
| Data Engineering Lead | Read/Write | Read/Write | Read/Write with audit log | Incident only |
| Security Team | Read | Read | Read/Write | Incident only |
| ML Engineer | Read | Read (via feature store API) | No direct access | No access |
| BI Developer | Read | No direct table access | No access | No access |

---

## Audit Logging

All accesses to Level 2 columns are logged to `audit.data_access_log`.

| Column | Description |
|---|---|
| access_id | UUID |
| accessed_at | TIMESTAMP UTC |
| accessor_email | The data team member's email |
| table_name | Table accessed |
| columns_accessed | Array of column names |
| query_hash | MD5 of the query |
| row_count | Approximate rows returned |
| justification | Business justification (required for Level 2) |

Audit logs are retained for 7 years and are immutable (append-only Delta table
with no DELETE permissions granted to any role).

---

## Data Classification Checklist (for new tables)

Before promoting a new table to Silver or Gold, the owning engineer must:

1. Complete the column-level PII inventory in this document
2. Set appropriate Row Access Policies if the table contains Level 2 data
3. Configure retention schedule in the lifecycle management config
4. If Level 2 data is present: request approval from data engineering lead
5. Add the table to the data quality monitoring dashboard
6. Document the table in data_catalog.md
