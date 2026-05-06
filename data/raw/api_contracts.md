# Internal API Contracts and Service Documentation

## Overview

This document describes the internal REST APIs and gRPC services that data
pipelines and ML models consume. These are internal services — not exposed
publicly. Authentication uses internal service-to-service JWT tokens issued
by the auth service.

---

## Auth Service

**Base URL (internal):** `http://auth-service.internal:8080`
**Protocol:** REST/JSON
**Authentication:** None (this service IS the auth provider for internal services)

### POST /v1/tokens/service

Issues a short-lived JWT for service-to-service authentication.

**Request:**
```
{
  "service_name": "event_ingestor",
  "scope": ["events:read", "events:write"]
}
```

**Response:**
```
{
  "token": "eyJhbGc...",
  "expires_in": 3600,
  "scope": ["events:read", "events:write"]
}
```

**Token scopes available:**
- `events:read` / `events:write` — access to user events APIs
- `orders:read` / `orders:write` — access to order APIs
- `users:read` / `users:write` — access to user profile APIs
- `features:read` / `features:write` — access to feature store API
- `ml:inference` — access to model serving endpoints

---

## User Service

**Base URL:** `http://user-service.internal:8081`
**Protocol:** REST/JSON
**Auth:** Bearer token with `users:read` scope

### GET /v1/users/{user_id}

Returns the current profile for a user.

**Path parameters:**
- `user_id` — UUID

**Response:**
```
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "username": "johndoe",
  "plan_type": "pro",
  "is_active": true,
  "created_at": "2022-03-15T14:22:00Z",
  "country_code": "US"
}
```

**Error codes:**
- 404: User not found
- 403: Insufficient scope

### GET /v1/users?email={email}

Lookup a user by email address.

**Query parameters:**
- `email` — URL-encoded email string

**Notes:** This endpoint is rate-limited to 100 requests/second per calling service.

### PATCH /v1/users/{user_id}

Update mutable user profile fields. Immutable fields (`user_id`, `created_at`,
`email`) are ignored if included in the request body.

**Request body:** Partial user object with fields to update.

---

## Order Service

**Base URL:** `http://order-service.internal:8082`
**Protocol:** REST/JSON
**Auth:** Bearer token with `orders:read` scope

### GET /v1/orders/{order_id}

**Response:**
```
{
  "order_id": "7f3d2a...",
  "user_id": "550e8400...",
  "product_id": "pro-monthly-usd",
  "status": "confirmed",
  "amount_usd": 29.99,
  "currency": "USD",
  "created_at": "2024-06-01T10:00:00Z",
  "updated_at": "2024-06-01T10:00:05Z",
  "payment_method": "card"
}
```

### GET /v1/orders?user_id={user_id}&status={status}&limit={limit}

List orders for a user with optional status filter.

**Query parameters:**
- `user_id` — required
- `status` — optional, comma-separated list of statuses
- `limit` — optional, max 100, default 20
- `cursor` — optional, pagination cursor from previous response

**Response:**
```
{
  "orders": [...],
  "next_cursor": "eyJpZCI6I...",
  "total_count": 47
}
```

### POST /v1/orders/{order_id}/refund

Initiates a refund for a confirmed order.

**Auth:** Requires `orders:write` scope.

**Request body:**
```
{
  "reason": "customer_request",
  "notes": "Customer reported non-delivery"
}
```

---

## Feature Store API

**Base URL:** `http://feature-store.internal:8083`
**Protocol:** REST/JSON + gRPC (port 9090)
**Auth:** Bearer token with `features:read` scope

### POST /v1/features/online

Retrieve online features for a list of entities.

**Request:**
```
{
  "features": [
    "user_engagement_features:sessions_last_7d",
    "user_purchase_features:is_paying",
    "user_risk_features:churn_probability_score"
  ],
  "entities": [
    {"user_id": "550e8400..."},
    {"user_id": "661f9511..."}
  ]
}
```

**Response:**
```
{
  "results": [
    {
      "user_id": "550e8400...",
      "features": {
        "sessions_last_7d": 12,
        "is_paying": true,
        "churn_probability_score": 0.23
      }
    }
  ],
  "metadata": {
    "latency_ms": 3.2,
    "cache_hit_rate": 0.87
  }
}
```

**SLA:** p99 latency < 10ms under normal load

### gRPC: FeatureStoreService.GetOnlineFeatures

For latency-critical use cases (fraud detection, real-time personalisation),
use the gRPC interface which is ~40% faster than REST due to binary encoding.

Proto definition is in `internal-protos/feature_store.proto`.

---

## Model Serving API

**Base URL:** `http://ml-serving.internal:8084`
**Protocol:** REST/JSON
**Auth:** Bearer token with `ml:inference` scope

### POST /v1/models/churn/predict

Runs the churn prediction model for a batch of users.

**Request:**
```
{
  "user_ids": ["550e8400...", "661f9511..."],
  "return_features": false
}
```

**Response:**
```
{
  "predictions": [
    {"user_id": "550e8400...", "churn_probability": 0.23, "risk_tier": "low"},
    {"user_id": "661f9511...", "churn_probability": 0.81, "risk_tier": "high"}
  ],
  "model_version": "v3.2",
  "inference_latency_ms": 18.4
}
```

**Risk tiers:**
- `low`: churn_probability < 0.4
- `medium`: 0.4 ≤ churn_probability < 0.65
- `high`: churn_probability ≥ 0.65

### POST /v1/models/fraud/score

Real-time fraud scoring for a single order. Called by the order service before
confirming payment.

**Request:**
```
{
  "order_id": "7f3d2a...",
  "user_id": "550e8400...",
  "amount_usd": 299.99,
  "payment_method": "card",
  "ip_country": "NG",
  "user_country": "US"
}
```

**Response:**
```
{
  "fraud_score": 0.92,
  "decision": "hold",
  "reasons": ["ip_country_mismatch", "high_velocity"],
  "model_version": "v1.8"
}
```

**Decisions:** `pass`, `hold`, `block`

---

## Event Registry API

**Base URL:** `http://event-registry.internal:8085`
**Protocol:** REST/JSON

### GET /v1/events/{event_type}/schema

Returns the JSON schema for a given event type. Used by the event_ingestor
pipeline to validate incoming events.

**Example:** `GET /v1/events/purchase.complete/schema`

**Response:**
```
{
  "event_type": "purchase.complete",
  "version": "2.1",
  "schema": {
    "type": "object",
    "required": ["order_id", "amount_usd", "currency"],
    "properties": {
      "order_id":   {"type": "string", "format": "uuid"},
      "amount_usd": {"type": "number", "minimum": 0},
      "currency":   {"type": "string", "pattern": "^[A-Z]{3}$"}
    }
  }
}
```

### GET /v1/events

List all registered event types with version and owner information.

---

## Service SLAs

| Service | p50 Latency | p99 Latency | Availability Target |
|---|---|---|---|
| Auth Service | 2ms | 10ms | 99.99% |
| User Service | 5ms | 25ms | 99.95% |
| Order Service | 8ms | 40ms | 99.95% |
| Feature Store (REST) | 4ms | 10ms | 99.9% |
| Feature Store (gRPC) | 2ms | 6ms | 99.9% |
| Model Serving (churn) | 15ms | 50ms | 99.9% |
| Model Serving (fraud) | 10ms | 30ms | 99.99% |

---

## Common Error Codes

| HTTP Code | Meaning | Action |
|---|---|---|
| 400 | Bad request — check request body schema | Fix client code |
| 401 | Token missing or expired | Re-fetch token from auth service |
| 403 | Insufficient scope | Request additional scope from auth service |
| 404 | Resource not found | Verify entity ID exists |
| 429 | Rate limit exceeded | Back off with exponential retry |
| 503 | Service temporarily unavailable | Retry with exponential backoff, max 3 attempts |

All services support the `Retry-After` header on 429 responses.
