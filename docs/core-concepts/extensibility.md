---
title: Extensibility
description: Use extension metadata and payload modes to shape how EventDBX integrates with the rest of your stack.
nav_id: guides
---

# Extensibility

EventDBX keeps the write path small and verifiable, then lets you hang read models, pipelines, and policy engines off the side. Two core tools make that possible: a namespaced extension envelope attached to every event and payload modes that tailor what each plugin receives.

## Extension envelopes on events

Writes accept an optional `--metadata` JSON object. The server stores it separately as `extensions` so system metadata (Snowflake id, timestamps, issuer) stays distinct from your own hints. CLI JSON output folds those namespaced entries into the `metadata` object alongside `created_at` and `event_id` so they are not dropped when you include events; plugin payloads still receive the `extensions` map explicitly.

- Keys must be namespaced with an `@` prefix (for example `@trace`, `@policy-rule`).
- The extension map must be a JSON object and is capped at 64 KiB.
- The bag travels with the event everywhere: CLI/API reads, filters, exports, and every plugin payload mode that includes events or extensions.

Example envelope delivered to plugins and event listings:

```json
{
  "aggregate_type": "patient",
  "aggregate_id": "p-001",
  "event_type": "patient-updated",
  "payload": { "status": "inactive" },
  "metadata": {
    "event_id": "1234567890123",
    "created_at": "2025-02-03T12:34:56.123Z",
    "issued_by": { "group": "admin", "user": "jane" },
    "@trace": { "correlation_id": "rest-1234" },
    "@policy": { "source": "rest-api" }
  },
  "version": 5,
  "hash": "<hash>",
  "merkle_root": "<merkle_root>"
}
```

## Writing and consuming extensions

- Attach namespaced hints as you append events:

  ```bash
  dbx aggregate apply patient p-001 patient-updated \
    --payload '{"status":"active"}' \
    --metadata '{"@trace":{"correlation_id":"cli-22"},"@policy":{"role":"doctor"}}'
  ```

- Filter event streams by extension data when debugging or replaying:

  ```bash
  dbx events --filter 'extensions.@trace.correlation_id = "cli-22"' --take 10
  ```

- Replay stored events through a plugin to validate how payload modes affect deliveries:

  ```bash
  dbx plugin replay search patient p-001 --payload-mode extensions-only
  ```

- Plugins see the same extension bag even when you hide payload/state; use it for correlation ids, audit markers, policy decisions, or routing keys.

## Plugin payload modes (including extensions-only)

Configure each plugin with the smallest payload it needs via `--payload`:

| Mode               | Includes                                                      | Use when you need…                           |
|--------------------|---------------------------------------------------------------|----------------------------------------------|
| `all`              | Event, materialised state, schema                             | Rich projections or migrations               |
| `event-only`       | Event document                                                | Webhooks, audit relays, simple streams       |
| `state-only`       | Latest state map                                              | Cache warmers, read replicas                 |
| `schema-only`      | Aggregate schema                                              | Schema registries, validators                |
| `event-and-schema` | Event + schema, no state                                      | Validation services, codegen                 |
| `extensions-only`  | Namespaced `extensions` map; payload/state nulled before send | Telemetry, policy gates, downstream fan-out without exposing payloads |

`extensions-only` strips payload and state before enqueueing jobs or delivering directly to plugins, while keeping identifiers and the extension map intact. Pair it with lightweight HTTP/process plugins when you just need correlation ids, routing hints, or policy labels to reach external systems. Because plugins consume events asynchronously, slow downstream systems never block writers; swap payload modes or add connectors as your architecture evolves.
