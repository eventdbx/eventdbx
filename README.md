# EventDBX

EventDBX is a local-first, single-tenant, write-side event database for developers who want an auditable system of record without standing up a full event platform.

It stores every accepted change as an immutable event, materializes current aggregate state from that history, and keeps the write path explicit and schema-checked. In practice, that makes it a good fit for applications where state transitions matter more than ad hoc document updates: order flows, workflow engines, internal tools, operational systems, and domain models that need replayable history.

The core model is intentionally narrow:

- one logical domain, pinned to `default`
- one authority for writes
- one local workspace discovered from `.dbx/config.toml`
- one schema source, authored in `schema.dbx`

What the supported core provides:

- authenticated control-plane and write operations
- immutable event history and aggregate reconstruction
- schema enforcement and reference validation from `schema.dbx`
- aggregate reads and event inspection from the local daemon
- Merkle-style integrity verification
- automatic snapshots on the write path
- a pull-based outbox for downstream consumers

What this lean build does not try to be:

- a multi-tenant hosting platform
- a built-in peer replication system
- a plugin runtime or queue orchestrator
- a general-purpose document database
- a full backup, restore, and upgrade management suite

## Quick Start

1. Initialize a local workspace in your project directory:

```bash
dbx init
```

This creates `./.dbx` and stores both configuration and runtime data there. Runtime commands discover the nearest `.dbx/config.toml` by walking up from the current directory, similar to Git repository discovery.
`dbx init` also emits a CLI bootstrap token and persists it under `.dbx/cli.token`; the default init-issued token expires after 86,400 seconds.
Use `--ttl` to override that bootstrap token lifetime with either raw seconds or human suffixes like `10m`, `24h`, `10d`, or `2w`. Use `--ttl 0` to issue a non-expiring bootstrap token.

2. Start the daemon:

```bash
dbx serve start --foreground
```

Use `dbx serve status` to inspect the daemon, `dbx serve restart` to restart it, and `dbx serve stop` when you want to shut it down.
The published Docker image sets `EVENTDBX_AUTO_INIT=1`, so a first run against an empty `/var/lib/eventdbx` volume will create `/var/lib/eventdbx/.dbx/config.toml` automatically before the server starts.

3. Author a schema in `schema.dbx`:

```text
aggregate order {
  snapshot_threshold = 100

  field status {
    type = "text"
  }

  event order_created {
    fields = ["status"]
  }

  event order_paid {
    fields = ["status"]
  }
}
```

4. Validate and preview the compiled runtime schema:

```bash
dbx schema validate
dbx schema show
dbx schema show order
```

Schema commands discover the nearest `schema.dbx` in the current directory tree. Pass `--file <PATH>` to `dbx schema validate`, `dbx schema show`, or `dbx schema apply` to override discovery.
`dbx schema show` is a deterministic preview of the compiled runtime shape. It intentionally normalizes volatile timestamps so unchanged source renders the same output across runs. Pass an aggregate name to render only that aggregate's compiled schema.

5. Apply the schema:

```bash
dbx schema apply
```

6. Bootstrap or mint a token:

```bash
dbx token bootstrap --stdout
dbx token generate --group ops --user alice --action aggregate.read --action aggregate.append --json
```

7. Write and read aggregates:

```bash
dbx aggregate create order order-42 --event order_created --field status=open --json
dbx aggregate apply order order-42 order_paid --field status=paid
dbx aggregate get order order-42 --include-events
dbx aggregate list order --json
dbx events order order-42 --json
dbx aggregate verify order order-42 --json
```

Automatic snapshots are still part of the core write path. Set `snapshot_threshold` in `schema.dbx`; EventDBX will create snapshots internally after qualifying writes. The standalone snapshot command surface has been removed.

## Schema Reference

`schema.dbx` is the authoring format for runtime schema compilation. The compiler parses a small DSL whose assigned values are JSON.

Top-level grammar:

```text
# comments with '#' or '//'
aggregate order {
  snapshot_threshold = 100

  field status {
    type = "text"
    rules = {"required": true}
  }

  event order_created {
    fields = ["status"]
  }
}
```

Rules of the format:

- top-level entries are `aggregate <name> { ... }`
- names may be bare tokens or quoted JSON strings
- `# ...` and `// ...` comments are supported
- each aggregate must define at least one event
- aggregate, field, and event names must be unique within their scope
- field and event property values are JSON values

Use snake_case names in practice. Quoted names are accepted by the parser, but snake_case names are the safest choice for CLI usage and reference compatibility.

### Aggregate Properties

| Property | Type | Meaning |
| --- | --- | --- |
| `snapshot_threshold` | integer or `null` | Enables automatic snapshots every N qualifying writes. |
| `locked` | boolean | Rejects writes for the aggregate when `true`. |
| `hidden` | boolean | Marks the aggregate hidden in schema metadata. |
| `hidden_fields` | array of strings | Marks named fields as hidden in schema metadata. |
| `field_locks` | array of strings | Rejects writes that attempt to mutate those fields. |
| `field <name> { ... }` | block | Declares field types and validation rules. |
| `event <name> { ... }` | block | Declares allowed event names and event-level field constraints. |

### Field Blocks

Each field block supports:

| Property | Type | Meaning |
| --- | --- | --- |
| `type` | string | Required. Declares the field column type or type alias. |
| `rules` | JSON object | Optional validation rules. Must be a JSON object when present. |
| `hidden` | boolean | Adds the field to the aggregate's hidden field list. |
| `locked` | boolean | Adds the field to the aggregate's locked field list. |

Supported field types and aliases:

| Type | Accepted values |
| --- | --- |
| integer | `integer`, `int` |
| float | `float`, `double` |
| decimal | `decimal(p,s)`, `numeric(p,s)` |
| boolean | `boolean`, `bool` |
| text | `text`, `string` |
| timestamp | `timestamp` |
| date | `date` |
| json | `json` |
| binary | `binary`, `bytes` |
| object | `object` |
| reference alias | `ref`, `reference`, `aggregate_ref`, `aggregate-reference` |

Reference aliases compile to a text column with `format = "reference"` semantics.

### Event Blocks

Each event block supports:

| Property | Type | Meaning |
| --- | --- | --- |
| `fields` | array of strings | Optional event field allowlist. When non-empty, these fields become both the required set and the permit list for that event payload. |
| `note` | string or `null` | Optional default note for the event. Maximum length is 128 characters. |

### Field Rules

`rules` must be a JSON object. Supported keys:

| Rule | Type | Meaning |
| --- | --- | --- |
| `required` | boolean | Field must be present. |
| `contains` | array of strings | Text value must contain every listed substring. |
| `does_not_contain` | array of strings | Text value must not contain any listed substring. |
| `regex` | array of strings | Text value must match every listed regular expression. |
| `length` | object | Length limits for text or binary values. Supports `min` and `max`. |
| `range` | object | Range limits for integer, float, decimal, timestamp, or date values. Supports `min` and `max`. |
| `format` | string | Built-in semantic validator. |
| `reference` | object | Additional constraints for reference-valued text fields. |
| `properties` | object | Nested property definitions for structured object fields. |

Supported `format` values:

- `email`
- `url`
- `credit_card`
- `country_code`
- `iso8601`
- `wgs_84`
- `camel_case`
- `snake_case`
- `kebab_case`
- `pascal_case`
- `upper_case_snake_case`
- `reference`

Important validation constraints:

- `length` applies only to text and binary values
- `range` applies only to integer, float, decimal, timestamp, and date values
- `format = "reference"` requires a text field
- `reference` rules without `format = "reference"` are invalid
- nested `properties` are only accepted on `object` or `json` field types
- use `object` for recursive nested validation and nested reference normalization

### Reference Rules

Reference-valued fields use a text column with `format = "reference"` and optional `reference` rules:

```text
field manager {
  type = "reference"
  rules = {
    "format": "reference",
    "reference": {
      "aggregate_type": "person",
      "integrity": "strong",
      "cascade": "restrict"
    }
  }
}
```

Supported `reference` keys:

| Key | Type | Meaning |
| --- | --- | --- |
| `integrity` | string | `strong` or `weak`. `strong` rejects missing/forbidden targets; `weak` allows unresolved targets. |
| `aggregate_type` | string | Restricts the reference to a specific aggregate type. |
| `cascade` | string | `none`, `restrict`, or `nullify`. Stored in schema metadata for downstream reference handling. |

Accepted reference string shapes:

- `aggregate#id`
- `#id`

Legacy `domain#aggregate#id` values are still accepted during reads, but reference values are canonicalized to `aggregate#id` during normalization. Invalid shapes, wrong target aggregate, or unresolved strong references fail validation.

### Nested Object Rules

Use `properties` to validate nested fields inside structured values:

```text
field profile {
  type = "object"
  rules = {
    "properties": {
      "country": {
        "type": "text",
        "format": "country_code"
      },
      "nickname": {
        "type": "text",
        "length": {"max": 32}
      }
    }
  }
}
```

Inside `properties`, each entry uses JSON schema settings:

- a simple type string such as `"postal_code": "text"`
- or an object containing `"type"` plus flattened rule keys such as `"country": {"type": "text", "format": "country_code"}`

### Full Example

```text
aggregate person {
  snapshot_threshold = 100
  locked = false
  hidden = false
  hidden_fields = ["internal_notes"]
  field_locks = ["id"]

  field email {
    type = "text"
    rules = {"required": true, "format": "email"}
  }

  field manager {
    type = "reference"
    hidden = true
    locked = true
    rules = {
      "format": "reference",
      "reference": {
        "tenant": "default",
        "aggregate_type": "person",
        "integrity": "strong",
        "cascade": "restrict"
      }
    }
  }

  field profile {
    type = "object"
    rules = {
      "properties": {
        "country": {
          "type": "text",
          "format": "country_code"
        },
        "nickname": {
          "type": "text",
          "length": {"max": 32}
        }
      }
    }
  }

  event person_created {
    fields = ["email", "manager"]
    note = "Initial import"
  }

  event person_updated {}
}
```

### Rules Cookbook

Common `rules` examples:

```text
field email {
  type = "text"
  rules = {"required": true, "format": "email"}
}

field sku {
  type = "text"
  rules = {
    "format": "upper_case_snake_case",
    "regex": ["^[A-Z0-9_]+$"],
    "length": {"min": 3, "max": 32}
  }
}

field amount {
  type = "decimal(12,2)"
  rules = {"range": {"min": "0.00", "max": "9999999999.99"}}
}

field effective_at {
  type = "timestamp"
  rules = {"format": "iso8601", "range": {"min": "2026-01-01T00:00:00Z"}}
}

field attachment {
  type = "binary"
  rules = {"length": {"max": 1048576}}
}
```

`dbx schema validate` checks the source file, and `dbx schema show` prints the normalized compiled runtime schema rather than echoing raw source text.

## Outbox

EventDBX exposes a pull-based control-socket outbox for downstream replication.

Contract:

- `readOutbox(afterEventId?, take?) -> { eventsJson, nextAfterEventId }`
- events are committed, active-only, and globally ordered by ascending `event_id`
- `afterEventId` is exclusive
- `nextAfterEventId` is the checkpoint a replicator should persist

Recommended replication shape:

1. bootstrap a remote once with replay or export/import
2. pull local outbox batches from the authoritative EventDBX instance
3. push those events to the downstream system
4. advance the checkpoint only after successful delivery

This replaces built-in peer replication. The core daemon does not manage destination config, retries, scheduling, or bidirectional reconciliation.

## CLI Surface

Supported commands:

- `dbx init`
- `dbx serve start|stop|status|restart|destroy`
- `dbx config`
- `dbx token generate|list|revoke|refresh|bootstrap`
- `dbx schema validate|show|apply`
- `dbx aggregate create|apply|list|get|verify`
- `dbx events`

Operational notes:

- Server lifecycle commands now live under `dbx serve`. The old top-level commands `dbx start`, `dbx stop`, `dbx status`, `dbx restart`, and `dbx destroy` are no longer supported.
- `dbx serve start --restrict [off|default|strict]` controls schema enforcement mode for the running daemon
- `dbx serve destroy` removes the active `.dbx` workspace directory and prompts for confirmation unless `--yes` is passed
- `dbx serve destroy` does not imply that an external `data_dir` outside the workspace will be removed
- `dbx schema validate|show|apply` discover the nearest `schema.dbx` by default and accept `--file <PATH>` to override
- `dbx schema show [AGGREGATE]` renders either the full compiled schema set or a single aggregate preview

Lifecycle examples:

```bash
dbx serve start
dbx serve status
dbx serve restart --foreground
dbx serve stop
dbx serve destroy --yes
```

## Configuration

Run `dbx config` with no update flags to print the current lean runtime config as TOML.
Writable configuration flags are:

- `--port`
- `--data-dir`
- `--cache-threshold`
- `--data-encryption-key`
- `--list-page-size`
- `--page-limit`
- `--bind-addr`
- `--snowflake-worker-id`
- `--noise`
- `--no-noise`

The default config location is the nearest `.dbx/config.toml` in the current directory tree.
Workspace initialization generates auth keys automatically; the printed config includes the `auth` block even though key material is not configured through dedicated `dbx config` flags.
Schema enforcement mode is a runtime/startup concern controlled via `dbx serve start --restrict`, not `dbx config`.

EventDBX now runs as a single-tenant core pinned to the logical domain `default`.

## Migration Note

This build refuses to start against legacy multi-domain or multi-tenant layouts.

Examples of rejected state:

- `config.toml` selecting a non-default domain
- `[tenants].multi_tenant = true`
- extra tenant/domain directories under the data root

The error is intentional. Export/import the data or run a dedicated migration before upgrading to the lean single-tenant build.
