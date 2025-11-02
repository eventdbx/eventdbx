# EventDBX

You’ll appreciate this database system. It lets you spend less time designing schemas and more time writing code that drives your application.

## Overview

EventDBX is an event-sourced, NoSQL write-side database designed to provide immutable, append-only storage for events across various domains. It is ideal for applications requiring detailed audit trails for compliance, complex business processes involving states, and high data integrity levels. The core engine focuses on the _write_ side of CQRS—capturing and validating events, persisting aggregate state, and ensuring integrity with Merkle trees.

The companion plugin framework turns those events into durable jobs and delivers them to other services, letting external systems specialise in the _read_ side. Whether you need to hydrate a search index, stream to analytics, feed caches, or trigger workflows, plugins let you extend EventDBX without altering the write path. Each plugin chooses the payload shape it needs (event-only, state-only, schema-only, or combinations) while the queue guarantees delivery and backoff.

## Getting Started

Follow the steps below to spin up EventDBX locally. The commands assume you installed the CLI globally; if you're contributing to the project, clone the repository and run them from the repo root instead.

The CLI installs as `dbx`. Older releases exposed an `eventdbx` alias, but the primary command is now `dbx`.

1. **Install the CLI**

   ```bash
   # Install prebuilt binaries via shell script
   curl --proto '=https' --tlsv1.2 -LsSf https://github.com/thachp/eventdbx/releases/download/v3.6.5/eventdbx-installer.sh | sh
   ```

   ```bash
   # Install prebuilt binaries via powershell script
   powershell -ExecutionPolicy Bypass -c "irm https://github.com/thachp/eventdbx/releases/download/v3.6.5/eventdbx-installer.ps1 | iex"
   ```

2. **Start the server**

   ```bash
   dbx start --foreground
   ```

   - Omit `--foreground` to daemonise the process.
   - Use `--data-dir <path>` to override the default `$HOME/.eventdbx` directory.

- Restriction (schema enforcement) defaults to `default`; switch to `--restrict=off` for permissive prototyping or `--restrict=strict` to require declared schemas on every write.

- Public REST/GraphQL/gRPC surfaces now live in the [dbx_plugins workspace](https://github.com/thachp/dbx_plugins). Install the `rest`, `graphql`, or `grpc` plugin alongside the daemon to expose those endpoints. The built-in server only retains the admin API and control socket.

  ```bash
  # install the restful api plugin
  dbx plugin install rest

  # config the api to not receive events stream
  dbx plugin config process rest --emit-events false

  # start the rest endpoint, default port is 8080
  dbx plugin start rest
  ```

3. **Switch domains (optional)**

   ```bash
   dbx checkout -d herds
   ```

   - New installs default to the `default` domain, which stores data directly under the configured `data_dir`.
   - `dbx checkout -d <domain>` (or `dbx checkout <domain>`) changes the active domain and creates isolated storage in `<data_dir>/domains/<domain>` so you can group aggregates and plugins per bounded context.
   - Domain names are case-insensitive, must begin with a letter or number, and may include dashes or underscores. Re-running the command for the current domain is a no-op.

4. **Define a schema (recommended when running in restricted mode)**

   ```bash
   dbx schema create patient \
     --events patient_added,patient_updated \
     --snapshot-threshold 100
   ```

   Omit `--snapshot-threshold` to inherit the default configured in `config.toml` (if any).

   EventDBX stores schemas in `~/.eventdbx/data/schemas.json` (or the `data_dir` you configured). Inspect the full definition at any time with `dbx schema <aggregate>`; the command prints the JSON the server enforces. For example, a `person` aggregate that validates email format and enforces name lengths can be declared as:

   ```bash
   dbx schema create person --events person_created,person_updated
   dbx schema person
   ```

   ```json
   {
     "person": {
       "aggregate": "person",
       "snapshot_threshold": null,
       "locked": false,
       "field_locks": [],
       "hidden": false,
       "hidden_fields": [],
       "column_types": {
         "email": { "type": "text", "required": true, "format": "email" },
         "name": {
           "type": "text",
           "rules": { "length": { "min": "1", "max": "64" } }
         }
       },
       "events": {
         "person_created": {
           "fields": ["first_name", "last_name"]
         },
         "person_updated": {
           "fields": []
         }
       },
       "created_at": "2025-10-26T22:25:24.028129Z",
       "updated_at": "2025-10-26T22:27:25.967615Z"
     }
   }
   ```

   - `column_types` declare data types, validation formats, and rule blocks per field. Rules can enforce length, numeric ranges, regexes, or cross-field matches.
   - `events.<event>.fields` restrict the properties an event may set; leaving the list empty keeps the event permissive.
   - `field_locks` and `hidden_fields` control which fields can be updated or returned in aggregate detail calls.

- `locked: true` freezes the schema to prevent further event writes until it is unlocked.

4. **Issue a token for CLI access**

```bash
dbx token generate --group admin --user jane --expiration 3600
```

5. **Append an event**
   - When the server is running the CLI proxies writes through the control socket automatically. Pass a token with `--token` or set `EVENTDBX_TOKEN` to reuse your own credentials; otherwise the CLI mints a short-lived token for the call.
     ```bash
     dbx aggregate apply person p-002 patient-added \
       --field name="Jane Doe" \
       --field status=active
     ```
   - If the server is stopped, the same CLI command writes to the local RocksDB store directly.
   - Inspect recent history at any point with `dbx events --filter 'payload.status = "active"' --sort created_at:desc --take 5` or drill into the payload of a specific event via `dbx events <snowflake_id> --json`.

You now have a working EventDBX instance with an initial aggregate. Explore the [Command-Line Reference](#command-line-reference) for the full set of supported operations.

## Run with Docker

### Use the published image

```bash
# Pull the latest published release
docker pull thachp/eventdbx:tagname

# Start the daemon with persistent storage and required ports
docker run \
  --name eventdbx \
  --detach \
  --publish 7070:7070 \
  --publish 6363:6363 \
  --volume "$PWD/data:/var/lib/eventdbx" \
  thachp/eventdbx:tagname
```

###

```typescript
// Connect with EventDBX using libraries in your language of choice.
// Only Javascript / Typescript is supported. Ie, pnpm install eventdbxjs

import { createClient } from 'eventdbxjs';

const client = createClient({
  ip: process.env.EVENTDBX_HOST,
  port: Number(process.env.EVENTDBX_PORT) || 6363,
  token: process.env.EVENTDBX_TOKEN,
  verbose: true, // should match verbose_responses = false on the server config file
});

await client.connect();

// create an aggregate
const snapshot = await client.create('person', 'p-110', 'person_registered', {
  payload: { name: 'Jane Doe', status: 'active' },
  metadata: { '@source': 'seed-script' },
  note: 'seed aggregate',
});
console.log('created aggregate version', snapshot.version);

// apply an event
await client.apply('person', 'p-110', 'person_contact_added', {
  payload: { name: 'Jane Doe', status: 'active' },
  metadata: { note: 'seed data' },
});

// return a list of events of a p-110
const history = await client.events('person', 'p-110');
console.log('event count:', history.length);
```

## Features

- **Flexible JSON payloads**: Events accept arbitrary JSON payloads; scalar values are normalized into strings for state tracking, while structured objects remain fully queryable.
- **Immutable Data Structure**: Once data is entered into EventDBX, it becomes immutable, meaning it cannot be altered or deleted. This characteristic is crucial for applications where the accuracy and traceability of historical data are paramount, such as medical records, financial transactions, and supply chain management. Data can be archived, moving from short-term to long-term storage, but cannot be deleted.
- **Event Sourcing and Replay**: EventDBX is built on the principle of event sourcing, storing all changes to the data as a sequence of events. This allows for the complete replay of events to reconstruct the database's state at any point in time, thereby enhancing data recovery and audit capabilities. Unlike traditional databases that execute update statements to modify data, this system is event-driven. Aggregate state changes are defined in the event object, allowing these events to be replayed at any time to reconstruct the aggregate's current state.
- **Merkle Tree Integration**: Each aggregate in EventDBX is associated with a Merkle tree of events, enabling verification of data integrity. The Merkle tree structure ensures that any data tampering can be detected, offering an additional security layer against data corruption.
- **Built-in Audit Trails**: EventDBX automatically maintains a comprehensive audit trail of all transactions, a feature invaluable for meeting compliance and regulatory requirements. It provides transparent and tamper-evident records. During audits, administrators can issue specific tokens to auditors to access and review specific aggregate instances and all relevant events associated with those instances.
- **Extensible Read Models via Plugins**: Every event that lands in the write-side store can be dispatched to external systems through the plugin job queue. Configure plugins to receive only the slices they care about (event, state snapshot, schema), and let specialized services—search, analytics, personalization, alerting—build optimized read models without touching the write path.
- **Built-in Observability**: A Prometheus-compatible `/metrics` endpoint reports HTTP traffic and plugin queue health so you can wire EventDBX into Grafana, Datadog, or any other monitoring stack out of the box.
- **Security with Token-Based Authorization**: EventDBX implements token-based authorization to manage database access. Tokens are signed with an Ed25519 key pair stored under `[auth]` in `config.toml`; keep `private_key` secret and distribute `public_key` to services that need to validate them. This approach allows for precise control over who can access and modify data, protecting against unauthorized changes.
- **Encrypted Payloads & Secrets at Rest**: Event payloads, aggregate snapshots, and `tokens.json` are encrypted transparently when a DEK is configured. Metadata such as aggregate identifiers, versions, and Merkle roots remain readable so plugins, replication, and integrity checks keep working without additional configuration.
- **Peer replication**: Keep deployments in sync with `dbx push` and `dbx pull`, selectively streaming aggregates, schemas, or both between trusted peers over the builtin Cap'n Proto transport.
- **Schema validation**: Enforce aggregate contracts with builtin formats (`email`, `url`, `wgs_84`, and more), rule blocks for length, range, regex, required fields, and nested `properties`, plus strict/relaxed enforcement modes.
- **Powered by RocksDB and Rust**: At its core, EventDBX utilizes RocksDB for storage, taking advantage of its high performance and efficiency. The system is developed in Rust, known for its safety, efficiency, and concurrency capabilities, ensuring that it is both rapid and dependable.

## Restriction Modes

EventDBX can run in three validation modes, tuned for different phases of development:

- **Off** (`--restrict=off` or `--restrict=false`): Ideal for prototyping and rapid application development. Event payloads bypass schema validation entirely, letting you iterate quickly without pre-registering aggregates, tables, or column types.
- **Default** (`--restrict=default` or `--restrict=true`): Validates events whenever a schema exists but allows aggregates without a declared schema. This matches prior behaviour and suits teams rolling out schema enforcement incrementally.
- **Strict** (`--restrict=strict`): Requires every aggregate to have a schema before events can be appended. Missing schemas fail fast with a clear error so production environments stay aligned with their contracts.

## Command-Line Reference

EventDBX ships a single `dbx` binary. Every command accepts an optional `--config <path>` to point at an alternate configuration file.

### Server lifecycle

- `dbx start [--port <u16>] [--data-dir <path>] [--foreground] [--restrict[=<off|default|strict>]]`  
  Launches the server. The default mode validates when schemas exist; set `--restrict=off` to bypass validation or `--restrict=strict` to require schemas up front.
- `dbx stop`  
  Stops the running daemon referenced by the PID file.
- `dbx status`  
  Prints the current port, PID, uptime, and whether restriction is enabled.
- `dbx restart [start options…]`  
  Stops the existing daemon (if any) and restarts it with the provided options.
- `dbx destroy [--yes]`  
  Removes the PID file, data directory, and configuration file after confirmation (or immediately with `--yes`).
- `dbx checkout -d <domain>`  
  Switches the active domain context for the CLI. Uses the `default` domain by default and creates per-domain data roots under `<data_dir>/domains/<domain>`.
- `dbx merge --from <domain> [--into <domain>] [--overwrite-schemas]`  
  Copies events (and their derived state) plus schema definitions from the source domain into the target domain (defaults to the active domain). Conflicting schemas require `--overwrite-schemas`; existing aggregates in the target abort the merge to prevent data loss.

### Configuration

- `dbx config [--port <u16>] [--data-dir <path>] [--cache-threshold <usize>] [--dek <base64>] [--list-page-size <usize>] [--page-limit <usize>] [--plugin-max-attempts <u32>] [--snapshot-threshold <u64>] [--clear-snapshot-threshold] [--admin-enabled <true|false>] [--admin-bind <addr>] [--admin-port <u16>]`  
  Persists configuration updates. Run without flags to print the current settings. The first invocation must include a data encryption key via `--dek` (32 bytes of base64). `--list-page-size` sets the default page size for aggregate listings (default 10), `--page-limit` caps any requested page size across list and event endpoints (default 1000), and `--plugin-max-attempts` controls how many retries are attempted before an event is marked dead (default 10). The Admin API relies on bearer tokens; set `--admin-enabled=false` to disable the HTTP surface entirely.

### Tokens

- `dbx token generate --group <name> --user <name> [--expiration <secs>] [--limit <writes>] [--keep-alive]`  
  Issues a new token tied to a Unix-style group and user.
- `dbx token list`  
  Lists all tokens with status, expiry, and remaining writes.
- `dbx token revoke --token <value>`  
  Revokes a token immediately.
- `dbx token refresh --token <value> [--expiration <secs>] [--limit <writes>]`  
  Extends the lifetime or write allowance of an existing token.

### Schemas

- `dbx schema create <name> --events <event1,event2,...> [--snapshot-threshold <u64>]`
- `dbx schema add <name> --events <event1,event2,...>`
- `dbx schema remove <name> <event>`
- `dbx schema annotate <name> <event> [--note <text>] [--clear]`
- `dbx schema list`

Schemas are stored on disk; when restriction is `default` or `strict`, incoming events must satisfy the recorded schema (and `strict` additionally requires every aggregate to declare one).

### Aggregates

- `dbx aggregate create --aggregate <type> --aggregate-id <id> --event <name> [--field KEY=VALUE...] [--payload <json>] [--metadata <json>] [--note <text>] [--token <value>] [--json]`
- `dbx aggregate apply --aggregate <type> --aggregate-id <id> --event <name> --field KEY=VALUE... [--payload <json>] [--stage] [--token <value>] [--note <text>]`  
  Appends an event immediately—use `--stage` to queue it for a later commit.
- `dbx aggregate patch --aggregate <type> --aggregate-id <id> --event <name> --patch <json> [--stage] [--token <value>] [--metadata <json>] [--note <text>]`  
  Applies an RFC 6902 JSON Patch to the current state and persists the delta as a new event.
- `dbx aggregate list [--skip <n>] [--take <n>] [--stage]`  
  Lists aggregates with version, Merkle root, and archive status; pass `--stage` to display queued events instead.
- `dbx aggregate get --aggregate <type> --aggregate-id <id> [--version <u64>] [--include-events]`
- `dbx aggregate replay --aggregate <type> --aggregate-id <id> [--skip <n>] [--take <n>]`
- `dbx aggregate verify --aggregate <type> --aggregate-id <id>`
- `dbx aggregate snapshot --aggregate <type> --aggregate-id <id> [--comment <text>]`
- `dbx aggregate archive --aggregate <type> --aggregate-id <id> [--comment <text>]`
- `dbx aggregate restore --aggregate <type> --aggregate-id <id> [--comment <text>]`
- `dbx aggregate remove --aggregate <type> --aggregate-id <id>` Removes an aggregate that has no events (version still 0).
- `dbx aggregate commit`  
  Flushes all staged events in a single atomic transaction.

### Events

- `dbx events [--aggregate <type>] [--aggregate-id <id>] [--skip <n>] [--take <n>] [--filter <expr>] [--sort <field[:order],...>] [--json] [--include-archived|--archived-only]`  
  Streams events with optional aggregate scoping, SQL-like filters (e.g. `payload.status = "open" AND metadata.note LIKE "retry%"`), and multi-key sorting. Prefix fields with `payload.`, `metadata.`, or `extensions.` to target nested JSON; `created_at`, `event_id`, `version`, and other top-level keys are also available.
- `dbx events --event <snowflake_id> [--json]`  
  Displays a single event by Snowflake identifier, including payload, metadata, and extensions. You can also omit `--event` when the first positional argument is a valid Snowflake id.
- `dbx aggregate export [<type>] [--all] --output <path> [--format csv|json] [--zip] [--pretty]`  
  Writes the current aggregate state (no metadata) as CSV or JSON. Exports default to one file per aggregate type; pass `--zip` to bundle the output into an archive.

`--payload` lets you provide the full event document explicitly; use `dbx aggregate patch` when you need to apply an RFC 6902 JSON Patch server-side.

#### Aggregate Operation Costs

Every aggregate command ultimately turns into a small set of RocksDB reads or writes. Ballpark complexities:

- `aggregate list`: iterates over the requested page of aggregates. Runtime is O(k) in the page size (default `list_page_size`).
- `aggregate get`: one state lookup (O(log N)) plus optional event scanning when you request `--include-events` or `--version` (adds O(Eₐ) for that aggregate’s events) and lightweight JSON parsing of the returned map.
- `aggregate select`: uses the same state lookup as `get` (O(log N)) and walks each requested dot path in-memory; no additional RocksDB reads are taken, so cost is dominated by the selected payload size.
- `aggregate apply`: validates the payload, merges it into the materialized state, and appends the event in a single batch write. Time is proportional to the payload size being processed.
- `aggregate patch`: reads the current state (same cost as `get`), applies the JSON Patch document, then appends the result—effectively O(payload + patch_ops).
- `aggregate replay` / `aggregate list --include-events`: scans the requested slice of the event stream for that aggregate (O(Eₐ)).
- `aggregate verify`: recomputes the Merkle root for the aggregate’s events (O(Eₐ)).

In practice those costs are dominated by payload size and the number of events you ask the CLI to stream; hot aggregates tend to stay in the RocksDB block cache, keeping per-operation latency close to constant.

| Operation          | Time complexity          | Notes                                                            |
| ------------------ | ------------------------ | ---------------------------------------------------------------- |
| `aggregate list`   | O(k)                     | `k` is the requested page size (defaults to `list_page_size`).   |
| `aggregate get`    | O(log N + Eₐ + P)        | Single state read plus optional event scan and JSON parsing.     |
| `aggregate select` | O(log N + P_selected)    | Same state read as `get`; dot-path traversal happens in memory.  |
| `aggregate apply`  | O(P)                     | Payload validation + merge + append in one RocksDB batch.        |
| `aggregate patch`  | O(log N + P + patch_ops) | Reads state, applies JSON Patch, then appends the patch payload. |
| `aggregate replay` | O(Eₐ)                    | Linear in the number of events streamed.                         |
| `aggregate verify` | O(Eₐ)                    | Recomputes the Merkle root across the aggregate’s events.        |

Staged events are stored in `.eventdbx/staged_events.json`. Use `aggregate apply --stage` to add entries to this queue, inspect them with `aggregate list --stage`, and persist the entire batch with `aggregate commit`. Events are validated against the active schema whenever restriction is `default` or `strict`; the strict mode also insists that a schema exists before anything can be staged. The commit operation writes every pending event in one RocksDB batch, guaranteeing all-or-nothing persistence.

### Plugins

- `dbx plugin install <plugin> <version> --source <path|url> [--bin <file>] [--checksum <sha256>] [--force]`
- `dbx plugin config tcp --name <label> --host <hostname> --port <u16> [--payload <all|event-only|state-only|schema-only|event-and-schema>] [--disable]`
- `dbx plugin config http --name <label> --endpoint <host|url> [--https] [--header KEY=VALUE]... [--payload <all|event-only|state-only|schema-only|event-and-schema>] [--disable]`
- `dbx plugin config log --name <label> --level <trace|debug|info|warn|error> [--template "text with {aggregate} {event} {id}"] [--payload <all|event-only|state-only|schema-only|event-and-schema>] [--disable]`
- `dbx plugin config process --name <instance> --plugin <id> --version <semver> [--arg <value>]... [--env KEY=VALUE]... [--working-dir <path>] [--payload <all|event-only|state-only|schema-only|event-and-schema>] [--disable]`
- `dbx plugin config capnp --name <label> --host <hostname> --port <u16> [--payload <all|event-only|state-only|schema-only|event-and-schema>] [--disable]`
- `dbx plugin enable <label>`
- `dbx plugin disable <label>`
- `dbx plugin remove <label>`
- `dbx plugin test`
- `dbx plugin list`
- `dbx queue`
- `dbx queue clear`
- `dbx queue retry [--event-id <job-id>]`
- `dbx plugin replay <plugin-name> <aggregate> [<aggregate_id>]`

Plugins consume jobs from a durable RocksDB-backed queue. EventDBX enqueues a job for every aggregate mutation, and each plugin can opt into the data it needs—event payloads, materialized state, schemas, or combinations thereof. Clearing dead entries prompts for confirmation to avoid accidental removal. Manual retries run the failed jobs immediately; use `--event-id` to target a specific entry.

### Replication

- `dbx remote add <name> <ip> --token <jwt> [--port <n>]`
  Registers a standby and stores the replication access token issued by the primary. The CLI formats the IP and port into a `tcp://` endpoint (default port `6363`).
- `dbx remote rm <name>`  
  Removes a configured remote.
- `dbx remote ls`  
  Lists remotes with their endpoints.
- `dbx remote show <name>`  
  Displays the endpoint and a shortened token fingerprint for a remote.
- `dbx push <name> [<domain>] [--dry-run] [--schema] [--schema-only] [--batch-size <n>] [--aggregate <type>...] [--aggregate-id <type:id>...]`  
  Streams local events to the remote in fast-forward mode; dry runs report pending changes.
- `dbx pull <name> [<domain>] [--dry-run] [--schema] [--schema-only] [--batch-size <n>] [--aggregate <type>...] [--aggregate-id <type:id>...]`  
  Fast-forwards the local node from the remote, reporting changes in dry-run mode.

Replication relies on JWT access tokens issued through `dbx token issue` (for example, grant `replication.*` or `*.*` actions). Share that token with trusted peers and store it with the remote configuration; it doubles as the symmetric key for the Noise channel that encrypts replication traffic. Pass an optional `<domain>` argument to operate on a different local domain than the currently active one; omit it to inherit the existing checkout. The Cap'n Proto listener that powers CLI automation and replication defaults to `[socket].bind_addr` (default `0.0.0.0:6363`); point remotes at that address with a `tcp://` endpoint or override the bind address in `config.toml` when you expose the replica on another interface. The `dbx push` and `dbx pull` commands connect over this socket—no gRPC listener is required—and every session is authenticated with the presenting token. Use `--aggregate` repeatedly to scope push/pull to specific aggregate types when you only need to sync a subset of data, `--aggregate-id TYPE:ID` to target individual aggregates, `--schema` to copy schema definitions alongside events, and `--schema-only` to synchronize schemas without touching event data.

### Admin API

The Admin API exposes token, schema, remote, and plugin management over HTTP so automation can operate EventDBX without shell access. It is disabled by default; enable it with:

```bash
dbx config \
  --admin-enabled true \
  --admin-bind 127.0.0.1 \
  --admin-port 7171
```

Authorization uses the same JWTs as the CLI. The first server start writes a token with full privileges to `~/.eventdbx/cli.token`; reuse it (or mint a dedicated token with `dbx token generate --group ops --user admin --action '*.*' --resource '*'`) and attach it as a bearer credential:

```bash
curl -H "Authorization: Bearer $(cat ~/.eventdbx/cli.token)" \
  http://127.0.0.1:7171/admin/remotes
```

Key routes:

| Method & Path        | Description                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------------ |
| `GET /admin/tokens`  | List issued tokens; `POST` issues a new token, `/revoke` and `/refresh` manage lifecycle.                    |
| `GET /admin/schemas` | Fetch declared schemas or append new ones with `POST`.                                                       |
| `GET /admin/remotes` | Show replication remotes; `PUT /admin/remotes/{name}` upserts a remote using the same validation as the CLI. |
| `GET /admin/plugins` | Inspect plugin configuration and toggle instances with `/enable` or `/disable`.                              |

Disable access any time with `dbx config --admin-enabled false` or issue a new root token and revoke the old one via `dbx token revoke`.

### Upgrades

- `dbx upgrade [<version>|latest] [--print-only]`  
  Downloads and runs the platform-specific installer to switch binaries. The CLI looks up releases from GitHub, so you can use `latest` or supply a tag like `v1.13.2` (omitting the leading `v` also works). Versions lower than `v1.13.2` are rejected because upgrades are unsupported before that release. Use `--print-only` to show the command without executing it. Shortcut syntax `dbx upgrade@<version>` resolves through the same lookup.
- `dbx upgrade --suppress <version>`  
  Suppresses the upgrade reminder for the provided release tag. Combine it with `latest` to ignore the current release until a newer one ships. Use `dbx upgrade --clear-suppress` to re-enable reminders for all releases.
- `dbx upgrade list [--limit <n>] [--json]`  
  Queries the GitHub releases API and prints the most recent versions. The default limit is 20; use `--json` for a machine-readable list, and call `dbx upgrade@list` for the same shortcut.

> The CLI checks for new releases on startup and prints a reminder when a newer version is available. Set `DBX_NO_UPGRADE_CHECK=1` to bypass the check for automation scenarios.

### Maintenance

- `dbx backup --output <path> [--force]`  
  Creates a compressed archive with the entire EventDBX data directory and configuration. Stop the server before running a backup to avoid partial snapshots.
- `dbx restore --input <path> [--data-dir <path>] [--force]`  
  Restores data from a backup archive. Use `--data-dir` to override the stored location, and `--force` to overwrite non-empty destinations. The server must be stopped before restoring.

Plugins fire after every committed event to keep external systems in sync. Remaining plugins deliver events through different channels:

- **TCP** – Writes a single-line JSON `EventRecord` to the configured socket.
- **HTTP** – POSTs the `EventRecord` JSON to an endpoint with optional headers; add `--https` during configuration to force HTTPS when the endpoint lacks a scheme.
- **gRPC** – Sends `EventRecord` batches to a remote gRPC endpoint compatible with the replication `ApplyEvents` API.
- **Log** – Emits a formatted line via `tracing` at the configured level. By default: `aggregate=<type> id=<id> event=<event>`.
- **Process** – Launches an installed plugin binary as a supervised subprocess and streams events to it over Cap'n Proto. Use this for first-party plugins from the [`dbx_plugins`](https://github.com/thachp/dbx_plugins) workspace or your own extensions.

Process plugins are distributed as zip/tar bundles. Install them with `dbx plugin install <plugin> <version> --source <path-or-url>`—the bundle is unpacked to `~/.eventdbx/plugins/<plugin>/<version>/<target>/`, where `<target>` matches the current OS/architecture (for example, `x86_64-apple-darwin`). Official bundles live in the `dbx_plugins` releases; pass the asset URL to `--source` or point it at a local build while developing. After installation, bind the binary to an instance:

```bash
dbx plugin config process \
  --name search \
  --plugin dbx_search \
  --version 1.0.0 \
  --arg "--listen=0.0.0.0:8081" \
  --env SEARCH_CLUSTER=https://example.com

dbx plugin enable search
```

EventDBX supervises the subprocess, restarts it on failure, and delivers every `EventRecord`/aggregate snapshot over the stream. Plugins that run as independent services can continue to use the TCP/HTTP/GRPC emitters instead.

Failed deliveries are automatically queued and retried with exponential backoff. The server keeps attempting until the plugin succeeds or the aggregate is removed, ensuring transient outages do not drop notifications. Use `dbx queue` to inspect pending/dead event IDs.

Plugin configurations are stored in `.eventdbx/plugins.json`. Each plugin instance requires a unique `--name` so you can update, enable, disable, remove, or replay it later. `plugin enable` validates connectivity (creating directories, touching files, or checking network access) before marking the plugin active. Remove a plugin only after disabling it with `plugin disable <name>`. `plugin replay` resends stored events for a single aggregate instance—or every instance of a type—through the selected plugin.

Need point-in-time snapshots instead of streaming plugins? Use `dbx aggregate export` to capture aggregate state as CSV or JSON on demand.

Example gRPC configuration:

```bash
# point at an existing replication-compatible listener
dbx plugin config grpc \
  --name audit-grpc \
  --endpoint grpc://replica.internal:8800

# enable the plugin once connectivity is confirmed
dbx plugin enable audit-grpc

# inspect status
dbx plugin list
```

Example HTTP/TCP payload (`EventRecord`):

```json
{
  "aggregate_type": "patient",
  "aggregate_id": "p-001",
  "event_type": "patient-updated",
  "payload": {
    "status": "inactive",
    "comment": "Archived via API"
  },
  "extensions": {
    "@analytics": {
      "correlation_id": "rest-1234"
    }
  },
  "metadata": {
    "event_id": "1234567890123",
    "created_at": "2024-12-01T17:22:43.512345Z",
    "issued_by": {
      "group": "admin",
      "user": "jane"
    }
  },
  "version": 5,
  "hash": "cafe…",
  "merkle_root": "deadbeef…"
}
```

> **Heads-up for plugin authors**  
> The [dbx_plugins](https://github.com/thachp/dbx_plugins) surfaces now receive `event_id` values as Snowflake strings and may optionally see an `extensions` object alongside `payload`. Update custom handlers to treat `metadata.event_id` as a stringified Snowflake and to ignore or consume the new `extensions` envelope as needed.

## Column definitions

Each entry in the `column_types` map declares both a storage type and the rules EventDBX enforces. Supported types:

| Type                       | Accepted input                                                 | Notes                                                                                                   |
| -------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `integer`                  | JSON numbers or strings that parse to a signed 64-bit integer  | Rejects values outside the i64 range.                                                                   |
| `float`                    | JSON numbers or numeric strings                                | Stored as `f64`; scientific notation is accepted.                                                       |
| `decimal(precision,scale)` | JSON numbers or strings                                        | Enforces total digits ≤ precision and fractional digits ≤ scale.                                        |
| `boolean`                  | JSON booleans, `0` / `1`, or `"true"`, `"false"`, `"1"`, `"0"` | Values are normalised to `true` / `false`.                                                              |
| `text`                     | UTF-8 strings                                                  | Use `length`, `contains`, or `regex` rules for additional constraints.                                  |
| `timestamp`                | RFC 3339 timestamps as strings                                 | Normalised to UTC.                                                                                      |
| `date`                     | `YYYY-MM-DD` strings                                           | Parsed as a calendar date without a timezone.                                                           |
| `json`                     | Any JSON value                                                 | No per-field validation is applied; use when you want to store free-form payloads.                      |
| `binary`                   | Base64-encoded strings                                         | Decoded to raw bytes before validation; `length` counts bytes after decoding.                           |
| `object`                   | JSON objects                                                   | Enable nested validation via the `properties` rule (see below). Extra keys not listed remain untouched. |

Rules are optional and can be combined when the target type supports them:

- `required`: the field must be present in every event payload.
- `contains` / `does_not_contain`: case-sensitive substring checks for `text` fields.
- `regex`: one or more regular expressions that `text` fields must satisfy.
- `format`: built-in string validators; choose `email`, `url`, `credit_card`, `country_code` (ISO&nbsp;3166-1 alpha-2), `iso_8601` (RFC&nbsp;3339 timestamp), `wgs_84` (latitude/longitude in decimal degrees), `camel_case`, `snake_case`, `kebab_case`, `pascal_case`, or `upper_case_snake_case`.
- `length`: `{ "min": <usize>, "max": <usize> }` bounds the length of `text` (characters) or `binary` (decoded bytes).
- `range`: `{ "min": <value>, "max": <value> }` for numeric and temporal types (`integer`, `float`, `decimal`, `timestamp`, `date`). Boundary values must parse to the column’s type.
- `properties`: nested `column_types` definitions for `object` columns, enabling recursion with the same rule set as top-level fields.

## Performance Testing

Performance benchmarks and workload scenarios live in the [eventdbx-perf repository](https://github.com/thachp/eventdbx-perf). Clone that project to review the current load profiles, adjust parameters, or submit new variations as you evaluate throughput and latency trade-offs.

All tests executed on the same host using Docker-based databases, single-thread client, datasets up to 10 M records.
Latency ≈ mean operation time; throughput = operations per second (converted from ns → µs).

| Engine              | Typical Throughput (ops/s) | Typical Latency (µs) |       Scaling Trend        | Summary & Observation                                                                                                                                                  |
| :------------------ | :------------------------: | :------------------: | :------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **EventDBX**        |     **1 400 – 2 000**      |    **0.5 – 0.8**     |   **Flat (1 K → 10 M)**    | RocksDB-based append-only core with hot-aggregate caching keeps performance nearly constant. Excels at write-heavy, event-sourced workloads with verifiable integrity. |
| **PostgreSQL 15**   |       1 000 – 1 900        |      0.6 – 1.0       | Stable / slightly variable | Strong transactional baseline; planner + WAL sync add moderate overhead. Excellent for mixed OLTP queries but heavier per-event cost.                                  |
| **MongoDB 7**       |        400 – 1 000         |      1.5 – 2.5       | Gradual decline with size  | Flexible JSON-document store with moderate efficiency. Serialization and journaling add overhead; roughly half of EventDBX throughput.                                 |
| **SQL Server 2022** |          50 – 180          |        5 – 20        |   Drops quickly > 100 K    | High latency and lowest throughput; B-tree structure and lock coordination dominate under load. Best suited for transactional consistency, not high-velocity writes.   |

EventDBX delivers the highest sustained throughput and lowest latency, outperforming PostgreSQL by ~2×, MongoDB by ~3×, and SQL Server by >10×—while maintaining sub-microsecond response times even at multi-million-record scale.

## Contributing

### **Quick Start**

1. **Report Issues**: Found a bug or have a suggestion? [Open an issue](https://chat.openai.com/g/g-kvXdAN8VA-eventdbx-guide/c/72d2fa39-c3e2-4e40-bd61-5694f7b82aab#) with detailed information.
2. **Contribute Code**:
   - **Fork & Clone**: Fork the EventDBX repo and clone your fork.
   - **Branch**: Create a branch for your changes from **`develop`**.
   - **Develop**: Make your changes, adhering to our coding standards. Add or update tests as necessary.
   - **Commit**: Use [Conventional Commits](https://www.conventionalcommits.org/) for clear, structured commit messages (e.g., **`feat: add new feature`** or **`fix: correct a bug`**).
   - **Pull Request**: Submit a pull request (PR) against the **`develop`** branch of the original repo. Describe your changes and link to any related issues.

### **Guidelines**

- **Formatting**: A project-wide Prettier configuration lives at `.prettierrc.json`; use it for Markdown/JSON/YAML changes.
- **Commit linting**: Conventional Commit headers are enforced through `.commitlintrc.json`.
- **Code Review**: Your PR will be reviewed by the team. Be open to feedback and make requested adjustments.
- **Merge**: Once approved, your PR will be merged into the project, and you'll be credited as a contributor.

## License

---

© 2025 Patrick Thach and contributors

EventDBX is open source software licensed under the [MIT License](./LICENSE).  
See the LICENSE file in the repository for details.
