use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Subcommand, ValueEnum};
use csv::WriterBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value, json};
use tempfile::tempdir;
use zip::{CompressionMethod, ZipWriter, write::FileOptions};

use eventdbx::{
    config::{Config, PluginPayloadMode, load_or_default},
    error::EventError,
    filter,
    merkle::compute_merkle_root,
    plugin::{JobPriority, PluginManager, PublishTarget},
    reference::{
        AggregateReference, ReferenceCascade, ReferenceContext, ReferenceFetchOutcome,
        ReferenceIntegrity, ReferenceResolutionStatus, Referrer, resolve_references,
    },
    restrict,
    schema::{MAX_EVENT_NOTE_LENGTH, SchemaManager},
    store::{
        self, ActorClaims, AggregateCursor, AggregateQueryScope, AggregateSort, AggregateState,
        AppendEvent, EventRecord, EventStore, payload_to_map, select_state_field,
    },
    tenant_store::{BYTES_PER_MEGABYTE, TenantAssignmentStore},
    token::{IssueTokenInput, JwtLimits, ROOT_ACTION, ROOT_RESOURCE, TokenManager},
    validation::{
        ensure_aggregate_id, ensure_first_event_rule, ensure_metadata_extensions,
        ensure_payload_size, ensure_snake_case,
    },
};

#[cfg(test)]
use eventdbx::restrict::RestrictMode;

use crate::commands::{cli_token, client::ServerClient, is_lock_error_message};
use tracing::warn;

#[derive(Clone, Debug)]
pub struct PublishTargetArg {
    pub plugin: String,
    pub mode: Option<PluginPayloadMode>,
    pub priority: Option<JobPriority>,
}

#[derive(Subcommand)]
pub enum AggregateCommands {
    /// Create a new aggregate instance
    Create(AggregateCreateArgs),
    /// Apply an event to an aggregate instance
    Apply(AggregateApplyArgs),
    /// Apply a JSON Patch event to an aggregate instance
    Patch(AggregatePatchArgs),
    /// List aggregates in the store
    List(AggregateListArgs),
    /// Select specific fields from an aggregate state
    Select(AggregateSelectArgs),
    /// Retrieve the state of an aggregate
    Get(AggregateGetArgs),
    /// Verify an aggregate's Merkle root
    Verify(AggregateVerifyArgs),
    /// Archive an aggregate instance
    Archive(AggregateArchiveArgs),
    /// Restore an archived aggregate instance
    Restore(AggregateArchiveArgs),
    /// Remove an aggregate that has no events
    Remove(AggregateRemoveArgs),
    /// List aggregates that reference a target aggregate
    Referrers(AggregateReferrersArgs),
    /// Commit events previously staged with `aggregate apply --stage`
    Commit,
    /// Export aggregate state to CSV or JSON
    Export(AggregateExportArgs),
}

#[derive(Args)]
pub struct AggregateCreateArgs {
    /// Aggregate type
    pub aggregate: String,

    /// Aggregate identifier
    pub aggregate_id: String,

    /// Event type to append as the initial write
    #[arg(long)]
    pub event: String,

    /// Event fields expressed as KEY=VALUE pairs
    #[arg(long = "field", value_parser = parse_key_value, value_name = "KEY=VALUE")]
    pub fields: Vec<KeyValue>,

    /// Raw JSON payload to use instead of key-value fields
    #[arg(long, value_name = "JSON")]
    pub payload: Option<String>,

    /// JSON metadata with plugin-specific keys prefixed by '@'
    #[arg(long)]
    pub metadata: Option<String>,

    /// Optional note associated with the event (up to 128 characters)
    #[arg(long, value_name = "NOTE")]
    pub note: Option<String>,

    /// Authorization token used when proxying through a running server
    #[arg(long, value_name = "TOKEN")]
    pub token: Option<String>,

    /// Emit results as JSON
    #[arg(long, default_value_t = false)]
    pub json: bool,

    /// Explicit plugin publish targets as PLUGIN[:MODE[:PRIORITY]]
    #[arg(long, value_parser = parse_publish_target, value_name = "PLUGIN[:MODE[:PRIORITY]]")]
    pub publish: Vec<PublishTargetArg>,
}

#[derive(Args)]
pub struct AggregateApplyArgs {
    /// Aggregate type
    pub aggregate: String,

    /// Aggregate identifier
    pub aggregate_id: String,

    /// Event type to append
    pub event: String,

    /// Event fields expressed as KEY=VALUE pairs
    #[arg(long = "field", value_parser = parse_key_value, value_name = "KEY=VALUE")]
    pub fields: Vec<KeyValue>,

    /// Stage the event for a later commit instead of writing immediately
    #[arg(long, default_value_t = false)]
    pub stage: bool,

    /// Authorization token used when proxying through a running server
    #[arg(long, value_name = "TOKEN")]
    pub token: Option<String>,

    /// Raw JSON payload to use instead of key-value fields
    #[arg(long, value_name = "JSON")]
    pub payload: Option<String>,

    /// JSON metadata with plugin-specific keys prefixed by '@'
    #[arg(long)]
    pub metadata: Option<String>,

    /// Optional note associated with the event (up to 128 characters)
    #[arg(long, value_name = "NOTE")]
    pub note: Option<String>,

    /// Explicit plugin publish targets as PLUGIN[:MODE[:PRIORITY]]
    #[arg(long, value_parser = parse_publish_target, value_name = "PLUGIN[:MODE[:PRIORITY]]")]
    pub publish: Vec<PublishTargetArg>,
}

#[derive(Args)]
pub struct AggregatePatchArgs {
    /// Aggregate type
    pub aggregate: String,

    /// Aggregate identifier
    pub aggregate_id: String,

    /// Event type to append
    pub event: String,

    /// Stage the patch for a later commit instead of writing immediately
    #[arg(long, default_value_t = false)]
    pub stage: bool,

    /// Authorization token used when proxying through a running server
    #[arg(long, value_name = "TOKEN")]
    pub token: Option<String>,

    /// JSON Patch (RFC 6902) document to apply server-side
    #[arg(long, value_name = "JSON")]
    pub patch: String,

    /// JSON metadata with plugin-specific keys prefixed by '@'
    #[arg(long)]
    pub metadata: Option<String>,

    /// Optional note associated with the event (up to 128 characters)
    #[arg(long, value_name = "NOTE")]
    pub note: Option<String>,

    /// Explicit plugin publish targets as PLUGIN[:MODE[:PRIORITY]] for this patch
    #[arg(long, value_parser = parse_publish_target, value_name = "PLUGIN[:MODE[:PRIORITY]]")]
    pub publish: Vec<PublishTargetArg>,
}

#[derive(Args)]
pub struct AggregateGetArgs {
    /// Aggregate type
    pub aggregate: String,

    /// Aggregate identifier
    pub aggregate_id: String,

    /// Optional version to compute state at
    #[arg(long)]
    pub version: Option<u64>,

    /// Include event history in the output
    #[arg(long, default_value_t = false)]
    pub include_events: bool,

    /// Resolve reference fields and include resolved payloads
    #[arg(long, default_value_t = false)]
    pub resolve: bool,

    /// Resolution depth (defaults to config reference_default_depth; capped at reference_max_depth)
    #[arg(long = "resolve-depth")]
    pub resolve_depth: Option<usize>,
}

#[derive(Args)]
pub struct AggregateSelectArgs {
    /// Aggregate type
    pub aggregate: String,

    /// Aggregate identifier
    pub aggregate_id: String,

    /// Field paths expressed as dot-delimited keys (arrays use numeric indices)
    #[arg(value_name = "FIELD", num_args = 1..)]
    pub fields: Vec<String>,
}

#[derive(Args)]
pub struct AggregateVerifyArgs {
    /// Aggregate type
    pub aggregate: String,

    /// Aggregate identifier
    pub aggregate_id: String,

    /// Emit results as JSON
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

#[derive(Args)]
pub struct AggregateArchiveArgs {
    /// Aggregate type
    pub aggregate: String,

    /// Aggregate identifier
    pub aggregate_id: String,

    /// Optional comment recorded with the action
    #[arg(long)]
    pub comment: Option<String>,
}

#[derive(Debug, Clone)]
pub struct KeyValue {
    pub key: String,
    pub value: String,
}

#[derive(Args)]
pub struct AggregateRemoveArgs {
    /// Aggregate type
    pub aggregate: String,

    /// Aggregate identifier
    pub aggregate_id: String,
}

#[derive(Args)]
pub struct AggregateReferrersArgs {
    /// Aggregate type to inspect for inbound references
    pub aggregate: String,

    /// Aggregate identifier to inspect for inbound references
    pub aggregate_id: String,

    /// Authorization token used when proxying through a running server
    #[arg(long, value_name = "TOKEN")]
    pub token: Option<String>,

    /// Emit results as JSON
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

#[derive(Args)]
pub struct AggregateListArgs {
    /// Limit results to a single aggregate type
    #[arg(value_name = "AGGREGATE")]
    pub aggregate: Option<String>,

    /// Resume listing after the provided cursor (`a:type:id`)
    #[arg(long)]
    pub cursor: Option<String>,

    /// Maximum number of aggregates to return
    #[arg(long)]
    pub take: Option<usize>,

    /// Show staged events instead of persisted aggregates
    #[arg(long, default_value_t = false)]
    pub stage: bool,

    /// Emit results as JSON
    #[arg(long, default_value_t = false)]
    pub json: bool,

    /// Filter aggregates using a SQL-like expression (e.g. `last_name = "thach"`)
    #[arg(long)]
    pub filter: Option<String>,

    /// Sort aggregates by comma-separated fields (e.g. `aggregate_type:asc,updated_at:desc`)
    #[arg(long, value_name = "FIELD[:ORDER][,...]")]
    pub sort: Option<String>,

    /// Include archived aggregates alongside active ones
    #[arg(long, default_value_t = false, conflicts_with = "archived_only")]
    pub include_archived: bool,

    /// Show only archived aggregates
    #[arg(long, default_value_t = false, conflicts_with = "include_archived")]
    pub archived_only: bool,

    /// Resolve reference fields for each aggregate (JSON output recommended)
    #[arg(long, default_value_t = false)]
    pub resolve: bool,

    /// Resolution depth (defaults to config reference_default_depth; capped at reference_max_depth)
    #[arg(long = "resolve-depth")]
    pub resolve_depth: Option<usize>,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum AggregateExportFormat {
    Csv,
    Json,
}

#[derive(Args)]
pub struct AggregateExportArgs {
    /// Aggregate type to export (omit when using --all)
    pub aggregate: Option<String>,

    /// Export every aggregate type
    #[arg(long)]
    pub all: bool,

    /// Output format
    #[arg(long, value_enum, default_value_t = AggregateExportFormat::Csv)]
    pub format: AggregateExportFormat,

    /// Output path (file or directory depending on context)
    #[arg(long)]
    pub output: PathBuf,

    /// Compress the export into a ZIP archive
    #[arg(long, default_value_t = false)]
    pub zip: bool,

    /// Pretty-print JSON output (no effect for CSV)
    #[arg(long, default_value_t = false)]
    pub pretty: bool,
}

#[derive(Clone)]
struct ExportRecord {
    aggregate_id: String,
    state: BTreeMap<String, String>,
}

const EXPORT_ID_KEY: &str = "__aggregate_id";

pub fn execute(config_path: Option<PathBuf>, command: AggregateCommands) -> Result<()> {
    let (config, _) = load_or_default(config_path)?;
    match command {
        AggregateCommands::Create(args) => {
            let AggregateCreateArgs {
                aggregate,
                aggregate_id,
                event,
                fields,
                payload: payload_arg,
                metadata,
                note,
                token,
                json,
                publish,
            } = args;

            if payload_arg.is_some() && !fields.is_empty() {
                bail!("--payload cannot be used together with --field");
            }

            let payload = if let Some(raw) = payload_arg {
                serde_json::from_str::<Value>(&raw)
                    .with_context(|| "failed to parse JSON payload provided via --payload")?
            } else {
                collect_payload(fields)
            };

            let metadata_value = match metadata {
                Some(raw) => Some(
                    serde_json::from_str::<Value>(&raw)
                        .with_context(|| "failed to parse JSON metadata provided via --metadata")?,
                ),
                None => None,
            };

            let command = CreateCommand {
                aggregate,
                aggregate_id,
                event,
                payload,
                metadata: metadata_value,
                note,
                token,
                json,
                publish,
            };

            execute_create_command(&config, command)?;
        }
        AggregateCommands::List(args) => {
            let aggregate_filter = args.aggregate.as_deref();
            if let Some(name) = aggregate_filter {
                ensure_snake_case("aggregate_type", name)?;
            }

            if args.stage {
                let staging_path = config.staging_path();
                let staged_events = load_staged_events(staging_path.as_path())?;
                let filtered: Vec<_> = staged_events
                    .into_iter()
                    .filter(|event| {
                        aggregate_filter
                            .map(|target| event.aggregate == target)
                            .unwrap_or(true)
                    })
                    .collect();
                if filtered.is_empty() {
                    println!("no staged events");
                } else {
                    for event in filtered {
                        println!("{}", serde_json::to_string_pretty(&event)?);
                    }
                }
                return Ok(());
            }

            let store =
                EventStore::open_read_only(config.event_store_path(), config.encryption_key()?)?;
            if args.resolve && !args.json {
                bail!("--resolve requires --json output for aggregate listings");
            }
            let tenant = config.active_domain().to_string();
            let resolve_depth = args
                .resolve_depth
                .unwrap_or(config.reference_default_depth)
                .min(config.reference_max_depth);
            let schemas_path = config.schema_store_path();
            let mut schemas_cache: Option<SchemaManager> = None;
            let mut print_json = |aggregates: Vec<AggregateState>| -> Result<()> {
                if !args.resolve {
                    println!("{}", serde_json::to_string_pretty(&aggregates)?);
                    return Ok(());
                }

                let schemas = if let Some(ref manager) = schemas_cache {
                    manager
                } else {
                    schemas_cache = Some(SchemaManager::load(schemas_path.clone())?);
                    schemas_cache.as_ref().expect("schema cache initialized")
                };

                let resolved: Vec<_> = aggregates
                    .into_iter()
                    .map(|aggregate| {
                        let resolved = resolve_references(
                            tenant.clone(),
                            aggregate,
                            schemas,
                            resolve_depth,
                            |reference| {
                                if !reference.domain.eq_ignore_ascii_case(&tenant) {
                                    return Ok(ReferenceFetchOutcome {
                                        status: ReferenceResolutionStatus::Forbidden,
                                        aggregate: None,
                                    });
                                }
                                match store.get_aggregate_state(
                                    &reference.aggregate_type,
                                    &reference.aggregate_id,
                                ) {
                                    Ok(agg) => Ok(ReferenceFetchOutcome {
                                        status: ReferenceResolutionStatus::Ok,
                                        aggregate: Some(agg),
                                    }),
                                    Err(EventError::AggregateNotFound) => {
                                        Ok(ReferenceFetchOutcome {
                                            status: ReferenceResolutionStatus::NotFound,
                                            aggregate: None,
                                        })
                                    }
                                    Err(err) => Err(err),
                                }
                            },
                        )?;
                        serde_json::to_value(resolved).map_err(Into::into)
                    })
                    .collect::<Result<Vec<_>>>()?;

                println!("{}", serde_json::to_string_pretty(&resolved)?);
                Ok(())
            };
            let mut filter_expr = match args.filter.as_ref() {
                Some(raw) => Some(
                    filter::parse_shorthand(raw)
                        .with_context(|| format!("invalid filter expression: {raw}"))?,
                ),
                None => None,
            };

            if let Some(name) = aggregate_filter {
                filter_expr = match filter_expr {
                    Some(existing) => Some(filter::FilterExpr::And(vec![
                        filter::FilterExpr::Comparison {
                            field: "aggregate_type".to_string(),
                            op: filter::ComparisonOp::Equals(filter::FilterValue::String(
                                name.to_string(),
                            )),
                        },
                        existing,
                    ])),
                    None => Some(filter::FilterExpr::Comparison {
                        field: "aggregate_type".to_string(),
                        op: filter::ComparisonOp::Equals(filter::FilterValue::String(
                            name.to_string(),
                        )),
                    }),
                };
            }
            let mut scope = if args.archived_only {
                AggregateQueryScope::ArchivedOnly
            } else if args.include_archived {
                AggregateQueryScope::IncludeArchived
            } else {
                AggregateQueryScope::ActiveOnly
            };

            if matches!(scope, AggregateQueryScope::ActiveOnly) {
                if let Some(expr) = filter_expr.as_ref() {
                    if expr.references_field("archived") {
                        scope = AggregateQueryScope::IncludeArchived;
                    }
                }
            }

            let sort_directives = if let Some(spec) = args.sort.as_deref() {
                Some(
                    AggregateSort::parse_directives(spec)
                        .map_err(|err| anyhow!("invalid sort specification: {err}"))?,
                )
            } else {
                None
            };
            let sort_keys = sort_directives.as_ref().map(|keys| keys.as_slice());
            let timestamp_sort = sort_keys.and_then(store::timestamp_sort_hint);
            let take = args.take.unwrap_or(config.list_page_size);
            if take == 0 {
                bail!("--take must be greater than zero");
            }

            let cursor_token = match args.cursor.as_deref() {
                Some(raw) => {
                    let cursor = if let Some((kind, descending)) = timestamp_sort {
                        store
                            .parse_timestamp_cursor(raw, kind, descending)
                            .with_context(|| format!("invalid cursor '{raw}'"))?
                    } else {
                        AggregateCursor::from_str(raw)
                            .with_context(|| format!("invalid cursor '{raw}'"))?
                    };
                    Some(cursor)
                }
                None => None,
            };
            if sort_directives.is_some() && timestamp_sort.is_none() && cursor_token.is_some() {
                bail!(
                    "--cursor can only be combined with --sort when sorting by created_at or updated_at"
                );
            }
            if sort_directives.is_none() {
                if let Some(token) = cursor_token.as_ref() {
                    if token.is_timestamp() {
                        bail!("timestamp cursors require --sort created_at or updated_at");
                    }
                }
            }

            if let Some(keys) = sort_keys {
                if let Some((kind, descending)) = timestamp_sort {
                    if let Some(cursor) = cursor_token.as_ref() {
                        store::ensure_timestamp_cursor(cursor, kind, descending, scope)?;
                    }
                    let aggregates = store.aggregates_paginated_with_transform(
                        0,
                        Some(take),
                        Some(keys),
                        scope,
                        cursor_token.as_ref(),
                        |aggregate| {
                            if let Some(expr) = filter_expr.as_ref() {
                                if !expr.matches_aggregate(&aggregate) {
                                    return None;
                                }
                            }
                            Some(aggregate)
                        },
                    );
                    if args.json {
                        print_json(aggregates)?;
                    } else {
                        let show_archived = matches!(
                            scope,
                            AggregateQueryScope::IncludeArchived
                                | AggregateQueryScope::ArchivedOnly
                        );
                        for aggregate in aggregates {
                            if show_archived {
                                println!(
                                    "aggregate_type={} aggregate_id={} version={} merkle_root={} archived={}",
                                    aggregate.aggregate_type,
                                    aggregate.aggregate_id,
                                    aggregate.version,
                                    aggregate.merkle_root,
                                    aggregate.archived
                                );
                            } else {
                                println!(
                                    "aggregate_type={} aggregate_id={} version={} merkle_root={}",
                                    aggregate.aggregate_type,
                                    aggregate.aggregate_id,
                                    aggregate.version,
                                    aggregate.merkle_root
                                );
                            }
                        }
                    }
                    return Ok(());
                }

                let aggregates = store.aggregates_paginated_with_transform(
                    0,
                    Some(take),
                    Some(keys),
                    scope,
                    None,
                    |aggregate| {
                        if let Some(expr) = filter_expr.as_ref() {
                            if !expr.matches_aggregate(&aggregate) {
                                return None;
                            }
                        }
                        Some(aggregate)
                    },
                );
                if args.json {
                    print_json(aggregates)?;
                } else {
                    let show_archived = matches!(
                        scope,
                        AggregateQueryScope::IncludeArchived | AggregateQueryScope::ArchivedOnly
                    );
                    for aggregate in aggregates {
                        if show_archived {
                            println!(
                                "aggregate_type={} aggregate_id={} version={} merkle_root={} archived={}",
                                aggregate.aggregate_type,
                                aggregate.aggregate_id,
                                aggregate.version,
                                aggregate.merkle_root,
                                aggregate.archived
                            );
                        } else {
                            println!(
                                "aggregate_type={} aggregate_id={} version={} merkle_root={}",
                                aggregate.aggregate_type,
                                aggregate.aggregate_id,
                                aggregate.version,
                                aggregate.merkle_root
                            );
                        }
                    }
                }
                return Ok(());
            }

            let (aggregates, _next_cursor) = store.aggregates_page_with_transform(
                cursor_token.as_ref(),
                take,
                scope,
                |aggregate| {
                    if let Some(expr) = filter_expr.as_ref() {
                        if !expr.matches_aggregate(&aggregate) {
                            return None;
                        }
                    }
                    Some(aggregate)
                },
            )?;

            if args.json {
                print_json(aggregates)?;
            } else {
                let show_archived = matches!(
                    scope,
                    AggregateQueryScope::IncludeArchived | AggregateQueryScope::ArchivedOnly
                );
                for aggregate in aggregates {
                    if show_archived {
                        println!(
                            "aggregate_type={} aggregate_id={} version={} merkle_root={} archived={}",
                            aggregate.aggregate_type,
                            aggregate.aggregate_id,
                            aggregate.version,
                            aggregate.merkle_root,
                            aggregate.archived
                        );
                    } else {
                        println!(
                            "aggregate_type={} aggregate_id={} version={} merkle_root={}",
                            aggregate.aggregate_type,
                            aggregate.aggregate_id,
                            aggregate.version,
                            aggregate.merkle_root
                        );
                    }
                }
            }
        }
        AggregateCommands::Remove(args) => {
            let store = EventStore::open(
                config.event_store_path(),
                config.encryption_key()?,
                config.snowflake_worker_id,
            )?;
            let schemas = SchemaManager::load(config.schema_store_path())?;
            let target = format!(
                "{}#{}#{}",
                config.active_domain(),
                args.aggregate,
                args.aggregate_id
            );
            let referrers = store.referrers_for_any_tenant(&target)?;
            let (nullify, blockers) =
                partition_referrers(&schemas, config.active_domain(), &referrers);
            if !blockers.is_empty() {
                bail!(
                    "aggregate {}:{} cannot be removed; referenced by {}",
                    args.aggregate,
                    args.aggregate_id,
                    blockers.into_iter().take(5).collect::<Vec<_>>().join(", ")
                );
            }
            nullify_referrers_offline(&store, &schemas, config.active_domain(), &target, &nullify)?;
            store.remove_aggregate(&args.aggregate, &args.aggregate_id)?;
            let _ = store.clear_reference_index(
                config.active_domain(),
                &args.aggregate,
                &args.aggregate_id,
            );
            let assignments = TenantAssignmentStore::open(config.tenant_meta_path())?;
            let usage = store.storage_usage_bytes()?;
            assignments.update_storage_usage_bytes(config.active_domain(), usage)?;
            println!(
                "aggregate_type={} aggregate_id={} removed",
                args.aggregate, args.aggregate_id
            );
        }
        AggregateCommands::Referrers(args) => {
            let AggregateReferrersArgs {
                aggregate,
                aggregate_id,
                token,
                json,
            } = args;
            let tenant = config.active_domain().to_string();
            let target = format!("{}#{}#{}", tenant, aggregate, aggregate_id);
            let refs = match EventStore::open_read_only(
                config.event_store_path(),
                config.encryption_key()?,
            ) {
                Ok(store) => store
                    .referrers_for_any_tenant(&target)?
                    .into_iter()
                    .map(|(_, aggregate_type, aggregate_id, path)| Referrer {
                        aggregate_type,
                        aggregate_id,
                        path,
                    })
                    .collect(),
                Err(EventError::Storage(message)) if is_lock_error_message(&message) => {
                    let token = ensure_proxy_token(&config, token)?;
                    let client = ServerClient::new(&config)?;
                    client.list_referrers(&token, &aggregate, &aggregate_id)?
                }
                Err(err) => return Err(err.into()),
            };

            if json {
                println!("{}", serde_json::to_string_pretty(&refs)?);
            } else if refs.is_empty() {
                println!("no referrers found");
            } else {
                for reference in refs {
                    println!(
                        "aggregate_type={} aggregate_id={} path={}",
                        reference.aggregate_type, reference.aggregate_id, reference.path
                    );
                }
            }
        }
        AggregateCommands::Get(args) => {
            let store =
                EventStore::open_read_only(config.event_store_path(), config.encryption_key()?)?;
            let mut state = store.get_aggregate_state(&args.aggregate, &args.aggregate_id)?;
            let mut events_cache = None;

            if args.version.is_some() || args.include_events {
                events_cache = Some(store.list_events(&args.aggregate, &args.aggregate_id)?);
            }

            if let Some(version) = args.version {
                let events = events_cache
                    .as_ref()
                    .expect("events cache should be populated");
                let (target_version, target_state, merkle_root) = state_at_version(events, version);
                state.version = target_version;
                state.state = target_state;
                state.merkle_root = merkle_root;
            }

            let mut output = serde_json::json!({
                "aggregate_type": state.aggregate_type,
                "aggregate_id": state.aggregate_id,
                "version": state.version,
                "state": state.state,
                "merkle_root": state.merkle_root,
                "updated_at": state.updated_at,
                "created_at": state.created_at
            });

            if args.include_events {
                let events = match events_cache.take() {
                    Some(events) => events,
                    None => store.list_events(&args.aggregate, &args.aggregate_id)?,
                };
                let filtered: Vec<_> = match args.version {
                    Some(version) => events
                        .into_iter()
                        .filter(|event| event.version <= version)
                        .collect(),
                    None => events,
                };
                output["events"] = serde_json::to_value(filtered)?;
            }

            if args.resolve {
                let schemas = SchemaManager::load(config.schema_store_path())?;
                let tenant = config.active_domain().to_string();
                let depth = args
                    .resolve_depth
                    .unwrap_or(config.reference_default_depth)
                    .min(config.reference_max_depth);
                let resolved = resolve_references(
                    tenant.clone(),
                    state.clone(),
                    &schemas,
                    depth,
                    |reference| {
                        if !reference.domain.eq_ignore_ascii_case(&tenant) {
                            return Ok(ReferenceFetchOutcome {
                                status: ReferenceResolutionStatus::Forbidden,
                                aggregate: None,
                            });
                        }
                        match store
                            .get_aggregate_state(&reference.aggregate_type, &reference.aggregate_id)
                        {
                            Ok(agg) => Ok(ReferenceFetchOutcome {
                                status: ReferenceResolutionStatus::Ok,
                                aggregate: Some(agg),
                            }),
                            Err(EventError::AggregateNotFound) => Ok(ReferenceFetchOutcome {
                                status: ReferenceResolutionStatus::NotFound,
                                aggregate: None,
                            }),
                            Err(err) => Err(err),
                        }
                    },
                )?;
                output["resolved"] = serde_json::to_value(resolved)?;
            }

            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        AggregateCommands::Apply(args) => {
            let AggregateApplyArgs {
                aggregate,
                aggregate_id,
                event,
                fields,
                stage,
                token,
                payload: payload_arg,
                metadata,
                note,
                publish,
            } = args;
            if payload_arg.is_some() && !fields.is_empty() {
                bail!("--payload cannot be used together with --field");
            }
            let payload_value = if let Some(raw) = payload_arg {
                Some(
                    serde_json::from_str(&raw)
                        .with_context(|| "failed to parse JSON payload provided via --payload")?,
                )
            } else {
                Some(collect_payload(fields))
            };
            let metadata_value = match metadata {
                Some(raw) => Some(
                    serde_json::from_str::<Value>(&raw)
                        .with_context(|| "failed to parse JSON metadata provided via --metadata")?,
                ),
                None => None,
            };

            let command = AppendCommand {
                aggregate,
                aggregate_id,
                event,
                stage,
                token,
                payload: payload_value,
                patch: None,
                metadata: metadata_value,
                note,
                publish,
            };

            execute_append_command(&config, command)?;
        }
        AggregateCommands::Patch(args) => {
            let AggregatePatchArgs {
                aggregate,
                aggregate_id,
                event,
                stage,
                token,
                patch,
                metadata,
                note,
                publish,
            } = args;

            let patch_value = serde_json::from_str::<Value>(&patch)
                .with_context(|| "failed to parse JSON patch provided via --patch")?;
            let metadata_value = match metadata {
                Some(raw) => Some(
                    serde_json::from_str::<Value>(&raw)
                        .with_context(|| "failed to parse JSON metadata provided via --metadata")?,
                ),
                None => None,
            };

            let command = AppendCommand {
                aggregate,
                aggregate_id,
                event,
                stage,
                token,
                payload: None,
                patch: Some(patch_value),
                metadata: metadata_value,
                note,
                publish,
            };

            execute_append_command(&config, command)?;
        }
        AggregateCommands::Select(args) => {
            let AggregateSelectArgs {
                aggregate,
                aggregate_id,
                fields,
            } = args;

            let store =
                EventStore::open_read_only(config.event_store_path(), config.encryption_key()?)?;
            let state = store.get_aggregate_state(&aggregate, &aggregate_id)?;

            let mut selection = JsonMap::new();
            for field in fields {
                let value = select_state_field(&state.state, &field).unwrap_or(Value::Null);
                selection.insert(field, value);
            }

            let output = serde_json::json!({
                "aggregate_type": state.aggregate_type,
                "aggregate_id": state.aggregate_id,
                "version": state.version,
                "selection": selection,
            });

            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        AggregateCommands::Verify(args) => {
            let store =
                EventStore::open_read_only(config.event_store_path(), config.encryption_key()?)?;
            let merkle_root = store.verify(&args.aggregate, &args.aggregate_id)?;
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "aggregate_type": args.aggregate,
                        "aggregate_id": args.aggregate_id,
                        "merkle_root": merkle_root,
                    }))?
                );
            } else {
                println!(
                    "aggregate_type={} aggregate_id={} merkle_root={}",
                    args.aggregate, args.aggregate_id, merkle_root
                );
            }
        }
        AggregateCommands::Archive(args) => {
            let store = EventStore::open(
                config.event_store_path(),
                config.encryption_key()?,
                config.snowflake_worker_id,
            )?;
            let schemas = SchemaManager::load(config.schema_store_path())?;
            let target = format!(
                "{}#{}#{}",
                config.active_domain(),
                args.aggregate,
                args.aggregate_id
            );
            let referrers = store.referrers_for_any_tenant(&target)?;
            let (nullify, blockers) =
                partition_referrers(&schemas, config.active_domain(), &referrers);
            if !blockers.is_empty() {
                bail!(
                    "aggregate {}:{} cannot be archived; referenced by {}",
                    args.aggregate,
                    args.aggregate_id,
                    blockers.into_iter().take(5).collect::<Vec<_>>().join(", ")
                );
            }
            nullify_referrers_offline(&store, &schemas, config.active_domain(), &target, &nullify)?;
            let meta = store.set_archive(
                &args.aggregate,
                &args.aggregate_id,
                true,
                args.comment.clone(),
            )?;
            if config.verbose_responses() {
                println!(
                    "aggregate_type={} aggregate_id={} archived={} comment={}",
                    meta.aggregate_type,
                    meta.aggregate_id,
                    meta.archived,
                    args.comment.unwrap_or_default()
                );
            } else {
                println!("Ok");
            }
        }
        AggregateCommands::Restore(args) => {
            let store = EventStore::open(
                config.event_store_path(),
                config.encryption_key()?,
                config.snowflake_worker_id,
            )?;
            let meta = store.set_archive(
                &args.aggregate,
                &args.aggregate_id,
                false,
                args.comment.clone(),
            )?;
            if config.verbose_responses() {
                println!(
                    "aggregate_type={} aggregate_id={} archived={} comment={}",
                    meta.aggregate_type,
                    meta.aggregate_id,
                    meta.archived,
                    args.comment.unwrap_or_default()
                );
            } else {
                println!("Ok");
            }
        }
        AggregateCommands::Commit => {
            let staging_path = config.staging_path();
            let staged_events = load_staged_events(staging_path.as_path())?;
            if staged_events.is_empty() {
                println!("no staged events to commit");
                return Ok(());
            }

            let schema_manager = SchemaManager::load(config.schema_store_path())?;
            let store = EventStore::open(
                config.event_store_path(),
                config.encryption_key()?,
                config.snowflake_worker_id,
            )?;
            let plugins = PluginManager::from_config(&config)?;
            let mut tx = store.transaction()?;
            let restrict_mode = config.restrict;

            for staged_event in &staged_events {
                ensure_snake_case("aggregate_type", &staged_event.aggregate)?;
                ensure_snake_case("event_type", &staged_event.event)?;
                ensure_aggregate_id(&staged_event.aggregate_id)?;
                ensure_payload_size(&staged_event.payload)?;
                if let Some(ref metadata) = staged_event.metadata {
                    ensure_metadata_extensions(metadata)?;
                }

                let is_new = match store
                    .aggregate_version(&staged_event.aggregate, &staged_event.aggregate_id)?
                {
                    Some(version) if version > 0 => false,
                    Some(_) | None => true,
                };
                ensure_first_event_rule(is_new, &staged_event.event)?;

                let schema_present = match schema_manager.get(&staged_event.aggregate) {
                    Ok(_) => true,
                    Err(EventError::SchemaNotFound) => false,
                    Err(err) => return Err(err.into()),
                };

                if !schema_present && restrict_mode.requires_declared_schema() {
                    bail!(restrict::strict_mode_missing_schema_message(
                        &staged_event.aggregate
                    ));
                }

                if schema_present {
                    schema_manager.validate_event(
                        &staged_event.aggregate,
                        &staged_event.event,
                        &staged_event.payload,
                    )?;
                }

                let mut evt = staged_event.to_append_event();
                evt.tenant = config.active_domain().to_string();
                evt.reference_targets = Vec::new();
                tx.append(evt)?;
            }

            let records = tx.commit()?;
            for record in &records {
                println!("{}", serde_json::to_string_pretty(record)?);
                maybe_auto_snapshot(&store, &schema_manager, record);
            }

            if !plugins.is_empty() {
                for record in &records {
                    let schema = schema_manager.get(&record.aggregate_type).ok();
                    match store.get_aggregate_state(&record.aggregate_type, &record.aggregate_id) {
                        Ok(current_state) => {
                            if let Err(err) =
                                plugins.notify_event(record, &current_state, schema.as_ref())
                            {
                                eprintln!("plugin notification failed: {}", err);
                            }
                        }
                        Err(err) => {
                            eprintln!(
                                "plugin notification skipped (failed to load state): {}",
                                err
                            );
                        }
                    }
                }
            }

            clear_staged_events(staging_path.as_path())?;
            println!("committed {} event(s)", records.len());
        }
        AggregateCommands::Export(args) => {
            export_aggregates(&config, args)?;
        }
    }

    Ok(())
}

fn maybe_auto_snapshot(store: &EventStore, schemas: &SchemaManager, record: &EventRecord) {
    if !schemas.should_snapshot(&record.aggregate_type, record.version) {
        return;
    }

    match store.create_snapshot(
        &record.aggregate_type,
        &record.aggregate_id,
        Some(format!("auto snapshot v{}", record.version)),
    ) {
        Ok(snapshot) => eprintln!(
            "auto snapshot created: aggregate={} aggregate_id={} version={}",
            snapshot.aggregate_type, snapshot.aggregate_id, snapshot.version
        ),
        Err(err) => eprintln!(
            "failed to create auto snapshot for {}::{} v{}: {}",
            record.aggregate_type, record.aggregate_id, record.version, err
        ),
    }
}

fn normalize_references_offline(
    schemas: &SchemaManager,
    store: &EventStore,
    tenant: &str,
    aggregate: &str,
    payload: Value,
) -> Result<(Value, Vec<(String, String)>)> {
    match schemas.get(aggregate) {
        Ok(_) => {}
        Err(EventError::SchemaNotFound) => return Ok((payload, Vec::new())),
        Err(err) => return Err(err.into()),
    };

    let context = ReferenceContext {
        domain: tenant,
        aggregate_type: aggregate,
    };
    let mut resolver = |reference: &AggregateReference,
                        integrity: ReferenceIntegrity|
     -> eventdbx::error::Result<ReferenceResolutionStatus> {
        if !reference.domain.eq_ignore_ascii_case(tenant) {
            return Ok(ReferenceResolutionStatus::Forbidden);
        }
        match store.aggregate_version(&reference.aggregate_type, &reference.aggregate_id) {
            Ok(Some(_)) => Ok(ReferenceResolutionStatus::Ok),
            Ok(None) => match integrity {
                ReferenceIntegrity::Weak => Ok(ReferenceResolutionStatus::NotFound),
                ReferenceIntegrity::Strong => Ok(ReferenceResolutionStatus::NotFound),
            },
            Err(err) => Err(err.into()),
        }
    };

    let (normalized_payload, outcomes) =
        schemas.normalize_references(aggregate, payload, context, &mut resolver)?;
    let targets = outcomes
        .iter()
        .map(|outcome| (outcome.reference.to_canonical(), outcome.path.clone()))
        .collect();
    Ok((normalized_payload, targets))
}

fn partition_referrers(
    schemas: &SchemaManager,
    tenant: &str,
    referrers: &[(String, String, String, String)],
) -> (Vec<(String, String, String, String)>, Vec<String>) {
    let mut nullify = Vec::new();
    let mut blockers = Vec::new();
    for (ref_tenant, agg, id, path) in referrers {
        if ref_tenant != tenant {
            blockers.push(format!("{ref_tenant}:{agg}:{id} ({path})"));
            continue;
        }
        let cascade = schemas
            .reference_rules_for_path(agg, path)
            .map(|rules| rules.cascade)
            .unwrap_or_default();
        match cascade {
            ReferenceCascade::Nullify => nullify.push((
                ref_tenant.to_string(),
                agg.to_string(),
                id.to_string(),
                path.to_string(),
            )),
            _ => blockers.push(format!("{ref_tenant}:{agg}:{id} ({path})")),
        }
    }
    (nullify, blockers)
}

fn nullify_referrers_offline(
    store: &EventStore,
    schemas: &SchemaManager,
    tenant: &str,
    target: &str,
    referrers: &[(String, String, String, String)],
) -> Result<()> {
    if referrers.is_empty() {
        return Ok(());
    }
    let mut grouped: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
    for (ref_tenant, agg, id, path) in referrers {
        if ref_tenant != tenant {
            continue;
        }
        if schemas
            .reference_rules_for_path(agg, path)
            .map(|rules| rules.cascade)
            .unwrap_or_default()
            != ReferenceCascade::Nullify
        {
            continue;
        }
        grouped
            .entry((agg.clone(), id.clone()))
            .or_default()
            .insert(json_pointer_from_path(path));
    }

    for ((aggregate, aggregate_id), paths) in grouped {
        let patch_ops: Vec<Value> = paths
            .into_iter()
            .map(|path| {
                json!({
                    "op": "replace",
                    "path": path,
                    "value": Value::Null
                })
            })
            .collect();
        if patch_ops.is_empty() {
            continue;
        }
        let patch = Value::Array(patch_ops);
        let payload = store.prepare_payload_from_patch(&aggregate, &aggregate_id, &patch)?;
        store.append(AppendEvent {
            aggregate_type: aggregate.clone(),
            aggregate_id: aggregate_id.clone(),
            event_type: "_ref_nullify".into(),
            payload,
            metadata: None,
            issued_by: None,
            note: Some(format!("auto-nullify due to {target}")),
            tenant: tenant.to_string(),
            reference_targets: Vec::new(),
        })?;
    }
    Ok(())
}

fn json_pointer_from_path(path: &str) -> String {
    if path.is_empty() {
        return "/".to_string();
    }
    format!("/{}", path.replace('.', "/"))
}

#[cfg(test)]
fn ensure_schema_for_mode(
    mode: RestrictMode,
    schema_manager: &SchemaManager,
    aggregate: &str,
) -> Result<()> {
    if mode.requires_declared_schema() {
        schema_manager.get(aggregate).map(|_| ()).map_err(|_| {
            EventError::SchemaViolation(restrict::strict_mode_missing_schema_message(aggregate))
        })?;
    }
    Ok(())
}

struct CreateCommand {
    aggregate: String,
    aggregate_id: String,
    event: String,
    payload: Value,
    metadata: Option<Value>,
    note: Option<String>,
    token: Option<String>,
    json: bool,
    publish: Vec<PublishTargetArg>,
}

fn execute_create_command(config: &Config, command: CreateCommand) -> Result<()> {
    let CreateCommand {
        aggregate,
        aggregate_id,
        event,
        payload,
        metadata,
        note,
        token,
        json,
        publish,
    } = command;

    let verbose = config.verbose_responses();

    if let Some(ref note_value) = note {
        if note_value.chars().count() > MAX_EVENT_NOTE_LENGTH {
            bail!("note cannot exceed {} characters", MAX_EVENT_NOTE_LENGTH);
        }
    }
    if let Some(ref metadata_value) = metadata {
        ensure_metadata_extensions(metadata_value)?;
    }

    ensure_snake_case("aggregate_type", &aggregate)?;
    ensure_snake_case("event_type", &event)?;
    ensure_aggregate_id(&aggregate_id)?;
    ensure_payload_size(&payload)?;

    let restrict_mode = config.restrict;
    let schema_manager = SchemaManager::load(config.schema_store_path())?;
    let schema_present = match schema_manager.get(&aggregate) {
        Ok(_) => true,
        Err(EventError::SchemaNotFound) => false,
        Err(err) => return Err(err.into()),
    };
    if !schema_present && restrict_mode.requires_declared_schema() {
        bail!(restrict::strict_mode_missing_schema_message(&aggregate));
    }
    if schema_present {
        schema_manager.validate_event(&aggregate, &event, &payload)?;
    }

    let assignments = if config.multi_tenant() {
        Some(TenantAssignmentStore::open(config.tenant_meta_path())?)
    } else {
        None
    };
    let encryption = config.encryption_key()?;
    let publish_targets = build_publish_targets(&publish)?;
    let publish_slice = (!publish_targets.is_empty()).then_some(publish_targets.as_slice());

    match EventStore::open(
        config.event_store_path(),
        encryption,
        config.snowflake_worker_id,
    ) {
        Ok(store) => {
            if store
                .aggregate_version(&aggregate, &aggregate_id)?
                .is_some()
            {
                bail!("aggregate {}::{} already exists", aggregate, aggregate_id);
            }

            if let Some(assignments) = assignments.as_ref() {
                enforce_offline_tenant_quota(config, &store, assignments)?;
            }

            let plugins = PluginManager::from_config(&config)?;
            let (normalized_payload, reference_targets) = normalize_references_offline(
                &schema_manager,
                &store,
                config.active_domain(),
                &aggregate,
                payload.clone(),
            )?;
            ensure_payload_size(&normalized_payload)?;
            let record = store.append(AppendEvent {
                aggregate_type: aggregate.clone(),
                aggregate_id: aggregate_id.clone(),
                event_type: event.clone(),
                payload: normalized_payload.clone(),
                metadata: metadata.clone(),
                issued_by: None,
                note: note.clone(),
                tenant: config.active_domain().to_string(),
                reference_targets,
            })?;

            maybe_auto_snapshot(&store, &schema_manager, &record);

            if let Some(assignments) = assignments.as_ref() {
                let usage = store.storage_usage_bytes()?;
                assignments.update_storage_usage_bytes(config.active_domain(), usage)?;
            }

            if !plugins.is_empty() {
                let schema = schema_manager.get(&record.aggregate_type).ok();
                match store.get_aggregate_state(&record.aggregate_type, &record.aggregate_id) {
                    Ok(current_state) => {
                        if let Err(err) = plugins.notify_event_for_targets(
                            &record,
                            &current_state,
                            schema.as_ref(),
                            publish_slice,
                        ) {
                            eprintln!("plugin notification failed: {}", err);
                        }
                    }
                    Err(err) => {
                        eprintln!(
                            "plugin notification skipped (failed to load state): {}",
                            err
                        );
                    }
                }
            }

            let state = store.get_aggregate_state(&aggregate, &aggregate_id)?;
            if verbose {
                if json {
                    println!("{}", serde_json::to_string_pretty(&state)?);
                } else {
                    println!(
                        "aggregate_type={} aggregate_id={} version={} archived={}",
                        state.aggregate_type, state.aggregate_id, state.version, state.archived
                    );
                }
            } else {
                println!("Ok");
            }
            Ok(())
        }
        Err(EventError::Storage(message)) if is_lock_error_message(&message) => {
            if publish_slice.is_some() {
                bail!(
                    "explicit publish targets are not supported when proxying to a running server"
                );
            }
            let state = proxy_create_via_socket(
                config,
                token,
                &aggregate,
                &aggregate_id,
                &event,
                &payload,
                metadata.as_ref(),
                note.as_deref(),
            )?;
            if verbose {
                if let Some(state) = state {
                    if json {
                        println!("{}", serde_json::to_string_pretty(&state)?);
                    } else {
                        println!(
                            "aggregate_type={} aggregate_id={} version={} archived={}",
                            state.aggregate_type, state.aggregate_id, state.version, state.archived
                        );
                    }
                } else {
                    println!("Ok");
                }
            } else {
                println!("Ok");
            }
            Ok(())
        }
        Err(err) => Err(err.into()),
    }
}

struct AppendCommand {
    aggregate: String,
    aggregate_id: String,
    event: String,
    stage: bool,
    token: Option<String>,
    payload: Option<Value>,
    patch: Option<Value>,
    metadata: Option<Value>,
    note: Option<String>,
    publish: Vec<PublishTargetArg>,
}

fn execute_append_command(config: &Config, command: AppendCommand) -> Result<()> {
    let AppendCommand {
        aggregate,
        aggregate_id,
        event,
        stage,
        token,
        payload,
        patch,
        metadata,
        note,
        publish,
    } = command;

    let verbose = config.verbose_responses();

    if let Some(ref note_value) = note {
        if note_value.chars().count() > MAX_EVENT_NOTE_LENGTH {
            bail!("note cannot exceed {} characters", MAX_EVENT_NOTE_LENGTH);
        }
    }
    if let Some(ref metadata_value) = metadata {
        ensure_metadata_extensions(metadata_value)?;
    }

    ensure_snake_case("aggregate_type", &aggregate)?;
    ensure_snake_case("event_type", &event)?;
    ensure_aggregate_id(&aggregate_id)?;

    let restrict_mode = config.restrict;
    let schema_manager = SchemaManager::load(config.schema_store_path())?;
    let schema_present = match schema_manager.get(&aggregate) {
        Ok(_) => true,
        Err(EventError::SchemaNotFound) => false,
        Err(err) => return Err(err.into()),
    };

    if !schema_present && restrict_mode.requires_declared_schema() {
        bail!(restrict::strict_mode_missing_schema_message(&aggregate));
    }

    if patch.is_none() {
        let payload_value = payload
            .as_ref()
            .ok_or_else(|| anyhow!("payload must be provided via --field or --payload"))?;
        ensure_payload_size(payload_value)?;
        if schema_present {
            schema_manager.validate_event(&aggregate, &event, payload_value)?;
        }
    }

    let publish_targets = build_publish_targets(&publish)?;
    let publish_slice = (!publish_targets.is_empty()).then_some(publish_targets.as_slice());

    if stage {
        if publish_slice.is_some() {
            bail!("--publish cannot be combined with --stage");
        }
        match EventStore::open(
            config.event_store_path(),
            config.encryption_key()?,
            config.snowflake_worker_id,
        ) {
            Ok(store) => {
                let effective_payload = if let Some(ref patch_ops) = patch {
                    store.prepare_payload_from_patch(&aggregate, &aggregate_id, patch_ops)?
                } else {
                    payload.clone().ok_or_else(|| {
                        anyhow!("payload must be provided via --field or --payload")
                    })?
                };
                if patch.is_some() {
                    ensure_payload_size(&effective_payload)?;
                    if schema_present {
                        schema_manager.validate_event(&aggregate, &event, &effective_payload)?;
                    }
                }
                let (exists, is_new) = match store.aggregate_version(&aggregate, &aggregate_id)? {
                    Some(version) => (true, version == 0),
                    None => (false, true),
                };
                if !exists {
                    bail!("aggregate {}::{} does not exist", aggregate, aggregate_id);
                }
                ensure_first_event_rule(is_new, &event)?;
                let (normalized_payload, reference_targets) = normalize_references_offline(
                    &schema_manager,
                    &store,
                    config.active_domain(),
                    &aggregate,
                    effective_payload.clone(),
                )?;
                ensure_payload_size(&normalized_payload)?;
                {
                    let mut tx = store.transaction()?;
                    tx.append(AppendEvent {
                        aggregate_type: aggregate.clone(),
                        aggregate_id: aggregate_id.clone(),
                        event_type: event.clone(),
                        payload: normalized_payload.clone(),
                        metadata: metadata.clone(),
                        issued_by: None,
                        note: note.clone(),
                        tenant: config.active_domain().to_string(),
                        reference_targets: reference_targets.clone(),
                    })?;
                }

                let staged_event = StagedEvent {
                    aggregate: aggregate.clone(),
                    aggregate_id: aggregate_id.clone(),
                    event: event.clone(),
                    payload: normalized_payload,
                    metadata: metadata.clone(),
                    issued_by: None,
                    note: note.clone(),
                };
                let staging_path = config.staging_path();
                append_staged_event(staging_path.as_path(), staged_event)?;
                println!("event staged for later commit");
                return Ok(());
            }
            Err(EventError::Storage(message)) if is_lock_error_message(&message) => {
                bail!(
                    "event store is locked by a running server.\nStop the server or omit --stage."
                );
            }
            Err(err) => return Err(err.into()),
        }
    }

    let encryption = config.encryption_key()?;
    match EventStore::open(
        config.event_store_path(),
        encryption,
        config.snowflake_worker_id,
    ) {
        Ok(store) => {
            let assignments = if config.multi_tenant() {
                Some(TenantAssignmentStore::open(config.tenant_meta_path())?)
            } else {
                None
            };
            if let Some(assignments) = assignments.as_ref() {
                enforce_offline_tenant_quota(config, &store, assignments)?;
            }
            let plugins = PluginManager::from_config(&config)?;
            let effective_payload = if let Some(ref patch_ops) = patch {
                store.prepare_payload_from_patch(&aggregate, &aggregate_id, patch_ops)?
            } else {
                payload
                    .clone()
                    .ok_or_else(|| anyhow!("payload must be provided via --field or --payload"))?
            };
            if patch.is_some() {
                ensure_payload_size(&effective_payload)?;
                if schema_present {
                    schema_manager.validate_event(&aggregate, &event, &effective_payload)?;
                }
            }
            let version_opt = store.aggregate_version(&aggregate, &aggregate_id)?;
            let is_new = match version_opt {
                Some(version) if version > 0 => false,
                _ => true,
            };
            let exists = version_opt.is_some();
            if !exists {
                bail!("aggregate {}::{} does not exist", aggregate, aggregate_id);
            }
            ensure_first_event_rule(is_new, &event)?;
            let (normalized_payload, reference_targets) = normalize_references_offline(
                &schema_manager,
                &store,
                config.active_domain(),
                &aggregate,
                effective_payload.clone(),
            )?;
            ensure_payload_size(&normalized_payload)?;
            let record = store.append(AppendEvent {
                aggregate_type: aggregate.clone(),
                aggregate_id: aggregate_id.clone(),
                event_type: event.clone(),
                payload: normalized_payload.clone(),
                metadata: metadata.clone(),
                issued_by: None,
                note: note.clone(),
                tenant: config.active_domain().to_string(),
                reference_targets,
            })?;
            if let Some(assignments) = assignments.as_ref() {
                let usage = store.storage_usage_bytes()?;
                assignments.update_storage_usage_bytes(config.active_domain(), usage)?;
            }

            maybe_auto_snapshot(&store, &schema_manager, &record);
            if verbose {
                println!("{}", serde_json::to_string_pretty(&record)?);
            } else {
                println!("Ok");
            }

            if !plugins.is_empty() {
                let schema = schema_manager.get(&record.aggregate_type).ok();
                match store.get_aggregate_state(&record.aggregate_type, &record.aggregate_id) {
                    Ok(current_state) => {
                        if let Err(err) = plugins.notify_event_for_targets(
                            &record,
                            &current_state,
                            schema.as_ref(),
                            publish_slice,
                        ) {
                            eprintln!("plugin notification failed: {}", err);
                        }
                    }
                    Err(err) => {
                        eprintln!(
                            "plugin notification skipped (failed to load state): {}",
                            err
                        );
                    }
                }
            }
            Ok(())
        }
        Err(EventError::Storage(message)) if is_lock_error_message(&message) => {
            if publish_slice.is_some() {
                bail!(
                    "explicit publish targets are not supported when proxying to a running server"
                );
            }
            let record = proxy_append_via_socket(
                config,
                token,
                &aggregate,
                &aggregate_id,
                &event,
                if patch.is_some() {
                    None
                } else {
                    payload.as_ref()
                },
                patch.as_ref(),
                metadata.as_ref(),
                note.as_deref(),
            )?;
            if verbose {
                if let Some(record) = record {
                    println!("{}", serde_json::to_string_pretty(&record)?);
                } else {
                    println!("Ok");
                }
            } else {
                println!("Ok");
            }
            Ok(())
        }
        Err(err) => Err(err.into()),
    }
}

fn proxy_append_via_socket(
    config: &Config,
    token: Option<String>,
    aggregate: &str,
    aggregate_id: &str,
    event: &str,
    payload: Option<&Value>,
    patch: Option<&Value>,
    metadata: Option<&Value>,
    note: Option<&str>,
) -> Result<Option<EventRecord>> {
    let token = ensure_proxy_token(config, token)?;
    let client = ServerClient::new(config)?;
    let record = if let Some(patch_value) = patch {
        client
            .patch_event(
                &token,
                aggregate,
                aggregate_id,
                event,
                patch_value,
                metadata,
                note,
            )
            .with_context(|| {
                format!(
                    "failed to append patch event via running server socket {}",
                    config.socket.bind_addr
                )
            })?
    } else {
        client
            .append_event(
                &token,
                aggregate,
                aggregate_id,
                event,
                payload,
                metadata,
                note,
            )
            .with_context(|| {
                format!(
                    "failed to append event via running server socket {}",
                    config.socket.bind_addr
                )
            })?
    };
    Ok(record)
}

fn proxy_create_via_socket(
    config: &Config,
    token: Option<String>,
    aggregate: &str,
    aggregate_id: &str,
    event: &str,
    payload: &Value,
    metadata: Option<&Value>,
    note: Option<&str>,
) -> Result<Option<AggregateState>> {
    let token = ensure_proxy_token(config, token)?;
    let client = ServerClient::new(config)?;
    let state = client
        .create_aggregate(
            &token,
            aggregate,
            aggregate_id,
            event,
            payload,
            metadata,
            note,
        )
        .with_context(|| {
            format!(
                "failed to create aggregate via running server socket {}",
                config.socket.bind_addr
            )
        })?;
    Ok(state)
}

pub(crate) fn ensure_proxy_token(config: &Config, token: Option<String>) -> Result<String> {
    if let Some(token) = token.and_then(normalize_token) {
        return Ok(token);
    }
    if let Some(token) = env::var("EVENTDBX_TOKEN").ok().and_then(normalize_token) {
        return Ok(token);
    }
    match cli_token::ensure_bootstrap_token(config, None) {
        Ok(token) => return Ok(token),
        Err(err) => {
            warn!(
                "failed to load CLI bootstrap token ({}); falling back to ephemeral token",
                err
            );
        }
    }
    issue_ephemeral_token(config)
}

fn issue_ephemeral_token(config: &Config) -> Result<String> {
    let encryptor = config.encryption_key()?;
    let jwt_config = config.jwt_manager_config()?;
    let manager = TokenManager::load(
        jwt_config,
        config.tokens_path(),
        config.jwt_revocations_path(),
        encryptor,
    )?;
    let user = proxy_user_identity();
    let subject = format!("cli:{}", user);
    let record = manager.issue(IssueTokenInput {
        subject,
        group: "cli".to_string(),
        user,
        actions: vec![ROOT_ACTION.to_string()],
        resources: vec![ROOT_RESOURCE.to_string()],
        tenants: Vec::new(),
        ttl_secs: Some(120),
        not_before: None,
        issued_by: "cli".to_string(),
        limits: JwtLimits {
            write_events: None,
            keep_alive: false,
        },
    })?;
    record
        .token
        .ok_or_else(|| anyhow!("ephemeral token missing value"))
}

fn proxy_user_identity() -> String {
    let env_value = env::var("USER")
        .or_else(|_| env::var("USERNAME"))
        .ok()
        .and_then(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        });

    env_value.unwrap_or_else(|| "local".to_string())
}

fn normalize_token(token: String) -> Option<String> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn parse_publish_target(raw: &str) -> std::result::Result<PublishTargetArg, String> {
    let parts: Vec<&str> = raw.split(':').collect();
    if parts.is_empty() {
        return Err("publish target cannot be empty".into());
    }
    if parts.len() > 3 {
        return Err("publish target format is PLUGIN[:MODE[:PRIORITY]]".into());
    }
    let plugin = parts[0].trim();
    if plugin.is_empty() {
        return Err("publish target name cannot be empty".into());
    }

    let mode = parts.get(1).and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return None;
        }
        parse_publish_mode(trimmed).ok()
    });

    let priority = parts.get(2).and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return None;
        }
        parse_publish_priority(trimmed).ok()
    });

    Ok(PublishTargetArg {
        plugin: plugin.to_string(),
        mode,
        priority,
    })
}

fn parse_publish_mode(raw: &str) -> std::result::Result<PluginPayloadMode, String> {
    match raw.to_ascii_lowercase().as_str() {
        "all" => Ok(PluginPayloadMode::All),
        "event-only" => Ok(PluginPayloadMode::EventOnly),
        "state-only" => Ok(PluginPayloadMode::StateOnly),
        "schema-only" => Ok(PluginPayloadMode::SchemaOnly),
        "event-and-schema" => Ok(PluginPayloadMode::EventAndSchema),
        "extensions-only" => Ok(PluginPayloadMode::ExtensionsOnly),
        other => Err(format!("invalid publish mode '{}'", other)),
    }
}

fn parse_publish_priority(raw: &str) -> std::result::Result<JobPriority, String> {
    match raw.to_ascii_lowercase().as_str() {
        "low" => Ok(JobPriority::Low),
        "normal" => Ok(JobPriority::Normal),
        "high" => Ok(JobPriority::High),
        other => Err(format!("invalid publish priority '{}'", other)),
    }
}

fn build_publish_targets(inputs: &[PublishTargetArg]) -> Result<Vec<PublishTarget>> {
    let mut targets = Vec::with_capacity(inputs.len());
    for input in inputs {
        targets.push(PublishTarget {
            plugin: input.plugin.clone(),
            mode: input.mode,
            priority: input.priority,
        });
    }
    Ok(targets)
}

fn export_aggregates(config: &Config, args: AggregateExportArgs) -> Result<()> {
    let AggregateExportArgs {
        aggregate,
        all,
        format,
        output,
        zip,
        pretty,
    } = args;

    if all && aggregate.is_some() {
        bail!("aggregate name cannot be provided when using --all");
    }
    if !all && aggregate.is_none() {
        bail!("aggregate name must be provided unless --all is specified");
    }

    let target_name = if all { None } else { aggregate };
    let store = EventStore::open_read_only(config.event_store_path(), config.encryption_key()?)?;
    let exports = collect_export_records(&store, target_name.as_deref(), all)?;

    if exports.is_empty() {
        if let Some(name) = target_name {
            println!("no aggregates found for '{}'", name);
        } else {
            println!("no aggregates found");
        }
        return Ok(());
    }

    if zip {
        let zip_path = export_as_zip(&exports, format, pretty, output)?;
        println!(
            "Exported {} aggregate type(s) to {} (zip archive)",
            exports.len(),
            zip_path.display()
        );
    } else {
        let files = export_to_files(&exports, format, pretty, output)?;
        for path in &files {
            println!("wrote {}", path.display());
        }
        println!(
            "Exported {} aggregate type(s) in {} format",
            exports.len(),
            export_suffix(format).to_uppercase()
        );
    }

    Ok(())
}

fn enforce_offline_tenant_quota(
    config: &Config,
    store: &EventStore,
    assignments: &TenantAssignmentStore,
) -> Result<()> {
    let tenant = config.active_domain();
    let usage = store.storage_usage_bytes()?;
    assignments.update_storage_usage_bytes(tenant, usage)?;
    if let Some(limit_mb) = assignments.quota_for(tenant)? {
        let limit_bytes = limit_mb.saturating_mul(BYTES_PER_MEGABYTE);
        if usage >= limit_bytes {
            bail!(
                "tenant '{}' reached storage quota ({} MB)",
                tenant,
                limit_mb
            );
        }
    }
    Ok(())
}

fn collect_export_records(
    store: &EventStore,
    target: Option<&str>,
    include_all: bool,
) -> Result<BTreeMap<String, Vec<ExportRecord>>> {
    let mut map: BTreeMap<String, Vec<ExportRecord>> = BTreeMap::new();
    let filter = if include_all {
        None
    } else {
        Some(
            target
                .ok_or_else(|| anyhow!("aggregate name must be provided unless --all is set"))?
                .to_string(),
        )
    };

    for aggregate in store.aggregates_paginated(0, None) {
        let store::AggregateState {
            aggregate_type,
            aggregate_id,
            state,
            ..
        } = aggregate;

        if let Some(ref filter_type) = filter {
            if &aggregate_type != filter_type {
                continue;
            }
        }

        map.entry(aggregate_type).or_default().push(ExportRecord {
            aggregate_id,
            state,
        });
    }

    for rows in map.values_mut() {
        rows.sort_by(|a, b| a.aggregate_id.cmp(&b.aggregate_id));
    }

    Ok(map)
}

fn export_to_files(
    exports: &BTreeMap<String, Vec<ExportRecord>>,
    format: AggregateExportFormat,
    pretty: bool,
    output: PathBuf,
) -> Result<Vec<PathBuf>> {
    let multiple = exports.len() > 1;

    if multiple {
        if output.exists() && output.is_file() {
            bail!("output path must be a directory when exporting multiple aggregate types");
        }
        let files = export_into_directory(output.as_path(), exports, format, pretty)?;
        return Ok(files.into_iter().map(|(path, _)| path).collect());
    }

    let (aggregate_type, rows) = exports.iter().next().expect("exports is not empty");
    let target = output;
    let as_file = should_treat_as_file(&target, format);

    if as_file {
        if let Some(parent) = target.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        match format {
            AggregateExportFormat::Csv => write_csv_file(&target, rows)?,
            AggregateExportFormat::Json => write_json_file(&target, rows, pretty)?,
        }
        return Ok(vec![target]);
    }

    fs::create_dir_all(&target)?;
    let file_name = export_file_name(aggregate_type, format);
    let file_path = target.join(file_name);
    match format {
        AggregateExportFormat::Csv => write_csv_file(&file_path, rows)?,
        AggregateExportFormat::Json => write_json_file(&file_path, rows, pretty)?,
    }
    Ok(vec![file_path])
}

fn should_treat_as_file(path: &Path, format: AggregateExportFormat) -> bool {
    if path.exists() {
        return path.is_file();
    }

    let expected_extension = export_suffix(format);
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case(expected_extension))
        .unwrap_or(false)
}

fn export_as_zip(
    exports: &BTreeMap<String, Vec<ExportRecord>>,
    format: AggregateExportFormat,
    pretty: bool,
    output: PathBuf,
) -> Result<PathBuf> {
    let temp = tempdir()?;
    let generated = export_into_directory(temp.path(), exports, format, pretty)?;
    let zip_path = normalize_zip_output(output, exports, format)?;
    create_zip_archive(&generated, &zip_path)?;
    Ok(zip_path)
}

fn export_into_directory(
    base_dir: &Path,
    exports: &BTreeMap<String, Vec<ExportRecord>>,
    format: AggregateExportFormat,
    pretty: bool,
) -> Result<Vec<(PathBuf, String)>> {
    fs::create_dir_all(base_dir)?;
    let mut files = Vec::new();

    for (aggregate_type, rows) in exports {
        let file_name = export_file_name(aggregate_type, format);
        let file_path = base_dir.join(&file_name);
        match format {
            AggregateExportFormat::Csv => write_csv_file(&file_path, rows)?,
            AggregateExportFormat::Json => write_json_file(&file_path, rows, pretty)?,
        }
        files.push((file_path, file_name));
    }

    Ok(files)
}

fn write_csv_file(path: &Path, rows: &[ExportRecord]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut dynamic_columns = BTreeSet::new();
    for row in rows {
        for key in row.state.keys() {
            dynamic_columns.insert(key.clone());
        }
    }

    let mut headers = Vec::with_capacity(dynamic_columns.len() + 1);
    headers.push(EXPORT_ID_KEY.to_string());
    headers.extend(dynamic_columns.into_iter());

    let mut writer = WriterBuilder::new().from_path(path)?;
    writer.write_record(&headers)?;

    for row in rows {
        let mut record = Vec::with_capacity(headers.len());
        for (idx, header) in headers.iter().enumerate() {
            if idx == 0 {
                record.push(row.aggregate_id.clone());
            } else {
                record.push(row.state.get(header).cloned().unwrap_or_default());
            }
        }
        writer.write_record(record)?;
    }

    writer.flush()?;
    Ok(())
}

fn write_json_file(path: &Path, rows: &[ExportRecord], pretty: bool) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut file = File::create(path)?;
    if pretty {
        let entries: Vec<_> = rows.iter().map(record_to_json).collect();
        serde_json::to_writer_pretty(&mut file, &entries)?;
    } else {
        file.write_all(b"[")?;
        for (idx, row) in rows.iter().enumerate() {
            if idx > 0 {
                file.write_all(b",")?;
            }
            let value = record_to_json(row);
            let payload = serde_json::to_vec(&value)?;
            file.write_all(&payload)?;
        }
        file.write_all(b"]")?;
    }
    file.write_all(b"\n")?;
    Ok(())
}

fn record_to_json(record: &ExportRecord) -> Value {
    let mut map = serde_json::Map::new();
    map.insert(
        EXPORT_ID_KEY.to_string(),
        Value::String(record.aggregate_id.clone()),
    );
    for (key, value) in &record.state {
        map.insert(key.clone(), Value::String(value.clone()));
    }
    Value::Object(map)
}

fn normalize_zip_output(
    output: PathBuf,
    exports: &BTreeMap<String, Vec<ExportRecord>>,
    format: AggregateExportFormat,
) -> Result<PathBuf> {
    if output.exists() && output.is_dir() {
        let file_name = default_zip_name(exports, format);
        let path = output.join(file_name);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        return Ok(path);
    }

    let mut path = output;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let has_zip_extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("zip"))
        .unwrap_or(false);

    if !has_zip_extension {
        if path.file_name().is_none() || path.file_name().unwrap().is_empty() {
            path = PathBuf::from(default_zip_name(exports, format));
        } else {
            path.set_extension("zip");
        }
    }

    Ok(path)
}

fn default_zip_name(
    exports: &BTreeMap<String, Vec<ExportRecord>>,
    format: AggregateExportFormat,
) -> String {
    let suffix = export_suffix(format);
    let stem = if exports.len() == 1 {
        let name = exports.keys().next().expect("exports not empty");
        format!("{}_{}", sanitize_component(name), suffix)
    } else {
        format!("aggregates_{}", suffix)
    };
    format!("{stem}.zip")
}

fn export_file_name(aggregate_type: &str, format: AggregateExportFormat) -> String {
    let stem = sanitize_component(aggregate_type);
    let suffix = export_suffix(format);
    format!("{stem}.{suffix}")
}

fn export_suffix(format: AggregateExportFormat) -> &'static str {
    match format {
        AggregateExportFormat::Csv => "csv",
        AggregateExportFormat::Json => "json",
    }
}

fn sanitize_component(input: &str) -> String {
    let mut sanitized = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            sanitized.push(ch);
        } else {
            sanitized.push('_');
        }
    }
    if sanitized.is_empty() {
        sanitized.push('_');
    }
    sanitized
}

fn create_zip_archive(files: &[(PathBuf, String)], output: &Path) -> Result<()> {
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let file = File::create(output)?;
    let mut zip = ZipWriter::new(file);
    let options = FileOptions::default().compression_method(CompressionMethod::Deflated);

    for (path, name) in files {
        zip.start_file(name, options)?;
        let mut reader = File::open(path)?;
        std::io::copy(&mut reader, &mut zip)?;
    }

    zip.finish()?;
    Ok(())
}

fn parse_key_value(raw: &str) -> Result<KeyValue, String> {
    let mut parts = raw.splitn(2, '=');
    let key = parts
        .next()
        .ok_or_else(|| "missing key".to_string())?
        .trim();
    let value = parts
        .next()
        .ok_or_else(|| "missing value".to_string())?
        .trim();

    if key.is_empty() {
        return Err("field key cannot be empty".to_string());
    }

    Ok(KeyValue {
        key: key.to_string(),
        value: value.to_string(),
    })
}

fn collect_payload(fields: Vec<KeyValue>) -> Value {
    let mut map = JsonMap::new();
    for kv in fields {
        map.insert(kv.key, Value::String(kv.value));
    }
    Value::Object(map)
}

fn state_at_version(
    events: &[store::EventRecord],
    version: u64,
) -> (u64, BTreeMap<String, String>, String) {
    let mut state = BTreeMap::new();
    let mut hashes = Vec::new();
    let mut last_version = 0;

    for event in events {
        if event.version > version {
            break;
        }
        last_version = event.version;
        hashes.push(event.hash.clone());
        for (key, value) in payload_to_map(&event.payload) {
            state.insert(key, value);
        }
    }

    let merkle_root = compute_merkle_root(&hashes);
    (last_version, state, merkle_root)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StagedEvent {
    aggregate: String,
    aggregate_id: String,
    event: String,
    #[serde(default = "default_event_payload")]
    payload: Value,
    #[serde(default)]
    metadata: Option<Value>,
    #[serde(default)]
    issued_by: Option<StagedIssuedBy>,
    #[serde(default)]
    note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StagedIssuedBy {
    group: String,
    user: String,
}

impl StagedEvent {
    fn to_append_event(&self) -> AppendEvent {
        AppendEvent {
            aggregate_type: self.aggregate.clone(),
            aggregate_id: self.aggregate_id.clone(),
            event_type: self.event.clone(),
            payload: self.payload.clone(),
            metadata: self.metadata.clone(),
            issued_by: self.issued_by.clone().map(Into::into),
            note: self.note.clone(),
            tenant: "default".to_string(),
            reference_targets: Vec::new(),
        }
    }
}

impl From<StagedIssuedBy> for ActorClaims {
    fn from(value: StagedIssuedBy) -> Self {
        ActorClaims {
            group: value.group,
            user: value.user,
        }
    }
}

fn default_event_payload() -> Value {
    Value::Null
}

fn load_staged_events(path: &Path) -> Result<Vec<StagedEvent>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let contents = fs::read_to_string(path)?;
    if contents.trim().is_empty() {
        return Ok(Vec::new());
    }

    let events = serde_json::from_str::<Vec<StagedEvent>>(&contents)?;
    Ok(events)
}

fn append_staged_event(path: &Path, event: StagedEvent) -> Result<()> {
    let mut events = load_staged_events(path)?;
    events.push(event);
    save_staged_events(path, &events)
}

fn clear_staged_events(path: &Path) -> Result<()> {
    if path.exists() {
        fs::write(path, "[]")?;
    }
    Ok(())
}

fn save_staged_events(path: &Path, events: &[StagedEvent]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = serde_json::to_string_pretty(events)?;
    fs::write(path, payload)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eventdbx::schema::CreateSchemaInput;

    #[test]
    fn ensure_schema_for_mode_respects_modes() {
        let dir = tempdir().expect("tempdir should be created");
        let manager = SchemaManager::load(dir.path().join("schemas.json"))
            .expect("schema manager should load");

        ensure_schema_for_mode(RestrictMode::Off, &manager, "account")
            .expect("off mode bypasses schema checks");
        ensure_schema_for_mode(RestrictMode::Default, &manager, "account")
            .expect("default mode allows missing schema");

        let err = ensure_schema_for_mode(RestrictMode::Strict, &manager, "account")
            .expect_err("strict mode should reject missing schemas");
        assert!(err.to_string().contains("restrict=strict"));

        manager
            .create(CreateSchemaInput {
                aggregate: "account".into(),
                events: vec!["opened".into()],
                snapshot_threshold: None,
            })
            .expect("schema creation should succeed");

        ensure_schema_for_mode(RestrictMode::Strict, &manager, "account")
            .expect("strict mode accepts existing schema");
    }
}
