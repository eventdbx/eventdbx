use std::{collections::BTreeMap, env, path::PathBuf, str::FromStr};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Subcommand};
use serde_json::{Map as JsonMap, Value, json};

use eventdbx::{
    config::{Config, load_or_default},
    error::EventError,
    merkle::compute_merkle_root,
    reference::{
        AggregateReference, ReferenceContext, ReferenceIntegrity, ReferenceResolutionStatus,
    },
    schema::{MAX_EVENT_NOTE_LENGTH, SchemaManager},
    store::{
        self, AggregateCursor, AggregateQueryScope, AppendEvent, EventRecord, EventStore,
        payload_to_map,
    },
    token::{IssueTokenInput, JwtLimits, ROOT_ACTION, ROOT_RESOURCE, TokenManager},
    validation::{
        ensure_aggregate_id, ensure_metadata_extensions, ensure_payload_size, ensure_snake_case,
        normalize_event_type,
    },
};
use tracing::warn;

use crate::commands::{cli_token, client::ServerClient, is_lock_error_message};

#[derive(Subcommand)]
pub enum AggregateCommands {
    /// Create a new aggregate instance
    Create(AggregateCreateArgs),
    /// Apply an event to an aggregate instance
    Apply(AggregateApplyArgs),
    /// List aggregates in the store
    List(AggregateListArgs),
    /// Retrieve the state of an aggregate
    Get(AggregateGetArgs),
    /// Verify an aggregate's Merkle root
    Verify(AggregateVerifyArgs),
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

    /// JSON metadata with extension keys prefixed by '@'
    #[arg(long)]
    pub metadata: Option<String>,

    /// Optional note associated with the event
    #[arg(long, value_name = "NOTE")]
    pub note: Option<String>,

    /// Authorization token used when proxying through a running server
    #[arg(long, value_name = "TOKEN")]
    pub token: Option<String>,

    /// Emit results as JSON
    #[arg(long, default_value_t = false)]
    pub json: bool,
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

    /// Authorization token used when proxying through a running server
    #[arg(long, value_name = "TOKEN")]
    pub token: Option<String>,

    /// Raw JSON payload to use instead of key-value fields
    #[arg(long, value_name = "JSON")]
    pub payload: Option<String>,

    /// JSON metadata with extension keys prefixed by '@'
    #[arg(long)]
    pub metadata: Option<String>,

    /// Optional note associated with the event
    #[arg(long, value_name = "NOTE")]
    pub note: Option<String>,
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

    /// Emit results as JSON
    #[arg(long, default_value_t = false)]
    pub json: bool,
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

#[derive(Debug, Clone)]
pub struct KeyValue {
    pub key: String,
    pub value: String,
}

pub fn execute(config_path: Option<PathBuf>, command: AggregateCommands) -> Result<()> {
    let (config, _) = load_or_default(config_path)?;
    match command {
        AggregateCommands::Create(args) => create_aggregate(&config, args),
        AggregateCommands::Apply(args) => apply_event(&config, args),
        AggregateCommands::List(args) => list_aggregates(&config, args),
        AggregateCommands::Get(args) => get_aggregate(&config, args),
        AggregateCommands::Verify(args) => verify_aggregate(&config, args),
    }
}

fn create_aggregate(config: &Config, args: AggregateCreateArgs) -> Result<()> {
    let payload = parse_payload(args.payload, args.fields)?;
    let metadata = parse_metadata(args.metadata)?;
    validate_write_input(
        &args.aggregate,
        &args.aggregate_id,
        &payload,
        metadata.as_ref(),
        args.note.as_deref(),
    )?;

    let normalized = normalize_event_type(&args.event)?;
    let schema_manager = SchemaManager::load(config.schema_store_path())?;
    validate_schema_for_write(
        config,
        &schema_manager,
        &args.aggregate,
        &normalized.normalized,
        &payload,
    )?;

    match EventStore::open(
        config.event_store_path(),
        config.encryption_key()?,
        config.snowflake_worker_id,
    ) {
        Ok(store) => {
            if store
                .aggregate_version(&args.aggregate, &args.aggregate_id)?
                .is_some()
            {
                bail!(
                    "aggregate {}::{} already exists",
                    args.aggregate,
                    args.aggregate_id
                );
            }

            let (normalized_payload, reference_targets) = normalize_references_offline(
                &schema_manager,
                &store,
                config.active_domain(),
                &args.aggregate,
                payload,
            )?;
            let record = store.append(AppendEvent {
                aggregate_type: args.aggregate.clone(),
                aggregate_id: args.aggregate_id.clone(),
                event_type: normalized.normalized,
                event_type_raw: normalized.original,
                payload: normalized_payload,
                metadata,
                issued_by: None,
                note: args.note,
                tenant: config.active_domain().to_string(),
                reference_targets,
            })?;
            maybe_auto_snapshot(&store, &schema_manager, &record);
            let state = store.get_aggregate_state(&args.aggregate, &args.aggregate_id)?;
            render_create_state(config, state, args.json)
        }
        Err(EventError::Storage(message)) if is_lock_error_message(&message) => {
            let state = proxy_create_via_socket(
                config,
                args.token,
                &args.aggregate,
                &args.aggregate_id,
                &normalized.normalized,
                &payload,
                metadata.as_ref(),
                args.note.as_deref(),
            )?;
            if let Some(state) = state {
                render_create_state(config, state, args.json)?;
            } else {
                println!("Ok");
            }
            Ok(())
        }
        Err(err) => Err(err.into()),
    }
}

fn apply_event(config: &Config, args: AggregateApplyArgs) -> Result<()> {
    let payload = parse_payload(args.payload, args.fields)?;
    let metadata = parse_metadata(args.metadata)?;
    validate_write_input(
        &args.aggregate,
        &args.aggregate_id,
        &payload,
        metadata.as_ref(),
        args.note.as_deref(),
    )?;

    let normalized = normalize_event_type(&args.event)?;
    let schema_manager = SchemaManager::load(config.schema_store_path())?;
    validate_schema_for_write(
        config,
        &schema_manager,
        &args.aggregate,
        &normalized.normalized,
        &payload,
    )?;

    match EventStore::open(
        config.event_store_path(),
        config.encryption_key()?,
        config.snowflake_worker_id,
    ) {
        Ok(store) => {
            if store
                .aggregate_version(&args.aggregate, &args.aggregate_id)?
                .is_none()
            {
                bail!(
                    "aggregate {}::{} does not exist",
                    args.aggregate,
                    args.aggregate_id
                );
            }

            let (normalized_payload, reference_targets) = normalize_references_offline(
                &schema_manager,
                &store,
                config.active_domain(),
                &args.aggregate,
                payload,
            )?;
            let record = store.append(AppendEvent {
                aggregate_type: args.aggregate.clone(),
                aggregate_id: args.aggregate_id.clone(),
                event_type: normalized.normalized.clone(),
                event_type_raw: normalized.original.clone(),
                payload: normalized_payload,
                metadata,
                issued_by: None,
                note: args.note,
                tenant: config.active_domain().to_string(),
                reference_targets,
            })?;
            maybe_auto_snapshot(&store, &schema_manager, &record);
            render_append_result(config, record)
        }
        Err(EventError::Storage(message)) if is_lock_error_message(&message) => {
            let record = proxy_append_via_socket(
                config,
                args.token,
                &args.aggregate,
                &args.aggregate_id,
                &normalized.normalized,
                Some(&payload),
                metadata.as_ref(),
                args.note.as_deref(),
            )?;
            if let Some(record) = record {
                render_append_result(config, record)?;
            } else {
                println!("Ok");
            }
            Ok(())
        }
        Err(err) => Err(err.into()),
    }
}

fn list_aggregates(config: &Config, args: AggregateListArgs) -> Result<()> {
    let store = EventStore::open_read_only(config.event_store_path(), config.encryption_key()?)?;
    let take = args.take.unwrap_or(config.list_page_size);
    if take == 0 {
        bail!("--take must be greater than zero");
    }
    let cursor = match args.cursor.as_deref() {
        Some(raw) => Some(
            AggregateCursor::from_str(raw).with_context(|| format!("invalid cursor '{raw}'"))?,
        ),
        None => None,
    };
    let aggregate_filter = args.aggregate.as_deref();
    if let Some(name) = aggregate_filter {
        ensure_snake_case("aggregate_type", name)?;
    }
    let (aggregates, next_cursor) = store.aggregates_page_with_transform(
        cursor.as_ref(),
        take,
        AggregateQueryScope::ActiveOnly,
        true,
        |aggregate| {
            if aggregate_filter
                .map(|target| aggregate.aggregate_type == target)
                .unwrap_or(true)
            {
                Some(aggregate)
            } else {
                None
            }
        },
    )?;

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "items": aggregates,
                "next_cursor": next_cursor.map(|cursor| cursor.to_string()),
            }))?
        );
        return Ok(());
    }

    if aggregates.is_empty() {
        println!("no aggregates");
        return Ok(());
    }

    for aggregate in aggregates {
        println!(
            "aggregate_type={} aggregate_id={} version={} merkle_root={}",
            aggregate.aggregate_type,
            aggregate.aggregate_id,
            aggregate.version,
            aggregate.merkle_root
        );
    }
    if let Some(next_cursor) = next_cursor {
        println!("next_cursor={}", next_cursor);
    }
    Ok(())
}

fn get_aggregate(config: &Config, args: AggregateGetArgs) -> Result<()> {
    let store = EventStore::open_read_only(config.event_store_path(), config.encryption_key()?)?;
    let mut state = store.get_aggregate_state(&args.aggregate, &args.aggregate_id)?;
    let mut events_cache = None;

    if args.version.is_some() || args.include_events {
        events_cache = Some(store.list_events(&args.aggregate, &args.aggregate_id)?);
    }

    if let Some(version) = args.version {
        let events = events_cache
            .as_ref()
            .expect("events cache should be populated when version is requested");
        let (target_version, target_state, merkle_root) = state_at_version(events, version);
        state.version = target_version;
        state.state = target_state;
        state.merkle_root = merkle_root;
    }

    let mut output = json!({
        "aggregate_type": state.aggregate_type,
        "aggregate_id": state.aggregate_id,
        "version": state.version,
        "state": state.state,
        "merkle_root": state.merkle_root,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
    });

    if let Some(object) = output.as_object_mut() {
        for (key, value) in state.extensions {
            object.insert(key, value);
        }
    }

    if args.include_events {
        let events = events_cache.unwrap_or_else(|| {
            store
                .list_events(&args.aggregate, &args.aggregate_id)
                .expect("event listing should succeed after aggregate lookup")
        });
        let rendered_events: Vec<Value> = events
            .into_iter()
            .filter(|event| {
                args.version
                    .map(|version| event.version <= version)
                    .unwrap_or(true)
            })
            .map(|event| event.to_output_value())
            .collect::<serde_json::Result<_>>()?;
        output["events"] = Value::Array(rendered_events);
    }

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn verify_aggregate(config: &Config, args: AggregateVerifyArgs) -> Result<()> {
    let store = EventStore::open_read_only(config.event_store_path(), config.encryption_key()?)?;
    let merkle_root = store.verify(&args.aggregate, &args.aggregate_id)?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
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
    Ok(())
}

fn render_create_state(
    config: &Config,
    state: store::AggregateState,
    json_output: bool,
) -> Result<()> {
    if config.verbose_responses() {
        if json_output {
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

fn render_append_result(config: &Config, record: EventRecord) -> Result<()> {
    if config.verbose_responses() {
        println!(
            "{}",
            serde_json::to_string_pretty(&record.to_output_value()?)?
        );
    } else {
        println!("Ok");
    }
    Ok(())
}

fn parse_payload(payload: Option<String>, fields: Vec<KeyValue>) -> Result<Value> {
    if payload.is_some() && !fields.is_empty() {
        bail!("--payload cannot be used together with --field");
    }
    if let Some(raw) = payload {
        serde_json::from_str::<Value>(&raw)
            .with_context(|| "failed to parse JSON payload provided via --payload")
    } else {
        Ok(collect_payload(fields))
    }
}

fn parse_metadata(metadata: Option<String>) -> Result<Option<Value>> {
    metadata
        .map(|raw| {
            serde_json::from_str::<Value>(&raw)
                .with_context(|| "failed to parse JSON metadata provided via --metadata")
        })
        .transpose()
}

fn validate_write_input(
    aggregate: &str,
    aggregate_id: &str,
    payload: &Value,
    metadata: Option<&Value>,
    note: Option<&str>,
) -> Result<()> {
    ensure_snake_case("aggregate_type", aggregate)?;
    ensure_aggregate_id(aggregate_id)?;
    ensure_payload_size(payload)?;
    if let Some(metadata) = metadata {
        ensure_metadata_extensions(metadata)?;
    }
    if let Some(note) = note {
        if note.chars().count() > MAX_EVENT_NOTE_LENGTH {
            bail!("note cannot exceed {} characters", MAX_EVENT_NOTE_LENGTH);
        }
    }
    Ok(())
}

fn validate_schema_for_write(
    config: &Config,
    schemas: &SchemaManager,
    aggregate: &str,
    event: &str,
    payload: &Value,
) -> Result<()> {
    let schema_present = match schemas.get(aggregate) {
        Ok(_) => true,
        Err(EventError::SchemaNotFound) => false,
        Err(err) => return Err(err.into()),
    };

    if !schema_present && config.restrict.requires_declared_schema() {
        bail!(eventdbx::restrict::strict_mode_missing_schema_message(
            aggregate
        ));
    }
    if schema_present {
        schemas.validate_event(aggregate, event, payload)?;
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
    _tenant: &str,
    aggregate: &str,
    payload: Value,
) -> Result<(Value, Vec<(String, String)>)> {
    match schemas.get(aggregate) {
        Ok(_) => {}
        Err(EventError::SchemaNotFound) => return Ok((payload, Vec::new())),
        Err(err) => return Err(err.into()),
    };

    let context = ReferenceContext {
        aggregate_type: aggregate,
    };
    let mut resolver = |reference: &AggregateReference,
                        _integrity: ReferenceIntegrity|
     -> eventdbx::error::Result<ReferenceResolutionStatus> {
        match store.aggregate_version(&reference.aggregate_type, &reference.aggregate_id) {
            Ok(Some(_)) => Ok(ReferenceResolutionStatus::Ok),
            Ok(None) => Ok(ReferenceResolutionStatus::NotFound),
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

fn proxy_append_via_socket(
    config: &Config,
    token: Option<String>,
    aggregate: &str,
    aggregate_id: &str,
    event: &str,
    payload: Option<&Value>,
    metadata: Option<&Value>,
    note: Option<&str>,
) -> Result<Option<EventRecord>> {
    let token = ensure_proxy_token(config, token)?;
    let client = ServerClient::new(config)?;
    client
        .append_event(
            token.as_str(),
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
        })
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
) -> Result<Option<store::AggregateState>> {
    let token = ensure_proxy_token(config, token)?;
    let client = ServerClient::new(config)?;
    client
        .create_aggregate(
            token.as_str(),
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
        })
}

pub(crate) fn ensure_proxy_token(config: &Config, token: Option<String>) -> Result<String> {
    if let Some(token) = token.and_then(normalize_token) {
        return Ok(token);
    }
    if let Some(token) = env::var("EVENTDBX_TOKEN").ok().and_then(normalize_token) {
        return Ok(token);
    }
    match cli_token::ensure_bootstrap_token(config, None) {
        Ok(token) => Ok(token),
        Err(err) => {
            warn!(
                "failed to load CLI bootstrap token ({}); falling back to ephemeral token",
                err
            );
            issue_ephemeral_token(config)
        }
    }
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
    let record = manager.issue(IssueTokenInput {
        subject: format!("cli:{user}"),
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
    env::var("USER")
        .or_else(|_| env::var("USERNAME"))
        .ok()
        .and_then(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
        .unwrap_or_else(|| "local".to_string())
}

fn normalize_token(token: String) -> Option<String> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
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
    events: &[EventRecord],
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

    (last_version, state, compute_merkle_root(&hashes))
}
