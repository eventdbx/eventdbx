use std::{
    collections::BTreeMap,
    env, fs,
    io::Cursor,
    path::{Path, PathBuf},
    process::Stdio,
    str::FromStr,
    sync::{Arc, RwLock},
    time::Instant,
};

use anyhow::{Context, Result, anyhow};
use capnp::serialize::{OwnedSegments, write_message_to_words};
use capnp::{message::ReaderOptions, struct_list};
use capnp_futures::serialize::{read_message, try_read_message};
use futures::AsyncWriteExt;
use serde_json::{self, Value};
use tokio::{
    net::{TcpListener, TcpStream},
    process::Command,
    task::{JoinHandle, spawn_blocking},
    time::{Duration, sleep},
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use tracing::{debug, error, info, warn};

use crate::{
    cli_capnp::{cli_request, cli_response},
    config::{Config, PluginPayloadMode},
    control_capnp::{
        control_hello, control_hello_response, control_request, control_response, publish_target,
    },
    error::EventError,
    filter::{self, FilterExpr},
    observability,
    plugin::{JobPriority, PluginManager, PublishTarget},
    reference::{
        ReferenceContext, ReferenceResolutionStatus, ResolvedAggregate, ResolvedReference,
    },
    replication_noise::{
        FrameTransport, MAX_NOISE_FRAME_PAYLOAD, perform_server_handshake, read_session_frame,
        write_session_frame,
    },
    schema::AggregateSchema,
    schema_history::{PublishOptions, SchemaHistoryManager},
    service::{
        AppendEventInput, CoreContext, CreateAggregateInput, SetAggregateArchiveInput,
        normalize_optional_note,
    },
    snowflake::SnowflakeId,
    store::{
        self, AggregateCursor, AggregateQueryScope, AggregateSort, AggregateState, EventCursor,
        EventRecord,
    },
    tenant::{CoreProvider, normalize_tenant_id},
    token::{JwtClaims, TokenManager},
};

#[derive(Debug, Clone)]
pub struct CliCommandResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

enum ControlReply {
    ListAggregates {
        aggregates: Vec<AggregateState>,
        next_cursor: Option<String>,
        resolved_json: Option<String>,
    },
    ListReferrers {
        referrers_json: String,
    },
    GetAggregate {
        found: bool,
        aggregate_json: Option<String>,
        resolved_json: Option<String>,
    },
    ListEvents {
        aggregate_type: String,
        aggregate_id: String,
        events: Vec<EventRecord>,
        next_cursor: Option<String>,
    },
    AppendEvent {
        event_json: Option<String>,
    },
    VerifyAggregate(String),
    SelectAggregate {
        found: bool,
        selection_json: Option<String>,
    },
    CreateAggregate {
        aggregate_json: Option<String>,
    },
    SetAggregateArchive {
        aggregate_json: Option<String>,
    },
    CreateSnapshot {
        snapshot_json: String,
    },
    ListSnapshots {
        snapshots_json: String,
    },
    GetSnapshot {
        found: bool,
        snapshot_json: Option<String>,
    },
    ListSchemas(String),
    ReplaceSchemas(u32),
    TenantAssign {
        changed: bool,
        shard: String,
    },
    TenantUnassign {
        changed: bool,
    },
    TenantQuotaSet {
        changed: bool,
        quota_mb: Option<u64>,
    },
    TenantQuotaClear {
        changed: bool,
    },
    TenantQuotaRecalc {
        storage_bytes: u64,
    },
    TenantReload {
        reloaded: bool,
    },
    TenantSchemaPublish {
        version_id: String,
        activated: bool,
        skipped: bool,
    },
}

enum ControlCommand {
    ListAggregates {
        token: String,
        cursor: Option<String>,
        take: Option<usize>,
        filter: Option<FilterExpr>,
        sort: Option<Vec<AggregateSort>>,
        include_archived: bool,
        archived_only: bool,
        resolve: bool,
        resolve_depth: Option<usize>,
    },
    ListReferrers {
        token: String,
        aggregate_type: String,
        aggregate_id: String,
    },
    GetAggregate {
        token: String,
        aggregate_type: String,
        aggregate_id: String,
        resolve: bool,
        resolve_depth: Option<usize>,
    },
    ListEvents {
        token: String,
        aggregate_type: String,
        aggregate_id: String,
        cursor: Option<EventCursor>,
        take: Option<usize>,
    },
    AppendEvent {
        token: String,
        aggregate_type: String,
        aggregate_id: String,
        event_type: String,
        payload: Option<Value>,
        metadata: Option<Value>,
        note: Option<String>,
        publish: Vec<PublishTarget>,
    },
    PatchEvent {
        token: String,
        aggregate_type: String,
        aggregate_id: String,
        event_type: String,
        patch: Value,
        metadata: Option<Value>,
        note: Option<String>,
        publish: Vec<PublishTarget>,
    },
    VerifyAggregate {
        aggregate_type: String,
        aggregate_id: String,
    },
    SelectAggregate {
        token: String,
        aggregate_type: String,
        aggregate_id: String,
        fields: Vec<String>,
    },
    CreateAggregate {
        token: String,
        aggregate_type: String,
        aggregate_id: String,
        event_type: String,
        payload: Value,
        metadata: Option<Value>,
        note: Option<String>,
        publish: Vec<PublishTarget>,
    },
    SetAggregateArchive {
        token: String,
        aggregate_type: String,
        aggregate_id: String,
        archived: bool,
        note: Option<String>,
    },
    CreateSnapshot {
        token: String,
        aggregate_type: String,
        aggregate_id: String,
        comment: Option<String>,
    },
    ListSnapshots {
        token: String,
        aggregate_type: Option<String>,
        aggregate_id: Option<String>,
        version: Option<u64>,
    },
    GetSnapshot {
        token: String,
        snapshot_id: SnowflakeId,
    },
    ListSchemas {
        token: String,
    },
    ReplaceSchemas {
        token: String,
        schemas_json: String,
    },
    TenantAssign {
        token: String,
        tenant: String,
        shard: String,
    },
    TenantUnassign {
        token: String,
        tenant: String,
    },
    TenantQuotaSet {
        token: String,
        tenant: String,
        max_megabytes: u64,
    },
    TenantQuotaClear {
        token: String,
        tenant: String,
    },
    TenantQuotaRecalc {
        token: String,
        tenant: String,
    },
    TenantReload {
        token: String,
        tenant: String,
    },
    TenantSchemaPublish {
        token: String,
        tenant: String,
        reason: Option<String>,
        actor: Option<String>,
        labels: Vec<String>,
        activate: bool,
        force: bool,
        reload: bool,
    },
}

const CONTROL_PROTOCOL_VERSION: u16 = 1;
const CONTROL_RESPONSE_HEADROOM: usize = 4 * 1024;
const TENANT_MANAGE_ACTION: &str = "tenant.manage";

enum FirstMessage {
    Cli(capnp::message::Reader<OwnedSegments>),
    Control {
        protocol_version: u16,
        token_result: Result<String, String>,
        tenant_result: Result<Option<String>, String>,
        no_noise: bool,
    },
}

fn control_error_code(err: &EventError) -> &'static str {
    match err {
        EventError::InvalidToken => "invalid_token",
        EventError::TokenExpired => "token_expired",
        EventError::TokenLimitReached => "token_limit_reached",
        EventError::Unauthorized => "unauthorized",
        EventError::AggregateNotFound => "aggregate_not_found",
        EventError::AggregateArchived => "aggregate_archived",
        EventError::SchemaExists => "schema_exists",
        EventError::SchemaNotFound => "schema_not_found",
        EventError::InvalidSchema(_) => "invalid_schema",
        EventError::SchemaViolation(_) => "schema_violation",
        EventError::InvalidCursor(_) => "invalid_cursor",
        EventError::Config(_) => "config",
        EventError::Storage(_) => "storage",
        EventError::Io(_) => "io",
        EventError::Serialization(_) => "serialization",
        EventError::TenantQuotaExceeded(_) => "tenant_quota",
    }
}

async fn refresh_tenant_context(core_provider: Arc<dyn CoreProvider>, tenant: &str) -> bool {
    let mut attempt = 0usize;
    loop {
        match core_provider.core_for(tenant) {
            Ok(_) => return true,
            Err(EventError::Storage(message)) => {
                let lower = message.to_ascii_lowercase();
                if lower.contains("lock") && attempt < 5 {
                    attempt += 1;
                    sleep(Duration::from_millis(100 * attempt as u64)).await;
                    continue;
                }
                warn!(
                    tenant = %tenant,
                    attempt = attempt,
                    "tenant reload skipped; failed to open context: {}",
                    message
                );
                return false;
            }
            Err(err) => {
                warn!(
                    tenant = %tenant,
                    "tenant reload skipped; failed to open context: {}",
                    err
                );
                return false;
            }
        }
    }
}

fn control_command_name(command: &ControlCommand) -> &'static str {
    match command {
        ControlCommand::ListAggregates { .. } => "list_aggregates",
        ControlCommand::ListReferrers { .. } => "list_referrers",
        ControlCommand::GetAggregate { .. } => "get_aggregate",
        ControlCommand::ListEvents { .. } => "list_events",
        ControlCommand::AppendEvent { .. } => "append_event",
        ControlCommand::PatchEvent { .. } => "patch_event",
        ControlCommand::VerifyAggregate { .. } => "verify_aggregate",
        ControlCommand::SelectAggregate { .. } => "select_aggregate",
        ControlCommand::CreateAggregate { .. } => "create_aggregate",
        ControlCommand::SetAggregateArchive { .. } => "set_aggregate_archive",
        ControlCommand::CreateSnapshot { .. } => "create_snapshot",
        ControlCommand::ListSnapshots { .. } => "list_snapshots",
        ControlCommand::GetSnapshot { .. } => "get_snapshot",
        ControlCommand::ListSchemas { .. } => "list_schemas",
        ControlCommand::ReplaceSchemas { .. } => "replace_schemas",
        ControlCommand::TenantAssign { .. } => "tenant_assign",
        ControlCommand::TenantUnassign { .. } => "tenant_unassign",
        ControlCommand::TenantQuotaSet { .. } => "tenant_quota_set",
        ControlCommand::TenantQuotaClear { .. } => "tenant_quota_clear",
        ControlCommand::TenantQuotaRecalc { .. } => "tenant_quota_recalc",
        ControlCommand::TenantReload { .. } => "tenant_reload",
        ControlCommand::TenantSchemaPublish { .. } => "tenant_schema_publish",
    }
}

fn read_control_text(
    field: capnp::Result<capnp::text::Reader<'_>>,
    label: &str,
) -> std::result::Result<String, EventError> {
    let reader =
        field.map_err(|err| EventError::Serialization(format!("failed to read {label}: {err}")))?;
    reader
        .to_string()
        .map_err(|err| EventError::Serialization(format!("invalid utf-8 in {label}: {err}")))
}

#[cfg_attr(not(test), doc(hidden))]
pub mod test_support {
    use super::*;
    use tokio::io::{DuplexStream, ReadHalf, WriteHalf, duplex, split};
    use tokio_util::compat::{Compat, TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

    pub fn spawn_control_session(
        core_provider: Arc<dyn CoreProvider>,
        tokens: Arc<TokenManager>,
        shared_config: Arc<RwLock<Config>>,
    ) -> (
        Compat<WriteHalf<DuplexStream>>,
        Compat<ReadHalf<DuplexStream>>,
        tokio::task::JoinHandle<Result<()>>,
    ) {
        let (client, server) = duplex(4096);
        let (server_reader_raw, server_writer_raw) = split(server);
        let mut server_reader = server_reader_raw.compat();
        let mut server_writer = server_writer_raw.compat_write();
        let task = tokio::spawn(async move {
            let message = read_message(&mut server_reader, ReaderOptions::new())
                .await
                .context("failed to read control hello")?;
            let (protocol_version, token_result, tenant_result, no_noise) = {
                let hello = message
                    .get_root::<control_hello::Reader>()
                    .context("failed to decode control hello")?;
                let protocol_version = hello.get_protocol_version();
                let token_result = hello
                    .get_token()
                    .map_err(|err| format!("failed to read control token: {err}"))
                    .and_then(|reader| {
                        reader
                            .to_str()
                            .map_err(|err| format!("invalid UTF-8 in control token: {err}"))
                            .map(|value| value.trim().to_string())
                    });
                let tenant_result = hello
                    .get_tenant_id()
                    .map_err(|err| format!("failed to read control tenant id: {err}"))
                    .and_then(|reader| {
                        reader
                            .to_str()
                            .map_err(|err| format!("invalid UTF-8 in control tenant id: {err}"))
                            .map(|value| value.trim().to_string())
                    })
                    .map(|value| if value.is_empty() { None } else { Some(value) });
                (
                    protocol_version,
                    token_result,
                    tenant_result,
                    hello.get_no_noise(),
                )
            };
            drop(message);
            handle_control_handshake(
                protocol_version,
                token_result,
                tenant_result,
                no_noise,
                &mut server_reader,
                &mut server_writer,
                Arc::clone(&tokens),
                Arc::clone(&core_provider),
                shared_config,
            )
            .await
        });

        let (client_reader_raw, client_writer_raw) = split(client);
        let client_writer = client_writer_raw.compat_write();
        let client_reader = client_reader_raw.compat();

        (client_writer, client_reader, task)
    }
}

pub async fn start(
    bind_addr: &str,
    config_path: Arc<PathBuf>,
    tokens: Arc<TokenManager>,
    core_provider: Arc<dyn CoreProvider>,
    shared_config: Arc<RwLock<Config>>,
) -> Result<JoinHandle<()>> {
    let listener = TcpListener::bind(bind_addr)
        .await
        .with_context(|| format!("failed to bind CLI Cap'n Proto listener on {bind_addr}"))?;
    let display_addr = listener
        .local_addr()
        .map(|addr| addr.to_string())
        .unwrap_or_else(|_| bind_addr.to_string());
    info!("CLI Cap'n Proto server listening on {}", display_addr);

    let handle = tokio::spawn({
        let core_provider = Arc::clone(&core_provider);
        let tokens = Arc::clone(&tokens);
        let shared_config = Arc::clone(&shared_config);
        let config_path = Arc::clone(&config_path);
        async move {
            if let Err(err) =
                serve(listener, config_path, tokens, core_provider, shared_config).await
            {
                warn!("CLI proxy server terminated: {err:?}");
            }
        }
    });
    Ok(handle)
}

async fn serve(
    listener: TcpListener,
    config_path: Arc<PathBuf>,
    tokens: Arc<TokenManager>,
    core_provider: Arc<dyn CoreProvider>,
    shared_config: Arc<RwLock<Config>>,
) -> Result<()> {
    loop {
        let (stream, peer) = listener
            .accept()
            .await
            .context("failed to accept CLI proxy connection")?;
        let config_path = Arc::clone(&config_path);
        let core_provider = Arc::clone(&core_provider);
        let tokens = Arc::clone(&tokens);
        let shared_config = Arc::clone(&shared_config);
        tokio::spawn(async move {
            if let Err(err) = handle_connection(
                stream,
                config_path,
                Arc::clone(&tokens),
                Arc::clone(&core_provider),
                shared_config,
            )
            .await
            {
                warn!(target: "cli_proxy", peer = %peer, "CLI proxy connection error: {err:?}");
            }
        });
    }
}

async fn handle_connection(
    stream: TcpStream,
    config_path: Arc<PathBuf>,
    tokens: Arc<TokenManager>,
    core_provider: Arc<dyn CoreProvider>,
    shared_config: Arc<RwLock<Config>>,
) -> Result<()> {
    let (reader, writer) = stream.into_split();
    let mut reader = reader.compat();
    let mut writer = writer.compat_write();

    let first_message = match try_read_message(&mut reader, ReaderOptions::new()).await {
        Ok(Some(message)) => message,
        Ok(None) => return Ok(()),
        Err(err) => {
            return Err(anyhow::Error::new(err).context("failed to read initial message"));
        }
    };

    match classify_first_message(first_message)? {
        FirstMessage::Cli(message) => {
            handle_cli_loop(
                Some(message),
                &mut reader,
                &mut writer,
                Arc::clone(&config_path),
            )
            .await?;
            Ok(())
        }
        FirstMessage::Control {
            protocol_version,
            token_result,
            tenant_result,
            no_noise,
        } => {
            handle_control_handshake(
                protocol_version,
                token_result,
                tenant_result,
                no_noise,
                &mut reader,
                &mut writer,
                Arc::clone(&tokens),
                core_provider,
                shared_config,
            )
            .await
        }
    }
}

async fn handle_cli_loop<R, W>(
    mut pending: Option<capnp::message::Reader<OwnedSegments>>,
    reader: &mut R,
    writer: &mut W,
    config_path: Arc<PathBuf>,
) -> Result<()>
where
    R: futures::AsyncRead + Unpin,
    W: futures::AsyncWrite + Unpin,
{
    loop {
        let message = if let Some(message) = pending.take() {
            message
        } else {
            match try_read_message(&mut *reader, ReaderOptions::new()).await {
                Ok(Some(message)) => message,
                Ok(None) => break,
                Err(err) => {
                    return Err(anyhow::Error::new(err).context("failed to read CLI request"));
                }
            }
        };

        let response_result = process_request(message, Arc::clone(&config_path)).await;

        let response_bytes = {
            let mut response_message = capnp::message::Builder::new_default();
            {
                let mut response = response_message.init_root::<cli_response::Builder>();
                match response_result {
                    Ok(result) => {
                        response.set_exit_code(result.exit_code);
                        response.set_stdout(&result.stdout);
                        response.set_stderr(&result.stderr);
                    }
                    Err(err) => {
                        response.set_exit_code(-1);
                        response.set_stdout("");
                        response.set_stderr(&err.to_string());
                    }
                }
            }
            write_message_to_words(&response_message)
        };

        writer
            .write_all(&response_bytes)
            .await
            .context("failed to write CLI response")?;
        writer
            .flush()
            .await
            .context("failed to flush CLI response")?;
    }

    Ok(())
}

async fn handle_control_handshake<R, W>(
    protocol_version: u16,
    token_result: Result<String, String>,
    tenant_result: Result<Option<String>, String>,
    no_noise_requested: bool,
    reader: &mut R,
    writer: &mut W,
    tokens: Arc<TokenManager>,
    core_provider: Arc<dyn CoreProvider>,
    shared_config: Arc<RwLock<Config>>,
) -> Result<()>
where
    R: futures::AsyncRead + Unpin,
    W: futures::AsyncWrite + Unpin,
{
    let handshake_start = Instant::now();

    let mut accepted = true;
    let mut response_text = "ok".to_string();
    let mut claims: Option<JwtClaims> = None;
    let mut token: Option<String> = None;
    let mut session_tenant: Option<String> = None;
    let mut session_core: Option<CoreContext> = None;

    if protocol_version != CONTROL_PROTOCOL_VERSION {
        accepted = false;
        response_text = format!("unsupported control protocol version {}", protocol_version);
    }

    if accepted {
        match token_result {
            Err(message) => {
                accepted = false;
                response_text = message;
            }
            Ok(value) if value.trim().is_empty() => {
                accepted = false;
                response_text = "missing control token".to_string();
            }
            Ok(value) => match tokens.verify(&value) {
                Ok(verified) => {
                    claims = Some(verified);
                    token = Some(value);
                }
                Err(err) => {
                    accepted = false;
                    response_text = "invalid control token".to_string();
                    warn!("control handshake rejected due to invalid token: {}", err);
                }
            },
        }
    }

    let requested_tenant = if accepted {
        match tenant_result {
            Err(message) => {
                accepted = false;
                response_text = message;
                None
            }
            Ok(value) => value,
        }
    } else {
        None
    };

    let config_snapshot = shared_config
        .read()
        .expect("config lock poisoned while resolving tenant")
        .clone();
    let require_tenant = config_snapshot.multi_tenant();

    if accepted {
        let tenant = match requested_tenant.filter(|value| !value.is_empty()) {
            Some(value) => value,
            None if require_tenant => {
                accepted = false;
                response_text =
                    "tenantId is required when multi-tenant mode is enabled".to_string();
                String::new()
            }
            None => config_snapshot.active_domain().to_string(),
        };
        if accepted {
            if let Some(claims) = &claims {
                if !claims.allows_tenant(&tenant) {
                    accepted = false;
                    response_text = format!("token does not grant access to tenant '{}'", tenant);
                }
            }
        }
        if accepted {
            match core_provider.core_for(&tenant) {
                Ok(core) => {
                    session_core = Some(core);
                    session_tenant = Some(tenant);
                }
                Err(err) => {
                    accepted = false;
                    response_text = format!("failed to initialise tenant '{}': {}", tenant, err);
                }
            }
        }
    }

    let use_no_noise = no_noise_requested && config_snapshot.no_noise;
    if no_noise_requested && !config_snapshot.no_noise {
        debug!(
            "plaintext control channel requested but disabled by server config; continuing with Noise"
        );
    }

    let response_bytes = {
        let mut message = capnp::message::Builder::new_default();
        {
            let mut response = message.init_root::<control_hello_response::Builder>();
            response.set_accepted(accepted);
            response.set_message(&response_text);
            response.set_no_noise(use_no_noise);
        }
        write_message_to_words(&message)
    };

    writer
        .write_all(&response_bytes)
        .await
        .context("failed to write control hello response")?;
    writer
        .flush()
        .await
        .context("failed to flush control hello response")?;

    let duration = handshake_start.elapsed().as_secs_f64();
    observability::record_capnp_control_request(
        "hello",
        if accepted { "accepted" } else { "rejected" },
        duration,
    );

    if !accepted {
        println!("control handshake rejected: {}", response_text);
        return Ok(());
    }

    if let Some(claims) = &claims {
        debug!(
            subject = %claims.sub,
            token_group = %claims.group,
            token_user = %claims.user,
            tenant = %session_tenant.as_deref().unwrap_or("<unknown>"),
            "control handshake accepted"
        );
    }

    let token = token.expect("token must be present when handshake accepted");
    let tenant = session_tenant
        .as_deref()
        .expect("tenant must be present when handshake accepted")
        .to_string();
    let core = session_core.expect("core context must be present when handshake accepted");
    println!(
        "control handshake accepted for token subject {} (tenant={}){}",
        claims.as_ref().map(|c| c.sub.as_str()).unwrap_or("unknown"),
        tenant,
        if use_no_noise {
            " [no-noise plaintext]"
        } else {
            ""
        }
    );
    let mut transport = if use_no_noise {
        FrameTransport::plain()
    } else {
        FrameTransport::from(
            perform_server_handshake(reader, writer, token.as_bytes())
                .await
                .context("failed to establish encrypted control channel")?,
        )
    };

    handle_control_session(
        reader,
        writer,
        &mut transport,
        core,
        shared_config,
        tenant,
        core_provider,
    )
    .await
}

fn classify_first_message(message: capnp::message::Reader<OwnedSegments>) -> Result<FirstMessage> {
    if message
        .get_root::<cli_request::Reader>()
        .and_then(|request| request.get_args())
        .is_ok()
    {
        return Ok(FirstMessage::Cli(message));
    }

    if let Ok(hello) = message.get_root::<control_hello::Reader>() {
        let token_result = hello
            .get_token()
            .map_err(|err| format!("failed to read control token: {err}"))
            .and_then(|reader| {
                reader
                    .to_str()
                    .map_err(|err| format!("invalid UTF-8 in control token: {err}"))
                    .map(|value| value.trim().to_string())
            });
        let tenant_result = hello
            .get_tenant_id()
            .map_err(|err| format!("failed to read control tenant id: {err}"))
            .and_then(|reader| {
                reader
                    .to_str()
                    .map_err(|err| format!("invalid UTF-8 in control tenant id: {err}"))
                    .map(|value| value.trim().to_string())
            })
            .map(|value| if value.is_empty() { None } else { Some(value) });
        return Ok(FirstMessage::Control {
            protocol_version: hello.get_protocol_version(),
            token_result,
            tenant_result,
            no_noise: hello.get_no_noise(),
        });
    }

    Err(anyhow!("unexpected first message"))
}

async fn handle_control_session<R, W>(
    reader: &mut R,
    writer: &mut W,
    transport: &mut FrameTransport,
    core: CoreContext,
    shared_config: Arc<RwLock<Config>>,
    tenant: String,
    core_provider: Arc<dyn CoreProvider>,
) -> Result<()>
where
    R: futures::AsyncRead + Unpin,
    W: futures::AsyncWrite + Unpin,
{
    loop {
        let request_bytes = match read_session_frame(reader, transport).await? {
            Some(bytes) => bytes,
            None => break,
        };
        let mut cursor = Cursor::new(&request_bytes);
        let message = capnp::serialize::read_message(&mut cursor, ReaderOptions::new())
            .context("failed to decode control request")?;

        let request = message
            .get_root::<control_request::Reader>()
            .context("failed to decode control request")?;
        let request_id = request.get_id();
        let command_start = Instant::now();
        let (command_label, status_label, response_result) = match parse_control_command(request) {
            Ok(command) => {
                let label = control_command_name(&command);
                let result = execute_control_command(
                    command,
                    core.clone(),
                    Arc::clone(&shared_config),
                    tenant.as_str(),
                    Arc::clone(&core_provider),
                )
                .await;
                let status = match &result {
                    Ok(_) => "ok",
                    Err(err) => control_error_code(err),
                };
                (label, status, result)
            }
            Err(err) => {
                let status = control_error_code(&err);
                ("parse", status, Err(err))
            }
        };

        let duration = command_start.elapsed().as_secs_f64();
        observability::record_capnp_control_request(command_label, status_label, duration);

        let response_bytes = {
            let mut response_message = capnp::message::Builder::new_default();
            {
                let mut response = response_message.init_root::<control_response::Builder>();
                response.set_id(request_id);
                match response_result {
                    Ok(reply) => {
                        let payload = response.reborrow().init_payload();
                        match reply {
                            ControlReply::ListAggregates {
                                aggregates,
                                next_cursor,
                                resolved_json,
                            } => {
                                let encoded = encode_json_list_response(
                                    request_id,
                                    &aggregates,
                                    |payload, json| {
                                        let mut builder = payload.init_list_aggregates();
                                        builder.set_aggregates_json(json);
                                    },
                                )?;
                                if encoded.included < aggregates.len() {
                                    debug!(
                                        target: "cli_proxy",
                                        requested = aggregates.len(),
                                        included = encoded.included,
                                        "truncated list_aggregates response to satisfy control frame limit"
                                    );
                                }
                                let mut builder = payload.init_list_aggregates();
                                builder.set_aggregates_json(&encoded.json);
                                if let Some(cursor) = next_cursor.as_deref() {
                                    builder.set_has_next_cursor(true);
                                    builder.set_next_cursor(cursor);
                                } else {
                                    builder.set_has_next_cursor(false);
                                    builder.set_next_cursor("");
                                }
                                if let Some(resolved) = resolved_json {
                                    builder.set_has_resolved_json(true);
                                    builder.set_resolved_json(&resolved);
                                } else {
                                    builder.set_has_resolved_json(false);
                                    builder.set_resolved_json("");
                                }
                            }
                            ControlReply::ListReferrers { referrers_json } => {
                                let mut builder = payload.init_list_referrers();
                                builder.set_referrers_json(&referrers_json);
                            }
                            ControlReply::GetAggregate {
                                found,
                                aggregate_json,
                                resolved_json,
                            } => {
                                let mut builder = payload.init_get_aggregate();
                                builder.set_found(found);
                                if let Some(json) = aggregate_json {
                                    builder.set_aggregate_json(&json);
                                } else {
                                    builder.set_aggregate_json("");
                                }
                                if let Some(resolved) = resolved_json {
                                    builder.set_has_resolved_json(true);
                                    builder.set_resolved_json(&resolved);
                                } else {
                                    builder.set_has_resolved_json(false);
                                    builder.set_resolved_json("");
                                }
                            }
                            ControlReply::ListEvents {
                                aggregate_type,
                                aggregate_id,
                                events,
                                next_cursor,
                            } => {
                                let encoded = encode_json_list_response(
                                    request_id,
                                    &events,
                                    |payload, json| {
                                        let mut builder = payload.init_list_events();
                                        builder.set_events_json(json);
                                    },
                                )?;
                                if encoded.included < events.len() {
                                    debug!(
                                        target: "cli_proxy",
                                        aggregate_type = %aggregate_type,
                                        aggregate_id = %aggregate_id,
                                        requested = events.len(),
                                        included = encoded.included,
                                        "truncated list_events response to satisfy control frame limit"
                                    );
                                }
                                let mut builder = payload.init_list_events();
                                builder.set_events_json(&encoded.json);
                                if let Some(cursor) = next_cursor.as_deref() {
                                    builder.set_has_next_cursor(true);
                                    builder.set_next_cursor(cursor);
                                } else {
                                    builder.set_has_next_cursor(false);
                                    builder.set_next_cursor("");
                                }
                            }
                            ControlReply::AppendEvent { event_json } => {
                                let mut builder = payload.init_append_event();
                                if let Some(json) = event_json {
                                    builder.set_event_json(&json);
                                } else {
                                    builder.set_event_json("");
                                }
                            }
                            ControlReply::VerifyAggregate(merkle_root) => {
                                let mut builder = payload.init_verify_aggregate();
                                builder.set_merkle_root(&merkle_root);
                            }
                            ControlReply::SelectAggregate {
                                found,
                                selection_json,
                            } => {
                                let mut builder = payload.init_select_aggregate();
                                builder.set_found(found);
                                if let Some(json) = selection_json {
                                    builder.set_selection_json(&json);
                                } else {
                                    builder.set_selection_json("");
                                }
                            }
                            ControlReply::CreateAggregate { aggregate_json } => {
                                let mut builder = payload.init_create_aggregate();
                                if let Some(json) = aggregate_json {
                                    builder.set_aggregate_json(&json);
                                } else {
                                    builder.set_aggregate_json("");
                                }
                            }
                            ControlReply::SetAggregateArchive { aggregate_json } => {
                                let mut builder = payload.init_set_aggregate_archive();
                                if let Some(json) = aggregate_json {
                                    builder.set_aggregate_json(&json);
                                } else {
                                    builder.set_aggregate_json("");
                                }
                            }
                            ControlReply::CreateSnapshot { snapshot_json } => {
                                let mut builder = payload.init_create_snapshot();
                                builder.set_snapshot_json(&snapshot_json);
                            }
                            ControlReply::GetSnapshot {
                                found,
                                snapshot_json,
                            } => {
                                let mut builder = payload.init_get_snapshot();
                                builder.set_found(found);
                                if let Some(json) = snapshot_json {
                                    builder.set_snapshot_json(&json);
                                } else {
                                    builder.set_snapshot_json("");
                                }
                            }
                            ControlReply::ListSnapshots { snapshots_json } => {
                                let mut builder = payload.init_list_snapshots();
                                builder.set_snapshots_json(&snapshots_json);
                            }
                            ControlReply::ListSchemas(json) => {
                                let mut builder = payload.init_list_schemas();
                                builder.set_schemas_json(&json);
                            }
                            ControlReply::ReplaceSchemas(replaced) => {
                                let mut builder = payload.init_replace_schemas();
                                builder.set_replaced(replaced);
                            }
                            ControlReply::TenantAssign { changed, shard } => {
                                let mut builder = payload.init_tenant_assign();
                                builder.set_changed(changed);
                                builder.set_shard_id(&shard);
                            }
                            ControlReply::TenantUnassign { changed } => {
                                let mut builder = payload.init_tenant_unassign();
                                builder.set_changed(changed);
                            }
                            ControlReply::TenantQuotaSet { changed, quota_mb } => {
                                let mut builder = payload.init_tenant_quota_set();
                                builder.set_changed(changed);
                                if let Some(value) = quota_mb {
                                    builder.set_has_quota(true);
                                    builder.set_quota_mb(value);
                                } else {
                                    builder.set_has_quota(false);
                                    builder.set_quota_mb(0);
                                }
                            }
                            ControlReply::TenantQuotaClear { changed } => {
                                let mut builder = payload.init_tenant_quota_clear();
                                builder.set_changed(changed);
                            }
                            ControlReply::TenantQuotaRecalc { storage_bytes } => {
                                let mut builder = payload.init_tenant_quota_recalc();
                                builder.set_storage_bytes(storage_bytes);
                            }
                            ControlReply::TenantReload { reloaded } => {
                                let mut builder = payload.init_tenant_reload();
                                builder.set_reloaded(reloaded);
                            }
                            ControlReply::TenantSchemaPublish {
                                version_id,
                                activated,
                                skipped,
                            } => {
                                let mut builder = payload.init_tenant_schema_publish();
                                builder.set_version_id(&version_id);
                                builder.set_activated(activated);
                                builder.set_skipped(skipped);
                            }
                        }
                    }
                    Err(err) => {
                        let payload = response.reborrow().init_payload();
                        let mut builder = payload.init_error();
                        builder.set_code(control_error_code(&err));
                        builder.set_message(&err.to_string());
                    }
                }
            }
            write_message_to_words(&response_message)
        };

        write_session_frame(writer, transport, &response_bytes)
            .await
            .context("failed to send control response")?;
    }

    Ok(())
}

fn parse_control_command(
    request: control_request::Reader<'_>,
) -> std::result::Result<ControlCommand, EventError> {
    use control_request::payload;

    match request
        .get_payload()
        .which()
        .map_err(|err| EventError::Serialization(err.to_string()))?
    {
        payload::ListAggregates(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let cursor = if req.get_has_cursor() {
                let raw = read_control_text(req.get_cursor(), "cursor")?;
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            } else {
                None
            };
            let take = if req.get_has_take() {
                Some(usize::try_from(req.get_take()).map_err(|_| {
                    EventError::InvalidSchema("take exceeds platform limits".into())
                })?)
            } else {
                None
            };

            let filter = if req.get_has_filter() {
                let raw = read_control_text(req.get_filter(), "filter")?;
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(filter::parse_shorthand(trimmed).map_err(|err| {
                        EventError::InvalidSchema(format!("invalid filter expression: {err}"))
                    })?)
                }
            } else {
                None
            };

            let sort = if req.get_has_sort() {
                let raw = read_control_text(req.get_sort(), "sort")?;
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(AggregateSort::parse_directives(trimmed).map_err(|err| {
                        EventError::InvalidSchema(format!("invalid sort specification: {err}"))
                    })?)
                }
            } else {
                None
            };

            let include_archived = req.get_include_archived();
            let archived_only = req.get_archived_only();
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let resolve = req.get_resolve();
            let resolve_depth = if req.get_has_resolve_depth() {
                Some(req.get_resolve_depth() as usize)
            } else {
                None
            };

            Ok(ControlCommand::ListAggregates {
                token,
                cursor,
                take,
                filter,
                sort,
                include_archived,
                archived_only,
                resolve,
                resolve_depth,
            })
        }
        payload::ListReferrers(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?;
            let aggregate_type = read_control_text(req.get_aggregate_type(), "aggregate_type")?;
            let aggregate_id = read_control_text(req.get_aggregate_id(), "aggregate_id")?;
            Ok(ControlCommand::ListReferrers {
                token,
                aggregate_type,
                aggregate_id,
            })
        }
        payload::GetAggregate(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let aggregate_type = read_control_text(req.get_aggregate_type(), "aggregate_type")?;
            let aggregate_id = read_control_text(req.get_aggregate_id(), "aggregate_id")?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let resolve = req.get_resolve();
            let resolve_depth = if req.get_has_resolve_depth() {
                Some(req.get_resolve_depth() as usize)
            } else {
                None
            };

            Ok(ControlCommand::GetAggregate {
                token,
                aggregate_type,
                aggregate_id,
                resolve,
                resolve_depth,
            })
        }
        payload::ListEvents(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let aggregate_type = read_control_text(req.get_aggregate_type(), "aggregate_type")?;
            let aggregate_id = read_control_text(req.get_aggregate_id(), "aggregate_id")?;
            let cursor = if req.get_has_cursor() {
                let raw = read_control_text(req.get_cursor(), "cursor")?;
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(EventCursor::from_str(trimmed)?)
                }
            } else {
                None
            };
            let take = if req.get_has_take() {
                Some(usize::try_from(req.get_take()).map_err(|_| {
                    EventError::InvalidSchema("take exceeds platform limits".into())
                })?)
            } else {
                None
            };

            Ok(ControlCommand::ListEvents {
                token: read_control_text(req.get_token(), "token")?
                    .trim()
                    .to_string(),
                aggregate_type,
                aggregate_id,
                cursor,
                take,
            })
        }
        payload::AppendEvent(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let aggregate_type = read_control_text(req.get_aggregate_type(), "aggregate_type")?;
            let aggregate_id = read_control_text(req.get_aggregate_id(), "aggregate_id")?;
            let event_type = read_control_text(req.get_event_type(), "event_type")?;

            let payload_raw = read_control_text(req.get_payload_json(), "payload_json")?;
            let payload_trimmed = payload_raw.trim();
            let payload = if payload_trimmed.is_empty() {
                None
            } else {
                Some(
                    serde_json::from_str::<Value>(payload_trimmed).map_err(|err| {
                        EventError::InvalidSchema(format!("invalid payload_json: {err}"))
                    })?,
                )
            };

            let metadata = if req.get_has_metadata() {
                let metadata_raw = read_control_text(req.get_metadata_json(), "metadata_json")?;
                let metadata_trimmed = metadata_raw.trim();
                if metadata_trimmed.is_empty() {
                    None
                } else {
                    Some(
                        serde_json::from_str::<Value>(metadata_trimmed).map_err(|err| {
                            EventError::InvalidSchema(format!("invalid metadata_json: {err}"))
                        })?,
                    )
                }
            } else {
                None
            };

            let note = if req.get_has_note() {
                Some(read_control_text(req.get_note(), "note")?)
            } else {
                None
            };
            let publish = if req.get_has_publish_targets() {
                let targets = req.get_publish_targets().map_err(|err| {
                    EventError::Serialization(format!("failed to read publish_targets: {err}"))
                })?;
                decode_publish_targets(targets)?
            } else {
                Vec::new()
            };

            Ok(ControlCommand::AppendEvent {
                token,
                aggregate_type,
                aggregate_id,
                event_type,
                payload,
                metadata,
                note,
                publish,
            })
        }
        payload::PatchEvent(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let aggregate_type = read_control_text(req.get_aggregate_type(), "aggregate_type")?;
            let aggregate_id = read_control_text(req.get_aggregate_id(), "aggregate_id")?;
            let event_type = read_control_text(req.get_event_type(), "event_type")?;

            let patch_raw = read_control_text(req.get_patch_json(), "patch_json")?;
            let patch_trimmed = patch_raw.trim();
            let patch = if patch_trimmed.is_empty() {
                return Err(EventError::InvalidSchema(
                    "patch_json must be provided for patch events".into(),
                ));
            } else {
                serde_json::from_str::<Value>(patch_trimmed).map_err(|err| {
                    EventError::InvalidSchema(format!("invalid patch_json: {err}"))
                })?
            };

            let metadata = if req.get_has_metadata() {
                let metadata_raw = read_control_text(req.get_metadata_json(), "metadata_json")?;
                let metadata_trimmed = metadata_raw.trim();
                if metadata_trimmed.is_empty() {
                    None
                } else {
                    Some(
                        serde_json::from_str::<Value>(metadata_trimmed).map_err(|err| {
                            EventError::InvalidSchema(format!("invalid metadata_json: {err}"))
                        })?,
                    )
                }
            } else {
                None
            };

            let note = if req.get_has_note() {
                Some(read_control_text(req.get_note(), "note")?)
            } else {
                None
            };
            let publish = if req.get_has_publish_targets() {
                let targets = req.get_publish_targets().map_err(|err| {
                    EventError::Serialization(format!("failed to read publish_targets: {err}"))
                })?;
                decode_publish_targets(targets)?
            } else {
                Vec::new()
            };

            Ok(ControlCommand::PatchEvent {
                token,
                aggregate_type,
                aggregate_id,
                event_type,
                patch,
                metadata,
                note,
                publish,
            })
        }
        payload::CreateAggregate(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let aggregate_type = read_control_text(req.get_aggregate_type(), "aggregate_type")?;
            let aggregate_id = read_control_text(req.get_aggregate_id(), "aggregate_id")?;
            let event_type = read_control_text(req.get_event_type(), "event_type")?;

            let payload_raw = read_control_text(req.get_payload_json(), "payload_json")?;
            let payload_trimmed = payload_raw.trim();
            if payload_trimmed.is_empty() {
                return Err(EventError::InvalidSchema(
                    "payload_json must be provided when creating an aggregate".into(),
                ));
            }
            let payload = serde_json::from_str::<Value>(payload_trimmed)
                .map_err(|err| EventError::InvalidSchema(format!("invalid payload_json: {err}")))?;

            let metadata = if req.get_has_metadata() {
                let metadata_raw = read_control_text(req.get_metadata_json(), "metadata_json")?;
                let metadata_trimmed = metadata_raw.trim();
                if metadata_trimmed.is_empty() {
                    None
                } else {
                    Some(
                        serde_json::from_str::<Value>(metadata_trimmed).map_err(|err| {
                            EventError::InvalidSchema(format!("invalid metadata_json: {err}"))
                        })?,
                    )
                }
            } else {
                None
            };

            let note = if req.get_has_note() {
                Some(read_control_text(req.get_note(), "note")?)
            } else {
                None
            };
            let publish = if req.get_has_publish_targets() {
                let targets = req.get_publish_targets().map_err(|err| {
                    EventError::Serialization(format!("failed to read publish_targets: {err}"))
                })?;
                decode_publish_targets(targets)?
            } else {
                Vec::new()
            };

            Ok(ControlCommand::CreateAggregate {
                token,
                aggregate_type,
                aggregate_id,
                event_type,
                payload,
                metadata,
                note,
                publish,
            })
        }
        payload::VerifyAggregate(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let aggregate_type = read_control_text(req.get_aggregate_type(), "aggregate_type")?;
            let aggregate_id = read_control_text(req.get_aggregate_id(), "aggregate_id")?;

            Ok(ControlCommand::VerifyAggregate {
                aggregate_type,
                aggregate_id,
            })
        }
        payload::SelectAggregate(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let aggregate_type = read_control_text(req.get_aggregate_type(), "aggregate_type")?;
            let aggregate_id = read_control_text(req.get_aggregate_id(), "aggregate_id")?;
            let fields_reader = req.get_fields().map_err(|err| {
                EventError::Serialization(format!("failed to read fields: {err}"))
            })?;
            let mut fields = Vec::new();
            for field in fields_reader.iter() {
                let value = field.map_err(|err| {
                    EventError::Serialization(format!("invalid utf-8 in fields entry: {err}"))
                })?;
                let field_str = value.to_str().map_err(|err| {
                    EventError::Serialization(format!("invalid utf-8 in fields entry: {err}"))
                })?;
                fields.push(field_str.to_string());
            }
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();

            Ok(ControlCommand::SelectAggregate {
                token,
                aggregate_type,
                aggregate_id,
                fields,
            })
        }
        payload::SetAggregateArchive(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let aggregate_type = read_control_text(req.get_aggregate_type(), "aggregate_type")?;
            let aggregate_id = read_control_text(req.get_aggregate_id(), "aggregate_id")?;
            let archived = req.get_archived();

            let note = normalize_optional_note(if req.get_has_note() {
                Some(read_control_text(req.get_note(), "note")?)
            } else {
                None
            });

            Ok(ControlCommand::SetAggregateArchive {
                token,
                aggregate_type,
                aggregate_id,
                archived,
                note,
            })
        }
        payload::CreateSnapshot(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let aggregate_type = read_control_text(req.get_aggregate_type(), "aggregate_type")?;
            let aggregate_id = read_control_text(req.get_aggregate_id(), "aggregate_id")?;
            let comment = if req.get_has_comment() {
                let value = read_control_text(req.get_comment(), "comment")?;
                let trimmed = value.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            } else {
                None
            };

            Ok(ControlCommand::CreateSnapshot {
                token,
                aggregate_type,
                aggregate_id,
                comment,
            })
        }
        payload::ListSnapshots(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let aggregate_type = if req.get_has_aggregate_type() {
                let value = read_control_text(req.get_aggregate_type(), "aggregate_type")?;
                let trimmed = value.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            } else {
                None
            };
            let aggregate_id = if req.get_has_aggregate_id() {
                let value = read_control_text(req.get_aggregate_id(), "aggregate_id")?;
                let trimmed = value.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            } else {
                None
            };
            let version = if req.get_has_version() {
                Some(req.get_version())
            } else {
                None
            };

            Ok(ControlCommand::ListSnapshots {
                token,
                aggregate_type,
                aggregate_id,
                version,
            })
        }
        payload::GetSnapshot(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            Ok(ControlCommand::GetSnapshot {
                token,
                snapshot_id: SnowflakeId::from_u64(req.get_snapshot_id()),
            })
        }
        payload::ListSchemas(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            Ok(ControlCommand::ListSchemas { token })
        }
        payload::ReplaceSchemas(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let schemas_json = read_control_text(req.get_schemas_json(), "schemas_json")?;
            Ok(ControlCommand::ReplaceSchemas {
                token,
                schemas_json,
            })
        }
        payload::TenantAssign(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let tenant = read_control_text(req.get_tenant_id(), "tenant_id")?;
            let shard = read_control_text(req.get_shard_id(), "shard_id")?;
            Ok(ControlCommand::TenantAssign {
                token,
                tenant,
                shard,
            })
        }
        payload::TenantUnassign(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let tenant = read_control_text(req.get_tenant_id(), "tenant_id")?;
            Ok(ControlCommand::TenantUnassign { token, tenant })
        }
        payload::TenantQuotaSet(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let tenant = read_control_text(req.get_tenant_id(), "tenant_id")?;
            let max_megabytes = req.get_max_storage_mb();
            Ok(ControlCommand::TenantQuotaSet {
                token,
                tenant,
                max_megabytes,
            })
        }
        payload::TenantQuotaClear(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let tenant = read_control_text(req.get_tenant_id(), "tenant_id")?;
            Ok(ControlCommand::TenantQuotaClear { token, tenant })
        }
        payload::TenantQuotaRecalc(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let tenant = read_control_text(req.get_tenant_id(), "tenant_id")?;
            Ok(ControlCommand::TenantQuotaRecalc { token, tenant })
        }
        payload::TenantReload(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let tenant = read_control_text(req.get_tenant_id(), "tenant_id")?;
            Ok(ControlCommand::TenantReload { token, tenant })
        }
        payload::TenantSchemaPublish(req) => {
            let req = req.map_err(|err| EventError::Serialization(err.to_string()))?;
            let token = read_control_text(req.get_token(), "token")?
                .trim()
                .to_string();
            let tenant = read_control_text(req.get_tenant_id(), "tenant_id")?;
            let reason = if req.get_has_reason() {
                Some(read_control_text(req.get_reason(), "reason")?)
            } else {
                None
            };
            let actor = if req.get_has_actor() {
                Some(read_control_text(req.get_actor(), "actor")?)
            } else {
                None
            };
            let labels_reader = req
                .get_labels()
                .map_err(|err| EventError::Serialization(err.to_string()))?;
            let mut labels = Vec::with_capacity(labels_reader.len() as usize);
            for entry in labels_reader.iter() {
                let value = entry
                    .map_err(|err| EventError::Serialization(err.to_string()))?
                    .to_string()
                    .map_err(|err| EventError::Serialization(err.to_string()))?;
                labels.push(value);
            }
            Ok(ControlCommand::TenantSchemaPublish {
                token,
                tenant,
                reason,
                actor,
                labels,
                activate: req.get_activate(),
                force: req.get_force(),
                reload: req.get_reload(),
            })
        }
    }
}

async fn execute_control_command(
    command: ControlCommand,
    core: CoreContext,
    shared_config: Arc<RwLock<Config>>,
    tenant_id: &str,
    core_provider: Arc<dyn CoreProvider>,
) -> std::result::Result<ControlReply, EventError> {
    let _tenant_id = tenant_id;

    fn resolve_cross_domain(
        core_provider: Arc<dyn CoreProvider>,
        token: &str,
        domain: &str,
        aggregate: AggregateState,
        depth: usize,
        stack: &mut Vec<(String, String, String)>,
    ) -> Result<ResolvedAggregate, EventError> {
        let core = core_provider.core_for(domain)?;
        let schemas = core.schemas();
        let context = ReferenceContext {
            domain,
            aggregate_type: &aggregate.aggregate_type,
        };
        let located =
            schemas.collect_references(&aggregate.aggregate_type, &aggregate.state, context)?;

        let mut resolved_refs = Vec::new();
        stack.push((
            domain.to_string(),
            aggregate.aggregate_type.clone(),
            aggregate.aggregate_id.clone(),
        ));

        for reference in located {
            if depth == 0 {
                resolved_refs.push(ResolvedReference {
                    path: reference.path,
                    reference: reference.reference,
                    status: ReferenceResolutionStatus::DepthExceeded,
                    resolved: None,
                });
                continue;
            }

            if stack.iter().any(|(d, t, id)| {
                d.eq_ignore_ascii_case(&reference.reference.domain)
                    && t == &reference.reference.aggregate_type
                    && id == &reference.reference.aggregate_id
            }) {
                resolved_refs.push(ResolvedReference {
                    path: reference.path,
                    reference: reference.reference,
                    status: ReferenceResolutionStatus::Cycle,
                    resolved: None,
                });
                continue;
            }

            let target_domain = reference.reference.domain.clone();
            let target_core = match core_provider.core_for(&target_domain) {
                Ok(core) => core,
                Err(_) => {
                    resolved_refs.push(ResolvedReference {
                        path: reference.path,
                        reference: reference.reference,
                        status: ReferenceResolutionStatus::Forbidden,
                        resolved: None,
                    });
                    continue;
                }
            };

            let resource = format!(
                "aggregate:{}:{}",
                reference.reference.aggregate_type, reference.reference.aggregate_id
            );
            if let Err(EventError::Unauthorized) = target_core.tokens().authorize_action(
                token,
                "aggregate.read",
                Some(resource.as_str()),
            ) {
                resolved_refs.push(ResolvedReference {
                    path: reference.path,
                    reference: reference.reference,
                    status: ReferenceResolutionStatus::Forbidden,
                    resolved: None,
                });
                continue;
            }

            match target_core.get_aggregate(
                token,
                &reference.reference.aggregate_type,
                &reference.reference.aggregate_id,
            ) {
                Ok(Some(child)) => {
                    let next = resolve_cross_domain(
                        Arc::clone(&core_provider),
                        token,
                        &target_domain,
                        child,
                        depth.saturating_sub(1),
                        stack,
                    )?;
                    resolved_refs.push(ResolvedReference {
                        path: reference.path,
                        reference: reference.reference,
                        status: ReferenceResolutionStatus::Ok,
                        resolved: Some(Box::new(next)),
                    });
                }
                Ok(None) => {
                    resolved_refs.push(ResolvedReference {
                        path: reference.path,
                        reference: reference.reference,
                        status: ReferenceResolutionStatus::NotFound,
                        resolved: None,
                    });
                }
                Err(EventError::AggregateNotFound) => {
                    resolved_refs.push(ResolvedReference {
                        path: reference.path,
                        reference: reference.reference,
                        status: ReferenceResolutionStatus::NotFound,
                        resolved: None,
                    });
                }
                Err(EventError::Unauthorized) => {
                    resolved_refs.push(ResolvedReference {
                        path: reference.path,
                        reference: reference.reference,
                        status: ReferenceResolutionStatus::Forbidden,
                        resolved: None,
                    });
                }
                Err(err) => return Err(err),
            }
        }

        stack.pop();
        Ok(ResolvedAggregate {
            domain: domain.to_string(),
            aggregate,
            references: resolved_refs,
        })
    }

    match command {
        ControlCommand::ListAggregates {
            token,
            cursor,
            take,
            filter,
            sort,
            include_archived,
            archived_only,
            resolve,
            resolve_depth,
        } => {
            let tenant_owned = tenant_id.to_string();
            let mut scope = if archived_only {
                AggregateQueryScope::ArchivedOnly
            } else if include_archived {
                AggregateQueryScope::IncludeArchived
            } else {
                AggregateQueryScope::ActiveOnly
            };
            if matches!(scope, AggregateQueryScope::ActiveOnly) {
                if let Some(expr) = filter.as_ref() {
                    if expr.references_field("archived") {
                        scope = AggregateQueryScope::IncludeArchived;
                    }
                }
            }

            let aggregates = spawn_blocking({
                let core = core.clone();
                let filter = filter;
                let sort = sort;
                let token = token.clone();
                let cursor = cursor.clone();
                let resolve = resolve;
                let resolve_depth = resolve_depth;
                let tenant_id = tenant_owned.clone();
                move || {
                    let sort_ref = sort.as_ref().map(|keys| keys.as_slice());
                    let timestamp_sort = sort_ref.and_then(store::timestamp_sort_hint);
                    if cursor.is_some() && sort.is_some() && timestamp_sort.is_none() {
                        return Err(EventError::InvalidCursor(
                            "cursor cannot be combined with sort directives".into(),
                        ));
                    }

                    let store = core.store();
                    let cursor_parsed = match cursor.as_deref() {
                        Some(raw) => {
                            let cursor = if let Some((kind, descending)) = timestamp_sort {
                                let expanded =
                                    store.parse_timestamp_cursor(raw, kind, descending)?;
                                store::ensure_timestamp_cursor(&expanded, kind, descending, scope)?;
                                expanded
                            } else {
                                AggregateCursor::from_str(raw)?
                            };
                            Some(cursor)
                        }
                        None => None,
                    };
                    let cursor_ref = cursor_parsed.as_ref();
                    let (aggregates, next_cursor) =
                        core.list_aggregates(&token, cursor_ref, take, filter, sort_ref, scope)?;

                    let resolved_json = if resolve {
                        let depth = resolve_depth
                            .unwrap_or(core.reference_default_depth())
                            .min(core.reference_max_depth());
                        let mut resolved = Vec::new();
                        for aggregate in &aggregates {
                            let mut stack = Vec::new();
                            let resolved_item = resolve_cross_domain(
                                Arc::clone(&core_provider),
                                &token,
                                &tenant_id,
                                aggregate.clone(),
                                depth,
                                &mut stack,
                            )?;
                            resolved.push(resolved_item);
                        }
                        Some(
                            serde_json::to_string(&resolved)
                                .map_err(|err| EventError::Serialization(err.to_string()))?,
                        )
                    } else {
                        None
                    };

                    Ok((aggregates, next_cursor, resolved_json))
                }
            })
            .await
            .map_err(|err| EventError::Storage(format!("list aggregates task failed: {err}")))?;
            let (aggregates, next_cursor, resolved_json) = aggregates?;
            Ok(ControlReply::ListAggregates {
                aggregates,
                next_cursor: next_cursor.map(|cursor| cursor.encode()),
                resolved_json,
            })
        }
        ControlCommand::ListReferrers {
            token,
            aggregate_type,
            aggregate_id,
        } => {
            let referrers = spawn_blocking({
                let core = core.clone();
                let token = token.clone();
                let aggregate_type = aggregate_type.clone();
                let aggregate_id = aggregate_id.clone();
                move || core.list_referrers(&token, &aggregate_type, &aggregate_id)
            })
            .await
            .map_err(|err| EventError::Storage(format!("list referrers task failed: {err}")))??;
            let referrers_json = serde_json::to_string(&referrers)
                .map_err(|err| EventError::Serialization(err.to_string()))?;
            Ok(ControlReply::ListReferrers { referrers_json })
        }
        ControlCommand::GetAggregate {
            token,
            aggregate_type,
            aggregate_id,
            resolve,
            resolve_depth,
        } => {
            let tenant_owned = tenant_id.to_string();
            let aggregate = spawn_blocking({
                let core = core.clone();
                let aggregate_type = aggregate_type.clone();
                let aggregate_id = aggregate_id.clone();
                let token = token.clone();
                let tenant_id = tenant_owned.clone();
                move || {
                    if resolve {
                        let aggregate =
                            core.get_aggregate(&token, &aggregate_type, &aggregate_id)?;
                        if let Some(aggregate) = aggregate {
                            let mut stack = Vec::new();
                            resolve_cross_domain(
                                Arc::clone(&core_provider),
                                &token,
                                &tenant_id,
                                aggregate,
                                resolve_depth
                                    .unwrap_or(core.reference_default_depth())
                                    .min(core.reference_max_depth()),
                                &mut stack,
                            )
                            .map(Some)
                        } else {
                            Ok(None)
                        }
                    } else {
                        core.get_aggregate(&token, &aggregate_type, &aggregate_id)
                            .map(|opt| {
                                opt.map(|aggregate| ResolvedAggregate {
                                    domain: core.tenant_id().to_string(),
                                    aggregate,
                                    references: Vec::new(),
                                })
                            })
                    }
                }
            })
            .await
            .map_err(|err| EventError::Storage(format!("get aggregate task failed: {err}")))??;

            let json = aggregate
                .as_ref()
                .map(|aggregate| serde_json::to_string(&aggregate.aggregate))
                .transpose()?;
            let resolved_json = aggregate
                .as_ref()
                .map(|aggregate| serde_json::to_string(aggregate))
                .transpose()?;

            Ok(ControlReply::GetAggregate {
                found: aggregate.is_some(),
                aggregate_json: json,
                resolved_json,
            })
        }
        ControlCommand::ListEvents {
            token,
            aggregate_type,
            aggregate_id,
            cursor,
            take,
        } => {
            let events = spawn_blocking({
                let core = core.clone();
                let aggregate_type = aggregate_type.clone();
                let aggregate_id = aggregate_id.clone();
                let token = token.clone();
                let cursor = cursor.clone();
                move || {
                    let cursor_ref = cursor.as_ref();
                    core.list_events(&token, &aggregate_type, &aggregate_id, cursor_ref, take)
                }
            })
            .await
            .map_err(|err| EventError::Storage(format!("list events task failed: {err}")))??;

            let (events, next_cursor) = events;
            Ok(ControlReply::ListEvents {
                aggregate_type,
                aggregate_id,
                events,
                next_cursor: next_cursor.map(|cursor| cursor.encode()),
            })
        }
        ControlCommand::AppendEvent {
            token,
            aggregate_type,
            aggregate_id,
            event_type,
            payload,
            metadata,
            note,
            publish,
        } => {
            let input = AppendEventInput {
                token,
                aggregate_type,
                aggregate_id,
                event_type,
                payload,
                patch: None,
                metadata,
                note,
                publish,
            };
            let verbose = config_verbose(&shared_config)?;
            handle_append_event_command(
                core.clone(),
                Arc::clone(&shared_config),
                input,
                "append event",
                verbose,
            )
            .await
        }
        ControlCommand::PatchEvent {
            token,
            aggregate_type,
            aggregate_id,
            event_type,
            patch,
            metadata,
            note,
            publish,
        } => {
            let input = AppendEventInput {
                token,
                aggregate_type,
                aggregate_id,
                event_type,
                payload: None,
                patch: Some(patch),
                metadata,
                note,
                publish,
            };
            let verbose = config_verbose(&shared_config)?;
            handle_append_event_command(
                core.clone(),
                Arc::clone(&shared_config),
                input,
                "append patch event",
                verbose,
            )
            .await
        }
        ControlCommand::VerifyAggregate {
            aggregate_type,
            aggregate_id,
        } => {
            let merkle = spawn_blocking({
                let core = core.clone();
                let aggregate_type = aggregate_type.clone();
                let aggregate_id = aggregate_id.clone();
                move || core.verify_aggregate(&aggregate_type, &aggregate_id)
            })
            .await
            .map_err(|err| EventError::Storage(format!("verify aggregate task failed: {err}")))??;

            Ok(ControlReply::VerifyAggregate(merkle))
        }
        ControlCommand::SelectAggregate {
            token,
            aggregate_type,
            aggregate_id,
            fields,
        } => {
            let selection = spawn_blocking({
                let core = core.clone();
                let aggregate_type = aggregate_type.clone();
                let aggregate_id = aggregate_id.clone();
                let fields = fields.clone();
                let token = token.clone();
                move || {
                    core.select_aggregate_fields(&token, &aggregate_type, &aggregate_id, &fields)
                }
            })
            .await
            .map_err(|err| EventError::Storage(format!("select aggregate task failed: {err}")))??;

            match selection {
                Some(map) => {
                    let json = serde_json::to_string(&map)?;
                    Ok(ControlReply::SelectAggregate {
                        found: true,
                        selection_json: Some(json),
                    })
                }
                None => Ok(ControlReply::SelectAggregate {
                    found: false,
                    selection_json: None,
                }),
            }
        }
        ControlCommand::CreateAggregate {
            token,
            aggregate_type,
            aggregate_id,
            event_type,
            payload,
            metadata,
            note,
            publish,
        } => {
            let verbose = config_verbose(&shared_config)?;
            let aggregate = spawn_blocking({
                let core = core.clone();
                let aggregate_type = aggregate_type.clone();
                let aggregate_id = aggregate_id.clone();
                let token = token.clone();
                let event_type = event_type.clone();
                let payload = payload.clone();
                let metadata = metadata.clone();
                let note = note.clone();
                let publish = publish.clone();
                move || {
                    core.create_aggregate(CreateAggregateInput {
                        token,
                        aggregate_type,
                        aggregate_id,
                        event_type,
                        payload,
                        metadata,
                        note,
                        publish,
                    })
                }
            })
            .await
            .map_err(|err| EventError::Storage(format!("create aggregate task failed: {err}")))??;

            let json = if verbose {
                Some(serde_json::to_string(&aggregate)?)
            } else {
                None
            };
            Ok(ControlReply::CreateAggregate {
                aggregate_json: json,
            })
        }
        ControlCommand::SetAggregateArchive {
            token,
            aggregate_type,
            aggregate_id,
            archived,
            note,
        } => {
            let verbose = config_verbose(&shared_config)?;
            let aggregate = spawn_blocking({
                let core = core.clone();
                let aggregate_type = aggregate_type.clone();
                let aggregate_id = aggregate_id.clone();
                let token = token.clone();
                let note = note.clone();
                move || {
                    core.set_aggregate_archive(SetAggregateArchiveInput {
                        token,
                        aggregate_type,
                        aggregate_id,
                        archived,
                        note,
                    })
                }
            })
            .await
            .map_err(|err| {
                EventError::Storage(format!("set aggregate archive task failed: {err}"))
            })??;

            let json = if verbose {
                Some(serde_json::to_string(&aggregate)?)
            } else {
                None
            };
            Ok(ControlReply::SetAggregateArchive {
                aggregate_json: json,
            })
        }
        ControlCommand::CreateSnapshot {
            token,
            aggregate_type,
            aggregate_id,
            comment,
        } => {
            let snapshot = spawn_blocking({
                let core = core.clone();
                let aggregate_type = aggregate_type.clone();
                let aggregate_id = aggregate_id.clone();
                let token = token.clone();
                let comment = comment.clone();
                move || core.create_snapshot(&token, &aggregate_type, &aggregate_id, comment)
            })
            .await
            .map_err(|err| EventError::Storage(format!("create snapshot task failed: {err}")))??;

            let snapshot_json = serde_json::to_string(&snapshot)?;
            Ok(ControlReply::CreateSnapshot { snapshot_json })
        }
        ControlCommand::ListSnapshots {
            token,
            aggregate_type,
            aggregate_id,
            version,
        } => {
            let snapshots = spawn_blocking({
                let core = core.clone();
                let aggregate_type = aggregate_type.clone();
                let aggregate_id = aggregate_id.clone();
                let token = token.clone();
                move || {
                    core.list_snapshots(
                        &token,
                        aggregate_type.as_deref(),
                        aggregate_id.as_deref(),
                        version,
                    )
                }
            })
            .await
            .map_err(|err| EventError::Storage(format!("list snapshots task failed: {err}")))??;

            let snapshots_json = serde_json::to_string(&snapshots)?;
            Ok(ControlReply::ListSnapshots { snapshots_json })
        }
        ControlCommand::GetSnapshot { token, snapshot_id } => {
            let snapshot = spawn_blocking({
                let core = core.clone();
                let token = token.clone();
                move || core.find_snapshot_by_id(&token, snapshot_id)
            })
            .await
            .map_err(|err| EventError::Storage(format!("get snapshot task failed: {err}")))??;

            let (found, snapshot_json) = if let Some(snapshot) = snapshot {
                (true, Some(serde_json::to_string(&snapshot)?))
            } else {
                (false, None)
            };
            Ok(ControlReply::GetSnapshot {
                found,
                snapshot_json,
            })
        }
        ControlCommand::ListSchemas { token } => {
            let tokens = core.tokens();
            let schemas = core.schemas();
            let token_clone = token.clone();
            let json = spawn_blocking(move || {
                tokens
                    .authorize_action(&token_clone, "schema.read", None)
                    .and_then(|_| {
                        let snapshot = schemas.snapshot();
                        serde_json::to_string(&snapshot)
                            .map_err(|err| EventError::Serialization(err.to_string()))
                    })
            })
            .await
            .map_err(|err| EventError::Storage(format!("list schemas task failed: {err}")))??;
            Ok(ControlReply::ListSchemas(json))
        }
        ControlCommand::ReplaceSchemas {
            token,
            schemas_json,
        } => {
            let tokens = core.tokens();
            let schemas = core.schemas();
            let token_clone = token.clone();
            let json = schemas_json.clone();
            let replaced = spawn_blocking(move || -> std::result::Result<u32, EventError> {
                tokens.authorize_action(&token_clone, "schema.write", None)?;
                let map: BTreeMap<String, AggregateSchema> = serde_json::from_str(&json)
                    .map_err(|err| EventError::Serialization(err.to_string()))?;
                let count = map.len() as u32;
                schemas.replace_all(map)?;
                Ok(count)
            })
            .await
            .map_err(|err| EventError::Storage(format!("replace schemas task failed: {err}")))??;
            Ok(ControlReply::ReplaceSchemas(replaced))
        }
        ControlCommand::TenantAssign {
            token,
            tenant,
            shard,
        } => {
            let tenant = normalize_tenant_id(&tenant)?;
            authorize_tenant_admin(&core, &token, &tenant)?;
            let assignments = core.assignments();
            let changed = assignments.assign(&tenant, &shard)?;
            core_provider.invalidate_tenant(&tenant);
            Ok(ControlReply::TenantAssign { changed, shard })
        }
        ControlCommand::TenantUnassign { token, tenant } => {
            let tenant = normalize_tenant_id(&tenant)?;
            authorize_tenant_admin(&core, &token, &tenant)?;
            let assignments = core.assignments();
            let changed = assignments.unassign(&tenant)?;
            core_provider.invalidate_tenant(&tenant);
            Ok(ControlReply::TenantUnassign { changed })
        }
        ControlCommand::TenantQuotaSet {
            token,
            tenant,
            max_megabytes,
        } => {
            let tenant = normalize_tenant_id(&tenant)?;
            authorize_tenant_admin(&core, &token, &tenant)?;
            let assignments = core.assignments();
            let changed = assignments.set_quota(&tenant, Some(max_megabytes))?;
            core_provider.invalidate_tenant(&tenant);
            Ok(ControlReply::TenantQuotaSet {
                changed,
                quota_mb: Some(max_megabytes),
            })
        }
        ControlCommand::TenantQuotaClear { token, tenant } => {
            let tenant = normalize_tenant_id(&tenant)?;
            authorize_tenant_admin(&core, &token, &tenant)?;
            let assignments = core.assignments();
            let changed = assignments.set_quota(&tenant, None)?;
            core_provider.invalidate_tenant(&tenant);
            Ok(ControlReply::TenantQuotaClear { changed })
        }
        ControlCommand::TenantQuotaRecalc { token, tenant } => {
            let tenant = normalize_tenant_id(&tenant)?;
            authorize_tenant_admin(&core, &token, &tenant)?;
            let target_core = if tenant.eq_ignore_ascii_case(core.tenant_id()) {
                core.clone()
            } else {
                core_provider.core_for(&tenant)?
            };
            let store = target_core.store();
            let usage_bytes = spawn_blocking(move || store.storage_usage_bytes())
                .await
                .map_err(|err| {
                    EventError::Storage(format!("storage usage task failed: {err}"))
                })??;
            let assignments = target_core.assignments();
            assignments.update_storage_usage_bytes(&tenant, usage_bytes)?;
            Ok(ControlReply::TenantQuotaRecalc {
                storage_bytes: usage_bytes,
            })
        }
        ControlCommand::TenantReload { token, tenant } => {
            let tenant = normalize_tenant_id(&tenant)?;
            authorize_tenant_admin(&core, &token, &tenant)?;
            drop(core);
            core_provider.invalidate_tenant(&tenant);
            let reloaded = refresh_tenant_context(Arc::clone(&core_provider), &tenant).await;
            Ok(ControlReply::TenantReload { reloaded })
        }
        ControlCommand::TenantSchemaPublish {
            token,
            tenant,
            reason,
            actor,
            labels,
            activate,
            force,
            reload,
        } => {
            let tenant = normalize_tenant_id(&tenant)?;
            authorize_tenant_admin(&core, &token, &tenant)?;
            let domain_root = {
                let guard = shared_config
                    .read()
                    .map_err(|_| EventError::Storage("failed to acquire config lock".into()))?;
                guard.domain_data_dir_for(&tenant)
            };
            drop(core);
            let manager = SchemaHistoryManager::new(&domain_root);
            let payload =
                fs::read_to_string(manager.active_schema_path()).map_err(EventError::Io)?;
            let outcome = manager.publish(PublishOptions {
                schema_json: &payload,
                actor: actor.as_deref(),
                reason: reason.as_deref(),
                labels: labels.as_slice(),
                activate,
                skip_if_identical: !force,
            })?;
            drop(manager);
            if reload {
                core_provider.invalidate_tenant(&tenant);
                let _ = refresh_tenant_context(Arc::clone(&core_provider), &tenant).await;
            }
            Ok(ControlReply::TenantSchemaPublish {
                version_id: outcome.version_id,
                activated: outcome.activated,
                skipped: outcome.skipped,
            })
        }
    }
}

struct EncodedJsonList {
    json: String,
    included: usize,
}

fn encode_json_list_response<T, F>(
    request_id: u64,
    items: &[T],
    mut set_payload: F,
) -> std::result::Result<EncodedJsonList, EventError>
where
    T: serde::Serialize,
    F: FnMut(control_response::payload::Builder<'_>, &str),
{
    let max_payload_len = MAX_NOISE_FRAME_PAYLOAD.saturating_sub(CONTROL_RESPONSE_HEADROOM);
    if max_payload_len == 0 {
        return Err(EventError::Serialization(
            "control frame limit too small to encode responses".to_string(),
        ));
    }

    if items.is_empty() {
        let empty_json = "[]".to_string();
        ensure_response_size_with_payload(
            request_id,
            &empty_json,
            max_payload_len,
            &mut set_payload,
        )?;
        return Ok(EncodedJsonList {
            json: empty_json,
            included: 0,
        });
    }

    let mut low = 1usize;
    let mut high = items.len();
    let mut best: Option<(usize, String)> = None;

    while low <= high {
        let mid = (low + high) / 2;
        let json = serde_json::to_string(&items[..mid])?;
        match ensure_response_size_with_payload(
            request_id,
            &json,
            max_payload_len,
            &mut set_payload,
        ) {
            Ok(()) => {
                best = Some((mid, json));
                low = mid + 1;
            }
            Err(err) => {
                if mid == 1 {
                    return Err(err);
                }
                high = mid - 1;
            }
        }
    }

    match best {
        Some((included, json)) => Ok(EncodedJsonList { json, included }),
        None => Ok(EncodedJsonList {
            json: "[]".to_string(),
            included: 0,
        }),
    }
}

fn ensure_response_size_with_payload<F>(
    request_id: u64,
    json: &str,
    max_payload_len: usize,
    set_payload: &mut F,
) -> std::result::Result<(), EventError>
where
    F: FnMut(control_response::payload::Builder<'_>, &str),
{
    let mut message = capnp::message::Builder::new_default();
    {
        let mut response = message.init_root::<control_response::Builder>();
        response.set_id(request_id);
        let payload = response.reborrow().init_payload();
        set_payload(payload, json);
    }
    let words = write_message_to_words(&message);
    let byte_len = words.len() * 8;
    if byte_len > max_payload_len {
        return Err(EventError::Serialization(format!(
            "control response exceeds frame limit ({} bytes)",
            MAX_NOISE_FRAME_PAYLOAD
        )));
    }
    Ok(())
}

fn decode_publish_targets(
    list: struct_list::Reader<'_, publish_target::Owned>,
) -> std::result::Result<Vec<PublishTarget>, EventError> {
    let mut targets = Vec::with_capacity(list.len() as usize);
    for entry in list.iter() {
        let plugin_reader = entry
            .get_plugin()
            .map_err(|err| EventError::Serialization(err.to_string()))?;
        let plugin = plugin_reader
            .to_str()
            .map_err(|err| EventError::Serialization(err.to_string()))?
            .to_string();
        if plugin.trim().is_empty() {
            return Err(EventError::InvalidSchema(
                "publish target plugin cannot be empty".into(),
            ));
        }

        let mode = if entry.get_has_mode() {
            let raw = entry
                .get_mode()
                .map_err(|err| EventError::Serialization(err.to_string()))?
                .to_str()
                .map_err(|err| EventError::Serialization(err.to_string()))?;
            Some(parse_publish_mode(raw)?)
        } else {
            None
        };

        let priority = if entry.get_has_priority() {
            let raw = entry
                .get_priority()
                .map_err(|err| EventError::Serialization(err.to_string()))?
                .to_str()
                .map_err(|err| EventError::Serialization(err.to_string()))?;
            Some(parse_publish_priority(raw)?)
        } else {
            None
        };

        targets.push(PublishTarget {
            plugin,
            mode,
            priority,
        });
    }
    Ok(targets)
}

fn parse_publish_mode(raw: &str) -> std::result::Result<PluginPayloadMode, EventError> {
    match raw.to_ascii_lowercase().as_str() {
        "all" => Ok(PluginPayloadMode::All),
        "event-only" => Ok(PluginPayloadMode::EventOnly),
        "state-only" => Ok(PluginPayloadMode::StateOnly),
        "schema-only" => Ok(PluginPayloadMode::SchemaOnly),
        "event-and-schema" => Ok(PluginPayloadMode::EventAndSchema),
        "extensions-only" => Ok(PluginPayloadMode::ExtensionsOnly),
        other => Err(EventError::InvalidSchema(format!(
            "invalid publish mode '{}'",
            other
        ))),
    }
}

fn parse_publish_priority(raw: &str) -> std::result::Result<JobPriority, EventError> {
    match raw.to_ascii_lowercase().as_str() {
        "low" => Ok(JobPriority::Low),
        "normal" => Ok(JobPriority::Normal),
        "high" => Ok(JobPriority::High),
        other => Err(EventError::InvalidSchema(format!(
            "invalid publish priority '{}'",
            other
        ))),
    }
}

fn authorize_tenant_admin(
    core: &CoreContext,
    token: &str,
    tenant: &str,
) -> std::result::Result<(), EventError> {
    let claims = core
        .tokens()
        .authorize_action(token, TENANT_MANAGE_ACTION, None)?;
    if !claims.allows_tenant(tenant) {
        return Err(EventError::Unauthorized);
    }
    Ok(())
}

async fn handle_append_event_command(
    core: CoreContext,
    shared_config: Arc<RwLock<Config>>,
    input: AppendEventInput,
    task_label: &'static str,
    verbose: bool,
) -> std::result::Result<ControlReply, EventError> {
    let publish_targets = input.publish.clone();
    let input_for_task = input;
    let record = spawn_blocking({
        let core = core.clone();
        move || core.append_event(input_for_task)
    })
    .await
    .map_err(|err| EventError::Storage(format!("{task_label} task failed: {err}")))??;

    let record_json = if verbose {
        Some(serde_json::to_string(&record)?)
    } else {
        None
    };

    let schemas = core.schemas();
    if schemas.should_snapshot(&record.aggregate_type, record.version) {
        let aggregate_type = record.aggregate_type.clone();
        let aggregate_id = record.aggregate_id.clone();
        let version = record.version;
        let snapshot_result = spawn_blocking({
            let store = core.store();
            let aggregate_type_key = aggregate_type.clone();
            let aggregate_id_key = aggregate_id.clone();
            move || {
                store.create_snapshot(
                    &aggregate_type_key,
                    &aggregate_id_key,
                    Some(format!("auto snapshot v{}", version)),
                )
            }
        })
        .await
        .map_err(|err| EventError::Storage(format!("failed to create auto snapshot: {err}")))?;
        if let Err(err) = snapshot_result {
            warn!(
                target: "server",
                "failed to create auto snapshot for {}::{} v{}: {}",
                aggregate_type, aggregate_id, version, err
            );
        }
    }

    let plugins = {
        let guard = shared_config
            .read()
            .map_err(|_| EventError::Storage("failed to acquire config lock".into()))?;
        PluginManager::from_config(&*guard)?
    };

    if !plugins.is_empty() {
        let schemas = core.schemas();
        let schema = schemas.get(&record.aggregate_type).ok();
        let aggregate_type = record.aggregate_type.clone();
        let aggregate_id = record.aggregate_id.clone();
        let state_result = spawn_blocking({
            let store = core.store();
            let aggregate_type_key = aggregate_type.clone();
            let aggregate_id_key = aggregate_id.clone();
            move || store.get_aggregate_state(&aggregate_type_key, &aggregate_id_key)
        })
        .await
        .map_err(|err| EventError::Storage(format!("failed to load aggregate state: {err}")))?;

        match state_result {
            Ok(current_state) => {
                let publish_slice =
                    (!publish_targets.is_empty()).then_some(publish_targets.as_slice());
                if let Err(err) = plugins.notify_event_for_targets(
                    &record,
                    &current_state,
                    schema.as_ref(),
                    publish_slice,
                ) {
                    error!("plugin notification failed: {}", err);
                }
            }
            Err(err) => {
                warn!(
                    "plugin notification skipped (failed to load state for {}::{}): {}",
                    aggregate_type, aggregate_id, err
                );
            }
        }
    }

    Ok(ControlReply::AppendEvent {
        event_json: record_json,
    })
}

fn config_verbose(shared_config: &Arc<RwLock<Config>>) -> std::result::Result<bool, EventError> {
    let guard = shared_config
        .read()
        .map_err(|_| EventError::Storage("failed to acquire config lock".into()))?;
    Ok(guard.verbose_responses())
}

async fn process_request(
    message: capnp::message::Reader<OwnedSegments>,
    config_path: Arc<PathBuf>,
) -> Result<CliCommandResult> {
    let request = message
        .get_root::<cli_request::Reader>()
        .context("failed to decode CLI request")?;
    let args = {
        let args_reader = request
            .get_args()
            .context("failed to read CLI request arguments")?;

        let mut collected = Vec::with_capacity(args_reader.len() as usize);
        for arg in args_reader.iter() {
            let value = arg.context("failed to read CLI argument")?;
            collected.push(value.to_string()?);
        }
        collected
    };

    execute_cli_command(args, &config_path).await
}

async fn execute_cli_command(args: Vec<String>, config_path: &PathBuf) -> Result<CliCommandResult> {
    let exe = resolve_cli_executable().context("failed to resolve CLI executable")?;

    let command_label = args.first().cloned().unwrap_or_else(|| "help".to_string());
    let final_args = augment_args_with_config(args, config_path);

    let mut command = Command::new(exe);
    command.args(&final_args);
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());

    let start = Instant::now();
    let output = match command.output().await {
        Ok(output) => output,
        Err(err) => {
            let duration = start.elapsed().as_secs_f64();
            observability::record_cli_proxy_command(&command_label, "spawn_error", None, duration);
            return Err(anyhow!("failed to execute CLI command: {err}"));
        }
    };

    let exit_code = output
        .status
        .code()
        .unwrap_or_else(|| if output.status.success() { 0 } else { -1 });

    let status_label = if output.status.success() {
        "ok"
    } else {
        "exit"
    };
    let duration = start.elapsed().as_secs_f64();
    observability::record_cli_proxy_command(
        &command_label,
        status_label,
        Some(exit_code),
        duration,
    );

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    debug!(
        target: "cli_proxy",
        args = ?final_args,
        exit_code,
        "CLI command executed"
    );

    Ok(CliCommandResult {
        exit_code,
        stdout,
        stderr,
    })
}

fn resolve_cli_executable() -> Result<PathBuf> {
    if let Ok(path) = env::var("EVENTDBX_CLI") {
        return Ok(PathBuf::from(path));
    }
    if let Ok(path) = env::var("CARGO_BIN_EXE_eventdbx") {
        return Ok(PathBuf::from(path));
    }
    if let Ok(path) = env::var("CARGO_BIN_EXE_dbx") {
        return Ok(PathBuf::from(path));
    }
    let current = std::env::current_exe().context("failed to resolve current executable")?;
    if let Some(dir) = current.parent() {
        if let Some(candidate) = probe_dir_for_cli(dir) {
            return Ok(candidate);
        }
        if let Some(parent) = dir.parent() {
            if let Some(candidate) = probe_dir_for_cli(parent) {
                return Ok(candidate);
            }
        }
    }
    Ok(current)
}

fn probe_dir_for_cli(dir: &Path) -> Option<PathBuf> {
    let candidates = ["eventdbx", "dbx"];
    for candidate in candidates {
        let unix_path = dir.join(candidate);
        if unix_path.exists() {
            return Some(unix_path);
        }
    }
    #[cfg(windows)]
    {
        let candidates = ["eventdbx.exe", "dbx.exe"];
        for candidate in candidates {
            let windows_path = dir.join(candidate);
            if windows_path.exists() {
                return Some(windows_path);
            }
        }
    }
    None
}

fn augment_args_with_config(mut args: Vec<String>, config_path: &PathBuf) -> Vec<String> {
    if has_config_arg(&args) {
        return args;
    }

    let mut final_args = Vec::with_capacity(args.len() + 2);
    final_args.push("--config".to_string());
    final_args.push(config_path.to_string_lossy().into_owned());
    final_args.extend(args.drain(..));
    final_args
}

fn has_config_arg(args: &[String]) -> bool {
    args.iter()
        .any(|arg| arg == "--config" || arg.starts_with("--config="))
}

pub async fn invoke(args: &[String], addr: &str) -> Result<CliCommandResult> {
    let stream = TcpStream::connect(addr)
        .await
        .with_context(|| format!("failed to connect to CLI proxy at {addr}"))?;
    let (reader, writer) = stream.into_split();
    let mut writer = writer.compat_write();
    let message_bytes = {
        let mut message = capnp::message::Builder::new_default();
        {
            let mut request = message.init_root::<cli_request::Builder>();
            let mut list = request.init_args(args.len() as u32);
            for (idx, arg) in args.iter().enumerate() {
                list.set(idx as u32, arg);
            }
        }
        write_message_to_words(&message)
    };

    writer
        .write_all(&message_bytes)
        .await
        .context("failed to send CLI request")?;
    writer
        .flush()
        .await
        .context("failed to flush CLI request")?;

    let mut reader = reader.compat();
    let response_message = read_message(&mut reader, ReaderOptions::new())
        .await
        .context("failed to read CLI response")?;
    let response = response_message
        .get_root::<cli_response::Reader>()
        .context("failed to decode CLI response")?;

    let stdout = response
        .get_stdout()
        .context("missing stdout field in CLI response")?
        .to_string()
        .map_err(|err| anyhow::Error::new(err).context("invalid UTF-8 in CLI stdout"))?;
    let stderr = response
        .get_stderr()
        .context("missing stderr field in CLI response")?
        .to_string()
        .map_err(|err| anyhow::Error::new(err).context("invalid UTF-8 in CLI stderr"))?;

    Ok(CliCommandResult {
        exit_code: response.get_exit_code(),
        stdout,
        stderr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use capnp::message::Builder;
    use tempfile::TempDir;

    #[test]
    fn probe_dir_prefers_eventdbx() {
        let dir = TempDir::new().unwrap();
        let eventdbx = dir.path().join("eventdbx");
        let dbx = dir.path().join("dbx");
        std::fs::write(&eventdbx, b"").unwrap();
        std::fs::write(&dbx, b"").unwrap();

        let resolved = probe_dir_for_cli(dir.path()).expect("expected CLI path");
        assert_eq!(resolved, eventdbx);
    }

    #[test]
    fn probe_dir_falls_back_to_dbx() {
        let dir = TempDir::new().unwrap();
        let dbx = dir.path().join("dbx");
        std::fs::write(&dbx, b"").unwrap();

        let resolved = probe_dir_for_cli(dir.path()).expect("expected CLI path");
        assert_eq!(resolved, dbx);
    }

    #[test]
    fn parses_publish_targets_from_append_control_request() -> crate::error::Result<()> {
        let mut message = Builder::new_default();
        {
            let mut request = message.init_root::<control_request::Builder>();
            request.set_id(42);
            let mut payload = request.reborrow().init_payload();
            let mut append = payload.reborrow().init_append_event();
            append.set_token("t");
            append.set_aggregate_type("animal");
            append.set_aggregate_id("a-1");
            append.set_event_type("created");
            append.set_payload_json("{}");
            append.set_has_note(false);
            append.set_note("");
            append.set_has_metadata(false);
            append.set_metadata_json("");

            let mut targets = append.reborrow().init_publish_targets(2);
            {
                let mut target = targets.reborrow().get(0);
                target.set_plugin("search-indexer");
                target.set_has_mode(true);
                target.set_mode("all");
                target.set_has_priority(true);
                target.set_priority("high");
            }
            {
                let mut target = targets.reborrow().get(1);
                target.set_plugin("analytics-engine");
                target.set_has_mode(false);
                target.set_has_priority(false);
                target.set_mode("");
                target.set_priority("");
            }
            append.set_has_publish_targets(true);
        }

        let words = write_message_to_words(&message);
        let mut words_slice: &[u8] = &words;
        let reader =
            capnp::serialize::read_message_from_flat_slice(&mut words_slice, ReaderOptions::new())
                .map_err(|err| EventError::Serialization(err.to_string()))?;
        let command = parse_control_command(
            reader
                .get_root::<control_request::Reader>()
                .map_err(|err| EventError::Serialization(err.to_string()))?,
        )?;

        match command {
            ControlCommand::AppendEvent { publish, .. } => {
                assert_eq!(publish.len(), 2);
                assert_eq!(publish[0].plugin, "search-indexer");
                assert_eq!(publish[0].mode, Some(PluginPayloadMode::All));
                assert_eq!(publish[0].priority, Some(JobPriority::High));
                assert_eq!(publish[1].plugin, "analytics-engine");
                assert!(publish[1].mode.is_none());
                assert!(publish[1].priority.is_none());
            }
            _ => panic!("unexpected command parsed"),
        }

        Ok(())
    }
}
