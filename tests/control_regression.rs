use std::sync::{Arc, RwLock};

use anyhow::{Context, Result, anyhow, bail};
use capnp::{message::Builder, message::ReaderOptions, serialize::write_message_to_words};
use capnp_futures::serialize::read_message;
use eventdbx::replication_noise::{
    FrameTransport, perform_client_handshake, read_session_frame, write_session_frame,
};
use eventdbx::{
    cli_proxy::test_support::spawn_control_session,
    config::Config,
    control_capnp::{control_hello, control_hello_response, control_request, control_response},
    schema::{CreateSchemaInput, SchemaManager},
    service::CoreContext,
    store::EventStore,
    tenant::{CoreProvider, StaticCoreProvider},
    tenant_store::TenantAssignmentStore,
    token::{IssueTokenInput, JwtLimits, TokenManager},
};
use futures::AsyncWriteExt;
use serde_json::{Value, json};
use tempfile::tempdir;
use tokio_util::compat::Compat;

const CONTROL_PROTOCOL_VERSION: u16 = 1;

async fn open_control_session(
    core_provider: Arc<dyn CoreProvider>,
    tokens: Arc<TokenManager>,
    shared_config: Arc<RwLock<Config>>,
    token: &str,
    tenant: &str,
    no_noise: bool,
) -> Result<(
    Compat<tokio::io::WriteHalf<tokio::io::DuplexStream>>,
    Compat<tokio::io::ReadHalf<tokio::io::DuplexStream>>,
    FrameTransport,
    tokio::task::JoinHandle<Result<()>>,
)> {
    let (mut writer, mut reader, task) =
        spawn_control_session(core_provider, tokens, shared_config);

    let mut hello_message = Builder::new_default();
    {
        let mut hello = hello_message.init_root::<control_hello::Builder>();
        hello.set_protocol_version(CONTROL_PROTOCOL_VERSION);
        hello.set_token(token);
        hello.set_tenant_id(tenant);
        hello.set_no_noise(no_noise);
    }
    let hello_bytes = write_message_to_words(&hello_message);
    writer
        .write_all(&hello_bytes)
        .await
        .context("failed to send control hello")?;
    writer
        .flush()
        .await
        .context("failed to flush control hello")?;

    let hello_response_message = read_message(&mut reader, ReaderOptions::new())
        .await
        .context("failed to read control hello response")?;
    let hello_response = hello_response_message
        .get_root::<control_hello_response::Reader>()
        .context("failed to decode control hello response")?;
    if !hello_response.get_accepted() {
        let reason = hello_response
            .get_message()
            .context("missing control hello message")?
            .to_str()
            .context("invalid UTF-8 in control hello message")?;
        bail!("control handshake rejected: {}", reason);
    }

    let transport = if hello_response.get_no_noise() {
        FrameTransport::plain()
    } else {
        FrameTransport::from(
            perform_client_handshake(&mut reader, &mut writer, token.as_bytes())
                .await
                .context("failed to establish encrypted control channel")?,
        )
    };
    if no_noise {
        assert!(
            matches!(&transport, FrameTransport::Plain),
            "server should honor no-noise request"
        );
    }

    Ok((writer, reader, transport, task))
}

#[tokio::test(flavor = "multi_thread")]
async fn control_capnp_regression_flows() -> Result<()> {
    let tempdir = tempdir().context("failed to create temp dir")?;
    let mut config = Config::default();
    config.data_dir = tempdir.path().to_path_buf();
    config.data_encryption_key = None;

    let tokens = Arc::new(TokenManager::load(
        config.jwt_manager_config()?,
        config.tokens_path(),
        config.jwt_revocations_path(),
        None,
    )?);
    let schemas = Arc::new(SchemaManager::load(config.schema_store_path())?);
    schemas.create(CreateSchemaInput {
        aggregate: "order".into(),
        events: vec!["order_created".into(), "order_status_updated".into()],
        snapshot_threshold: None,
    })?;
    let store = Arc::new(EventStore::open(
        config.event_store_path(),
        None,
        config.snowflake_worker_id,
    )?);
    let assignments = Arc::new(TenantAssignmentStore::open(config.tenant_meta_path())?);
    let usage =
        assignments.ensure_storage_usage_bytes("default", || store.storage_usage_bytes())?;

    let core = CoreContext::new(
        Arc::clone(&tokens),
        Arc::clone(&schemas),
        Arc::clone(&store),
        config.restrict,
        config.list_page_size,
        config.page_limit,
        "default",
        None,
        Some(usage),
        Arc::clone(&assignments),
        config.reference_default_depth,
        config.reference_max_depth,
    );
    let shared_config = Arc::new(RwLock::new(config.clone()));
    let core_provider: Arc<dyn CoreProvider> = Arc::new(StaticCoreProvider::new(core.clone()));

    let token_record = tokens.issue(IssueTokenInput {
        subject: "system:admin".into(),
        group: "system".into(),
        user: "admin".into(),
        actions: vec![
            "aggregate.create".into(),
            "aggregate.append".into(),
            "aggregate.archive".into(),
            "aggregate.read".into(),
            // Needed for the later plaintext tenant_reload section; existing encrypted
            // flows do not rely on this capability.
            "tenant.manage".into(),
        ],
        resources: vec!["*".to_string()],
        tenants: Vec::new(),
        ttl_secs: Some(3_600),
        not_before: None,
        issued_by: "control-test".into(),
        limits: JwtLimits::default(),
    })?;
    let token_value = token_record
        .token
        .clone()
        .expect("issued token missing value");

    let (mut writer, mut reader, mut transport, server_task) = open_control_session(
        Arc::clone(&core_provider),
        Arc::clone(&tokens),
        Arc::clone(&shared_config),
        &token_value,
        config.active_domain(),
        false,
    )
    .await?;

    let mut next_request_id: u64 = 1;

    let aggregates: Vec<Value> = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut list = payload.init_list_aggregates();
            list.set_token(&token_value);
            list.set_has_cursor(false);
            list.set_cursor("");
            list.set_take(0);
            list.set_has_take(false);
            list.set_has_sort(false);
            list.set_sort("");
            list.set_include_archived(false);
            list.set_archived_only(false);
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::ListAggregates(Ok(list)) => {
                let json = list
                    .get_aggregates_json()
                    .context("missing aggregates_json")?
                    .to_str()
                    .context("aggregates_json utf-8 error")?;
                let parsed: Vec<Value> =
                    serde_json::from_str(json).context("failed to parse aggregates_json")?;
                Ok(parsed)
            }
            control_response::payload::ListAggregates(Err(err)) => {
                Err(anyhow!("failed to decode list_aggregates response: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert!(aggregates.is_empty());
    next_request_id += 1;

    let event_record: Value = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut append = payload.init_append_event();
            append.set_token(&token_value);
            append.set_aggregate_type("order");
            append.set_aggregate_id("order-1");
            append.set_event_type("order_created");
            append.set_payload_json(
                r#"{"status":"created","firstName":"John","address":{"street":"123 Main","zipCode":"94107"}}"#,
            );
            append.set_metadata_json("");
            append.set_note("");
            append.set_has_note(false);
            append.set_has_metadata(false);
        },
        |response| {
            use control_response::payload;

            match response
                .get_payload()
                .which()
                .context("payload decode failed")?
            {
                payload::AppendEvent(Ok(append)) => {
                    let json = append
                        .get_event_json()
                        .context("missing event_json")?
                        .to_str()
                        .context("event_json utf-8 error")?;
                    let parsed: Value =
                        serde_json::from_str(json).context("failed to parse event_json")?;
                    Ok(parsed)
                }
                payload::AppendEvent(Err(err)) => {
                    Err(anyhow!("failed to decode append_event response: {err}"))
                }
                payload::Error(Ok(error)) => {
                    let code = error
                        .get_code()
                        .context("missing error code")?
                        .to_str()
                        .context("error code utf-8 error")?
                        .to_string();
                    let message = error
                        .get_message()
                        .context("missing error message")?
                        .to_str()
                        .context("error message utf-8 error")?
                        .to_string();
                    Err(anyhow!("patch_event returned {}: {}", code, message))
                }
                payload::Error(Err(err)) => Err(anyhow!("failed to decode error payload: {err}")),
                payload::GetAggregate(_) => {
                    Err(anyhow!("unexpected get_aggregate response to patch_event"))
                }
                payload::ListAggregates(_) => Err(anyhow!(
                    "unexpected list_aggregates response to patch_event"
                )),
                payload::ListEvents(_) => {
                    Err(anyhow!("unexpected list_events response to patch_event"))
                }
                payload::VerifyAggregate(_) => Err(anyhow!(
                    "unexpected verify_aggregate response to patch_event"
                )),
                payload::CreateAggregate(_) => Err(anyhow!(
                    "unexpected create_aggregate response to patch_event"
                )),
                payload::SelectAggregate(_) => Err(anyhow!(
                    "unexpected select_aggregate response to patch_event"
                )),
                payload::SetAggregateArchive(_) => Err(anyhow!(
                    "unexpected set_aggregate_archive response to patch_event"
                )),
                _ => Err(anyhow!(
                    "unexpected control response variant returned to patch_event"
                )),
            }
        },
    )
    .await?;
    assert_eq!(event_record["aggregate_type"], "order");
    assert_eq!(event_record["aggregate_id"], "order-1");
    assert_eq!(event_record["version"], 1);
    next_request_id += 1;

    let aggregates_after: Vec<Value> = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut list = payload.init_list_aggregates();
            list.set_token(&token_value);
            list.set_has_cursor(false);
            list.set_cursor("");
            list.set_take(10);
            list.set_has_take(true);
            list.set_has_sort(false);
            list.set_sort("");
            list.set_include_archived(false);
            list.set_archived_only(false);
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::ListAggregates(Ok(list)) => {
                let json = list
                    .get_aggregates_json()
                    .context("missing aggregates_json")?
                    .to_str()
                    .context("aggregates_json utf-8 error")?;
                let parsed: Vec<Value> =
                    serde_json::from_str(json).context("failed to parse aggregates_json")?;
                Ok(parsed)
            }
            control_response::payload::ListAggregates(Err(err)) => {
                Err(anyhow!("failed to decode list_aggregates response: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert_eq!(aggregates_after.len(), 1);
    assert_eq!(aggregates_after[0]["aggregate_type"], "order");
    assert_eq!(aggregates_after[0]["version"], 1);
    next_request_id += 1;

    let filtered_aggregates: Vec<Value> = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut list = payload.init_list_aggregates();
            list.set_token(&token_value);
            list.set_has_cursor(false);
            list.set_cursor("");
            list.set_take(10);
            list.set_has_take(true);
            list.set_has_sort(false);
            list.set_sort("");
            list.set_include_archived(false);
            list.set_archived_only(false);
            list.set_has_filter(true);
            list.set_filter(r#"aggregate_id = "order-1""#);
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::ListAggregates(Ok(list)) => {
                let json = list
                    .get_aggregates_json()
                    .context("missing aggregates_json")?
                    .to_str()
                    .context("aggregates_json utf-8 error")?;
                let parsed: Vec<Value> =
                    serde_json::from_str(json).context("failed to parse aggregates_json")?;
                Ok(parsed)
            }
            control_response::payload::ListAggregates(Err(err)) => {
                Err(anyhow!("failed to decode list_aggregates response: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert_eq!(filtered_aggregates.len(), 1);
    assert_eq!(filtered_aggregates[0]["aggregate_id"], "order-1");
    next_request_id += 1;

    let aggregate_detail = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut get = payload.init_get_aggregate();
            get.set_token(&token_value);
            get.set_aggregate_type("order");
            get.set_aggregate_id("order-1");
        },
        |response| {
            use control_response::payload;

            match response
                .get_payload()
                .which()
                .context("payload decode failed")?
            {
                payload::GetAggregate(Ok(get)) => {
                    let found = get.get_found();
                    let json = get
                        .get_aggregate_json()
                        .context("missing aggregate_json")?
                        .to_str()
                        .context("aggregate_json utf-8 error")?;
                    let parsed: Option<Value> = if json.trim().is_empty() {
                        None
                    } else {
                        Some(serde_json::from_str(json).context("failed to parse aggregate_json")?)
                    };
                    Ok((found, parsed))
                }
                payload::GetAggregate(Err(err)) => {
                    Err(anyhow!("failed to decode get_aggregate response: {err}"))
                }
                payload::Error(Ok(error)) => {
                    let code = error
                        .get_code()
                        .context("missing error code")?
                        .to_str()
                        .context("error code utf-8 error")?
                        .to_string();
                    let message = error
                        .get_message()
                        .context("missing error message")?
                        .to_str()
                        .context("error message utf-8 error")?
                        .to_string();
                    Err(anyhow!("get_aggregate returned {}: {}", code, message))
                }
                payload::Error(Err(err)) => Err(anyhow!("failed to decode error payload: {err}")),
                payload::AppendEvent(_) => {
                    Err(anyhow!("unexpected append_event response to get_aggregate"))
                }
                payload::ListAggregates(_) => Err(anyhow!(
                    "unexpected list_aggregates response to get_aggregate"
                )),
                payload::ListEvents(_) => {
                    Err(anyhow!("unexpected list_events response to get_aggregate"))
                }
                payload::VerifyAggregate(_) => Err(anyhow!(
                    "unexpected verify_aggregate response to get_aggregate"
                )),
                payload::CreateAggregate(_) => Err(anyhow!(
                    "unexpected create_aggregate response to get_aggregate"
                )),
                payload::SelectAggregate(_) => Err(anyhow!(
                    "unexpected select_aggregate response to get_aggregate"
                )),
                payload::SetAggregateArchive(_) => Err(anyhow!(
                    "unexpected set_aggregate_archive response to get_aggregate"
                )),
                _ => Err(anyhow!(
                    "unexpected control response variant returned to get_aggregate"
                )),
            }
        },
    )
    .await?;
    assert!(aggregate_detail.0);
    let aggregate_body = aggregate_detail
        .1
        .expect("aggregate payload expected when found");
    assert_eq!(aggregate_body["aggregate_type"], "order");
    assert_eq!(aggregate_body["state"]["status"], "created");
    next_request_id += 1;

    let events: Vec<Value> = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut list = payload.init_list_events();
            list.set_token(&token_value);
            list.set_aggregate_type("order");
            list.set_aggregate_id("order-1");
            list.set_has_cursor(false);
            list.set_cursor("");
            list.set_take(25);
            list.set_has_take(true);
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::ListEvents(Ok(list)) => {
                let json = list
                    .get_events_json()
                    .context("missing events_json")?
                    .to_str()
                    .context("events_json utf-8 error")?;
                let parsed: Vec<Value> =
                    serde_json::from_str(json).context("failed to parse events_json")?;
                Ok(parsed)
            }
            control_response::payload::ListEvents(Err(err)) => {
                Err(anyhow!("failed to decode list_events response: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["event_type"], "order_created");
    next_request_id += 1;

    let merkle = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut verify = payload.init_verify_aggregate();
            verify.set_aggregate_type("order");
            verify.set_aggregate_id("order-1");
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::VerifyAggregate(Ok(verify)) => {
                let root = verify
                    .get_merkle_root()
                    .context("missing merkle_root")?
                    .to_str()
                    .context("merkle_root utf-8 error")?
                    .to_string();
                Ok(root)
            }
            control_response::payload::VerifyAggregate(Err(err)) => {
                Err(anyhow!("failed to decode verify_aggregate response: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert!(!merkle.trim().is_empty());
    assert_eq!(merkle, event_record["merkle_root"]);
    next_request_id += 1;

    let missing = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut get = payload.init_get_aggregate();
            get.set_token(&token_value);
            get.set_aggregate_type("inventory");
            get.set_aggregate_id("sku-1");
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::GetAggregate(Ok(get)) => {
                let found = get.get_found();
                Ok(found)
            }
            control_response::payload::GetAggregate(Err(err)) => {
                Err(anyhow!("failed to decode get_aggregate response: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert!(!missing);
    next_request_id += 1;

    let error_response = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut append = payload.init_append_event();
            append.set_token("invalid-token");
            append.set_aggregate_type("order");
            append.set_aggregate_id("order-1");
            append.set_event_type("order_created");
            append.set_payload_json(r#"{"status":"ignored"}"#);
            append.set_metadata_json("");
            append.set_note("");
            append.set_has_note(false);
            append.set_has_metadata(false);
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::Error(Ok(error)) => {
                let code = error
                    .get_code()
                    .context("missing error code")?
                    .to_str()
                    .context("code utf-8 error")?
                    .to_string();
                let message = error
                    .get_message()
                    .context("missing error message")?
                    .to_str()
                    .context("message utf-8 error")?
                    .to_string();
                Ok((code, message))
            }
            control_response::payload::Error(Err(err)) => {
                Err(anyhow!("failed to decode control error payload: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert_eq!(error_response.0, "invalid_token");
    assert!(!error_response.1.is_empty());

    drop(writer);
    drop(reader);

    let outcome = server_task.await.context("control handler task panicked")?;
    outcome.context("control handler returned error")?;

    {
        let mut guard = shared_config
            .write()
            .expect("config lock poisoned while enabling no_noise for plaintext test");
        guard.no_noise = true;
    }

    let (mut plain_writer, mut plain_reader, mut plain_transport, plain_task) =
        open_control_session(
            Arc::clone(&core_provider),
            Arc::clone(&tokens),
            Arc::clone(&shared_config),
            &token_value,
            config.active_domain(),
            true,
        )
        .await?;
    let plaintext_request_id = next_request_id + 1;
    let reloaded = send_control_request(
        &mut plain_writer,
        &mut plain_reader,
        &mut plain_transport,
        plaintext_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut reload = payload.init_tenant_reload();
            reload.set_token(&token_value);
            reload.set_tenant_id(config.active_domain());
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::TenantReload(Ok(result)) => Ok(result.get_reloaded()),
            control_response::payload::Error(Ok(error)) => {
                let code = error
                    .get_code()
                    .context("missing error code")?
                    .to_str()
                    .context("error code utf-8")?;
                let message = error
                    .get_message()
                    .context("missing error message")?
                    .to_str()
                    .context("error message utf-8")?;
                Err(anyhow!("tenant_reload error {}: {}", code, message))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert!(reloaded);
    drop(plain_writer);
    drop(plain_reader);
    let plaintext_outcome = plain_task
        .await
        .context("plaintext control handler task panicked")?;
    plaintext_outcome.context("plaintext control handler returned error")?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn control_capnp_patch_requires_existing() -> Result<()> {
    let tempdir = tempdir().context("failed to create temp dir")?;
    let mut config = Config::default();
    config.data_dir = tempdir.path().to_path_buf();
    config.data_encryption_key = None;

    let tokens = Arc::new(TokenManager::load(
        config.jwt_manager_config()?,
        config.tokens_path(),
        config.jwt_revocations_path(),
        None,
    )?);
    let schemas = Arc::new(SchemaManager::load(config.schema_store_path())?);
    schemas.create(CreateSchemaInput {
        aggregate: "order".into(),
        events: vec!["order_created".into(), "order_status_updated".into()],
        snapshot_threshold: None,
    })?;
    let store = Arc::new(EventStore::open(
        config.event_store_path(),
        None,
        config.snowflake_worker_id,
    )?);
    let assignments = Arc::new(TenantAssignmentStore::open(config.tenant_meta_path())?);
    let usage =
        assignments.ensure_storage_usage_bytes("default", || store.storage_usage_bytes())?;

    let core = CoreContext::new(
        Arc::clone(&tokens),
        Arc::clone(&schemas),
        Arc::clone(&store),
        config.restrict,
        config.list_page_size,
        config.page_limit,
        "default",
        None,
        Some(usage),
        Arc::clone(&assignments),
        config.reference_default_depth,
        config.reference_max_depth,
    );
    let shared_config = Arc::new(RwLock::new(config.clone()));
    let core_provider: Arc<dyn CoreProvider> = Arc::new(StaticCoreProvider::new(core.clone()));

    let token_record = tokens.issue(IssueTokenInput {
        subject: "system:admin".into(),
        group: "system".into(),
        user: "admin".into(),
        actions: vec![
            "aggregate.create".into(),
            "aggregate.append".into(),
            "aggregate.archive".into(),
            "aggregate.read".into(),
        ],
        resources: vec!["*".to_string()],
        tenants: Vec::new(),
        ttl_secs: Some(3_600),
        not_before: None,
        issued_by: "control-batch-test".into(),
        limits: JwtLimits::default(),
    })?;
    let token_value = token_record
        .token
        .clone()
        .expect("issued token missing value");

    let (mut writer, mut reader, mut transport, server_task) = open_control_session(
        Arc::clone(&core_provider),
        Arc::clone(&tokens),
        Arc::clone(&shared_config),
        &token_value,
        config.active_domain(),
        false,
    )
    .await?;

    let mut next_request_id: u64 = 1;

    let missing_patch_error = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut patch = payload.init_patch_event();
            patch.set_token(&token_value);
            patch.set_aggregate_type("order");
            patch.set_aggregate_id("order-unknown");
            patch.set_event_type("order_status_updated");
            patch.set_patch_json(
                r#"[
                    {"op":"replace","path":"/firstName","value":"Jane"},
                    {"op":"add","path":"/age","value":30},
                    {"op":"remove","path":"/address/zipCode"}
                ]"#,
            );
            patch.set_note("");
            patch.set_has_note(false);
            patch.set_metadata_json("");
            patch.set_has_metadata(false);
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::Error(Ok(error)) => {
                let code = error
                    .get_code()
                    .context("missing error code")?
                    .to_str()
                    .context("code utf-8 error")?
                    .to_string();
                let message = error
                    .get_message()
                    .context("missing error message")?
                    .to_str()
                    .context("message utf-8 error")?
                    .to_string();
                Ok((code, message))
            }
            control_response::payload::Error(Err(err)) => {
                Err(anyhow!("failed to decode control error payload: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert_eq!(missing_patch_error.0, "aggregate_not_found");
    next_request_id += 1;

    let append_record: Value = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut append = payload.init_append_event();
            append.set_token(&token_value);
            append.set_aggregate_type("order");
            append.set_aggregate_id("order-1");
            append.set_event_type("order_created");
            append.set_payload_json(
                r#"{"status":"created","firstName":"John","address":{"street":"123 Main","zipCode":"94107"}}"#,
            );
            append.set_metadata_json("");
            append.set_note("");
            append.set_has_note(false);
            append.set_has_metadata(false);
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::AppendEvent(Ok(append)) => {
                let json = append
                    .get_event_json()
                    .context("missing event_json")?
                    .to_str()
                    .context("event_json utf-8 error")?;
                let parsed: Value =
                    serde_json::from_str(json).context("failed to parse event_json")?;
                Ok(parsed)
            }
            control_response::payload::AppendEvent(Err(err)) => {
                Err(anyhow!("failed to decode append_event response: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert_eq!(append_record["version"], 1);
    assert_eq!(append_record["payload"]["firstName"], "John");
    let initial_address = append_record["payload"]["address"]
        .as_object()
        .expect("initial address object");
    assert_eq!(
        initial_address.get("zipCode").and_then(Value::as_str),
        Some("94107")
    );
    next_request_id += 1;

    let patch_record: Value = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut patch = payload.init_patch_event();
            patch.set_token(&token_value);
            patch.set_aggregate_type("order");
            patch.set_aggregate_id("order-1");
            patch.set_event_type("order_status_updated");
            patch.set_patch_json(
                r#"[
                    {"op":"replace","path":"/firstName","value":"Jane"},
                    {"op":"add","path":"/age","value":30},
                    {"op":"remove","path":"/address/zipCode"}
                ]"#,
            );
            patch.set_note("");
            patch.set_has_note(false);
            patch.set_metadata_json("");
            patch.set_has_metadata(false);
        },
        |response| {
            use control_response::payload;

            match response
                .get_payload()
                .which()
                .context("payload decode failed")?
            {
                payload::AppendEvent(Ok(append)) => {
                    let json = append
                        .get_event_json()
                        .context("missing event_json")?
                        .to_str()
                        .context("event_json utf-8 error")?;
                    let parsed: Value =
                        serde_json::from_str(json).context("failed to parse event_json")?;
                    Ok(parsed)
                }
                payload::AppendEvent(Err(err)) => {
                    Err(anyhow!("failed to decode append_event response: {err}"))
                }
                payload::Error(Ok(error)) => {
                    let code = error
                        .get_code()
                        .context("missing error code")?
                        .to_str()
                        .context("error code utf-8 error")?
                        .to_string();
                    let message = error
                        .get_message()
                        .context("missing error message")?
                        .to_str()
                        .context("error message utf-8 error")?
                        .to_string();
                    Err(anyhow!("patch_event returned {}: {}", code, message))
                }
                payload::Error(Err(err)) => Err(anyhow!("failed to decode error payload: {err}")),
                payload::GetAggregate(_) => {
                    Err(anyhow!("unexpected get_aggregate response to patch_event"))
                }
                payload::ListAggregates(_) => Err(anyhow!(
                    "unexpected list_aggregates response to patch_event"
                )),
                payload::ListEvents(_) => {
                    Err(anyhow!("unexpected list_events response to patch_event"))
                }
                payload::VerifyAggregate(_) => Err(anyhow!(
                    "unexpected verify_aggregate response to patch_event"
                )),
                payload::CreateAggregate(_) => Err(anyhow!(
                    "unexpected create_aggregate response to patch_event"
                )),
                payload::SelectAggregate(_) => Err(anyhow!(
                    "unexpected select_aggregate response to patch_event"
                )),
                payload::SetAggregateArchive(_) => Err(anyhow!(
                    "unexpected set_aggregate_archive response to patch_event"
                )),
                _ => Err(anyhow!(
                    "unexpected control response variant returned to patch_event"
                )),
            }
        },
    )
    .await?;
    assert_eq!(patch_record["version"], 2);
    assert_eq!(patch_record["event_type"], "order_status_updated");
    assert_eq!(patch_record["payload"]["firstName"], "Jane");
    assert_eq!(patch_record["payload"]["age"], 30);
    let patch_address = patch_record["payload"]["address"]
        .as_object()
        .expect("address payload object");
    assert!(
        !patch_address.contains_key("zipCode"),
        "zipCode should be removed from address payload"
    );
    next_request_id += 1;

    let events: Vec<Value> = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut list = payload.init_list_events();
            list.set_token(&token_value);
            list.set_aggregate_type("order");
            list.set_aggregate_id("order-1");
            list.set_has_cursor(false);
            list.set_cursor("");
            list.set_take(25);
            list.set_has_take(true);
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::ListEvents(Ok(list)) => {
                let json = list
                    .get_events_json()
                    .context("missing events_json")?
                    .to_str()
                    .context("events_json utf-8 error")?;
                let parsed: Vec<Value> =
                    serde_json::from_str(json).context("failed to parse events_json")?;
                Ok(parsed)
            }
            control_response::payload::ListEvents(Err(err)) => {
                Err(anyhow!("failed to decode list_events response: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert_eq!(events.len(), 2);
    assert_eq!(events[1]["event_type"], "order_status_updated");
    assert_eq!(events[1]["payload"]["firstName"], "Jane");
    assert_eq!(events[1]["payload"]["age"], 30);
    let patched_address = events[1]["payload"]["address"]
        .as_object()
        .expect("patched address payload object");
    assert!(
        !patched_address.contains_key("zipCode"),
        "zipCode should be removed in event payload"
    );

    next_request_id += 1;

    let aggregate_after_patch = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut get = payload.init_get_aggregate();
            get.set_token(&token_value);
            get.set_aggregate_type("order");
            get.set_aggregate_id("order-1");
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::GetAggregate(Ok(get)) => {
                let found = get.get_found();
                let json = get
                    .get_aggregate_json()
                    .context("missing aggregate_json")?
                    .to_str()
                    .context("aggregate_json utf-8 error")?;
                let parsed: Option<Value> = if json.trim().is_empty() {
                    None
                } else {
                    Some(serde_json::from_str(json).context("failed to parse aggregate_json")?)
                };
                Ok((found, parsed))
            }
            control_response::payload::GetAggregate(Err(err)) => {
                Err(anyhow!("failed to decode get_aggregate response: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert!(aggregate_after_patch.0);
    let aggregate_body = aggregate_after_patch.1.expect("aggregate body expected");
    let state = aggregate_body["state"].as_object().expect("state object");
    assert_eq!(state.get("firstName").and_then(Value::as_str), Some("Jane"));
    assert_eq!(
        state.get("age").and_then(Value::as_str),
        Some("30"),
        "age should be stored as stringified value"
    );
    let address_state = state
        .get("address")
        .and_then(Value::as_str)
        .expect("address state entry");
    let parsed_address: Value =
        serde_json::from_str(address_state).context("failed to parse address state JSON")?;
    assert!(
        !parsed_address.get("zipCode").is_some(),
        "address state should not include zipCode"
    );

    next_request_id += 1;

    let selection = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut select = payload.init_select_aggregate();
            select.set_token(&token_value);
            select.set_aggregate_type("order");
            select.set_aggregate_id("order-1");
            let mut fields = select.reborrow().init_fields(3);
            fields.set(0, "firstName");
            fields.set(1, "address.street");
            fields.set(2, "address.zipCode");
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::SelectAggregate(Ok(select)) => {
                let found = select.get_found();
                let json = select
                    .get_selection_json()
                    .context("missing selection_json")?
                    .to_str()
                    .context("selection_json utf-8 error")?;
                let parsed: Option<Value> = if json.trim().is_empty() {
                    None
                } else {
                    Some(serde_json::from_str(json).context("failed to parse selection_json")?)
                };
                Ok((found, parsed))
            }
            control_response::payload::SelectAggregate(Err(err)) => {
                Err(anyhow!("failed to decode select_aggregate response: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert!(selection.0, "selection should report aggregate found");
    let selection_body = selection
        .1
        .expect("selection body expected when aggregate found");
    assert_eq!(selection_body["firstName"], json!("Jane"));
    assert_eq!(selection_body["address.street"], json!("123 Main"));
    assert!(selection_body["address.zipCode"].is_null());

    next_request_id += 1;

    let archived_state: Value = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut archive = payload.init_set_aggregate_archive();
            archive.set_token(&token_value);
            archive.set_aggregate_type("order");
            archive.set_aggregate_id("order-1");
            archive.set_archived(true);
            archive.set_note("closed via control");
            archive.set_has_note(true);
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::SetAggregateArchive(Ok(archive)) => {
                let json = archive
                    .get_aggregate_json()
                    .context("missing aggregate_json")?
                    .to_str()
                    .context("aggregate_json utf-8 error")?;
                let parsed: Value =
                    serde_json::from_str(json).context("failed to parse aggregate_json")?;
                Ok(parsed)
            }
            control_response::payload::SetAggregateArchive(Err(err)) => Err(anyhow!(
                "failed to decode set_aggregate_archive response: {err}"
            )),
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert_eq!(archived_state["aggregate_type"], json!("order"));
    assert_eq!(archived_state["archived"], json!(true));

    next_request_id += 1;

    let active_after_archive: Vec<Value> = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut list = payload.init_list_aggregates();
            list.set_token(&token_value);
            list.set_has_cursor(false);
            list.set_cursor("");
            list.set_take(10);
            list.set_has_take(true);
            list.set_has_filter(false);
            list.set_has_sort(false);
            list.set_sort("");
            list.set_include_archived(false);
            list.set_archived_only(false);
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::ListAggregates(Ok(list)) => {
                let json = list
                    .get_aggregates_json()
                    .context("missing aggregates_json")?
                    .to_str()
                    .context("aggregates_json utf-8 error")?;
                let parsed: Vec<Value> =
                    serde_json::from_str(json).context("failed to parse aggregates_json")?;
                Ok(parsed)
            }
            control_response::payload::ListAggregates(Err(err)) => {
                Err(anyhow!("failed to decode list_aggregates response: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert!(
        active_after_archive.is_empty(),
        "archived aggregates should be hidden when include_archived=false"
    );

    next_request_id += 1;

    let archived_only_listing: Vec<Value> = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut list = payload.init_list_aggregates();
            list.set_token(&token_value);
            list.set_has_cursor(false);
            list.set_cursor("");
            list.set_take(10);
            list.set_has_take(true);
            list.set_has_filter(false);
            list.set_has_sort(false);
            list.set_sort("");
            list.set_include_archived(false);
            list.set_archived_only(true);
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::ListAggregates(Ok(list)) => {
                let json = list
                    .get_aggregates_json()
                    .context("missing aggregates_json")?
                    .to_str()
                    .context("aggregates_json utf-8 error")?;
                let parsed: Vec<Value> =
                    serde_json::from_str(json).context("failed to parse aggregates_json")?;
                Ok(parsed)
            }
            control_response::payload::ListAggregates(Err(err)) => {
                Err(anyhow!("failed to decode list_aggregates response: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert_eq!(archived_only_listing.len(), 1);
    assert_eq!(archived_only_listing[0]["archived"], json!(true));

    next_request_id += 1;

    let restored_state: Value = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut archive = payload.init_set_aggregate_archive();
            archive.set_token(&token_value);
            archive.set_aggregate_type("order");
            archive.set_aggregate_id("order-1");
            archive.set_archived(false);
            archive.set_note("");
            archive.set_has_note(false);
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::SetAggregateArchive(Ok(archive)) => {
                let json = archive
                    .get_aggregate_json()
                    .context("missing aggregate_json")?
                    .to_str()
                    .context("aggregate_json utf-8 error")?;
                let parsed: Value =
                    serde_json::from_str(json).context("failed to parse aggregate_json")?;
                Ok(parsed)
            }
            control_response::payload::SetAggregateArchive(Err(err)) => Err(anyhow!(
                "failed to decode set_aggregate_archive response: {err}"
            )),
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    let restored_archived = restored_state
        .get("archived")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    assert!(
        !restored_archived,
        "archived flag should be absent or false after restoration"
    );

    next_request_id += 1;

    let active_after_restore: Vec<Value> = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut list = payload.init_list_aggregates();
            list.set_token(&token_value);
            list.set_has_cursor(false);
            list.set_cursor("");
            list.set_take(10);
            list.set_has_take(true);
            list.set_has_filter(false);
            list.set_has_sort(false);
            list.set_sort("");
            list.set_include_archived(false);
            list.set_archived_only(false);
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::ListAggregates(Ok(list)) => {
                let json = list
                    .get_aggregates_json()
                    .context("missing aggregates_json")?
                    .to_str()
                    .context("aggregates_json utf-8 error")?;
                let parsed: Vec<Value> =
                    serde_json::from_str(json).context("failed to parse aggregates_json")?;
                Ok(parsed)
            }
            control_response::payload::ListAggregates(Err(err)) => {
                Err(anyhow!("failed to decode list_aggregates response: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert_eq!(active_after_restore.len(), 1);
    let restored_entry_archived = active_after_restore[0]
        .get("archived")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    assert!(
        !restored_entry_archived,
        "active aggregates should not report archived flag"
    );

    drop(writer);
    drop(reader);

    let outcome = server_task.await.context("control handler task panicked")?;
    outcome.context("control handler returned error")?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn control_tenant_admin_commands() -> Result<()> {
    let tempdir = tempdir().context("failed to create temp dir")?;
    let mut config = Config::default();
    config.data_dir = tempdir.path().to_path_buf();
    config.data_encryption_key = None;

    let tokens = Arc::new(TokenManager::load(
        config.jwt_manager_config()?,
        config.tokens_path(),
        config.jwt_revocations_path(),
        None,
    )?);
    let schemas = Arc::new(SchemaManager::load(config.schema_store_path())?);
    schemas.create(CreateSchemaInput {
        aggregate: "order".into(),
        events: vec!["order_created".into()],
        snapshot_threshold: None,
    })?;
    let store = Arc::new(EventStore::open(
        config.event_store_path(),
        None,
        config.snowflake_worker_id,
    )?);
    let assignments = Arc::new(TenantAssignmentStore::open(config.tenant_meta_path())?);
    let usage =
        assignments.ensure_storage_usage_bytes("default", || store.storage_usage_bytes())?;

    let core = CoreContext::new(
        Arc::clone(&tokens),
        Arc::clone(&schemas),
        Arc::clone(&store),
        config.restrict,
        config.list_page_size,
        config.page_limit,
        "default",
        None,
        Some(usage),
        Arc::clone(&assignments),
        config.reference_default_depth,
        config.reference_max_depth,
    );
    let shared_config = Arc::new(RwLock::new(config.clone()));
    let core_provider: Arc<dyn CoreProvider> = Arc::new(StaticCoreProvider::new(core.clone()));

    let token_record = tokens.issue(IssueTokenInput {
        subject: "system:admin".into(),
        group: "system".into(),
        user: "admin".into(),
        actions: vec!["tenant.manage".into()],
        resources: vec!["*".to_string()],
        tenants: Vec::new(),
        ttl_secs: Some(3_600),
        not_before: None,
        issued_by: "control-tenant-test".into(),
        limits: JwtLimits::default(),
    })?;
    let token_value = token_record
        .token
        .clone()
        .expect("issued token missing value");

    let (mut writer, mut reader, mut transport, server_task) = open_control_session(
        Arc::clone(&core_provider),
        Arc::clone(&tokens),
        Arc::clone(&shared_config),
        &token_value,
        config.active_domain(),
        false,
    )
    .await?;

    let mut next_request_id: u64 = 1;

    let assign_changed = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut assign = payload.init_tenant_assign();
            assign.set_token(&token_value);
            assign.set_tenant_id("default");
            assign.set_shard_id("shard-0007");
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::TenantAssign(Ok(result)) => Ok(result.get_changed()),
            control_response::payload::TenantAssign(Err(err)) => {
                Err(anyhow!("failed to decode tenant_assign response: {err}"))
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert!(assign_changed);
    next_request_id += 1;

    let duplicate_assign = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut assign = payload.init_tenant_assign();
            assign.set_token(&token_value);
            assign.set_tenant_id("default");
            assign.set_shard_id("shard-0007");
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::TenantAssign(Ok(result)) => Ok(result.get_changed()),
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert!(!duplicate_assign);
    next_request_id += 1;

    let unassign_changed = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut unassign = payload.init_tenant_unassign();
            unassign.set_token(&token_value);
            unassign.set_tenant_id("default");
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::TenantUnassign(Ok(result)) => Ok(result.get_changed()),
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert!(unassign_changed);
    next_request_id += 1;

    let quota_set = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut quota = payload.init_tenant_quota_set();
            quota.set_token(&token_value);
            quota.set_tenant_id("default");
            quota.set_max_storage_mb(5);
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::TenantQuotaSet(Ok(result)) => Ok(result.get_changed()),
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert!(quota_set);
    next_request_id += 1;

    let quota_clear = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut quota = payload.init_tenant_quota_clear();
            quota.set_token(&token_value);
            quota.set_tenant_id("default");
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::TenantQuotaClear(Ok(result)) => Ok(result.get_changed()),
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    assert!(quota_clear);
    next_request_id += 1;

    let storage_bytes = send_control_request(
        &mut writer,
        &mut reader,
        &mut transport,
        next_request_id,
        |request| {
            let payload = request.reborrow().init_payload();
            let mut quota = payload.init_tenant_quota_recalc();
            quota.set_token(&token_value);
            quota.set_tenant_id("default");
        },
        |response| match response
            .get_payload()
            .which()
            .context("payload decode failed")?
        {
            control_response::payload::TenantQuotaRecalc(Ok(result)) => {
                Ok(result.get_storage_bytes())
            }
            _ => Err(anyhow!("unexpected response variant")),
        },
    )
    .await?;
    let local_usage = store.storage_usage_bytes()?;
    assert_eq!(storage_bytes, local_usage);

    drop(writer);
    drop(reader);

    let outcome = server_task.await.context("control handler task panicked")?;
    outcome.context("control handler returned error")?;

    Ok(())
}

async fn send_control_request<W, R, Build, Handle, T>(
    writer: &mut W,
    reader: &mut R,
    transport: &mut FrameTransport,
    request_id: u64,
    build: Build,
    handle: Handle,
) -> Result<T>
where
    W: futures::AsyncWrite + Unpin,
    R: futures::AsyncRead + Unpin,
    Build: FnOnce(&mut control_request::Builder<'_>),
    Handle: FnOnce(control_response::Reader<'_>) -> Result<T>,
{
    let mut message = Builder::new_default();
    {
        let mut request = message.init_root::<control_request::Builder>();
        request.set_id(request_id);
        build(&mut request);
    }
    let bytes = write_message_to_words(&message);
    write_session_frame(writer, transport, &bytes)
        .await
        .context("failed to send control request")?;

    let response_bytes = read_session_frame(reader, transport)
        .await?
        .ok_or_else(|| anyhow!("control channel closed before response"))?;
    let mut cursor = std::io::Cursor::new(&response_bytes);
    let response_message = capnp::serialize::read_message(&mut cursor, ReaderOptions::new())
        .context("failed to decode control response frame")?;
    let response = response_message
        .get_root::<control_response::Reader>()
        .context("failed to decode control response")?;

    if response.get_id() != request_id {
        return Err(anyhow!(
            "response id {} did not match request {}",
            response.get_id(),
            request_id
        ));
    }

    handle(response)
}
