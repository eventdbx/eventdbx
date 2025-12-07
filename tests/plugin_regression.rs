use std::{collections::BTreeMap, io};

use anyhow::{Context, Result};
use axum::{Router, body::Bytes, extract::State, http::StatusCode, routing::post};
use chrono::Utc;
use eventdbx::{
    config::{Config, HttpPluginConfig, PluginConfig, PluginDefinition, PluginPayloadMode},
    plugin::{PluginDelivery, instantiate_plugin},
    snowflake::SnowflakeId,
    store::{AggregateState, EventMetadata, EventRecord},
};
use serde_json::{Value, json};
use tokio::{net::TcpListener, sync::mpsc};

#[tokio::test(flavor = "multi_thread")]
async fn http_plugin_posts_event_payload() -> Result<()> {
    let (tx, mut rx) = mpsc::channel::<Vec<u8>>(1);
    let listener = match TcpListener::bind("127.0.0.1:0").await {
        Ok(listener) => listener,
        Err(err) if err.kind() == io::ErrorKind::PermissionDenied => {
            eprintln!("http plugin regression skipped: {}", err);
            return Ok(());
        }
        Err(err) => {
            return Err(err).context("failed to bind HTTP hook listener");
        }
    };
    let addr = listener
        .local_addr()
        .context("failed to read listener address")?;
    let app = Router::new()
        .route(
            "/hook",
            post(
                |State(tx): State<mpsc::Sender<Vec<u8>>>, body: Bytes| async move {
                    if tx.send(body.to_vec()).await.is_err() {
                        return StatusCode::INTERNAL_SERVER_ERROR;
                    }
                    StatusCode::OK
                },
            ),
        )
        .with_state(tx.clone());

    let server_handle = tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app).await {
            eprintln!("hook server error: {err}");
        }
    });

    let endpoint = format!("http://{}/hook", addr);
    let plugin_definition = PluginDefinition {
        enabled: true,
        emit_events: true,
        name: Some("http-regression".to_string()),
        payload_mode: PluginPayloadMode::All,
        config: PluginConfig::Http(HttpPluginConfig {
            endpoint,
            headers: BTreeMap::from([("X-Test".to_string(), "ok".to_string())]),
            https: false,
        }),
    };

    let plugin = instantiate_plugin(&plugin_definition, &Config::default());

    let record = EventRecord {
        aggregate_type: "order".into(),
        aggregate_id: "order-123".into(),
        event_type: "OrderCreated".into(),
        event_type_raw: Some("OrderCreated".into()),
        payload: json!({ "status": "created" }),
        extensions: None,
        metadata: EventMetadata {
            event_id: SnowflakeId::from_u64(42),
            created_at: Utc::now(),
            issued_by: None,
            note: Some("regression".into()),
        },
        version: 5,
        hash: "hash-order-123-v5".into(),
        merkle_root: "merkle-order-123-v5".into(),
    };
    let state = AggregateState {
        aggregate_type: record.aggregate_type.clone(),
        aggregate_id: record.aggregate_id.clone(),
        version: 5,
        state: BTreeMap::from([("status".into(), "created".into())]),
        extensions: BTreeMap::new(),
        merkle_root: "state-root".into(),
        created_at: None,
        updated_at: None,
        archived: false,
    };

    plugin
        .notify_event(PluginDelivery {
            record: Some(&record),
            state: Some(&state),
            schema: None,
        })
        .context("plugin notify_event failed")?;

    let body = rx
        .recv()
        .await
        .context("server did not receive plugin request")?;
    server_handle.abort();

    let received: Value =
        serde_json::from_slice(&body).context("failed to decode posted payload as JSON")?;
    let event = received.get("event").context("missing event payload")?;
    assert_eq!(event["aggregate_type"], "order");
    assert_eq!(event["aggregate_id"], "order-123");
    assert_eq!(event["event_type"], "OrderCreated");
    assert_eq!(event["version"], 5);
    assert_eq!(event["metadata"]["note"], "regression");
    assert_eq!(
        event["metadata"]["event_id"],
        record.metadata.event_id.to_string()
    );
    let state = received.get("state").context("missing state payload")?;
    assert_eq!(state["aggregate_type"], "order");
    assert_eq!(state["aggregate_id"], "order-123");
    assert_eq!(state["version"], 5);
    assert_eq!(state["state"]["status"], "created");
    assert!(received.get("schema").is_none());

    Ok(())
}
