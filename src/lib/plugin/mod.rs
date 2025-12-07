use std::sync::Arc;

use chrono::Utc;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use tracing::{error, warn};

use crate::{
    config::{
        Config, PluginConfig, PluginDefinition, PluginKind, PluginPayloadMode,
        PluginQueuePruneConfig,
    },
    error::{EventError, Result},
    schema::AggregateSchema,
    store::{AggregateState, EventRecord},
};

mod process;
pub mod queue;
pub mod registry;
use queue::{JobPayload, PluginQueueStore};
mod tcp;
use tcp::TcpPlugin;
mod http;
use http::HttpPlugin;
mod log;
use log::LogPlugin;
use process::ProcessPlugin;
pub use process::status_file_path;

pub struct PluginDelivery<'a> {
    pub record: Option<&'a EventRecord>,
    pub state: Option<&'a AggregateState>,
    pub schema: Option<&'a AggregateSchema>,
}

pub trait Plugin: Send + Sync {
    fn name(&self) -> &'static str;
    fn notify_event(&self, delivery: PluginDelivery<'_>) -> Result<()>;
}

struct PluginEntry {
    identifier: String,
    label: String,
    mode: PluginPayloadMode,
    plugin: Box<dyn Plugin>,
    emit_events: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobPriority {
    Low,
    Normal,
    High,
}

impl Default for JobPriority {
    fn default() -> Self {
        JobPriority::Normal
    }
}

#[derive(Debug, Clone)]
pub struct PublishTarget {
    pub plugin: String,
    pub mode: Option<PluginPayloadMode>,
    pub priority: Option<JobPriority>,
}

struct QueuePruner {
    policy: PluginQueuePruneConfig,
    last_prune_ms: Mutex<Option<i64>>,
}

impl QueuePruner {
    fn from_config(policy: PluginQueuePruneConfig) -> Option<Self> {
        if policy.done_ttl_secs == 0 && policy.max_done_jobs.unwrap_or(0) == 0 {
            return None;
        }
        Some(Self {
            policy,
            last_prune_ms: Mutex::new(None),
        })
    }

    fn interval_millis(&self) -> i64 {
        let secs = self.policy.interval_secs.max(60);
        let millis = secs.saturating_mul(1_000);
        millis.min(i64::MAX as u64) as i64
    }

    fn cutoff_millis(&self, now: i64) -> Option<i64> {
        if self.policy.done_ttl_secs == 0 {
            return None;
        }
        let millis = self.policy.done_ttl_secs.saturating_mul(1_000);
        let ttl = millis.min(i64::MAX as u64) as i64;
        Some(now.saturating_sub(ttl))
    }

    fn max_done_jobs(&self) -> Option<usize> {
        self.policy.max_done_jobs
    }

    fn should_run(&self, now: i64) -> bool {
        let mut guard = self.last_prune_ms.lock();
        if let Some(previous) = *guard {
            if now - previous < self.interval_millis() {
                return false;
            }
        }
        *guard = Some(now);
        true
    }
}

#[derive(Clone)]
pub struct PluginManager {
    plugins: Arc<Vec<PluginEntry>>,
    queue: Option<PluginQueueStore>,
    max_attempts: u8,
    queue_pruner: Option<Arc<QueuePruner>>,
}

impl PluginManager {
    pub fn from_config(config: &Config) -> Result<Self> {
        let mut entries = Vec::new();
        let mut definitions = config.load_plugins()?;
        if definitions.is_empty() && !config.plugins.is_empty() {
            definitions = config.plugins.clone();
        }

        for definition in definitions.into_iter() {
            if !definition.enabled {
                continue;
            }
            let label = plugin_label(&definition);
            let identifier = definition
                .name
                .clone()
                .unwrap_or_else(|| plugin_kind_name(definition.config.kind()).to_string());
            let plugin = instantiate_plugin(&definition, config);
            entries.push(PluginEntry {
                identifier,
                label,
                mode: definition.payload_mode,
                plugin,
                emit_events: definition.emit_events,
            });
        }

        // Config migrations run during load_or_default; open_with_legacy still tolerates older paths.
        let queue = match PluginQueueStore::open_with_legacy(
            config.plugin_queue_db_path().as_path(),
            config.plugin_queue_path().as_path(),
        ) {
            Ok(store) => Some(store),
            Err(err) => {
                warn!(
                    target: "eventdbx.plugin",
                    "failed to initialize plugin queue store: {}",
                    err
                );
                None
            }
        };

        let queue_pruner = queue
            .as_ref()
            .and_then(|_| QueuePruner::from_config(config.plugin_queue.prune.clone()))
            .map(Arc::new);

        Ok(Self {
            plugins: Arc::new(entries),
            queue,
            max_attempts: config.plugin_max_attempts.min(u8::MAX as u32) as u8,
            queue_pruner,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty() || self.plugins.iter().all(|entry| !entry.emit_events)
    }

    pub fn notify_event(
        &self,
        record: &EventRecord,
        state: &AggregateState,
        schema: Option<&AggregateSchema>,
    ) -> Result<()> {
        self.notify_event_for_targets(record, state, schema, None)
    }

    pub fn notify_event_for_targets(
        &self,
        record: &EventRecord,
        state: &AggregateState,
        schema: Option<&AggregateSchema>,
        targets: Option<&[PublishTarget]>,
    ) -> Result<()> {
        if self.plugins.is_empty() || self.plugins.iter().all(|entry| !entry.emit_events) {
            return Ok(());
        }

        let selected: Vec<(&PluginEntry, PluginPayloadMode, JobPriority)> =
            if let Some(targets) = targets {
                if targets.is_empty() {
                    return Ok(());
                }
                let mut seen = HashSet::new();
                let mut matched = Vec::new();
                for target in targets {
                    let key = target.plugin.trim();
                    if key.is_empty() {
                        return Err(EventError::Config(
                            "publish target name cannot be empty".into(),
                        ));
                    }
                    if !seen.insert(key.to_ascii_lowercase()) {
                        continue;
                    }
                    let entry = self
                        .plugins
                        .iter()
                        .find(|entry| entry.identifier.eq_ignore_ascii_case(key))
                        .ok_or_else(|| {
                            EventError::Config(format!("plugin '{}' is not configured", key))
                        })?;
                    if !entry.emit_events {
                        return Err(EventError::Config(format!("plugin '{}' is disabled", key)));
                    }
                    let mode = target.mode.unwrap_or(entry.mode);
                    let priority = target.priority.unwrap_or_default();
                    matched.push((entry, mode, priority));
                }
                matched
            } else {
                self.plugins
                    .iter()
                    .filter(|entry| entry.emit_events)
                    .map(|entry| (entry, entry.mode, JobPriority::Normal))
                    .collect()
            };

        for (entry, mode, priority) in selected {
            if let Some(queue) = &self.queue {
                let job_payload = build_job_payload(mode, record, state, schema);
                let payload_value = serde_json::to_value(&job_payload)?;
                let now = Utc::now().timestamp_millis();
                match queue.enqueue_job(&entry.label, payload_value, now, priority) {
                    Ok(_) => {
                        self.maybe_prune_done(queue, now);
                        if let Err(err) = self.process_jobs_for_plugin(queue, entry) {
                            warn!(
                                target: "eventdbx.plugin",
                                "failed to process plugin queue for {}: {}",
                                entry.label,
                                err
                            );
                        }
                    }
                    Err(err) => {
                        warn!(
                            target: "eventdbx.plugin",
                            "failed to enqueue plugin job for {}: {}",
                            entry.label,
                            err
                        );
                        let owned_record = sanitized_record_for_mode(mode, record);
                        let record_ref = owned_record
                            .as_ref()
                            .or_else(|| mode.includes_event().then_some(record));
                        let state_ref = if mode.includes_state() {
                            Some(state)
                        } else {
                            None
                        };
                        let schema_ref = if mode.includes_schema() { schema } else { None };

                        let delivery = PluginDelivery {
                            record: record_ref,
                            state: state_ref,
                            schema: schema_ref,
                        };
                        if let Err(err) = entry.plugin.notify_event(delivery) {
                            error!("plugin {} failed: {}", entry.label, err);
                        }
                    }
                }
            } else {
                let owned_record = sanitized_record_for_mode(mode, record);
                let record_ref = owned_record
                    .as_ref()
                    .or_else(|| mode.includes_event().then_some(record));
                let state_ref = if mode.includes_state() {
                    Some(state)
                } else {
                    None
                };
                let schema_ref = if mode.includes_schema() { schema } else { None };

                let delivery = PluginDelivery {
                    record: record_ref,
                    state: state_ref,
                    schema: schema_ref,
                };
                if let Err(err) = entry.plugin.notify_event(delivery) {
                    error!("plugin {} failed: {}", entry.label, err);
                }
            }
        }

        Ok(())
    }

    fn process_jobs_for_plugin(&self, queue: &PluginQueueStore, entry: &PluginEntry) -> Result<()> {
        if !entry.emit_events {
            return Ok(());
        }
        let now = Utc::now().timestamp_millis();
        self.maybe_prune_done(queue, now);
        let _ = queue.recover_stuck_jobs(&entry.label, now, self.max_attempts);

        loop {
            let dispatch_now = Utc::now().timestamp_millis();
            let jobs = queue.dispatch_jobs(&entry.label, 8, dispatch_now)?;
            if jobs.is_empty() {
                break;
            }

            for job in jobs {
                let payload: JobPayload =
                    serde_json::from_value(job.payload.clone()).unwrap_or_default();

                if payload.legacy.is_some() && payload.is_empty() {
                    // legacy queue entry without enough data—mark completed.
                    queue.complete_job(job.id)?;
                    continue;
                }

                if payload.is_empty() {
                    queue.complete_job(job.id)?;
                    continue;
                }

                let delivery = PluginDelivery {
                    record: payload.record.as_ref(),
                    state: payload.state.as_ref(),
                    schema: payload.schema.as_ref(),
                };

                match entry.plugin.notify_event(delivery) {
                    Ok(_) => {
                        queue.complete_job(job.id)?;
                    }
                    Err(err) => {
                        let error = err.to_string();
                        let now = Utc::now().timestamp_millis();
                        queue.fail_job(job.id, error, now, self.max_attempts)?;
                    }
                }
            }
        }

        Ok(())
    }

    pub fn dispatch_pending(&self) -> Result<()> {
        if let Some(queue) = &self.queue {
            for entry in self.plugins.iter() {
                if !entry.emit_events {
                    continue;
                }
                self.process_jobs_for_plugin(queue, entry)?;
            }
        }
        Ok(())
    }

    pub fn has_label(&self, label: &str) -> bool {
        self.plugins.iter().any(|entry| entry.label == label)
    }

    pub fn dispatch_for_label(&self, label: &str) -> Result<()> {
        if let Some(queue) = &self.queue {
            if let Some(entry) = self.plugins.iter().find(|entry| entry.label == label) {
                if !entry.emit_events {
                    return Ok(());
                }
                self.process_jobs_for_plugin(queue, entry)?;
            }
        }
        Ok(())
    }

    fn maybe_prune_done(&self, queue: &PluginQueueStore, now: i64) {
        let Some(pruner) = self.queue_pruner.as_ref() else {
            return;
        };

        if !pruner.should_run(now) {
            return;
        }

        if let Some(cutoff) = pruner.cutoff_millis(now) {
            if let Err(err) = queue.clear_done(Some(cutoff)) {
                warn!(
                    target: "eventdbx.plugin",
                    "failed to prune done plugin jobs older than {}: {}",
                    cutoff,
                    err
                );
            }
        }

        if let Some(limit) = pruner.max_done_jobs() {
            if let Err(err) = queue.truncate_done(limit) {
                warn!(
                    target: "eventdbx.plugin",
                    "failed to enforce done plugin job limit {}: {}",
                    limit,
                    err
                );
            }
        }
    }
}

fn sanitized_record_for_mode(mode: PluginPayloadMode, record: &EventRecord) -> Option<EventRecord> {
    match mode {
        PluginPayloadMode::ExtensionsOnly => {
            let mut sanitized = record.clone();
            sanitized.payload = Value::Null;
            Some(sanitized)
        }
        _ => None,
    }
}

fn build_job_payload(
    mode: PluginPayloadMode,
    record: &EventRecord,
    state: &AggregateState,
    schema: Option<&AggregateSchema>,
) -> JobPayload {
    let record_payload = if let Some(sanitized) = sanitized_record_for_mode(mode, record) {
        Some(sanitized)
    } else if mode.includes_event() {
        Some(record.clone())
    } else {
        None
    };

    let state_payload = if mode.includes_state() {
        Some(state.clone())
    } else {
        None
    };

    let schema_payload = if mode.includes_schema() {
        schema.cloned()
    } else {
        None
    };

    JobPayload {
        record: record_payload,
        state: state_payload,
        schema: schema_payload,
        legacy: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::PluginQueuePruneConfig,
        schema::AggregateSchema,
        snowflake::SnowflakeId,
        store::{AggregateState, EventMetadata, EventRecord},
    };
    use chrono::Utc;
    use serde_json::json;
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    struct FailingPlugin;

    impl Plugin for FailingPlugin {
        fn name(&self) -> &'static str {
            "failing"
        }

        fn notify_event(&self, _delivery: PluginDelivery<'_>) -> Result<()> {
            Err(EventError::Storage("simulated failure".into()))
        }
    }

    struct SuccessfulPlugin;

    impl Plugin for SuccessfulPlugin {
        fn name(&self) -> &'static str {
            "successful"
        }

        fn notify_event(&self, _delivery: PluginDelivery<'_>) -> Result<()> {
            Ok(())
        }
    }

    fn sample_record(event_id: u64) -> (EventRecord, AggregateState) {
        let record = EventRecord {
            aggregate_type: "order".into(),
            aggregate_id: "order-1".into(),
            event_type: "created".into(),
            event_type_raw: Some("created".into()),
            payload: json!({"status": "created"}),
            extensions: None,
            metadata: EventMetadata {
                event_id: SnowflakeId::from_u64(event_id),
                created_at: Utc::now(),
                issued_by: None,
                note: None,
            },
            version: 1,
            hash: "hash".into(),
            merkle_root: "merkle".into(),
        };
        let state = AggregateState {
            aggregate_type: record.aggregate_type.clone(),
            aggregate_id: record.aggregate_id.clone(),
            version: record.version,
            state: BTreeMap::from([("status".into(), "created".into())]),
            extensions: BTreeMap::new(),
            merkle_root: record.merkle_root.clone(),
            created_at: None,
            updated_at: None,
            archived: false,
        };
        (record, state)
    }

    #[test]
    fn extensions_only_mode_sanitizes_job_payload() {
        let (mut record, state) = sample_record(99);
        record.payload = json!({"secret": "value"});
        record.extensions = Some(json!({"trace_id": "abc123"}));

        let job_payload =
            build_job_payload(PluginPayloadMode::ExtensionsOnly, &record, &state, None);

        let job_record = job_payload.record.expect("record is present");
        assert!(job_payload.state.is_none());
        assert!(job_payload.schema.is_none());
        assert!(job_record.payload.is_null());
        assert_eq!(job_record.extensions, record.extensions);
        assert_eq!(job_record.aggregate_id, record.aggregate_id);
    }

    #[test]
    fn extensions_only_mode_sanitizes_direct_delivery() {
        let (mut record, state) = sample_record(101);
        record.payload = json!({"secret": "value"});
        record.extensions = Some(json!({"trace_id": "xyz"}));

        let mode = PluginPayloadMode::ExtensionsOnly;
        let owned_record = sanitized_record_for_mode(mode, &record);
        let record_ref = owned_record
            .as_ref()
            .or_else(|| mode.includes_event().then_some(&record))
            .expect("record reference is present");

        let state_ref = if mode.includes_state() {
            Some(&state)
        } else {
            None
        };
        let schema_input: Option<&AggregateSchema> = None;
        let schema_ref = if mode.includes_schema() {
            schema_input
        } else {
            None
        };

        let delivery = PluginDelivery {
            record: Some(record_ref),
            state: state_ref,
            schema: schema_ref,
        };

        let record_from_delivery = delivery.record.expect("record is present");
        assert!(record_from_delivery.payload.is_null());
        assert_eq!(record_from_delivery.extensions, record.extensions);
        assert!(delivery.state.is_none());
        assert!(delivery.schema.is_none());
    }

    #[test]
    fn queues_failed_plugin_events() -> Result<()> {
        let temp = TempDir::new().expect("tempdir");
        let queue_path = temp.path().join("queue.db");
        let queue_store = PluginQueueStore::open(queue_path.as_path())?;

        let manager = PluginManager {
            plugins: Arc::new(vec![PluginEntry {
                identifier: "failing".into(),
                label: "failing".into(),
                mode: PluginPayloadMode::All,
                plugin: Box::new(FailingPlugin),
                emit_events: true,
            }]),
            queue: Some(queue_store.clone()),
            max_attempts: 3,
            queue_pruner: None,
        };

        let (record, state) = sample_record(7);
        manager.notify_event(&record, &state, None)?;

        let status = queue_store.status()?;
        assert_eq!(status.dead.len(), 0);
        assert_eq!(status.pending.len(), 1);
        assert_eq!(status.pending[0].attempts, 1);

        Ok(())
    }

    #[test]
    fn clears_queue_on_success() -> Result<()> {
        let temp = TempDir::new().expect("tempdir");
        let queue_path = temp.path().join("queue.db");
        let queue_store = PluginQueueStore::open(queue_path.as_path())?;

        let failing_manager = PluginManager {
            plugins: Arc::new(vec![PluginEntry {
                identifier: "failing".into(),
                label: "failing".into(),
                mode: PluginPayloadMode::All,
                plugin: Box::new(FailingPlugin),
                emit_events: true,
            }]),
            queue: Some(queue_store.clone()),
            max_attempts: 0,
            queue_pruner: None,
        };

        let (record, state) = sample_record(15);
        failing_manager.notify_event(&record, &state, None)?;

        let status_after_fail = queue_store.status()?;
        assert_eq!(status_after_fail.dead.len(), 1);
        let job_id = status_after_fail.dead[0].id;
        queue_store.retry_dead_job(job_id, Utc::now().timestamp_millis())?;

        let success_manager = PluginManager {
            plugins: Arc::new(vec![PluginEntry {
                identifier: "failing".into(),
                label: "failing".into(),
                mode: PluginPayloadMode::All,
                plugin: Box::new(SuccessfulPlugin),
                emit_events: true,
            }]),
            queue: Some(queue_store.clone()),
            max_attempts: 3,
            queue_pruner: None,
        };
        success_manager.dispatch_pending()?;

        let status = queue_store.status()?;
        assert!(status.pending.is_empty());
        assert!(status.processing.is_empty());
        assert_eq!(status.done.len(), 1);
        Ok(())
    }

    #[test]
    fn prune_removes_old_done_jobs() -> Result<()> {
        let temp = TempDir::new().expect("tempdir");
        let queue_path = temp.path().join("queue.db");
        let queue_store = PluginQueueStore::open(queue_path.as_path())?;

        let now = Utc::now().timestamp_millis();
        let old_job = queue_store.enqueue_job(
            "rest",
            json!({"kind": "old"}),
            now - 120_000,
            JobPriority::Normal,
        )?;
        queue_store.complete_job(old_job.id)?;

        let fresh_job =
            queue_store.enqueue_job("rest", json!({"kind": "fresh"}), now, JobPriority::Normal)?;
        queue_store.complete_job(fresh_job.id)?;

        let pruner = QueuePruner::from_config(PluginQueuePruneConfig {
            done_ttl_secs: 60,
            interval_secs: 0,
            max_done_jobs: None,
        })
        .expect("pruner configured");

        let manager = PluginManager {
            plugins: Arc::new(Vec::new()),
            queue: Some(queue_store.clone()),
            max_attempts: 3,
            queue_pruner: Some(Arc::new(pruner)),
        };

        manager.maybe_prune_done(&queue_store, now);

        let status = queue_store.status()?;
        assert_eq!(status.done.len(), 1);
        assert_eq!(status.done[0].id, fresh_job.id);
        Ok(())
    }

    #[test]
    fn prune_enforces_done_limit() -> Result<()> {
        let temp = TempDir::new().expect("tempdir");
        let queue_path = temp.path().join("queue.db");
        let queue_store = PluginQueueStore::open(queue_path.as_path())?;

        let now = Utc::now().timestamp_millis();
        for idx in 0..4 {
            let created_at = now + idx as i64;
            let job = queue_store.enqueue_job(
                "rest",
                json!({"seq": idx}),
                created_at,
                JobPriority::Normal,
            )?;
            queue_store.complete_job(job.id)?;
        }

        let pruner = QueuePruner::from_config(PluginQueuePruneConfig {
            done_ttl_secs: 0,
            interval_secs: 0,
            max_done_jobs: Some(2),
        })
        .expect("pruner configured");

        let manager = PluginManager {
            plugins: Arc::new(Vec::new()),
            queue: Some(queue_store.clone()),
            max_attempts: 3,
            queue_pruner: Some(Arc::new(pruner)),
        };

        manager.maybe_prune_done(&queue_store, now);

        let status = queue_store.status()?;
        assert_eq!(status.done.len(), 2);
        Ok(())
    }
}

pub fn establish_connection(definition: &PluginDefinition) -> Result<()> {
    match &definition.config {
        PluginConfig::Tcp(settings) => {
            let plugin = TcpPlugin::new(settings.clone());
            plugin.ensure_ready()
        }
        PluginConfig::Http(settings) => {
            let plugin = HttpPlugin::new(settings.clone());
            plugin.ensure_ready()
        }
        PluginConfig::Log(settings) => {
            let plugin = LogPlugin::new(settings.clone());
            plugin.ensure_ready()
        }
        PluginConfig::Process(_) => Ok(()),
    }
}

pub fn instantiate_plugin(definition: &PluginDefinition, config: &Config) -> Box<dyn Plugin> {
    match &definition.config {
        PluginConfig::Tcp(settings) => Box::new(TcpPlugin::new(settings.clone())),
        PluginConfig::Http(settings) => Box::new(HttpPlugin::new(settings.clone())),
        PluginConfig::Log(settings) => Box::new(LogPlugin::new(settings.clone())),
        PluginConfig::Process(settings) => {
            let identifier = definition
                .name
                .clone()
                .unwrap_or_else(|| settings.name.clone());
            let data_root = config.data_dir.clone();
            match ProcessPlugin::new(identifier.clone(), settings.clone(), data_root.as_path()) {
                Ok(plugin) => Box::new(plugin),
                Err(err) => {
                    tracing::error!(
                        target: "eventdbx.plugin",
                        "failed to initialize process plugin {}: {}",
                        identifier,
                        err
                    );
                    Box::new(UnavailablePlugin::new(identifier, err.to_string()))
                }
            }
        }
    }
}

struct UnavailablePlugin {
    label: String,
    reason: String,
}

impl UnavailablePlugin {
    fn new(label: String, reason: String) -> Self {
        Self { label, reason }
    }
}

impl Plugin for UnavailablePlugin {
    fn name(&self) -> &'static str {
        "unavailable"
    }

    fn notify_event(&self, _delivery: PluginDelivery<'_>) -> Result<()> {
        Err(EventError::Config(format!(
            "plugin '{}' is unavailable: {}",
            self.label, self.reason
        )))
    }
}

fn plugin_label(definition: &PluginDefinition) -> String {
    match definition.name.as_deref() {
        Some(name) if !name.trim().is_empty() => {
            format!("{} ({})", plugin_kind_name(definition.config.kind()), name)
        }
        _ => plugin_kind_name(definition.config.kind()).to_string(),
    }
}

fn plugin_kind_name(kind: PluginKind) -> &'static str {
    match kind {
        PluginKind::Tcp => "tcp",
        PluginKind::Http => "http",
        PluginKind::Log => "log",
        PluginKind::Process => "process",
    }
}
