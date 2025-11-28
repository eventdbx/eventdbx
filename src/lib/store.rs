use std::{
    cmp::Ordering as StdOrdering,
    collections::{BTreeMap, BTreeSet},
    convert::TryInto,
    fmt, fs, io,
    path::{Path, PathBuf},
    str::{self, FromStr},
    thread,
    time::{Duration, Instant},
};

use chrono::{DateTime, Utc};
use parking_lot::{Mutex, MutexGuard};
use rocksdb::{
    DBIterator, DBWithThreadMode, Direction, IteratorMode, MultiThreaded, Options, WriteBatch,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use json_patch::Patch;
use metrics::{counter, histogram};
use tracing::warn;

#[cfg(unix)]
use std::os::fd::AsRawFd;

use super::{
    encryption::{self, Encryptor},
    error::{EventError, Result},
    merkle::{compute_merkle_root, empty_root},
    schema::MAX_EVENT_NOTE_LENGTH,
};
use crate::{
    filter::FilterExpr,
    snowflake::{MAX_WORKER_ID, SnowflakeGenerator, SnowflakeId},
};

const SEP: u8 = 0x1F;
const PREFIX_EVENT: &str = "evt";
const PREFIX_EVENT_ARCHIVED: &str = "evt-arch";
const PREFIX_META: &str = "meta";
const PREFIX_STATE: &str = "state";
const PREFIX_META_ARCHIVED: &str = "meta-arch";
const PREFIX_STATE_ARCHIVED: &str = "state-arch";
const PREFIX_SNAPSHOT: &str = "snapshot";
const PREFIX_SNAPSHOT_ID: &str = "snapidx";
const PREFIX_META_CREATED_INDEX: &str = "meta-created";
const PREFIX_META_UPDATED_INDEX: &str = "meta-updated";
const PREFIX_META_CREATED_INDEX_ARCHIVED: &str = "meta-created-arch";
const PREFIX_META_UPDATED_INDEX_ARCHIVED: &str = "meta-updated-arch";
const PREFIX_REF_SOURCE: &str = "refsrc";
const PREFIX_REF_TARGET: &str = "reftgt";
const LOCK_RETRY_ATTEMPTS: usize = 5;
const LOCK_RETRY_BACKOFF: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRecord {
    pub aggregate_type: String,
    pub aggregate_id: String,
    pub event_type: String,
    pub payload: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Value>,
    pub metadata: EventMetadata,
    pub version: u64,
    pub hash: String,
    pub merkle_root: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    pub event_id: SnowflakeId,
    pub created_at: DateTime<Utc>,
    pub issued_by: Option<ActorClaims>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorClaims {
    pub group: String,
    pub user: String,
}

impl<'a> Transaction<'a> {
    pub fn append(&mut self, input: AppendEvent) -> Result<EventRecord> {
        if let Some(note) = input.note.as_ref() {
            if note.chars().count() > MAX_EVENT_NOTE_LENGTH {
                return Err(EventError::InvalidSchema(format!(
                    "event note cannot exceed {} characters",
                    MAX_EVENT_NOTE_LENGTH
                )));
            }
        }
        let tenant = input.tenant.clone();
        let references = input.reference_targets.clone();
        let key = AggregateKey::new(input.aggregate_type.clone(), input.aggregate_id.clone());
        let event_id = self.store.next_event_id();
        let (meta, state) = self.ensure_cache(&key)?;
        if meta.archived {
            return Err(EventError::AggregateArchived);
        }
        let record = apply_event(meta, state, input, event_id);
        let stored_record = self.store.encode_record(&record)?;
        let event_value = serde_json::to_vec(&stored_record)?;

        self.pending_events.push(PendingEvent {
            key: event_key(&record.aggregate_type, &record.aggregate_id, record.version),
            value: event_value,
        });
        self.pending_records.push(record.clone());

        self.pending_refs.push(PendingRefUpdate {
            tenant,
            aggregate_type: record.aggregate_type.clone(),
            aggregate_id: record.aggregate_id.clone(),
            references,
        });
        Ok(record)
    }

    pub fn commit(mut self) -> Result<Vec<EventRecord>> {
        if self.pending_events.is_empty()
            && self.meta_cache.is_empty()
            && self.state_cache.is_empty()
        {
            return Ok(Vec::new());
        }

        let pending_events = std::mem::take(&mut self.pending_events);
        let meta_cache = std::mem::take(&mut self.meta_cache);
        let state_cache = std::mem::take(&mut self.state_cache);
        let records = std::mem::take(&mut self.pending_records);

        let mut batch = WriteBatch::default();

        for pending in pending_events {
            batch_put(&mut batch, pending.key, pending.value)?;
        }

        for meta in meta_cache.into_values() {
            let previous = self
                .store
                .load_meta(&meta.aggregate_type, &meta.aggregate_id)?;
            self.store.sync_timestamp_indexes(
                &mut batch,
                AggregateIndex::Active,
                previous.as_ref(),
                Some(&meta),
            );
            let key = meta_key(&meta.aggregate_type, &meta.aggregate_id);
            batch_put(&mut batch, key, serde_json::to_vec(&meta)?)?;
        }

        for (key, state) in state_cache {
            let (aggregate_type, aggregate_id) = key.parts();
            let state_bytes = self.store.encode_state_map(&state)?;
            batch_put(
                &mut batch,
                state_key(aggregate_type, aggregate_id),
                state_bytes,
            )?;
        }

        for pending in self.pending_refs {
            update_reference_index_inner(
                &self.store.db,
                &mut batch,
                &pending.tenant,
                &pending.aggregate_type,
                &pending.aggregate_id,
                &pending.references,
            )?;
        }

        self.store.write_batch(batch)?;
        Ok(records)
    }

    fn ensure_cache(
        &mut self,
        key: &AggregateKey,
    ) -> Result<(&mut AggregateMeta, &mut BTreeMap<String, String>)> {
        if !self.meta_cache.contains_key(key) {
            let (aggregate_type, aggregate_id) = key.parts();
            let meta = self
                .store
                .load_meta(aggregate_type, aggregate_id)?
                .unwrap_or_else(|| {
                    AggregateMeta::new(aggregate_type.to_string(), aggregate_id.to_string())
                });
            self.meta_cache.insert(key.clone(), meta);
        }

        if !self.state_cache.contains_key(key) {
            let (aggregate_type, aggregate_id) = key.parts();
            let state = self.store.load_state_map(aggregate_type, aggregate_id)?;
            self.state_cache.insert(key.clone(), state);
        }

        let meta = self
            .meta_cache
            .get_mut(key)
            .expect("meta cache missing after insertion");
        let state = self
            .state_cache
            .get_mut(key)
            .expect("state cache missing after insertion");
        Ok((meta, state))
    }
}

fn apply_event(
    meta: &mut AggregateMeta,
    state: &mut BTreeMap<String, String>,
    input: AppendEvent,
    event_id: SnowflakeId,
) -> EventRecord {
    let AppendEvent {
        aggregate_type,
        aggregate_id,
        event_type,
        payload,
        metadata,
        issued_by,
        note,
        tenant: _,
        reference_targets: _,
    } = input;

    debug_assert_eq!(&meta.aggregate_type, &aggregate_type);
    debug_assert_eq!(&meta.aggregate_id, &aggregate_id);

    let event_time = Utc::now();
    let version = meta.version + 1;
    meta.record_write(event_time);
    let payload_map = payload_to_map(&payload);
    let hash = hash_event(
        &aggregate_type,
        &aggregate_id,
        version,
        &event_type,
        &payload_map,
    );

    meta.event_hashes.push(hash.clone());
    meta.version = version;
    meta.merkle_root = compute_merkle_root(&meta.event_hashes);

    for (key, value) in payload_map {
        state.insert(key, value);
    }

    EventRecord {
        aggregate_type,
        aggregate_id,
        event_type,
        payload,
        extensions: metadata,
        metadata: EventMetadata {
            event_id,
            created_at: event_time,
            issued_by,
            note,
        },
        version,
        hash,
        merkle_root: meta.merkle_root.clone(),
    }
}

fn batch_put(batch: &mut WriteBatch, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
    batch.put(key, value);
    Ok(())
}

fn decode_pointer_segment(segment: &str) -> String {
    let mut result = String::with_capacity(segment.len());
    let mut chars = segment.chars();
    while let Some(ch) = chars.next() {
        if ch == '~' {
            if let Some(next) = chars.next() {
                match next {
                    '0' => result.push('~'),
                    '1' => result.push('/'),
                    other => {
                        result.push('~');
                        result.push(other);
                    }
                }
            } else {
                result.push('~');
            }
        } else {
            result.push(ch);
        }
    }
    result
}

fn top_level_key_from_path(path: &str) -> std::result::Result<String, String> {
    if !path.starts_with('/') {
        return Err(format!("patch path '{}' must start with '/'", path));
    }
    let trimmed = &path[1..];
    if trimmed.is_empty() {
        return Err("patch operations targeting the document root are not supported".into());
    }
    let segment = trimmed
        .split('/')
        .next()
        .expect("split returns at least one segment");
    Ok(decode_pointer_segment(segment))
}

fn collect_top_level_keys(patch: &Value) -> std::result::Result<BTreeSet<String>, String> {
    let array = patch
        .as_array()
        .ok_or_else(|| "patch must be an array".to_string())?;
    let mut keys = BTreeSet::new();
    for entry in array {
        let op = entry
            .get("op")
            .and_then(Value::as_str)
            .ok_or_else(|| "patch entry is missing 'op'".to_string())?;
        if op != "add" && op != "replace" && op != "remove" {
            return Err(format!("unsupported patch operation '{}'", op));
        }
        let path = entry
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| "patch entry is missing 'path'".to_string())?;
        let key = top_level_key_from_path(path)?;
        keys.insert(key);
    }
    Ok(keys)
}

fn parse_state_value(raw: &str) -> Value {
    if let Ok(parsed) = serde_json::from_str(raw) {
        return parsed;
    }
    Value::String(raw.to_string())
}

fn state_map_to_value(map: &BTreeMap<String, String>) -> Value {
    let mut object = serde_json::Map::new();
    for (key, value) in map {
        object.insert(key.clone(), parse_state_value(value));
    }
    Value::Object(object)
}

pub fn select_state_field(map: &BTreeMap<String, String>, path: &str) -> Option<Value> {
    if path.is_empty() {
        return None;
    }

    let mut segments = path.split('.');
    let first = segments.next()?;
    if first.is_empty() {
        return None;
    }

    let mut current = match map.get(first) {
        Some(raw) => parse_state_value(raw),
        None => return None,
    };

    for segment in segments {
        if segment.is_empty() {
            return None;
        }
        current = match &current {
            Value::Object(object) => object.get(segment)?.clone(),
            Value::Array(array) => {
                let index = segment.parse::<usize>().ok()?;
                array.get(index)?.clone()
            }
            _ => return None,
        };
    }

    Some(current)
}

#[derive(Debug, Clone)]
pub struct AppendEvent {
    pub aggregate_type: String,
    pub aggregate_id: String,
    pub event_type: String,
    pub payload: Value,
    pub metadata: Option<Value>,
    pub issued_by: Option<ActorClaims>,
    pub note: Option<String>,
    pub tenant: String,
    pub reference_targets: Vec<(String, String)>, // (canonical_target, path)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateState {
    pub aggregate_type: String,
    pub aggregate_id: String,
    pub version: u64,
    pub state: BTreeMap<String, String>,
    pub merkle_root: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub archived: bool,
}

#[derive(Debug, Clone)]
struct AggregateSummary {
    aggregate_type: String,
    aggregate_id: String,
    version: u64,
    merkle_root: String,
    archived: bool,
    created_at: Option<DateTime<Utc>>,
    updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct StoreCounts {
    pub active_aggregates: usize,
    pub archived_aggregates: usize,
    pub active_events: u64,
    pub archived_events: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorIndex {
    Active,
    Archived,
}

impl CursorIndex {
    fn code(self) -> char {
        match self {
            CursorIndex::Active => 'a',
            CursorIndex::Archived => 'r',
        }
    }

    fn from_str_tag(tag: &str) -> Result<Self> {
        match tag {
            "a" | "active" => Ok(CursorIndex::Active),
            "r" | "archived" => Ok(CursorIndex::Archived),
            other => Err(EventError::InvalidCursor(format!(
                "unknown cursor segment '{other}'"
            ))),
        }
    }
}

impl fmt::Display for CursorIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CursorIndex::Active => write!(f, "active"),
            CursorIndex::Archived => write!(f, "archived"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AggregateCursorKind {
    Natural,
    Timestamp {
        kind: TimestampIndexKind,
        descending: bool,
        timestamp_ms: i64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregateCursor {
    index: CursorIndex,
    aggregate_type: String,
    aggregate_id: String,
    kind: AggregateCursorKind,
}

impl AggregateCursor {
    pub fn new(
        index: CursorIndex,
        aggregate_type: impl Into<String>,
        aggregate_id: impl Into<String>,
    ) -> Self {
        Self {
            index,
            aggregate_type: aggregate_type.into(),
            aggregate_id: aggregate_id.into(),
            kind: AggregateCursorKind::Natural,
        }
    }

    pub fn new_timestamp(
        index: CursorIndex,
        aggregate_type: impl Into<String>,
        aggregate_id: impl Into<String>,
        kind: TimestampIndexKind,
        descending: bool,
        timestamp_ms: i64,
    ) -> Self {
        Self {
            index,
            aggregate_type: aggregate_type.into(),
            aggregate_id: aggregate_id.into(),
            kind: AggregateCursorKind::Timestamp {
                kind,
                descending,
                timestamp_ms,
            },
        }
    }

    pub fn index(&self) -> CursorIndex {
        self.index
    }

    pub fn aggregate_type(&self) -> &str {
        &self.aggregate_type
    }

    pub fn aggregate_id(&self) -> &str {
        &self.aggregate_id
    }

    pub fn encode(&self) -> String {
        match &self.kind {
            AggregateCursorKind::Natural => format!(
                "{}:{}:{}",
                self.index.code(),
                self.aggregate_type,
                self.aggregate_id
            ),
            AggregateCursorKind::Timestamp {
                kind,
                descending,
                timestamp_ms,
            } => format!(
                "ts:{}:{}:{}:{}:{}:{}",
                kind.as_str(),
                if *descending { "desc" } else { "asc" },
                self.index.code(),
                timestamp_ms,
                self.aggregate_type,
                self.aggregate_id
            ),
        }
    }

    pub fn is_timestamp(&self) -> bool {
        matches!(self.kind, AggregateCursorKind::Timestamp { .. })
    }

    pub fn timestamp_components(&self) -> Option<(TimestampIndexKind, bool, i64)> {
        match self.kind {
            AggregateCursorKind::Timestamp {
                kind,
                descending,
                timestamp_ms,
            } => Some((kind, descending, timestamp_ms)),
            _ => None,
        }
    }

    pub fn as_timestamp_key(
        &self,
    ) -> Option<(TimestampIndexKind, bool, i64, CursorIndex, &str, &str)> {
        match self.kind {
            AggregateCursorKind::Timestamp {
                kind,
                descending,
                timestamp_ms,
            } => Some((
                kind,
                descending,
                timestamp_ms,
                self.index,
                self.aggregate_type(),
                self.aggregate_id(),
            )),
            _ => None,
        }
    }
}

impl fmt::Display for AggregateCursor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.encode())
    }
}

impl std::str::FromStr for AggregateCursor {
    type Err = EventError;

    fn from_str(value: &str) -> Result<Self> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(EventError::InvalidCursor("cursor cannot be empty".into()));
        }
        if trimmed.starts_with("ts:") {
            let mut segments = trimmed.split(':');
            segments.next(); // consume "ts"
            let kind_str = segments
                .next()
                .ok_or_else(|| EventError::InvalidCursor("timestamp cursor missing kind".into()))?;
            let kind = TimestampIndexKind::from_str_token(kind_str).ok_or_else(|| {
                EventError::InvalidCursor(format!("unknown timestamp cursor kind '{kind_str}'"))
            })?;
            let order = segments.next().ok_or_else(|| {
                EventError::InvalidCursor("timestamp cursor missing order".into())
            })?;
            let descending = match order {
                "asc" => false,
                "desc" => true,
                _ => {
                    return Err(EventError::InvalidCursor(format!(
                        "timestamp cursor order must be 'asc' or 'desc', found '{order}'"
                    )));
                }
            };
            let index_tag = segments.next().ok_or_else(|| {
                EventError::InvalidCursor("timestamp cursor missing scope".into())
            })?;
            let index = CursorIndex::from_str_tag(index_tag)?;
            let ts_str = segments.next().ok_or_else(|| {
                EventError::InvalidCursor("timestamp cursor missing timestamp".into())
            })?;
            let timestamp_ms = ts_str.parse::<i64>().map_err(|_| {
                EventError::InvalidCursor(format!(
                    "timestamp cursor contains invalid timestamp '{ts_str}'"
                ))
            })?;
            let aggregate_type = segments.next().ok_or_else(|| {
                EventError::InvalidCursor("timestamp cursor missing aggregate_type".into())
            })?;
            if aggregate_type.is_empty() {
                return Err(EventError::InvalidCursor(
                    "timestamp cursor aggregate_type cannot be empty".into(),
                ));
            }
            let aggregate_id = segments.next().ok_or_else(|| {
                EventError::InvalidCursor("timestamp cursor missing aggregate_id".into())
            })?;
            if aggregate_id.is_empty() {
                return Err(EventError::InvalidCursor(
                    "timestamp cursor aggregate_id cannot be empty".into(),
                ));
            }
            if segments.next().is_some() {
                return Err(EventError::InvalidCursor(
                    "timestamp cursor has too many segments".into(),
                ));
            }
            return Ok(Self::new_timestamp(
                index,
                aggregate_type,
                aggregate_id,
                kind,
                descending,
                timestamp_ms,
            ));
        }
        let mut segments = trimmed.split(':');
        let Some(index_tag) = segments.next() else {
            return Err(EventError::InvalidCursor("cursor is incomplete".into()));
        };
        let index = CursorIndex::from_str_tag(index_tag)?;
        let aggregate_type = segments
            .next()
            .ok_or_else(|| EventError::InvalidCursor("cursor missing aggregate_type".into()))?;
        if aggregate_type.is_empty() {
            return Err(EventError::InvalidCursor(
                "cursor aggregate_type cannot be empty".into(),
            ));
        }
        let aggregate_id = segments
            .next()
            .ok_or_else(|| EventError::InvalidCursor("cursor missing aggregate_id".into()))?;
        if aggregate_id.is_empty() {
            return Err(EventError::InvalidCursor(
                "cursor aggregate_id cannot be empty".into(),
            ));
        }
        if segments.next().is_some() {
            return Err(EventError::InvalidCursor(
                "cursor has too many segments".into(),
            ));
        }

        Ok(Self::new(index, aggregate_type, aggregate_id))
    }
}

pub fn ensure_timestamp_cursor(
    cursor: &AggregateCursor,
    kind: TimestampIndexKind,
    descending: bool,
    scope: AggregateQueryScope,
) -> Result<()> {
    let Some((cursor_kind, cursor_desc, _, cursor_index, _, _)) = cursor.as_timestamp_key() else {
        return Err(EventError::InvalidCursor(
            "timestamp cursors must use the 'ts:' prefix (ts:<field>:<order>:...)".into(),
        ));
    };
    if cursor_kind != kind {
        return Err(EventError::InvalidCursor(format!(
            "timestamp cursor targets '{}' but sort is '{}'",
            cursor_kind.as_str(),
            kind.as_str()
        )));
    }
    if cursor_desc != descending {
        return Err(EventError::InvalidCursor(
            "timestamp cursor order does not match the requested sort direction".into(),
        ));
    }
    match scope {
        AggregateQueryScope::ActiveOnly => {
            if cursor_index != CursorIndex::Active {
                return Err(EventError::InvalidCursor(
                    "timestamp cursor must reference active aggregates in this scope".into(),
                ));
            }
        }
        AggregateQueryScope::ArchivedOnly => {
            if cursor_index != CursorIndex::Archived {
                return Err(EventError::InvalidCursor(
                    "timestamp cursor must reference archived aggregates in this scope".into(),
                ));
            }
        }
        AggregateQueryScope::IncludeArchived => {}
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventCursor {
    index: CursorIndex,
    aggregate_type: String,
    aggregate_id: String,
    version: u64,
}

impl EventCursor {
    pub fn new(
        index: CursorIndex,
        aggregate_type: impl Into<String>,
        aggregate_id: impl Into<String>,
        version: u64,
    ) -> Self {
        Self {
            index,
            aggregate_type: aggregate_type.into(),
            aggregate_id: aggregate_id.into(),
            version,
        }
    }

    pub fn index(&self) -> CursorIndex {
        self.index
    }

    pub fn aggregate_type(&self) -> &str {
        &self.aggregate_type
    }

    pub fn aggregate_id(&self) -> &str {
        &self.aggregate_id
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn encode(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.index.code(),
            self.aggregate_type,
            self.aggregate_id,
            self.version
        )
    }
}

impl fmt::Display for EventCursor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}:{}",
            self.index.code(),
            self.aggregate_type,
            self.aggregate_id,
            self.version
        )
    }
}

impl std::str::FromStr for EventCursor {
    type Err = EventError;

    fn from_str(value: &str) -> Result<Self> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(EventError::InvalidCursor("cursor cannot be empty".into()));
        }
        let mut segments = trimmed.split(':');
        let Some(index_tag) = segments.next() else {
            return Err(EventError::InvalidCursor("cursor is incomplete".into()));
        };
        let index = CursorIndex::from_str_tag(index_tag)?;
        let aggregate_type = segments
            .next()
            .ok_or_else(|| EventError::InvalidCursor("cursor missing aggregate_type".into()))?;
        if aggregate_type.is_empty() {
            return Err(EventError::InvalidCursor(
                "cursor aggregate_type cannot be empty".into(),
            ));
        }
        let aggregate_id = segments
            .next()
            .ok_or_else(|| EventError::InvalidCursor("cursor missing aggregate_id".into()))?;
        if aggregate_id.is_empty() {
            return Err(EventError::InvalidCursor(
                "cursor aggregate_id cannot be empty".into(),
            ));
        }
        let version_str = segments
            .next()
            .ok_or_else(|| EventError::InvalidCursor("cursor missing version".into()))?;
        if segments.next().is_some() {
            return Err(EventError::InvalidCursor(
                "cursor has too many segments".into(),
            ));
        }
        let version = version_str.parse::<u64>().map_err(|_| {
            EventError::InvalidCursor(format!("cursor version '{}' is not a number", version_str))
        })?;

        Ok(Self::new(index, aggregate_type, aggregate_id, version))
    }
}

impl StoreCounts {
    pub fn total_aggregates(&self) -> usize {
        self.active_aggregates + self.archived_aggregates
    }

    pub fn total_events(&self) -> u64 {
        self.active_events + self.archived_events
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateSortField {
    AggregateType,
    AggregateId,
    Archived,
    CreatedAt,
    UpdatedAt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AggregateSort {
    pub field: AggregateSortField,
    pub descending: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateQueryScope {
    ActiveOnly,
    ArchivedOnly,
    IncludeArchived,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventSortField {
    AggregateType,
    AggregateId,
    EventType,
    Version,
    EventId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventSort {
    pub field: EventSortField,
    pub descending: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventQueryScope<'a> {
    All,
    AggregateType(&'a str),
    Aggregate {
        aggregate_type: &'a str,
        aggregate_id: &'a str,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventArchiveScope {
    ActiveOnly,
    ArchivedOnly,
    IncludeArchived,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AggregateIndex {
    Active,
    Archived,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimestampIndexKind {
    CreatedAt,
    UpdatedAt,
}

impl From<AggregateIndex> for CursorIndex {
    fn from(value: AggregateIndex) -> Self {
        match value {
            AggregateIndex::Active => CursorIndex::Active,
            AggregateIndex::Archived => CursorIndex::Archived,
        }
    }
}

impl From<CursorIndex> for AggregateIndex {
    fn from(value: CursorIndex) -> Self {
        match value {
            CursorIndex::Active => AggregateIndex::Active,
            CursorIndex::Archived => AggregateIndex::Archived,
        }
    }
}

impl AggregateSortField {
    pub fn as_str(self) -> &'static str {
        match self {
            AggregateSortField::AggregateType => "aggregate_type",
            AggregateSortField::AggregateId => "aggregate_id",
            AggregateSortField::Archived => "archived",
            AggregateSortField::CreatedAt => "created_at",
            AggregateSortField::UpdatedAt => "updated_at",
        }
    }

    pub fn compare(self, lhs: &AggregateState, rhs: &AggregateState) -> StdOrdering {
        match self {
            AggregateSortField::AggregateType => lhs.aggregate_type.cmp(&rhs.aggregate_type),
            AggregateSortField::AggregateId => lhs.aggregate_id.cmp(&rhs.aggregate_id),
            AggregateSortField::Archived => lhs.archived.cmp(&rhs.archived),
            AggregateSortField::CreatedAt => lhs.created_at.cmp(&rhs.created_at),
            AggregateSortField::UpdatedAt => lhs.updated_at.cmp(&rhs.updated_at),
        }
    }
}

impl AggregateSort {
    pub fn compare(self, lhs: &AggregateState, rhs: &AggregateState) -> StdOrdering {
        let ordering = self.field.compare(lhs, rhs);
        if self.descending {
            ordering.reverse()
        } else {
            ordering
        }
    }

    pub fn parse_directives(raw: &str) -> std::result::Result<Vec<Self>, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err("sort specification cannot be empty".to_string());
        }

        let mut directives = Vec::new();
        for segment in trimmed.split(',') {
            let spec = segment.trim();
            if spec.is_empty() {
                return Err("sort segments cannot be empty".to_string());
            }
            directives.push(Self::parse_single(spec)?);
        }

        Ok(directives)
    }

    fn parse_single(spec: &str) -> std::result::Result<Self, String> {
        let mut parts = spec.split(':');
        let field_str = parts
            .next()
            .ok_or_else(|| "missing sort field".to_string())?
            .trim();

        if field_str.is_empty() {
            return Err("sort field cannot be empty".to_string());
        }

        let field = field_str.parse::<AggregateSortField>()?;
        let descending = match parts.next() {
            Some(order) => match order.trim().to_ascii_lowercase().as_str() {
                "asc" => false,
                "desc" => true,
                other => {
                    return Err(format!(
                        "invalid sort order '{other}' (expected 'asc' or 'desc')"
                    ));
                }
            },
            None => false,
        };

        if parts.next().is_some() {
            return Err("sort specification contains too many ':' separators".to_string());
        }

        Ok(Self { field, descending })
    }
}

impl EventSortField {
    pub fn as_str(self) -> &'static str {
        match self {
            EventSortField::AggregateType => "aggregate_type",
            EventSortField::AggregateId => "aggregate_id",
            EventSortField::EventType => "event_type",
            EventSortField::Version => "version",
            EventSortField::EventId => "event_id",
        }
    }

    pub fn compare(self, lhs: &EventRecord, rhs: &EventRecord) -> StdOrdering {
        match self {
            EventSortField::AggregateType => lhs.aggregate_type.cmp(&rhs.aggregate_type),
            EventSortField::AggregateId => lhs.aggregate_id.cmp(&rhs.aggregate_id),
            EventSortField::EventType => lhs.event_type.cmp(&rhs.event_type),
            EventSortField::Version => lhs.version.cmp(&rhs.version),
            EventSortField::EventId => lhs.metadata.event_id.cmp(&rhs.metadata.event_id),
        }
    }
}

impl EventSort {
    pub fn compare(self, lhs: &EventRecord, rhs: &EventRecord) -> StdOrdering {
        let ordering = self.field.compare(lhs, rhs);
        if self.descending {
            ordering.reverse()
        } else {
            ordering
        }
    }
}

impl std::str::FromStr for AggregateSortField {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "aggregate_type" | "aggregatetype" => Ok(AggregateSortField::AggregateType),
            "aggregate_id" | "aggregateid" => Ok(AggregateSortField::AggregateId),
            "archived" => Ok(AggregateSortField::Archived),
            "created_at" | "createdat" => Ok(AggregateSortField::CreatedAt),
            "updated_at" | "updatedat" => Ok(AggregateSortField::UpdatedAt),
            _ => Err(format!("unsupported aggregate sort field '{value}'")),
        }
    }
}

impl std::str::FromStr for EventSortField {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "aggregate_type" | "aggregatetype" => Ok(EventSortField::AggregateType),
            "aggregate_id" | "aggregateid" => Ok(EventSortField::AggregateId),
            "event_type" | "eventtype" => Ok(EventSortField::EventType),
            "version" => Ok(EventSortField::Version),
            "event_id" | "eventid" => Ok(EventSortField::EventId),
            _ => Err(format!("unsupported event sort field '{value}'")),
        }
    }
}

impl AggregateIndex {
    fn meta_prefix(self) -> &'static str {
        match self {
            AggregateIndex::Active => PREFIX_META,
            AggregateIndex::Archived => PREFIX_META_ARCHIVED,
        }
    }

    fn state_prefix(self) -> &'static str {
        match self {
            AggregateIndex::Active => PREFIX_STATE,
            AggregateIndex::Archived => PREFIX_STATE_ARCHIVED,
        }
    }

    fn event_prefix(self) -> &'static str {
        match self {
            AggregateIndex::Active => PREFIX_EVENT,
            AggregateIndex::Archived => PREFIX_EVENT_ARCHIVED,
        }
    }
}

impl TimestampIndexKind {
    fn prefix(self, index: AggregateIndex) -> &'static str {
        match (self, index) {
            (TimestampIndexKind::CreatedAt, AggregateIndex::Active) => PREFIX_META_CREATED_INDEX,
            (TimestampIndexKind::CreatedAt, AggregateIndex::Archived) => {
                PREFIX_META_CREATED_INDEX_ARCHIVED
            }
            (TimestampIndexKind::UpdatedAt, AggregateIndex::Active) => PREFIX_META_UPDATED_INDEX,
            (TimestampIndexKind::UpdatedAt, AggregateIndex::Archived) => {
                PREFIX_META_UPDATED_INDEX_ARCHIVED
            }
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            TimestampIndexKind::CreatedAt => "created_at",
            TimestampIndexKind::UpdatedAt => "updated_at",
        }
    }

    fn from_str_token(value: &str) -> Option<Self> {
        match value {
            "created_at" | "created" => Some(TimestampIndexKind::CreatedAt),
            "updated_at" | "updated" => Some(TimestampIndexKind::UpdatedAt),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct TimestampIndexEntry {
    aggregate_type: String,
    aggregate_id: String,
    timestamp_ms: i64,
}

struct TimestampIndexIterator<'a> {
    iter: DBIterator<'a>,
    prefix: Vec<u8>,
    kind: TimestampIndexKind,
    index: AggregateIndex,
    finished: bool,
    had_error: bool,
    start_time: Instant,
}

impl<'a> TimestampIndexIterator<'a> {
    fn new(
        store: &'a EventStore,
        kind: TimestampIndexKind,
        index: AggregateIndex,
        descending: bool,
    ) -> Self {
        let prefix = timestamp_index_prefix(kind, index);
        let mut start_key = prefix.clone();
        let direction = if descending {
            start_key.push(u8::MAX);
            Direction::Reverse
        } else {
            Direction::Forward
        };
        let iter = store
            .db
            .iterator(IteratorMode::From(start_key.as_slice(), direction));
        Self {
            iter,
            prefix,
            kind,
            index,
            finished: false,
            had_error: false,
            start_time: Instant::now(),
        }
    }

    fn next_entry(&mut self) -> Option<TimestampIndexEntry> {
        if self.finished {
            return None;
        }
        while let Some(item) = self.iter.next() {
            let (key, _) = match item {
                Ok(pair) => pair,
                Err(_) => {
                    self.had_error = true;
                    continue;
                }
            };
            if !key.starts_with(self.prefix.as_slice()) {
                self.finished = true;
                break;
            }
            if let Some(entry) = parse_timestamp_index_entry(self.prefix.as_slice(), &key) {
                return Some(entry);
            }
        }
        None
    }

    fn mark_error(&mut self) {
        self.had_error = true;
    }
}

impl Drop for TimestampIndexIterator<'_> {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed().as_secs_f64();
        let status = if self.had_error { "err" } else { "ok" };
        record_store_op(
            timestamp_index_metric(self.kind, self.index),
            status,
            duration,
        );
    }
}

fn compare_aggregate_sort_keys(
    lhs: &AggregateState,
    rhs: &AggregateState,
    keys: &[AggregateSort],
) -> StdOrdering {
    for key in keys {
        let ordering = key.compare(lhs, rhs);
        if ordering != StdOrdering::Equal {
            return ordering;
        }
    }
    StdOrdering::Equal
}

fn compare_event_sort_keys(
    lhs: &EventRecord,
    rhs: &EventRecord,
    keys: &[EventSort],
) -> StdOrdering {
    for key in keys {
        let ordering = key.compare(lhs, rhs);
        if ordering != StdOrdering::Equal {
            return ordering;
        }
    }
    StdOrdering::Equal
}

fn aggregate_sort_matches_key_order(keys: &[AggregateSort]) -> bool {
    if keys.is_empty() {
        return true;
    }
    const NATURAL_ORDER: [AggregateSortField; 2] = [
        AggregateSortField::AggregateType,
        AggregateSortField::AggregateId,
    ];
    keys.iter().enumerate().all(|(idx, key)| {
        !key.descending
            && NATURAL_ORDER
                .get(idx)
                .map(|field| key.field == *field)
                .unwrap_or(false)
    })
}

pub fn timestamp_sort_hint(keys: &[AggregateSort]) -> Option<(TimestampIndexKind, bool)> {
    if keys.len() != 1 {
        return None;
    }
    let directive = keys[0];
    let kind = match directive.field {
        AggregateSortField::CreatedAt => TimestampIndexKind::CreatedAt,
        AggregateSortField::UpdatedAt => TimestampIndexKind::UpdatedAt,
        _ => return None,
    };
    Some((kind, directive.descending))
}

fn event_sort_matches_key_order(keys: &[EventSort]) -> bool {
    if keys.is_empty() {
        return true;
    }
    const NATURAL_ORDER: [EventSortField; 3] = [
        EventSortField::AggregateType,
        EventSortField::AggregateId,
        EventSortField::Version,
    ];
    keys.iter().enumerate().all(|(idx, key)| {
        !key.descending
            && NATURAL_ORDER
                .get(idx)
                .map(|field| key.field == *field)
                .unwrap_or(false)
    })
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AggregateMeta {
    aggregate_type: String,
    aggregate_id: String,
    version: u64,
    event_hashes: Vec<String>,
    merkle_root: String,
    #[serde(default)]
    archived: bool,
    #[serde(default)]
    archived_at: Option<DateTime<Utc>>,
    #[serde(default)]
    archive_comment: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    created_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    updated_at: Option<DateTime<Utc>>,
}

impl AggregateMeta {
    fn new(aggregate_type: String, aggregate_id: String) -> Self {
        Self {
            aggregate_type,
            aggregate_id,
            version: 0,
            event_hashes: Vec::new(),
            merkle_root: empty_root(),
            archived: false,
            archived_at: None,
            archive_comment: None,
            created_at: None,
            updated_at: None,
        }
    }

    fn record_write(&mut self, event_time: DateTime<Utc>) {
        if self.version == 0 && self.created_at.is_none() {
            self.created_at = Some(event_time);
        }
        self.updated_at = Some(event_time);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct AggregateKey {
    aggregate_type: String,
    aggregate_id: String,
}

impl AggregateKey {
    fn new(aggregate_type: impl Into<String>, aggregate_id: impl Into<String>) -> Self {
        Self {
            aggregate_type: aggregate_type.into(),
            aggregate_id: aggregate_id.into(),
        }
    }

    fn parts(&self) -> (&str, &str) {
        (&self.aggregate_type, &self.aggregate_id)
    }
}

struct PendingEvent {
    key: Vec<u8>,
    value: Vec<u8>,
}

struct PendingRefUpdate {
    tenant: String,
    aggregate_type: String,
    aggregate_id: String,
    references: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredReferencePath {
    target: String,
    path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredReferrer {
    aggregate_type: String,
    aggregate_id: String,
    path: String,
}

pub struct EventStore {
    db: DBWithThreadMode<MultiThreaded>,
    write_lock: Mutex<()>,
    read_only: bool,
    encryptor: Option<Encryptor>,
    id_generator: Mutex<SnowflakeGenerator>,
}

pub struct Transaction<'a> {
    store: &'a EventStore,
    _guard: MutexGuard<'a, ()>,
    pending_events: Vec<PendingEvent>,
    pending_records: Vec<EventRecord>,
    pending_refs: Vec<PendingRefUpdate>,
    meta_cache: BTreeMap<AggregateKey, AggregateMeta>,
    state_cache: BTreeMap<AggregateKey, BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_id: Option<SnowflakeId>,
    pub aggregate_type: String,
    pub aggregate_id: String,
    pub version: u64,
    pub state: BTreeMap<String, String>,
    pub merkle_root: String,
    pub created_at: DateTime<Utc>,
    pub comment: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AggregatePositionEntry {
    pub aggregate_type: String,
    pub aggregate_id: String,
    pub version: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LockStatus {
    NotLocked,
    LockedBy(Option<u32>),
    Unknown,
}

fn open_db_with_lock_handling(
    path: &Path,
    options: &Options,
    read_only: bool,
) -> Result<DBWithThreadMode<MultiThreaded>> {
    let lock_path = path.join("LOCK");
    let mut last_error = None;

    for attempt in 0..=LOCK_RETRY_ATTEMPTS {
        match try_open_db(path, options, read_only) {
            Ok(db) => return Ok(db),
            Err(err) => {
                let message = err.to_string();
                last_error = Some(message.clone());
                if !is_lock_error_message(&message) {
                    return Err(EventError::Storage(message));
                }

                let status = lock_status(&lock_path)?;
                match status {
                    LockStatus::NotLocked => {
                        cleanup_stale_lock(&lock_path);
                    }
                    LockStatus::LockedBy(pid_opt) => {
                        if pid_opt.map_or(false, pid_running) {
                            if attempt == LOCK_RETRY_ATTEMPTS {
                                return Err(EventError::Storage(store_in_use_message(
                                    path, pid_opt,
                                )));
                            }
                        } else {
                            cleanup_stale_lock(&lock_path);
                        }
                    }
                    LockStatus::Unknown => {
                        if attempt == LOCK_RETRY_ATTEMPTS {
                            return Err(EventError::Storage(store_in_use_message(path, None)));
                        }
                    }
                }
            }
        }

        if attempt < LOCK_RETRY_ATTEMPTS {
            thread::sleep(LOCK_RETRY_BACKOFF);
        }
    }

    Err(EventError::Storage(
        last_error.unwrap_or_else(|| "failed to open event store".into()),
    ))
}

fn try_open_db(
    path: &Path,
    options: &Options,
    read_only: bool,
) -> std::result::Result<DBWithThreadMode<MultiThreaded>, rocksdb::Error> {
    if read_only {
        DBWithThreadMode::<MultiThreaded>::open_for_read_only(options, path, false)
    } else {
        DBWithThreadMode::<MultiThreaded>::open(options, path)
    }
}

fn lock_status(lock_path: &Path) -> io::Result<LockStatus> {
    if !lock_path.exists() {
        return Ok(LockStatus::NotLocked);
    }

    #[cfg(unix)]
    {
        let file = std::fs::File::open(lock_path)?;
        let mut flock = libc::flock {
            l_type: libc::F_WRLCK as libc::c_short,
            l_whence: libc::SEEK_SET as libc::c_short,
            l_start: 0,
            l_len: 0,
            l_pid: 0,
        };
        let status = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETLK, &mut flock) };
        if status == -1 {
            return Ok(LockStatus::Unknown);
        }
        if flock.l_type == libc::F_UNLCK as libc::c_short {
            return Ok(LockStatus::NotLocked);
        }
        return Ok(LockStatus::LockedBy(Some(flock.l_pid as u32)));
    }

    #[cfg(not(unix))]
    {
        let _ = lock_path;
        Ok(LockStatus::Unknown)
    }
}

fn pid_running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
        if result == 0 {
            return true;
        }
        if let Some(errno) = io::Error::last_os_error().raw_os_error() {
            return errno != libc::ESRCH;
        }
        return false;
    }

    #[cfg(windows)]
    {
        use windows_sys::Win32::{
            Foundation::{CloseHandle, STILL_ACTIVE},
            System::Threading::{
                GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
            },
        };

        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle == 0 {
                return false;
            }
            let mut exit_code = 0;
            let success = GetExitCodeProcess(handle, &mut exit_code);
            let _ = CloseHandle(handle);
            success != 0 && exit_code == STILL_ACTIVE as u32
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        true
    }
}

fn cleanup_stale_lock(lock_path: &Path) {
    if let Err(err) = fs::remove_file(lock_path) {
        if err.kind() != io::ErrorKind::NotFound {
            warn!(
                "failed to remove stale lock file {}: {}",
                lock_path.display(),
                err
            );
        }
    } else {
        warn!(
            "removed stale event store lock file at {}",
            lock_path.display()
        );
    }
}

fn store_in_use_message(path: &Path, pid: Option<u32>) -> String {
    match pid {
        Some(pid) => format!(
            "event store lock at {} is held by process {pid}; stop the running dbx instance and retry",
            path.display()
        ),
        None => format!(
            "event store at {} is currently in use; stop the running dbx instance and retry",
            path.display()
        ),
    }
}

pub fn is_lock_error_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("lock file")
        || lower.contains("resource temporarily unavailable")
        || lower.contains("lock is held")
        || lower.contains("is held by process")
        || lower.contains("lock hold")
        || lower.contains("currently in use")
        || lower.contains("no locks available")
}

impl EventStore {
    pub fn open(path: PathBuf, encryptor: Option<Encryptor>, worker_id: u16) -> Result<Self> {
        let mut options = Options::default();
        options.create_if_missing(true);
        configure_db_options(&mut options, false);
        let db = open_db_with_lock_handling(&path, &options, false)?;

        if worker_id > MAX_WORKER_ID {
            return Err(EventError::Config(format!(
                "snowflake worker id {} exceeds maximum {}",
                worker_id, MAX_WORKER_ID
            )));
        }

        Ok(Self {
            db,
            write_lock: Mutex::new(()),
            read_only: false,
            encryptor,
            id_generator: Mutex::new(SnowflakeGenerator::new(worker_id)),
        })
    }

    pub fn open_read_only(path: PathBuf, encryptor: Option<Encryptor>) -> Result<Self> {
        let mut options = Options::default();
        options.create_if_missing(false);
        configure_db_options(&mut options, true);
        let db = open_db_with_lock_handling(&path, &options, true)?;

        Ok(Self {
            db,
            write_lock: Mutex::new(()),
            read_only: true,
            encryptor,
            id_generator: Mutex::new(SnowflakeGenerator::new(0)),
        })
    }

    pub fn storage_usage_bytes(&self) -> Result<u64> {
        let sst = self.rocksdb_property_bytes("rocksdb.total-sst-files-size")?;
        let mem = self.rocksdb_property_bytes("rocksdb.size-all-mem-tables")?;
        let wal = self.rocksdb_property_bytes("rocksdb.live-wal-file-size")?;
        Ok(sst.saturating_add(mem).saturating_add(wal))
    }

    fn rocksdb_property_bytes(&self, name: &str) -> Result<u64> {
        match self.db.property_int_value(name) {
            Ok(Some(value)) => Ok(value),
            Ok(None) => Ok(0),
            Err(err) => Err(EventError::Storage(format!(
                "failed to read RocksDB property '{name}': {err}"
            ))),
        }
    }

    pub fn counts(&self) -> Result<StoreCounts> {
        let (active_aggregates, active_events) =
            self.count_index_entries(AggregateIndex::Active)?;
        let (archived_aggregates, archived_events) =
            self.count_index_entries(AggregateIndex::Archived)?;

        Ok(StoreCounts {
            active_aggregates,
            archived_aggregates,
            active_events,
            archived_events,
        })
    }

    fn count_index_entries(&self, index: AggregateIndex) -> Result<(usize, u64)> {
        let start = Instant::now();
        let metric = match index {
            AggregateIndex::Active => "rocksdb_iter_meta_count",
            AggregateIndex::Archived => "rocksdb_iter_meta_archived_count",
        };

        let result = (|| {
            let prefix = meta_prefix_for(index);
            let iter = self
                .db
                .iterator(IteratorMode::From(prefix.as_slice(), Direction::Forward));

            let mut aggregates = 0usize;
            let mut events = 0u64;

            for item in iter {
                let (key, value) = item.map_err(|err| EventError::Storage(err.to_string()))?;
                if !key.starts_with(prefix.as_slice()) {
                    break;
                }
                if key.len() > prefix.len() && key[prefix.len()] != SEP {
                    break;
                }

                let meta: AggregateMeta = serde_json::from_slice(&value).map_err(|err| {
                    EventError::Storage(format!("failed to deserialize aggregate metadata: {err}"))
                })?;
                aggregates += 1;
                events += meta.version;
            }

            Ok((aggregates, events))
        })();

        let duration = start.elapsed().as_secs_f64();
        record_store_op(metric, if result.is_ok() { "ok" } else { "err" }, duration);
        result
    }

    fn next_event_id(&self) -> SnowflakeId {
        self.next_snowflake_id()
    }

    fn next_snowflake_id(&self) -> SnowflakeId {
        let mut guard = self.id_generator.lock();
        guard.next_id()
    }

    pub fn prepare_payload_from_patch(
        &self,
        aggregate_type: &str,
        aggregate_id: &str,
        patch_value: &Value,
    ) -> Result<Value> {
        if !patch_value.is_array() {
            return Err(EventError::InvalidSchema(
                "patch must be provided as a JSON array".into(),
            ));
        }

        let top_level_keys =
            collect_top_level_keys(patch_value).map_err(EventError::InvalidSchema)?;
        if top_level_keys.is_empty() {
            return Err(EventError::InvalidSchema(
                "patch must modify at least one path".into(),
            ));
        }

        let patch: Patch = serde_json::from_value(patch_value.clone())
            .map_err(|err| EventError::InvalidSchema(format!("failed to parse patch: {}", err)))?;

        let state_map = self.load_state_map(aggregate_type, aggregate_id)?;
        let mut document = state_map_to_value(&state_map);
        if !document.is_object() {
            document = Value::Object(serde_json::Map::new());
        }

        json_patch::patch(&mut document, &patch)
            .map_err(|err| EventError::InvalidSchema(format!("failed to apply patch: {}", err)))?;

        let document_object = document.as_object().ok_or_else(|| {
            EventError::InvalidSchema("patched document must be a JSON object".into())
        })?;

        let mut payload = serde_json::Map::new();
        for key in top_level_keys {
            let value = document_object.get(&key).ok_or_else(|| {
                EventError::InvalidSchema(format!(
                    "patch removed key '{}' and removals are not supported",
                    key
                ))
            })?;
            payload.insert(key, value.clone());
        }

        Ok(Value::Object(payload))
    }

    pub fn append(&self, input: AppendEvent) -> Result<EventRecord> {
        self.ensure_writable()?;
        let _guard = self.write_lock.lock();

        let tenant = input.tenant.clone();
        let references = input.reference_targets.clone();
        if let Some(note) = input.note.as_ref() {
            if note.chars().count() > MAX_EVENT_NOTE_LENGTH {
                return Err(EventError::InvalidSchema(format!(
                    "event note cannot exceed {} characters",
                    MAX_EVENT_NOTE_LENGTH
                )));
            }
        }

        let aggregate_type = input.aggregate_type.clone();
        let aggregate_id = input.aggregate_id.clone();
        let event_id = self.next_event_id();

        let (mut meta, mut state, previous_meta) =
            match self.load_meta_any(&aggregate_type, &aggregate_id)? {
                Some((_, AggregateIndex::Archived)) => {
                    return Err(EventError::AggregateArchived);
                }
                Some((meta, AggregateIndex::Active)) => {
                    let previous = meta.clone();
                    (
                        meta,
                        self.load_state_map_from(
                            AggregateIndex::Active,
                            &aggregate_type,
                            &aggregate_id,
                        )?,
                        Some(previous),
                    )
                }
                None => (
                    AggregateMeta::new(aggregate_type.clone(), aggregate_id.clone()),
                    BTreeMap::new(),
                    None,
                ),
            };
        let record = apply_event(&mut meta, &mut state, input, event_id);
        let stored_record = self.encode_record(&record)?;
        let encoded_state = self.encode_state_map(&state)?;

        let mut batch = WriteBatch::default();
        batch_put(
            &mut batch,
            event_key(&record.aggregate_type, &record.aggregate_id, record.version),
            serde_json::to_vec(&stored_record)?,
        )?;
        batch_put(
            &mut batch,
            meta_key(&record.aggregate_type, &record.aggregate_id),
            serde_json::to_vec(&meta)?,
        )?;
        batch_put(
            &mut batch,
            state_key(&record.aggregate_type, &record.aggregate_id),
            encoded_state,
        )?;
        update_reference_index_inner(
            &self.db,
            &mut batch,
            &tenant,
            &record.aggregate_type,
            &record.aggregate_id,
            &references,
        )?;
        self.sync_timestamp_indexes(
            &mut batch,
            AggregateIndex::Active,
            previous_meta.as_ref(),
            Some(&meta),
        );

        self.write_batch(batch)?;

        Ok(record)
    }

    pub fn append_imported_event(&self, record: EventRecord) -> Result<()> {
        self.ensure_writable()?;
        let _guard = self.write_lock.lock();

        let aggregate_type = record.aggregate_type.clone();
        let aggregate_id = record.aggregate_id.clone();

        let (mut meta, mut state, previous_meta) =
            match self.load_meta_any(&aggregate_type, &aggregate_id)? {
                Some((_, AggregateIndex::Archived)) => {
                    return Err(EventError::AggregateArchived);
                }
                Some((meta, AggregateIndex::Active)) => {
                    let previous = meta.clone();
                    (
                        meta,
                        self.load_state_map_from(
                            AggregateIndex::Active,
                            &aggregate_type,
                            &aggregate_id,
                        )?,
                        Some(previous),
                    )
                }
                None => (
                    AggregateMeta::new(aggregate_type.clone(), aggregate_id.clone()),
                    BTreeMap::new(),
                    None,
                ),
            };

        if meta.version + 1 != record.version {
            return Err(EventError::Storage(format!(
                "imported event version mismatch for {}::{} (expected {}, found {})",
                aggregate_type,
                aggregate_id,
                meta.version + 1,
                record.version
            )));
        }
        let payload_map = payload_to_map(&record.payload);

        let computed_hash = hash_event(
            &aggregate_type,
            &aggregate_id,
            record.version,
            &record.event_type,
            &payload_map,
        );

        if computed_hash != record.hash {
            return Err(EventError::Storage(format!(
                "imported event hash mismatch for {}::{} (expected {}, found {})",
                aggregate_type, aggregate_id, computed_hash, record.hash
            )));
        }

        for (key, value) in payload_map {
            state.insert(key, value);
        }

        let record_created_at = record.metadata.created_at;
        meta.record_write(record_created_at);
        meta.event_hashes.push(record.hash.clone());
        meta.version = record.version;
        meta.merkle_root = compute_merkle_root(&meta.event_hashes);

        if meta.merkle_root != record.merkle_root {
            return Err(EventError::Storage(format!(
                "imported event merkle mismatch for {}::{} (expected {}, found {})",
                aggregate_type, aggregate_id, meta.merkle_root, record.merkle_root
            )));
        }

        let stored_record = self.encode_record(&record)?;

        let mut batch = WriteBatch::default();
        batch_put(
            &mut batch,
            event_key(&aggregate_type, &aggregate_id, record.version),
            serde_json::to_vec(&stored_record)?,
        )?;
        self.sync_timestamp_indexes(
            &mut batch,
            AggregateIndex::Active,
            previous_meta.as_ref(),
            Some(&meta),
        );
        batch_put(
            &mut batch,
            meta_key(&aggregate_type, &aggregate_id),
            serde_json::to_vec(&meta)?,
        )?;
        batch_put(
            &mut batch,
            state_key(&aggregate_type, &aggregate_id),
            self.encode_state_map(&state)?,
        )?;

        self.write_batch(batch)?;
        Ok(())
    }

    pub fn transaction(&self) -> Result<Transaction<'_>> {
        self.ensure_writable()?;
        let guard = self.write_lock.lock();
        Ok(Transaction {
            store: self,
            _guard: guard,
            pending_events: Vec::new(),
            pending_records: Vec::new(),
            pending_refs: Vec::new(),
            meta_cache: BTreeMap::new(),
            state_cache: BTreeMap::new(),
        })
    }

    pub fn get_aggregate_state(
        &self,
        aggregate_type: &str,
        aggregate_id: &str,
    ) -> Result<AggregateState> {
        let (meta, index) = self
            .load_meta_any(aggregate_type, aggregate_id)?
            .ok_or(EventError::AggregateNotFound)?;
        let state = self.load_state_map_from(index, aggregate_type, aggregate_id)?;

        let AggregateMeta {
            aggregate_type,
            aggregate_id,
            version,
            merkle_root,
            archived,
            created_at,
            updated_at,
            ..
        } = meta;

        Ok(AggregateState {
            aggregate_type,
            aggregate_id,
            version,
            state,
            merkle_root,
            created_at,
            updated_at,
            archived,
        })
    }

    pub fn parse_timestamp_cursor(
        &self,
        raw: &str,
        kind: TimestampIndexKind,
        descending: bool,
    ) -> Result<AggregateCursor> {
        if let Some(cursor) = self.try_shorthand_timestamp_cursor(raw, kind, descending)? {
            return Ok(cursor);
        }

        AggregateCursor::from_str(raw)
    }

    fn try_shorthand_timestamp_cursor(
        &self,
        raw: &str,
        kind: TimestampIndexKind,
        descending: bool,
    ) -> Result<Option<AggregateCursor>> {
        let parts: Vec<_> = raw.trim().split(':').collect();
        if parts.len() != 3 || parts[0] != "ts" {
            return Ok(None);
        }
        let aggregate_type = parts[1];
        let aggregate_id = parts[2];
        if aggregate_type.is_empty() {
            return Err(EventError::InvalidCursor(
                "timestamp cursor aggregate_type cannot be empty".into(),
            ));
        }
        if aggregate_id.is_empty() {
            return Err(EventError::InvalidCursor(
                "timestamp cursor aggregate_id cannot be empty".into(),
            ));
        }

        let aggregate = match self.get_aggregate_state(aggregate_type, aggregate_id) {
            Ok(aggregate) => aggregate,
            Err(EventError::AggregateNotFound) => {
                return Err(EventError::InvalidCursor(format!(
                    "cursor references unknown aggregate '{}:{}'",
                    aggregate_type, aggregate_id
                )));
            }
            Err(err) => return Err(err),
        };
        let timestamp = match kind {
            TimestampIndexKind::CreatedAt => aggregate.created_at,
            TimestampIndexKind::UpdatedAt => aggregate.updated_at,
        }
        .ok_or_else(|| {
            EventError::InvalidCursor(format!(
                "aggregate {}:{} missing {} timestamp",
                aggregate_type,
                aggregate_id,
                match kind {
                    TimestampIndexKind::CreatedAt => "created_at",
                    TimestampIndexKind::UpdatedAt => "updated_at",
                }
            ))
        })?;

        let index = if aggregate.archived {
            CursorIndex::Archived
        } else {
            CursorIndex::Active
        };

        Ok(Some(AggregateCursor::new_timestamp(
            index,
            aggregate.aggregate_type,
            aggregate.aggregate_id,
            kind,
            descending,
            timestamp.timestamp_millis(),
        )))
    }

    pub fn list_events(
        &self,
        aggregate_type: &str,
        aggregate_id: &str,
    ) -> Result<Vec<EventRecord>> {
        self.list_events_paginated(aggregate_type, aggregate_id, 0, None)
    }

    pub fn merkle_root_at(
        &self,
        aggregate_type: &str,
        aggregate_id: &str,
        version: u64,
    ) -> Result<Option<String>> {
        if version == 0 {
            return Ok(Some(empty_root()));
        }
        let Some((meta, _)) = self.load_meta_any(aggregate_type, aggregate_id)? else {
            return Ok(None);
        };
        if version > meta.version {
            return Ok(None);
        }
        if version == meta.version {
            return Ok(Some(meta.merkle_root.clone()));
        }
        let end = version as usize;
        let hashes = &meta.event_hashes[..end];
        Ok(Some(compute_merkle_root(hashes)))
    }

    pub fn event_hashes(
        &self,
        aggregate_type: &str,
        aggregate_id: &str,
    ) -> Result<Option<Vec<String>>> {
        let Some((meta, _)) = self.load_meta_any(aggregate_type, aggregate_id)? else {
            return Ok(None);
        };
        Ok(Some(meta.event_hashes.clone()))
    }

    pub fn event_by_version(
        &self,
        aggregate_type: &str,
        aggregate_id: &str,
        version: u64,
    ) -> Result<Option<EventRecord>> {
        if version == 0 {
            return Ok(None);
        }
        let (meta, index) = match self.load_meta_any(aggregate_type, aggregate_id)? {
            Some((meta, idx)) => (meta, idx),
            None => return Ok(None),
        };
        if version > meta.version {
            return Ok(None);
        }

        let key = event_key_for(index, aggregate_type, aggregate_id, version);
        let value = match self
            .db
            .get(key)
            .map_err(|err| EventError::Storage(err.to_string()))?
        {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let stored: EventRecord = serde_json::from_slice(&value)?;
        let decoded = self.decode_record(stored)?;
        Ok(Some(decoded))
    }

    pub fn list_events_paginated(
        &self,
        aggregate_type: &str,
        aggregate_id: &str,
        skip: usize,
        take: Option<usize>,
    ) -> Result<Vec<EventRecord>> {
        match self.aggregate_version(aggregate_type, aggregate_id)? {
            Some(_) => {}
            None => return Err(EventError::AggregateNotFound),
        }

        if matches!(take, Some(0)) {
            return Ok(Vec::new());
        }

        let from_version = skip as u64;
        self.events_after(aggregate_type, aggregate_id, from_version, take)
    }

    pub fn events_after(
        &self,
        aggregate_type: &str,
        aggregate_id: &str,
        from_version: u64,
        limit: Option<usize>,
    ) -> Result<Vec<EventRecord>> {
        let start = Instant::now();
        let result = (|| {
            let (_, index) = self
                .load_meta_any(aggregate_type, aggregate_id)?
                .ok_or(EventError::AggregateNotFound)?;
            let start_key = event_key_for(index, aggregate_type, aggregate_id, from_version + 1);
            let prefix = event_prefix_for(index, aggregate_type, aggregate_id);
            let iter = self
                .db
                .iterator(IteratorMode::From(start_key.as_slice(), Direction::Forward));

            let mut events = Vec::new();
            for item in iter {
                let (key, value) = item.map_err(|err| EventError::Storage(err.to_string()))?;
                if !key.starts_with(prefix.as_slice()) {
                    break;
                }
                let record: EventRecord = serde_json::from_slice(&value)?;
                let record = self.decode_record(record)?;
                if record.version <= from_version {
                    continue;
                }
                events.push(record);
                if let Some(limit) = limit {
                    if events.len() >= limit {
                        break;
                    }
                }
            }

            Ok(events)
        })();
        let duration = start.elapsed().as_secs_f64();
        record_store_op(
            "rocksdb_iter_events",
            if result.is_ok() { "ok" } else { "err" },
            duration,
        );
        result
    }

    pub fn events_paginated(
        &self,
        scope: EventQueryScope<'_>,
        archive_scope: EventArchiveScope,
        skip: usize,
        take: Option<usize>,
        sort: Option<&[EventSort]>,
        filter: Option<&FilterExpr>,
    ) -> Result<Vec<EventRecord>> {
        match archive_scope {
            EventArchiveScope::ActiveOnly => Ok(self.collect_events_paginated(
                AggregateIndex::Active,
                scope,
                skip,
                take,
                sort,
                filter,
            )),
            EventArchiveScope::ArchivedOnly => Ok(self.collect_events_paginated(
                AggregateIndex::Archived,
                scope,
                skip,
                take,
                sort,
                filter,
            )),
            EventArchiveScope::IncludeArchived => {
                let mut active = self.collect_events_paginated(
                    AggregateIndex::Active,
                    scope,
                    0,
                    None,
                    None,
                    filter,
                );
                let mut archived = self.collect_events_paginated(
                    AggregateIndex::Archived,
                    scope,
                    0,
                    None,
                    None,
                    filter,
                );
                active.append(&mut archived);

                if let Some(keys) = sort {
                    active.sort_by(|a, b| compare_event_sort_keys(a, b, keys));
                }

                let total = active.len();
                let start = skip.min(total);
                let end = if let Some(limit) = take {
                    start.saturating_add(limit).min(total)
                } else {
                    total
                };
                Ok(active[start..end].to_vec())
            }
        }
    }

    pub fn find_event_by_id(&self, event_id: SnowflakeId) -> Result<Option<EventRecord>> {
        if let Some(event) = self.find_event_by_id_in_index(AggregateIndex::Active, event_id)? {
            return Ok(Some(event));
        }
        self.find_event_by_id_in_index(AggregateIndex::Archived, event_id)
    }

    pub fn aggregate_version(
        &self,
        aggregate_type: &str,
        aggregate_id: &str,
    ) -> Result<Option<u64>> {
        let meta = self.load_meta_any(aggregate_type, aggregate_id)?;
        Ok(meta.map(|(meta, _)| meta.version))
    }

    pub fn list_aggregate_ids(&self, aggregate_type: &str) -> Result<Vec<String>> {
        let start = Instant::now();
        let result = (|| {
            let mut prefix = key_with_segments(&[PREFIX_META, aggregate_type]);
            prefix.push(SEP);

            let iter = self
                .db
                .iterator(IteratorMode::From(prefix.as_slice(), Direction::Forward));

            let mut ids = Vec::new();
            for item in iter {
                let (key, _) = item.map_err(|err| EventError::Storage(err.to_string()))?;
                if !key.starts_with(prefix.as_slice()) {
                    break;
                }
                let remainder = &key[prefix.len()..];
                if remainder.is_empty() {
                    continue;
                }
                let id = std::str::from_utf8(remainder)
                    .map_err(|err| EventError::Storage(err.to_string()))?;
                ids.push(id.to_string());
            }
            Ok(ids)
        })();
        let duration = start.elapsed().as_secs_f64();
        record_store_op(
            "rocksdb_iter_meta_ids",
            if result.is_ok() { "ok" } else { "err" },
            duration,
        );
        result
    }

    pub fn verify(&self, aggregate_type: &str, aggregate_id: &str) -> Result<String> {
        let (meta, _) = self
            .load_meta_any(aggregate_type, aggregate_id)?
            .ok_or(EventError::AggregateNotFound)?;
        Ok(meta.merkle_root)
    }

    pub fn create_snapshot(
        &self,
        aggregate_type: &str,
        aggregate_id: &str,
        comment: Option<String>,
    ) -> Result<SnapshotRecord> {
        self.ensure_writable()?;
        let _guard = self.write_lock.lock();
        let state = self.get_aggregate_state(aggregate_type, aggregate_id)?;
        let created_at = Utc::now();
        let record = SnapshotRecord {
            snapshot_id: Some(self.next_snowflake_id()),
            aggregate_type: state.aggregate_type.clone(),
            aggregate_id: state.aggregate_id.clone(),
            version: state.version,
            state: state.state.clone(),
            merkle_root: state.merkle_root.clone(),
            created_at,
            comment,
        };

        let key = snapshot_key(aggregate_type, aggregate_id, created_at);
        self.db
            .put(key.clone(), serde_json::to_vec(&record)?)
            .map_err(|err| EventError::Storage(err.to_string()))?;
        if let Some(snapshot_id) = record.snapshot_id {
            let index_key = snapshot_index_key(snapshot_id);
            self.db
                .put(index_key, key)
                .map_err(|err| EventError::Storage(err.to_string()))?;
        }

        Ok(record)
    }

    pub fn list_snapshots(
        &self,
        aggregate_type: Option<&str>,
        aggregate_id: Option<&str>,
        version: Option<u64>,
    ) -> Result<Vec<SnapshotRecord>> {
        let start = Instant::now();
        let mut segments = vec![PREFIX_SNAPSHOT];
        if let Some(aggregate_type) = aggregate_type {
            segments.push(aggregate_type);
            if let Some(aggregate_id) = aggregate_id {
                segments.push(aggregate_id);
            }
        }
        let prefix = key_with_segments(&segments);
        let result = (|| {
            let mut snapshots = Vec::new();
            let iter = self
                .db
                .iterator(IteratorMode::From(prefix.as_slice(), Direction::Forward));

            for item in iter {
                let (key, value) = item.map_err(|err| EventError::Storage(err.to_string()))?;
                if !key.starts_with(prefix.as_slice()) {
                    break;
                }

                let snapshot: SnapshotRecord = serde_json::from_slice(&value).map_err(|err| {
                    EventError::Storage(format!("failed to deserialize snapshot: {err}"))
                })?;
                if version.map_or(true, |v| v == snapshot.version) {
                    snapshots.push(snapshot);
                }
            }
            Ok(snapshots)
        })();

        let duration = start.elapsed().as_secs_f64();
        record_store_op(
            "rocksdb_list_snapshots",
            if result.is_ok() { "ok" } else { "err" },
            duration,
        );
        result
    }

    pub fn find_snapshot_by_id(&self, snapshot_id: SnowflakeId) -> Result<Option<SnapshotRecord>> {
        let start = Instant::now();
        let result = (|| {
            // Fast path: use secondary index to jump to the snapshot key.
            let index_key = snapshot_index_key(snapshot_id);
            if let Some(snapshot_key) = self
                .db
                .get(index_key)
                .map_err(|err| EventError::Storage(err.to_string()))?
            {
                let Some(value) = self
                    .db
                    .get(snapshot_key)
                    .map_err(|err| EventError::Storage(err.to_string()))?
                else {
                    return Ok(None);
                };
                let snapshot: SnapshotRecord = serde_json::from_slice(&value).map_err(|err| {
                    EventError::Storage(format!("failed to deserialize snapshot: {err}"))
                })?;
                return Ok(Some(snapshot));
            }

            // Backward-compatibility: scan in case the index is missing for older snapshots.
            let prefix = key_with_segments(&[PREFIX_SNAPSHOT]);
            let iter = self
                .db
                .iterator(IteratorMode::From(prefix.as_slice(), Direction::Forward));
            for item in iter {
                let (key, value) = item.map_err(|err| EventError::Storage(err.to_string()))?;
                if !key.starts_with(prefix.as_slice()) {
                    break;
                }
                let snapshot: SnapshotRecord = serde_json::from_slice(&value).map_err(|err| {
                    EventError::Storage(format!("failed to deserialize snapshot: {err}"))
                })?;
                if snapshot.snapshot_id == Some(snapshot_id) {
                    return Ok(Some(snapshot));
                }
            }
            Ok(None)
        })();

        let duration = start.elapsed().as_secs_f64();
        record_store_op(
            "rocksdb_find_snapshot",
            if result.is_ok() { "ok" } else { "err" },
            duration,
        );
        result
    }

    pub fn set_archive(
        &self,
        aggregate_type: &str,
        aggregate_id: &str,
        archived: bool,
        comment: Option<String>,
    ) -> Result<AggregateState> {
        self.ensure_writable()?;
        let _guard = self.write_lock.lock();
        let (mut meta, current_index) = self
            .load_meta_any(aggregate_type, aggregate_id)?
            .ok_or(EventError::AggregateNotFound)?;
        let previous_meta = meta.clone();
        let state_map = self.load_state_map_from(current_index, aggregate_type, aggregate_id)?;

        let changed_at = Utc::now();
        meta.archived = archived;
        if archived {
            meta.archived_at = Some(changed_at);
            meta.archive_comment = comment;
        } else {
            meta.archived_at = None;
            meta.archive_comment = None;
        }
        meta.updated_at = Some(changed_at);

        let destination_index = if archived {
            AggregateIndex::Archived
        } else {
            AggregateIndex::Active
        };

        let encoded_meta = serde_json::to_vec(&meta)?;
        let encoded_state = self.encode_state_map(&state_map)?;

        let mut batch = WriteBatch::default();
        if current_index != destination_index {
            self.move_event_stream(
                aggregate_type,
                aggregate_id,
                current_index,
                destination_index,
                &mut batch,
            )?;
        }
        match (current_index, destination_index) {
            (AggregateIndex::Active, AggregateIndex::Archived) => {
                self.sync_timestamp_indexes(
                    &mut batch,
                    AggregateIndex::Active,
                    Some(&previous_meta),
                    None,
                );
                self.sync_timestamp_indexes(
                    &mut batch,
                    AggregateIndex::Archived,
                    None,
                    Some(&meta),
                );
                batch.delete(meta_key(aggregate_type, aggregate_id));
                batch.delete(state_key(aggregate_type, aggregate_id));
                batch_put(
                    &mut batch,
                    meta_archived_key(aggregate_type, aggregate_id),
                    encoded_meta.clone(),
                )?;
                batch_put(
                    &mut batch,
                    state_archived_key(aggregate_type, aggregate_id),
                    encoded_state.clone(),
                )?;
            }
            (AggregateIndex::Archived, AggregateIndex::Active) => {
                self.sync_timestamp_indexes(
                    &mut batch,
                    AggregateIndex::Archived,
                    Some(&previous_meta),
                    None,
                );
                self.sync_timestamp_indexes(&mut batch, AggregateIndex::Active, None, Some(&meta));
                batch.delete(meta_archived_key(aggregate_type, aggregate_id));
                batch.delete(state_archived_key(aggregate_type, aggregate_id));
                batch_put(
                    &mut batch,
                    meta_key(aggregate_type, aggregate_id),
                    encoded_meta.clone(),
                )?;
                batch_put(
                    &mut batch,
                    state_key(aggregate_type, aggregate_id),
                    encoded_state.clone(),
                )?;
            }
            (AggregateIndex::Active, AggregateIndex::Active) => {
                self.sync_timestamp_indexes(
                    &mut batch,
                    AggregateIndex::Active,
                    Some(&previous_meta),
                    Some(&meta),
                );
                batch_put(
                    &mut batch,
                    meta_key(aggregate_type, aggregate_id),
                    encoded_meta.clone(),
                )?;
                batch_put(
                    &mut batch,
                    state_key(aggregate_type, aggregate_id),
                    encoded_state.clone(),
                )?;
            }
            (AggregateIndex::Archived, AggregateIndex::Archived) => {
                self.sync_timestamp_indexes(
                    &mut batch,
                    AggregateIndex::Archived,
                    Some(&previous_meta),
                    Some(&meta),
                );
                batch_put(
                    &mut batch,
                    meta_archived_key(aggregate_type, aggregate_id),
                    encoded_meta.clone(),
                )?;
                batch_put(
                    &mut batch,
                    state_archived_key(aggregate_type, aggregate_id),
                    encoded_state.clone(),
                )?;
            }
        }

        self.write_batch(batch)?;

        self.get_aggregate_state(aggregate_type, aggregate_id)
    }

    fn move_event_stream(
        &self,
        aggregate_type: &str,
        aggregate_id: &str,
        from: AggregateIndex,
        to: AggregateIndex,
        batch: &mut WriteBatch,
    ) -> Result<()> {
        if from == to {
            return Ok(());
        }

        let prefix = event_prefix_for(from, aggregate_type, aggregate_id);
        let mut iter = self
            .db
            .iterator(IteratorMode::From(prefix.as_slice(), Direction::Forward));
        let mut records = Vec::new();

        while let Some(item) = iter.next() {
            let (key, value) = item.map_err(|err| EventError::Storage(err.to_string()))?;
            if !key.starts_with(prefix.as_slice()) {
                break;
            }
            let version = parse_event_version(&key)?;
            records.push((key, value.to_vec(), version));
        }

        for (old_key, value, version) in records {
            batch.delete(old_key);
            batch.put(
                event_key_for(to, aggregate_type, aggregate_id, version),
                value,
            );
        }

        Ok(())
    }

    fn encode_record(&self, record: &EventRecord) -> Result<EventRecord> {
        if let Some(enc) = &self.encryptor {
            if encryption::extract_encrypted_value(&record.payload).is_some() {
                return Ok(record.clone());
            }
            let payload_bytes = serde_json::to_vec(&record.payload)?;
            let ciphertext = enc.encrypt_to_string(&payload_bytes)?;
            let mut stored = record.clone();
            stored.payload = encryption::wrap_encrypted_value(ciphertext);
            Ok(stored)
        } else {
            Ok(record.clone())
        }
    }

    fn decode_record(&self, mut record: EventRecord) -> Result<EventRecord> {
        if let Some(enc) = &self.encryptor {
            if let Some(ciphertext) = encryption::extract_encrypted_value(&record.payload) {
                let bytes = enc.decrypt_from_str(ciphertext)?;
                record.payload = serde_json::from_slice(&bytes)?;
            }
            Ok(record)
        } else {
            if encryption::extract_encrypted_value(&record.payload).is_some() {
                return Err(EventError::Config(
                    "data encryption key must be configured to read encrypted events".to_string(),
                ));
            }
            Ok(record)
        }
    }

    fn encode_state_map(&self, state: &BTreeMap<String, String>) -> Result<Vec<u8>> {
        let json = serde_json::to_vec(state)?;
        if let Some(enc) = &self.encryptor {
            let ciphertext = enc.encrypt_to_string(&json)?;
            Ok(ciphertext.into_bytes())
        } else {
            Ok(json)
        }
    }

    fn decode_state_map(&self, bytes: &[u8]) -> Result<BTreeMap<String, String>> {
        if let Some(enc) = &self.encryptor {
            let text = str::from_utf8(bytes).map_err(|err| EventError::Storage(err.to_string()))?;
            if encryption::is_encrypted_blob(text.trim()) {
                let decrypted = enc.decrypt_from_str(text.trim())?;
                Ok(serde_json::from_slice(&decrypted)?)
            } else if text.trim().is_empty() {
                Ok(BTreeMap::new())
            } else {
                Ok(serde_json::from_str(text)?)
            }
        } else {
            if let Ok(text) = str::from_utf8(bytes) {
                if encryption::is_encrypted_blob(text.trim()) {
                    return Err(EventError::Config(
                        "data encryption key must be configured to read encrypted aggregates"
                            .to_string(),
                    ));
                }
            }
            Ok(serde_json::from_slice(bytes)?)
        }
    }

    fn ensure_writable(&self) -> Result<()> {
        if self.read_only {
            Err(EventError::Storage(
                "event store opened in read-only mode".into(),
            ))
        } else {
            Ok(())
        }
    }

    fn write_batch(&self, batch: WriteBatch) -> Result<()> {
        let start = Instant::now();
        let result = self
            .db
            .write(batch)
            .map_err(|err| EventError::Storage(err.to_string()));
        let duration = start.elapsed().as_secs_f64();
        record_store_op(
            "rocksdb_write",
            if result.is_ok() { "ok" } else { "err" },
            duration,
        );
        result
    }

    fn sync_timestamp_indexes(
        &self,
        batch: &mut WriteBatch,
        index: AggregateIndex,
        previous: Option<&AggregateMeta>,
        current: Option<&AggregateMeta>,
    ) {
        self.sync_single_timestamp_index(
            batch,
            TimestampIndexKind::CreatedAt,
            index,
            previous,
            current,
        );
        self.sync_single_timestamp_index(
            batch,
            TimestampIndexKind::UpdatedAt,
            index,
            previous,
            current,
        );
    }

    fn sync_single_timestamp_index(
        &self,
        batch: &mut WriteBatch,
        kind: TimestampIndexKind,
        index: AggregateIndex,
        previous: Option<&AggregateMeta>,
        current: Option<&AggregateMeta>,
    ) {
        let meta_identity = current.or(previous);
        let Some(meta) = meta_identity else {
            return;
        };
        let prev_ts = previous.and_then(|meta| meta_timestamp_ms(meta, kind));
        let curr_ts = current.and_then(|meta| meta_timestamp_ms(meta, kind));
        if prev_ts == curr_ts {
            return;
        }
        if let Some(prev) = prev_ts {
            batch.delete(timestamp_index_key(
                kind,
                index,
                prev,
                &meta.aggregate_type,
                &meta.aggregate_id,
            ));
        }
        if let Some(curr) = curr_ts {
            batch.put(
                timestamp_index_key(kind, index, curr, &meta.aggregate_type, &meta.aggregate_id),
                Vec::new(),
            );
        }
    }

    pub fn aggregates(&self) -> Vec<AggregateState> {
        self.aggregates_paginated(0, None)
    }

    pub fn aggregates_paginated(&self, skip: usize, take: Option<usize>) -> Vec<AggregateState> {
        self.aggregates_paginated_with_transform(
            skip,
            take,
            None,
            AggregateQueryScope::ActiveOnly,
            None,
            |aggregate| Some(aggregate),
        )
    }

    pub fn aggregates_paginated_with_transform<F>(
        &self,
        skip: usize,
        take: Option<usize>,
        sort: Option<&[AggregateSort]>,
        scope: AggregateQueryScope,
        cursor: Option<&AggregateCursor>,
        mut transform: F,
    ) -> Vec<AggregateState>
    where
        F: FnMut(AggregateState) -> Option<AggregateState>,
    {
        if let Some(keys) = sort {
            if let Some((kind, descending)) = timestamp_sort_hint(keys) {
                return self.collect_timestamp_sorted(
                    kind,
                    scope,
                    skip,
                    take,
                    descending,
                    cursor,
                    &mut transform,
                );
            }
        }
        match scope {
            AggregateQueryScope::ActiveOnly => self.collect_index_paginated(
                AggregateIndex::Active,
                skip,
                take,
                sort,
                &mut transform,
            ),
            AggregateQueryScope::ArchivedOnly => self.collect_index_paginated(
                AggregateIndex::Archived,
                skip,
                take,
                sort,
                &mut transform,
            ),
            AggregateQueryScope::IncludeArchived => {
                let mut items = self.collect_index_paginated(
                    AggregateIndex::Active,
                    0,
                    None,
                    None,
                    &mut transform,
                );
                let mut archived_items = self.collect_index_paginated(
                    AggregateIndex::Archived,
                    0,
                    None,
                    None,
                    &mut transform,
                );
                items.append(&mut archived_items);

                if let Some(keys) = sort {
                    items.sort_by(|a, b| compare_aggregate_sort_keys(a, b, keys));
                }

                let total = items.len();
                let start = skip.min(total);
                let end = if let Some(limit) = take {
                    start.saturating_add(limit).min(total)
                } else {
                    total
                };
                items[start..end].to_vec()
            }
        }
    }

    fn collect_timestamp_sorted<F>(
        &self,
        kind: TimestampIndexKind,
        scope: AggregateQueryScope,
        skip: usize,
        take: Option<usize>,
        descending: bool,
        cursor: Option<&AggregateCursor>,
        transform: &mut F,
    ) -> Vec<AggregateState>
    where
        F: FnMut(AggregateState) -> Option<AggregateState>,
    {
        match scope {
            AggregateQueryScope::ActiveOnly => self.collect_timestamp_index_paginated(
                kind,
                AggregateIndex::Active,
                skip,
                take,
                descending,
                cursor,
                transform,
            ),
            AggregateQueryScope::ArchivedOnly => self.collect_timestamp_index_paginated(
                kind,
                AggregateIndex::Archived,
                skip,
                take,
                descending,
                cursor,
                transform,
            ),
            AggregateQueryScope::IncludeArchived => self
                .collect_timestamp_indexes_merged(kind, skip, take, descending, cursor, transform),
        }
    }

    fn collect_timestamp_index_paginated<F>(
        &self,
        kind: TimestampIndexKind,
        index: AggregateIndex,
        skip: usize,
        take: Option<usize>,
        descending: bool,
        cursor: Option<&AggregateCursor>,
        transform: &mut F,
    ) -> Vec<AggregateState>
    where
        F: FnMut(AggregateState) -> Option<AggregateState>,
    {
        let mut stream = TimestampIndexIterator::new(self, kind, index, descending);
        let mut items = Vec::new();
        let mut skipped = 0usize;
        let cursor_key = cursor.and_then(|token| {
            token.as_timestamp_key().and_then(
                |(cursor_kind, cursor_desc, cursor_ts, cursor_index, agg_type, agg_id)| {
                    if cursor_kind == kind
                        && cursor_desc == descending
                        && cursor_index == CursorIndex::from(index)
                    {
                        Some((cursor_ts, agg_type.to_owned(), agg_id.to_owned()))
                    } else {
                        None
                    }
                },
            )
        });
        let mut resume = cursor_key;

        while let Some(entry) = stream.next_entry() {
            let aggregate = match self.load_aggregate_from_index(
                index,
                &entry.aggregate_type,
                &entry.aggregate_id,
            ) {
                Ok(Some(aggregate)) => aggregate,
                Ok(None) => continue,
                Err(_) => {
                    stream.mark_error();
                    continue;
                }
            };

            let mut consume_cursor = false;
            if let Some((cursor_ts, cursor_type, cursor_id)) = resume.as_ref() {
                let ordering = compare_timestamp_values(
                    entry.timestamp_ms,
                    &entry.aggregate_type,
                    &entry.aggregate_id,
                    *cursor_ts,
                    cursor_type,
                    cursor_id,
                );
                let skip_entry = if descending {
                    ordering != StdOrdering::Less
                } else {
                    ordering != StdOrdering::Greater
                };
                if skip_entry {
                    continue;
                }
                consume_cursor = true;
            }
            if consume_cursor {
                resume = None;
            }

            let Some(aggregate) = transform(aggregate) else {
                continue;
            };

            if skipped < skip {
                skipped += 1;
                continue;
            }
            items.push(aggregate);
            if let Some(limit) = take {
                if items.len() >= limit {
                    break;
                }
            }
        }

        items
    }

    fn collect_timestamp_indexes_merged<F>(
        &self,
        kind: TimestampIndexKind,
        skip: usize,
        take: Option<usize>,
        descending: bool,
        cursor: Option<&AggregateCursor>,
        transform: &mut F,
    ) -> Vec<AggregateState>
    where
        F: FnMut(AggregateState) -> Option<AggregateState>,
    {
        let mut active_stream =
            TimestampIndexIterator::new(self, kind, AggregateIndex::Active, descending);
        let mut archived_stream =
            TimestampIndexIterator::new(self, kind, AggregateIndex::Archived, descending);
        let mut next_active = active_stream.next_entry();
        let mut next_archived = archived_stream.next_entry();
        let mut skipped = 0usize;
        let mut items = Vec::new();
        let cursor_key = cursor.and_then(|token| {
            token.as_timestamp_key().and_then(
                |(cursor_kind, cursor_desc, cursor_ts, _cursor_index, agg_type, agg_id)| {
                    if cursor_kind == kind && cursor_desc == descending {
                        Some((cursor_ts, agg_type.to_owned(), agg_id.to_owned()))
                    } else {
                        None
                    }
                },
            )
        });
        let mut resume = cursor_key;

        loop {
            let (source_index, entry) = match (&next_active, &next_archived) {
                (None, None) => break,
                (Some(_), None) => (AggregateIndex::Active, next_active.take()),
                (None, Some(_)) => (AggregateIndex::Archived, next_archived.take()),
                (Some(active), Some(archived)) => {
                    if compare_timestamp_entries(active, archived, descending)
                        != StdOrdering::Greater
                    {
                        (AggregateIndex::Active, next_active.take())
                    } else {
                        (AggregateIndex::Archived, next_archived.take())
                    }
                }
            };

            let Some(entry) = entry else {
                break;
            };

            let aggregate = match self.load_aggregate_from_index(
                source_index,
                &entry.aggregate_type,
                &entry.aggregate_id,
            ) {
                Ok(Some(aggregate)) => aggregate,
                Ok(None) => {
                    match source_index {
                        AggregateIndex::Active => next_active = active_stream.next_entry(),
                        AggregateIndex::Archived => next_archived = archived_stream.next_entry(),
                    }
                    continue;
                }
                Err(_) => {
                    match source_index {
                        AggregateIndex::Active => active_stream.mark_error(),
                        AggregateIndex::Archived => archived_stream.mark_error(),
                    }
                    match source_index {
                        AggregateIndex::Active => next_active = active_stream.next_entry(),
                        AggregateIndex::Archived => next_archived = archived_stream.next_entry(),
                    }
                    continue;
                }
            };

            let mut consume_cursor = false;
            if let Some((cursor_ts, cursor_type, cursor_id)) = resume.as_ref() {
                let ordering = compare_timestamp_values(
                    entry.timestamp_ms,
                    &entry.aggregate_type,
                    &entry.aggregate_id,
                    *cursor_ts,
                    cursor_type,
                    cursor_id,
                );
                let skip_entry = if descending {
                    ordering != StdOrdering::Less
                } else {
                    ordering != StdOrdering::Greater
                };
                if skip_entry {
                    match source_index {
                        AggregateIndex::Active => next_active = active_stream.next_entry(),
                        AggregateIndex::Archived => next_archived = archived_stream.next_entry(),
                    }
                    continue;
                }
                consume_cursor = true;
            }
            if consume_cursor {
                resume = None;
            }

            let Some(aggregate) = transform(aggregate) else {
                match source_index {
                    AggregateIndex::Active => next_active = active_stream.next_entry(),
                    AggregateIndex::Archived => next_archived = archived_stream.next_entry(),
                }
                continue;
            };

            if skipped < skip {
                skipped += 1;
            } else {
                items.push(aggregate);
            }

            if let Some(limit) = take {
                if items.len() >= limit {
                    break;
                }
            }

            match source_index {
                AggregateIndex::Active => next_active = active_stream.next_entry(),
                AggregateIndex::Archived => next_archived = archived_stream.next_entry(),
            }
        }

        items
    }

    fn load_aggregate_from_index(
        &self,
        index: AggregateIndex,
        aggregate_type: &str,
        aggregate_id: &str,
    ) -> Result<Option<AggregateState>> {
        let meta = match self.load_meta_from(index, aggregate_type, aggregate_id)? {
            Some(meta) => meta,
            None => return Ok(None),
        };
        let AggregateMeta {
            aggregate_type,
            aggregate_id,
            version,
            merkle_root,
            archived,
            created_at,
            updated_at,
            ..
        } = meta;
        let state = self.load_state_map_from(index, &aggregate_type, &aggregate_id)?;
        Ok(Some(AggregateState {
            aggregate_type,
            aggregate_id,
            version,
            state,
            merkle_root,
            created_at,
            updated_at,
            archived,
        }))
    }

    fn collect_index_paginated<F>(
        &self,
        index: AggregateIndex,
        skip: usize,
        take: Option<usize>,
        sort: Option<&[AggregateSort]>,
        transform: &mut F,
    ) -> Vec<AggregateState>
    where
        F: FnMut(AggregateState) -> Option<AggregateState>,
    {
        let start = Instant::now();
        let prefix = meta_prefix_for(index);
        let iter = self
            .db
            .iterator(IteratorMode::From(prefix.as_slice(), Direction::Forward));
        let mut items = Vec::new();
        let mut skipped = 0usize;
        let should_sort = sort.map_or(false, |keys| !aggregate_sort_matches_key_order(keys));
        let mut status = "ok";
        let mut pending = if should_sort { Some(Vec::new()) } else { None };

        for item in iter {
            let Ok((key, value)) = item else {
                continue;
            };
            if !key.starts_with(prefix.as_slice()) {
                break;
            }
            if key.len() > prefix.len() && key[prefix.len()] != SEP {
                break;
            }

            let meta: AggregateMeta = match serde_json::from_slice(&value) {
                Ok(meta) => meta,
                Err(_) => continue,
            };

            let AggregateMeta {
                aggregate_type,
                aggregate_id,
                version,
                merkle_root,
                archived,
                created_at,
                updated_at,
                ..
            } = meta;
            if should_sort {
                if let Some(list) = pending.as_mut() {
                    list.push(AggregateSummary {
                        aggregate_type,
                        aggregate_id,
                        version,
                        merkle_root,
                        archived,
                        created_at,
                        updated_at,
                    });
                }
                continue;
            }
            let state = match self.load_state_map_from(index, &aggregate_type, &aggregate_id) {
                Ok(state) => state,
                Err(_) => {
                    status = "err";
                    continue;
                }
            };

            let aggregate = AggregateState {
                aggregate_type,
                aggregate_id,
                version,
                state,
                merkle_root,
                created_at,
                updated_at,
                archived,
            };

            let Some(aggregate) = transform(aggregate) else {
                continue;
            };

            if skipped < skip && !should_sort {
                skipped += 1;
                continue;
            }

            if should_sort {
                items.push(aggregate);
                continue;
            }

            items.push(aggregate);

            if let Some(limit) = take {
                if items.len() >= limit {
                    break;
                }
            }
        }

        if should_sort {
            let summaries = pending.unwrap_or_default();
            let (mut aggregates, had_errors) = self.load_states_parallel(index, summaries);
            if had_errors {
                status = "err";
            }
            for aggregate in aggregates.drain(..) {
                let Some(aggregate) = transform(aggregate) else {
                    continue;
                };
                items.push(aggregate);
            }
            if let Some(keys) = sort {
                items.sort_by(|a, b| compare_aggregate_sort_keys(a, b, keys));
            }

            let total = items.len();
            let start = skip.min(total);
            let end = if let Some(limit) = take {
                start.saturating_add(limit).min(total)
            } else {
                total
            };
            items = items[start..end].to_vec();
        }

        let duration = start.elapsed().as_secs_f64();
        let metric = match index {
            AggregateIndex::Active => "rocksdb_iter_meta",
            AggregateIndex::Archived => "rocksdb_iter_meta_archived",
        };
        record_store_op(metric, status, duration);
        items
    }

    pub fn aggregates_page_with_transform<F>(
        &self,
        cursor: Option<&AggregateCursor>,
        take: usize,
        scope: AggregateQueryScope,
        mut transform: F,
    ) -> Result<(Vec<AggregateState>, Option<AggregateCursor>)>
    where
        F: FnMut(AggregateState) -> Option<AggregateState>,
    {
        if take == 0 {
            return Ok((Vec::new(), None));
        }

        match scope {
            AggregateQueryScope::ActiveOnly => {
                self.collect_index_page(AggregateIndex::Active, cursor, take, &mut transform)
            }
            AggregateQueryScope::ArchivedOnly => {
                if let Some(cursor) = cursor {
                    if cursor.index() != CursorIndex::Archived {
                        return Err(EventError::InvalidCursor(
                            "archived listings require archived cursor values".into(),
                        ));
                    }
                }
                self.collect_index_page(AggregateIndex::Archived, cursor, take, &mut transform)
            }
            AggregateQueryScope::IncludeArchived => {
                let mut remaining = take;
                let mut items = Vec::new();

                if let Some(cursor) = cursor {
                    if cursor.index() == CursorIndex::Archived {
                        let (mut archived, next) = self.collect_index_page(
                            AggregateIndex::Archived,
                            Some(cursor),
                            remaining,
                            &mut transform,
                        )?;
                        items.append(&mut archived);
                        return Ok((items, next));
                    }
                }

                let (mut active_items, active_cursor) = self.collect_index_page(
                    AggregateIndex::Active,
                    cursor,
                    remaining,
                    &mut transform,
                )?;
                remaining = remaining.saturating_sub(active_items.len());
                items.append(&mut active_items);
                if let Some(next) = active_cursor {
                    return Ok((items, Some(next)));
                }
                if remaining == 0 {
                    return Ok((items, None));
                }
                let (mut archived_items, archived_cursor) = self.collect_index_page(
                    AggregateIndex::Archived,
                    None,
                    remaining,
                    &mut transform,
                )?;
                items.append(&mut archived_items);
                Ok((items, archived_cursor))
            }
        }
    }

    pub fn aggregates_page(
        &self,
        cursor: Option<&AggregateCursor>,
        take: usize,
        scope: AggregateQueryScope,
    ) -> Result<(Vec<AggregateState>, Option<AggregateCursor>)> {
        self.aggregates_page_with_transform(cursor, take, scope, |aggregate| Some(aggregate))
    }

    fn collect_index_page<F>(
        &self,
        index: AggregateIndex,
        cursor: Option<&AggregateCursor>,
        take: usize,
        transform: &mut F,
    ) -> Result<(Vec<AggregateState>, Option<AggregateCursor>)>
    where
        F: FnMut(AggregateState) -> Option<AggregateState>,
    {
        let start = Instant::now();
        let prefix = meta_prefix_for(index);
        let mut start_key = prefix.clone();
        let mut skip_current = false;
        if let Some(position) = cursor {
            if position.index() != CursorIndex::from(index) {
                return Err(EventError::InvalidCursor(format!(
                    "cursor points to {} aggregates but scope uses {} records",
                    position.index(),
                    CursorIndex::from(index)
                )));
            }
            start_key = meta_key_for(index, position.aggregate_type(), position.aggregate_id());
            skip_current = true;
        }
        let iter = self
            .db
            .iterator(IteratorMode::From(start_key.as_slice(), Direction::Forward));
        let mut items = Vec::new();
        let mut status = "ok";
        let mut last_cursor = None;

        for item in iter {
            let Ok((key, value)) = item else {
                status = "err";
                continue;
            };
            if !key.starts_with(prefix.as_slice()) {
                break;
            }
            if key.len() > prefix.len() && key[prefix.len()] != SEP {
                break;
            }
            if skip_current && key.as_ref() == start_key.as_slice() {
                skip_current = false;
                continue;
            } else if skip_current {
                skip_current = false;
            }

            let meta: AggregateMeta = match serde_json::from_slice(&value) {
                Ok(meta) => meta,
                Err(_) => {
                    status = "err";
                    continue;
                }
            };

            let AggregateMeta {
                aggregate_type,
                aggregate_id,
                version,
                merkle_root,
                archived,
                created_at,
                updated_at,
                ..
            } = meta;
            let state = match self.load_state_map_from(index, &aggregate_type, &aggregate_id) {
                Ok(state) => state,
                Err(_) => {
                    status = "err";
                    continue;
                }
            };

            let aggregate = AggregateState {
                aggregate_type,
                aggregate_id,
                version,
                state,
                merkle_root,
                created_at,
                updated_at,
                archived,
            };

            let Some(aggregate) = transform(aggregate) else {
                continue;
            };

            last_cursor = Some(AggregateCursor::new(
                CursorIndex::from(index),
                aggregate.aggregate_type.clone(),
                aggregate.aggregate_id.clone(),
            ));
            items.push(aggregate);

            if items.len() >= take {
                break;
            }
        }

        let duration = start.elapsed().as_secs_f64();
        let metric = match index {
            AggregateIndex::Active => "rocksdb_iter_meta",
            AggregateIndex::Archived => "rocksdb_iter_meta_archived",
        };
        record_store_op(metric, status, duration);

        let next_cursor = if items.len() >= take {
            last_cursor
        } else {
            None
        };

        Ok((items, next_cursor))
    }

    fn load_states_parallel(
        &self,
        index: AggregateIndex,
        summaries: Vec<AggregateSummary>,
    ) -> (Vec<AggregateState>, bool) {
        use rayon::prelude::*;

        let results: Vec<std::result::Result<AggregateState, EventError>> = summaries
            .into_par_iter()
            .map(|summary| {
                let AggregateSummary {
                    aggregate_type,
                    aggregate_id,
                    version,
                    merkle_root,
                    archived,
                    created_at,
                    updated_at,
                } = summary;
                let state = self.load_state_map_from(index, &aggregate_type, &aggregate_id)?;
                Ok(AggregateState {
                    aggregate_type,
                    aggregate_id,
                    version,
                    state,
                    merkle_root,
                    created_at,
                    updated_at,
                    archived,
                })
            })
            .collect();

        let mut had_error = false;
        let aggregates = results
            .into_iter()
            .filter_map(|result| match result {
                Ok(aggregate) => Some(aggregate),
                Err(_) => {
                    had_error = true;
                    None
                }
            })
            .collect();

        (aggregates, had_error)
    }

    fn collect_events_paginated(
        &self,
        index: AggregateIndex,
        scope: EventQueryScope<'_>,
        skip: usize,
        take: Option<usize>,
        sort: Option<&[EventSort]>,
        filter: Option<&FilterExpr>,
    ) -> Vec<EventRecord> {
        let start = Instant::now();
        let prefix = event_prefix_for_scope(index, scope);
        let iter = self
            .db
            .iterator(IteratorMode::From(prefix.as_slice(), Direction::Forward));
        let mut items = Vec::new();
        let mut skipped = 0usize;
        let should_sort = sort.map_or(false, |keys| !event_sort_matches_key_order(keys));
        let mut status = "ok";

        for item in iter {
            let Ok((key, value)) = item else {
                status = "err";
                continue;
            };
            if !key.starts_with(prefix.as_slice()) {
                break;
            }
            if key.len() > prefix.len() && key[prefix.len()] != SEP {
                break;
            }

            let raw: EventRecord = match serde_json::from_slice(&value) {
                Ok(record) => record,
                Err(_) => {
                    status = "err";
                    continue;
                }
            };
            let record = match self.decode_record(raw) {
                Ok(record) => record,
                Err(_) => {
                    status = "err";
                    continue;
                }
            };

            if let Some(expr) = filter {
                if !expr.matches_event(&record) {
                    continue;
                }
            }

            if skipped < skip && !should_sort {
                skipped += 1;
                continue;
            }

            if should_sort {
                items.push(record);
                continue;
            }

            items.push(record);

            if let Some(limit) = take {
                if items.len() >= limit {
                    break;
                }
            }
        }

        if should_sort {
            if let Some(keys) = sort {
                items.sort_by(|a, b| compare_event_sort_keys(a, b, keys));
            }

            let total = items.len();
            let start = skip.min(total);
            let end = if let Some(limit) = take {
                start.saturating_add(limit).min(total)
            } else {
                total
            };
            items = items[start..end].to_vec();
        }

        let duration = start.elapsed().as_secs_f64();
        let metric = match index {
            AggregateIndex::Active => "rocksdb_iter_events",
            AggregateIndex::Archived => "rocksdb_iter_events_archived",
        };
        record_store_op(metric, status, duration);
        items
    }

    pub fn events_page(
        &self,
        scope: EventQueryScope<'_>,
        archive_scope: EventArchiveScope,
        cursor: Option<&EventCursor>,
        take: usize,
        filter: Option<&FilterExpr>,
    ) -> Result<(Vec<EventRecord>, Option<EventCursor>)> {
        if take == 0 {
            return Ok((Vec::new(), None));
        }

        match archive_scope {
            EventArchiveScope::ActiveOnly => {
                self.collect_events_page(AggregateIndex::Active, scope, cursor, take, filter)
            }
            EventArchiveScope::ArchivedOnly => {
                if let Some(cursor) = cursor {
                    if cursor.index() != CursorIndex::Archived {
                        return Err(EventError::InvalidCursor(
                            "archived listings require archived cursor values".into(),
                        ));
                    }
                }
                self.collect_events_page(AggregateIndex::Archived, scope, cursor, take, filter)
            }
            EventArchiveScope::IncludeArchived => {
                if let Some(cursor) = cursor {
                    if cursor.index() == CursorIndex::Archived {
                        return self.collect_events_page(
                            AggregateIndex::Archived,
                            scope,
                            Some(cursor),
                            take,
                            filter,
                        );
                    }
                }

                let (mut active, next_active) =
                    self.collect_events_page(AggregateIndex::Active, scope, cursor, take, filter)?;
                if let Some(cursor) = next_active {
                    return Ok((active, Some(cursor)));
                }
                if active.len() >= take {
                    return Ok((active, None));
                }
                let remaining = take.saturating_sub(active.len());
                let (mut archived, next_archived) = self.collect_events_page(
                    AggregateIndex::Archived,
                    scope,
                    None,
                    remaining,
                    filter,
                )?;
                active.append(&mut archived);
                Ok((active, next_archived))
            }
        }
    }

    fn collect_events_page(
        &self,
        index: AggregateIndex,
        scope: EventQueryScope<'_>,
        cursor: Option<&EventCursor>,
        take: usize,
        filter: Option<&FilterExpr>,
    ) -> Result<(Vec<EventRecord>, Option<EventCursor>)> {
        let start = Instant::now();
        let prefix = event_prefix_for_scope(index, scope);
        let mut start_key = prefix.clone();
        let mut skip_current = false;
        if let Some(token) = cursor {
            if token.index() != CursorIndex::from(index) {
                return Err(EventError::InvalidCursor(format!(
                    "cursor points to {} events but scope uses {} records",
                    token.index(),
                    CursorIndex::from(index)
                )));
            }
            if !event_cursor_matches_scope(token, scope) {
                return Err(EventError::InvalidCursor(
                    "cursor does not match the requested event scope".into(),
                ));
            }
            start_key = event_key_for(
                index,
                token.aggregate_type(),
                token.aggregate_id(),
                token.version(),
            );
            skip_current = true;
        }
        let iter = self
            .db
            .iterator(IteratorMode::From(start_key.as_slice(), Direction::Forward));
        let mut items = Vec::new();
        let mut status = "ok";
        let mut last_cursor = None;

        for item in iter {
            let Ok((key, value)) = item else {
                status = "err";
                continue;
            };
            if !key.starts_with(prefix.as_slice()) {
                break;
            }
            if skip_current && key.as_ref() == start_key.as_slice() {
                skip_current = false;
                continue;
            } else if skip_current {
                skip_current = false;
            }

            let raw: EventRecord = match serde_json::from_slice(&value) {
                Ok(record) => record,
                Err(_) => {
                    status = "err";
                    continue;
                }
            };
            let record = match self.decode_record(raw) {
                Ok(record) => record,
                Err(_) => {
                    status = "err";
                    continue;
                }
            };

            if let Some(expr) = filter {
                if !expr.matches_event(&record) {
                    continue;
                }
            }

            last_cursor = Some(EventCursor::new(
                CursorIndex::from(index),
                record.aggregate_type.clone(),
                record.aggregate_id.clone(),
                record.version,
            ));
            items.push(record);

            if items.len() >= take {
                break;
            }
        }

        let duration = start.elapsed().as_secs_f64();
        let metric = match index {
            AggregateIndex::Active => "rocksdb_iter_events",
            AggregateIndex::Archived => "rocksdb_iter_events_archived",
        };
        record_store_op(metric, status, duration);

        let next_cursor = if items.len() >= take {
            last_cursor
        } else {
            None
        };

        Ok((items, next_cursor))
    }

    fn find_event_by_id_in_index(
        &self,
        index: AggregateIndex,
        event_id: SnowflakeId,
    ) -> Result<Option<EventRecord>> {
        let start = Instant::now();
        let prefix = key_with_segments(&[index.event_prefix()]);
        let iter = self
            .db
            .iterator(IteratorMode::From(prefix.as_slice(), Direction::Forward));
        let mut status = "ok";

        for item in iter {
            let Ok((key, value)) = item else {
                status = "err";
                continue;
            };
            if !key.starts_with(prefix.as_slice()) {
                break;
            }
            if key.len() > prefix.len() && key[prefix.len()] != SEP {
                break;
            }

            let raw: EventRecord = match serde_json::from_slice(&value) {
                Ok(record) => record,
                Err(_) => {
                    status = "err";
                    continue;
                }
            };
            let record = match self.decode_record(raw) {
                Ok(record) => record,
                Err(_) => {
                    status = "err";
                    continue;
                }
            };

            if record.metadata.event_id == event_id {
                let duration = start.elapsed().as_secs_f64();
                let metric = match index {
                    AggregateIndex::Active => "rocksdb_scan_event_id",
                    AggregateIndex::Archived => "rocksdb_scan_event_id_archived",
                };
                record_store_op(metric, status, duration);
                return Ok(Some(record));
            }
        }

        let duration = start.elapsed().as_secs_f64();
        let metric = match index {
            AggregateIndex::Active => "rocksdb_scan_event_id",
            AggregateIndex::Archived => "rocksdb_scan_event_id_archived",
        };
        record_store_op(metric, status, duration);
        Ok(None)
    }

    pub fn aggregate_positions(&self) -> Result<Vec<AggregatePositionEntry>> {
        let start = Instant::now();
        let result = (|| {
            let prefix = meta_prefix();
            let iter = self
                .db
                .iterator(IteratorMode::From(prefix.as_slice(), Direction::Forward));
            let mut items = Vec::new();

            for item in iter {
                let (key, value) = item.map_err(|err| EventError::Storage(err.to_string()))?;
                if !key.starts_with(prefix.as_slice()) {
                    break;
                }
                if key.len() > prefix.len() && key[prefix.len()] != SEP {
                    break;
                }

                let meta: AggregateMeta = serde_json::from_slice(&value)?;
                items.push(AggregatePositionEntry {
                    aggregate_type: meta.aggregate_type,
                    aggregate_id: meta.aggregate_id,
                    version: meta.version,
                });
            }

            Ok(items)
        })();
        let duration = start.elapsed().as_secs_f64();
        record_store_op(
            "rocksdb_iter_meta_positions",
            if result.is_ok() { "ok" } else { "err" },
            duration,
        );
        result
    }

    fn load_meta_from(
        &self,
        index: AggregateIndex,
        aggregate_type: &str,
        aggregate_id: &str,
    ) -> Result<Option<AggregateMeta>> {
        let start = Instant::now();
        let result = (|| {
            let key = meta_key_for(index, aggregate_type, aggregate_id);
            let value = self
                .db
                .get(key)
                .map_err(|err| EventError::Storage(err.to_string()))?;
            if let Some(value) = value {
                Ok(Some(serde_json::from_slice(&value)?))
            } else {
                Ok(None)
            }
        })();
        let duration = start.elapsed().as_secs_f64();
        record_store_op(
            "rocksdb_get_meta",
            if result.is_ok() { "ok" } else { "err" },
            duration,
        );
        result
    }

    fn load_meta_any(
        &self,
        aggregate_type: &str,
        aggregate_id: &str,
    ) -> Result<Option<(AggregateMeta, AggregateIndex)>> {
        if let Some(meta) =
            self.load_meta_from(AggregateIndex::Active, aggregate_type, aggregate_id)?
        {
            return Ok(Some((meta, AggregateIndex::Active)));
        }
        if let Some(meta) =
            self.load_meta_from(AggregateIndex::Archived, aggregate_type, aggregate_id)?
        {
            return Ok(Some((meta, AggregateIndex::Archived)));
        }
        Ok(None)
    }

    fn load_meta(&self, aggregate_type: &str, aggregate_id: &str) -> Result<Option<AggregateMeta>> {
        self.load_meta_from(AggregateIndex::Active, aggregate_type, aggregate_id)
    }

    pub fn create_aggregate(&self, aggregate_type: &str, aggregate_id: &str) -> Result<()> {
        self.ensure_writable()?;
        let _guard = self.write_lock.lock();

        if self.load_meta_any(aggregate_type, aggregate_id)?.is_some() {
            return Err(EventError::Storage(format!(
                "aggregate {}:{} already exists",
                aggregate_type, aggregate_id
            )));
        }

        let meta = AggregateMeta::new(aggregate_type.to_string(), aggregate_id.to_string());
        let state: BTreeMap<String, String> = BTreeMap::new();

        let start_meta = Instant::now();
        let meta_result = self
            .db
            .put(
                meta_key(aggregate_type, aggregate_id),
                serde_json::to_vec(&meta)?,
            )
            .map_err(|err| EventError::Storage(err.to_string()));
        let duration_meta = start_meta.elapsed().as_secs_f64();
        record_store_op(
            "rocksdb_put_meta",
            if meta_result.is_ok() { "ok" } else { "err" },
            duration_meta,
        );
        meta_result?;

        let start_state = Instant::now();
        let state_result = self
            .db
            .put(
                state_key(aggregate_type, aggregate_id),
                serde_json::to_vec(&state)?,
            )
            .map_err(|err| EventError::Storage(err.to_string()));
        let duration_state = start_state.elapsed().as_secs_f64();
        record_store_op(
            "rocksdb_put_state",
            if state_result.is_ok() { "ok" } else { "err" },
            duration_state,
        );
        state_result?;

        Ok(())
    }

    pub fn remove_aggregate(&self, aggregate_type: &str, aggregate_id: &str) -> Result<()> {
        self.ensure_writable()?;
        let _guard = self.write_lock.lock();

        let Some(meta) = self.load_meta(aggregate_type, aggregate_id)? else {
            return Err(EventError::AggregateNotFound);
        };

        if meta.version > 0 {
            return Err(EventError::Storage(format!(
                "aggregate {}:{} cannot be removed while events exist",
                aggregate_type, aggregate_id
            )));
        }

        let mut batch = WriteBatch::default();
        batch.delete(meta_key(aggregate_type, aggregate_id));
        batch.delete(state_key(aggregate_type, aggregate_id));
        self.sync_timestamp_indexes(&mut batch, AggregateIndex::Active, Some(&meta), None);
        // best-effort cleanup of reference index entries (tenant/domain handled by caller)
        self.write_batch(batch)
    }

    pub fn update_reference_index(
        &self,
        tenant: &str,
        aggregate_type: &str,
        aggregate_id: &str,
        references: &[(String, String)],
    ) -> Result<()> {
        self.ensure_writable()?;
        let _guard = self.write_lock.lock();

        let source_key = refsrc_key(tenant, aggregate_type, aggregate_id);
        let prev: Vec<StoredReferencePath> = self
            .db
            .get(source_key.clone())
            .map_err(|err| EventError::Storage(err.to_string()))?
            .map(|bytes| serde_json::from_slice(&bytes))
            .transpose()
            .map_err(|err| EventError::Serialization(err.to_string()))?
            .unwrap_or_default();

        let mut batch = WriteBatch::default();

        // Remove old target mappings
        for entry in prev {
            let target_key = reftgt_key(tenant, &entry.target);
            let existing: Vec<StoredReferrer> = self
                .db
                .get(&target_key)
                .map_err(|err| EventError::Storage(err.to_string()))?
                .map(|bytes| serde_json::from_slice(&bytes))
                .transpose()
                .map_err(|err| EventError::Serialization(err.to_string()))?
                .unwrap_or_default();
            let filtered: Vec<StoredReferrer> = existing
                .into_iter()
                .filter(|r| !(r.aggregate_type == aggregate_type && r.aggregate_id == aggregate_id))
                .collect();
            if filtered.is_empty() {
                batch.delete(target_key);
            } else {
                batch.put(target_key, serde_json::to_vec(&filtered)?);
            }
        }

        // Add new mappings
        let mut new_paths = Vec::new();
        for (target, path) in references {
            let path_entry = StoredReferencePath {
                target: target.clone(),
                path: path.clone(),
            };
            new_paths.push(path_entry.clone());

            let target_key = reftgt_key(tenant, target);
            let mut existing: Vec<StoredReferrer> = self
                .db
                .get(&target_key)
                .map_err(|err| EventError::Storage(err.to_string()))?
                .map(|bytes| serde_json::from_slice(&bytes))
                .transpose()
                .map_err(|err| EventError::Serialization(err.to_string()))?
                .unwrap_or_default();
            if !existing.iter().any(|r| {
                r.aggregate_type == aggregate_type
                    && r.aggregate_id == aggregate_id
                    && r.path == *path
            }) {
                existing.push(StoredReferrer {
                    aggregate_type: aggregate_type.to_string(),
                    aggregate_id: aggregate_id.to_string(),
                    path: path.clone(),
                });
                batch.put(target_key, serde_json::to_vec(&existing)?);
            }
        }

        batch.put(source_key, serde_json::to_vec(&new_paths)?);
        self.write_batch(batch)
    }

    pub fn clear_reference_index(
        &self,
        tenant: &str,
        aggregate_type: &str,
        aggregate_id: &str,
    ) -> Result<()> {
        self.update_reference_index(tenant, aggregate_type, aggregate_id, &[])
    }

    pub fn referrers_for(
        &self,
        tenant: &str,
        target: &str,
    ) -> Result<Vec<(String, String, String)>> {
        let target_key = reftgt_key(tenant, target);
        let existing: Vec<StoredReferrer> = self
            .db
            .get(&target_key)
            .map_err(|err| EventError::Storage(err.to_string()))?
            .map(|bytes| serde_json::from_slice(&bytes))
            .transpose()
            .map_err(|err| EventError::Serialization(err.to_string()))?
            .unwrap_or_default();
        Ok(existing
            .into_iter()
            .map(|r| (r.aggregate_type, r.aggregate_id, r.path))
            .collect())
    }

    pub fn referrers_for_any_tenant(
        &self,
        target: &str,
    ) -> Result<Vec<(String, String, String, String)>> {
        let mut matches = Vec::new();
        let prefix = PREFIX_REF_TARGET.as_bytes();
        for entry in self.db.prefix_iterator(prefix) {
            let (key, value) = entry.map_err(|err| EventError::Storage(err.to_string()))?;
            let segments: Vec<&[u8]> = key.split(|byte| *byte == SEP).collect();
            if segments.len() != 3 {
                continue;
            }
            let key_target = match str::from_utf8(segments[2]) {
                Ok(value) => value,
                Err(_) => continue,
            };
            if key_target != target {
                continue;
            }
            let tenant = match str::from_utf8(segments[1]) {
                Ok(value) => value.to_string(),
                Err(_) => continue,
            };
            let referrers: Vec<StoredReferrer> = serde_json::from_slice(&value).map_err(|err| {
                EventError::Serialization(format!("failed to decode referrer index: {err}"))
            })?;
            for r in referrers {
                matches.push((tenant.clone(), r.aggregate_type, r.aggregate_id, r.path));
            }
        }
        Ok(matches)
    }

    fn load_state_map_from(
        &self,
        index: AggregateIndex,
        aggregate_type: &str,
        aggregate_id: &str,
    ) -> Result<BTreeMap<String, String>> {
        let start = Instant::now();
        let result = (|| {
            let key = state_key_for(index, aggregate_type, aggregate_id);
            let value = self
                .db
                .get(key)
                .map_err(|err| EventError::Storage(err.to_string()))?;
            if let Some(value) = value {
                self.decode_state_map(&value)
            } else {
                Ok(BTreeMap::new())
            }
        })();
        let duration = start.elapsed().as_secs_f64();
        record_store_op(
            "rocksdb_get_state",
            if result.is_ok() { "ok" } else { "err" },
            duration,
        );
        result
    }

    fn load_state_map(
        &self,
        aggregate_type: &str,
        aggregate_id: &str,
    ) -> Result<BTreeMap<String, String>> {
        self.load_state_map_from(AggregateIndex::Active, aggregate_type, aggregate_id)
    }
}

fn configure_db_options(options: &mut Options, read_only: bool) {
    if !read_only {
        if let Ok(parallelism) = std::thread::available_parallelism() {
            let workers = parallelism.get().max(2) as i32;
            options.increase_parallelism(workers);
            options.set_max_background_jobs(workers);
        }
        options.set_allow_concurrent_memtable_write(true);
        options.set_enable_write_thread_adaptive_yield(true);
        options.set_write_buffer_size(64 * 1024 * 1024);
        options.set_max_write_buffer_number(4);
        options.set_min_write_buffer_number_to_merge(2);
        options.set_level_zero_file_num_compaction_trigger(8);
        options.set_level_zero_slowdown_writes_trigger(17);
        options.set_level_zero_stop_writes_trigger(24);
        options.set_target_file_size_base(128 * 1024 * 1024);
        options.set_max_total_wal_size(256 * 1024 * 1024);
    }
    options.set_bytes_per_sync(2 * 1024 * 1024);
    options.set_optimize_filters_for_hits(true);
}

fn meta_prefix_for(index: AggregateIndex) -> Vec<u8> {
    key_with_segments(&[index.meta_prefix()])
}

fn meta_prefix() -> Vec<u8> {
    meta_prefix_for(AggregateIndex::Active)
}

fn meta_key(aggregate_type: &str, aggregate_id: &str) -> Vec<u8> {
    meta_key_for(AggregateIndex::Active, aggregate_type, aggregate_id)
}

fn meta_archived_key(aggregate_type: &str, aggregate_id: &str) -> Vec<u8> {
    meta_key_for(AggregateIndex::Archived, aggregate_type, aggregate_id)
}

fn state_key(aggregate_type: &str, aggregate_id: &str) -> Vec<u8> {
    state_key_for(AggregateIndex::Active, aggregate_type, aggregate_id)
}

fn state_archived_key(aggregate_type: &str, aggregate_id: &str) -> Vec<u8> {
    state_key_for(AggregateIndex::Archived, aggregate_type, aggregate_id)
}

fn meta_key_for(index: AggregateIndex, aggregate_type: &str, aggregate_id: &str) -> Vec<u8> {
    key_with_segments(&[index.meta_prefix(), aggregate_type, aggregate_id])
}

fn state_key_for(index: AggregateIndex, aggregate_type: &str, aggregate_id: &str) -> Vec<u8> {
    key_with_segments(&[index.state_prefix(), aggregate_type, aggregate_id])
}

fn event_prefix_for(index: AggregateIndex, aggregate_type: &str, aggregate_id: &str) -> Vec<u8> {
    let prefix = match index {
        AggregateIndex::Active => PREFIX_EVENT,
        AggregateIndex::Archived => PREFIX_EVENT_ARCHIVED,
    };
    let mut key = key_with_segments(&[prefix, aggregate_type, aggregate_id]);
    key.push(SEP);
    key
}

fn event_key_for(
    index: AggregateIndex,
    aggregate_type: &str,
    aggregate_id: &str,
    version: u64,
) -> Vec<u8> {
    let mut key = event_prefix_for(index, aggregate_type, aggregate_id);
    key.extend_from_slice(&version.to_be_bytes());
    key
}

fn event_key(aggregate_type: &str, aggregate_id: &str, version: u64) -> Vec<u8> {
    event_key_for(
        AggregateIndex::Active,
        aggregate_type,
        aggregate_id,
        version,
    )
}

fn timestamp_index_prefix(kind: TimestampIndexKind, index: AggregateIndex) -> Vec<u8> {
    key_with_segments(&[kind.prefix(index)])
}

fn timestamp_index_key(
    kind: TimestampIndexKind,
    index: AggregateIndex,
    timestamp_ms: i64,
    aggregate_type: &str,
    aggregate_id: &str,
) -> Vec<u8> {
    let mut key = timestamp_index_prefix(kind, index);
    key.push(SEP);
    key.extend_from_slice(&timestamp_ms.to_be_bytes());
    key.push(SEP);
    key.extend_from_slice(aggregate_type.as_bytes());
    key.push(SEP);
    key.extend_from_slice(aggregate_id.as_bytes());
    key
}

fn parse_timestamp_index_entry(prefix: &[u8], key: &[u8]) -> Option<TimestampIndexEntry> {
    if !key.starts_with(prefix) {
        return None;
    }
    let mut cursor = prefix.len();
    if key.get(cursor)? != &SEP {
        return None;
    }
    cursor += 1;
    let end_ts = cursor.checked_add(8)?;
    let timestamp_bytes: [u8; 8] = key.get(cursor..end_ts)?.try_into().ok()?;
    let timestamp_ms = i64::from_be_bytes(timestamp_bytes);
    cursor = end_ts;
    if key.get(cursor)? != &SEP {
        return None;
    }
    cursor += 1;
    let next_sep = key[cursor..]
        .iter()
        .position(|byte| *byte == SEP)
        .map(|offset| cursor + offset)?;
    let aggregate_type = std::str::from_utf8(&key[cursor..next_sep])
        .ok()?
        .to_string();
    cursor = next_sep + 1;
    if cursor > key.len() {
        return None;
    }
    let aggregate_id = std::str::from_utf8(&key[cursor..]).ok()?.to_string();
    Some(TimestampIndexEntry {
        aggregate_type,
        aggregate_id,
        timestamp_ms,
    })
}

fn meta_timestamp_ms(meta: &AggregateMeta, kind: TimestampIndexKind) -> Option<i64> {
    match kind {
        TimestampIndexKind::CreatedAt => meta.created_at.as_ref().map(|dt| dt.timestamp_millis()),
        TimestampIndexKind::UpdatedAt => meta.updated_at.as_ref().map(|dt| dt.timestamp_millis()),
    }
}

fn timestamp_index_metric(kind: TimestampIndexKind, index: AggregateIndex) -> &'static str {
    match (kind, index) {
        (TimestampIndexKind::CreatedAt, AggregateIndex::Active) => {
            "rocksdb_iter_meta_created_index"
        }
        (TimestampIndexKind::CreatedAt, AggregateIndex::Archived) => {
            "rocksdb_iter_meta_created_index_archived"
        }
        (TimestampIndexKind::UpdatedAt, AggregateIndex::Active) => {
            "rocksdb_iter_meta_updated_index"
        }
        (TimestampIndexKind::UpdatedAt, AggregateIndex::Archived) => {
            "rocksdb_iter_meta_updated_index_archived"
        }
    }
}

fn compare_timestamp_entries(
    lhs: &TimestampIndexEntry,
    rhs: &TimestampIndexEntry,
    descending: bool,
) -> StdOrdering {
    let mut ordering = lhs.timestamp_ms.cmp(&rhs.timestamp_ms);
    if ordering == StdOrdering::Equal {
        ordering = lhs.aggregate_type.cmp(&rhs.aggregate_type);
    }
    if ordering == StdOrdering::Equal {
        ordering = lhs.aggregate_id.cmp(&rhs.aggregate_id);
    }
    if descending {
        ordering.reverse()
    } else {
        ordering
    }
}

fn compare_timestamp_values(
    lhs_ts: i64,
    lhs_type: &str,
    lhs_id: &str,
    rhs_ts: i64,
    rhs_type: &str,
    rhs_id: &str,
) -> StdOrdering {
    let mut ordering = lhs_ts.cmp(&rhs_ts);
    if ordering == StdOrdering::Equal {
        ordering = lhs_type.cmp(rhs_type);
    }
    if ordering == StdOrdering::Equal {
        ordering = lhs_id.cmp(rhs_id);
    }
    ordering
}

fn event_prefix_for_scope(index: AggregateIndex, scope: EventQueryScope<'_>) -> Vec<u8> {
    match scope {
        EventQueryScope::All => key_with_segments(&[index.event_prefix()]),
        EventQueryScope::AggregateType(aggregate_type) => {
            key_with_segments(&[index.event_prefix(), aggregate_type])
        }
        EventQueryScope::Aggregate {
            aggregate_type,
            aggregate_id,
        } => key_with_segments(&[index.event_prefix(), aggregate_type, aggregate_id]),
    }
}

fn event_cursor_matches_scope(cursor: &EventCursor, scope: EventQueryScope<'_>) -> bool {
    match scope {
        EventQueryScope::All => true,
        EventQueryScope::AggregateType(aggregate_type) => cursor.aggregate_type() == aggregate_type,
        EventQueryScope::Aggregate {
            aggregate_type,
            aggregate_id,
        } => cursor.aggregate_type() == aggregate_type && cursor.aggregate_id() == aggregate_id,
    }
}

fn snapshot_key(aggregate_type: &str, aggregate_id: &str, created_at: DateTime<Utc>) -> Vec<u8> {
    let mut key = snapshot_prefix(aggregate_type, aggregate_id);
    key.push(SEP);
    key.extend_from_slice(&created_at.timestamp_millis().to_be_bytes());
    key
}

fn snapshot_prefix(aggregate_type: &str, aggregate_id: &str) -> Vec<u8> {
    key_with_segments(&[PREFIX_SNAPSHOT, aggregate_type, aggregate_id])
}

fn snapshot_index_key(snapshot_id: SnowflakeId) -> Vec<u8> {
    key_with_segments(&[PREFIX_SNAPSHOT_ID, &snapshot_id.to_string()])
}

fn parse_event_version(key: &[u8]) -> Result<u64> {
    if key.len() < 8 {
        return Err(EventError::Storage("event key too short".into()));
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&key[key.len() - 8..]);
    Ok(u64::from_be_bytes(buf))
}

fn key_with_segments(parts: &[&str]) -> Vec<u8> {
    let mut key = Vec::new();
    let mut iter = parts.iter();
    if let Some(first) = iter.next() {
        key.extend_from_slice(first.as_bytes());
    }
    for part in iter {
        key.push(SEP);
        key.extend_from_slice(part.as_bytes());
    }
    key
}

fn refsrc_key(tenant: &str, aggregate_type: &str, aggregate_id: &str) -> Vec<u8> {
    key_with_segments(&[PREFIX_REF_SOURCE, tenant, aggregate_type, aggregate_id])
}

fn reftgt_key(tenant: &str, target: &str) -> Vec<u8> {
    key_with_segments(&[PREFIX_REF_TARGET, tenant, target])
}

fn update_reference_index_inner(
    db: &DBWithThreadMode<MultiThreaded>,
    batch: &mut WriteBatch,
    tenant: &str,
    aggregate_type: &str,
    aggregate_id: &str,
    references: &[(String, String)],
) -> Result<()> {
    let source_key = refsrc_key(tenant, aggregate_type, aggregate_id);
    let prev: Vec<StoredReferencePath> = db
        .get(source_key.clone())
        .map_err(|err| EventError::Storage(err.to_string()))?
        .map(|bytes| serde_json::from_slice(&bytes))
        .transpose()
        .map_err(|err| EventError::Serialization(err.to_string()))?
        .unwrap_or_default();

    // Remove old target mappings
    for entry in prev {
        let target_key = reftgt_key(tenant, &entry.target);
        let existing: Vec<StoredReferrer> = db
            .get(&target_key)
            .map_err(|err| EventError::Storage(err.to_string()))?
            .map(|bytes| serde_json::from_slice(&bytes))
            .transpose()
            .map_err(|err| EventError::Serialization(err.to_string()))?
            .unwrap_or_default();
        let filtered: Vec<StoredReferrer> = existing
            .into_iter()
            .filter(|r| !(r.aggregate_type == aggregate_type && r.aggregate_id == aggregate_id))
            .collect();
        if filtered.is_empty() {
            batch.delete(target_key);
        } else {
            batch.put(target_key, serde_json::to_vec(&filtered)?);
        }
    }

    // Add new mappings
    let mut new_paths = Vec::new();
    for (target, path) in references {
        let path_entry = StoredReferencePath {
            target: target.clone(),
            path: path.clone(),
        };
        new_paths.push(path_entry.clone());

        let target_key = reftgt_key(tenant, target);
        let mut existing: Vec<StoredReferrer> = db
            .get(&target_key)
            .map_err(|err| EventError::Storage(err.to_string()))?
            .map(|bytes| serde_json::from_slice(&bytes))
            .transpose()
            .map_err(|err| EventError::Serialization(err.to_string()))?
            .unwrap_or_default();
        if !existing.iter().any(|r| {
            r.aggregate_type == aggregate_type && r.aggregate_id == aggregate_id && r.path == *path
        }) {
            existing.push(StoredReferrer {
                aggregate_type: aggregate_type.to_string(),
                aggregate_id: aggregate_id.to_string(),
                path: path.clone(),
            });
            batch.put(target_key, serde_json::to_vec(&existing)?);
        }
    }

    batch.put(source_key, serde_json::to_vec(&new_paths)?);
    Ok(())
}

fn record_store_op(operation: &'static str, status: &'static str, duration: f64) {
    let labels = [("operation", operation), ("status", status)];
    counter!("eventdbx_store_operations_total", &labels).increment(1);
    histogram!("eventdbx_store_operation_duration_seconds", &labels).record(duration);
}

fn hash_event(
    aggregate_type: &str,
    aggregate_id: &str,
    version: u64,
    event_type: &str,
    payload: &BTreeMap<String, String>,
) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(aggregate_type.as_bytes());
    hasher.update(aggregate_id.as_bytes());
    hasher.update(version.to_be_bytes());
    hasher.update(event_type.as_bytes());

    for (key, value) in payload {
        hasher.update(key.as_bytes());
        hasher.update(value.as_bytes());
    }

    hex::encode(hasher.finalize())
}

pub fn payload_to_map(value: &Value) -> BTreeMap<String, String> {
    fn normalize(value: &Value) -> String {
        match value {
            Value::String(s) => s.clone(),
            Value::Null => String::new(),
            _ => value.to_string(),
        }
    }

    match value {
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| (k.clone(), normalize(v)))
            .collect::<BTreeMap<_, _>>(),
        other => {
            let mut map = BTreeMap::new();
            if !other.is_null() {
                map.insert("_value".into(), normalize(other));
            }
            map
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_consistent_merkle_root() {
        let root = compute_merkle_root(&[]);
        assert_eq!(root, empty_root());

        let root = compute_merkle_root(&["abc".into()]);
        let root2 = compute_merkle_root(&["abc".into()]);
        assert_eq!(root, root2);
    }

    #[test]
    fn append_and_retrieve_events() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");

        {
            let store = EventStore::open(path.clone(), None, 0).unwrap();

            let payload = serde_json::json!({ "name": "Alice" });

            let record = store
                .append(AppendEvent {
                    aggregate_type: "patient".into(),
                    aggregate_id: "patient-1".into(),
                    event_type: "patient-created".into(),
                    payload: payload.clone(),
                    metadata: None,
                    issued_by: None,
                    note: None,
                    tenant: "default".into(),
                    reference_targets: Vec::new(),
                })
                .unwrap();

            assert_eq!(record.version, 1);
            assert_eq!(record.payload["name"], "Alice");

            let state = store.get_aggregate_state("patient", "patient-1").unwrap();
            assert_eq!(state.version, 1);
            assert_eq!(state.state["name"], "Alice");
            assert!(!state.archived);

            let events = store.list_events("patient", "patient-1").unwrap();
            assert_eq!(events.len(), 1);
        }
    }

    #[test]
    fn transaction_appends_multiple_events() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");
        let store = EventStore::open(path, None, 0).unwrap();

        let mut tx = store.transaction().unwrap();
        let first = tx
            .append(AppendEvent {
                aggregate_type: "order".into(),
                aggregate_id: "order-42".into(),
                event_type: "order-created".into(),
                payload: serde_json::json!({ "status": "processing" }),
                metadata: None,
                issued_by: None,
                note: None,
                tenant: "default".into(),
                reference_targets: Vec::new(),
            })
            .unwrap();
        assert_eq!(first.version, 1);

        let second = tx
            .append(AppendEvent {
                aggregate_type: "order".into(),
                aggregate_id: "order-42".into(),
                event_type: "order-updated".into(),
                payload: serde_json::json!({
                    "status": "shipped",
                    "tracking": "abc123"
                }),
                metadata: None,
                issued_by: None,
                note: None,
                tenant: "default".into(),
                reference_targets: Vec::new(),
            })
            .unwrap();
        assert_eq!(second.version, 2);

        let committed = tx.commit().unwrap();
        assert_eq!(committed.len(), 2);
        assert_eq!(committed[0].version, 1);
        assert_eq!(committed[1].version, 2);

        let state = store.get_aggregate_state("order", "order-42").unwrap();
        assert_eq!(state.version, 2);
        assert_eq!(state.state["status"], "shipped");
        assert_eq!(state.state["tracking"], "abc123");

        let events = store.list_events("order", "order-42").unwrap();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn aggregate_sort_key_order_detection() {
        assert!(aggregate_sort_matches_key_order(&[]));

        let sorts = vec![AggregateSort {
            field: AggregateSortField::AggregateType,
            descending: false,
        }];
        assert!(aggregate_sort_matches_key_order(&sorts));

        let sorts = vec![
            AggregateSort {
                field: AggregateSortField::AggregateType,
                descending: false,
            },
            AggregateSort {
                field: AggregateSortField::AggregateId,
                descending: false,
            },
        ];
        assert!(aggregate_sort_matches_key_order(&sorts));

        let invalid = vec![AggregateSort {
            field: AggregateSortField::AggregateId,
            descending: false,
        }];
        assert!(!aggregate_sort_matches_key_order(&invalid));

        let descending = vec![AggregateSort {
            field: AggregateSortField::AggregateType,
            descending: true,
        }];
        assert!(!aggregate_sort_matches_key_order(&descending));
    }

    #[test]
    fn event_sort_key_order_detection() {
        assert!(event_sort_matches_key_order(&[]));

        let sorts = vec![EventSort {
            field: EventSortField::AggregateType,
            descending: false,
        }];
        assert!(event_sort_matches_key_order(&sorts));

        let sorts = vec![
            EventSort {
                field: EventSortField::AggregateType,
                descending: false,
            },
            EventSort {
                field: EventSortField::AggregateId,
                descending: false,
            },
            EventSort {
                field: EventSortField::Version,
                descending: false,
            },
        ];
        assert!(event_sort_matches_key_order(&sorts));

        let invalid_field = vec![EventSort {
            field: EventSortField::EventType,
            descending: false,
        }];
        assert!(!event_sort_matches_key_order(&invalid_field));

        let descending = vec![EventSort {
            field: EventSortField::AggregateType,
            descending: true,
        }];
        assert!(!event_sort_matches_key_order(&descending));
    }

    #[test]
    fn aggregates_sort_by_created_at_desc() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");
        let store = EventStore::open(path, None, 0).unwrap();
        let base = Utc::now();
        let cases = vec![
            ("agg-1", base + chrono::Duration::milliseconds(10)),
            ("agg-2", base + chrono::Duration::milliseconds(20)),
            ("agg-3", base + chrono::Duration::milliseconds(30)),
        ];

        for (idx, (aggregate_id, created_at)) in cases.iter().enumerate() {
            import_history(
                &store,
                aggregate_id,
                &[created_at.to_owned()],
                (idx as u64) * 10,
            );
        }

        let sorts = vec![AggregateSort {
            field: AggregateSortField::CreatedAt,
            descending: true,
        }];
        let aggregates = store.aggregates_paginated_with_transform(
            0,
            Some(3),
            Some(&sorts),
            AggregateQueryScope::ActiveOnly,
            None,
            |aggregate| Some(aggregate),
        );
        let ids: Vec<_> = aggregates.into_iter().map(|agg| agg.aggregate_id).collect();
        assert_eq!(ids, vec!["agg-3", "agg-2", "agg-1"]);
    }

    #[test]
    fn aggregates_sort_by_updated_at_desc() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");
        let store = EventStore::open(path, None, 0).unwrap();
        let base = Utc::now();
        import_history(
            &store,
            "agg-1",
            &[
                base + chrono::Duration::milliseconds(5),
                base + chrono::Duration::milliseconds(50),
            ],
            100,
        );
        import_history(
            &store,
            "agg-2",
            &[
                base + chrono::Duration::milliseconds(10),
                base + chrono::Duration::milliseconds(30),
            ],
            200,
        );

        let sorts = vec![AggregateSort {
            field: AggregateSortField::UpdatedAt,
            descending: true,
        }];
        let aggregates = store.aggregates_paginated_with_transform(
            0,
            Some(2),
            Some(&sorts),
            AggregateQueryScope::ActiveOnly,
            None,
            |aggregate| Some(aggregate),
        );
        let ids: Vec<_> = aggregates.into_iter().map(|agg| agg.aggregate_id).collect();
        assert_eq!(ids, vec!["agg-1", "agg-2"]);
    }

    fn import_history(
        store: &EventStore,
        aggregate_id: &str,
        timestamps: &[DateTime<Utc>],
        base_event_id: u64,
    ) {
        let aggregate_type = "chronicle";
        let event_type = "created";
        let mut hashes = Vec::new();
        for (idx, timestamp) in timestamps.iter().enumerate() {
            let version = (idx as u64) + 1;
            let payload = serde_json::json!({ "seq": version });
            let payload_map = payload_to_map(&payload);
            let hash = hash_event(
                aggregate_type,
                aggregate_id,
                version,
                event_type,
                &payload_map,
            );
            hashes.push(hash.clone());
            let merkle_root = compute_merkle_root(&hashes);
            let record = EventRecord {
                aggregate_type: aggregate_type.into(),
                aggregate_id: aggregate_id.to_string(),
                event_type: event_type.into(),
                payload: payload.clone(),
                extensions: None,
                metadata: EventMetadata {
                    event_id: SnowflakeId::from_u64(base_event_id + version),
                    created_at: *timestamp,
                    issued_by: None,
                    note: None,
                },
                version,
                hash,
                merkle_root,
            };
            store.append_imported_event(record).unwrap();
        }
    }

    #[test]
    fn transaction_without_commit_discards_changes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");
        let store = EventStore::open(path.clone(), None, 0).unwrap();

        {
            let mut tx = store.transaction().unwrap();
            tx.append(AppendEvent {
                aggregate_type: "invoice".into(),
                aggregate_id: "inv-1".into(),
                event_type: "invoice-created".into(),
                payload: serde_json::json!({ "total": "100.00" }),
                metadata: None,
                issued_by: None,
                note: None,
                tenant: "default".into(),
                reference_targets: Vec::new(),
            })
            .unwrap();
        }

        let err = store.list_events("invoice", "inv-1").unwrap_err();
        assert!(matches!(err, crate::error::EventError::AggregateNotFound));
    }

    #[test]
    fn snapshots_and_archive_flow() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");
        let store = EventStore::open(path, None, 0).unwrap();

        let payload = serde_json::json!({ "status": "active" });

        store
            .append(AppendEvent {
                aggregate_type: "order".into(),
                aggregate_id: "order-1".into(),
                event_type: "order-created".into(),
                payload,
                metadata: None,
                issued_by: None,
                note: None,
                tenant: "default".into(),
                reference_targets: Vec::new(),
            })
            .unwrap();

        let snapshot = store
            .create_snapshot("order", "order-1", Some("initial".into()))
            .unwrap();
        assert_eq!(snapshot.version, 1);
        assert_eq!(snapshot.comment.as_deref(), Some("initial"));

        let state = store
            .set_archive("order", "order-1", true, Some("closed".into()))
            .unwrap();
        assert!(state.archived);

        let err = store
            .append(AppendEvent {
                aggregate_type: "order".into(),
                aggregate_id: "order-1".into(),
                event_type: "order-updated".into(),
                payload: serde_json::json!({}),
                metadata: None,
                issued_by: None,
                note: None,
                tenant: "default".into(),
                reference_targets: Vec::new(),
            })
            .unwrap_err();
        assert!(matches!(err, crate::error::EventError::AggregateArchived));

        let state = store.set_archive("order", "order-1", false, None).unwrap();
        assert!(!state.archived);

        // Verify snapshot lookup by id uses index.
        let found = store
            .find_snapshot_by_id(snapshot.snapshot_id.expect("snapshot id"))
            .unwrap();
        assert!(found.is_some());
    }

    #[test]
    fn append_event_with_note_persists_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");
        let store = EventStore::open(path, None, 0).unwrap();

        let note_text = "initial import".to_string();
        let record = store
            .append(AppendEvent {
                aggregate_type: "account".into(),
                aggregate_id: "acct-1".into(),
                event_type: "account-created".into(),
                payload: serde_json::json!({ "status": "active" }),
                metadata: None,
                issued_by: None,
                note: Some(note_text.clone()),
                tenant: "default".into(),
                reference_targets: Vec::new(),
            })
            .unwrap();

        assert_eq!(record.metadata.note.as_deref(), Some(note_text.as_str()));
    }

    #[test]
    fn reject_event_note_exceeding_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");
        let store = EventStore::open(path, None, 0).unwrap();

        let long_note = "x".repeat(MAX_EVENT_NOTE_LENGTH + 1);
        let err = store
            .append(AppendEvent {
                aggregate_type: "account".into(),
                aggregate_id: "acct-2".into(),
                event_type: "account-created".into(),
                payload: serde_json::json!({ "status": "pending" }),
                metadata: None,
                issued_by: None,
                note: Some(long_note),
                tenant: "default".into(),
                reference_targets: Vec::new(),
            })
            .unwrap_err();

        assert!(matches!(err, crate::error::EventError::InvalidSchema(_)));
    }

    #[test]
    fn append_event_with_extensions_persists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");
        let store = EventStore::open(path.clone(), None, 0).unwrap();

        let metadata = serde_json::json!({"@plugin": {"mode": "debug"}});
        let record = store
            .append(AppendEvent {
                aggregate_type: "account".into(),
                aggregate_id: "acct-3".into(),
                event_type: "account-created".into(),
                payload: serde_json::json!({ "status": "active" }),
                metadata: Some(metadata.clone()),
                issued_by: None,
                note: None,
                tenant: "default".into(),
                reference_targets: Vec::new(),
            })
            .unwrap();

        assert_eq!(record.extensions, Some(metadata.clone()));

        drop(store);
        let reopened = EventStore::open(path, None, 0).unwrap();
        let events = reopened.list_events("account", "acct-3").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].extensions, Some(metadata));
    }

    #[test]
    fn prepare_payload_from_patch_updates_nested_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");
        let store = EventStore::open(path.clone(), None, 0).unwrap();

        store
            .append(AppendEvent {
                aggregate_type: "patient".into(),
                aggregate_id: "p-1".into(),
                event_type: "patient-created".into(),
                payload: serde_json::json!({
                    "status": "active",
                    "contact": {
                        "address": { "city": "Portland" }
                    }
                }),
                metadata: None,
                issued_by: None,
                note: None,
                tenant: "default".into(),
                reference_targets: Vec::new(),
            })
            .unwrap();

        let patch = serde_json::json!([
            { "op": "replace", "path": "/status", "value": "inactive" },
            { "op": "replace", "path": "/contact/address/city", "value": "Spokane" }
        ]);

        let payload = store
            .prepare_payload_from_patch("patient", "p-1", &patch)
            .unwrap();

        assert_eq!(
            payload,
            serde_json::json!({
                "contact": { "address": { "city": "Spokane" } },
                "status": "inactive"
            })
        );
    }

    #[test]
    fn prepare_payload_from_patch_rejects_unsupported_operations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");
        let store = EventStore::open(path, None, 0).unwrap();

        let patch = serde_json::json!([
            { "op": "remove", "path": "/status" }
        ]);

        let err = store
            .prepare_payload_from_patch("patient", "p-unknown", &patch)
            .unwrap_err();

        assert!(matches!(err, crate::error::EventError::InvalidSchema(_)));
    }

    #[test]
    fn prepare_payload_from_patch_supports_add_on_new_aggregate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");
        let store = EventStore::open(path, None, 0).unwrap();

        let patch = serde_json::json!([
            { "op": "add", "path": "/status", "value": "inactive" }
        ]);

        let payload = store
            .prepare_payload_from_patch("patient", "p-new", &patch)
            .unwrap();

        assert_eq!(
            payload,
            serde_json::json!({
                "status": "inactive"
            })
        );
    }

    #[test]
    fn select_state_field_handles_nested_paths() {
        let mut map = BTreeMap::new();
        map.insert("status".into(), "\"active\"".into());
        map.insert(
            "profile".into(),
            r#"{"name":{"first":"Ada"},"scores":[10,20]}"#.into(),
        );

        assert_eq!(
            super::select_state_field(&map, "status"),
            Some(Value::String("active".into()))
        );
        assert_eq!(
            super::select_state_field(&map, "profile.name.first"),
            Some(Value::String("Ada".into()))
        );
        assert_eq!(
            super::select_state_field(&map, "profile.scores.1"),
            Some(Value::from(20))
        );
        assert_eq!(super::select_state_field(&map, "missing"), None);
        assert_eq!(super::select_state_field(&map, ""), None);
        assert_eq!(super::select_state_field(&map, "profile.scores.5"), None);
    }

    #[test]
    fn reports_store_in_use_on_active_lock() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");
        let _holder = EventStore::open(path.clone(), None, 0).unwrap();

        let result = EventStore::open(path, None, 0);
        match result {
            Err(EventError::Storage(message)) => {
                let lower = message.to_ascii_lowercase();
                assert!(
                    lower.contains("in use") || lower.contains("lock"),
                    "expected lock error message, got {message}"
                );
            }
            Err(other) => panic!("expected storage error for active lock, got {other:?}"),
            Ok(_) => panic!("expected store open to fail while lock is held"),
        }
    }

    #[test]
    fn recovers_after_lock_is_released() {
        use std::{thread, time::Duration};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");

        // Create the store so the directory exists before the holder grabs the lock.
        EventStore::open(path.clone(), None, 0).unwrap();

        let holder_path = path.clone();
        let handle = thread::spawn(move || {
            let store = EventStore::open(holder_path, None, 0).unwrap();
            thread::sleep(Duration::from_millis(150));
            drop(store);
        });

        thread::sleep(Duration::from_millis(25));
        let reopened = EventStore::open(path.clone(), None, 0).unwrap();
        drop(reopened);

        handle.join().unwrap();
    }

    #[test]
    fn snapshot_does_not_leave_lock_held() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");

        {
            let store = EventStore::open(path.clone(), None, 0).unwrap();
            store
                .append(AppendEvent {
                    aggregate_type: "account".into(),
                    aggregate_id: "acct-42".into(),
                    event_type: "created".into(),
                    payload: serde_json::json!({ "status": "active" }),
                    metadata: None,
                    issued_by: None,
                    note: None,
                    tenant: "default".into(),
                    reference_targets: Vec::new(),
                })
                .unwrap();
            store
                .create_snapshot("account", "acct-42", Some("first".into()))
                .unwrap();
        }

        let reopened = EventStore::open(path, None, 0).unwrap();
        let snapshot = reopened
            .create_snapshot("account", "acct-42", Some("second".into()))
            .unwrap();
        assert_eq!(snapshot.aggregate_id, "acct-42");
        assert_eq!(snapshot.aggregate_type, "account");
    }

    #[test]
    fn list_snapshots_filters_by_version() {
        use std::{thread, time::Duration};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");
        let store = EventStore::open(path, None, 0).unwrap();

        store
            .append(AppendEvent {
                aggregate_type: "order".into(),
                aggregate_id: "order-9".into(),
                event_type: "created".into(),
                payload: serde_json::json!({ "status": "new" }),
                metadata: None,
                issued_by: None,
                note: None,
                tenant: "default".into(),
                reference_targets: Vec::new(),
            })
            .unwrap();
        store
            .create_snapshot("order", "order-9", Some("v1".into()))
            .unwrap();

        thread::sleep(Duration::from_millis(2));

        store
            .append(AppendEvent {
                aggregate_type: "order".into(),
                aggregate_id: "order-9".into(),
                event_type: "updated".into(),
                payload: serde_json::json!({ "status": "shipped" }),
                metadata: None,
                issued_by: None,
                note: None,
                tenant: "default".into(),
                reference_targets: Vec::new(),
            })
            .unwrap();
        store
            .create_snapshot("order", "order-9", Some("v2".into()))
            .unwrap();

        let all = store
            .list_snapshots(Some("order"), Some("order-9"), None)
            .expect("list snapshots");
        assert_eq!(all.len(), 2);
        assert!(all.iter().any(|s| s.version == 1));
        assert!(all.iter().any(|s| s.version == 2));

        let filtered = store
            .list_snapshots(Some("order"), Some("order-9"), Some(1))
            .expect("list snapshots by version");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].version, 1);
    }

    #[test]
    fn list_snapshots_all_aggregates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");
        let store = EventStore::open(path, None, 0).unwrap();

        for agg_id in &["a-1", "b-2"] {
            store
                .append(AppendEvent {
                    aggregate_type: "person".into(),
                    aggregate_id: agg_id.to_string(),
                    event_type: "created".into(),
                    payload: serde_json::json!({ "name": agg_id }),
                    metadata: None,
                    issued_by: None,
                    note: None,
                    tenant: "default".into(),
                    reference_targets: Vec::new(),
                })
                .unwrap();
            store
                .create_snapshot("person", agg_id, Some("checkpoint".into()))
                .unwrap();
        }

        let snapshots = store
            .list_snapshots(None, None, None)
            .expect("list all snapshots");
        assert_eq!(snapshots.len(), 2);
    }

    #[test]
    fn referrers_for_any_tenant_returns_cross_tenant_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("event_store");
        let store = EventStore::open(path, None, 0).unwrap();

        // target aggregate in tenant beta
        store
            .append(AppendEvent {
                aggregate_type: "address".into(),
                aggregate_id: "addr-1".into(),
                event_type: "created".into(),
                payload: serde_json::json!({ "status": "active" }),
                metadata: None,
                issued_by: None,
                note: None,
                tenant: "beta".into(),
                reference_targets: Vec::new(),
            })
            .unwrap();

        // referrer stored under tenant alpha pointing at the beta address
        store
            .append(AppendEvent {
                aggregate_type: "farm".into(),
                aggregate_id: "farm-1".into(),
                event_type: "created".into(),
                payload: serde_json::json!({ "address": "beta#address#addr-1" }),
                metadata: None,
                issued_by: None,
                note: None,
                tenant: "alpha".into(),
                reference_targets: vec![("beta#address#addr-1".into(), "/address".into())],
            })
            .unwrap();

        let refs = store
            .referrers_for_any_tenant("beta#address#addr-1")
            .unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].0, "alpha");
        assert_eq!(refs[0].1, "farm");
        assert_eq!(refs[0].2, "farm-1");
        assert_eq!(refs[0].3, "/address");
    }
}
