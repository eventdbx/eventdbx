use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    error::{EventError, Result},
    filter::FilterExpr,
    plugin::PublishTarget,
    reference::{
        AggregateReference, ReferenceCascade, ReferenceContext, ReferenceFetchOutcome,
        ReferenceIntegrity, ReferenceResolutionStatus, Referrer, ResolvedAggregate,
        resolve_references,
    },
    restrict::{self, RestrictMode},
    schema::SchemaManager,
    snowflake::SnowflakeId,
    store::{
        self, ActorClaims, AggregateCursor, AggregateQueryScope, AggregateSort, AggregateState,
        AppendEvent, EventArchiveScope, EventCursor, EventQueryScope, EventRecord, EventStore,
        SnapshotRecord, select_state_field,
    },
    tenant_store::{BYTES_PER_MEGABYTE, TenantAssignmentStore},
    token::{JwtClaims, TokenManager},
    validation::{
        ensure_aggregate_id, ensure_first_event_rule, ensure_metadata_extensions,
        ensure_payload_size, ensure_snake_case,
    },
};
use parking_lot::Mutex;
use serde_json::{self, Value};
/// Shared context exposing the core store, schema, and token managers so
/// higher-level surfaces (REST, GraphQL, gRPC, plugins) can be layered on
/// top of a consistent API.
#[derive(Clone)]
pub struct CoreContext {
    tokens: Arc<TokenManager>,
    schemas: Arc<SchemaManager>,
    store: Arc<EventStore>,
    restrict: RestrictMode,
    list_page_size: usize,
    page_limit: usize,
    tenant_id: String,
    storage_quota_mb: Option<u64>,
    assignments: Arc<TenantAssignmentStore>,
    usage_cache: Arc<Mutex<StorageUsageCache>>,
    reference_default_depth: usize,
    reference_max_depth: usize,
}

const STORAGE_USAGE_REFRESH_EVENTS: u64 = 128;
const STORAGE_USAGE_REFRESH_INTERVAL: Duration = Duration::from_secs(30);

struct StorageUsageCache {
    bytes: u64,
    last_refresh: Instant,
    events_since_refresh: u64,
    initialized: bool,
    dirty: bool,
}

impl StorageUsageCache {
    fn new(initial: Option<u64>) -> Self {
        let now = Instant::now();
        match initial {
            Some(bytes) => Self {
                bytes,
                last_refresh: now,
                events_since_refresh: 0,
                initialized: true,
                dirty: false,
            },
            None => Self {
                bytes: 0,
                last_refresh: now,
                events_since_refresh: 0,
                initialized: false,
                dirty: false,
            },
        }
    }
}

impl CoreContext {
    pub fn new(
        tokens: Arc<TokenManager>,
        schemas: Arc<SchemaManager>,
        store: Arc<EventStore>,
        restrict: RestrictMode,
        list_page_size: usize,
        page_limit: usize,
        tenant_id: impl Into<String>,
        storage_quota_mb: Option<u64>,
        initial_usage_bytes: Option<u64>,
        assignments: Arc<TenantAssignmentStore>,
        reference_default_depth: usize,
        reference_max_depth: usize,
    ) -> Self {
        Self {
            tokens,
            schemas,
            store,
            restrict,
            list_page_size,
            page_limit,
            tenant_id: tenant_id.into(),
            storage_quota_mb,
            assignments,
            usage_cache: Arc::new(Mutex::new(StorageUsageCache::new(initial_usage_bytes))),
            reference_default_depth,
            reference_max_depth,
        }
    }

    pub fn tokens(&self) -> Arc<TokenManager> {
        Arc::clone(&self.tokens)
    }

    pub fn schemas(&self) -> Arc<SchemaManager> {
        Arc::clone(&self.schemas)
    }

    pub fn store(&self) -> Arc<EventStore> {
        Arc::clone(&self.store)
    }

    pub fn assignments(&self) -> Arc<TenantAssignmentStore> {
        Arc::clone(&self.assignments)
    }

    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }

    pub fn restrict(&self) -> RestrictMode {
        self.restrict
    }

    pub fn reference_default_depth(&self) -> usize {
        self.reference_default_depth
    }

    pub fn reference_max_depth(&self) -> usize {
        self.reference_max_depth
    }

    pub fn list_page_size(&self) -> usize {
        self.list_page_size
    }

    pub fn page_limit(&self) -> usize {
        self.page_limit
    }

    fn inbound_referrers_for(
        &self,
        aggregate_type: &str,
        aggregate_id: &str,
    ) -> Result<Vec<(String, String, String, String)>> {
        let target = format!("{}#{}#{}", self.tenant_id(), aggregate_type, aggregate_id);
        self.store
            .referrers_for_any_tenant(&target)
            .map_err(Into::into)
    }

    fn nullify_referrers(
        &self,
        referrers: &[(String, String, String, String)],
        target: &str,
    ) -> Result<()> {
        if referrers.is_empty() {
            return Ok(());
        }
        let store = self.store();
        let mut grouped: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
        for (_, agg_type, agg_id, path) in referrers {
            let pointer = json_pointer_from_path(path);
            grouped
                .entry((agg_type.clone(), agg_id.clone()))
                .or_default()
                .insert(pointer);
        }

        for ((agg_type, agg_id), paths) in grouped {
            let patch_ops: Vec<Value> = paths
                .into_iter()
                .map(|path| {
                    serde_json::json!({
                        "op": "replace",
                        "path": path,
                        "value": Value::Null
                    })
                })
                .collect();
            let patch = Value::Array(patch_ops);
            let payload = store.prepare_payload_from_patch(&agg_type, &agg_id, &patch)?;
            store.append(AppendEvent {
                aggregate_type: agg_type.clone(),
                aggregate_id: agg_id.clone(),
                event_type: "_ref_nullify".into(),
                payload,
                metadata: None,
                issued_by: None,
                note: Some(format!("auto-nullify due to {}", target)),
                tenant: self.tenant_id.clone(),
                reference_targets: Vec::new(),
            })?;
            self.note_storage_mutation()?;
        }
        Ok(())
    }

    pub fn is_hidden_aggregate(&self, aggregate_type: &str) -> bool {
        self.schemas
            .get(aggregate_type)
            .map(|schema| schema.hidden)
            .unwrap_or(false)
    }

    pub fn sanitize_aggregate(&self, mut aggregate: AggregateState) -> AggregateState {
        aggregate.state = self.filter_state_map(&aggregate.aggregate_type, aggregate.state);
        aggregate
    }

    #[allow(dead_code)]
    fn storage_usage_bytes(&self) -> Result<u64> {
        self.maybe_refresh_usage(false)
    }

    fn enforce_storage_quota(&self) -> Result<()> {
        if let Some(limit_mb) = self.storage_quota_mb {
            let usage = self.maybe_refresh_usage(false)?;
            let limit_bytes = limit_mb.saturating_mul(BYTES_PER_MEGABYTE);
            if usage >= limit_bytes {
                return Err(EventError::TenantQuotaExceeded(format!(
                    "tenant '{}' reached storage quota ({limit_mb} MB)",
                    self.tenant_id
                )));
            }
        }
        Ok(())
    }

    pub fn list_aggregates(
        &self,
        token: &str,
        cursor: Option<&AggregateCursor>,
        take: Option<usize>,
        filter: Option<FilterExpr>,
        sort: Option<&[AggregateSort]>,
        scope: AggregateQueryScope,
    ) -> Result<(Vec<AggregateState>, Option<AggregateCursor>)> {
        let mut effective_take = take.unwrap_or(self.list_page_size);
        if effective_take == 0 {
            return Ok((Vec::new(), None));
        }
        if effective_take > self.page_limit {
            effective_take = self.page_limit;
        }

        let claims: JwtClaims = self
            .tokens()
            .authorize_action(token, "aggregate.read", None)?;
        let filter_ref = filter.as_ref();

        let mut transform = |aggregate: AggregateState| {
            if self.is_hidden_aggregate(&aggregate.aggregate_type) {
                return None;
            }
            let resource =
                Self::aggregate_resource(&aggregate.aggregate_type, &aggregate.aggregate_id);
            if !claims.allows_resource(Some(resource.as_str())) {
                return None;
            }
            let sanitized = self.sanitize_aggregate(aggregate);
            if let Some(expr) = filter_ref {
                if !expr.matches_aggregate(&sanitized) {
                    return None;
                }
            }
            Some(sanitized)
        };

        let timestamp_sort = sort.and_then(store::timestamp_sort_hint);
        if cursor.is_some() && sort.is_some() && timestamp_sort.is_none() {
            return Err(EventError::InvalidCursor(
                "cursor cannot be combined with sort directives".into(),
            ));
        }

        if let Some(keys) = sort {
            if let Some((kind, descending)) = timestamp_sort {
                if let Some(cursor) = cursor {
                    store::ensure_timestamp_cursor(cursor, kind, descending, scope)?;
                }
                let aggregates = self.store.aggregates_paginated_with_transform(
                    0,
                    Some(effective_take),
                    Some(keys),
                    scope,
                    cursor,
                    &mut transform,
                );
                return Ok((aggregates, None));
            }
            let aggregates = self.store.aggregates_paginated_with_transform(
                0,
                Some(effective_take),
                Some(keys),
                scope,
                None,
                &mut transform,
            );
            return Ok((aggregates, None));
        }

        self.store
            .aggregates_page_with_transform(cursor, effective_take, scope, &mut transform)
    }

    pub fn get_aggregate(
        &self,
        token: &str,
        aggregate_type: &str,
        aggregate_id: &str,
    ) -> Result<Option<AggregateState>> {
        let resource = Self::aggregate_resource(aggregate_type, aggregate_id);
        self.tokens()
            .authorize_action(token, "aggregate.read", Some(resource.as_str()))?;

        if self.is_hidden_aggregate(aggregate_type) {
            return Ok(None);
        }
        match self.store.get_aggregate_state(aggregate_type, aggregate_id) {
            Ok(aggregate) => Ok(Some(self.sanitize_aggregate(aggregate))),
            Err(EventError::AggregateNotFound) => Ok(None),
            Err(err) => Err(err),
        }
    }

    pub fn resolve_aggregate(
        &self,
        token: &str,
        aggregate_type: &str,
        aggregate_id: &str,
        depth: Option<usize>,
    ) -> Result<Option<ResolvedAggregate>> {
        let Some(root) = self.get_aggregate(token, aggregate_type, aggregate_id)? else {
            return Ok(None);
        };

        let schemas = self.schemas();
        let store = self.store();
        let tokens = self.tokens();
        let tenant = self.tenant_id().to_string();
        let hops = depth
            .unwrap_or(self.reference_default_depth)
            .min(self.reference_max_depth);

        let resolved = resolve_references(tenant.clone(), root, &schemas, hops, |reference| {
            if !reference.domain.eq_ignore_ascii_case(&tenant) {
                return Ok(ReferenceFetchOutcome {
                    status: ReferenceResolutionStatus::Forbidden,
                    aggregate: None,
                });
            }

            let resource = format!(
                "aggregate:{}:{}",
                reference.aggregate_type, reference.aggregate_id
            );
            match tokens.authorize_action(token, "aggregate.read", Some(resource.as_str())) {
                Ok(_) => {}
                Err(EventError::Unauthorized) => {
                    return Ok(ReferenceFetchOutcome {
                        status: ReferenceResolutionStatus::Forbidden,
                        aggregate: None,
                    });
                }
                Err(err) => return Err(err),
            }

            match store.get_aggregate_state(&reference.aggregate_type, &reference.aggregate_id) {
                Ok(aggregate) => Ok(ReferenceFetchOutcome {
                    status: ReferenceResolutionStatus::Ok,
                    aggregate: Some(self.sanitize_aggregate(aggregate)),
                }),
                Err(EventError::AggregateNotFound) => Ok(ReferenceFetchOutcome {
                    status: ReferenceResolutionStatus::NotFound,
                    aggregate: None,
                }),
                Err(err) => Err(err),
            }
        })?;

        Ok(Some(resolved))
    }

    pub fn select_aggregate_fields(
        &self,
        token: &str,
        aggregate_type: &str,
        aggregate_id: &str,
        fields: &[String],
    ) -> Result<Option<BTreeMap<String, Value>>> {
        let resource = Self::aggregate_resource(aggregate_type, aggregate_id);
        self.tokens()
            .authorize_action(token, "aggregate.read", Some(resource.as_str()))?;

        if self.is_hidden_aggregate(aggregate_type) {
            return Ok(None);
        }

        let aggregate = match self.store.get_aggregate_state(aggregate_type, aggregate_id) {
            Ok(aggregate) => self.sanitize_aggregate(aggregate),
            Err(EventError::AggregateNotFound) => return Ok(None),
            Err(err) => return Err(err),
        };

        let mut selections = BTreeMap::new();
        for field in fields {
            let trimmed = field.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value = select_state_field(&aggregate.state, trimmed).unwrap_or(Value::Null);
            selections.insert(trimmed.to_string(), value);
        }

        Ok(Some(selections))
    }

    pub fn list_events(
        &self,
        token: &str,
        aggregate_type: &str,
        aggregate_id: &str,
        cursor: Option<&EventCursor>,
        take: Option<usize>,
    ) -> Result<(Vec<EventRecord>, Option<EventCursor>)> {
        let resource = Self::aggregate_resource(aggregate_type, aggregate_id);
        self.tokens()
            .authorize_action(token, "aggregate.read", Some(resource.as_str()))?;

        if self.is_hidden_aggregate(aggregate_type) {
            return Err(EventError::AggregateNotFound);
        }

        let mut effective_take = take.unwrap_or(self.page_limit);
        if effective_take == 0 {
            return Ok((Vec::new(), None));
        }
        if effective_take > self.page_limit {
            effective_take = self.page_limit;
        }

        self.store.events_page(
            EventQueryScope::Aggregate {
                aggregate_type,
                aggregate_id,
            },
            EventArchiveScope::ActiveOnly,
            cursor,
            effective_take,
            None,
        )
    }

    pub fn list_referrers(
        &self,
        token: &str,
        aggregate_type: &str,
        aggregate_id: &str,
    ) -> Result<Vec<Referrer>> {
        ensure_snake_case("aggregate_type", aggregate_type)?;
        ensure_aggregate_id(aggregate_id)?;

        if self.is_hidden_aggregate(aggregate_type) {
            return Err(EventError::AggregateNotFound);
        }

        let resource = Self::aggregate_resource(aggregate_type, aggregate_id);
        let claims: JwtClaims =
            self.tokens()
                .authorize_action(token, "aggregate.read", Some(resource.as_str()))?;

        if self
            .store
            .aggregate_version(aggregate_type, aggregate_id)?
            .is_none()
        {
            return Err(EventError::AggregateNotFound);
        }

        let target = format!("{}#{}#{}", self.tenant_id(), aggregate_type, aggregate_id);
        let referrers = self.store.referrers_for(self.tenant_id(), &target)?;
        Ok(referrers
            .into_iter()
            .filter_map(|(agg_type, agg_id, path)| {
                if self.is_hidden_aggregate(&agg_type) {
                    return None;
                }
                let resource = Self::aggregate_resource(&agg_type, &agg_id);
                if !claims.allows_resource(Some(resource.as_str())) {
                    return None;
                }
                Some(Referrer {
                    aggregate_type: agg_type,
                    aggregate_id: agg_id,
                    path,
                })
            })
            .collect())
    }

    pub fn verify_aggregate(&self, aggregate_type: &str, aggregate_id: &str) -> Result<String> {
        if self.is_hidden_aggregate(aggregate_type) {
            return Err(EventError::AggregateNotFound);
        }
        self.store.verify(aggregate_type, aggregate_id)
    }

    pub fn create_aggregate(&self, input: CreateAggregateInput) -> Result<AggregateState> {
        let CreateAggregateInput {
            token,
            aggregate_type,
            aggregate_id,
            event_type,
            payload,
            metadata,
            note,
            publish: _,
        } = input;

        ensure_snake_case("aggregate_type", &aggregate_type)?;
        ensure_snake_case("event_type", &event_type)?;
        ensure_aggregate_id(&aggregate_id)?;
        if let Some(ref metadata) = metadata {
            ensure_metadata_extensions(metadata)?;
        }
        ensure_payload_size(&payload)?;

        let store = self.store();
        if store
            .aggregate_version(&aggregate_type, &aggregate_id)?
            .is_some()
        {
            return Err(EventError::SchemaViolation(format!(
                "aggregate {}::{} already exists",
                aggregate_type, aggregate_id
            )));
        }

        let resource = format!("aggregate:{}:{}", aggregate_type, aggregate_id);
        let claims =
            self.tokens()
                .authorize_action(&token, "aggregate.create", Some(resource.as_str()))?;

        if claims.actor_claims().is_none() {
            return Err(EventError::Unauthorized);
        }

        let append_input = AppendEventInput {
            token,
            aggregate_type: aggregate_type.clone(),
            aggregate_id: aggregate_id.clone(),
            event_type,
            payload: Some(payload),
            patch: None,
            metadata,
            note,
            publish: Vec::new(),
        };
        let _record = self.append_event_internal(append_input, true)?;

        let aggregate = store.get_aggregate_state(&aggregate_type, &aggregate_id)?;
        Ok(self.sanitize_aggregate(aggregate))
    }

    pub fn set_aggregate_archive(&self, input: SetAggregateArchiveInput) -> Result<AggregateState> {
        let SetAggregateArchiveInput {
            token,
            aggregate_type,
            aggregate_id,
            archived,
            note,
        } = input;

        ensure_snake_case("aggregate_type", &aggregate_type)?;
        ensure_aggregate_id(&aggregate_id)?;

        let resource = format!("aggregate:{}:{}", aggregate_type, aggregate_id);
        let claims =
            self.tokens()
                .authorize_action(&token, "aggregate.archive", Some(resource.as_str()))?;

        if claims.actor_claims().is_none() {
            return Err(EventError::Unauthorized);
        }

        let comment = normalize_optional_note(note);

        let store = self.store();
        if archived {
            let referrers = self.inbound_referrers_for(&aggregate_type, &aggregate_id)?;
            let mut blockers = Vec::new();
            let mut nullify = Vec::new();
            for (tenant, agg, id, path) in referrers {
                if tenant != self.tenant_id() {
                    blockers.push(format!("{tenant}:{agg}:{id} ({path})"));
                    continue;
                }
                let cascade = self
                    .schemas
                    .reference_rules_for_path(&agg, &path)
                    .map(|rules| rules.cascade)
                    .unwrap_or_default();
                match cascade {
                    ReferenceCascade::Nullify => nullify.push((tenant, agg, id, path)),
                    _ => blockers.push(format!("{tenant}:{agg}:{id} ({path})")),
                }
            }

            if !blockers.is_empty() {
                return Err(EventError::SchemaViolation(format!(
                    "aggregate {}:{} cannot be archived; referenced by {}",
                    aggregate_type,
                    aggregate_id,
                    blockers.into_iter().take(5).collect::<Vec<_>>().join(", ")
                )));
            }

            let target = format!("{}#{}#{}", self.tenant_id(), aggregate_type, aggregate_id);
            self.nullify_referrers(&nullify, &target)?;
        }
        let state = store.set_archive(&aggregate_type, &aggregate_id, archived, comment)?;
        Ok(self.sanitize_aggregate(state))
    }

    pub fn create_snapshot(
        &self,
        token: &str,
        aggregate_type: &str,
        aggregate_id: &str,
        comment: Option<String>,
    ) -> Result<crate::store::SnapshotRecord> {
        ensure_snake_case("aggregate_type", aggregate_type)?;
        ensure_aggregate_id(aggregate_id)?;

        if self.is_hidden_aggregate(aggregate_type) {
            return Err(EventError::AggregateNotFound);
        }

        self.enforce_storage_quota()?;

        let resource = Self::aggregate_resource(aggregate_type, aggregate_id);
        self.tokens()
            .authorize_action(token, "aggregate.append", Some(resource.as_str()))?;

        self.store
            .create_snapshot(aggregate_type, aggregate_id, comment)
    }

    pub fn list_snapshots(
        &self,
        token: &str,
        aggregate_type: Option<&str>,
        aggregate_id: Option<&str>,
        version: Option<u64>,
    ) -> Result<Vec<SnapshotRecord>> {
        if let Some(value) = aggregate_type {
            ensure_snake_case("aggregate_type", value)?;
        }
        if let Some(value) = aggregate_id {
            ensure_aggregate_id(value)?;
        }
        if aggregate_id.is_some() && aggregate_type.is_none() {
            return Err(EventError::InvalidSchema(
                "aggregate type is required when providing aggregate id".into(),
            ));
        }

        let resource = aggregate_type
            .zip(aggregate_id)
            .map(|(agg, id)| Self::aggregate_resource(agg, id));
        if let Some(resource) = resource.as_deref() {
            self.tokens()
                .authorize_action(token, "aggregate.read", Some(resource))?;
        } else {
            self.tokens()
                .authorize_action(token, "aggregate.read", None)?;
        }

        let snapshots = self
            .store
            .list_snapshots(aggregate_type, aggregate_id, version)?;
        Ok(snapshots
            .into_iter()
            .filter_map(|mut snapshot| {
                if self.is_hidden_aggregate(&snapshot.aggregate_type) {
                    return None;
                }
                snapshot.state =
                    self.filter_state_map(&snapshot.aggregate_type, snapshot.state.clone());
                Some(snapshot)
            })
            .collect())
    }

    pub fn find_snapshot_by_id(
        &self,
        token: &str,
        snapshot_id: SnowflakeId,
    ) -> Result<Option<SnapshotRecord>> {
        let Some(mut snapshot) = self.store.find_snapshot_by_id(snapshot_id)? else {
            return Ok(None);
        };

        if self.is_hidden_aggregate(&snapshot.aggregate_type) {
            return Ok(None);
        }

        let resource = Self::aggregate_resource(&snapshot.aggregate_type, &snapshot.aggregate_id);
        self.tokens()
            .authorize_action(token, "aggregate.read", Some(resource.as_str()))?;
        snapshot.state = self.filter_state_map(&snapshot.aggregate_type, snapshot.state.clone());
        Ok(Some(snapshot))
    }

    pub fn append_event(&self, input: AppendEventInput) -> Result<EventRecord> {
        let allow_new = input.payload.is_some();
        self.append_event_internal(input, allow_new)
    }

    fn append_event_internal(
        &self,
        input: AppendEventInput,
        allow_new_aggregate: bool,
    ) -> Result<EventRecord> {
        let AppendEventInput {
            token,
            aggregate_type,
            aggregate_id,
            event_type,
            payload,
            patch,
            metadata,
            note,
            publish: _,
        } = input;

        enum EventBody {
            Payload(Value),
            Patch(Value),
        }

        let event_body = match (payload, patch) {
            (Some(payload), None) => EventBody::Payload(payload),
            (None, Some(patch_spec)) => EventBody::Patch(patch_spec),
            (Some(_), Some(_)) => {
                return Err(EventError::InvalidSchema(
                    "payload_json and patch_json cannot both be provided".into(),
                ));
            }
            (None, None) => {
                return Err(EventError::InvalidSchema(
                    "payload_json must be provided when patch_json is absent".into(),
                ));
            }
        };

        ensure_snake_case("aggregate_type", &aggregate_type)?;
        ensure_snake_case("event_type", &event_type)?;
        ensure_aggregate_id(&aggregate_id)?;
        if let Some(ref metadata) = metadata {
            ensure_metadata_extensions(metadata)?;
        }

        self.enforce_storage_quota()?;

        let aggregate_version = self
            .store
            .aggregate_version(&aggregate_type, &aggregate_id)?;
        let is_new_aggregate = match aggregate_version {
            Some(version) if version > 0 => false,
            Some(_) | None => true,
        };
        let is_patch_event = matches!(&event_body, EventBody::Patch(_));
        if (!allow_new_aggregate || is_patch_event) && is_new_aggregate {
            return Err(EventError::AggregateNotFound);
        }
        ensure_first_event_rule(is_new_aggregate, &event_type)?;

        let resource = format!("aggregate:{}:{}", aggregate_type, aggregate_id);
        let claims =
            self.tokens()
                .authorize_action(&token, "aggregate.append", Some(resource.as_str()))?;

        let mut reference_targets = Vec::new();
        let mut effective_payload = match event_body {
            EventBody::Payload(payload) => {
                ensure_payload_size(&payload)?;
                payload
            }
            EventBody::Patch(patch_spec) => {
                if !patch_spec.is_array() {
                    return Err(EventError::InvalidSchema(
                        "patch_json must be an array of operations".into(),
                    ));
                }
                let payload = self.store.prepare_payload_from_patch(
                    &aggregate_type,
                    &aggregate_id,
                    &patch_spec,
                )?;
                ensure_payload_size(&payload)?;
                payload
            }
        };

        let schemas = self.schemas();
        let schema_present = match schemas.get(&aggregate_type) {
            Ok(_) => true,
            Err(EventError::SchemaNotFound) => false,
            Err(err) => return Err(err),
        };

        if !schema_present && self.restrict.requires_declared_schema() {
            return Err(EventError::SchemaViolation(
                restrict::strict_mode_missing_schema_message(&aggregate_type),
            ));
        }

        if schema_present {
            schemas.validate_event(&aggregate_type, &event_type, &effective_payload)?;
            let reference_context = ReferenceContext {
                domain: self.tenant_id(),
                aggregate_type: &aggregate_type,
            };
            let store = self.store();
            let tokens = self.tokens();
            let tenant = self.tenant_id().to_string();
            let mut resolver = move |reference: &AggregateReference,
                                     _integrity: ReferenceIntegrity|
                  -> Result<ReferenceResolutionStatus> {
                if !reference.domain.eq_ignore_ascii_case(tenant.as_str()) {
                    return Ok(ReferenceResolutionStatus::Forbidden);
                }
                let resource = format!(
                    "aggregate:{}:{}",
                    reference.aggregate_type, reference.aggregate_id
                );
                match tokens.authorize_action(&token, "aggregate.read", Some(resource.as_str())) {
                    Ok(_) => {}
                    Err(EventError::Unauthorized) => {
                        return Ok(ReferenceResolutionStatus::Forbidden);
                    }
                    Err(err) => return Err(err),
                }
                match store.aggregate_version(&reference.aggregate_type, &reference.aggregate_id) {
                    Ok(Some(_)) => Ok(ReferenceResolutionStatus::Ok),
                    Ok(None) => Ok(ReferenceResolutionStatus::NotFound),
                    Err(err) => Err(err),
                }
            };
            let (normalized_payload, outcomes) = schemas.normalize_references(
                &aggregate_type,
                effective_payload,
                reference_context,
                &mut resolver,
            )?;
            effective_payload = normalized_payload;
            ensure_payload_size(&effective_payload)?;
            reference_targets = outcomes
                .iter()
                .map(|outcome| (outcome.reference.to_canonical(), outcome.path.clone()))
                .collect();
        }

        let issued_by: Option<ActorClaims> = claims.actor_claims();
        if issued_by.is_none() {
            return Err(EventError::Unauthorized);
        }

        let record = self.store.append(AppendEvent {
            aggregate_type,
            aggregate_id,
            event_type,
            payload: effective_payload,
            metadata,
            issued_by,
            note,
            tenant: self.tenant_id.clone(),
            reference_targets,
        })?;
        self.note_storage_mutation()?;
        Ok(record)
    }

    fn aggregate_resource(aggregate_type: &str, aggregate_id: &str) -> String {
        format!("aggregate:{}:{}", aggregate_type, aggregate_id)
    }

    fn filter_state_map(
        &self,
        aggregate_type: &str,
        mut state: BTreeMap<String, String>,
    ) -> BTreeMap<String, String> {
        if let Ok(schema) = self.schemas.get(aggregate_type) {
            if !schema.hidden_fields.is_empty() {
                state.retain(|key, _| !schema.hidden_fields.contains(key));
            }
        }
        state
    }

    fn maybe_refresh_usage(&self, force: bool) -> Result<u64> {
        let (needs_refresh, current_bytes) = {
            let cache = self.usage_cache.lock();
            let refresh = force
                || !cache.initialized
                || cache.events_since_refresh >= STORAGE_USAGE_REFRESH_EVENTS
                || cache.last_refresh.elapsed() >= STORAGE_USAGE_REFRESH_INTERVAL
                || cache.dirty;
            (refresh, cache.bytes)
        };
        if needs_refresh {
            self.refresh_usage_from_store()
        } else {
            Ok(current_bytes)
        }
    }

    fn refresh_usage_from_store(&self) -> Result<u64> {
        let store = self.store();
        let usage = store.storage_usage_bytes()?;
        self.assignments
            .update_storage_usage_bytes(&self.tenant_id, usage)?;
        let mut cache = self.usage_cache.lock();
        cache.bytes = usage;
        cache.last_refresh = Instant::now();
        cache.events_since_refresh = 0;
        cache.initialized = true;
        cache.dirty = false;
        Ok(usage)
    }

    fn note_storage_mutation(&self) -> Result<()> {
        let should_refresh = {
            let mut cache = self.usage_cache.lock();
            if !cache.initialized {
                true
            } else {
                cache.events_since_refresh = cache.events_since_refresh.saturating_add(1);
                if cache.events_since_refresh >= STORAGE_USAGE_REFRESH_EVENTS {
                    true
                } else {
                    cache.dirty = true;
                    false
                }
            }
        };
        if should_refresh {
            self.refresh_usage_from_store()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AppendEventInput {
    pub token: String,
    pub aggregate_type: String,
    pub aggregate_id: String,
    pub event_type: String,
    pub payload: Option<Value>,
    pub patch: Option<Value>,
    pub metadata: Option<Value>,
    pub note: Option<String>,
    pub publish: Vec<PublishTarget>,
}

#[derive(Debug, Clone)]
pub struct CreateAggregateInput {
    pub token: String,
    pub aggregate_type: String,
    pub aggregate_id: String,
    pub event_type: String,
    pub payload: Value,
    pub metadata: Option<Value>,
    pub note: Option<String>,
    pub publish: Vec<PublishTarget>,
}

#[derive(Debug, Clone)]
pub struct SetAggregateArchiveInput {
    pub token: String,
    pub aggregate_type: String,
    pub aggregate_id: String,
    pub archived: bool,
    pub note: Option<String>,
}

pub(crate) fn normalize_optional_note(note: Option<String>) -> Option<String> {
    note.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn json_pointer_from_path(path: &str) -> String {
    if path.is_empty() {
        return "/".to_string();
    }
    format!("/{}", path.replace('.', "/"))
}
