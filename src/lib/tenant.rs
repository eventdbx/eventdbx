use std::{collections::HashMap, fs, sync::Arc};

use parking_lot::Mutex;

use crate::{
    config::{Config, DEFAULT_DOMAIN_NAME},
    encryption::Encryptor,
    error::{EventError, Result},
    restrict::RestrictMode,
    schema::SchemaManager,
    service::CoreContext,
    store::EventStore,
    tenant_store::{SHARD_PREFIX, TenantAssignmentStore, compute_default_shard, format_shard_id},
    token::TokenManager,
};

pub trait CoreProvider: Send + Sync {
    fn core_for(&self, tenant: &str) -> Result<CoreContext>;
    fn invalidate_tenant(&self, tenant: &str);
}

pub struct TenantRegistry {
    config: Config,
    encryption: Option<Encryptor>,
    tokens: Arc<TokenManager>,
    restrict: RestrictMode,
    list_page_size: usize,
    page_limit: usize,
    snowflake_worker_id: u16,
    shard_count: u16,
    multi_tenant: bool,
    assignments: Arc<TenantAssignmentStore>,
    cache: Mutex<HashMap<String, Arc<CoreContext>>>,
}

impl TenantRegistry {
    pub fn new(
        config: Config,
        tokens: Arc<TokenManager>,
        encryption: Option<Encryptor>,
    ) -> Result<Self> {
        let shard_count = config.shard_count();
        let assignments = Arc::new(TenantAssignmentStore::open(config.tenant_meta_path())?);
        Ok(Self {
            restrict: config.restrict,
            list_page_size: config.list_page_size,
            page_limit: config.page_limit,
            snowflake_worker_id: config.snowflake_worker_id,
            shard_count,
            multi_tenant: config.multi_tenant(),
            assignments,
            config,
            encryption,
            tokens,
            cache: Mutex::new(HashMap::new()),
        })
    }

    fn ensure_tenant_context(&self, tenant: &str) -> Result<Arc<CoreContext>> {
        let normalized = normalize_tenant_id(tenant)?;
        {
            let cache = self.cache.lock();
            if let Some(existing) = cache.get(&normalized) {
                return Ok(Arc::clone(existing));
            }
        }

        let context = self.build_core_context(&normalized)?;
        let mut cache = self.cache.lock();
        let entry = cache
            .entry(normalized)
            .or_insert_with(|| Arc::clone(&context));
        Ok(Arc::clone(entry))
    }

    fn drop_cached_tenant(&self, tenant: &str) {
        let mut cache = self.cache.lock();
        cache.remove(tenant);
    }

    fn build_core_context(&self, tenant: &str) -> Result<Arc<CoreContext>> {
        let quota = self.assignments.quota_for(tenant)?;
        if self.multi_tenant {
            let shard = self.resolve_shard(tenant)?;
            self.build_sharded_context(tenant, &shard, quota)
        } else {
            self.build_legacy_context(tenant, quota)
        }
    }

    fn build_legacy_context(&self, tenant: &str, quota: Option<u64>) -> Result<Arc<CoreContext>> {
        let tenant_dir = self.config.domain_data_dir_for(tenant);
        fs::create_dir_all(&tenant_dir)?;

        let mut tenant_config = self.config.clone();
        tenant_config.domain = tenant.to_string();
        tenant_config.ensure_data_dir()?;

        let store = Arc::new(EventStore::open(
            tenant_config.event_store_path(),
            self.encryption.clone(),
            self.snowflake_worker_id,
        )?);
        let schemas = Arc::new(SchemaManager::load(tenant_config.schema_store_path())?);
        let usage = self
            .assignments
            .ensure_storage_usage_bytes(tenant, || store.storage_usage_bytes())?;

        Ok(Arc::new(CoreContext::new(
            Arc::clone(&self.tokens),
            schemas,
            store,
            self.restrict,
            self.list_page_size,
            self.page_limit,
            tenant.to_string(),
            quota,
            Some(usage),
            Arc::clone(&self.assignments),
            self.config.reference_default_depth,
            self.config.reference_max_depth,
        )))
    }

    fn build_sharded_context(
        &self,
        tenant: &str,
        shard: &str,
        quota: Option<u64>,
    ) -> Result<Arc<CoreContext>> {
        let tenant_dir = self.config.tenant_shard_dir(shard, tenant);
        fs::create_dir_all(&tenant_dir)?;

        let event_store_path = tenant_dir.join("event_store");
        let schemas_path = tenant_dir.join("schemas.json");

        let store = Arc::new(EventStore::open(
            event_store_path,
            self.encryption.clone(),
            self.snowflake_worker_id,
        )?);
        let schemas = Arc::new(SchemaManager::load(schemas_path)?);
        let usage = self
            .assignments
            .ensure_storage_usage_bytes(tenant, || store.storage_usage_bytes())?;

        Ok(Arc::new(CoreContext::new(
            Arc::clone(&self.tokens),
            schemas,
            store,
            self.restrict,
            self.list_page_size,
            self.page_limit,
            tenant.to_string(),
            quota,
            Some(usage),
            Arc::clone(&self.assignments),
            self.config.reference_default_depth,
            self.config.reference_max_depth,
        )))
    }

    fn resolve_shard(&self, tenant: &str) -> Result<String> {
        if let Some(explicit) = self.assignments.shard_for(tenant)? {
            return Ok(explicit);
        }
        Ok(compute_default_shard(tenant, self.shard_count))
    }
}

impl CoreProvider for TenantRegistry {
    fn core_for(&self, tenant: &str) -> Result<CoreContext> {
        let context = self.ensure_tenant_context(tenant)?;
        Ok(context.as_ref().clone())
    }

    fn invalidate_tenant(&self, tenant: &str) {
        match normalize_tenant_id(tenant) {
            Ok(normalized) => self.drop_cached_tenant(&normalized),
            Err(_) => self.drop_cached_tenant(tenant),
        }
    }
}

pub fn normalize_tenant_id(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(EventError::Config("tenant id cannot be empty".to_string()));
    }

    if trimmed.eq_ignore_ascii_case(DEFAULT_DOMAIN_NAME) {
        return Ok(DEFAULT_DOMAIN_NAME.to_string());
    }

    let lower = trimmed.to_ascii_lowercase();
    if !matches!(lower.chars().next(), Some(ch) if ch.is_ascii_alphanumeric()) {
        return Err(EventError::Config(
            "tenant id must begin with an ASCII letter or digit".to_string(),
        ));
    }
    if !lower
        .chars()
        .skip(1)
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        return Err(EventError::Config(
            "tenant id may only contain letters, numbers, '-' or '_'".to_string(),
        ));
    }
    Ok(lower)
}

pub fn normalize_tenant_list<I, S>(values: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut cleaned: Vec<String> = values
        .into_iter()
        .map(|value| value.as_ref().trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect();
    cleaned.sort();
    cleaned.dedup();
    cleaned
}

pub fn normalize_shard_id(raw: &str, shard_count: u16) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(EventError::Config("shard id must not be empty".to_string()));
    }
    let normalized = trimmed.to_ascii_lowercase();
    let value = if let Some(suffix) = normalized.strip_prefix(SHARD_PREFIX) {
        suffix
    } else {
        normalized.as_str()
    };
    let index: u16 = value
        .parse()
        .map_err(|_| EventError::Config(format!("invalid shard id '{}'", raw)))?;
    let max = shard_count.max(1);
    if index >= max {
        return Err(EventError::Config(format!(
            "shard '{}' exceeds configured shard-count ({})",
            raw, max
        )));
    }
    Ok(format_shard_id(index))
}

pub struct StaticCoreProvider {
    core: CoreContext,
}

impl StaticCoreProvider {
    pub fn new(core: CoreContext) -> Self {
        Self { core }
    }
}

impl CoreProvider for StaticCoreProvider {
    fn core_for(&self, _tenant: &str) -> Result<CoreContext> {
        Ok(self.core.clone())
    }

    fn invalidate_tenant(&self, _tenant: &str) {}
}
