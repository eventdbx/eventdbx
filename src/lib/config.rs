use std::{
    collections::BTreeMap,
    env, fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use chrono::{DateTime, Duration, Utc};
use rand_core::{OsRng, RngCore};
use ring::{
    rand::SystemRandom,
    signature::{ED25519_PUBLIC_KEY_LEN, Ed25519KeyPair, KeyPair},
};
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};

use super::{
    encryption::Encryptor,
    error::{EventError, Result},
    restrict::{self, RestrictMode},
    store::EventStore,
    tenant_store::{TenantAssignmentStore, compute_default_shard},
    token::JwtManagerConfig,
};

pub const DEFAULT_PORT: u16 = 7070;
pub const DEFAULT_SOCKET_PORT: u16 = 6363;
pub const DEFAULT_CACHE_THRESHOLD: usize = 10_000;
pub const DEFAULT_DOMAIN_NAME: &str = "default";
pub const WORKSPACE_DIR_NAME: &str = ".dbx";

const CONFIG_FILE_NAME: &str = "config.toml";

const ENV_AUTO_INIT_WORKSPACE: &str = "EVENTDBX_AUTO_INIT";
const ENV_DATA_ENCRYPTION_KEY: &str = "EVENTDBX_DATA_ENCRYPTION_KEY";
const ENV_AUTH_PRIVATE_KEY: &str = "EVENTDBX_AUTH_PRIVATE_KEY";
const ENV_AUTH_PUBLIC_KEY: &str = "EVENTDBX_AUTH_PUBLIC_KEY";
const ENV_AUTH_KEY_ID: &str = "EVENTDBX_AUTH_KEY_ID";
const DEFAULT_NO_NOISE: bool = false;
fn secret_from_env(vars: &[&str]) -> Option<String> {
    for key in vars {
        if let Ok(value) = env::var(key) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn normalize_non_empty(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    #[serde(default = "default_jwt_issuer")]
    pub issuer: String,
    #[serde(default = "default_jwt_audience")]
    pub audience: String,
    #[serde(default)]
    pub private_key: Option<String>,
    #[serde(default)]
    pub public_key: Option<String>,
    #[serde(default)]
    pub key_id: Option<String>,
    #[serde(default = "default_jwt_ttl_secs")]
    pub default_ttl_secs: u64,
    #[serde(default = "default_jwt_clock_skew_secs")]
    pub clock_skew_secs: u64,
}

impl Default for AuthConfig {
    fn default() -> Self {
        let (private_key, public_key) = generate_jwt_keypair()
            .unwrap_or_else(|err| panic!("failed to generate default jwt key pair: {err}"));
        Self {
            issuer: default_jwt_issuer(),
            audience: default_jwt_audience(),
            private_key: Some(private_key),
            public_key: Some(public_key),
            key_id: Some(format!("key-{}", Utc::now().format("%Y%m%d%H%M%S"))),
            default_ttl_secs: default_jwt_ttl_secs(),
            clock_skew_secs: default_jwt_clock_skew_secs(),
        }
    }
}

impl AuthConfig {
    pub fn resolved_private_key(&self) -> Option<String> {
        secret_from_env(&[ENV_AUTH_PRIVATE_KEY]).or_else(|| {
            self.private_key
                .as_ref()
                .and_then(|value| normalize_non_empty(value))
        })
    }

    pub fn resolved_public_key(&self) -> Option<String> {
        secret_from_env(&[ENV_AUTH_PUBLIC_KEY]).or_else(|| {
            self.public_key
                .as_ref()
                .and_then(|value| normalize_non_empty(value))
        })
    }

    pub fn resolved_key_id(&self) -> Option<String> {
        secret_from_env(&[ENV_AUTH_KEY_ID]).or_else(|| {
            self.key_id
                .as_ref()
                .and_then(|value| normalize_non_empty(value))
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginQueueConfig {
    #[serde(default = "default_plugin_queue_prune")]
    pub prune: PluginQueuePruneConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantRoutingConfig {
    #[serde(default)]
    pub multi_tenant: bool,
    #[serde(default = "default_tenant_shard_count")]
    pub shard_count: u16,
    #[serde(default)]
    pub shard_map_path: Option<PathBuf>,
}

impl Default for TenantRoutingConfig {
    fn default() -> Self {
        Self {
            multi_tenant: false,
            shard_count: default_tenant_shard_count(),
            shard_map_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginQueuePruneConfig {
    #[serde(default = "default_queue_prune_done_ttl_secs")]
    pub done_ttl_secs: u64,
    #[serde(default = "default_queue_prune_interval_secs")]
    pub interval_secs: u64,
    #[serde(default)]
    pub max_done_jobs: Option<usize>,
}

impl Default for PluginQueuePruneConfig {
    fn default() -> Self {
        Self {
            done_ttl_secs: default_queue_prune_done_ttl_secs(),
            interval_secs: default_queue_prune_interval_secs(),
            max_done_jobs: None,
        }
    }
}

fn default_tenant_shard_count() -> u16 {
    16
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub port: u16,
    pub data_dir: PathBuf,
    #[serde(default = "default_domain")]
    pub domain: String,
    #[serde(default = "default_cache_threshold", alias = "memory_threshold")]
    pub cache_threshold: usize,
    #[serde(default)]
    pub snapshot_threshold: Option<u64>,
    pub data_encryption_key: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(
        default = "default_restrict",
        deserialize_with = "deserialize_restrict_mode"
    )]
    pub restrict: RestrictMode,
    #[serde(default, skip_serializing)]
    pub plugins: Vec<PluginDefinition>,
    #[serde(default = "default_list_page_size")]
    pub list_page_size: usize,
    #[serde(default = "default_page_limit")]
    pub page_limit: usize,
    #[serde(default = "default_plugin_max_attempts")]
    pub plugin_max_attempts: u32,
    #[serde(default)]
    pub plugin_queue: PluginQueueConfig,
    #[serde(default)]
    pub socket: SocketConfig,
    #[serde(default)]
    pub tenants: TenantRoutingConfig,
    #[serde(default = "default_verbose_responses")]
    pub verbose_responses: bool,
    #[serde(default = "default_reference_default_depth")]
    pub reference_default_depth: usize,
    #[serde(default = "default_reference_max_depth")]
    pub reference_max_depth: usize,
    #[serde(default)]
    pub auth: AuthConfig,
    #[serde(default = "default_snowflake_worker_id")]
    pub snowflake_worker_id: u16,
    /// Disable Noise encryption for the control channel (Cap'n Proto framing only).
    /// WARNING: Sends control protocol messages in plaintext; only safe on trusted networks
    /// such as localhost or other secured private links.
    #[serde(default = "default_no_noise")]
    pub no_noise: bool,
}

impl Default for Config {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            port: DEFAULT_PORT,
            data_dir: default_data_dir(),
            domain: default_domain(),
            cache_threshold: default_cache_threshold(),
            snapshot_threshold: None,
            data_encryption_key: Some(generate_data_encryption_key()),
            created_at: now,
            updated_at: now,
            restrict: default_restrict(),
            plugins: Vec::new(),
            list_page_size: default_list_page_size(),
            page_limit: default_page_limit(),
            plugin_max_attempts: default_plugin_max_attempts(),
            plugin_queue: PluginQueueConfig::default(),
            socket: SocketConfig::default(),
            tenants: TenantRoutingConfig::default(),
            verbose_responses: default_verbose_responses(),
            reference_default_depth: default_reference_default_depth(),
            reference_max_depth: default_reference_max_depth(),
            auth: AuthConfig::default(),
            snowflake_worker_id: default_snowflake_worker_id(),
            no_noise: default_no_noise(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ConfigUpdate {
    pub port: Option<u16>,
    pub data_dir: Option<PathBuf>,
    pub cache_threshold: Option<usize>,
    pub snapshot_threshold: Option<Option<u64>>,
    pub data_encryption_key: Option<String>,
    pub restrict: Option<RestrictMode>,
    pub list_page_size: Option<usize>,
    pub page_limit: Option<usize>,
    pub verbose_responses: Option<bool>,
    pub reference_default_depth: Option<usize>,
    pub reference_max_depth: Option<usize>,
    pub plugin_max_attempts: Option<u32>,
    pub socket: Option<SocketConfigUpdate>,
    pub tenants: Option<TenantRoutingConfigUpdate>,
    pub snowflake_worker_id: Option<u16>,
    pub no_noise: Option<bool>,
}

#[derive(Debug, Clone, Default)]
pub struct TenantRoutingConfigUpdate {
    pub multi_tenant: Option<bool>,
    pub shard_count: Option<u16>,
    pub shard_map_path: Option<PathBuf>,
}

pub fn default_config_path() -> Result<PathBuf> {
    find_workspace_config_path(None)
}

pub fn init_workspace(path: Option<PathBuf>) -> Result<(Config, PathBuf, bool)> {
    let config_path = init_config_path(path)?;
    if config_path.exists() {
        let (config, path) = load_config(config_path)?;
        return Ok((config, path, false));
    }

    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let data_dir = config_path
        .parent()
        .ok_or_else(|| EventError::Config("config path must have a parent directory".to_string()))?
        .to_path_buf();

    let mut cfg = Config {
        data_dir,
        ..Config::default()
    };
    cfg.ensure_single_tenant_layout()?;
    let _ = cfg.ensure_encryption_key();
    cfg.ensure_auth_keys()?;
    cfg.ensure_data_dir()?;
    let _ = cfg.migrate_plugins()?;
    let _ = cfg.migrate_plugin_config_to_root()?;
    let _ = cfg.migrate_plugin_queue_to_root()?;
    let _ = cfg.migrate_plugin_runtime_to_root()?;
    cfg.save(&config_path)?;
    Ok((cfg, config_path, true))
}

pub fn load_or_default(path: Option<PathBuf>) -> Result<(Config, PathBuf)> {
    let auto_init = auto_init_workspace_enabled();
    let config_path = resolve_config_path_for_load(path, auto_init)?;
    maybe_auto_init_workspace(config_path.as_path(), auto_init)?;
    load_config(config_path)
}

fn resolve_config_path_for_load(path: Option<PathBuf>, auto_init: bool) -> Result<PathBuf> {
    match path {
        Some(path) => normalize_path(path),
        None if auto_init => init_config_path(None),
        None => default_config_path(),
    }
}

fn auto_init_workspace_enabled() -> bool {
    matches!(
        env::var(ENV_AUTO_INIT_WORKSPACE),
        Ok(value)
            if matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
    )
}

fn maybe_auto_init_workspace(config_path: &Path, auto_init: bool) -> Result<()> {
    if !auto_init || config_path.exists() {
        return Ok(());
    }

    let (_, resolved_path, created) = init_workspace(Some(config_path.to_path_buf()))?;
    if created {
        tracing::info!(
            "auto-initialized EventDBX workspace at {}",
            resolved_path.display()
        );
    }

    Ok(())
}

impl Config {
    pub fn resolved_data_encryption_key(&self) -> Option<String> {
        secret_from_env(&[ENV_DATA_ENCRYPTION_KEY]).or_else(|| {
            self.data_encryption_key
                .as_ref()
                .and_then(|value| normalize_non_empty(value))
        })
    }

    pub fn migrate_plugins(&mut self) -> Result<bool> {
        if self.plugins.is_empty() {
            return Ok(false);
        }

        let existing = self.load_plugins()?;
        if existing.is_empty() {
            self.save_plugins(&self.plugins)?;
        }

        self.plugins.clear();
        self.updated_at = Utc::now();
        Ok(true)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let contents = toml::to_string_pretty(self)?;
        fs::write(path, contents)?;
        Ok(())
    }

    pub fn apply_update(&mut self, update: ConfigUpdate) {
        if let Some(port) = update.port {
            self.port = port;
        }
        if let Some(dir) = update.data_dir {
            self.data_dir = dir;
        }
        if let Some(threshold) = update.cache_threshold {
            self.cache_threshold = threshold;
        }
        if let Some(snapshot_threshold) = update.snapshot_threshold {
            self.snapshot_threshold = snapshot_threshold;
        }
        if let Some(dek) = update.data_encryption_key {
            self.data_encryption_key = Some(dek);
        }
        if let Some(restrict) = update.restrict {
            self.restrict = restrict;
        }
        if let Some(list_page_size) = update.list_page_size {
            self.list_page_size = list_page_size;
        }
        if let Some(page_limit) = update.page_limit {
            self.page_limit = page_limit;
        }
        if let Some(verbose_responses) = update.verbose_responses {
            self.verbose_responses = verbose_responses;
        }
        if let Some(reference_default_depth) = update.reference_default_depth {
            self.reference_default_depth =
                reference_default_depth.min(default_reference_max_depth());
        }
        if let Some(reference_max_depth) = update.reference_max_depth {
            self.reference_max_depth = reference_max_depth.max(self.reference_default_depth);
        }
        if let Some(max_attempts) = update.plugin_max_attempts {
            self.plugin_max_attempts = max_attempts.max(1);
        }
        if let Some(worker_id) = update.snowflake_worker_id {
            self.snowflake_worker_id = worker_id;
        }
        if let Some(no_noise) = update.no_noise {
            self.no_noise = no_noise;
        }
        if let Some(socket) = update.socket {
            if let Some(bind_addr) = socket.bind_addr {
                self.socket.bind_addr = bind_addr;
            }
        }
        if let Some(tenant_update) = update.tenants {
            if let Some(enabled) = tenant_update.multi_tenant {
                self.tenants.multi_tenant = enabled;
            }
            if let Some(shards) = tenant_update.shard_count {
                self.tenants.shard_count = shards.max(1);
            }
            if let Some(path) = tenant_update.shard_map_path {
                self.tenants.shard_map_path = Some(path);
            }
        }
        self.updated_at = Utc::now();
    }

    pub fn ensure_data_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.data_dir)?;
        fs::create_dir_all(self.domain_data_dir().as_path())?;
        if self.multi_tenant() {
            fs::create_dir_all(self.shards_root())?;
        }
        self.ensure_event_store_initialized()?;
        Ok(())
    }

    pub fn ensure_event_store_initialized(&self) -> Result<()> {
        let path = self.event_store_path();
        if path.join("CURRENT").exists() {
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let _store = EventStore::open(path, self.encryption_key()?, self.snowflake_worker_id)?;
        Ok(())
    }

    pub fn active_domain(&self) -> &str {
        DEFAULT_DOMAIN_NAME
    }

    pub fn is_default_domain(&self) -> bool {
        true
    }

    pub fn domains_root(&self) -> PathBuf {
        self.data_dir.join("domains")
    }

    pub fn domain_data_dir(&self) -> PathBuf {
        self.domain_data_dir_for(DEFAULT_DOMAIN_NAME)
    }

    pub fn domain_data_dir_for(&self, domain: &str) -> PathBuf {
        let tenant = domain.to_ascii_lowercase();
        if let Some(shard) = self.lookup_assigned_shard(&tenant) {
            return self.tenant_shard_dir(&shard, &tenant);
        }
        let shard = compute_default_shard(&tenant, self.shard_count());
        self.tenant_shard_dir(&shard, &tenant)
    }

    pub fn shards_root(&self) -> PathBuf {
        self.data_dir.join("shards")
    }

    pub fn tenant_shard_dir(&self, shard: &str, tenant: &str) -> PathBuf {
        self.shards_root().join(shard).join("tenants").join(tenant)
    }

    pub fn tenant_meta_path(&self) -> PathBuf {
        if let Some(path) = &self.tenants.shard_map_path {
            path.clone()
        } else {
            self.data_dir.join("tenant_meta")
        }
    }

    pub fn multi_tenant(&self) -> bool {
        false
    }

    pub fn shard_count(&self) -> u16 {
        self.tenants.shard_count.max(1)
    }

    pub fn shard_map_path(&self) -> Option<PathBuf> {
        self.tenants.shard_map_path.clone()
    }

    fn lookup_assigned_shard(&self, tenant: &str) -> Option<String> {
        let path = self.tenant_meta_path();
        let store = TenantAssignmentStore::open_read_only(path).ok()?;
        store.shard_for(tenant).ok()?
    }

    pub fn ensure_encryption_key(&mut self) -> bool {
        if self.resolved_data_encryption_key().is_some() {
            return false;
        }

        let encoded = generate_data_encryption_key();
        self.data_encryption_key = Some(encoded);
        self.updated_at = Utc::now();
        true
    }

    pub fn ensure_auth_keys(&mut self) -> Result<bool> {
        if secret_from_env(&[ENV_AUTH_PRIVATE_KEY]).is_some()
            || secret_from_env(&[ENV_AUTH_PUBLIC_KEY]).is_some()
        {
            return Ok(false);
        }

        let needs_private = self
            .auth
            .private_key
            .as_ref()
            .map(|value| value.trim().is_empty())
            .unwrap_or(true);
        let needs_public = self
            .auth
            .public_key
            .as_ref()
            .map(|value| value.trim().is_empty())
            .unwrap_or(true);
        if !needs_private && !needs_public {
            return Ok(false);
        }

        let (private_key, public_key) = generate_jwt_keypair()?;
        self.auth.private_key = Some(private_key);
        self.auth.public_key = Some(public_key);
        if self.auth.key_id.is_none() {
            self.auth.key_id = Some(format!("key-{}", Utc::now().format("%Y%m%d%H%M%S")));
        }
        self.updated_at = Utc::now();
        Ok(true)
    }

    pub fn event_store_path(&self) -> PathBuf {
        self.domain_data_dir().join("event_store")
    }

    pub fn event_store_path_for(&self, domain: &str) -> PathBuf {
        self.domain_data_dir_for(domain).join("event_store")
    }

    pub fn tokens_path(&self) -> PathBuf {
        self.domain_data_dir().join("tokens.json")
    }

    pub fn cli_token_path(&self) -> PathBuf {
        self.data_dir.join("cli.token")
    }

    pub fn jwt_revocations_path(&self) -> PathBuf {
        self.domain_data_dir().join("jwt_revocations.json")
    }

    pub fn jwt_manager_config(&self) -> Result<JwtManagerConfig> {
        let private_b64 = self
            .auth
            .resolved_private_key()
            .ok_or_else(|| EventError::Config("jwt private key not configured".to_string()))?;
        let private_key = STANDARD
            .decode(private_b64.trim())
            .map_err(|err| EventError::Config(format!("invalid jwt private key: {err}")))?;
        if private_key.is_empty() {
            return Err(EventError::Config(
                "jwt private key must contain data".to_string(),
            ));
        }
        let key_pair = Ed25519KeyPair::from_pkcs8(private_key.as_slice())
            .map_err(|err| EventError::Config(format!("invalid jwt private key: {err}")))?;
        let public_key = if let Some(public_b64) = self.auth.resolved_public_key() {
            let decoded = STANDARD
                .decode(public_b64.trim())
                .map_err(|err| EventError::Config(format!("invalid jwt public key: {err}")))?;
            if decoded.len() != ED25519_PUBLIC_KEY_LEN {
                return Err(EventError::Config(format!(
                    "jwt public key must decode to {} bytes (got {})",
                    ED25519_PUBLIC_KEY_LEN,
                    decoded.len()
                )));
            }
            if key_pair.public_key().as_ref() != decoded.as_slice() {
                return Err(EventError::Config(
                    "jwt public key does not match private key".to_string(),
                ));
            }
            decoded
        } else {
            key_pair.public_key().as_ref().to_vec()
        };
        let ttl = self.auth.default_ttl_secs.max(60);
        let clock_skew = self.auth.clock_skew_secs.clamp(0, 300);
        Ok(JwtManagerConfig {
            issuer: self.auth.issuer.clone(),
            audience: self.auth.audience.clone(),
            private_key,
            public_key,
            key_id: self.auth.resolved_key_id(),
            default_ttl: Duration::seconds(ttl as i64),
            clock_skew: Duration::seconds(clock_skew as i64),
        })
    }

    pub fn schema_store_path(&self) -> PathBuf {
        self.domain_data_dir().join("schemas.json")
    }

    pub fn schema_store_path_for(&self, domain: &str) -> PathBuf {
        self.domain_data_dir_for(domain).join("schemas.json")
    }

    pub fn staging_path(&self) -> PathBuf {
        self.domain_data_dir().join("staged_events.json")
    }

    pub fn plugins_path(&self) -> PathBuf {
        self.data_dir.join("plugins.json")
    }

    pub fn legacy_plugins_path(&self) -> PathBuf {
        self.domain_data_dir().join("plugins.json")
    }

    pub fn plugin_queue_path(&self) -> PathBuf {
        self.data_dir.join("plugin_queue.json")
    }

    pub fn legacy_plugin_queue_path(&self) -> PathBuf {
        self.domain_data_dir().join("plugin_queue.json")
    }

    pub fn plugin_queue_db_path(&self) -> PathBuf {
        self.data_dir.join("plugin_queue.db")
    }

    pub fn legacy_plugin_queue_db_path(&self) -> PathBuf {
        self.domain_data_dir().join("plugin_queue.db")
    }

    pub fn pid_file_path(&self) -> PathBuf {
        self.data_dir.join("eventdbx.pid")
    }

    pub fn verbose_responses(&self) -> bool {
        self.verbose_responses
    }

    pub fn is_initialized(&self) -> bool {
        self.resolved_data_encryption_key().is_some()
    }

    pub fn encryption_key(&self) -> Result<Option<Encryptor>> {
        match self.resolved_data_encryption_key() {
            Some(value) => Ok(Some(Encryptor::new_from_base64(&value)?)),
            None => Ok(None),
        }
    }

    pub fn load_plugins(&self) -> Result<Vec<PluginDefinition>> {
        self.migrate_plugin_config_to_root()?;

        let path = self.plugins_path();
        if !path.exists() {
            let legacy_path = self.legacy_plugins_path();
            if !legacy_path.exists() {
                return Ok(Vec::new());
            }
            return self.load_plugins_from(&legacy_path);
        }

        self.load_plugins_from(&path)
    }

    fn load_plugins_from(&self, path: &Path) -> Result<Vec<PluginDefinition>> {
        let contents = fs::read_to_string(&path)?;
        if contents.trim().is_empty() {
            return Ok(Vec::new());
        }

        let plugins = serde_json::from_str(&contents)?;
        Ok(plugins)
    }

    pub fn save_plugins(&self, plugins: &[PluginDefinition]) -> Result<()> {
        let path = self.plugins_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let payload = serde_json::to_string_pretty(plugins)?;
        fs::write(path, payload)?;
        Ok(())
    }

    pub fn migrate_plugin_config_to_root(&self) -> Result<bool> {
        let source = self.legacy_plugins_path();
        let dest = self.plugins_path();
        migrate_path_if_absent(&source, &dest)
    }

    pub fn migrate_plugin_queue_to_root(&self) -> Result<bool> {
        let mut migrated = false;

        let legacy_json = self.legacy_plugin_queue_path();
        let json_dest = self.plugin_queue_path();
        if migrate_path_if_absent(&legacy_json, &json_dest)? {
            migrated = true;
        }

        let legacy_db = self.legacy_plugin_queue_db_path();
        let db_dest = self.plugin_queue_db_path();
        if migrate_path_if_absent(&legacy_db, &db_dest)? {
            migrated = true;
        }

        Ok(migrated)
    }

    pub fn migrate_plugin_runtime_to_root(&self) -> Result<bool> {
        let legacy_run = self.domain_data_dir().join("plugins").join("run");
        let dest = self.data_dir.join("plugins").join("run");
        migrate_path_if_absent(&legacy_run, &dest)
    }

    pub fn ensure_single_tenant_layout(&self) -> Result<()> {
        if self.domain != DEFAULT_DOMAIN_NAME {
            return Err(EventError::Config(format!(
                "single-tenant core only supports domain '{DEFAULT_DOMAIN_NAME}', but config.toml selects '{}'. Export/import or run a dedicated migration before upgrading.",
                self.domain
            )));
        }
        if self.tenants.multi_tenant {
            return Err(EventError::Config(
                "single-tenant core does not support `[tenants].multi_tenant = true`. Export/import or run a dedicated migration before upgrading."
                    .to_string(),
            ));
        }
        if self.tenants.shard_map_path.is_some() {
            return Err(EventError::Config(
                "single-tenant core does not support a custom tenant shard map. Export/import or run a dedicated migration before upgrading."
                    .to_string(),
            ));
        }

        let mut conflicts = Vec::new();
        if let Some(entries) = collect_non_default_entries(self.domains_root().as_path())? {
            conflicts.extend(entries.into_iter().map(|entry| format!("domains/{entry}")));
        }
        if let Some(entries) = collect_non_default_shard_tenants(self.shards_root().as_path())? {
            conflicts.extend(entries.into_iter());
        }
        if self.tenant_meta_path().exists() {
            let store = TenantAssignmentStore::open_read_only(self.tenant_meta_path())?;
            for (tenant, _) in store.list()? {
                if tenant != DEFAULT_DOMAIN_NAME {
                    conflicts.push(format!("tenant_meta/{tenant}"));
                }
            }
        }

        if conflicts.is_empty() {
            return Ok(());
        }

        conflicts.sort();
        conflicts.dedup();
        Err(EventError::Config(format!(
            "single-tenant core only supports one local domain ('{DEFAULT_DOMAIN_NAME}'); found legacy tenant/domain data: {}. Export/import or run a dedicated migration before starting this build.",
            conflicts.join(", ")
        )))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocketConfig {
    #[serde(default = "default_socket_bind_addr")]
    pub bind_addr: String,
}

impl Default for SocketConfig {
    fn default() -> Self {
        Self {
            bind_addr: default_socket_bind_addr(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SocketConfigUpdate {
    pub bind_addr: Option<String>,
}

fn find_workspace_config_path(start: Option<&Path>) -> Result<PathBuf> {
    let start = match start {
        Some(path) => path.to_path_buf(),
        None => env::current_dir().map_err(|err| {
            EventError::Config(format!("failed to resolve current directory: {err}"))
        })?,
    };
    find_workspace_config_path_from(start.as_path())
}

fn find_workspace_config_path_from(start: &Path) -> Result<PathBuf> {
    let mut current = normalize_path(start.to_path_buf())?;
    loop {
        let candidate = current.join(WORKSPACE_DIR_NAME).join(CONFIG_FILE_NAME);
        if candidate.exists() {
            return Ok(candidate);
        }
        if !current.pop() {
            break;
        }
    }

    Err(EventError::Config(format!(
        "no EventDBX workspace found from {} upward.\nRun `dbx init` to create {} in the current directory or pass `--config <path>`.",
        start.display(),
        WORKSPACE_DIR_NAME
    )))
}

fn init_config_path(path: Option<PathBuf>) -> Result<PathBuf> {
    match path {
        Some(path) => normalize_path(path),
        None => {
            let cwd = env::current_dir().map_err(|err| {
                EventError::Config(format!("failed to resolve current directory: {err}"))
            })?;
            Ok(cwd.join(WORKSPACE_DIR_NAME).join(CONFIG_FILE_NAME))
        }
    }
}

fn load_config(config_path: PathBuf) -> Result<(Config, PathBuf)> {
    if !config_path.exists() {
        return Err(EventError::Config(format!(
            "config file not found at {}.\nRun `dbx init` to create a workspace or pass an existing `--config <path>`.",
            config_path.display()
        )));
    }

    let contents = fs::read_to_string(&config_path)?;
    let mut cfg: Config = toml::from_str(&contents)?;
    cfg.ensure_single_tenant_layout()?;

    let updated = cfg.ensure_encryption_key();
    let auth_updated = cfg.ensure_auth_keys()?;
    cfg.ensure_data_dir()?;
    let migrated = cfg.migrate_plugins()?;
    let plugins_relocated = cfg.migrate_plugin_config_to_root()?;
    let queue_relocated = cfg.migrate_plugin_queue_to_root()?;
    let runtime_relocated = cfg.migrate_plugin_runtime_to_root()?;
    if updated
        || migrated
        || auth_updated
        || plugins_relocated
        || queue_relocated
        || runtime_relocated
    {
        cfg.save(&config_path)?;
    }
    Ok((cfg, config_path))
}

fn normalize_path(path: PathBuf) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path)
    } else {
        env::current_dir().map(|cwd| cwd.join(path)).map_err(|err| {
            EventError::Config(format!("failed to resolve current directory: {err}"))
        })
    }
}

fn collect_non_default_entries(root: &Path) -> Result<Option<Vec<String>>> {
    if !root.exists() {
        return Ok(None);
    }

    let mut entries = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name != DEFAULT_DOMAIN_NAME {
            entries.push(name);
        }
    }

    Ok(Some(entries))
}

fn collect_non_default_shard_tenants(root: &Path) -> Result<Option<Vec<String>>> {
    if !root.exists() {
        return Ok(None);
    }

    let mut entries = Vec::new();
    for shard_entry in fs::read_dir(root)? {
        let shard_entry = shard_entry?;
        if !shard_entry.file_type()?.is_dir() {
            continue;
        }
        let shard_name = shard_entry.file_name().to_string_lossy().to_string();
        let tenants_root = shard_entry.path().join("tenants");
        if !tenants_root.exists() {
            continue;
        }
        for tenant_entry in fs::read_dir(&tenants_root)? {
            let tenant_entry = tenant_entry?;
            if !tenant_entry.file_type()?.is_dir() {
                continue;
            }
            let tenant = tenant_entry.file_name().to_string_lossy().to_string();
            if tenant != DEFAULT_DOMAIN_NAME {
                entries.push(format!("shards/{shard_name}/tenants/{tenant}"));
            }
        }
    }

    Ok(Some(entries))
}

fn generate_data_encryption_key() -> String {
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
    STANDARD.encode(bytes)
}

fn default_jwt_issuer() -> String {
    "eventdbx://self".to_string()
}

fn default_jwt_audience() -> String {
    "eventdbx-clients".to_string()
}

fn default_jwt_ttl_secs() -> u64 {
    3_600
}

fn default_jwt_clock_skew_secs() -> u64 {
    30
}

fn generate_jwt_keypair() -> Result<(String, String)> {
    let rng = SystemRandom::new();
    let pkcs8_doc = Ed25519KeyPair::generate_pkcs8(&rng)
        .map_err(|err| EventError::Config(format!("failed to generate jwt key pair: {err}")))?;
    let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_doc.as_ref())
        .map_err(|err| EventError::Config(format!("failed to parse jwt key pair: {err}")))?;
    let private_key = STANDARD.encode(pkcs8_doc.as_ref());
    let public_key = STANDARD.encode(key_pair.public_key().as_ref());
    Ok((private_key, public_key))
}

fn default_data_dir() -> PathBuf {
    env::current_dir()
        .map(|dir| dir.join(WORKSPACE_DIR_NAME))
        .unwrap_or_else(|_| PathBuf::from(WORKSPACE_DIR_NAME))
}

fn default_domain() -> String {
    DEFAULT_DOMAIN_NAME.to_string()
}

fn default_restrict() -> RestrictMode {
    RestrictMode::default()
}

fn default_list_page_size() -> usize {
    10
}

fn default_page_limit() -> usize {
    1000
}

fn default_plugin_max_attempts() -> u32 {
    10
}

fn default_plugin_payload_mode() -> PluginPayloadMode {
    PluginPayloadMode::All
}

fn default_snowflake_worker_id() -> u16 {
    0
}

fn default_no_noise() -> bool {
    DEFAULT_NO_NOISE
}

fn default_socket_bind_addr() -> String {
    format!("0.0.0.0:{}", DEFAULT_SOCKET_PORT)
}

fn default_cache_threshold() -> usize {
    DEFAULT_CACHE_THRESHOLD
}

fn default_plugin_queue_prune() -> PluginQueuePruneConfig {
    PluginQueuePruneConfig::default()
}

fn default_reference_default_depth() -> usize {
    crate::reference::DEFAULT_RESOLUTION_DEPTH
}

fn default_reference_max_depth() -> usize {
    crate::reference::MAX_RESOLUTION_DEPTH
}

fn default_queue_prune_done_ttl_secs() -> u64 {
    86_400
}

fn default_queue_prune_interval_secs() -> u64 {
    300
}

fn default_verbose_responses() -> bool {
    true
}

fn deserialize_restrict_mode<'de, D>(deserializer: D) -> std::result::Result<RestrictMode, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum RestrictInput {
        Bool(bool),
        String(String),
    }

    let value = Option::<RestrictInput>::deserialize(deserializer)?;
    Ok(match value {
        Some(RestrictInput::Bool(value)) => restrict::legacy_bool(value),
        Some(RestrictInput::String(value)) => {
            restrict::parse_mode(&value).unwrap_or_else(default_restrict)
        }
        None => default_restrict(),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDefinition {
    pub enabled: bool,
    #[serde(default = "default_emit_events")]
    pub emit_events: bool,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default = "default_plugin_payload_mode")]
    pub payload_mode: PluginPayloadMode,
    pub config: PluginConfig,
}

fn default_emit_events() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum PluginConfig {
    Tcp(TcpPluginConfig),
    Http(HttpPluginConfig),
    Log(LogPluginConfig),
    Process(ProcessPluginConfig),
}

impl PluginConfig {
    pub fn kind(&self) -> PluginKind {
        match self {
            PluginConfig::Tcp(_) => PluginKind::Tcp,
            PluginConfig::Http(_) => PluginKind::Http,
            PluginConfig::Log(_) => PluginKind::Log,
            PluginConfig::Process(_) => PluginKind::Process,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PluginKind {
    Tcp,
    Http,
    Log,
    Process,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PluginPayloadMode {
    All,
    EventOnly,
    StateOnly,
    SchemaOnly,
    EventAndSchema,
    ExtensionsOnly,
}

impl PluginPayloadMode {
    pub fn includes_event(self) -> bool {
        matches!(
            self,
            PluginPayloadMode::All
                | PluginPayloadMode::EventOnly
                | PluginPayloadMode::EventAndSchema
        )
    }

    pub fn includes_state(self) -> bool {
        matches!(self, PluginPayloadMode::All | PluginPayloadMode::StateOnly)
    }

    pub fn includes_schema(self) -> bool {
        matches!(
            self,
            PluginPayloadMode::All
                | PluginPayloadMode::SchemaOnly
                | PluginPayloadMode::EventAndSchema
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpPluginConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpPluginConfig {
    pub endpoint: String,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub https: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogPluginConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub detail: LogDetailMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LogDetailMode {
    #[default]
    Summary,
    Full,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessPluginConfig {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub working_dir: Option<PathBuf>,
}

fn default_log_level() -> String {
    "info".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn applies_snapshot_threshold_updates() {
        let mut config = Config::default();
        assert_eq!(config.snapshot_threshold, None);

        config.apply_update(ConfigUpdate {
            snapshot_threshold: Some(Some(42)),
            ..ConfigUpdate::default()
        });
        assert_eq!(config.snapshot_threshold, Some(42));

        config.apply_update(ConfigUpdate {
            snapshot_threshold: Some(None),
            ..ConfigUpdate::default()
        });
        assert_eq!(config.snapshot_threshold, None);
    }

    #[test]
    fn applies_verbose_response_updates() {
        let mut config = Config::default();
        assert!(config.verbose_responses());

        config.apply_update(ConfigUpdate {
            verbose_responses: Some(false),
            ..ConfigUpdate::default()
        });
        assert!(!config.verbose_responses());
    }

    #[test]
    fn log_plugin_config_defaults_detail_to_summary() {
        let config: LogPluginConfig =
            serde_json::from_str(r#"{"level":"debug","template":"[{aggregate}] {event}"}"#)
                .expect("log config should deserialize");

        assert_eq!(config.detail, LogDetailMode::Summary);
    }

    #[derive(Deserialize)]
    struct RestrictWrapper {
        #[serde(
            default = "default_restrict",
            deserialize_with = "deserialize_restrict_mode"
        )]
        restrict: RestrictMode,
    }

    #[test]
    fn restrict_mode_deserializes_from_bool() {
        let wrapper: RestrictWrapper =
            toml::from_str("restrict = false").expect("bool value should parse");
        assert_eq!(wrapper.restrict, RestrictMode::Off);

        let wrapper: RestrictWrapper =
            toml::from_str("restrict = true").expect("bool value should parse");
        assert_eq!(wrapper.restrict, RestrictMode::Default);
    }

    #[test]
    fn restrict_mode_deserializes_from_string() {
        let wrapper: RestrictWrapper =
            toml::from_str("restrict = \"strict\"").expect("string value should parse");
        assert_eq!(wrapper.restrict, RestrictMode::Strict);

        let wrapper: RestrictWrapper = toml::from_str("restrict = \"unknown\"")
            .expect("unknown value should fall back to default");
        assert_eq!(wrapper.restrict, RestrictMode::Default);
    }

    #[test]
    fn restrict_mode_serializes_as_lowercase_string() {
        let mut config = Config::default();
        config.restrict = RestrictMode::Strict;
        let payload = toml::to_string(&config).expect("config should serialize");
        assert!(payload.contains("restrict = \"strict\""));
    }

    #[test]
    fn finds_nearest_workspace_config_in_parent_tree() {
        let temp = tempdir().expect("temp dir");
        let project = temp.path().join("project");
        let nested = project.join("nested").join("deeper");
        let config_path = project.join(WORKSPACE_DIR_NAME).join(CONFIG_FILE_NAME);
        fs::create_dir_all(&nested).expect("nested dir");
        fs::create_dir_all(config_path.parent().expect("config parent")).expect("workspace dir");
        fs::write(&config_path, "").expect("config file");

        let resolved =
            find_workspace_config_path_from(&nested).expect("workspace discovery should succeed");
        assert_eq!(resolved, config_path);
    }

    #[test]
    fn reports_missing_workspace_when_no_parent_contains_dbx() {
        let temp = tempdir().expect("temp dir");
        let nested = temp.path().join("project").join("nested");
        fs::create_dir_all(&nested).expect("nested dir");

        let err = find_workspace_config_path_from(&nested).expect_err("workspace should be absent");
        assert!(err.to_string().contains("Run `dbx init`"));
    }

    #[test]
    fn init_workspace_creates_local_config_and_data_dir() {
        let temp = tempdir().expect("temp dir");
        let config_path = temp
            .path()
            .join("project")
            .join(WORKSPACE_DIR_NAME)
            .join(CONFIG_FILE_NAME);

        let (config, resolved_path, created) =
            init_workspace(Some(config_path.clone())).expect("workspace init should succeed");

        assert!(created);
        assert_eq!(resolved_path, config_path);
        assert_eq!(
            config.data_dir,
            config_path.parent().expect("config parent").to_path_buf()
        );
        assert!(resolved_path.exists());
        assert!(config.event_store_path().join("CURRENT").exists());
    }

    #[test]
    fn auto_init_workspace_creates_missing_config_when_enabled() {
        let temp = tempdir().expect("temp dir");
        let config_path = temp
            .path()
            .join("project")
            .join(WORKSPACE_DIR_NAME)
            .join(CONFIG_FILE_NAME);

        assert!(!config_path.exists());

        maybe_auto_init_workspace(config_path.as_path(), true)
            .expect("auto init should create workspace");

        assert!(config_path.exists());
        let (config, resolved_path) =
            load_config(config_path.clone()).expect("auto-initialized config should load");
        assert_eq!(resolved_path, config_path);
        assert!(config.event_store_path().join("CURRENT").exists());
    }

    #[test]
    fn auto_init_workspace_leaves_missing_config_when_disabled() {
        let temp = tempdir().expect("temp dir");
        let config_path = temp
            .path()
            .join("project")
            .join(WORKSPACE_DIR_NAME)
            .join(CONFIG_FILE_NAME);

        maybe_auto_init_workspace(config_path.as_path(), false)
            .expect("disabled auto init should be a no-op");

        assert!(!config_path.exists());
    }

    #[test]
    fn resolve_config_path_for_load_uses_local_workspace_path_when_auto_init_enabled() {
        let temp = tempdir().expect("temp dir");
        let original_cwd = env::current_dir().expect("current dir");
        env::set_current_dir(temp.path()).expect("change current dir");

        let resolved = resolve_config_path_for_load(None, true)
            .expect("auto-init path resolution should succeed");

        env::set_current_dir(original_cwd).expect("restore current dir");

        assert!(resolved.ends_with(Path::new(WORKSPACE_DIR_NAME).join(CONFIG_FILE_NAME)));
    }
}

fn migrate_path_if_absent(source: &Path, dest: &Path) -> Result<bool> {
    if dest.exists() || !source.exists() {
        return Ok(false);
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    if let Err(err) = fs::rename(source, dest) {
        copy_path_recursively(source, dest).map_err(|copy_err| {
            EventError::Config(format!(
                "failed to move {} to {} (rename error: {}; copy error: {})",
                source.display(),
                dest.display(),
                err,
                copy_err
            ))
        })?;

        // Clean up the legacy path after a successful copy to avoid duplicate configs.
        match fs::metadata(source) {
            Ok(metadata) => {
                let cleanup = if metadata.is_dir() {
                    fs::remove_dir_all(source)
                } else {
                    fs::remove_file(source).or_else(|remove_err| {
                        if remove_err.kind() == ErrorKind::NotFound {
                            Ok(())
                        } else {
                            Err(remove_err)
                        }
                    })
                };

                if let Err(remove_err) = cleanup {
                    return Err(EventError::Config(format!(
                        "migrated {} to {} but failed to remove legacy source: {}",
                        source.display(),
                        dest.display(),
                        remove_err
                    )));
                }
            }
            Err(meta_err) if meta_err.kind() == ErrorKind::NotFound => return Ok(true),
            Err(meta_err) => {
                return Err(EventError::Config(format!(
                    "migrated {} to {} but failed to inspect legacy source: {}",
                    source.display(),
                    dest.display(),
                    meta_err
                )));
            }
        }
    }

    Ok(true)
}

fn copy_path_recursively(source: &Path, dest: &Path) -> Result<()> {
    let metadata = fs::metadata(source)?;
    if metadata.is_dir() {
        fs::create_dir_all(dest)?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            let child_source = entry.path();
            let child_dest = dest.join(entry.file_name());
            copy_path_recursively(&child_source, &child_dest)?;
        }
    } else {
        fs::copy(source, dest).map_err(|err| {
            EventError::Config(format!(
                "failed to copy {} to {}: {}",
                source.display(),
                dest.display(),
                err
            ))
        })?;
    }
    Ok(())
}
