use std::path::PathBuf;

use anyhow::{Result, anyhow};
use clap::Args;

use eventdbx::config::{Config, ConfigUpdate, TenantRoutingConfigUpdate, load_or_default};

#[derive(Args)]
pub struct ConfigArgs {
    #[arg(long)]
    pub port: Option<u16>,

    #[arg(long)]
    pub data_dir: Option<PathBuf>,

    #[arg(long = "cache-threshold")]
    pub cache_threshold: Option<usize>,

    #[arg(long, alias = "dek")]
    pub data_encryption_key: Option<String>,

    #[arg(long)]
    pub list_page_size: Option<usize>,

    #[arg(long = "page-limit")]
    pub page_limit: Option<usize>,

    #[arg(long = "verbose-responses")]
    pub verbose_responses: Option<bool>,

    #[arg(long = "plugin-max-attempts")]
    pub plugin_max_attempts: Option<u32>,

    #[arg(long = "snapshot-threshold")]
    pub snapshot_threshold: Option<u64>,

    #[arg(
        long = "clear-snapshot-threshold",
        conflicts_with = "snapshot_threshold"
    )]
    pub clear_snapshot_threshold: bool,
    #[arg(long = "snowflake-worker-id")]
    pub snowflake_worker_id: Option<u16>,

    #[arg(long = "multi-tenant")]
    pub multi_tenant: Option<bool>,

    #[arg(long = "shard-count")]
    pub shard_count: Option<u16>,

    #[arg(long = "shard-map-path")]
    pub shard_map_path: Option<PathBuf>,

    /// Default depth for reference resolution
    #[arg(long = "reference-default-depth")]
    pub reference_default_depth: Option<usize>,

    /// Maximum depth allowed for reference resolution
    #[arg(long = "reference-max-depth")]
    pub reference_max_depth: Option<usize>,

    /// Disable Noise encryption for the control channel (Cap'n Proto framing only).
    /// WARNING: Sends control protocol messages in plaintext; use only on trusted networks
    /// such as localhost or private links.
    #[arg(long = "no-noise", action = clap::ArgAction::SetTrue)]
    pub no_noise: bool,

    #[arg(long = "noise", action = clap::ArgAction::SetTrue, conflicts_with = "no_noise")]
    pub noise: bool,
}

pub fn execute(config_path: Option<PathBuf>, args: ConfigArgs) -> Result<()> {
    let (mut config, path) = load_or_default(config_path)?;
    if !has_updates(&args, &config) {
        let contents = toml::to_string_pretty(&config)?;
        println!("{contents}");
        return Ok(());
    }
    let was_initialized = config.is_initialized();

    let ConfigArgs {
        port,
        data_dir,
        cache_threshold,
        data_encryption_key,
        list_page_size,
        page_limit,
        verbose_responses,
        plugin_max_attempts,
        snapshot_threshold,
        clear_snapshot_threshold,
        snowflake_worker_id,
        multi_tenant,
        shard_count,
        shard_map_path,
        reference_default_depth,
        reference_max_depth,
        no_noise,
        noise,
    } = args;

    let data_encryption_key = normalize_secret(data_encryption_key);
    let snapshot_threshold = if clear_snapshot_threshold {
        Some(None)
    } else {
        snapshot_threshold.map(Some)
    };
    let tenant_update =
        if multi_tenant.is_some() || shard_count.is_some() || shard_map_path.is_some() {
            Some(TenantRoutingConfigUpdate {
                multi_tenant,
                shard_count,
                shard_map_path,
            })
        } else {
            None
        };
    let no_noise_update = if no_noise {
        Some(true)
    } else if noise {
        Some(false)
    } else {
        None
    };

    config.apply_update(ConfigUpdate {
        port,
        data_dir,
        cache_threshold,
        snapshot_threshold,
        data_encryption_key,
        restrict: None,
        list_page_size,
        page_limit,
        verbose_responses,
        reference_default_depth,
        reference_max_depth,
        plugin_max_attempts,
        socket: None,
        snowflake_worker_id,
        tenants: tenant_update,
        no_noise: no_noise_update,
    });

    if !was_initialized && !config.is_initialized() {
        return Err(anyhow!(
            "initial setup requires --dek to be provided (32-byte base64 value)"
        ));
    }

    config.ensure_data_dir()?;
    config.save(&path)?;

    tracing::info!("Configuration saved to {}", path.display());
    Ok(())
}

pub fn ensure_secrets_configured(config: &Config) -> Result<()> {
    if config.is_initialized() {
        Ok(())
    } else {
        Err(anyhow!(
            "data encryption key must be configured.\nRun `eventdbx config --dek <value>` during initial setup."
        ))
    }
}

pub fn normalize_secret(input: Option<String>) -> Option<String> {
    input.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn has_updates(args: &ConfigArgs, config: &Config) -> bool {
    args.port.is_some()
        || args.data_dir.is_some()
        || args.cache_threshold.is_some()
        || args
            .data_encryption_key
            .as_ref()
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false)
        || args.list_page_size.is_some()
        || args.page_limit.is_some()
        || args.verbose_responses.is_some()
        || args.reference_default_depth.is_some()
        || args.reference_max_depth.is_some()
        || args.plugin_max_attempts.is_some()
        || args.snapshot_threshold.is_some()
        || args.clear_snapshot_threshold
        || args.snowflake_worker_id.is_some()
        || args.multi_tenant.is_some()
        || args.shard_count.is_some()
        || args.shard_map_path.is_some()
        || (args.no_noise && !config.no_noise)
        || (args.noise && config.no_noise)
}
