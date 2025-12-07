use std::path::PathBuf;
use std::{fs, io::Read};

use anyhow::{Context, Result};
use assert_cmd::{Command, cargo::cargo_bin_cmd};
use base64::Engine as _;
use csv::StringRecord;
use serde_json::{Value, json};
use tempfile::TempDir;
use zip::ZipArchive;

use chrono::{DateTime, Utc};
use eventdbx::{
    plugin::{JobPriority, queue::PluginQueueStore},
    snowflake::SnowflakeId,
    token::JwtClaims,
    validation::MAX_EVENT_PAYLOAD_BYTES,
};

struct CliTest {
    _tmp: TempDir,
    home: PathBuf,
    log_dir: PathBuf,
    config_home: PathBuf,
}

struct FailureOutput {
    stdout: String,
    stderr: String,
}

macro_rules! cli_failure_test {
    ($name:ident, [$($arg:expr),+ $(,)?], $pattern:expr) => {
        #[test]
        fn $name() -> Result<()> {
            let cli = CliTest::new()?;
            let failure = cli.run_failure(&[$($arg),+])?;
            assert!(
                failure.stderr.contains($pattern),
                "expected stderr to contain {:?}\nstdout:\n{}\nstderr:\n{}",
                $pattern,
                failure.stdout,
                failure.stderr
            );
            Ok(())
        }
    };
}

macro_rules! cli_success_contains_test {
    ($name:ident, [$($arg:expr),+ $(,)?], $pattern:expr) => {
        #[test]
        fn $name() -> Result<()> {
            let cli = CliTest::new()?;
            let stdout = cli.run(&[$($arg),+])?;
            assert!(
                stdout.contains($pattern),
                "expected stdout to contain {:?}, got:\n{}",
                $pattern,
                stdout
            );
            Ok(())
        }
    };
}

fn decode_jwt_claims(token: &str) -> Result<JwtClaims> {
    let parts: Vec<&str> = token.trim().split('.').collect();
    if parts.len() != 3 {
        anyhow::bail!("token must contain three segments");
    }
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(parts[1])
        .context("failed to decode token payload")?;
    let claims: JwtClaims =
        serde_json::from_slice(&payload).context("failed to parse token claims")?;
    Ok(claims)
}

impl CliTest {
    fn new() -> Result<Self> {
        let tmp = tempfile::tempdir().context("failed to create temp dir")?;
        let home = tmp.path().join("home");
        fs::create_dir_all(&home).context("failed to create temporary home directory")?;
        let log_dir = home.join("logs");
        fs::create_dir_all(&log_dir).context("failed to create temporary log directory")?;
        let config_home = home.join(".config");
        fs::create_dir_all(&config_home).context("failed to create temporary config directory")?;
        Ok(Self {
            _tmp: tmp,
            home,
            log_dir,
            config_home,
        })
    }

    fn command(&self) -> Result<Command> {
        let mut cmd = cargo_bin_cmd!("dbx");
        cmd.env("HOME", &self.home);
        cmd.env("EVENTDBX_LOG_DIR", &self.log_dir);
        cmd.env("DBX_NO_UPGRADE_CHECK", "1");
        cmd.env("XDG_CONFIG_HOME", &self.config_home);
        Ok(cmd)
    }

    fn run(&self, args: &[&str]) -> Result<String> {
        let output = self.exec(args)?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("dbx {:?} exited with {}: {}", args, output.status, stderr);
        }
        Ok(normalize_output(output.stdout)?)
    }

    fn run_json(&self, args: &[&str]) -> Result<Value> {
        let stdout = self.run(args)?;
        let parsed = serde_json::from_str(stdout.trim())
            .with_context(|| format!("failed to parse JSON output from dbx {:?}", args))?;
        Ok(parsed)
    }

    fn create_aggregate(&self, aggregate: &str, aggregate_id: &str, event: &str) -> Result<()> {
        self.create_aggregate_with_fields(aggregate, aggregate_id, event, &[])
    }

    fn create_aggregate_with_fields(
        &self,
        aggregate: &str,
        aggregate_id: &str,
        event: &str,
        fields: &[(&str, &str)],
    ) -> Result<()> {
        let mut args = vec![
            "aggregate".to_string(),
            "create".to_string(),
            aggregate.to_string(),
            aggregate_id.to_string(),
            "--event".to_string(),
            event.to_string(),
            "--json".to_string(),
        ];

        for (key, value) in fields {
            args.push("--field".to_string());
            args.push(format!("{}={}", key, value));
        }

        let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        let stdout = self.run(&arg_refs)?;
        serde_json::from_str::<Value>(stdout.trim())
            .context("failed to parse aggregate create output")?;
        Ok(())
    }

    fn create_aggregate_with_payload(
        &self,
        aggregate: &str,
        aggregate_id: &str,
        event: &str,
        payload: &str,
    ) -> Result<()> {
        let stdout = self.run(&[
            "aggregate",
            "create",
            aggregate,
            aggregate_id,
            "--event",
            event,
            "--payload",
            payload,
            "--json",
        ])?;
        serde_json::from_str::<Value>(stdout.trim())
            .context("failed to parse aggregate create output")?;
        Ok(())
    }

    fn run_failure(&self, args: &[&str]) -> Result<FailureOutput> {
        let output = self.exec(args)?;
        if output.status.success() {
            anyhow::bail!("expected dbx {:?} to fail but it succeeded", args);
        }
        Ok(FailureOutput {
            stdout: normalize_output(output.stdout)?,
            stderr: normalize_output(output.stderr)?,
        })
    }

    fn exec(&self, args: &[&str]) -> Result<std::process::Output> {
        let mut cmd = self.command()?;
        cmd.args(args);
        cmd.output()
            .with_context(|| format!("failed to run dbx {:?}", args))
    }

    fn run_with_input(&self, args: &[&str], input: &str) -> Result<String> {
        let mut cmd = self.command()?;
        cmd.args(args);
        let output = cmd
            .write_stdin(input)
            .output()
            .with_context(|| format!("failed to run dbx {:?}", args))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("dbx {:?} exited with {}: {}", args, output.status, stderr);
        }
        Ok(normalize_output(output.stdout)?)
    }

    fn data_dir(&self) -> PathBuf {
        self.home.join(".eventdbx")
    }

    fn config_path(&self) -> PathBuf {
        self.data_dir().join("config.toml")
    }

    fn load_config(&self) -> Result<eventdbx::config::Config> {
        let path = self.config_path();
        if !path.exists() {
            let _ = self.run(&["config"])?;
        }
        let contents = fs::read_to_string(&path).context("failed to read config.toml")?;
        let cfg: eventdbx::config::Config =
            toml::from_str(&contents).context("failed to parse config.toml")?;
        Ok(cfg)
    }

    fn plugin_queue_db_path(&self) -> Result<PathBuf> {
        let config = self.load_config()?;
        Ok(config.plugin_queue_db_path())
    }
}

#[test]
fn token_list_empty_prints_hint() -> Result<()> {
    let cli = CliTest::new()?;
    let stdout = cli.run(&["token", "list"])?;
    assert!(
        stdout.contains("no issued tokens"),
        "unexpected token list output:\n{}",
        stdout
    );
    Ok(())
}

#[test]
fn token_generate_and_list_json_round_trip() -> Result<()> {
    let cli = CliTest::new()?;
    let stdout = cli.run(&[
        "token",
        "generate",
        "--group",
        "ops",
        "--user",
        "alice",
        "--action",
        "aggregate.read",
        "--tenant",
        "default",
        "--json",
    ])?;
    let record: Value =
        serde_json::from_str(stdout.trim()).context("failed to parse token generate output")?;
    assert_eq!(record["group"], "ops");
    assert_eq!(record["user"], "alice");
    assert_eq!(record["actions"], json!(["aggregate.read"]));

    let stdout = cli.run(&["token", "list", "--json"])?;
    let records: Value =
        serde_json::from_str(stdout.trim()).context("failed to parse token list output")?;
    let Some(array) = records.as_array() else {
        anyhow::bail!("expected token list JSON to be an array, got {}", records);
    };
    assert_eq!(
        array.len(),
        1,
        "expected one token record, got {}",
        array.len()
    );
    assert_eq!(array[0]["group"], "ops");
    assert_eq!(array[0]["user"], "alice");

    Ok(())
}

#[test]
fn token_generate_requires_action_for_non_root() -> Result<()> {
    let cli = CliTest::new()?;
    let failure = cli.run_failure(&[
        "token", "generate", "--group", "ops", "--user", "alice", "--tenant", "default",
    ])?;
    assert!(
        failure
            .stderr
            .contains("at least one --action must be provided"),
        "unexpected validation message:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn token_bootstrap_writes_file_by_default() -> Result<()> {
    let cli = CliTest::new()?;
    let token_path = cli.data_dir().join("cli.token");
    if token_path.exists() {
        fs::remove_file(&token_path).context("failed to remove existing cli.token")?;
    }
    let stdout = cli.run(&["token", "bootstrap"])?;
    assert!(
        stdout.contains("Bootstrap token stored at"),
        "unexpected bootstrap output:\n{}",
        stdout
    );
    let contents =
        fs::read_to_string(&token_path).context("failed to read bootstrap token from disk")?;
    let token = contents.trim();
    assert!(
        token.split('.').count() == 3,
        "expected bootstrap token written to disk to contain three segments, got: {}",
        token
    );
    let claims = decode_jwt_claims(token)?;
    let exp = claims
        .exp
        .ok_or_else(|| anyhow::anyhow!("bootstrap token missing expiration in claims"))?;
    let ttl_secs = exp - claims.iat;
    assert!(
        (ttl_secs - 7_200).abs() <= 2,
        "expected bootstrap token TTL close to 7200 seconds, got {ttl_secs}"
    );
    Ok(())
}

#[test]
fn token_bootstrap_stdout_prints_token_without_writing_file() -> Result<()> {
    let cli = CliTest::new()?;
    let token_path = cli.data_dir().join("cli.token");
    if token_path.exists() {
        fs::remove_file(&token_path).context("failed to remove existing cli.token")?;
    }
    let stdout = cli.run(&["token", "bootstrap", "--stdout"])?;
    let token = stdout.trim();
    assert!(
        token.split('.').count() == 3,
        "expected stdout bootstrap token to contain three segments, got: {}",
        token
    );
    assert!(
        !token_path.exists(),
        "expected --stdout bootstrap to skip writing {}, but file exists",
        token_path.display()
    );
    Ok(())
}

#[test]
fn token_bootstrap_override_expires() -> Result<()> {
    let cli = CliTest::new()?;
    let token_path = cli.data_dir().join("cli.token");
    if token_path.exists() {
        fs::remove_file(&token_path).context("failed to remove existing cli.token")?;
    }
    let mut config = cli.load_config()?;
    config.auth.clock_skew_secs = 0;
    fs::write(cli.config_path(), toml::to_string(&config)?)
        .context("failed to write updated config")?;

    let stdout = cli.run(&["token", "bootstrap", "--stdout", "--ttl", "1"])?;
    let token = stdout.trim();
    let claims = decode_jwt_claims(token)?;
    assert_eq!(
        claims.exp.map(|exp| exp - claims.iat),
        Some(1),
        "TTL override should propagate to bootstrap token"
    );

    std::thread::sleep(std::time::Duration::from_secs(2));
    let failure = cli.run_failure(&["token", "refresh", "--token", token])?;
    assert!(
        failure.stderr.contains("token has expired") || failure.stderr.contains("token_expired"),
        "expected expiration error after TTL\nstdout:\n{}\nstderr:\n{}",
        failure.stdout,
        failure.stderr
    );

    Ok(())
}

#[test]
fn token_refresh_replaces_token() -> Result<()> {
    let cli = CliTest::new()?;
    let original = cli.run_json(&[
        "token",
        "generate",
        "--group",
        "ops",
        "--user",
        "alice",
        "--action",
        "aggregate.read",
        "--tenant",
        "default",
        "--json",
    ])?;
    let original_token = original["token"]
        .as_str()
        .context("missing token field from token generate")?
        .to_string();
    let original_jti = original["jti"]
        .as_str()
        .context("missing jti from token generate output")?
        .to_string();

    let refreshed = cli.run_json(&["token", "refresh", "--token", &original_token, "--json"])?;
    let new_token = refreshed["token"]
        .as_str()
        .context("missing token field from refresh response")?;
    assert_ne!(
        new_token, original_token,
        "refresh should issue a replacement token"
    );
    let refreshed_jti = refreshed["jti"]
        .as_str()
        .context("missing jti from refresh response")?
        .to_string();

    let list = cli.run_json(&["token", "list", "--json"])?;
    let array = list
        .as_array()
        .context("token list did not produce array after refresh")?;
    assert_eq!(
        array.len(),
        2,
        "expected refreshed token plus revoked original"
    );
    let mut active_found = false;
    let mut revoked_found = false;
    for record in array {
        match record["jti"].as_str() {
            Some(jti) if jti == refreshed_jti => {
                active_found = true;
                assert_eq!(
                    record["status"],
                    json!("active"),
                    "refreshed token should remain active"
                );
            }
            Some(jti) if jti == original_jti => {
                revoked_found = true;
                assert_eq!(
                    record["status"],
                    json!("revoked"),
                    "original token should be marked revoked"
                );
            }
            Some("active") => {
                active_found = true;
            }
            Some("revoked") => {
                revoked_found = true;
            }
            _ => {}
        }
    }
    assert!(active_found, "missing active refreshed token entry");
    assert!(revoked_found, "missing revoked original token entry");

    Ok(())
}

cli_failure_test!(
    token_refresh_unknown_token_fails,
    ["token", "refresh", "--token", "unknown"],
    "token not found or inactive"
);

cli_failure_test!(
    token_refresh_missing_token_argument,
    ["token", "refresh", "--json"],
    "the following required arguments were not provided"
);

#[test]
fn token_revoke_unknown_token_fails() -> Result<()> {
    let cli = CliTest::new()?;
    let failure = cli.run_failure(&["token", "revoke", "missing-token"])?;
    assert!(
        failure.stderr.contains("token not found or inactive"),
        "unexpected revoke failure message:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn plugin_list_empty_reports_none() -> Result<()> {
    let cli = CliTest::new()?;
    let stdout = cli.run(&["plugin", "list"])?;
    assert_eq!(stdout.trim(), "(no plugins configured)");

    let stdout = cli.run(&["plugin", "list", "--json"])?;
    let plugins: Value =
        serde_json::from_str(stdout.trim()).context("failed to parse plugin list JSON")?;
    assert_eq!(plugins, json!([]));

    Ok(())
}

#[test]
fn queue_status_empty() -> Result<()> {
    let cli = CliTest::new()?;
    let stdout = cli.run(&["queue"])?;
    assert_eq!(stdout.trim(), "pending=0 processing=0 done=0 dead=0");
    Ok(())
}

#[test]
fn queue_retry_without_dead_events() -> Result<()> {
    let cli = CliTest::new()?;
    let stdout = cli.run(&["queue", "retry"])?;
    assert!(
        stdout.contains("no dead plugin jobs to retry"),
        "unexpected queue retry output:\n{}",
        stdout
    );
    Ok(())
}

cli_success_contains_test!(
    queue_clear_no_dead_events,
    ["queue", "clear"],
    "no dead plugin jobs to clear"
);

cli_success_contains_test!(
    queue_retry_specific_event_no_match,
    ["queue", "retry", "--event-id", "1"],
    "no dead plugin jobs match the provided id"
);

#[test]
fn queue_clear_prompts_and_respects_confirmation() -> Result<()> {
    let cli = CliTest::new()?;
    {
        let queue_path = cli.plugin_queue_db_path()?;
        let store = PluginQueueStore::open(queue_path.as_path()).map_err(anyhow::Error::from)?;
        let job = store
            .enqueue_job(
                "test",
                serde_json::json!({"event_id": SnowflakeId::from_u64(42).to_string()}),
                Utc::now().timestamp_millis(),
                JobPriority::Normal,
            )
            .map_err(anyhow::Error::from)?;
        let dispatched = store
            .dispatch_jobs("test", 1, Utc::now().timestamp_millis())
            .map_err(anyhow::Error::from)?;
        assert_eq!(dispatched.len(), 1);
        store
            .fail_job(job.id, "boom".into(), Utc::now().timestamp_millis(), 0)
            .map_err(anyhow::Error::from)?;
    }

    let cancelled = cli.run_with_input(&["queue", "clear"], "\n")?;
    assert!(
        cancelled.contains("clear cancelled"),
        "expected clear cancellation message, got:\n{}",
        cancelled
    );

    let cleared = cli.run_with_input(&["queue", "clear"], "clear\n")?;
    assert!(
        cleared.contains("cleared 1 dead job(s) from the plugin queue"),
        "expected clear confirmation, got:\n{}",
        cleared
    );
    Ok(())
}

#[test]
fn queue_status_displays_dead_events() -> Result<()> {
    let cli = CliTest::new()?;
    {
        let queue_path = cli.plugin_queue_db_path()?;
        let store = PluginQueueStore::open(queue_path.as_path()).map_err(anyhow::Error::from)?;
        store
            .enqueue_job(
                "alpha",
                serde_json::json!({"event_id": SnowflakeId::from_u64(99).to_string()}),
                Utc::now().timestamp_millis(),
                JobPriority::Normal,
            )
            .map_err(anyhow::Error::from)?;
        let job = store
            .enqueue_job(
                "alpha",
                serde_json::json!({"event_id": SnowflakeId::from_u64(100).to_string()}),
                Utc::now().timestamp_millis(),
                JobPriority::Normal,
            )
            .map_err(anyhow::Error::from)?;
        let dispatched = store
            .dispatch_jobs("alpha", 1, Utc::now().timestamp_millis())
            .map_err(anyhow::Error::from)?;
        assert_eq!(dispatched.len(), 1);
        store
            .fail_job(job.id, "error".into(), Utc::now().timestamp_millis(), 0)
            .map_err(anyhow::Error::from)?;
    }

    let status_output = cli.run(&["queue"])?;
    assert!(
        status_output.contains("dead=1"),
        "unexpected queue status:\n{}",
        status_output
    );
    assert!(
        status_output.contains("status=\"dead\""),
        "expected dead job details in output:\n{}",
        status_output
    );
    Ok(())
}

#[test]
fn schema_list_empty_json() -> Result<()> {
    let cli = CliTest::new()?;
    let stdout = cli.run(&["schema", "list", "--json"])?;
    let schemas: Value =
        serde_json::from_str(stdout.trim()).context("failed to parse schema list JSON")?;
    assert_eq!(schemas, json!([]));
    Ok(())
}

#[test]
fn schema_add_remove_annotate_flow() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "inventory",
        "--events",
        "item_created",
        "--json",
    ])?;

    let add_out = cli.run(&["schema", "add", "inventory", "item_restocked"])?;
    assert!(
        add_out.contains("schema=inventory added_events=item_restocked total_events=2"),
        "unexpected schema add output:\n{}",
        add_out
    );

    let schema_after_add = cli.run_json(&["schema", "inventory"])?;
    assert!(
        schema_after_add["events"]
            .get("item_restocked")
            .and_then(|event| event.get("fields"))
            .is_some(),
        "expected schema to contain item_restocked event: {}",
        schema_after_add
    );

    let annotate_out = cli.run(&[
        "schema",
        "annotate",
        "inventory",
        "item_created",
        "--note",
        "Initial load event",
    ])?;
    assert!(
        annotate_out.contains("note_set=\"Initial load event\""),
        "unexpected schema annotate output:\n{}",
        annotate_out
    );

    let schema_with_note = cli.run_json(&["schema", "inventory"])?;
    assert_eq!(
        schema_with_note["events"]["item_created"]["notes"],
        json!("Initial load event")
    );

    let clear_out = cli.run(&["schema", "annotate", "inventory", "item_created", "--clear"])?;
    assert!(
        clear_out.contains("note_cleared"),
        "unexpected schema annotate clear output:\n{}",
        clear_out
    );

    let schema_cleared = cli.run_json(&["schema", "inventory"])?;
    assert!(
        schema_cleared["events"]["item_created"]
            .get("notes")
            .is_none(),
        "expected note to be cleared from schema: {}",
        schema_cleared
    );

    let remove_out = cli.run(&["schema", "remove", "inventory", "item_restocked"])?;
    assert!(
        remove_out.contains("schema=inventory removed_event=item_restocked remaining_events=1"),
        "unexpected schema remove output:\n{}",
        remove_out
    );

    Ok(())
}

#[test]
fn schema_hide_field() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "orders",
        "--events",
        "order_created",
        "--json",
    ])?;

    let hide_out = cli.run(&[
        "schema",
        "hide",
        "--aggregate",
        "orders",
        "--field",
        "internal_notes",
    ])?;
    assert!(
        hide_out.contains("aggregate=orders field=internal_notes hidden"),
        "unexpected schema hide output:\n{}",
        hide_out
    );

    let schema_after_hide = cli.run_json(&["schema", "orders"])?;
    let hidden = schema_after_hide["hidden_fields"]
        .as_array()
        .context("hidden_fields not present in schema json")?;
    assert!(
        hidden.iter().any(|value| value == "internal_notes"),
        "expected hidden_fields to include internal_notes: {}",
        schema_after_hide
    );

    Ok(())
}

#[test]
fn schema_unhide_field() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "orders",
        "--events",
        "order_created",
        "--json",
    ])?;

    cli.run(&[
        "schema",
        "hide",
        "--aggregate",
        "orders",
        "--field",
        "internal_notes",
    ])?;

    let unhide_out = cli.run(&[
        "schema",
        "hide",
        "--aggregate",
        "orders",
        "--field",
        "internal_notes",
        "--unhide",
    ])?;
    assert!(
        unhide_out.contains("aggregate=orders field=internal_notes unhidden"),
        "unexpected schema unhide output:\n{}",
        unhide_out
    );

    let schema_after_unhide = cli.run_json(&["schema", "orders"])?;
    let hidden = schema_after_unhide["hidden_fields"]
        .as_array()
        .context("hidden_fields not present in schema json")?;
    assert!(
        !hidden.iter().any(|value| value == "internal_notes"),
        "expected hidden_fields to exclude internal_notes after unhide: {}",
        schema_after_unhide
    );

    Ok(())
}

#[test]
fn schema_field_sets_type_and_rules() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&[
        "schema",
        "create",
        "contacts",
        "--events",
        "contact_created",
    ])?;

    let type_out = cli.run(&["schema", "field", "contacts", "email", "--type", "text"])?;
    assert!(
        type_out.contains("type=text"),
        "unexpected schema field type output:\n{}",
        type_out
    );

    let rules_out = cli.run(&[
        "schema",
        "field",
        "contacts",
        "email",
        "--format",
        "email",
        "--required",
    ])?;
    assert!(
        rules_out.contains("rules=updated"),
        "unexpected schema field rules output:\n{}",
        rules_out
    );

    let schema = cli.run_json(&["schema", "contacts"])?;
    let column = schema["column_types"]["email"]
        .as_object()
        .context("email column missing from schema json")?;
    assert_eq!(column.get("type"), Some(&json!("text")));
    assert_eq!(column.get("required"), Some(&json!(true)));
    assert_eq!(column.get("format"), Some(&json!("email")));

    let clear_rules = cli.run(&[
        "schema",
        "field",
        "contacts",
        "email",
        "--clear-format",
        "--not-required",
    ])?;
    assert!(
        clear_rules.contains("rules=updated"),
        "unexpected schema field clear output:\n{}",
        clear_rules
    );
    let after = cli.run_json(&["schema", "contacts"])?;
    assert_eq!(after["column_types"]["email"], json!("text"));

    Ok(())
}

#[test]
fn schema_field_clear_type_removes_column() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "orders", "--events", "order_created"])?;
    cli.run(&["schema", "field", "orders", "total", "--type", "integer"])?;

    let cleared = cli.run(&["schema", "field", "orders", "total", "--clear-type"])?;
    assert!(
        cleared.contains("type=cleared"),
        "unexpected schema field clear type output:\n{}",
        cleared
    );

    let schema = cli.run_json(&["schema", "orders"])?;
    let columns = schema["column_types"]
        .as_object()
        .context("column_types missing from schema json")?;
    assert!(
        !columns.contains_key("total"),
        "expected total column to be removed: {}",
        schema
    );
    Ok(())
}

#[test]
fn schema_alter_add_remove_flow() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "person", "--events", "person_created"])?;

    let add = cli.run(&[
        "schema",
        "alter",
        "person",
        "person_created",
        "--add",
        "first_name,last_name",
    ])?;
    assert!(
        add.contains("added=first_name,last_name"),
        "unexpected schema alter add output:\n{}",
        add
    );

    let schema = cli.run_json(&["schema", "person"])?;
    let fields = schema["events"]["person_created"]["fields"]
        .as_array()
        .context("expected fields array after add")?;
    assert!(
        fields.iter().any(|value| value == "first_name")
            && fields.iter().any(|value| value == "last_name"),
        "field list missing expected entries: {}",
        schema
    );

    let remove = cli.run(&[
        "schema",
        "alter",
        "person",
        "person_created",
        "--remove",
        "last_name",
    ])?;
    assert!(
        remove.contains("removed=last_name"),
        "unexpected schema alter remove output:\n{}",
        remove
    );

    let after_remove = cli.run_json(&["schema", "person"])?;
    let remaining = after_remove["events"]["person_created"]["fields"]
        .as_array()
        .context("expected fields array after remove")?;
    let remaining_values: Vec<_> = remaining
        .iter()
        .map(|value| value.as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(
        remaining_values,
        vec!["first_name".to_string()],
        "expected only first_name after removal: {}",
        after_remove
    );

    Ok(())
}

#[test]
fn schema_alter_set_and_clear() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "order", "--events", "order_created"])?;

    let set = cli.run(&[
        "schema",
        "alter",
        "order",
        "order_created",
        "--set",
        "order_id,status",
    ])?;
    assert!(
        set.contains("fields=order_id,status"),
        "unexpected schema alter set output:\n{}",
        set
    );

    let schema = cli.run_json(&["schema", "order"])?;
    let fields = schema["events"]["order_created"]["fields"]
        .as_array()
        .context("expected fields array after set")?;
    let set_values: Vec<_> = fields
        .iter()
        .map(|value| value.as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(
        set_values,
        vec!["order_id".to_string(), "status".to_string()],
        "expected set fields to replace list: {}",
        schema
    );

    let cleared = cli.run(&["schema", "alter", "order", "order_created", "--clear"])?;
    assert!(
        cleared.contains("fields=cleared"),
        "unexpected schema alter clear output:\n{}",
        cleared
    );

    let after_clear = cli.run_json(&["schema", "order"])?;
    let cleared_fields = after_clear["events"]["order_created"]["fields"]
        .as_array()
        .context("expected fields array after clear")?;
    assert!(
        cleared_fields.is_empty(),
        "expected fields to be empty after clear: {}",
        after_clear
    );

    Ok(())
}

#[test]
fn schema_create_duplicate_fails() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "dupe", "--events", "event_a"])?;
    let failure = cli.run_failure(&["schema", "create", "dupe", "--events", "event_b"])?;
    assert!(
        failure.stderr.contains("schema already exists"),
        "unexpected duplicate schema message:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn schema_list_after_create_contains_entry() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&[
        "schema",
        "create",
        "customers",
        "--events",
        "customer_created",
    ])?;
    let list = cli.run_json(&["schema", "list", "--json"])?;
    let array = list
        .as_array()
        .context("schema list did not return an array")?;
    assert_eq!(array.len(), 1);
    assert_eq!(array[0]["aggregate"], json!("customers"));
    Ok(())
}

#[test]
fn schema_remove_unknown_event_fails() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "inventory", "--events", "item_created"])?;
    let failure = cli.run_failure(&["schema", "remove", "inventory", "item_deleted"])?;
    assert!(
        failure
            .stderr
            .contains("event item_deleted is not defined for aggregate inventory"),
        "unexpected remove failure:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn schema_validate_unknown_event_fails() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "prices", "--events", "price_created"])?;
    let failure = cli.run_failure(&[
        "schema",
        "validate",
        "--aggregate",
        "prices",
        "--event",
        "price_updated",
        "--payload",
        "{}",
    ])?;
    assert!(
        failure
            .stderr
            .contains("event price_updated is not defined for aggregate prices"),
        "unexpected schema validate failure:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn schema_validate_invalid_json_fails() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "audit", "--events", "log_created"])?;
    let failure = cli.run_failure(&[
        "schema",
        "validate",
        "--aggregate",
        "audit",
        "--event",
        "log_created",
        "--payload",
        "{invalid",
    ])?;
    assert!(
        failure.stderr.contains("key must be a string")
            || failure.stderr.contains("expected value"),
        "unexpected invalid JSON message:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn schema_hide_empty_field_fails() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "orders", "--events", "order_created"])?;
    let failure = cli.run_failure(&["schema", "hide", "--aggregate", "orders", "--field", ""])?;
    assert!(
        failure.stderr.contains("field name cannot be empty"),
        "unexpected hide validation message:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn schema_create_empty_aggregate_fails() -> Result<()> {
    let cli = CliTest::new()?;
    let failure = cli.run_failure(&["schema", "create", "", "--events", "evt"])?;
    assert!(
        failure.stderr.contains("aggregate name must be provided"),
        "unexpected empty aggregate message:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn schema_annotate_requires_note_or_clear() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "ledger",
        "--events",
        "entry_logged",
        "--json",
    ])?;

    let failure = cli.run_failure(&["schema", "annotate", "ledger", "entry_logged"])?;
    assert!(
        failure
            .stderr
            .contains("either --note must be provided or use --clear"),
        "unexpected annotate validation message:\n{}",
        failure.stderr
    );

    Ok(())
}

#[test]
fn schema_annotate_clear_with_note_fails() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "ledger",
        "--events",
        "entry_logged",
        "--json",
    ])?;
    let failure = cli.run_failure(&[
        "schema",
        "annotate",
        "ledger",
        "entry_logged",
        "--note",
        "example",
        "--clear",
    ])?;
    assert!(
        failure
            .stderr
            .contains("--note cannot be combined with --clear"),
        "unexpected annotate clear message:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn schema_hide_field_idempotent_and_error_cases() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "billing",
        "--events",
        "invoice_created",
        "--json",
    ])?;

    let hide_again = cli.run(&[
        "schema",
        "hide",
        "--aggregate",
        "billing",
        "--field",
        "secret_code",
    ])?;
    assert!(
        hide_again.contains("aggregate=billing field=secret_code hidden"),
        "unexpected schema hide output:\n{}",
        hide_again
    );

    let schema_hidden = cli.run_json(&["schema", "billing"])?;
    let hidden = schema_hidden["hidden_fields"]
        .as_array()
        .context("hidden_fields not present after hide")?;
    assert!(
        hidden.iter().any(|value| value == "secret_code"),
        "expected hidden_fields to contain secret_code: {}",
        schema_hidden
    );

    let hide_duplicate = cli.run(&[
        "schema",
        "hide",
        "--aggregate",
        "billing",
        "--field",
        "secret_code",
    ])?;
    assert!(
        hide_duplicate.contains("aggregate=billing field=secret_code hidden"),
        "duplicate hide should still report hidden:\n{}",
        hide_duplicate
    );

    let failure = cli.run_failure(&[
        "schema",
        "hide",
        "--aggregate",
        "",
        "--field",
        "secret_code",
    ])?;
    assert!(
        failure.stderr.contains("aggregate name cannot be empty"),
        "expected empty aggregate error, got:\n{}",
        failure.stderr
    );

    Ok(())
}

#[test]
fn schema_show_outputs_json() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "customers",
        "--events",
        "customer_created",
        "--json",
    ])?;
    let schema = cli.run_json(&["schema", "customers"])?;
    assert_eq!(schema["aggregate"], json!("customers"));
    assert!(schema["events"].get("customer_created").is_some());
    Ok(())
}

#[test]
fn backup_output_inside_data_dir_fails() -> Result<()> {
    let cli = CliTest::new()?;
    let output_path = cli.data_dir().join("backup.tar.gz");
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let failure =
        cli.run_failure(&["backup", "--output", output_path.to_string_lossy().as_ref()])?;
    assert!(
        failure.stderr.contains("backup output path")
            && failure.stderr.contains("resides inside the data directory"),
        "unexpected backup path failure:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn backup_creates_archive() -> Result<()> {
    let cli = CliTest::new()?;
    let output_path = cli.home.join("backup.tar.gz");
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let stdout = cli.run(&["backup", "--output", output_path.to_string_lossy().as_ref()])?;
    assert!(
        stdout.contains("Backup created at"),
        "unexpected backup output:\n{}",
        stdout
    );
    assert!(output_path.exists(), "backup archive was not created");
    Ok(())
}

#[test]
fn restore_missing_archive_fails() -> Result<()> {
    let cli = CliTest::new()?;
    let missing = cli.home.join("missing.tar.gz");
    let failure = cli.run_failure(&["restore", "--input", missing.to_string_lossy().as_ref()])?;
    assert!(
        failure.stderr.contains("backup archive") && failure.stderr.contains("does not exist"),
        "unexpected restore failure:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn config_prints_toml_when_no_updates() -> Result<()> {
    let cli = CliTest::new()?;
    let stdout = cli.run(&["config"])?;
    assert!(
        stdout.contains("data_dir") && stdout.contains("port"),
        "unexpected config output:\n{}",
        stdout
    );
    Ok(())
}

#[test]
fn config_updates_port_writes_file() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["config", "--port", "8080"])?;
    let config_path = cli.data_dir().join("config.toml");
    let contents = fs::read_to_string(config_path)?;
    assert!(
        contents.contains("port = 8080"),
        "expected config to include updated port: {}",
        contents
    );
    Ok(())
}

#[test]
fn token_revoke_updates_status() -> Result<()> {
    let cli = CliTest::new()?;
    let record = cli.run_json(&[
        "token",
        "generate",
        "--group",
        "ops",
        "--user",
        "alice",
        "--action",
        "aggregate.read",
        "--tenant",
        "default",
        "--json",
    ])?;
    let token = record["token"]
        .as_str()
        .context("token generate output missing token field")?
        .to_string();

    let revoke = cli.run_json(&["token", "revoke", &token, "--reason", "cleanup", "--json"])?;
    assert_eq!(revoke["token"], json!(token));
    assert_eq!(revoke["revoked"], json!(true));
    assert_eq!(revoke["reason"], json!("cleanup"));

    let list = cli.run_json(&["token", "list", "--json"])?;
    let array = list
        .as_array()
        .context("token list json did not return array")?;
    assert_eq!(array.len(), 1);
    assert_eq!(array[0]["status"], json!("revoked"));

    Ok(())
}

#[test]
fn tenant_assign_list_and_unassign_flow() -> Result<()> {
    let cli = CliTest::new()?;

    cli.run(&["tenant", "assign", "people", "--shard", "shard-0003"])?;
    cli.run(&["tenant", "assign", "billing", "--shard", "4"])?;

    let list_all = cli.run_json(&["tenant", "list", "--json"])?;
    let entries = list_all
        .as_array()
        .context("tenant list did not return an array")?;
    assert_eq!(
        entries.len(),
        2,
        "expected two tenant assignments: {entries:?}"
    );
    assert!(
        entries.iter().any(|entry| {
            entry["tenant"] == json!("people")
                && entry["shard"] == json!("shard-0003")
                && entry["quota_mb"].is_null()
        }),
        "missing people assignment: {entries:?}"
    );
    assert!(
        entries
            .iter()
            .any(|entry| entry["tenant"] == json!("billing")
                && entry["shard"] == json!("shard-0004")
                && entry["quota_mb"].is_null()),
        "missing billing assignment: {entries:?}"
    );

    let filtered = cli.run_json(&["tenant", "list", "--json", "--shard", "4"])?;
    let filtered_array = filtered
        .as_array()
        .context("filtered tenant list did not return an array")?;
    assert_eq!(filtered_array.len(), 1, "expected one entry for shard-0004");
    assert_eq!(filtered_array[0]["tenant"], json!("billing"));
    assert!(filtered_array[0]["quota_mb"].is_null());

    cli.run(&["tenant", "unassign", "people"])?;
    let after_unassign = cli.run_json(&["tenant", "list", "--json"])?;
    let remaining = after_unassign
        .as_array()
        .context("tenant list did not return an array after unassign")?;
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0]["tenant"], json!("billing"));
    assert_eq!(remaining[0]["shard"], json!("shard-0004"));
    assert!(remaining[0]["quota_mb"].is_null());
    Ok(())
}

#[test]
fn tenant_stats_reports_counts() -> Result<()> {
    let cli = CliTest::new()?;

    cli.run(&["tenant", "assign", "alpha", "--shard", "1"])?;
    cli.run(&["tenant", "assign", "bravo", "--shard", "1"])?;
    cli.run(&["tenant", "assign", "charlie", "--shard", "2"])?;

    let stats = cli.run_json(&["tenant", "stats", "--json"])?;
    let stats_map = stats
        .as_object()
        .context("tenant stats did not return an object")?;
    assert_eq!(stats_map.get("shard-0001").and_then(Value::as_u64), Some(2));
    assert_eq!(stats_map.get("shard-0002").and_then(Value::as_u64), Some(1));

    cli.run(&["tenant", "unassign", "alpha"])?;
    let after = cli.run_json(&["tenant", "stats", "--json"])?;
    let after_map = after
        .as_object()
        .context("tenant stats did not return an object after unassign")?;
    assert_eq!(after_map.get("shard-0001").and_then(Value::as_u64), Some(1));
    assert_eq!(after_map.get("shard-0002").and_then(Value::as_u64), Some(1));
    Ok(())
}

#[test]
fn tenant_schema_publish_records_history() -> Result<()> {
    let cli = CliTest::new()?;

    cli.run(&[
        "schema",
        "create",
        "orders",
        "--events",
        "order_created",
        "--json",
    ])?;

    let publish_output = cli.run(&[
        "tenant",
        "schema",
        "publish",
        "default",
        "--activate",
        "--reason",
        "initial load",
        "--no-reload",
    ])?;
    assert!(
        publish_output.contains("published version="),
        "unexpected publish output: {}",
        publish_output
    );

    let manifest = cli.run_json(&["tenant", "schema", "history", "default", "--json"])?;
    let active = manifest["active_version"]
        .as_str()
        .context("history output missing active_version")?;
    let versions = manifest["versions"]
        .as_array()
        .context("history output missing versions array")?;
    assert_eq!(versions.len(), 1, "expected a single schema version");
    assert_eq!(
        versions[0]["id"],
        json!(active),
        "first version id should be active"
    );
    assert_eq!(
        versions[0]["reason"],
        json!("initial load"),
        "version reason should match publish flag"
    );
    let audit = manifest["audit_log"]
        .as_array()
        .context("history output missing audit log")?;
    assert_eq!(
        audit.len(),
        2,
        "expected publish + activate audit entries, got {:?}",
        audit
    );
    assert_eq!(audit[0]["action"], json!("publish"));
    assert_eq!(audit[1]["action"], json!("activate"));
    Ok(())
}

#[test]
fn tenant_schema_diff_and_rollback_flow() -> Result<()> {
    let cli = CliTest::new()?;

    cli.run(&[
        "schema",
        "create",
        "orders",
        "--events",
        "order_created",
        "--json",
    ])?;
    cli.run(&[
        "tenant",
        "schema",
        "publish",
        "default",
        "--activate",
        "--no-reload",
    ])?;

    cli.run(&["schema", "add", "orders", "order_shipped"])?;
    cli.run(&[
        "tenant",
        "schema",
        "publish",
        "default",
        "--activate",
        "--reason",
        "add shipped",
        "--no-reload",
    ])?;

    let manifest = cli.run_json(&["tenant", "schema", "history", "default", "--json"])?;
    let versions = manifest["versions"]
        .as_array()
        .context("expected versions array after second publish")?;
    assert!(
        versions.len() >= 2,
        "history should include at least two versions: {:?}",
        versions
    );
    let first_id = versions
        .first()
        .and_then(|value| value["id"].as_str())
        .context("missing id for first version")?
        .to_string();
    let latest_id = versions
        .last()
        .and_then(|value| value["id"].as_str())
        .context("missing id for latest version")?
        .to_string();
    assert_ne!(
        first_id, latest_id,
        "second publish should create new version"
    );

    let diff = cli.run_json(&[
        "tenant",
        "schema",
        "diff",
        "default",
        "--from",
        first_id.as_str(),
        "--to",
        latest_id.as_str(),
        "--json",
    ])?;
    let operations = diff
        .as_array()
        .context("diff output was not a JSON array")?;
    assert!(
        !operations.is_empty(),
        "expected diff to include at least one operation"
    );

    let unified_output = cli.run(&[
        "tenant",
        "schema",
        "diff",
        "default",
        "--from",
        first_id.as_str(),
        "--to",
        latest_id.as_str(),
        "--style",
        "unified",
        "--color",
        "always",
    ])?;
    assert!(
        (unified_output.contains("@@") || unified_output.contains("+++ "))
            && unified_output.contains("\u{1b}[32m")
            && unified_output.contains("\u{1b}[31m"),
        "unified diff output did not include expected formatting:\n{}",
        unified_output
    );

    let split_output = cli.run(&[
        "tenant",
        "schema",
        "diff",
        "default",
        "--from",
        first_id.as_str(),
        "--to",
        latest_id.as_str(),
        "--style",
        "split",
        "--color",
        "always",
    ])?;
    assert!(
        split_output.contains('|') && split_output.contains("schema@"),
        "split diff output missing expected columns:\n{}",
        split_output
    );

    let rollback_output = cli.run(&[
        "tenant",
        "schema",
        "rollback",
        "default",
        "--version",
        first_id.as_str(),
        "--no-reload",
    ])?;
    assert!(
        rollback_output.contains("rolled back") || rollback_output.contains("already active"),
        "unexpected rollback output: {}",
        rollback_output
    );

    let post_manifest = cli.run_json(&["tenant", "schema", "history", "default", "--json"])?;
    assert_eq!(
        post_manifest["active_version"],
        json!(first_id),
        "rollback should update active version"
    );
    Ok(())
}

#[test]
fn tenant_quota_limits_aggregate_creation() -> Result<()> {
    let cli = CliTest::new()?;

    cli.run(&["config", "--multi-tenant", "true"])?;

    cli.run(&["tenant", "quota", "set", "default", "--max-mb", "1"])?;

    let list = cli.run_json(&["tenant", "list", "--json"])?;
    let quota_entry = list
        .as_array()
        .context("tenant list did not return an array after quota set")?
        .iter()
        .find(|entry| entry["tenant"] == json!("default"))
        .cloned()
        .context("missing default tenant entry with quota")?;
    assert_eq!(quota_entry["quota_mb"], json!(1));

    cli.run(&["schema", "create", "order", "--events", "order_created"])?;

    let payload_blob = "x".repeat((MAX_EVENT_PAYLOAD_BYTES.saturating_sub(32)) as usize);
    let payload_json = format!(r#"{{"blob":"{}"}}"#, payload_blob);

    let mut quota_hit = false;
    for index in 0..10 {
        let aggregate_id = format!("order-{}", index);
        let args = vec![
            "aggregate",
            "create",
            "order",
            aggregate_id.as_str(),
            "--event",
            "order_created",
            "--payload",
            payload_json.as_str(),
            "--json",
        ];
        let output = cli.exec(&args)?;
        if output.status.success() {
            continue;
        }
        let stderr = normalize_output(output.stderr)?;
        assert!(
            stderr.contains("quota"),
            "unexpected failure once quota tripped:\n{}",
            stderr
        );
        quota_hit = true;
        break;
    }
    assert!(quota_hit, "expected storage quota to block new events");

    cli.run(&["tenant", "quota", "clear", "default"])?;
    let list_after_clear = cli.run_json(&["tenant", "list", "--json"])?;
    assert!(
        !list_after_clear
            .as_array()
            .context("tenant list did not return an array after quota clear")?
            .iter()
            .any(|entry| entry["tenant"] == json!("default")),
        "default tenant entry should disappear once quota is cleared"
    );
    cli.create_aggregate("order", "order-99", "order_created")?;
    Ok(())
}

#[test]
fn plugin_http_config_enable_disable() -> Result<()> {
    let cli = CliTest::new()?;

    let output = cli.run(&[
        "plugin",
        "config",
        "http",
        "--name",
        "hook",
        "--endpoint",
        "http://127.0.0.1:9",
        "--header",
        "X-Test=ok",
        "--disable",
    ])?;
    assert!(
        output.contains("HTTP plugin 'hook' disabled"),
        "unexpected config output:\n{}",
        output
    );

    let list = cli.run(&["plugin", "list"])?;
    assert!(
        list.contains("http (hook) - disabled (enable with plugin enable hook)"),
        "unexpected plugin list output:\n{}",
        list
    );

    let failure = cli.run_failure(&["plugin", "enable", "hook"])?;
    assert!(
        failure
            .stderr
            .contains("failed to establish connection for 'hook'"),
        "unexpected enable failure message:\n{}",
        failure.stderr
    );

    let plugins = cli.run_json(&["plugin", "list", "--json"])?;
    let array = plugins
        .as_array()
        .context("plugin list json did not return array")?;
    assert_eq!(array.len(), 1);
    let plugin = &array[0];
    assert_eq!(plugin["enabled"], json!(false));
    assert_eq!(plugin["name"], json!("hook"));
    assert_eq!(plugin["config"]["type"], json!("http"));
    assert_eq!(plugin["config"]["endpoint"], json!("http://127.0.0.1:9"));
    assert_eq!(plugin["config"]["headers"], json!({ "X-Test": "ok" }));

    Ok(())
}

cli_failure_test!(
    plugin_disable_unknown_fails,
    ["plugin", "disable", "ghost"],
    "no plugin named 'ghost' is configured"
);

cli_failure_test!(
    plugin_config_process_requires_installation,
    [
        "plugin",
        "config",
        "process",
        "--name",
        "worker",
        "--plugin",
        "search",
        "--version",
        "1.2.3"
    ],
    "plugin 'search' version 1.2.3 is not installed"
);

cli_success_contains_test!(
    plugin_test_reports_empty,
    ["plugin", "test"],
    "(no plugins enabled)"
);

#[test]
fn plugin_enable_unknown_fails() -> Result<()> {
    let cli = CliTest::new()?;
    let failure = cli.run_failure(&["plugin", "enable", "ghost"])?;
    assert!(
        failure
            .stderr
            .contains("no plugin named 'ghost' is configured"),
        "unexpected enable failure message:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn plugin_remove_unknown_fails() -> Result<()> {
    let cli = CliTest::new()?;
    let failure = cli.run_failure(&["plugin", "remove", "ghost"])?;
    assert!(
        failure
            .stderr
            .contains("no plugin named 'ghost' is configured"),
        "unexpected remove failure message:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn plugin_config_log_sets_template() -> Result<()> {
    let cli = CliTest::new()?;
    let out = cli.run(&[
        "plugin",
        "config",
        "log",
        "--name",
        "audit",
        "--level",
        "debug",
        "--template",
        "[{aggregate}] {event}",
    ])?;
    assert!(
        out.contains("Log plugin 'audit' configured"),
        "unexpected log config output:\n{}",
        out
    );

    let list = cli.run_json(&["plugin", "list", "--json"])?;
    let array = list
        .as_array()
        .context("log plugin list did not return array")?;
    assert_eq!(array.len(), 1);
    let plugin = &array[0];
    assert_eq!(plugin["config"]["type"], json!("log"));
    assert_eq!(plugin["config"]["level"], json!("debug"));
    assert_eq!(plugin["config"]["template"], json!("[{aggregate}] {event}"));
    Ok(())
}

#[test]
fn plugin_config_tcp_sets_values() -> Result<()> {
    let cli = CliTest::new()?;
    let out = cli.run(&[
        "plugin",
        "config",
        "tcp",
        "--name",
        "sink",
        "--host",
        "127.0.0.1",
        "--port",
        "7000",
        "--disable",
    ])?;
    assert!(
        out.contains("TCP plugin 'sink' disabled"),
        "unexpected tcp config output:\n{}",
        out
    );

    let list = cli.run_json(&["plugin", "list", "--json"])?;
    let array = list
        .as_array()
        .context("tcp plugin list did not return array")?;
    assert_eq!(array.len(), 1);
    let plugin = &array[0];
    assert_eq!(plugin["config"]["type"], json!("tcp"));
    assert_eq!(plugin["config"]["host"], json!("127.0.0.1"));
    assert_eq!(plugin["config"]["port"], json!(7000));
    Ok(())
}

#[test]
fn plugin_disable_already_disabled() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&[
        "plugin",
        "config",
        "http",
        "--name",
        "hook",
        "--endpoint",
        "http://127.0.0.1:9",
        "--disable",
    ])?;
    let output = cli.run(&["plugin", "disable", "hook"])?;
    assert!(
        output.contains("Plugin 'hook' is already disabled"),
        "unexpected disable repeat message:\n{}",
        output
    );
    Ok(())
}

#[test]
fn plugin_config_http_enables_plugin() -> Result<()> {
    let cli = CliTest::new()?;
    let output = cli.run(&[
        "plugin",
        "config",
        "http",
        "--name",
        "hooks",
        "--endpoint",
        "http://127.0.0.1:8080",
        "--header",
        "X-App=test",
    ])?;
    assert!(
        output.contains("HTTP plugin 'hooks' configured"),
        "unexpected http config output:\n{}",
        output
    );
    let list = cli.run(&["plugin", "list"])?;
    assert!(
        list.contains("http (hooks) - enabled"),
        "unexpected plugin list output:\n{}",
        list
    );
    Ok(())
}

#[test]
fn plugin_list_multiple_entries() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&[
        "plugin", "config", "log", "--name", "audit", "--level", "info",
    ])?;
    cli.run(&[
        "plugin",
        "config",
        "http",
        "--name",
        "hooks",
        "--endpoint",
        "http://127.0.0.1:8080",
        "--disable",
    ])?;
    let list = cli.run_json(&["plugin", "list", "--json"])?;
    let array = list
        .as_array()
        .context("plugin list did not return array")?;
    assert_eq!(array.len(), 2);
    Ok(())
}

#[test]
fn aggregate_get_unknown_fails() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&[
        "schema",
        "create",
        "existing",
        "--events",
        "existing_created",
    ])?;
    cli.create_aggregate("existing", "existing-1", "existing_created")?;
    cli.run(&[
        "aggregate",
        "apply",
        "existing",
        "existing-1",
        "existing_created",
        "--field",
        "status=ok",
    ])?;
    let failure = cli.run_failure(&["aggregate", "get", "existing", "missing-id"])?;
    assert!(
        failure.stderr.contains("aggregate not found"),
        "unexpected aggregate get error:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn aggregate_remove_unknown_fails() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&[
        "schema",
        "create",
        "existing",
        "--events",
        "existing_created",
    ])?;
    cli.create_aggregate("existing", "existing-1", "existing_created")?;
    cli.run(&[
        "aggregate",
        "apply",
        "existing",
        "existing-1",
        "existing_created",
        "--field",
        "status=ok",
    ])?;
    let failure = cli.run_failure(&["aggregate", "remove", "existing", "missing-id"])?;
    assert!(
        failure.stderr.contains("aggregate not found"),
        "unexpected aggregate remove error:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn aggregate_verify_unknown_fails() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&[
        "schema",
        "create",
        "existing",
        "--events",
        "existing_created",
    ])?;
    cli.create_aggregate("existing", "existing-1", "existing_created")?;
    cli.run(&[
        "aggregate",
        "apply",
        "existing",
        "existing-1",
        "existing_created",
        "--field",
        "status=ok",
    ])?;
    let failure = cli.run_failure(&["aggregate", "verify", "existing", "missing-id"])?;
    assert!(
        failure.stderr.contains("aggregate not found"),
        "unexpected aggregate verify error:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn aggregate_apply_requires_schema_in_strict_mode() -> Result<()> {
    let cli = CliTest::new()?;
    let _ = cli.run(&["config"])?;
    let config_path = cli.home.join(".eventdbx").join("config.toml");
    let mut config_contents =
        fs::read_to_string(&config_path).context("failed to read config.toml")?;
    config_contents = config_contents.replace("restrict = \"default\"", "restrict = \"strict\"");
    fs::write(&config_path, config_contents).context("failed to update restrict mode")?;
    let failure = cli.run_failure(&[
        "aggregate",
        "apply",
        "order",
        "order-1",
        "order_created",
        "--field",
        "status=created",
    ])?;
    assert!(
        failure
            .stderr
            .contains("aggregate order requires a schema before events can be appended"),
        "missing schema error not reported\nstdout:\n{}\nstderr:\n{}",
        failure.stdout,
        failure.stderr
    );
    Ok(())
}

#[test]
fn aggregate_apply_allows_missing_schema_in_default_mode() -> Result<()> {
    let cli = CliTest::new()?;
    cli.create_aggregate("order", "order-1", "order_created")?;
    cli.run(&[
        "aggregate",
        "apply",
        "order",
        "order-1",
        "order_created",
        "--field",
        "status=created",
    ])?;
    Ok(())
}

#[test]
fn aggregate_apply_without_create_requires_flag() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "legacy",
        "--events",
        "legacy_created",
        "--json",
    ])?;

    let failure = cli.run_failure(&[
        "aggregate",
        "apply",
        "legacy",
        "legacy-1",
        "legacy_created",
        "--field",
        "status=pending",
    ])?;
    assert!(
        failure.stderr.contains("does not exist"),
        "unexpected error when applying without create:\n{}",
        failure.stderr
    );

    cli.create_aggregate_with_fields(
        "legacy",
        "legacy-1",
        "legacy_created",
        &[("status", "pending")],
    )?;
    let state = cli.run_json(&["aggregate", "get", "legacy", "legacy-1", "--include-events"])?;
    assert_eq!(state["aggregate_type"], json!("legacy"));
    assert_eq!(state["aggregate_id"], json!("legacy-1"));
    assert_eq!(state["version"], json!(1));
    assert_eq!(state["state"]["status"], json!("pending"));
    Ok(())
}

#[test]
fn aggregate_create_outputs_state() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "patient",
        "--events",
        "patient_created",
        "--json",
    ])?;

    let state = cli.run_json(&[
        "aggregate",
        "create",
        "patient",
        "patient-1",
        "--event",
        "patient_created",
        "--json",
    ])?;
    assert_eq!(state["aggregate_type"], json!("patient"));
    assert_eq!(state["aggregate_id"], json!("patient-1"));
    assert_eq!(state["version"], json!(1));
    assert!(state["state"].as_object().unwrap().is_empty());
    Ok(())
}

#[test]
fn aggregate_list_json_empty() -> Result<()> {
    let cli = CliTest::new()?;
    let output = cli.run_json(&["aggregate", "list", "--json"])?;
    let array = output
        .as_array()
        .context("aggregate list did not return an array")?;
    assert!(array.is_empty(), "expected no aggregates, got {}", output);
    Ok(())
}

cli_failure_test!(
    aggregate_list_resolve_requires_json,
    ["aggregate", "list", "--resolve"],
    "--resolve requires --json output"
);

#[test]
fn aggregate_list_accepts_positional_type() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "patient", "--events", "patient_created"])?;
    cli.run(&["schema", "create", "order", "--events", "order_created"])?;

    cli.create_aggregate("patient", "patient-1", "patient_created")?;
    cli.run(&[
        "aggregate",
        "apply",
        "patient",
        "patient-1",
        "patient_created",
        "--payload",
        r#"{"status":"active"}"#,
    ])?;
    cli.create_aggregate("order", "order-1", "order_created")?;

    let patient_list = cli.run_json(&["aggregate", "list", "patient", "--json"])?;
    let patient_array = patient_list
        .as_array()
        .context("patient aggregate list did not return an array")?;
    assert_eq!(patient_array.len(), 1);
    assert!(
        patient_array
            .iter()
            .all(|entry| entry["aggregate_type"] == "patient")
    );

    let order_list = cli.run_json(&["aggregate", "list", "order", "--json"])?;
    let order_array = order_list
        .as_array()
        .context("order aggregate list did not return an array")?;
    assert_eq!(order_array.len(), 1);
    assert!(
        order_array
            .iter()
            .all(|entry| entry["aggregate_type"] == "order")
    );

    Ok(())
}

#[test]
fn aggregate_list_resolves_references() -> Result<()> {
    let cli = CliTest::new()?;
    // Schemas
    cli.run(&["schema", "create", "address", "--events", "address_created"])?;
    cli.run(&["schema", "create", "farm", "--events", "farm_created"])?;
    cli.run(&["schema", "field", "--type", "reference", "farm", "address"])?;

    // Target aggregate
    cli.create_aggregate("address", "addr-1", "address_created")?;
    // Aggregate with reference (same domain shorthand)
    cli.run(&[
        "aggregate",
        "create",
        "farm",
        "farm-1",
        "--event",
        "farm_created",
        "--field",
        "address=address#addr-1",
    ])?;

    let output = cli.run_json(&["aggregate", "list", "farm", "--json", "--resolve"])?;
    let array = output
        .as_array()
        .context("aggregate list did not return an array")?;
    assert_eq!(array.len(), 1);
    let resolved = &array[0];
    assert_eq!(resolved["aggregate"]["aggregate_id"], json!("farm-1"));

    let references = resolved["references"]
        .as_array()
        .context("resolved references missing")?;
    assert_eq!(references.len(), 1);
    assert_eq!(references[0]["status"], json!("ok"));
    assert_eq!(
        references[0]["reference"],
        json!({"domain":"default","aggregate_type":"address","aggregate_id":"addr-1"})
    );
    assert_eq!(
        references[0]["resolved"]
            .get("aggregate")
            .and_then(|agg| agg.get("aggregate_type"))
            .cloned(),
        Some(json!("address"))
    );

    Ok(())
}

#[test]
fn aggregate_referrers_reports_sources() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "address", "--events", "address_created"])?;
    cli.run(&["schema", "create", "farm", "--events", "farm_created"])?;
    cli.run(&["schema", "field", "--type", "reference", "farm", "address"])?;

    cli.create_aggregate("address", "addr-1", "address_created")?;
    cli.run(&[
        "aggregate",
        "create",
        "farm",
        "farm-1",
        "--event",
        "farm_created",
        "--field",
        "address=address#addr-1",
    ])?;

    let refs = cli.run_json(&["aggregate", "referrers", "address", "addr-1", "--json"])?;
    let arr = refs
        .as_array()
        .context("referrers output was not an array")?;
    assert_eq!(arr.len(), 1, "expected a single referrer: {}", refs);
    assert_eq!(arr[0]["aggregate_type"], json!("farm"));
    assert_eq!(arr[0]["aggregate_id"], json!("farm-1"));
    assert!(
        arr[0]["path"]
            .as_str()
            .map(|path| path.contains("address"))
            .unwrap_or(false),
        "expected path to mention address field: {}",
        refs
    );

    Ok(())
}

#[test]
fn aggregate_archive_nullifies_references() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "address", "--events", "address_created"])?;
    cli.run(&["schema", "create", "farm", "--events", "farm_created"])?;
    cli.run(&[
        "schema",
        "field",
        "--type",
        "reference",
        "--reference-cascade",
        "nullify",
        "farm",
        "address",
    ])?;

    cli.create_aggregate("address", "addr-1", "address_created")?;
    cli.run(&[
        "aggregate",
        "create",
        "farm",
        "farm-1",
        "--event",
        "farm_created",
        "--field",
        "address=address#addr-1",
    ])?;

    cli.run(&["aggregate", "archive", "address", "addr-1"])?;
    let farms = cli.run_json(&["aggregate", "list", "farm", "--json"])?;
    let farm = farms
        .as_array()
        .and_then(|arr| arr.first())
        .cloned()
        .context("expected farm aggregate after archive")?;
    assert_eq!(farm["state"]["address"], json!(""));

    Ok(())
}

#[test]
fn aggregate_commit_no_staged_events() -> Result<()> {
    let cli = CliTest::new()?;
    let output = cli.run(&["aggregate", "commit"])?;
    assert!(
        output.contains("no staged events to commit"),
        "unexpected commit output:\n{}",
        output
    );
    Ok(())
}

#[test]
fn aggregate_apply_metadata_invalid_key_fails() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&[
        "schema",
        "create",
        "metadata",
        "--events",
        "metadata_created",
    ])?;
    cli.create_aggregate("metadata", "id-1", "metadata_created")?;
    let failure = cli.run_failure(&[
        "aggregate",
        "apply",
        "metadata",
        "id-1",
        "metadata_created",
        "--payload",
        "{}",
        "--metadata",
        "{\"foo\":1}",
    ])?;
    assert!(
        failure
            .stderr
            .contains("metadata key 'foo' must start with '@'"),
        "unexpected metadata validation message:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn aggregate_apply_note_too_long_fails() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "notes", "--events", "note_created"])?;
    cli.create_aggregate("notes", "id-1", "note_created")?;
    let long_note = "n".repeat(129);
    let failure = cli.run_failure(&[
        "aggregate",
        "apply",
        "notes",
        "id-1",
        "note_created",
        "--field",
        "status=ok",
        "--note",
        &long_note,
    ])?;
    assert!(
        failure.stderr.contains("note cannot exceed"),
        "unexpected note length message:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn aggregate_list_stage_empty_shows_message() -> Result<()> {
    let cli = CliTest::new()?;
    let stdout = cli.run(&["aggregate", "list", "--stage"])?;
    assert_eq!(stdout.trim(), "no staged events");
    Ok(())
}

#[test]
fn aggregate_stage_accepts_first_event_without_created_suffix() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "order",
        "--events",
        "order_created,order_updated",
        "--json",
    ])?;

    cli.create_aggregate("order", "order-1", "order_created")?;
    let staged = cli.run(&[
        "aggregate",
        "apply",
        "order",
        "order-1",
        "order_updated",
        "--field",
        "status=processing",
        "--stage",
    ])?;
    assert!(
        staged.contains("event staged for later commit"),
        "unexpected staging output:\n{}",
        staged
    );

    let staged_list = cli.run(&["aggregate", "list", "--stage"])?;
    assert!(
        staged_list.contains("\"event\": \"order_updated\""),
        "expected staged list to include order_updated event:\n{}",
        staged_list
    );
    Ok(())
}

#[test]
fn aggregate_apply_payload_and_field_conflict() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "conflict",
        "--events",
        "conflict_created",
        "--json",
    ])?;
    cli.create_aggregate("conflict", "id-1", "conflict_created")?;
    let failure = cli.run_failure(&[
        "aggregate",
        "apply",
        "conflict",
        "id-1",
        "conflict_created",
        "--payload",
        r#"{"status":"ok"}"#,
        "--field",
        "status=ok",
    ])?;
    assert!(
        failure
            .stderr
            .contains("--payload cannot be used together with --field"),
        "unexpected aggregate apply validation message:\n{}",
        failure.stderr
    );
    Ok(())
}

#[test]
fn aggregate_apply_and_get_flow() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "order",
        "--events",
        "order_created,order_updated",
        "--json",
    ])?;

    cli.create_aggregate_with_fields(
        "order",
        "order-1",
        "order_created",
        &[("status", "pending")],
    )?;
    let record = cli.run_json(&[
        "aggregate",
        "apply",
        "order",
        "order-1",
        "order_updated",
        "--field",
        "status=created",
    ])?;
    assert_eq!(record["aggregate_type"], "order");
    assert_eq!(record["aggregate_id"], "order-1");
    assert_eq!(record["event_type"], "order_updated");
    assert_eq!(record["payload"]["status"], "created");
    assert_eq!(record["version"], 2);

    let detail = cli.run_json(&["aggregate", "get", "order", "order-1", "--include-events"])?;
    assert_eq!(detail["aggregate_type"], "order");
    assert_eq!(detail["aggregate_id"], "order-1");
    assert_eq!(detail["version"], 2);
    assert_eq!(detail["state"]["status"], "created");
    let events = detail["events"]
        .as_array()
        .context("aggregate get response missing events array")?;
    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["event_type"], "order_created");
    assert_eq!(events[0]["payload"]["status"], "pending");
    assert_eq!(events[1]["event_type"], "order_updated");
    assert_eq!(events[1]["payload"]["status"], "created");

    let aggregates = cli.run_json(&["aggregate", "list", "--json"])?;
    let list = aggregates
        .as_array()
        .context("aggregate list did not return an array")?;
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["aggregate_type"], "order");
    assert_eq!(list[0]["aggregate_id"], "order-1");
    assert_eq!(list[0]["version"], 2);

    Ok(())
}

#[test]
fn aggregate_get_includes_extension_metadata_inline() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "person",
        "--events",
        "person_updated",
        "--json",
    ])?;

    cli.run_json(&[
        "aggregate",
        "create",
        "person",
        "person-1",
        "--event",
        "person.updated",
        "--metadata",
        r#"{"@trace":{"id":"abc123"},"@channel":"beta"}"#,
        "--json",
    ])?;

    let detail = cli.run_json(&["aggregate", "get", "person", "person-1", "--include-events"])?;
    let events = detail["events"]
        .as_array()
        .context("aggregate get response missing events array")?;
    assert_eq!(events.len(), 1);
    let event = events[0]
        .as_object()
        .context("event should render as an object")?;
    assert!(
        !event.contains_key("extensions"),
        "extensions should be folded into metadata: {event:?}"
    );
    assert!(
        !event.contains_key("event_type_raw"),
        "raw event type should be hidden when rendering events: {event:?}"
    );
    let metadata = event["metadata"]
        .as_object()
        .context("event metadata should be present")?;
    assert_eq!(metadata.get("@channel"), Some(&json!("beta")));
    assert_eq!(
        metadata
            .get("@trace")
            .and_then(Value::as_object)
            .and_then(|trace| trace.get("id")),
        Some(&json!("abc123"))
    );
    assert!(metadata.contains_key("created_at"));
    Ok(())
}

#[test]
fn aggregate_patch_and_select_flow() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&[
        "schema",
        "create",
        "person",
        "--events",
        "person_created,person_updated",
    ])?;

    cli.create_aggregate_with_payload(
        "person",
        "person-1",
        "person_created",
        r#"{"status":"active","profile":{"name":{"first":"Ada"}}}"#,
    )?;

    let patch_ops = r#"[{"op":"replace","path":"/status","value":"inactive"},{"op":"replace","path":"/profile/name/first","value":"Grace"}]"#;
    let record = cli.run_json(&[
        "aggregate",
        "patch",
        "person",
        "person-1",
        "person_updated",
        "--patch",
        patch_ops,
    ])?;
    assert_eq!(record["aggregate_type"], "person");
    assert_eq!(record["aggregate_id"], "person-1");
    assert_eq!(record["event_type"], "person_updated");
    assert_eq!(record["version"], 2);
    assert_eq!(record["payload"]["status"], "inactive");
    assert_eq!(record["payload"]["profile"]["name"]["first"], "Grace");

    let selection = cli.run_json(&[
        "aggregate",
        "select",
        "person",
        "person-1",
        "status",
        "profile.name.first",
    ])?;
    assert_eq!(selection["aggregate_type"], "person");
    assert_eq!(selection["aggregate_id"], "person-1");
    assert_eq!(selection["version"], 2);
    let map = selection["selection"]
        .as_object()
        .context("selection output missing selection map")?;
    assert_eq!(map["status"], json!("inactive"));
    assert_eq!(map["profile.name.first"], json!("Grace"));

    Ok(())
}

#[test]
fn aggregate_list_supports_sorting() -> Result<()> {
    let cli = CliTest::new()?;

    cli.create_aggregate_with_fields(
        "order",
        "order-1",
        "order_created",
        &[("status", "created")],
    )?;
    cli.create_aggregate_with_fields(
        "order",
        "order-2",
        "order_created",
        &[("status", "created")],
    )?;
    cli.create_aggregate_with_fields(
        "invoice",
        "invoice-1",
        "invoice_created",
        &[("total", "100")],
    )?;
    cli.run(&[
        "aggregate",
        "apply",
        "order",
        "order-1",
        "order_updated",
        "--field",
        "status=confirmed",
    ])?;

    let aggregates = cli.run_json(&[
        "aggregate",
        "list",
        "--json",
        "--sort",
        "aggregate_type:asc,aggregate_id:desc",
    ])?;
    let list = aggregates
        .as_array()
        .context("sorted aggregate list did not return an array")?;
    assert_eq!(list.len(), 3, "expected three aggregates in list");
    assert_eq!(list[0]["aggregate_type"], "invoice");
    assert_eq!(list[0]["aggregate_id"], "invoice-1");
    assert_eq!(list[1]["aggregate_type"], "order");
    assert_eq!(list[1]["aggregate_id"], "order-2");
    assert_eq!(list[2]["aggregate_id"], "order-1");

    let updated_sorted = cli.run_json(&[
        "aggregate",
        "list",
        "--json",
        "--sort",
        "updated_at:desc,aggregate_id:asc",
    ])?;
    let updated_list = updated_sorted
        .as_array()
        .context("timestamp-sorted list did not return an array")?;
    assert_eq!(updated_list.len(), 3);
    assert_eq!(updated_list[0]["aggregate_id"], "order-1");
    assert_eq!(updated_list[1]["aggregate_id"], "invoice-1");
    assert_eq!(updated_list[2]["aggregate_id"], "order-2");

    let first_page = cli.run_json(&["aggregate", "list", "--json", "--take", "1"])?;
    let first_list = first_page
        .as_array()
        .context("first aggregate page did not return an array")?;
    assert_eq!(first_list.len(), 1);
    assert_eq!(
        first_list[0]["aggregate_id"], "invoice-1",
        "expected invoice-1 to appear first without an explicit sort"
    );
    let aggregate_type = first_list[0]["aggregate_type"]
        .as_str()
        .context("first page missing aggregate_type")?;
    let aggregate_id = first_list[0]["aggregate_id"]
        .as_str()
        .context("first page missing aggregate_id")?;
    let cursor = format!("a:{}:{}", aggregate_type, aggregate_id);
    let paged = cli.run_json(&[
        "aggregate",
        "list",
        "--json",
        "--take",
        "1",
        "--cursor",
        &cursor,
    ])?;
    let paged_list = paged
        .as_array()
        .context("cursor-based aggregate list did not return an array")?;
    assert_eq!(paged_list.len(), 1);
    assert_eq!(paged_list[0]["aggregate_id"], "order-1");

    Ok(())
}

#[test]
fn aggregate_list_timestamp_cursor() -> Result<()> {
    let cli = CliTest::new()?;

    cli.create_aggregate_with_fields(
        "patient",
        "patient-1",
        "patient_created",
        &[("status", "new")],
    )?;
    std::thread::sleep(std::time::Duration::from_millis(5));
    cli.create_aggregate_with_fields(
        "patient",
        "patient-2",
        "patient_created",
        &[("status", "pending")],
    )?;

    let page = cli.run_json(&[
        "aggregate",
        "list",
        "--json",
        "--take",
        "1",
        "--sort",
        "updated_at:desc",
    ])?;
    let list = page
        .as_array()
        .context("timestamp-sorted list did not return an array")?;
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["aggregate_id"], "patient-2");
    let updated_at = list[0]["updated_at"]
        .as_str()
        .context("sorted row missing updated_at")?;
    let parsed =
        DateTime::parse_from_rfc3339(updated_at).context("failed to parse updated_at timestamp")?;
    let cursor = format!(
        "ts:updated_at:desc:a:{}:{}:{}",
        parsed.timestamp_millis(),
        list[0]["aggregate_type"]
            .as_str()
            .context("sorted row missing aggregate_type")?,
        list[0]["aggregate_id"]
            .as_str()
            .context("sorted row missing aggregate_id")?
    );

    let next_page = cli.run_json(&[
        "aggregate",
        "list",
        "--json",
        "--take",
        "1",
        "--sort",
        "updated_at:desc",
        "--cursor",
        &cursor,
    ])?;
    let next = next_page
        .as_array()
        .context("timestamp cursor page did not return an array")?;
    assert_eq!(next.len(), 1);
    assert_eq!(next[0]["aggregate_id"], "patient-1");

    Ok(())
}

#[test]
fn aggregate_list_timestamp_cursor_shorthand() -> Result<()> {
    let cli = CliTest::new()?;

    cli.create_aggregate_with_fields(
        "account",
        "account-1",
        "account_created",
        &[("status", "open")],
    )?;
    std::thread::sleep(std::time::Duration::from_millis(5));
    cli.create_aggregate_with_fields(
        "account",
        "account-2",
        "account_created",
        &[("status", "approved")],
    )?;

    let page = cli.run_json(&[
        "aggregate",
        "list",
        "--json",
        "--take",
        "1",
        "--sort",
        "created_at:desc",
    ])?;
    let list = page
        .as_array()
        .context("timestamp-sorted list did not return an array")?;
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["aggregate_id"], "account-2");
    let cursor = format!(
        "ts:{}:{}",
        list[0]["aggregate_type"]
            .as_str()
            .context("sorted row missing aggregate_type")?,
        list[0]["aggregate_id"]
            .as_str()
            .context("sorted row missing aggregate_id")?
    );

    let next_page = cli.run_json(&[
        "aggregate",
        "list",
        "--json",
        "--take",
        "1",
        "--sort",
        "created_at:desc",
        "--cursor",
        &cursor,
    ])?;
    let next = next_page
        .as_array()
        .context("timestamp cursor page did not return an array")?;
    assert_eq!(next.len(), 1);
    assert_eq!(next[0]["aggregate_id"], "account-1");

    Ok(())
}

#[test]
fn aggregate_list_hides_archived_by_default() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&[
        "schema",
        "create",
        "archive_demo",
        "--events",
        "archive_created",
    ])?;

    cli.create_aggregate("archive_demo", "demo-1", "archive_created")?;
    cli.run(&[
        "aggregate",
        "apply",
        "archive_demo",
        "demo-1",
        "archive_created",
        "--field",
        "status=active",
    ])?;

    cli.run(&["aggregate", "archive", "archive_demo", "demo-1"])?;

    let active_only = cli.run_json(&["aggregate", "list", "--json"])?;
    let active_list = active_only
        .as_array()
        .context("active list did not return an array")?;
    assert!(
        active_list.is_empty(),
        "archived aggregate should be hidden by default"
    );

    let include_archived = cli.run_json(&["aggregate", "list", "--json", "--include-archived"])?;
    let include_list = include_archived
        .as_array()
        .context("include archived list did not return an array")?;
    assert_eq!(include_list.len(), 1);
    assert_eq!(include_list[0]["aggregate_type"], json!("archive_demo"));
    assert_eq!(include_list[0]["aggregate_id"], json!("demo-1"));
    assert_eq!(include_list[0]["archived"], json!(true));

    let archived_only = cli.run_json(&["aggregate", "list", "--json", "--archived-only"])?;
    let archived_list = archived_only
        .as_array()
        .context("archived-only list did not return an array")?;
    assert_eq!(archived_list.len(), 1);
    assert_eq!(archived_list[0]["aggregate_id"], json!("demo-1"));

    Ok(())
}

#[test]
fn aggregate_snapshot_creates_record() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "snap",
        "--events",
        "snap_created",
        "--json",
    ])?;
    cli.create_aggregate_with_fields("snap", "snap-1", "snap_created", &[("status", "ready")])?;
    let output = cli.run(&["snapshots", "create", "snap", "snap-1"])?;
    let snapshot: Value =
        serde_json::from_str(output.trim()).context("failed to parse snapshot output")?;
    assert_eq!(snapshot["aggregate_id"], json!("snap-1"));
    assert_eq!(snapshot["state"]["status"], json!("ready"));
    assert!(
        snapshot
            .get("snapshot_id")
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false),
        "snapshot_id should be present"
    );
    let list = cli.run(&["snapshots", "list"])?;
    assert!(
        list.contains("snapshot_id="),
        "list output should include snapshot_id:\n{}",
        list
    );
    let snapshot_id = snapshot["snapshot_id"]
        .as_str()
        .context("snapshot missing snapshot_id string")?;
    let fetched = cli.run_json(&["snapshots", "get", snapshot_id, "--json"])?;
    assert_eq!(fetched["snapshot_id"], json!(snapshot_id));
    assert_eq!(fetched["aggregate_id"], json!("snap-1"));
    assert_eq!(fetched["state"]["status"], json!("ready"));
    Ok(())
}

#[test]
fn aggregate_verify_success_json() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "verify",
        "--events",
        "verify_created",
        "--json",
    ])?;
    cli.create_aggregate_with_fields("verify", "verify-1", "verify_created", &[("status", "ok")])?;
    let output = cli.run(&["aggregate", "verify", "verify", "verify-1", "--json"])?;
    let body: Value =
        serde_json::from_str(output.trim()).context("failed to parse verify output")?;
    assert_eq!(body["aggregate_id"], json!("verify-1"));
    assert!(
        body["merkle_root"]
            .as_str()
            .map(|s| !s.is_empty())
            .unwrap_or(false)
    );
    Ok(())
}

#[test]
fn aggregate_archive_and_restore_flow() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "archive",
        "--events",
        "archive_created",
        "--json",
    ])?;
    cli.create_aggregate_with_fields(
        "archive",
        "archive-1",
        "archive_created",
        &[("status", "active")],
    )?;

    let archived = cli.run(&[
        "aggregate",
        "archive",
        "archive",
        "archive-1",
        "--comment",
        "archived",
    ])?;
    assert!(
        archived.contains("archived=true"),
        "unexpected archive output:\n{}",
        archived
    );

    let restored = cli.run(&[
        "aggregate",
        "restore",
        "archive",
        "archive-1",
        "--comment",
        "restored",
    ])?;
    assert!(
        restored.contains("archived=false"),
        "unexpected restore output:\n{}",
        restored
    );
    Ok(())
}

#[test]
fn aggregate_export_json_creates_file() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run_json(&[
        "schema",
        "create",
        "export",
        "--events",
        "export_created",
        "--json",
    ])?;
    cli.create_aggregate_with_fields(
        "export",
        "export-1",
        "export_created",
        &[("status", "ready")],
    )?;
    let output_path = cli.home.join("export.json");
    let stdout = cli.run(&[
        "aggregate",
        "export",
        "export",
        "--format",
        "json",
        "--output",
        output_path.to_string_lossy().as_ref(),
        "--pretty",
    ])?;
    assert!(
        stdout.contains("Exported"),
        "unexpected export output:\n{}",
        stdout
    );
    assert!(output_path.exists(), "export file not created");
    let contents = fs::read_to_string(&output_path)?;
    assert!(
        contents.contains("export-1"),
        "export file missing aggregate id: {}",
        contents
    );
    Ok(())
}

#[test]
fn aggregate_export_csv_sorts_headers_and_rows() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&[
        "schema",
        "create",
        "exportcsv",
        "--events",
        "export_created",
    ])?;
    cli.create_aggregate_with_fields(
        "exportcsv",
        "exp-2",
        "export_created",
        &[("status", "shipped"), ("total", "12")],
    )?;
    cli.create_aggregate_with_fields(
        "exportcsv",
        "exp-1",
        "export_created",
        &[("status", "ready"), ("total", "10")],
    )?;

    let output_path = cli.home.join("export.csv");
    let stdout = cli.run(&[
        "aggregate",
        "export",
        "exportcsv",
        "--format",
        "csv",
        "--output",
        output_path.to_string_lossy().as_ref(),
    ])?;
    assert!(
        stdout.contains("Exported 1 aggregate type(s) in CSV format"),
        "unexpected export output:\n{}",
        stdout
    );
    let contents = fs::read_to_string(&output_path)?;
    let mut reader = csv::Reader::from_reader(contents.as_bytes());
    let headers = reader.headers()?.clone();
    assert_eq!(
        headers,
        vec!["__aggregate_id", "status", "total"],
        "headers should include aggregate id plus sorted field names"
    );
    let rows = reader
        .records()
        .collect::<std::result::Result<Vec<_>, csv::Error>>()?;
    assert_eq!(
        rows,
        vec![
            StringRecord::from(vec!["exp-1", "ready", "10"]),
            StringRecord::from(vec!["exp-2", "shipped", "12"])
        ],
        "rows should be sorted by aggregate id"
    );

    Ok(())
}

#[test]
fn aggregate_export_all_errors_when_output_is_file() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "first", "--events", "first_created"])?;
    cli.run(&["schema", "create", "second", "--events", "second_created"])?;
    cli.create_aggregate("first", "first-1", "first_created")?;
    cli.create_aggregate("second", "second-1", "second_created")?;

    let output_path = cli.home.join("multi.json");
    fs::write(&output_path, "{}")?;
    let failure = cli.run_failure(&[
        "aggregate",
        "export",
        "--all",
        "--format",
        "json",
        "--output",
        output_path.to_string_lossy().as_ref(),
    ])?;
    assert!(
        failure
            .stderr
            .contains("output path must be a directory when exporting multiple aggregate types"),
        "unexpected export error message:\n{}\nstdout:\n{}",
        failure.stderr,
        failure.stdout
    );
    Ok(())
}

#[test]
fn aggregate_export_all_to_zip_includes_each_type() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "alpha", "--events", "alpha_created"])?;
    cli.run(&["schema", "create", "beta", "--events", "beta_created"])?;
    cli.create_aggregate_with_fields("alpha", "alpha-1", "alpha_created", &[("status", "ok")])?;
    cli.create_aggregate_with_fields("beta", "beta-1", "beta_created", &[("count", "1")])?;

    let output_dir = cli.home.join("exports");
    fs::create_dir_all(&output_dir)?;
    let stdout = cli.run(&[
        "aggregate",
        "export",
        "--all",
        "--format",
        "json",
        "--zip",
        "--output",
        output_dir.to_string_lossy().as_ref(),
    ])?;
    let zip_path = output_dir.join("aggregates_json.zip");
    assert!(
        zip_path.exists(),
        "expected zip archive at {}",
        zip_path.display()
    );
    assert!(
        stdout.contains(zip_path.to_string_lossy().as_ref()),
        "export output should mention archive path:\n{}",
        stdout
    );

    let file = fs::File::open(&zip_path)?;
    let mut archive = ZipArchive::new(file)?;

    let mut alpha_contents = String::new();
    {
        let mut alpha_file = archive
            .by_name("alpha.json")
            .context("missing alpha.json in archive")?;
        alpha_file.read_to_string(&mut alpha_contents)?;
    }
    let alpha_records: Vec<Value> = serde_json::from_str(&alpha_contents)?;
    assert_eq!(alpha_records.len(), 1);
    assert_eq!(alpha_records[0]["__aggregate_id"], json!("alpha-1"));
    assert_eq!(alpha_records[0]["status"], json!("ok"));

    let mut beta_contents = String::new();
    {
        let mut beta_file = archive
            .by_name("beta.json")
            .context("missing beta.json in archive")?;
        beta_file.read_to_string(&mut beta_contents)?;
    }
    let beta_records: Vec<Value> = serde_json::from_str(&beta_contents)?;
    assert_eq!(beta_records.len(), 1);
    assert_eq!(beta_records[0]["__aggregate_id"], json!("beta-1"));
    assert_eq!(beta_records[0]["count"], json!("1"));

    Ok(())
}

#[test]
fn aggregate_stage_and_commit_flow() -> Result<()> {
    let cli = CliTest::new()?;
    let schema = cli.run_json(&[
        "schema",
        "create",
        "order",
        "--events",
        "order_created,order_updated",
        "--json",
    ])?;
    assert_eq!(schema["aggregate"], "order");

    cli.create_aggregate("order", "order-1", "order_created")?;

    let staged_update = cli.run(&[
        "aggregate",
        "apply",
        "order",
        "order-1",
        "order_updated",
        "--field",
        "status=processing",
        "--stage",
    ])?;
    assert!(
        staged_update.contains("event staged for later commit"),
        "unexpected staging output for update:\n{}",
        staged_update
    );

    let commit_update = cli.run(&["aggregate", "commit"])?;
    assert!(
        commit_update.contains("committed 1 event(s)"),
        "unexpected commit output for update:\n{}",
        commit_update
    );

    let staged_list_after = cli.run(&["aggregate", "list", "--stage"])?;
    assert_eq!(
        staged_list_after.trim(),
        "no staged events",
        "expected staging queue to be empty after commit:\n{}",
        staged_list_after
    );

    let detail = cli.run_json(&["aggregate", "get", "order", "order-1", "--include-events"])?;
    assert_eq!(detail["version"], 2);
    assert_eq!(detail["state"]["status"], "processing");
    let events = detail["events"]
        .as_array()
        .context("aggregate get response missing events array")?;
    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["event_type"], "order_created");
    assert_eq!(events[1]["event_type"], "order_updated");
    assert_eq!(events[1]["payload"]["status"], "processing");

    Ok(())
}

#[test]
fn events_list_filters_by_payload_field() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "orders", "--events", "order_created"])?;
    cli.create_aggregate("orders", "order-1", "order_created")?;
    cli.run(&[
        "aggregate",
        "apply",
        "orders",
        "order-1",
        "order_created",
        "--field",
        "status=open",
    ])?;

    let events = cli.run_json(&["events", "--filter", r#"payload.status = "open""#, "--json"])?;
    let array = events.as_array().context("events output missing array")?;
    assert_eq!(array.len(), 1);
    assert_eq!(array[0]["event_type"], "order_created");

    let filtered = cli.run_json(&[
        "events",
        "--filter",
        r#"payload.status = "closed""#,
        "--json",
    ])?;
    let filtered_array = filtered
        .as_array()
        .context("filtered events output missing array")?;
    assert!(
        filtered_array.is_empty(),
        "expected filtered events to be empty"
    );

    Ok(())
}

#[test]
fn event_show_returns_single_event() -> Result<()> {
    let cli = CliTest::new()?;
    cli.run(&["schema", "create", "orders", "--events", "order_created"])?;
    cli.create_aggregate("orders", "order-1", "order_created")?;
    let record = cli.run_json(&[
        "aggregate",
        "apply",
        "orders",
        "order-1",
        "order_created",
        "--field",
        "status=open",
    ])?;
    let event_id = record["metadata"]["event_id"]
        .as_str()
        .context("aggregate apply output missing event_id")?;

    let detail = cli.run_json(&["events", event_id, "--json"])?;
    assert_eq!(detail["event_type"], "order_created");
    assert_eq!(detail["aggregate_id"], "order-1");
    assert_eq!(detail["payload"]["status"], "open");

    Ok(())
}

fn normalize_output(bytes: Vec<u8>) -> Result<String> {
    Ok(String::from_utf8(bytes)?.replace("\r\n", "\n"))
}
