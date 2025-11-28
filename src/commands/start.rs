use std::{
    env, fs,
    io::{self, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Result, anyhow};
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};

use eventdbx::{
    config::{Config, ConfigUpdate, load_or_default},
    restrict::{self, RESTRICT_ENV},
    server,
};

#[derive(Args, Clone)]
pub struct StartArgs {
    /// Override the configured server port
    #[arg(long)]
    pub port: Option<u16>,

    /// Override the configured data directory
    #[arg(long)]
    pub data_dir: Option<PathBuf>,

    /// Run the server in the foreground instead of daemonizing
    #[arg(long)]
    pub foreground: bool,

    /// Choose schema enforcement mode (`off`, `default`, or `strict`)
    #[arg(
        long,
        value_enum,
        default_value_t = RestrictModeArg::Default,
        num_args = 0..=1,
        default_missing_value = "default"
    )]
    pub restrict: RestrictModeArg,
}

impl Default for StartArgs {
    fn default() -> Self {
        Self {
            port: None,
            data_dir: None,
            foreground: false,
            restrict: RestrictModeArg::from(restrict::from_env()),
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
pub enum RestrictModeArg {
    #[value(
        alias = "false",
        alias = "0",
        alias = "no",
        alias = "none",
        alias = "unrestricted"
    )]
    Off,
    #[value(
        alias = "true",
        alias = "1",
        alias = "yes",
        alias = "on",
        alias = "prod",
        alias = "restricted",
        alias = "hybrid"
    )]
    Default,
    #[value(alias = "strict", alias = "all", alias = "enforced")]
    Strict,
}

impl From<RestrictModeArg> for restrict::RestrictMode {
    fn from(value: RestrictModeArg) -> Self {
        match value {
            RestrictModeArg::Off => restrict::RestrictMode::Off,
            RestrictModeArg::Default => restrict::RestrictMode::Default,
            RestrictModeArg::Strict => restrict::RestrictMode::Strict,
        }
    }
}

impl From<restrict::RestrictMode> for RestrictModeArg {
    fn from(value: restrict::RestrictMode) -> Self {
        match value {
            restrict::RestrictMode::Off => RestrictModeArg::Off,
            restrict::RestrictMode::Default => RestrictModeArg::Default,
            restrict::RestrictMode::Strict => RestrictModeArg::Strict,
        }
    }
}

#[derive(Args)]
pub struct DestroyArgs {
    /// Skip confirmation prompt
    #[arg(long)]
    pub yes: bool,
}

pub async fn execute(config_path: Option<PathBuf>, args: StartArgs) -> Result<()> {
    restrict::set_env(args.restrict.into());
    if args.foreground {
        start_foreground(config_path, args).await
    } else {
        start_daemon(config_path, args)?;
        Ok(())
    }
}

pub async fn run_internal(config_path: Option<PathBuf>) -> Result<()> {
    let args = StartArgs::default();
    restrict::set_env(args.restrict.into());
    start_foreground(config_path, args).await
}

pub fn stop(config_path: Option<PathBuf>) -> Result<()> {
    let (config, _) = load_or_default(config_path)?;
    let pid_path = config.pid_file_path();

    let Some(record) = read_pid_record(&pid_path)? else {
        println!("No running EventDBX server found.");
        return Ok(());
    };
    let pid = record.pid;

    if !process_is_running(pid) {
        remove_pid_file(&pid_path)?;
        println!("Removed stale EventDBX server pid file.");
        return Ok(());
    }

    terminate_process(pid)?;
    if !wait_for_exit(pid, Duration::from_secs(5)) {
        #[cfg(unix)]
        {
            force_kill_process(pid)?;
            if !wait_for_exit(pid, Duration::from_secs(2)) {
                return Err(anyhow!(
                    "failed to stop EventDBX server (pid {pid}); process is still running"
                ));
            }
        }
        #[cfg(not(unix))]
        {
            return Err(anyhow!(
                "failed to stop EventDBX server (pid {pid}); process is still running"
            ));
        }
    }

    remove_pid_file(&pid_path)?;
    if let Some(started_at) = record.started_at {
        println!(
            "EventDBX server stopped (pid {}) after {} (started {})",
            pid,
            describe_uptime(started_at),
            started_at.to_rfc3339()
        );
    } else {
        println!("EventDBX server stopped (pid {})", pid);
    }
    Ok(())
}

fn remove_pid_file(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(_) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn remove_pid_file_ignores_missing_path() {
        let dir = tempdir().unwrap();
        let pid_path = dir.path().join("eventdbx.pid");
        assert!(!pid_path.exists());
        remove_pid_file(pid_path.as_path()).expect("removing missing pid file should succeed");
    }

    #[test]
    fn remove_pid_file_deletes_existing_file() {
        let dir = tempdir().unwrap();
        let pid_path = dir.path().join("eventdbx.pid");
        fs::write(&pid_path, "1234").unwrap();
        assert!(pid_path.exists());
        remove_pid_file(pid_path.as_path()).expect("should remove pid file");
        assert!(!pid_path.exists());
    }
}

pub fn status(config_path: Option<PathBuf>) -> Result<()> {
    let (config, _) = load_or_default(config_path)?;
    let pid_path = config.pid_file_path();
    let domain = config.active_domain().to_string();

    match read_pid_record(&pid_path)? {
        Some(record) => {
            let pid = record.pid;
            if process_is_running(pid) {
                let socket_port =
                    resolve_socket_port(&config.socket.bind_addr).unwrap_or(config.port);
                if let Some(started_at) = record.started_at {
                    println!(
                        "EventDBX server is running on port {} (pid {}) — restrict={} — domain={} — up for {} (since {})",
                        socket_port,
                        pid,
                        config.restrict,
                        domain,
                        describe_uptime(started_at),
                        started_at.to_rfc3339()
                    );
                } else {
                    println!(
                        "EventDBX server is running on port {} (pid {}) — restrict={} — domain={}",
                        socket_port, pid, config.restrict, domain
                    );
                }
            } else {
                let _ = fs::remove_file(&pid_path);
                println!("EventDBX server is not running (removed stale pid file).");
                println!("Active domain: {}", domain);
            }
        }
        None => {
            println!("EventDBX server is not running.");
            println!("Active domain: {}", domain);
        }
    }

    Ok(())
}

pub fn destroy(config_path: Option<PathBuf>, args: DestroyArgs) -> Result<()> {
    let (config, path) = load_or_default(config_path)?;

    if !args.yes {
        eprint!(
            "This will permanently delete all EventDBX data under {} and remove the config file at {}.\nType \"destroy\" to continue: ",
            config.data_dir.display(),
            path.display()
        );
        io::stderr().flush()?;
        let mut confirmation = String::new();
        io::stdin().read_line(&mut confirmation)?;
        if confirmation.trim() != "destroy" {
            println!("Destroy command cancelled.");
            return Ok(());
        }
    }

    if let Err(err) = stop(Some(path.clone())) {
        tracing::warn!("failed to stop running server before destroy: {err}");
    }

    if config.data_dir.exists() {
        fs::remove_dir_all(&config.data_dir)?;
    }

    if path.exists() {
        fs::remove_file(&path)?;
    }

    println!(
        "All EventDBX data and configuration removed from {}",
        config.data_dir.display()
    );
    Ok(())
}

async fn start_foreground(config_path: Option<PathBuf>, args: StartArgs) -> Result<()> {
    let (config, path) = load_and_update_config(config_path, &args)?;
    let pid_path = config.pid_file_path();
    ensure_pid_slot(&pid_path)?;
    let _pid_guard = PidFileGuard::new(&pid_path)?;
    eprintln!(
        "configuration loaded; starting server (pid={})",
        std::process::id()
    );
    server::run(config, path).await?;
    Ok(())
}

fn start_daemon(config_path: Option<PathBuf>, args: StartArgs) -> Result<()> {
    let (config, path) = load_and_update_config(config_path, &args)?;
    let pid_path = config.pid_file_path();
    let restrict_mode: restrict::RestrictMode = args.restrict.into();

    ensure_pid_slot(&pid_path)?;

    let mut command = Command::new(env::current_exe()?);
    command.arg("--config").arg(&path);
    command.arg("__internal:server");
    command.stdin(Stdio::null());
    command.stdout(Stdio::null());
    command.stderr(Stdio::null());
    command.env(RESTRICT_ENV, restrict_mode.as_str());

    let mut child = command.spawn()?;
    let pid = child.id();

    let wait_deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if let Some(status) = child.try_wait()? {
            let message = if let Some(code) = status.code() {
                format!(
                    "EventDBX server failed to start (process exited with status {code}). \
                     Re-run with `eventdbx start --foreground` for details."
                )
            } else {
                "EventDBX server failed to start (process terminated unexpectedly). \
                 Re-run with `eventdbx start --foreground` for details."
                    .to_string()
            };
            return Err(anyhow!(message));
        }

        if Instant::now() >= wait_deadline {
            break;
        }

        thread::sleep(Duration::from_millis(100));
    }

    let started_at = chrono::Utc::now();
    let record = PidRecord {
        pid,
        started_at: Some(started_at),
    };
    write_pid_record(&pid_path, &record)?;

    drop(child);

    let socket_port = resolve_socket_port(&config.socket.bind_addr).unwrap_or(config.port);

    println!(
        "EventDBX server is running on port {} (pid {}) since {} (restrict={})",
        socket_port,
        pid,
        started_at.to_rfc3339(),
        restrict_mode
    );
    Ok(())
}

fn load_and_update_config(
    config_path: Option<PathBuf>,
    args: &StartArgs,
) -> Result<(Config, PathBuf)> {
    let (mut config, path) = load_or_default(config_path)?;
    apply_start_overrides(&mut config, args);
    config.ensure_data_dir()?;
    config.save(&path)?;
    Ok((config, path))
}

fn apply_start_overrides(config: &mut Config, args: &StartArgs) {
    config.apply_update(ConfigUpdate {
        port: args.port,
        data_dir: args.data_dir.clone(),
        cache_threshold: None,
        snapshot_threshold: None,
        data_encryption_key: None,
        restrict: Some(args.restrict.into()),
        list_page_size: None,
        page_limit: None,
        verbose_responses: None,
        reference_default_depth: None,
        reference_max_depth: None,
        plugin_max_attempts: None,
        socket: None,
        tenants: None,
        snowflake_worker_id: None,
        no_noise: None,
    });
}

#[derive(Debug, Serialize, Deserialize)]
struct PidRecord {
    pid: u32,
    #[serde(default)]
    started_at: Option<chrono::DateTime<chrono::Utc>>,
}

struct PidFileGuard {
    path: PathBuf,
}

impl PidFileGuard {
    fn new(path: &Path) -> Result<Self> {
        let record = PidRecord {
            pid: std::process::id(),
            started_at: Some(chrono::Utc::now()),
        };
        write_pid_record(path, &record)?;
        Ok(Self {
            path: path.to_path_buf(),
        })
    }
}

impl Drop for PidFileGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn write_pid_record(path: &Path, record: &PidRecord) -> Result<()> {
    let contents = serde_json::to_string(record)?;
    fs::write(path, contents)?;
    Ok(())
}

fn ensure_pid_slot(pid_path: &Path) -> Result<()> {
    if let Some(existing) = read_pid_record(pid_path)? {
        if process_is_running(existing.pid) {
            return Err(anyhow!(
                "EventDBX server already running (pid {})",
                existing.pid
            ));
        }
        fs::remove_file(pid_path)?;
    }

    Ok(())
}

fn read_pid_record(path: &Path) -> Result<Option<PidRecord>> {
    if !path.exists() {
        return Ok(None);
    }

    let contents = fs::read_to_string(path)?;
    let trimmed = contents.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    if let Ok(record) = serde_json::from_str::<PidRecord>(trimmed) {
        return Ok(Some(record));
    }

    if let Ok(pid) = trimmed.parse::<u32>() {
        return Ok(Some(PidRecord {
            pid,
            started_at: None,
        }));
    }

    Err(anyhow!("invalid pid file at {}", path.display()))
}

fn wait_for_exit(pid: u32, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if !process_is_running(pid) {
            return true;
        }
        if Instant::now() >= deadline {
            return !process_is_running(pid);
        }
        thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(unix)]
fn process_is_running(pid: u32) -> bool {
    unsafe {
        if libc::kill(pid as libc::pid_t, 0) == 0 {
            true
        } else {
            let err = io::Error::last_os_error();
            !matches!(err.raw_os_error(), Some(libc::ESRCH))
        }
    }
}

#[cfg(windows)]
fn process_is_running(pid: u32) -> bool {
    use windows_sys::Win32::{
        Foundation::CloseHandle,
        System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION},
    };

    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle == 0 {
            false
        } else {
            CloseHandle(handle);
            true
        }
    }
}

#[cfg(not(any(unix, windows)))]
fn process_is_running(_pid: u32) -> bool {
    false
}

#[cfg(unix)]
fn terminate_process(pid: u32) -> Result<()> {
    unsafe {
        if libc::kill(pid as libc::pid_t, libc::SIGTERM) == 0 {
            Ok(())
        } else {
            let err = io::Error::last_os_error();
            if matches!(err.raw_os_error(), Some(libc::ESRCH)) {
                Ok(())
            } else {
                Err(anyhow!("failed to send SIGTERM to pid {pid}: {err}"))
            }
        }
    }
}

#[cfg(windows)]
fn terminate_process(pid: u32) -> Result<()> {
    use windows_sys::Win32::{
        Foundation::CloseHandle,
        System::Threading::{OpenProcess, PROCESS_TERMINATE, TerminateProcess},
    };

    unsafe {
        let handle = OpenProcess(PROCESS_TERMINATE, 0, pid);
        if handle == 0 {
            return Err(anyhow!("failed to open process {pid} for termination"));
        }
        let result = TerminateProcess(handle, 0);
        CloseHandle(handle);
        if result == 0 {
            return Err(anyhow!("failed to terminate process {pid}"));
        }
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn terminate_process(pid: u32) -> Result<()> {
    Err(anyhow!(
        "process control is not supported on this platform (pid {pid})"
    ))
}

#[cfg(unix)]
fn force_kill_process(pid: u32) -> Result<()> {
    unsafe {
        if libc::kill(pid as libc::pid_t, libc::SIGKILL) == 0 {
            Ok(())
        } else {
            let err = io::Error::last_os_error();
            if matches!(err.raw_os_error(), Some(libc::ESRCH)) {
                Ok(())
            } else {
                Err(anyhow!("failed to send SIGKILL to pid {pid}: {err}"))
            }
        }
    }
}

fn resolve_socket_port(bind_addr: &str) -> Option<u16> {
    bind_addr
        .parse::<SocketAddr>()
        .map(|addr| addr.port())
        .ok()
        .or_else(|| {
            bind_addr
                .rsplit_once(':')
                .and_then(|(_, port)| port.trim().parse::<u16>().ok())
        })
}

fn describe_uptime(started_at: chrono::DateTime<chrono::Utc>) -> String {
    let now = chrono::Utc::now();
    let elapsed = now.signed_duration_since(started_at);
    match elapsed.to_std() {
        Ok(duration) => format_human_duration(duration),
        Err(_) => "unknown duration".to_string(),
    }
}

fn format_human_duration(duration: Duration) -> String {
    let mut secs = duration.as_secs();
    if secs == 0 {
        return "under 1s".to_string();
    }

    let days = secs / 86_400;
    secs %= 86_400;
    let hours = secs / 3_600;
    secs %= 3_600;
    let minutes = secs / 60;
    let seconds = secs % 60;

    let mut parts = Vec::new();
    if days > 0 {
        parts.push(format!("{}d", days));
    }
    if hours > 0 {
        parts.push(format!("{}h", hours));
    }
    if minutes > 0 {
        parts.push(format!("{}m", minutes));
    }
    if seconds > 0 && parts.len() < 3 {
        parts.push(format!("{}s", seconds));
    }

    if parts.is_empty() {
        "under 1s".to_string()
    } else {
        parts.join(" ")
    }
}
