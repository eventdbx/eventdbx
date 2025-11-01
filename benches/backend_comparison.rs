use std::{sync::Arc, time::Instant};

use anyhow::{Context, Result, anyhow};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use json_patch::Patch;
use once_cell::sync::Lazy;
use rand::{Rng, SeedableRng, distributions::Alphanumeric, rngs::StdRng};
use report::{Operation, Scenario};
use serde_json::{Value, json};

const AGGREGATE_TYPE: &str = "orders";
const EVENT_APPLY_TYPE: &str = "order_created";
const EVENT_PATCH_TYPE: &str = "order_patched";
const APPLY_PAYLOAD_SIZES: &[usize] = &[256, 1024, 4096];
const SEED_COUNT: usize = 1024;
const LIST_EVENT_COUNT: usize = 512;
const LIST_LIMIT: usize = 128;
const LIST_AGGREGATE_ID: &str = "list-benchmark";

static PATCH_TEMPLATE: Lazy<Value> = Lazy::new(|| {
    json!([
        { "op": "replace", "path": "/description", "value": "patched description" },
        { "op": "add", "path": "/status", "value": "patched" }
    ])
});

static PATCH_DOC: Lazy<Patch> =
    Lazy::new(|| serde_json::from_value(PATCH_TEMPLATE.clone()).expect("patch template is valid"));

mod report {
    use once_cell::sync::Lazy;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Mutex;
    use std::time::Duration;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub(super) enum Operation {
        Apply,
        Get,
        Patch,
        List,
    }

    impl Operation {
        pub(super) fn display(self) -> &'static str {
            match self {
                Operation::Apply => "Apply",
                Operation::Get => "Get",
                Operation::Patch => "Patch",
                Operation::List => "List",
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub(super) enum Scenario {
        PayloadBytes(usize),
        SeedCount(usize),
        ListLimit { limit: usize, events: usize },
    }

    impl Scenario {
        pub(super) fn label(self) -> String {
            match self {
                Scenario::PayloadBytes(bytes) => format!("{bytes} B payload"),
                Scenario::SeedCount(count) => format!("{count} seeded events"),
                Scenario::ListLimit { limit, events } => format!("limit {limit}, events {events}"),
            }
        }
    }

    #[derive(Default, Clone)]
    struct OperationData {
        rows: BTreeMap<Scenario, RowData>,
        columns: BTreeSet<String>,
    }

    #[derive(Default, Clone)]
    struct RowData {
        samples: BTreeMap<String, Vec<f64>>,
    }

    #[derive(Default, Clone)]
    struct Report {
        data: BTreeMap<Operation, OperationData>,
    }

    static REPORT: Lazy<Mutex<Report>> = Lazy::new(|| Mutex::new(Report::default()));

    pub(super) fn record(
        operation: Operation,
        scenario: Scenario,
        backend: &str,
        elapsed: Duration,
        iters: u64,
    ) {
        if iters == 0 {
            return;
        }

        let per_iter = elapsed.as_secs_f64() / iters as f64;
        let mut report = REPORT.lock().expect("report mutex poisoned");
        let op_data = report.data.entry(operation).or_default();
        op_data.columns.insert(backend.to_string());
        let row = op_data.rows.entry(scenario).or_default();
        row.samples
            .entry(backend.to_string())
            .or_default()
            .push(per_iter);
    }

    pub(super) fn print_summary() {
        let snapshot = {
            let report = REPORT.lock().expect("report mutex poisoned");
            report.data.clone()
        };

        if snapshot.is_empty() {
            return;
        }

        println!();
        println!("=== Backend Comparison Summary ===");
        for operation in [
            Operation::Apply,
            Operation::Get,
            Operation::Patch,
            Operation::List,
        ] {
            if let Some(data) = snapshot.get(&operation) {
                render(operation, data);
            }
        }
    }

    fn render(operation: Operation, data: &OperationData) {
        if data.columns.is_empty() || data.rows.is_empty() {
            return;
        }

        println!();
        println!("{} report (median duration per op)", operation.display());

        let mut headers = Vec::with_capacity(data.columns.len() + 1);
        headers.push("Scenario".to_string());
        headers.extend(data.columns.iter().cloned());

        let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
        let mut rows = Vec::with_capacity(data.rows.len());

        for (scenario, row_data) in &data.rows {
            let mut cells = Vec::with_capacity(headers.len());
            let label = scenario.label();
            widths[0] = widths[0].max(label.len());
            cells.push(label);

            for (idx, backend) in data.columns.iter().enumerate() {
                let cell = row_data
                    .samples
                    .get(backend)
                    .map(|samples| format_duration(median(samples)))
                    .unwrap_or_else(|| "n/a".to_string());
                widths[idx + 1] = widths[idx + 1].max(cell.len());
                cells.push(cell);
            }

            rows.push(cells);
        }

        let header_align = vec![false; headers.len()];
        let mut data_align = vec![true; headers.len()];
        if let Some(first) = data_align.first_mut() {
            *first = false;
        }

        print_separator(&widths);
        print_row(&headers, &widths, &header_align);
        print_separator(&widths);
        for row in rows {
            print_row(&row, &widths, &data_align);
        }
        print_separator(&widths);
    }

    fn print_separator(widths: &[usize]) {
        let mut line = String::from("+");
        for width in widths {
            line.push_str(&"-".repeat(width + 2));
            line.push('+');
        }
        println!("{}", line);
    }

    fn print_row(cells: &[String], widths: &[usize], align_right: &[bool]) {
        let mut line = String::from("|");
        for ((cell, width), right_align) in cells.iter().zip(widths).zip(align_right) {
            line.push(' ');
            if *right_align {
                line.push_str(&format!("{:>width$}", cell, width = width));
            } else {
                line.push_str(&format!("{:<width$}", cell, width = width));
            }
            line.push(' ');
            line.push('|');
        }
        println!("{}", line);
    }

    fn median(samples: &[f64]) -> f64 {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    fn format_duration(seconds: f64) -> String {
        if seconds >= 1.0 {
            format!("{:.3} s/op", seconds)
        } else if seconds >= 1e-3 {
            format!("{:.3} ms/op", seconds * 1e3)
        } else if seconds >= 1e-6 {
            format!("{:.3} us/op", seconds * 1e6)
        } else {
            format!("{:.3} ns/op", seconds * 1e9)
        }
    }
}

fn criterion_benches() -> Criterion {
    Criterion::default().warm_up_time(std::time::Duration::from_secs(3))
}

fn bench_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply");
    let payloads: Vec<Value> = APPLY_PAYLOAD_SIZES
        .iter()
        .map(|&size| build_payload(size))
        .collect();

    let mut eventdbx = EventdbxBackend::new().expect("eventdbx backend");
    for (idx, &size) in APPLY_PAYLOAD_SIZES.iter().enumerate() {
        let payload = &payloads[idx];
        group.bench_with_input(BenchmarkId::new("eventdbx", size), payload, |b, payload| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let key = eventdbx.apply_new(payload).expect("eventdbx apply");
                    black_box(key.event_id);
                }
                let elapsed = start.elapsed();
                report::record(
                    Operation::Apply,
                    Scenario::PayloadBytes(size),
                    "eventdbx",
                    elapsed,
                    iters,
                );
                elapsed
            });
        });
    }

    #[cfg(feature = "bench-postgres")]
    {
        if let Some(mut backend) = postgres::PostgresBackend::from_env().expect("postgres init") {
            for (idx, &size) in APPLY_PAYLOAD_SIZES.iter().enumerate() {
                let payload = &payloads[idx];
                group.bench_with_input(
                    BenchmarkId::new("postgres", size),
                    payload,
                    |b, payload| {
                        b.iter_custom(|iters| {
                            let start = Instant::now();
                            for _ in 0..iters {
                                let key = backend.apply_new(payload).expect("postgres apply");
                                black_box(key.id);
                            }
                            let elapsed = start.elapsed();
                            report::record(
                                Operation::Apply,
                                Scenario::PayloadBytes(size),
                                "postgres",
                                elapsed,
                                iters,
                            );
                            elapsed
                        });
                    },
                );
            }
        } else {
            eprintln!("Skipping postgres apply benchmark: set EVENTDBX_PG_DSN");
        }
    }

    #[cfg(feature = "bench-mongodb")]
    {
        if let Some(mut backend) = mongodb_backend::MongoBackend::from_env().expect("mongodb init")
        {
            for (idx, &size) in APPLY_PAYLOAD_SIZES.iter().enumerate() {
                let payload = &payloads[idx];
                group.bench_with_input(BenchmarkId::new("mongodb", size), payload, |b, payload| {
                    b.iter_custom(|iters| {
                        let start = Instant::now();
                        for _ in 0..iters {
                            let key = backend.apply_new(payload).expect("mongodb apply");
                            black_box(key.id.clone());
                        }
                        let elapsed = start.elapsed();
                        report::record(
                            Operation::Apply,
                            Scenario::PayloadBytes(size),
                            "mongodb",
                            elapsed,
                            iters,
                        );
                        elapsed
                    });
                });
            }
        } else {
            eprintln!("Skipping mongodb apply benchmark: set EVENTDBX_MONGO_URI");
        }
    }

    #[cfg(feature = "bench-mssql")]
    {
        if let Some(mut backend) = mssql::MssqlBackend::from_env().expect("mssql init") {
            for (idx, &size) in APPLY_PAYLOAD_SIZES.iter().enumerate() {
                let payload = &payloads[idx];
                group.bench_with_input(BenchmarkId::new("mssql", size), payload, |b, payload| {
                    b.iter_custom(|iters| {
                        let start = Instant::now();
                        for _ in 0..iters {
                            let key = backend.apply_new(payload).expect("mssql apply");
                            black_box(key.id);
                        }
                        let elapsed = start.elapsed();
                        report::record(
                            Operation::Apply,
                            Scenario::PayloadBytes(size),
                            "mssql",
                            elapsed,
                            iters,
                        );
                        elapsed
                    });
                });
            }
        } else {
            eprintln!("Skipping SQL Server apply benchmark: set EVENTDBX_MSSQL_DSN");
        }
    }

    group.finish();
}

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("get");
    let seed_payload = build_payload(512);

    let mut eventdbx = EventdbxBackend::new().expect("eventdbx backend");
    let eventdbx_keys = eventdbx
        .seed(&seed_payload, SEED_COUNT)
        .expect("seed eventdbx get dataset");
    let mut eventdbx_cycle = KeyCycle::new(eventdbx_keys);
    group.bench_function("eventdbx", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let key = eventdbx_cycle.next();
                let payload = eventdbx.get(&key).expect("eventdbx get");
                black_box(payload);
            }
            let elapsed = start.elapsed();
            report::record(
                Operation::Get,
                Scenario::SeedCount(SEED_COUNT),
                "eventdbx",
                elapsed,
                iters,
            );
            elapsed
        });
    });

    #[cfg(feature = "bench-postgres")]
    {
        if let Some(mut backend) = postgres::PostgresBackend::from_env().expect("postgres init") {
            let keys = backend
                .seed(&seed_payload, SEED_COUNT)
                .expect("seed postgres get dataset");
            let mut cycle = KeyCycle::new(keys);
            group.bench_function("postgres", |b| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let key = cycle.next();
                        let payload = backend.get(&key).expect("postgres get");
                        black_box(payload);
                    }
                    let elapsed = start.elapsed();
                    report::record(
                        Operation::Get,
                        Scenario::SeedCount(SEED_COUNT),
                        "postgres",
                        elapsed,
                        iters,
                    );
                    elapsed
                });
            });
        }
    }

    #[cfg(feature = "bench-mongodb")]
    {
        if let Some(mut backend) = mongodb_backend::MongoBackend::from_env().expect("mongodb init")
        {
            let keys = backend
                .seed(&seed_payload, SEED_COUNT)
                .expect("seed mongodb get dataset");
            let mut cycle = KeyCycle::new(keys);
            group.bench_function("mongodb", |b| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let key = cycle.next();
                        let payload = backend.get(&key).expect("mongodb get");
                        black_box(payload);
                    }
                    let elapsed = start.elapsed();
                    report::record(
                        Operation::Get,
                        Scenario::SeedCount(SEED_COUNT),
                        "mongodb",
                        elapsed,
                        iters,
                    );
                    elapsed
                });
            });
        }
    }

    #[cfg(feature = "bench-mssql")]
    {
        if let Some(mut backend) = mssql::MssqlBackend::from_env().expect("mssql init") {
            let keys = backend
                .seed(&seed_payload, SEED_COUNT)
                .expect("seed mssql get dataset");
            let mut cycle = KeyCycle::new(keys);
            group.bench_function("mssql", |b| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let key = cycle.next();
                        let payload = backend.get(&key).expect("mssql get");
                        black_box(payload);
                    }
                    let elapsed = start.elapsed();
                    report::record(
                        Operation::Get,
                        Scenario::SeedCount(SEED_COUNT),
                        "mssql",
                        elapsed,
                        iters,
                    );
                    elapsed
                });
            });
        }
    }

    group.finish();
}

fn bench_patch(c: &mut Criterion) {
    let mut group = c.benchmark_group("patch");
    let seed_payload = build_payload(512);

    let mut eventdbx = EventdbxBackend::new().expect("eventdbx backend");
    let eventdbx_keys = eventdbx
        .seed(&seed_payload, SEED_COUNT / 2)
        .expect("seed eventdbx patch dataset");
    let mut eventdbx_cycle = KeyCycle::new(eventdbx_keys);
    group.bench_function("eventdbx", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let key = eventdbx_cycle.next();
                eventdbx
                    .patch(&key, &PATCH_TEMPLATE, &PATCH_DOC)
                    .expect("eventdbx patch");
            }
            let elapsed = start.elapsed();
            report::record(
                Operation::Patch,
                Scenario::SeedCount(SEED_COUNT / 2),
                "eventdbx",
                elapsed,
                iters,
            );
            elapsed
        });
    });

    #[cfg(feature = "bench-postgres")]
    {
        if let Some(mut backend) = postgres::PostgresBackend::from_env().expect("postgres init") {
            let keys = backend
                .seed(&seed_payload, SEED_COUNT / 2)
                .expect("seed postgres patch dataset");
            let mut cycle = KeyCycle::new(keys);
            group.bench_function("postgres", |b| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let key = cycle.next();
                        backend
                            .patch(&key, &PATCH_TEMPLATE, &PATCH_DOC)
                            .expect("postgres patch");
                    }
                    let elapsed = start.elapsed();
                    report::record(
                        Operation::Patch,
                        Scenario::SeedCount(SEED_COUNT / 2),
                        "postgres",
                        elapsed,
                        iters,
                    );
                    elapsed
                });
            });
        }
    }

    #[cfg(feature = "bench-mongodb")]
    {
        if let Some(mut backend) = mongodb_backend::MongoBackend::from_env().expect("mongodb init")
        {
            let keys = backend
                .seed(&seed_payload, SEED_COUNT / 2)
                .expect("seed mongodb patch dataset");
            let mut cycle = KeyCycle::new(keys);
            group.bench_function("mongodb", |b| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let key = cycle.next();
                        backend
                            .patch(&key, &PATCH_TEMPLATE, &PATCH_DOC)
                            .expect("mongodb patch");
                    }
                    let elapsed = start.elapsed();
                    report::record(
                        Operation::Patch,
                        Scenario::SeedCount(SEED_COUNT / 2),
                        "mongodb",
                        elapsed,
                        iters,
                    );
                    elapsed
                });
            });
        }
    }

    #[cfg(feature = "bench-mssql")]
    {
        if let Some(mut backend) = mssql::MssqlBackend::from_env().expect("mssql init") {
            let keys = backend
                .seed(&seed_payload, SEED_COUNT / 2)
                .expect("seed mssql patch dataset");
            let mut cycle = KeyCycle::new(keys);
            group.bench_function("mssql", |b| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let key = cycle.next();
                        backend
                            .patch(&key, &PATCH_TEMPLATE, &PATCH_DOC)
                            .expect("mssql patch");
                    }
                    let elapsed = start.elapsed();
                    report::record(
                        Operation::Patch,
                        Scenario::SeedCount(SEED_COUNT / 2),
                        "mssql",
                        elapsed,
                        iters,
                    );
                    elapsed
                });
            });
        }
    }

    group.finish();
}

fn bench_list(c: &mut Criterion) {
    let mut group = c.benchmark_group("list");
    let seed_payload = build_payload(512);

    let mut eventdbx = EventdbxBackend::new().expect("eventdbx backend");
    eventdbx
        .seed_list_dataset(LIST_AGGREGATE_ID, &seed_payload, LIST_EVENT_COUNT)
        .expect("seed eventdbx list dataset");
    group.bench_function("eventdbx", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let events = eventdbx
                    .list(LIST_AGGREGATE_ID, LIST_LIMIT)
                    .expect("eventdbx list");
                black_box(events);
            }
            let elapsed = start.elapsed();
            report::record(
                Operation::List,
                Scenario::ListLimit {
                    limit: LIST_LIMIT,
                    events: LIST_EVENT_COUNT,
                },
                "eventdbx",
                elapsed,
                iters,
            );
            elapsed
        });
    });

    #[cfg(feature = "bench-postgres")]
    {
        if let Some(mut backend) = postgres::PostgresBackend::from_env().expect("postgres init") {
            backend
                .seed_list_dataset(LIST_AGGREGATE_ID, &seed_payload, LIST_EVENT_COUNT)
                .expect("seed postgres list dataset");
            group.bench_function("postgres", |b| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let events = backend
                            .list(LIST_AGGREGATE_ID, LIST_LIMIT)
                            .expect("postgres list");
                        black_box(events);
                    }
                    let elapsed = start.elapsed();
                    report::record(
                        Operation::List,
                        Scenario::ListLimit {
                            limit: LIST_LIMIT,
                            events: LIST_EVENT_COUNT,
                        },
                        "postgres",
                        elapsed,
                        iters,
                    );
                    elapsed
                });
            });
        }
    }

    #[cfg(feature = "bench-mongodb")]
    {
        if let Some(mut backend) = mongodb_backend::MongoBackend::from_env().expect("mongodb init")
        {
            backend
                .seed_list_dataset(LIST_AGGREGATE_ID, &seed_payload, LIST_EVENT_COUNT)
                .expect("seed mongodb list dataset");
            group.bench_function("mongodb", |b| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let events = backend
                            .list(LIST_AGGREGATE_ID, LIST_LIMIT)
                            .expect("mongodb list");
                        black_box(events);
                    }
                    let elapsed = start.elapsed();
                    report::record(
                        Operation::List,
                        Scenario::ListLimit {
                            limit: LIST_LIMIT,
                            events: LIST_EVENT_COUNT,
                        },
                        "mongodb",
                        elapsed,
                        iters,
                    );
                    elapsed
                });
            });
        }
    }

    #[cfg(feature = "bench-mssql")]
    {
        if let Some(mut backend) = mssql::MssqlBackend::from_env().expect("mssql init") {
            backend
                .seed_list_dataset(LIST_AGGREGATE_ID, &seed_payload, LIST_EVENT_COUNT)
                .expect("seed mssql list dataset");
            group.bench_function("mssql", |b| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let events = backend
                            .list(LIST_AGGREGATE_ID, LIST_LIMIT)
                            .expect("mssql list");
                        black_box(events);
                    }
                    let elapsed = start.elapsed();
                    report::record(
                        Operation::List,
                        Scenario::ListLimit {
                            limit: LIST_LIMIT,
                            events: LIST_EVENT_COUNT,
                        },
                        "mssql",
                        elapsed,
                        iters,
                    );
                    elapsed
                });
            });
        }
    }

    group.finish();
    report::print_summary();
}

criterion_group! {
    name = benches;
    config = criterion_benches();
    targets = bench_apply, bench_get, bench_patch, bench_list
}
criterion_main!(benches);

fn build_payload(size: usize) -> Value {
    let mut rng = StdRng::from_entropy();
    let text: String = (0..size)
        .map(|_| rng.sample(Alphanumeric) as char)
        .collect();
    json!({
        "amount": 42,
        "currency": "USD",
        "description": text,
    })
}

struct KeyCycle<K> {
    keys: Arc<Vec<K>>,
    index: usize,
}

impl<K: Clone> KeyCycle<K> {
    fn new(keys: Vec<K>) -> Self {
        assert!(!keys.is_empty(), "key cycle requires at least one key");
        Self {
            keys: Arc::new(keys),
            index: 0,
        }
    }

    fn next(&mut self) -> K {
        let key = self.keys[self.index].clone();
        self.index = (self.index + 1) % self.keys.len();
        key
    }
}

fn next_aggregate_id(counter: &mut u64) -> String {
    *counter += 1;
    format!("agg-{}", counter)
}

#[derive(Clone)]
struct EventdbxKey {
    aggregate_id: String,
    event_id: u64,
}

mod eventdbx_client {
    use super::*;
    use anyhow::{Context, Result, anyhow};
    use capnp::{
        message::{Builder, ReaderOptions},
        serialize,
        serialize::write_message_to_words,
    };
    use eventdbx::control_capnp::{
        control_hello, control_hello_response, control_request, control_response,
    };
    use eventdbx::replication_noise::{
        perform_client_handshake_blocking, read_encrypted_frame_blocking,
        write_encrypted_frame_blocking,
    };
    use eventdbx::store::EventRecord;
    use serde::Deserialize;
    use serde_json::Value;
    use std::{
        fs,
        io::{Cursor, ErrorKind, Write},
        net::TcpStream,
        path::{Path, PathBuf},
        thread,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    const CONFIG_WAIT_RETRIES: usize = 360;
    const CONFIG_WAIT_BACKOFF: Duration = Duration::from_millis(250);
    const CONNECT_RETRIES: usize = 60;
    const CONNECT_BACKOFF: Duration = Duration::from_millis(250);
    const TOKEN_WAIT_RETRIES: usize = 360;
    const TOKEN_WAIT_BACKOFF: Duration = Duration::from_millis(250);

    #[derive(Deserialize)]
    struct SocketSection {
        bind_addr: String,
    }

    #[derive(Deserialize)]
    struct ConfigToml {
        data_dir: PathBuf,
        socket: SocketSection,
    }

    impl ConfigToml {
        fn load(path: &Path) -> Result<Self> {
            let contents = fs::read_to_string(path)
                .with_context(|| format!("read EventDBX config at {}", path.display()))?;
            toml::from_str(&contents)
                .with_context(|| format!("parse EventDBX config at {}", path.display()))
        }

        fn base_dir(&self) -> PathBuf {
            self.data_dir.clone()
        }
    }

    pub struct EventdbxClient {
        connect_addr: String,
        token: String,
        fallback_token_path: Option<PathBuf>,
    }

    impl EventdbxClient {
        pub fn from_env() -> Result<Self> {
            let config_path = std::env::var("EVENTDBX_CONFIG_PATH")
                .unwrap_or_else(|_| "/eventdbx/config.toml".to_string());
            let config = load_config_with_retry(Path::new(&config_path))
                .with_context(|| format!("load EventDBX config from {}", config_path))?;

            let connect_addr = std::env::var("EVENTDBX_SOCKET_ADDR")
                .unwrap_or_else(|_| normalize_connect_addr(&config.socket.bind_addr));

            let cli_token_path = std::env::var("EVENTDBX_CLI_TOKEN_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|_| config.base_dir().join("cli.token"));

            let (token, fallback_path) = if let Ok(token) = std::env::var("EVENTDBX_TOKEN") {
                (token, Some(cli_token_path))
            } else if let Ok(custom_path) = std::env::var("EVENTDBX_TOKEN_PATH") {
                let path = PathBuf::from(custom_path);
                (read_token_once(&path)?, Some(path))
            } else {
                let path = cli_token_path;
                (read_token_with_retry(&path)?, Some(path))
            };

            Ok(Self {
                connect_addr,
                token: token.trim().to_string(),
                fallback_token_path: fallback_path,
            })
        }

        pub fn append_event(
            &mut self,
            aggregate_type: &str,
            aggregate_id: &str,
            event_type: &str,
            payload: &Value,
            metadata: Option<&Value>,
            note: Option<&str>,
        ) -> Result<EventRecord> {
            let payload_json =
                serde_json::to_string(payload).context("serialize payload for append request")?;
            let (metadata_json, has_metadata) = match metadata {
                Some(value) => (
                    serde_json::to_string(value)
                        .context("serialize metadata for append request")?,
                    true,
                ),
                None => (String::new(), false),
            };
            let (note_text, has_note) = match note {
                Some(value) => (value.to_string(), true),
                None => (String::new(), false),
            };

            let connect_addr = self.connect_addr.clone();
            self.retry_request(|request_id, token| {
                send_control_request_blocking(
                    &connect_addr,
                    token,
                    request_id,
                    |request| {
                        let payload_builder = request.reborrow().init_payload();
                        let mut append = payload_builder.init_append_event();
                        append.set_token(token);
                        append.set_aggregate_type(aggregate_type);
                        append.set_aggregate_id(aggregate_id);
                        append.set_event_type(event_type);
                        append.set_payload_json(&payload_json);
                        append.set_metadata_json(&metadata_json);
                        append.set_has_metadata(has_metadata);
                        append.set_note(&note_text);
                        append.set_has_note(has_note);
                        Ok(())
                    },
                    |response| {
                        use control_response::payload;

                        match response
                            .get_payload()
                            .which()
                            .context("decode append response payload")?
                        {
                            payload::AppendEvent(Ok(append)) => {
                                let event_json = read_text(append.get_event_json(), "event_json")?;
                                if event_json.trim().is_empty() {
                                    Err(anyhow!("append event response missing payload"))
                                } else {
                                    serde_json::from_str(&event_json)
                                        .context("parse append response event")
                                }
                            }
                            payload::AppendEvent(Err(err)) => Err(anyhow!(
                                "decode append_event payload from CLI proxy: {}",
                                err
                            )),
                            payload::Error(Ok(error)) => {
                                let code = read_text(error.get_code(), "code")?;
                                let message = read_text(error.get_message(), "message")?;
                                Err(anyhow!("server returned {}: {}", code, message))
                            }
                            payload::Error(Err(err)) => {
                                Err(anyhow!("decode error payload from CLI proxy: {}", err))
                            }
                            _ => Err(anyhow!(
                                "unexpected payload returned from CLI proxy response"
                            )),
                        }
                    },
                )
            })
        }

        pub fn patch_event(
            &mut self,
            aggregate_type: &str,
            aggregate_id: &str,
            event_type: &str,
            patch_ops: &Patch,
            metadata: Option<&Value>,
            note: Option<&str>,
        ) -> Result<EventRecord> {
            let patch_json =
                serde_json::to_string(patch_ops).context("serialize patch for patch request")?;
            let (metadata_json, has_metadata) = match metadata {
                Some(value) => (
                    serde_json::to_string(value).context("serialize metadata for patch request")?,
                    true,
                ),
                None => (String::new(), false),
            };
            let (note_text, has_note) = match note {
                Some(value) => (value.to_string(), true),
                None => (String::new(), false),
            };

            let connect_addr = self.connect_addr.clone();
            self.retry_request(|request_id, token| {
                send_control_request_blocking(
                    &connect_addr,
                    token,
                    request_id,
                    |request| {
                        let payload_builder = request.reborrow().init_payload();
                        let mut patch = payload_builder.init_patch_event();
                        patch.set_token(token);
                        patch.set_aggregate_type(aggregate_type);
                        patch.set_aggregate_id(aggregate_id);
                        patch.set_event_type(event_type);
                        patch.set_patch_json(&patch_json);
                        patch.set_metadata_json(&metadata_json);
                        patch.set_has_metadata(has_metadata);
                        patch.set_note(&note_text);
                        patch.set_has_note(has_note);
                        Ok(())
                    },
                    |response| {
                        use control_response::payload;

                        match response
                            .get_payload()
                            .which()
                            .context("decode patch response payload")?
                        {
                            payload::AppendEvent(Ok(append)) => {
                                let event_json = read_text(append.get_event_json(), "event_json")?;
                                if event_json.trim().is_empty() {
                                    Err(anyhow!("patch event response missing payload"))
                                } else {
                                    serde_json::from_str(&event_json)
                                        .context("parse patch response event")
                                }
                            }
                            payload::AppendEvent(Err(err)) => Err(anyhow!(
                                "decode patch_event payload from CLI proxy: {}",
                                err
                            )),
                            payload::Error(Ok(error)) => {
                                let code = read_text(error.get_code(), "code")?;
                                let message = read_text(error.get_message(), "message")?;
                                Err(anyhow!("server returned {}: {}", code, message))
                            }
                            payload::Error(Err(err)) => {
                                Err(anyhow!("decode error payload from CLI proxy: {}", err))
                            }
                            _ => Err(anyhow!(
                                "unexpected payload returned from CLI proxy response"
                            )),
                        }
                    },
                )
            })
        }

        pub fn list_event_payloads(
            &mut self,
            aggregate_type: &str,
            aggregate_id: &str,
            limit: usize,
        ) -> Result<Vec<Value>> {
            let records =
                self.list_events_for_aggregate(aggregate_type, aggregate_id, limit as u64, None)?;
            Ok(records.into_iter().map(|record| record.payload).collect())
        }

        pub fn get_event(
            &mut self,
            aggregate_type: &str,
            aggregate_id: &str,
            event_id: u64,
        ) -> Result<EventRecord> {
            let filter = format!("event_id = {}", event_id);
            let records =
                self.list_events_for_aggregate(aggregate_type, aggregate_id, 1, Some(&filter))?;
            records
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("event {} not found", event_id))
        }

        fn list_events_for_aggregate(
            &mut self,
            aggregate_type: &str,
            aggregate_id: &str,
            take: u64,
            filter: Option<&str>,
        ) -> Result<Vec<EventRecord>> {
            let connect_addr = self.connect_addr.clone();
            let filter_owned = filter.map(|s| s.to_string());
            self.retry_request(|request_id, token| {
                send_control_request_blocking(
                    &connect_addr,
                    token,
                    request_id,
                    |request| {
                        let payload_builder = request.reborrow().init_payload();
                        let mut list = payload_builder.init_list_events();
                        list.set_token(token);
                        list.set_aggregate_type(aggregate_type);
                        list.set_aggregate_id(aggregate_id);
                        list.set_skip(0);
                        list.set_take(take);
                        list.set_has_take(true);
                        if let Some(filter) = filter_owned.as_deref() {
                            list.set_filter(filter);
                            list.set_has_filter(true);
                        } else {
                            list.set_has_filter(false);
                        }
                        Ok(())
                    },
                    |response| {
                        use control_response::payload;

                        match response
                            .get_payload()
                            .which()
                            .context("decode list events payload")?
                        {
                            payload::ListEvents(Ok(events)) => {
                                let events_json =
                                    read_text(events.get_events_json(), "events_json")?;
                                if events_json.trim().is_empty() {
                                    Ok(Vec::new())
                                } else {
                                    serde_json::from_str(&events_json)
                                        .context("parse list events payload")
                                }
                            }
                            payload::ListEvents(Err(err)) => {
                                Err(anyhow!("decode listEvents payload from CLI proxy: {}", err))
                            }
                            payload::Error(Ok(error)) => {
                                let code = read_text(error.get_code(), "code")?;
                                let message = read_text(error.get_message(), "message")?;
                                Err(anyhow!("server returned {}: {}", code, message))
                            }
                            payload::Error(Err(err)) => {
                                Err(anyhow!("decode error payload from CLI proxy: {}", err))
                            }
                            _ => Err(anyhow!(
                                "unexpected payload returned from CLI proxy response"
                            )),
                        }
                    },
                )
            })
        }

        fn retry_request<F, T>(&mut self, mut op: F) -> Result<T>
        where
            F: FnMut(u64, &str) -> Result<T>,
        {
            let mut last_err = None;
            let mut refreshed_token = false;
            for attempt in 0..CONNECT_RETRIES {
                let request_id = next_request_id();
                let token_snapshot = self.token.clone();
                match op(request_id, &token_snapshot) {
                    Ok(result) => return Ok(result),
                    Err(err) => {
                        if !refreshed_token && self.try_refresh_token(&err)? {
                            refreshed_token = true;
                            continue;
                        }
                        if attempt + 1 == CONNECT_RETRIES {
                            return Err(err);
                        }
                        last_err = Some(err);
                        thread::sleep(CONNECT_BACKOFF);
                    }
                }
            }
            Err(last_err.unwrap_or_else(|| anyhow!("control request failed without error")))
        }

        fn try_refresh_token(&mut self, err: &anyhow::Error) -> Result<bool> {
            if let Some(path) = self.fallback_token_path.clone() {
                let message = err.to_string();
                if message.contains("control handshake rejected") {
                    if let Ok(token) = read_token_once(&path) {
                        self.token = token.trim().to_string();
                        return Ok(true);
                    }
                }
            }
            Ok(false)
        }
    }

    fn load_config_with_retry(path: &Path) -> Result<ConfigToml> {
        for attempt in 0..CONFIG_WAIT_RETRIES {
            match ConfigToml::load(path) {
                Ok(config) => return Ok(config),
                Err(err) => {
                    if let Some(io_err) = err.root_cause().downcast_ref::<std::io::Error>() {
                        if io_err.kind() == ErrorKind::NotFound {
                            if attempt + 1 == CONFIG_WAIT_RETRIES {
                                return Err(err);
                            }
                            thread::sleep(CONFIG_WAIT_BACKOFF);
                            continue;
                        }
                    }
                    return Err(err);
                }
            }
        }
        ConfigToml::load(path)
    }

    fn read_token_with_retry(path: &Path) -> Result<String> {
        for attempt in 0..TOKEN_WAIT_RETRIES {
            match read_token_once(path) {
                Ok(token) => return Ok(token),
                Err(err) => {
                    if let Some(io_err) = err.root_cause().downcast_ref::<std::io::Error>() {
                        if io_err.kind() == ErrorKind::NotFound && attempt + 1 != TOKEN_WAIT_RETRIES
                        {
                            thread::sleep(TOKEN_WAIT_BACKOFF);
                            continue;
                        }
                    }
                    return Err(err);
                }
            }
        }
        read_token_once(path)
    }

    fn read_token_once(path: &Path) -> Result<String> {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("read CLI token from {}", path.display()))?;
        extract_token(&contents).with_context(|| format!("parse CLI token from {}", path.display()))
    }

    fn extract_token(contents: &str) -> Result<String> {
        let trimmed = contents.trim();
        if trimmed.is_empty() {
            anyhow::bail!("token contents are empty");
        }

        if let Ok(value) = trimmed.parse::<toml::Value>() {
            if let Some(token) = value
                .get("token")
                .and_then(|entry| entry.as_str())
                .map(str::trim)
                .filter(|token| !token.is_empty())
            {
                return Ok(token.to_string());
            }
        }

        Ok(trimmed.to_string())
    }

    fn send_control_request_blocking<Build, Handle, T>(
        connect_addr: &str,
        handshake_token: &str,
        request_id: u64,
        build: Build,
        handle: Handle,
    ) -> Result<T>
    where
        Build: FnOnce(&mut control_request::Builder<'_>) -> Result<()>,
        Handle: FnOnce(control_response::Reader<'_>) -> Result<T>,
    {
        let mut stream = TcpStream::connect(connect_addr)
            .with_context(|| format!("connect to CLI proxy at {}", connect_addr))?;
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .context("configure proxy write timeout")?;
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .context("configure proxy read timeout")?;

        let mut hello_message = Builder::new_default();
        {
            let mut hello = hello_message.init_root::<control_hello::Builder>();
            hello.set_protocol_version(1);
            hello.set_token(handshake_token);
        }
        serialize::write_message(&mut stream, &hello_message).context("send control hello")?;
        stream.flush().context("flush control hello")?;

        let response_message = serialize::read_message(&mut stream, ReaderOptions::new())
            .context("read hello response")?;
        let response = response_message
            .get_root::<control_hello_response::Reader>()
            .context("decode hello response")?;
        if !response.get_accepted() {
            let reason = read_text(response.get_message(), "control handshake message")?;
            anyhow::bail!("control handshake rejected: {}", reason);
        }

        let mut noise = perform_client_handshake_blocking(&mut stream, handshake_token.as_bytes())
            .context("establish encrypted control channel")?;

        let mut message = Builder::new_default();
        {
            let mut request = message.init_root::<control_request::Builder>();
            request.set_id(request_id);
            build(&mut request)?;
        }
        let request_bytes = write_message_to_words(&message);
        write_encrypted_frame_blocking(&mut stream, &mut noise, &request_bytes)
            .context("send control request")?;

        let response_bytes = read_encrypted_frame_blocking(&mut stream, &mut noise)?
            .ok_or_else(|| anyhow!("CLI proxy closed control channel before response"))?;
        let mut cursor = Cursor::new(&response_bytes);
        let response_message = serialize::read_message(&mut cursor, ReaderOptions::new())
            .context("decode control response")?;
        let response = response_message
            .get_root::<control_response::Reader>()
            .context("decode control response")?;

        if response.get_id() != request_id {
            anyhow::bail!(
                "CLI proxy returned response id {} but expected {}",
                response.get_id(),
                request_id
            );
        }

        handle(response)
    }

    fn read_text(field: capnp::Result<capnp::text::Reader<'_>>, label: &str) -> Result<String> {
        let reader = field.with_context(|| format!("missing {label} in CLI proxy response"))?;
        reader
            .to_str()
            .map(|value| value.to_string())
            .map_err(|err| anyhow!("invalid utf-8 in {}: {}", label, err))
    }

    fn normalize_connect_addr(bind_addr: &str) -> String {
        use std::net::{IpAddr, SocketAddr};

        if let Ok(addr) = bind_addr.parse::<SocketAddr>() {
            match addr.ip() {
                IpAddr::V4(ip) if ip.is_unspecified() => format!("127.0.0.1:{}", addr.port()),
                IpAddr::V6(ip) if ip.is_unspecified() => format!("[::1]:{}", addr.port()),
                _ => addr.to_string(),
            }
        } else {
            bind_addr.to_string()
        }
    }

    fn next_request_id() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0)
    }
}

struct EventdbxBackend {
    client: eventdbx_client::EventdbxClient,
    write_counter: u64,
}

impl EventdbxBackend {
    fn new() -> Result<Self> {
        let client = eventdbx_client::EventdbxClient::from_env()?;
        Ok(Self {
            client,
            write_counter: 0,
        })
    }

    fn apply_new(&mut self, payload: &Value) -> Result<EventdbxKey> {
        let aggregate_id = next_aggregate_id(&mut self.write_counter);
        self.apply_with(&aggregate_id, EVENT_APPLY_TYPE, payload)
    }

    fn apply_with(
        &mut self,
        aggregate_id: &str,
        event_type: &str,
        payload: &Value,
    ) -> Result<EventdbxKey> {
        let record = self.client.append_event(
            AGGREGATE_TYPE,
            aggregate_id,
            event_type,
            payload,
            None,
            None,
        )?;
        Ok(EventdbxKey {
            aggregate_id: aggregate_id.to_string(),
            event_id: record.metadata.event_id.as_u64(),
        })
    }

    fn seed(&mut self, payload: &Value, count: usize) -> Result<Vec<EventdbxKey>> {
        let mut keys = Vec::with_capacity(count);
        for _ in 0..count {
            keys.push(self.apply_new(payload)?);
        }
        Ok(keys)
    }

    fn get(&mut self, key: &EventdbxKey) -> Result<Value> {
        let record = self
            .client
            .get_event(AGGREGATE_TYPE, &key.aggregate_id, key.event_id)?;
        Ok(record.payload)
    }

    fn patch(&mut self, key: &EventdbxKey, _patch_value: &Value, patch_ops: &Patch) -> Result<()> {
        self.client.patch_event(
            AGGREGATE_TYPE,
            &key.aggregate_id,
            EVENT_PATCH_TYPE,
            patch_ops,
            None,
            None,
        )?;
        Ok(())
    }

    fn list(&mut self, aggregate_id: &str, limit: usize) -> Result<Vec<Value>> {
        self.client
            .list_event_payloads(AGGREGATE_TYPE, aggregate_id, limit)
    }

    fn seed_list_dataset(
        &mut self,
        aggregate_id: &str,
        payload: &Value,
        count: usize,
    ) -> Result<()> {
        for _ in 0..count {
            let _ = self.apply_with(aggregate_id, EVENT_APPLY_TYPE, payload)?;
        }
        Ok(())
    }
}

#[cfg(feature = "bench-postgres")]
mod postgres {
    use super::*;
    use tokio::runtime::Runtime;
    use tokio_postgres::{Client, NoTls, types::Json};

    #[derive(Clone)]
    pub struct PostgresKey {
        pub id: i64,
    }

    pub struct PostgresBackend {
        runtime: Runtime,
        client: Client,
        table: String,
        insert_sql: String,
        select_sql: String,
        update_sql: String,
        list_sql: String,
        write_counter: u64,
    }

    impl PostgresBackend {
        pub fn from_env() -> Result<Option<Self>> {
            let url = match std::env::var("EVENTDBX_PG_DSN") {
                Ok(url) => url,
                Err(_) => return Ok(None),
            };
            let table = std::env::var("EVENTDBX_PG_TABLE")
                .unwrap_or_else(|_| "eventdbx_bench_events".into());

            let runtime = Runtime::new().context("create tokio runtime for postgres")?;
            let (client, connection) = runtime
                .block_on(tokio_postgres::connect(&url, NoTls))
                .context("connect to postgres")?;

            runtime.spawn(async move {
                if let Err(err) = connection.await {
                    eprintln!("postgres connection error: {err}");
                }
            });

            let insert_sql = format!(
                "INSERT INTO {table} (aggregate_type, aggregate_id, event_type, payload) \
                 VALUES ($1, $2, $3, $4::jsonb) RETURNING id",
            );
            let select_sql = format!("SELECT payload FROM {table} WHERE id = $1");
            let update_sql = format!("UPDATE {table} SET payload = $1::jsonb WHERE id = $2");
            let list_sql = format!(
                "SELECT payload FROM {table} WHERE aggregate_id = $1 \
                 ORDER BY id DESC LIMIT $2"
            );

            let mut backend = Self {
                runtime,
                client,
                table,
                insert_sql,
                select_sql,
                update_sql,
                list_sql,
                write_counter: 0,
            };
            backend.init_schema()?;

            Ok(Some(backend))
        }

        fn init_schema(&mut self) -> Result<()> {
            self.runtime.block_on(self.client.batch_execute(&format!(
                "CREATE TABLE IF NOT EXISTS {table} (
                    id BIGSERIAL PRIMARY KEY,
                    aggregate_type TEXT NOT NULL,
                    aggregate_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload JSONB NOT NULL
                );",
                table = self.table
            )))?;
            self.runtime.block_on(
                self.client
                    .batch_execute(&format!("TRUNCATE TABLE {};", self.table)),
            )?;
            Ok(())
        }

        pub fn apply_new(&mut self, payload: &Value) -> Result<PostgresKey> {
            let aggregate_id = super::next_aggregate_id(&mut self.write_counter);
            self.apply_with(&aggregate_id, EVENT_APPLY_TYPE, payload)
        }

        fn apply_with(
            &mut self,
            aggregate_id: &str,
            event_type: &str,
            payload: &Value,
        ) -> Result<PostgresKey> {
            let json_payload = Json(payload.clone());
            let row = self.runtime.block_on(self.client.query_one(
                &self.insert_sql,
                &[&AGGREGATE_TYPE, &aggregate_id, &event_type, &json_payload],
            ))?;
            Ok(PostgresKey { id: row.get(0) })
        }

        pub fn seed(&mut self, payload: &Value, count: usize) -> Result<Vec<PostgresKey>> {
            let mut keys = Vec::with_capacity(count);
            for _ in 0..count {
                keys.push(self.apply_new(payload)?);
            }
            Ok(keys)
        }

        pub fn get(&mut self, key: &PostgresKey) -> Result<Value> {
            let row = self
                .runtime
                .block_on(self.client.query_opt(&self.select_sql, &[&key.id]))?
                .ok_or_else(|| anyhow!("postgres row {} missing", key.id))?;
            let payload: Json<Value> = row.get(0);
            Ok(payload.0)
        }

        pub fn patch(
            &mut self,
            key: &PostgresKey,
            _patch_value: &Value,
            patch_ops: &Patch,
        ) -> Result<()> {
            let row = self
                .runtime
                .block_on(self.client.query_opt(&self.select_sql, &[&key.id]))?
                .ok_or_else(|| anyhow!("postgres row {} missing", key.id))?;
            let payload: Json<Value> = row.get(0);
            let mut document = payload.0;
            json_patch::patch(&mut document, patch_ops)
                .context("apply patch to postgres payload")?;
            let updated = Json(document);
            self.runtime
                .block_on(self.client.execute(&self.update_sql, &[&updated, &key.id]))?;
            Ok(())
        }

        pub fn list(&mut self, aggregate_id: &str, limit: usize) -> Result<Vec<Value>> {
            let rows = self.runtime.block_on(
                self.client
                    .query(&self.list_sql, &[&aggregate_id, &(limit as i64)]),
            )?;
            let mut values = Vec::with_capacity(rows.len());
            for row in rows {
                let payload: Json<Value> = row.get(0);
                values.push(payload.0);
            }
            Ok(values)
        }

        pub fn seed_list_dataset(
            &mut self,
            aggregate_id: &str,
            payload: &Value,
            count: usize,
        ) -> Result<()> {
            for _ in 0..count {
                let _ = self.apply_with(aggregate_id, EVENT_APPLY_TYPE, payload)?;
            }
            Ok(())
        }
    }
}

#[cfg(feature = "bench-mongodb")]
mod mongodb_backend {
    use super::*;
    use futures_util::TryStreamExt;
    use mongodb::{
        Client, Collection,
        bson::{Bson, Document, doc, to_bson},
        options::FindOptions,
    };
    use tokio::runtime::Runtime;

    #[derive(Clone)]
    pub struct MongoKey {
        pub id: Bson,
    }

    pub struct MongoBackend {
        runtime: Runtime,
        collection: Collection<Document>,
        write_counter: u64,
    }

    impl MongoBackend {
        pub fn from_env() -> Result<Option<Self>> {
            let uri = match std::env::var("EVENTDBX_MONGO_URI") {
                Ok(uri) => uri,
                Err(_) => return Ok(None),
            };
            let database =
                std::env::var("EVENTDBX_MONGO_DB").unwrap_or_else(|_| "eventdbx_bench".into());
            let collection =
                std::env::var("EVENTDBX_MONGO_COLLECTION").unwrap_or_else(|_| "events".into());

            let runtime = Runtime::new().context("create tokio runtime for mongodb")?;
            let client = runtime
                .block_on(Client::with_uri_str(uri))
                .context("connect to mongodb")?;
            let collection = client
                .database(&database)
                .collection::<Document>(&collection);

            runtime
                .block_on(collection.delete_many(doc! {}, None))
                .context("clean mongodb collection")?;

            Ok(Some(Self {
                runtime,
                collection,
                write_counter: 0,
            }))
        }

        pub fn apply_new(&mut self, payload: &Value) -> Result<MongoKey> {
            let aggregate_id = super::next_aggregate_id(&mut self.write_counter);
            self.apply_with(&aggregate_id, EVENT_APPLY_TYPE, payload)
        }

        fn apply_with(
            &mut self,
            aggregate_id: &str,
            event_type: &str,
            payload: &Value,
        ) -> Result<MongoKey> {
            let payload_bson = to_bson(payload).context("payload to bson")?;
            let document = doc! {
                "aggregate_type": AGGREGATE_TYPE,
                "aggregate_id": aggregate_id,
                "event_type": event_type,
                "payload": payload_bson,
            };
            let result = self
                .runtime
                .block_on(self.collection.insert_one(document, None))?;
            let id = result.inserted_id;
            Ok(MongoKey { id })
        }

        pub fn seed(&mut self, payload: &Value, count: usize) -> Result<Vec<MongoKey>> {
            let mut keys = Vec::with_capacity(count);
            for _ in 0..count {
                keys.push(self.apply_new(payload)?);
            }
            Ok(keys)
        }

        pub fn get(&mut self, key: &MongoKey) -> Result<Value> {
            let filter = doc! { "_id": key.id.clone() };
            let document = self
                .runtime
                .block_on(self.collection.find_one(filter, None))?
                .ok_or_else(|| anyhow!("mongodb document not found"))?;
            let payload = document
                .get("payload")
                .ok_or_else(|| anyhow!("mongodb payload missing"))?;
            mongodb::bson::from_bson::<Value>(payload.clone()).context("decode mongodb payload")
        }

        pub fn patch(
            &mut self,
            key: &MongoKey,
            _patch_value: &Value,
            patch_ops: &Patch,
        ) -> Result<()> {
            let filter = doc! { "_id": key.id.clone() };
            let document = self
                .runtime
                .block_on(self.collection.find_one(filter.clone(), None))?
                .ok_or_else(|| anyhow!("mongodb document not found"))?;
            let payload = document
                .get("payload")
                .ok_or_else(|| anyhow!("mongodb payload missing"))?;
            let mut value = mongodb::bson::from_bson::<Value>(payload.clone())
                .context("decode mongodb payload")?;
            json_patch::patch(&mut value, patch_ops).context("apply patch to mongodb payload")?;
            let updated = to_bson(&value).context("encode patched payload to bson")?;
            self.runtime.block_on(self.collection.update_one(
                filter,
                doc! { "$set": { "payload": updated } },
                None,
            ))?;
            Ok(())
        }

        pub fn list(&mut self, aggregate_id: &str, limit: usize) -> Result<Vec<Value>> {
            let options = FindOptions::builder()
                .limit(limit as i64)
                .sort(doc! { "_id": -1 })
                .build();
            let mut cursor = self.runtime.block_on(
                self.collection
                    .find(doc! { "aggregate_id": aggregate_id }, options),
            )?;
            let mut values = Vec::new();
            while let Some(doc) = self
                .runtime
                .block_on(cursor.try_next())
                .context("iterate mongodb cursor")?
            {
                let payload = doc
                    .get("payload")
                    .ok_or_else(|| anyhow!("mongodb payload missing"))?;
                values.push(
                    mongodb::bson::from_bson::<Value>(payload.clone())
                        .context("decode mongodb payload")?,
                );
            }
            Ok(values)
        }

        pub fn seed_list_dataset(
            &mut self,
            aggregate_id: &str,
            payload: &Value,
            count: usize,
        ) -> Result<()> {
            for _ in 0..count {
                let _ = self.apply_with(aggregate_id, EVENT_APPLY_TYPE, payload)?;
            }
            Ok(())
        }
    }
}

#[cfg(feature = "bench-mssql")]
mod mssql {
    use super::*;
    use futures_util::TryStreamExt;
    use tiberius::{Client, Config, Query};
    use tokio::{net::TcpStream, runtime::Runtime, sync::Mutex};
    use tokio_util::compat::TokioAsyncWriteCompatExt;

    #[derive(Clone)]
    pub struct MssqlKey {
        pub id: i64,
        pub aggregate_id: String,
    }

    pub struct MssqlBackend {
        runtime: Runtime,
        client: Arc<Mutex<Client<tokio_util::compat::Compat<TcpStream>>>>,
        table: String,
        write_counter: u64,
    }

    impl MssqlBackend {
        pub fn from_env() -> Result<Option<Self>> {
            let dsn = match std::env::var("EVENTDBX_MSSQL_DSN") {
                Ok(value) => value,
                Err(_) => return Ok(None),
            };
            let table = std::env::var("EVENTDBX_MSSQL_TABLE")
                .unwrap_or_else(|_| "eventdbx_bench_events".into());

            let runtime = Runtime::new().context("create tokio runtime for mssql")?;
            let client = runtime
                .block_on(Self::connect(&dsn))
                .context("connect to mssql")?;
            let client = Arc::new(Mutex::new(client));

            let mut backend = Self {
                runtime,
                client,
                table,
                write_counter: 0,
            };
            backend.init_schema()?;

            Ok(Some(backend))
        }

        async fn connect(dsn: &str) -> Result<Client<tokio_util::compat::Compat<TcpStream>>> {
            let mut config = Config::from_ado_string(dsn).context("parse mssql dsn")?;
            config.trust_cert();
            let addr = config.get_addr();
            let tcp = TcpStream::connect(addr).await.context("connect tcp")?;
            tcp.set_nodelay(true).context("set nodelay")?;
            Client::connect(config, tcp.compat_write())
                .await
                .context("connect client")
        }

        fn init_schema(&mut self) -> Result<()> {
            let table = self.table.clone();
            self.runtime.block_on(async {
                let mut client = self.client.lock().await;
                client
                    .simple_query(&format!(
                        "IF OBJECT_ID('{table}', 'U') IS NULL BEGIN
                            CREATE TABLE {table} (
                                id BIGINT IDENTITY(1,1) PRIMARY KEY,
                                aggregate_type NVARCHAR(255) NOT NULL,
                                aggregate_id NVARCHAR(255) NOT NULL,
                                event_type NVARCHAR(255) NOT NULL,
                                payload NVARCHAR(MAX) NOT NULL
                            );
                        END;",
                        table = table
                    ))
                    .await
                    .context("create table")?;
                client
                    .simple_query(&format!("TRUNCATE TABLE {table};", table = table))
                    .await
                    .context("truncate table")?;
                Ok::<_, anyhow::Error>(())
            })?;
            Ok(())
        }

        pub fn apply_new(&mut self, payload: &Value) -> Result<MssqlKey> {
            let aggregate_id = super::next_aggregate_id(&mut self.write_counter);
            self.apply_with(&aggregate_id, EVENT_APPLY_TYPE, payload)
        }

        fn apply_with(
            &mut self,
            aggregate_id: &str,
            event_type: &str,
            payload: &Value,
        ) -> Result<MssqlKey> {
            let payload_string = payload.to_string();
            let table = self.table.clone();
            self.runtime.block_on(async {
                let mut client = self.client.lock().await;
                let mut stream = client
                    .query(
                        &format!(
                            "INSERT INTO {table} (aggregate_type, aggregate_id, event_type, payload)
                             OUTPUT INSERTED.id
                             VALUES (@P1, @P2, @P3, @P4);",
                            table = table
                        ),
                        &[&AGGREGATE_TYPE, &aggregate_id, &event_type, &payload_string],
                    )
                    .await
                    .context("mssql insert")?;

                let row = stream.try_next().await.context("fetch inserted id")?;
                let row = row.ok_or_else(|| anyhow!("mssql missing output row"))?;
                let id: i64 = row.try_get(0).context("read inserted id")?;
                Ok(MssqlKey {
                    id,
                    aggregate_id: aggregate_id.to_string(),
                })
            })
        }

        pub fn seed(&mut self, payload: &Value, count: usize) -> Result<Vec<MssqlKey>> {
            let mut keys = Vec::with_capacity(count);
            for _ in 0..count {
                keys.push(self.apply_new(payload)?);
            }
            Ok(keys)
        }

        pub fn get(&mut self, key: &MssqlKey) -> Result<Value> {
            let table = self.table.clone();
            self.runtime.block_on(async {
                let mut client = self.client.lock().await;
                let mut stream = client
                    .query(
                        &format!("SELECT payload FROM {table} WHERE id = @P1;", table = table),
                        &[&key.id],
                    )
                    .await
                    .context("mssql select")?;

                let row = stream.try_next().await.context("fetch row")?;
                let row = row.ok_or_else(|| anyhow!("mssql row {} missing", key.id))?;
                let payload: &str = row.try_get(0).context("get payload")?;
                serde_json::from_str(payload).context("decode mssql payload")
            })
        }

        pub fn patch(
            &mut self,
            key: &MssqlKey,
            _patch_value: &Value,
            patch_ops: &Patch,
        ) -> Result<()> {
            let table = self.table.clone();
            self.runtime.block_on(async {
                let mut client = self.client.lock().await;
                let mut stream = client
                    .query(
                        &format!("SELECT payload FROM {table} WHERE id = @P1;", table = table),
                        &[&key.id],
                    )
                    .await
                    .context("mssql select for patch")?;
                let row = stream.try_next().await.context("fetch row")?;
                let row = row.ok_or_else(|| anyhow!("mssql row {} missing", key.id))?;
                let payload: &str = row.try_get(0).context("get payload")?;
                let mut value: Value =
                    serde_json::from_str(payload).context("decode mssql payload")?;
                json_patch::patch(&mut value, patch_ops).context("apply patch to mssql payload")?;
                let updated = value.to_string();
                client
                    .execute(
                        &format!(
                            "UPDATE {table} SET payload = @P1 WHERE id = @P2;",
                            table = table
                        ),
                        &[&updated, &key.id],
                    )
                    .await
                    .context("update mssql payload")?;
                Ok(())
            })
        }

        pub fn list(&mut self, aggregate_id: &str, limit: usize) -> Result<Vec<Value>> {
            let table = self.table.clone();
            self.runtime.block_on(async {
                let mut client = self.client.lock().await;
                let mut stream = client
                    .query(
                        &format!(
                            "SELECT payload FROM {table} WHERE aggregate_id = @P1 \
                             ORDER BY id DESC OFFSET 0 ROWS FETCH NEXT @P2 ROWS ONLY;",
                            table = table
                        ),
                        &[&aggregate_id, &(limit as i32)],
                    )
                    .await
                    .context("mssql list query")?;

                let mut values = Vec::new();
                while let Some(row) = stream.try_next().await.context("fetch row")? {
                    let payload: &str = row.try_get(0).context("get payload")?;
                    values.push(serde_json::from_str(payload).context("decode mssql payload")?);
                }
                Ok(values)
            })
        }

        pub fn seed_list_dataset(
            &mut self,
            aggregate_id: &str,
            payload: &Value,
            count: usize,
        ) -> Result<()> {
            for _ in 0..count {
                let _ = self.apply_with(aggregate_id, EVENT_APPLY_TYPE, payload)?;
            }
            Ok(())
        }
    }
}
