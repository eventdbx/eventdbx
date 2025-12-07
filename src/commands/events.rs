use std::{path::PathBuf, str::FromStr};

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
use eventdbx::{
    config::load_or_default,
    filter,
    snowflake::SnowflakeId,
    store::{
        EventArchiveScope, EventCursor, EventQueryScope, EventSort, EventSortField, EventStore,
    },
};
use serde_json::{self, Value};

#[derive(Args)]
pub struct EventsArgs {
    /// Optional aggregate type to scope results. Provide an event id instead to inspect a single event.
    #[arg(value_name = "TARGET")]
    pub aggregate: Option<String>,

    /// Optional aggregate identifier to scope results
    #[arg(value_name = "AGGREGATE_ID")]
    pub aggregate_id: Option<String>,

    /// Inspect a single event by Snowflake identifier
    #[arg(long = "event", short = 'e', value_name = "EVENT_ID")]
    pub event_id: Option<String>,

    /// Resume listing after the provided cursor (`a:type:id:version`)
    #[arg(long)]
    pub cursor: Option<String>,

    /// Maximum number of events to return
    #[arg(long)]
    pub take: Option<usize>,

    /// Filter events using a SQL-like expression (e.g. `event_type = "order_created"`)
    #[arg(long)]
    pub filter: Option<String>,

    /// Sort events by comma-separated fields (e.g. `created_at:desc,aggregate_id`)
    #[arg(long, value_name = "FIELD[:ORDER][,...]")]
    pub sort: Option<String>,

    /// Emit results as JSON
    #[arg(long)]
    pub json: bool,

    /// Include archived events alongside active ones
    #[arg(long, conflicts_with = "archived_only")]
    pub include_archived: bool,

    /// Show only archived events
    #[arg(long)]
    pub archived_only: bool,
}

pub fn list(config_path: Option<PathBuf>, args: EventsArgs) -> Result<()> {
    if let Some(event_id) = selected_event_id(&args)? {
        return show_event(config_path, &event_id, args.json);
    }

    let (config, _) = load_or_default(config_path)?;
    let store = EventStore::open_read_only(config.event_store_path(), config.encryption_key()?)?;

    let filter_expr = if let Some(raw) = args.filter.as_ref() {
        Some(
            filter::parse_shorthand(raw)
                .with_context(|| format!("invalid filter expression: {raw}"))?,
        )
    } else {
        None
    };

    let sort_directives = if let Some(spec) = args.sort.as_deref() {
        Some(
            parse_event_sort_directives(spec)
                .map_err(|err| anyhow!("invalid sort specification: {err}"))?,
        )
    } else {
        None
    };

    let archive_scope = if args.archived_only {
        EventArchiveScope::ArchivedOnly
    } else if args.include_archived {
        EventArchiveScope::IncludeArchived
    } else {
        EventArchiveScope::ActiveOnly
    };

    let take = args.take.unwrap_or(config.list_page_size);
    if take == 0 {
        bail!("--take must be greater than zero");
    }

    let scope = match (&args.aggregate, &args.aggregate_id) {
        (Some(aggregate), Some(aggregate_id)) => EventQueryScope::Aggregate {
            aggregate_type: aggregate.as_str(),
            aggregate_id: aggregate_id.as_str(),
        },
        (Some(aggregate), None) => EventQueryScope::AggregateType(aggregate.as_str()),
        (None, _) => EventQueryScope::All,
    };

    if args.cursor.is_some() && sort_directives.is_some() {
        bail!("--cursor cannot be combined with --sort");
    }

    let events = if let Some(directives) = sort_directives.as_ref() {
        store.events_paginated(
            scope,
            archive_scope,
            0,
            Some(take),
            Some(directives.as_slice()),
            filter_expr.as_ref(),
        )?
    } else {
        let cursor = match args.cursor.as_deref() {
            Some(raw) => Some(EventCursor::from_str(raw).with_context(|| {
                format!(
                    "invalid cursor '{}'; expected format a:aggregate_type:aggregate_id:version",
                    raw
                )
            })?),
            None => None,
        };
        let (events, _next_cursor) = store.events_page(
            scope,
            archive_scope,
            cursor.as_ref(),
            take,
            filter_expr.as_ref(),
        )?;
        events
    };

    if args.json {
        let rendered: Vec<Value> = events
            .iter()
            .map(|event| event.to_output_value())
            .collect::<serde_json::Result<_>>()?;
        println!("{}", serde_json::to_string_pretty(&rendered)?);
        return Ok(());
    }

    if events.is_empty() {
        println!("no events");
        return Ok(());
    }

    for record in events {
        let issued = record
            .metadata
            .issued_by
            .as_ref()
            .map(|claims| format!("{}:{}", claims.group, claims.user))
            .unwrap_or_else(|| "-".to_string());
        let note = record.metadata.note.as_deref().unwrap_or("-");
        println!(
            "aggregate_type={} aggregate_id={} version={} event_type={} event_id={} created_at={} issued_by={} note={}",
            record.aggregate_type,
            record.aggregate_id,
            record.version,
            record.event_type,
            record.metadata.event_id,
            record.metadata.created_at.to_rfc3339(),
            issued,
            note,
        );
    }

    Ok(())
}

fn show_event(config_path: Option<PathBuf>, event_id_raw: &str, json: bool) -> Result<()> {
    let (config, _) = load_or_default(config_path)?;
    let store = EventStore::open_read_only(config.event_store_path(), config.encryption_key()?)?;

    let event_id = event_id_raw
        .parse::<SnowflakeId>()
        .with_context(|| format!("invalid snowflake id '{}'", event_id_raw))?;

    let Some(event) = store.find_event_by_id(event_id)? else {
        bail!("event {} not found", event_id_raw);
    };

    if json {
        let rendered = event.to_output_value()?;
        println!("{}", serde_json::to_string_pretty(&rendered)?);
        return Ok(());
    }

    println!("aggregate_type={}", event.aggregate_type);
    println!("aggregate_id={}", event.aggregate_id);
    println!("version={}", event.version);
    println!("event_type={}", event.event_type);
    println!("event_id={}", event.metadata.event_id);
    println!("created_at={}", event.metadata.created_at.to_rfc3339());
    if let Some(claims) = event.metadata.issued_by.as_ref() {
        println!("issued_by={}:{}", claims.group, claims.user);
    }
    if let Some(note) = event.metadata.note.as_ref() {
        println!("note={}", note);
    }
    println!("hash={}", event.hash);
    println!("merkle_root={}", event.merkle_root);

    println!("payload={}", serde_json::to_string_pretty(&event.payload)?);

    if let Some(extensions) = event.extensions.as_ref() {
        println!("extensions={}", serde_json::to_string_pretty(extensions)?);
    }

    Ok(())
}

fn selected_event_id(args: &EventsArgs) -> Result<Option<String>> {
    let listing_options_used = args.cursor.is_some()
        || args.take.is_some()
        || args.filter.is_some()
        || args.sort.is_some()
        || args.include_archived
        || args.archived_only;

    if let Some(event_id) = args.event_id.as_ref() {
        if args.aggregate.is_some() || args.aggregate_id.is_some() {
            bail!("aggregate filters cannot be combined with --event");
        }
        if listing_options_used {
            bail!("listing options cannot be combined with --event");
        }
        return Ok(Some(event_id.clone()));
    }

    if args.aggregate_id.is_none() {
        if let Some(candidate) = args.aggregate.as_ref() {
            if SnowflakeId::from_str(candidate).is_ok() {
                if listing_options_used {
                    bail!("event inspection cannot be combined with listing filters");
                }
                return Ok(Some(candidate.clone()));
            }
        }
    }

    Ok(None)
}

fn parse_event_sort_directives(raw: &str) -> Result<Vec<EventSort>, String> {
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
        directives.push(parse_event_sort(spec)?);
    }

    Ok(directives)
}

fn parse_event_sort(spec: &str) -> Result<EventSort, String> {
    let mut parts = spec.split(':');
    let field_str = parts
        .next()
        .ok_or_else(|| "missing sort field".to_string())?
        .trim();

    if field_str.is_empty() {
        return Err("sort field cannot be empty".to_string());
    }

    let field = EventSortField::from_str(field_str)?;
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

    Ok(EventSort { field, descending })
}
