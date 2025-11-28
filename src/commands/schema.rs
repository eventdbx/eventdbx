use std::{collections::HashSet, fs, path::PathBuf, str::FromStr};

use anyhow::{Context, Result, anyhow};
use clap::{Args, Subcommand, ValueEnum};

use crate::commands::schema_version::{
    SchemaActivateArgs, SchemaDiffArgs, SchemaHistoryArgs, SchemaPublishArgs, SchemaReloadArgs,
    SchemaRollbackArgs, SchemaShowArgs, schema_activate, schema_diff, schema_history,
    schema_publish, schema_reload, schema_rollback, schema_show,
};
use eventdbx::{
    config::load_or_default,
    reference::{ReferenceCascade, ReferenceIntegrity, ReferenceRules},
    schema::{
        ColumnType, CreateSchemaInput, FieldFormat, FieldRules, MAX_EVENT_NOTE_LENGTH,
        SchemaManager, SchemaUpdate,
    },
    schema_history::SchemaAuditAction,
};
use serde::de::DeserializeOwned;
use serde_json;

#[derive(Subcommand)]
pub enum SchemaCommands {
    /// Create a new schema definition
    Create(SchemaCreateArgs),
    /// Add events to an existing schema definition
    Add(SchemaAddEventArgs),
    /// Remove an event definition from an aggregate
    Remove(SchemaRemoveEventArgs),
    /// Set or clear the default note for an event
    Annotate(SchemaAnnotateArgs),
    /// List available schemas
    List(SchemaListArgs),
    /// Hide a field from aggregate detail responses
    Hide(SchemaHideArgs),
    /// Modify column types and validation rules for a field
    Field(SchemaFieldArgs),
    /// Manage the field allow-list for a specific event
    Alter(SchemaAlterArgs),
    /// Validate payload against a schema definition
    Validate(SchemaValidateArgs),
    /// Publish a tenant-level schema snapshot
    Publish(SchemaPublishArgs),
    /// Show schema history for a tenant
    History(SchemaHistoryArgs),
    /// Show a specific schema version
    Show(SchemaShowArgs),
    /// Diff two schema versions
    Diff(SchemaDiffArgs),
    /// Activate a schema version
    Activate(SchemaActivateArgs),
    /// Roll back to a previous schema version
    Rollback(SchemaRollbackArgs),
    /// Reload the tenant schema cache
    Reload(SchemaReloadArgs),
    /// Fallback handler for positional aggregate commands
    #[command(external_subcommand)]
    External(Vec<String>),
}

#[derive(Args)]
pub struct SchemaCreateArgs {
    /// Aggregate name to initialize
    #[arg(value_name = "AGGREGATE")]
    pub aggregate: String,

    /// Comma-delimited list of events to seed the schema
    #[arg(short, long, required = true, value_delimiter = ',')]
    pub events: Vec<String>,

    /// Optional override for the snapshot threshold
    #[arg(short, long)]
    pub snapshot_threshold: Option<u64>,

    /// Emit JSON output
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

#[derive(Args, Default)]
pub struct SchemaListArgs {
    /// Emit JSON output
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

#[derive(Args)]
pub struct SchemaAddEventArgs {
    /// Aggregate name to modify
    pub aggregate: String,

    /// Event names to add (comma-delimited or repeated)
    #[arg(required = true, value_delimiter = ',')]
    pub events: Vec<String>,
}

#[derive(Args)]
pub struct SchemaRemoveEventArgs {
    /// Aggregate name
    pub aggregate: String,

    /// Event name to remove
    pub event: String,
}

#[derive(Args)]
pub struct SchemaAnnotateArgs {
    /// Aggregate name to modify
    pub aggregate: String,

    /// Event name to annotate
    pub event: String,

    /// Note text to set (omit to use --clear)
    #[arg(long, value_name = "NOTE")]
    pub note: Option<String>,

    /// Clear the existing note for the event
    #[arg(long, default_value_t = false)]
    pub clear: bool,
}

#[derive(Args)]
pub struct SchemaHideArgs {
    /// Aggregate name
    #[arg(long)]
    pub aggregate: String,

    /// Field/property name to hide
    #[arg(long)]
    pub field: String,

    /// Show the field again (removes it from the hidden list)
    #[arg(long, default_value_t = false)]
    pub show: bool,
}

#[derive(Args)]
pub struct SchemaFieldArgs {
    /// Aggregate name to modify
    #[arg(value_name = "AGGREGATE")]
    pub aggregate: String,

    /// Field/property name to alter
    #[arg(value_name = "FIELD")]
    pub field: String,

    /// Column data type to enforce (use 'reference' for ref fields)
    #[arg(long = "type", value_name = "TYPE", conflicts_with = "clear_type")]
    pub column_type: Option<String>,

    /// Remove the current column type and validation rules
    #[arg(long = "clear-type", default_value_t = false)]
    pub clear_type: bool,

    /// Replace the entire rules block with JSON
    #[arg(
        long = "rules",
        value_name = "JSON",
        conflicts_with_all = [
            "required",
            "optional",
            "format",
            "clear_format",
            "contains",
            "clear_contains",
            "does_not_contain",
            "clear_does_not_contain",
            "regex",
            "clear_regex",
            "length_min",
            "length_max",
            "clear_length",
            "range_min",
            "range_max",
            "clear_range",
            "properties",
            "clear_properties"
        ],
        conflicts_with = "clear_rules"
    )]
    pub rules: Option<String>,

    /// Clear all validation rules
    #[arg(long = "clear-rules", default_value_t = false)]
    pub clear_rules: bool,

    /// Require the field to be present in payloads
    #[arg(long, conflicts_with = "optional")]
    pub required: bool,

    /// Mark the field as optional
    #[arg(long = "not-required", conflicts_with = "required")]
    pub optional: bool,

    /// Built-in format validation to apply
    #[arg(long = "format", value_enum, value_name = "FORMAT")]
    pub format: Option<FieldFormatArg>,

    /// Remove any existing format rule
    #[arg(long = "clear-format", default_value_t = false)]
    pub clear_format: bool,

    /// Replace the allowed substring list
    #[arg(long = "contains", value_name = "TEXT", value_delimiter = ',')]
    pub contains: Option<Vec<String>>,

    /// Clear all `contains` checks
    #[arg(long = "clear-contains", default_value_t = false)]
    pub clear_contains: bool,

    /// Replace the disallowed substring list
    #[arg(long = "does-not-contain", value_name = "TEXT", value_delimiter = ',')]
    pub does_not_contain: Option<Vec<String>>,

    /// Clear all `does_not_contain` checks
    #[arg(long = "clear-does-not-contain", default_value_t = false)]
    pub clear_does_not_contain: bool,

    /// Replace the enforced regex patterns
    #[arg(long = "regex", value_name = "PATTERN", value_delimiter = ',')]
    pub regex: Option<Vec<String>>,

    /// Remove all regex validations
    #[arg(long = "clear-regex", default_value_t = false)]
    pub clear_regex: bool,

    /// Minimum accepted length for text/binary fields
    #[arg(long = "length-min", value_name = "N")]
    pub length_min: Option<usize>,

    /// Maximum accepted length for text/binary fields
    #[arg(long = "length-max", value_name = "N")]
    pub length_max: Option<usize>,

    /// Remove any length constraint
    #[arg(long = "clear-length", default_value_t = false)]
    pub clear_length: bool,

    /// Minimum accepted value for numeric/date/timestamp fields
    #[arg(long = "range-min", value_name = "VALUE")]
    pub range_min: Option<String>,

    /// Maximum accepted value for numeric/date/timestamp fields
    #[arg(long = "range-max", value_name = "VALUE")]
    pub range_max: Option<String>,

    /// Remove range constraints
    #[arg(long = "clear-range", default_value_t = false)]
    pub clear_range: bool,

    /// Replace nested property rules for object columns
    #[arg(long = "properties", value_name = "JSON")]
    pub properties: Option<String>,

    /// Clear nested object property rules
    #[arg(long = "clear-properties", default_value_t = false)]
    pub clear_properties: bool,

    /// Reference integrity (strong or weak)
    #[arg(long = "reference-integrity", value_enum)]
    pub reference_integrity: Option<ReferenceIntegrityArg>,

    /// Reference tenant/domain to enforce (only valid for reference fields)
    #[arg(long = "reference-tenant", value_name = "TENANT")]
    pub reference_tenant: Option<String>,

    /// Reference aggregate type to enforce (only valid for reference fields)
    #[arg(long = "reference-aggregate", value_name = "AGGREGATE")]
    pub reference_aggregate: Option<String>,

    /// Reference delete policy (none, restrict, nullify)
    #[arg(long = "reference-cascade", value_enum)]
    pub reference_cascade: Option<ReferenceCascadeArg>,

    /// Lock the field to prevent future updates
    #[arg(long = "lock", default_value_t = false, conflicts_with = "unlock")]
    pub lock: bool,

    /// Unlock the field to allow updates again
    #[arg(long = "unlock", default_value_t = false)]
    pub unlock: bool,
}

impl SchemaFieldArgs {
    fn has_type_mutation(&self) -> bool {
        self.column_type.is_some() || self.clear_type
    }

    fn has_rule_flag_updates(&self) -> bool {
        self.required
            || self.optional
            || self.format.is_some()
            || self.clear_format
            || self.contains.is_some()
            || self.clear_contains
            || self.does_not_contain.is_some()
            || self.clear_does_not_contain
            || self.regex.is_some()
            || self.clear_regex
            || self.length_min.is_some()
            || self.length_max.is_some()
            || self.clear_length
            || self.range_min.is_some()
            || self.range_max.is_some()
            || self.clear_range
            || self.properties.is_some()
            || self.clear_properties
            || self.reference_integrity.is_some()
            || self.reference_tenant.is_some()
            || self.reference_aggregate.is_some()
            || self.reference_cascade.is_some()
    }

    fn has_any_changes(&self) -> bool {
        self.has_type_mutation()
            || self.rules.is_some()
            || self.clear_rules
            || self.has_rule_flag_updates()
            || self.lock
            || self.unlock
    }

    fn apply_rule_overrides(&self, rules: &mut FieldRules, is_reference_field: bool) -> Result<()> {
        if self.required {
            rules.required = true;
        }
        if self.optional {
            rules.required = false;
        }
        if let Some(format) = self.format {
            rules.format = Some(FieldFormat::from(format));
        } else if self.clear_format {
            rules.format = None;
        }
        if let Some(values) = &self.contains {
            rules.contains = clone_non_empty(values, "contains")?;
        } else if self.clear_contains {
            rules.contains.clear();
        }
        if let Some(values) = &self.does_not_contain {
            rules.does_not_contain = clone_non_empty(values, "does-not-contain")?;
        } else if self.clear_does_not_contain {
            rules.does_not_contain.clear();
        }
        if let Some(values) = &self.regex {
            rules.regex = clone_non_empty(values, "regex")?;
        } else if self.clear_regex {
            rules.regex.clear();
        }
        if self.clear_length {
            rules.length = None;
        } else if self.length_min.is_some() || self.length_max.is_some() {
            let mut length = rules.length.clone().unwrap_or_default();
            if let Some(min) = self.length_min {
                length.min = Some(min);
            }
            if let Some(max) = self.length_max {
                length.max = Some(max);
            }
            rules.length = Some(length);
        }
        if self.clear_range {
            rules.range = None;
        } else if self.range_min.is_some() || self.range_max.is_some() {
            let mut range = rules.range.clone().unwrap_or_default();
            if let Some(min) = &self.range_min {
                range.min = Some(min.clone());
            }
            if let Some(max) = &self.range_max {
                range.max = Some(max.clone());
            }
            rules.range = Some(range);
        }
        if self.clear_properties {
            rules.properties.clear();
        } else if let Some(json) = &self.properties {
            rules.properties = parse_json_input(json, "properties")?;
        }

        let reference_flags_present = self.reference_integrity.is_some()
            || self.reference_tenant.is_some()
            || self.reference_aggregate.is_some()
            || self.reference_cascade.is_some();

        if reference_flags_present {
            if rules.format != Some(FieldFormat::Reference) && !is_reference_field {
                return Err(anyhow!(
                    "reference flags require a reference field; set --type reference or --format reference"
                ));
            }
            rules.format = Some(FieldFormat::Reference);
            let mut ref_rules = rules.reference.clone().unwrap_or_default();
            if let Some(value) = self.reference_integrity {
                ref_rules.integrity = value.into();
            }
            if let Some(value) = &self.reference_tenant {
                ref_rules.tenant = Some(value.trim().to_string());
            }
            if let Some(value) = &self.reference_aggregate {
                ref_rules.aggregate_type = Some(value.trim().to_string());
            }
            if let Some(value) = self.reference_cascade {
                ref_rules.cascade = value.into();
            }
            rules.reference = Some(ref_rules);
        }
        Ok(())
    }
}

fn is_reference_type_alias(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "ref" | "reference" | "aggregate_ref" | "aggregate-reference"
    )
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
pub enum FieldFormatArg {
    #[value(name = "email")]
    Email,
    #[value(name = "url")]
    Url,
    #[value(name = "credit_card")]
    CreditCard,
    #[value(name = "country_code")]
    CountryCode,
    #[value(name = "iso_8601")]
    Iso8601,
    #[value(name = "wgs_84")]
    Wgs84,
    #[value(name = "camel_case")]
    CamelCase,
    #[value(name = "snake_case")]
    SnakeCase,
    #[value(name = "kebab_case")]
    KebabCase,
    #[value(name = "pascal_case")]
    PascalCase,
    #[value(name = "upper_case_snake_case")]
    UpperCaseSnakeCase,
    #[value(name = "reference")]
    Reference,
}

impl From<FieldFormatArg> for FieldFormat {
    fn from(value: FieldFormatArg) -> Self {
        match value {
            FieldFormatArg::Email => FieldFormat::Email,
            FieldFormatArg::Url => FieldFormat::Url,
            FieldFormatArg::CreditCard => FieldFormat::CreditCard,
            FieldFormatArg::CountryCode => FieldFormat::CountryCode,
            FieldFormatArg::Iso8601 => FieldFormat::Iso8601,
            FieldFormatArg::Wgs84 => FieldFormat::Wgs84,
            FieldFormatArg::CamelCase => FieldFormat::CamelCase,
            FieldFormatArg::SnakeCase => FieldFormat::SnakeCase,
            FieldFormatArg::KebabCase => FieldFormat::KebabCase,
            FieldFormatArg::PascalCase => FieldFormat::PascalCase,
            FieldFormatArg::UpperCaseSnakeCase => FieldFormat::UpperCaseSnakeCase,
            FieldFormatArg::Reference => FieldFormat::Reference,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
pub enum ReferenceIntegrityArg {
    #[value(name = "strong")]
    Strong,
    #[value(name = "weak")]
    Weak,
}

impl From<ReferenceIntegrityArg> for ReferenceIntegrity {
    fn from(value: ReferenceIntegrityArg) -> Self {
        match value {
            ReferenceIntegrityArg::Strong => ReferenceIntegrity::Strong,
            ReferenceIntegrityArg::Weak => ReferenceIntegrity::Weak,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
pub enum ReferenceCascadeArg {
    #[value(name = "none")]
    None,
    #[value(name = "restrict")]
    Restrict,
    #[value(name = "nullify")]
    Nullify,
}

impl From<ReferenceCascadeArg> for ReferenceCascade {
    fn from(value: ReferenceCascadeArg) -> Self {
        match value {
            ReferenceCascadeArg::None => ReferenceCascade::None,
            ReferenceCascadeArg::Restrict => ReferenceCascade::Restrict,
            ReferenceCascadeArg::Nullify => ReferenceCascade::Nullify,
        }
    }
}

#[derive(Args)]
pub struct SchemaAlterArgs {
    /// Aggregate name whose event definition will change
    #[arg(value_name = "AGGREGATE")]
    pub aggregate: String,

    /// Event name to update
    #[arg(value_name = "EVENT")]
    pub event: String,

    /// Append fields to the allow-list
    #[arg(
        long = "add",
        value_name = "FIELD",
        value_delimiter = ',',
        conflicts_with_all = ["set", "clear"]
    )]
    pub add: Option<Vec<String>>,

    /// Remove fields from the allow-list
    #[arg(
        long = "remove",
        value_name = "FIELD",
        value_delimiter = ',',
        conflicts_with_all = ["set", "clear"]
    )]
    pub remove: Option<Vec<String>>,

    /// Replace the allow-list entirely
    #[arg(
        long = "set",
        value_name = "FIELD",
        value_delimiter = ',',
        conflicts_with_all = ["add", "remove", "clear"]
    )]
    pub set: Option<Vec<String>>,

    /// Clear all allowed fields (equivalent to an empty set)
    #[arg(long = "clear", default_value_t = false, conflicts_with_all = ["set", "add", "remove"])]
    pub clear: bool,
}

impl SchemaAlterArgs {
    fn has_mutation(&self) -> bool {
        self.set.is_some() || self.clear || self.add.is_some() || self.remove.is_some()
    }
}

#[derive(Args)]
pub struct SchemaValidateArgs {
    /// Aggregate name to validate against
    #[arg(long)]
    pub aggregate: String,

    /// Event name to validate against
    #[arg(long)]
    pub event: String,

    /// JSON payload to validate
    #[arg(long)]
    pub payload: String,
}

pub fn execute(config_path: Option<PathBuf>, command: SchemaCommands) -> Result<()> {
    match command {
        SchemaCommands::Hide(args) => hide_field(config_path, args),
        SchemaCommands::Field(args) => schema_field(config_path, args),
        SchemaCommands::Alter(args) => schema_alter(config_path, args),
        SchemaCommands::Create(args) => {
            let (config, _) = load_or_default(config_path)?;
            let manager = SchemaManager::load(config.schema_store_path())?;
            let schema = manager.create(CreateSchemaInput {
                aggregate: args.aggregate,
                events: args.events,
                snapshot_threshold: args.snapshot_threshold.or(config.snapshot_threshold),
            })?;
            if args.json {
                println!("{}", serde_json::to_string_pretty(&schema)?);
            } else {
                println!(
                    "schema={} events={} snapshot_threshold={:?}",
                    schema.aggregate,
                    schema.events.len(),
                    schema.snapshot_threshold
                );
            }
            Ok(())
        }
        SchemaCommands::Add(args) => {
            let (config, _) = load_or_default(config_path)?;
            let manager = SchemaManager::load(config.schema_store_path())?;
            manager.get(&args.aggregate)?;
            let mut update = SchemaUpdate::default();
            for event in &args.events {
                update.event_add_fields.entry(event.clone()).or_default();
            }
            let schema = manager.update(&args.aggregate, update)?;
            println!(
                "schema={} added_events={} total_events={}",
                schema.aggregate,
                args.events.join(","),
                schema.events.len()
            );
            Ok(())
        }
        SchemaCommands::Remove(args) => {
            let (config, _) = load_or_default(config_path)?;
            let manager = SchemaManager::load(config.schema_store_path())?;
            let schema = manager.remove_event(&args.aggregate, &args.event)?;
            println!(
                "schema={} removed_event={} remaining_events={}",
                schema.aggregate,
                args.event,
                schema.events.len()
            );
            Ok(())
        }
        SchemaCommands::Annotate(args) => annotate_event(config_path, args),
        SchemaCommands::List(args) => {
            let (config, _) = load_or_default(config_path)?;
            let manager = SchemaManager::load(config.schema_store_path())?;
            let schemas = manager.list();
            if args.json {
                println!("{}", serde_json::to_string_pretty(&schemas)?);
            } else {
                for schema in schemas {
                    println!(
                        "schema={} events={} locked={} snapshot_threshold={:?}",
                        schema.aggregate,
                        schema.events.len(),
                        schema.locked,
                        schema.snapshot_threshold
                    );
                }
            }
            Ok(())
        }
        SchemaCommands::Validate(args) => {
            let (config, _) = load_or_default(config_path)?;
            let manager = SchemaManager::load(config.schema_store_path())?;
            let payload: serde_json::Value = serde_json::from_str(&args.payload)?;
            manager.validate_event(&args.aggregate, &args.event, &payload)?;
            println!(
                "aggregate={} event={} validation=ok",
                args.aggregate, args.event
            );
            Ok(())
        }
        SchemaCommands::Publish(args) => schema_publish(config_path, args),
        SchemaCommands::History(args) => schema_history(config_path, args),
        SchemaCommands::Show(args) => schema_show(config_path, args),
        SchemaCommands::Diff(args) => schema_diff(config_path, args),
        SchemaCommands::Activate(args) => {
            schema_activate(config_path, args, SchemaAuditAction::Activate, "activated")
        }
        SchemaCommands::Rollback(args) => schema_rollback(config_path, args),
        SchemaCommands::Reload(args) => schema_reload(config_path, args),
        SchemaCommands::External(args) => {
            let (config, _) = load_or_default(config_path)?;
            let manager = SchemaManager::load(config.schema_store_path())?;
            match args.as_slice() {
                [] => Err(anyhow!(
                    "missing aggregate name; try `eventdbx schema <aggregate>`"
                )),
                [aggregate] => {
                    let schema = manager.get(aggregate)?;
                    println!("{}", serde_json::to_string_pretty(&schema)?);
                    Ok(())
                }
                [command, aggregate] if command.eq_ignore_ascii_case("show") => {
                    let schema = manager.get(aggregate)?;
                    println!("{}", serde_json::to_string_pretty(&schema)?);
                    Ok(())
                }
                _ => Err(anyhow!(
                    "unsupported schema command; available subcommands are create, add, remove, list"
                )),
            }
        }
    }
}

fn annotate_event(config_path: Option<PathBuf>, args: SchemaAnnotateArgs) -> Result<()> {
    let (config, _) = load_or_default(config_path)?;
    let manager = SchemaManager::load(config.schema_store_path())?;

    let aggregate = args.aggregate.trim();
    if aggregate.is_empty() {
        return Err(anyhow!("aggregate name cannot be empty"));
    }

    let event = args.event.trim();
    if event.is_empty() {
        return Err(anyhow!("event name cannot be empty"));
    }

    if args.clear && args.note.is_some() {
        return Err(anyhow!("--note cannot be combined with --clear"));
    }

    let desired_note = if args.clear {
        None
    } else if let Some(note) = args.note {
        if note.chars().count() > MAX_EVENT_NOTE_LENGTH {
            return Err(anyhow!(
                "note cannot exceed {} characters",
                MAX_EVENT_NOTE_LENGTH
            ));
        }
        Some(note)
    } else {
        return Err(anyhow!("either --note must be provided or use --clear"));
    };

    let schema = manager.get(aggregate)?;
    if !schema.events.contains_key(event) {
        return Err(anyhow!(
            "event {} is not defined for aggregate {}",
            event,
            aggregate
        ));
    }

    let mut update = SchemaUpdate::default();
    update
        .event_notes
        .insert(event.to_string(), desired_note.clone());
    manager.update(aggregate, update)?;

    match desired_note {
        Some(note) => println!(
            "aggregate={} event={} note_set=\"{}\"",
            aggregate, event, note
        ),
        None => println!("aggregate={} event={} note_cleared", aggregate, event),
    }

    Ok(())
}

fn hide_field(config_path: Option<PathBuf>, args: SchemaHideArgs) -> Result<()> {
    let (config, _) = load_or_default(config_path)?;
    let manager = SchemaManager::load(config.schema_store_path())?;

    let aggregate = args.aggregate.trim();
    if aggregate.is_empty() {
        return Err(anyhow!("aggregate name cannot be empty"));
    }
    let field = args.field.trim();
    if field.is_empty() {
        return Err(anyhow!("field name cannot be empty"));
    }

    // Ensure aggregate exists (best-effort)
    if manager.get(aggregate).is_err() {
        tracing::warn!(
            "hiding field `{}` for unknown aggregate `{}`; field visibility will still be enforced",
            field,
            aggregate
        );
    }

    let mut update = SchemaUpdate::default();
    update.hidden_field = Some((field.to_string(), !args.show));
    manager.update(aggregate, update)?;

    if args.show {
        println!("aggregate={} field={} shown", aggregate, field);
    } else {
        println!("aggregate={} field={} hidden", aggregate, field);
    }
    Ok(())
}

fn schema_field(config_path: Option<PathBuf>, args: SchemaFieldArgs) -> Result<()> {
    let aggregate = args.aggregate.trim();
    if aggregate.is_empty() {
        return Err(anyhow!("aggregate name cannot be empty"));
    }

    let field = args.field.trim();
    if field.is_empty() {
        return Err(anyhow!("field name cannot be empty"));
    }

    if !args.has_any_changes() {
        return Err(anyhow!(
            "no changes requested; pass --type, --rules, or at least one rule flag"
        ));
    }

    let reference_alias = args
        .column_type
        .as_deref()
        .map(is_reference_type_alias)
        .unwrap_or(false);
    let has_explicit_rule_change =
        args.rules.is_some() || args.clear_rules || args.has_rule_flag_updates();

    if args.clear_type && (args.rules.is_some() || args.clear_rules || args.has_rule_flag_updates())
    {
        return Err(anyhow!(
            "--clear-type cannot be combined with rule changes; removing the type deletes all rules automatically"
        ));
    }

    let (config, _) = load_or_default(config_path)?;
    let manager = SchemaManager::load(config.schema_store_path())?;
    let schema = manager.get(aggregate)?;

    let mut update = SchemaUpdate::default();
    let mut actions = Vec::new();

    if let Some(ref value) = args.column_type {
        if reference_alias {
            update.column_type = Some((field.to_string(), Some(ColumnType::Text)));
            actions.push("type=reference".to_string());
        } else {
            let column_type =
                ColumnType::from_str(value).map_err(|err| anyhow!("invalid column type: {err}"))?;
            update.column_type = Some((field.to_string(), Some(column_type.clone())));
            actions.push(format!("type={column_type}"));
        }
    } else if args.clear_type {
        update.column_type = Some((field.to_string(), None));
        actions.push("type=cleared".to_string());
    }

    let mut reference_flags_present = args.reference_integrity.is_some()
        || args.reference_tenant.is_some()
        || args.reference_aggregate.is_some()
        || args.reference_cascade.is_some();

    if let Some(ref rules_json) = args.rules {
        let rules: FieldRules = parse_json_input(rules_json, "rules")?;
        update.column_rules = Some((field.to_string(), Some(rules)));
        actions.push("rules=updated".to_string());
    } else if args.clear_rules {
        if schema.column_types.get(field).is_none() && args.column_type.is_none() {
            return Err(anyhow!("field {} does not have rules to clear", field));
        }
        update.column_rules = Some((field.to_string(), None));
        actions.push("rules=cleared".to_string());
    } else if args.has_rule_flag_updates() {
        let mut rules = if let Some(settings) = schema.column_types.get(field) {
            settings.rules.clone()
        } else if args.column_type.is_some() {
            FieldRules::default()
        } else {
            return Err(anyhow!(
                "field {} has no column type; set --type before editing rules",
                field
            ));
        };
        let is_reference_field = reference_alias
            || args.format == Some(FieldFormatArg::Reference)
            || rules.format == Some(FieldFormat::Reference);
        args.apply_rule_overrides(&mut rules, is_reference_field)?;
        reference_flags_present |= args.reference_integrity.is_some()
            || args.reference_tenant.is_some()
            || args.reference_aggregate.is_some()
            || args.reference_cascade.is_some();
        update.column_rules = Some((field.to_string(), Some(rules)));
        actions.push("rules=updated".to_string());
    }

    if reference_alias && !has_explicit_rule_change && !args.clear_type {
        let mut rules = if let Some(settings) = schema.column_types.get(field) {
            settings.rules.clone()
        } else {
            FieldRules::default()
        };
        rules.format = Some(FieldFormat::Reference);
        rules.reference = Some(rules.reference.unwrap_or_else(ReferenceRules::default));
        update.column_rules = Some((field.to_string(), Some(rules)));
        actions.push("rules=reference".to_string());
    }

    if reference_flags_present
        && !reference_alias
        && args.format != Some(FieldFormatArg::Reference)
        && update.column_rules.as_ref().map_or(
            true,
            |(_, rules)| matches!(rules, Some(r) if r.format == Some(FieldFormat::Reference)),
        )
    {
        // if rules were directly set via JSON, allow; otherwise enforce format
        if update.column_rules.is_none() {
            return Err(anyhow!(
                "reference flags require a reference field; set --type reference or --format reference"
            ));
        }
    }

    if args.lock {
        update.field_lock = Some((field.to_string(), true));
        actions.push("locked=true".to_string());
    } else if args.unlock {
        update.field_lock = Some((field.to_string(), false));
        actions.push("locked=false".to_string());
    }

    if actions.is_empty() {
        // Should be unreachable thanks to earlier guard, but double-check to avoid silent no-ops.
        return Err(anyhow!(
            "no valid updates were derived from the provided options"
        ));
    }

    manager.update(aggregate, update)?;

    println!(
        "aggregate={} field={} {}",
        aggregate,
        field,
        actions.join(" ")
    );
    Ok(())
}

fn schema_alter(config_path: Option<PathBuf>, args: SchemaAlterArgs) -> Result<()> {
    let aggregate = args.aggregate.trim();
    if aggregate.is_empty() {
        return Err(anyhow!("aggregate name cannot be empty"));
    }

    let event = args.event.trim();
    if event.is_empty() {
        return Err(anyhow!("event name cannot be empty"));
    }

    if !args.has_mutation() {
        return Err(anyhow!(
            "no changes requested; supply --add, --remove, --set, or --clear"
        ));
    }

    let (config, _) = load_or_default(config_path)?;
    let manager = SchemaManager::load(config.schema_store_path())?;
    let schema = manager.get(aggregate)?;
    if !schema.events.contains_key(event) {
        return Err(anyhow!(
            "event {} is not defined for aggregate {}",
            event,
            aggregate
        ));
    }

    let mut update = SchemaUpdate::default();
    let mut actions = Vec::new();

    if let Some(set_fields) = &args.set {
        let normalized = normalize_field_names(set_fields, "--set")?;
        update
            .event_set_fields
            .insert(event.to_string(), normalized.clone());
        actions.push(format!(
            "fields={}",
            if normalized.is_empty() {
                "[]".to_string()
            } else {
                normalized.join(",")
            }
        ));
    } else if args.clear {
        update
            .event_set_fields
            .insert(event.to_string(), Vec::new());
        actions.push("fields=cleared".to_string());
    } else {
        if let Some(add_fields) = &args.add {
            let normalized = normalize_field_names(add_fields, "--add")?;
            update
                .event_add_fields
                .entry(event.to_string())
                .or_default()
                .extend(normalized.clone());
            actions.push(format!("added={}", normalized.join(",")));
        }
        if let Some(remove_fields) = &args.remove {
            let normalized = normalize_field_names(remove_fields, "--remove")?;
            update
                .event_remove_fields
                .entry(event.to_string())
                .or_default()
                .extend(normalized.clone());
            actions.push(format!("removed={}", normalized.join(",")));
        }
    }

    if actions.is_empty() {
        return Err(anyhow!(
            "no valid field names supplied; please check the provided arguments"
        ));
    }

    manager.update(aggregate, update)?;

    println!(
        "aggregate={} event={} {}",
        aggregate,
        event,
        actions.join(" ")
    );
    Ok(())
}

fn parse_json_input<T>(input: &str, label: &str) -> Result<T>
where
    T: DeserializeOwned,
{
    let source = if let Some(path) = input.strip_prefix('@') {
        fs::read_to_string(path).with_context(|| format!("failed to read {label} from {path}"))?
    } else {
        input.to_string()
    };
    serde_json::from_str(&source)
        .with_context(|| format!("failed to parse {label} JSON: {}", input))
}

fn clone_non_empty(values: &[String], label: &str) -> Result<Vec<String>> {
    let mut items = Vec::with_capacity(values.len());
    for value in values {
        if value.trim().is_empty() {
            return Err(anyhow!("{label} entries cannot be empty"));
        }
        items.push(value.clone());
    }
    Ok(items)
}

fn normalize_field_names(values: &[String], flag: &str) -> Result<Vec<String>> {
    if values.is_empty() {
        return Err(anyhow!("{flag} requires at least one field name"));
    }
    let mut seen = HashSet::new();
    let mut normalized = Vec::with_capacity(values.len());
    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(anyhow!("{flag} entries cannot be empty"));
        }
        if seen.insert(trimmed.to_string()) {
            normalized.push(trimmed.to_string());
        }
    }
    Ok(normalized)
}
