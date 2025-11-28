use std::{collections::BTreeMap, fmt, fs, path::PathBuf, str::FromStr};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_ENGINE};
use chrono::{DateTime, NaiveDate, Utc};
use parking_lot::RwLock;
use regex::Regex;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value};
use validator::{validate_credit_card, validate_email, validate_url};

use crate::reference::{
    AggregateReference, ReferenceContext, ReferenceIntegrity, ReferenceOutcome,
    ReferenceResolutionStatus, ReferenceRules,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum ColumnType {
    Integer,
    Float,
    Decimal { precision: u8, scale: u8 },
    Boolean,
    Text,
    Timestamp,
    Date,
    Json,
    Binary,
    Object,
}

type ColumnTypeResult<T> = std::result::Result<T, ColumnTypeParseError>;

impl ColumnType {
    fn as_str(&self) -> String {
        match self {
            ColumnType::Integer => "integer".to_string(),
            ColumnType::Float => "float".to_string(),
            ColumnType::Decimal { precision, scale } => {
                format!("decimal({},{})", precision, scale)
            }
            ColumnType::Boolean => "boolean".to_string(),
            ColumnType::Text => "text".to_string(),
            ColumnType::Timestamp => "timestamp".to_string(),
            ColumnType::Date => "date".to_string(),
            ColumnType::Json => "json".to_string(),
            ColumnType::Binary => "binary".to_string(),
            ColumnType::Object => "object".to_string(),
        }
    }

    fn parse_decimal(value: &str) -> ColumnTypeResult<Self> {
        let start = value
            .find('(')
            .ok_or_else(|| ColumnTypeParseError(value.to_string()))?;
        let end = value
            .rfind(')')
            .filter(|pos| *pos > start)
            .ok_or_else(|| ColumnTypeParseError(value.to_string()))?;
        let inner = &value[start + 1..end];
        let mut parts = inner.split(',').map(|part| part.trim());
        let precision = parts
            .next()
            .ok_or_else(|| ColumnTypeParseError(value.to_string()))?
            .parse()
            .map_err(|_| ColumnTypeParseError(value.to_string()))?;
        let scale = parts
            .next()
            .ok_or_else(|| ColumnTypeParseError(value.to_string()))?
            .parse()
            .map_err(|_| ColumnTypeParseError(value.to_string()))?;

        if parts.next().is_some() {
            return Err(ColumnTypeParseError(value.to_string()));
        }

        Ok(ColumnType::Decimal { precision, scale })
    }
}

impl fmt::Display for ColumnType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<ColumnType> for String {
    fn from(value: ColumnType) -> Self {
        value.as_str()
    }
}

impl TryFrom<String> for ColumnType {
    type Error = ColumnTypeParseError;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        ColumnType::from_str(&value)
    }
}

impl FromStr for ColumnType {
    type Err = ColumnTypeParseError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(ColumnTypeParseError(value.to_string()));
        }
        let lower = trimmed.to_ascii_lowercase();
        match lower.as_str() {
            "integer" | "int" => Ok(ColumnType::Integer),
            "float" | "double" => Ok(ColumnType::Float),
            "boolean" | "bool" => Ok(ColumnType::Boolean),
            "text" | "string" => Ok(ColumnType::Text),
            "timestamp" => Ok(ColumnType::Timestamp),
            "date" => Ok(ColumnType::Date),
            "json" => Ok(ColumnType::Json),
            "binary" | "bytes" => Ok(ColumnType::Binary),
            "object" => Ok(ColumnType::Object),
            _ if lower.starts_with("decimal(") || lower.starts_with("numeric(") => {
                ColumnType::parse_decimal(trimmed)
            }
            _ => Err(ColumnTypeParseError(value.to_string())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ColumnTypeParseError(String);

impl fmt::Display for ColumnTypeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid column type '{}'", self.0)
    }
}

impl std::error::Error for ColumnTypeParseError {}

// Manual implementation of `Deserialize` for `FieldRules` is provided elsewhere.
// This is necessary to ignore the deprecated `must_match` field during deserialization.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, Default)]
#[serde(default)]
pub struct FieldRules {
    #[serde(skip_serializing_if = "is_false")]
    pub required: bool,
    #[serde(skip_serializing_if = "vec_is_empty")]
    pub contains: Vec<String>,
    #[serde(skip_serializing_if = "vec_is_empty")]
    pub does_not_contain: Vec<String>,
    #[serde(skip_serializing_if = "vec_is_empty")]
    pub regex: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub length: Option<LengthRule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<RangeRule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<FieldFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<ReferenceRules>,
    #[serde(skip_serializing_if = "map_is_empty")]
    pub properties: BTreeMap<String, ColumnSettings>,
}

impl FieldRules {
    fn is_default(&self) -> bool {
        !self.required
            && self.contains.is_empty()
            && self.does_not_contain.is_empty()
            && self.regex.is_empty()
            && self.length.is_none()
            && self.range.is_none()
            && self.format.is_none()
            && self
                .reference
                .as_ref()
                .map_or(true, |rules| rules == &ReferenceRules::default())
            && self.properties.is_empty()
    }

    fn validate_rules(
        &self,
        path: &str,
        column_type: &ColumnType,
        value: &ComparableValue,
    ) -> Result<()> {
        if self.reference.is_some() && self.format != Some(FieldFormat::Reference) {
            return Err(EventError::SchemaViolation(format!(
                "field {} specifies reference rules without reference format",
                path
            )));
        }

        if !self.contains.is_empty()
            || !self.does_not_contain.is_empty()
            || !self.regex.is_empty()
            || self.format.is_some()
        {
            let text = value.as_text().ok_or_else(|| {
                EventError::SchemaViolation(format!("field {} must be a string", path))
            })?;

            for needle in &self.contains {
                if !text.contains(needle) {
                    return Err(EventError::SchemaViolation(format!(
                        "field {} must contain '{}'",
                        path, needle
                    )));
                }
            }

            for needle in &self.does_not_contain {
                if text.contains(needle) {
                    return Err(EventError::SchemaViolation(format!(
                        "field {} must not contain '{}'",
                        path, needle
                    )));
                }
            }

            for pattern in &self.regex {
                let regex = Regex::new(pattern).map_err(|err| {
                    EventError::SchemaViolation(format!(
                        "field {} has invalid regex '{}': {}",
                        path, pattern, err
                    ))
                })?;
                if !regex.is_match(text) {
                    return Err(EventError::SchemaViolation(format!(
                        "field {} must match pattern {}",
                        path, pattern
                    )));
                }
            }

            if let Some(format) = &self.format {
                match format {
                    FieldFormat::Email => {
                        if !validate_email(text) {
                            return Err(EventError::SchemaViolation(format!(
                                "field {} must be a valid email address",
                                path
                            )));
                        }
                    }
                    FieldFormat::Url => {
                        if !validate_url(text) {
                            return Err(EventError::SchemaViolation(format!(
                                "field {} must be a valid URL",
                                path
                            )));
                        }
                    }
                    FieldFormat::CreditCard => {
                        if !validate_credit_card(text) {
                            return Err(EventError::SchemaViolation(format!(
                                "field {} must be a valid credit card number",
                                path
                            )));
                        }
                    }
                    FieldFormat::CountryCode => {
                        if !validate_country_code(text) {
                            return Err(EventError::SchemaViolation(format!(
                                "field {} must be an ISO 3166-1 alpha-2 country code",
                                path
                            )));
                        }
                    }
                    FieldFormat::Iso8601 => {
                        if DateTime::parse_from_rfc3339(text).is_err() {
                            return Err(EventError::SchemaViolation(format!(
                                "field {} must be a valid ISO 8601 timestamp",
                                path
                            )));
                        }
                    }
                    FieldFormat::Wgs84 => {
                        if !validate_wgs84(text) {
                            return Err(EventError::SchemaViolation(format!(
                                "field {} must be a valid WGS 84 coordinate (lat,lon)",
                                path
                            )));
                        }
                    }
                    FieldFormat::CamelCase => {
                        if !validate_camel_case(text) {
                            return Err(EventError::SchemaViolation(format!(
                                "field {} must be camelCase",
                                path
                            )));
                        }
                    }
                    FieldFormat::SnakeCase => {
                        if !validate_snake_case(text) {
                            return Err(EventError::SchemaViolation(format!(
                                "field {} must be snake_case",
                                path
                            )));
                        }
                    }
                    FieldFormat::KebabCase => {
                        if !validate_kebab_case(text) {
                            return Err(EventError::SchemaViolation(format!(
                                "field {} must be kebab-case",
                                path
                            )));
                        }
                    }
                    FieldFormat::PascalCase => {
                        if !validate_pascal_case(text) {
                            return Err(EventError::SchemaViolation(format!(
                                "field {} must be PascalCase",
                                path
                            )));
                        }
                    }
                    FieldFormat::UpperCaseSnakeCase => {
                        if !validate_upper_case_snake_case(text) {
                            return Err(EventError::SchemaViolation(format!(
                                "field {} must be UPPER_CASE_SNAKE_CASE",
                                path
                            )));
                        }
                    }
                    FieldFormat::Reference => {
                        validate_reference_shape(path, text)?;
                    }
                }
            }
        } else if let Some(length) = &self.length {
            match value {
                ComparableValue::Text(text) => length.check(path, text.chars().count())?,
                ComparableValue::Binary(bytes) => length.check(path, bytes.len())?,
                _ => {
                    return Err(EventError::SchemaViolation(format!(
                        "field {} does not support length validation",
                        path
                    )));
                }
            }
        }

        if let Some(range) = &self.range {
            match value {
                ComparableValue::Integer(current) => {
                    check_integer_range(path, *current, range)?;
                }
                ComparableValue::Float(current) => {
                    check_float_range(path, *current, range)?;
                }
                ComparableValue::Decimal(current) => {
                    check_decimal_range(path, current, range)?;
                }
                ComparableValue::Timestamp(current) => {
                    check_timestamp_range(path, current, range)?;
                }
                ComparableValue::Date(current) => {
                    check_date_range(path, current, range)?;
                }
                _ => {
                    return Err(EventError::SchemaViolation(format!(
                        "field {} does not support range validation",
                        path
                    )));
                }
            }
        }

        if !self.properties.is_empty()
            && !matches!(column_type, ColumnType::Object | ColumnType::Json)
        {
            return Err(EventError::SchemaViolation(format!(
                "field {} cannot define nested properties on non-object type",
                path
            )));
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct LocatedReference {
    pub path: String,
    pub reference: AggregateReference,
    pub integrity: ReferenceIntegrity,
}

impl<'de> Deserialize<'de> for FieldRules {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize, Default)]
        #[serde(default)]
        struct FieldRulesHelper {
            #[serde(default)]
            required: bool,
            #[serde(default)]
            contains: Vec<String>,
            #[serde(default)]
            does_not_contain: Vec<String>,
            #[serde(default)]
            // Retained for backward compatibility during deserialization of old schema files.
            #[allow(dead_code)]
            must_match: Option<String>,
            #[serde(default)]
            regex: Vec<String>,
            #[serde(default)]
            length: Option<LengthRule>,
            #[serde(default)]
            range: Option<RangeRule>,
            #[serde(default)]
            format: Option<FieldFormat>,
            #[serde(default)]
            reference: Option<ReferenceRules>,
            #[serde(default)]
            properties: BTreeMap<String, ColumnSettings>,
        }

        let helper = FieldRulesHelper::deserialize(deserializer)?;
        Ok(FieldRules {
            required: helper.required,
            contains: helper.contains,
            does_not_contain: helper.does_not_contain,
            regex: helper.regex,
            length: helper.length,
            range: helper.range,
            format: helper.format,
            reference: helper.reference,
            properties: helper.properties,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(default)]
pub struct LengthRule {
    pub min: Option<usize>,
    pub max: Option<usize>,
}

impl LengthRule {
    fn check(&self, path: &str, len: usize) -> Result<()> {
        if let Some(min) = self.min {
            if len < min {
                return Err(EventError::SchemaViolation(format!(
                    "field {} must have length >= {}",
                    path, min
                )));
            }
        }
        if let Some(max) = self.max {
            if len > max {
                return Err(EventError::SchemaViolation(format!(
                    "field {} must have length <= {}",
                    path, max
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(default)]
pub struct RangeRule {
    pub min: Option<String>,
    pub max: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FieldFormat {
    Email,
    Url,
    CreditCard,
    CountryCode,
    Iso8601,
    #[serde(rename = "wgs_84")]
    Wgs84,
    CamelCase,
    SnakeCase,
    KebabCase,
    PascalCase,
    UpperCaseSnakeCase,
    Reference,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnSettings {
    pub column_type: ColumnType,
    pub rules: FieldRules,
}

impl ColumnSettings {
    fn from_type(column_type: ColumnType) -> Self {
        Self {
            column_type,
            rules: FieldRules::default(),
        }
    }

    fn validate_with_path(
        &self,
        path: &str,
        value_opt: Option<&Value>,
        root_value: &Value,
        root_definitions: &BTreeMap<String, ColumnSettings>,
    ) -> Result<()> {
        if self.rules.format == Some(FieldFormat::Reference)
            && !matches!(self.column_type, ColumnType::Text)
        {
            return Err(EventError::SchemaViolation(format!(
                "field {} uses reference format but must be a text column",
                path
            )));
        }
        if self.rules.reference.is_some() && self.rules.format != Some(FieldFormat::Reference) {
            return Err(EventError::SchemaViolation(format!(
                "field {} specifies reference rules without reference format",
                path
            )));
        }

        if value_opt.is_none() {
            if self.rules.required {
                return Err(EventError::SchemaViolation(format!(
                    "field {} is required",
                    path
                )));
            }
            return Ok(());
        }

        let value = value_opt.expect("value checked above");
        let comparable = self.coerce_value(path, value, root_value, root_definitions)?;
        self.rules
            .validate_rules(path, &self.column_type, &comparable)?;
        Ok(())
    }

    fn coerce_value(
        &self,
        path: &str,
        value: &Value,
        root_value: &Value,
        root_definitions: &BTreeMap<String, ColumnSettings>,
    ) -> Result<ComparableValue> {
        match &self.column_type {
            ColumnType::Integer => Ok(ComparableValue::Integer(coerce_integer(value, path)?)),
            ColumnType::Float => Ok(ComparableValue::Float(coerce_float(value, path)?)),
            ColumnType::Decimal { precision, scale } => {
                let decimal = coerce_decimal(value, path)?;
                check_decimal_constraints(&decimal, *precision, *scale, path)?;
                Ok(ComparableValue::Decimal(decimal))
            }
            ColumnType::Boolean => Ok(ComparableValue::Boolean(coerce_boolean(value, path)?)),
            ColumnType::Text => Ok(ComparableValue::Text(coerce_text(value, path)?)),
            ColumnType::Timestamp => Ok(ComparableValue::Timestamp(coerce_timestamp(value, path)?)),
            ColumnType::Date => Ok(ComparableValue::Date(coerce_date(value, path)?)),
            ColumnType::Json => Ok(ComparableValue::Json(value.clone())),
            ColumnType::Binary => Ok(ComparableValue::Binary(coerce_binary(value, path)?)),
            ColumnType::Object => {
                let object = value.as_object().ok_or_else(|| {
                    EventError::SchemaViolation(format!("field {} must be a JSON object", path))
                })?;
                validate_columns(
                    &self.rules.properties,
                    object,
                    root_value,
                    root_definitions,
                    path,
                )?;
                Ok(ComparableValue::Json(value.clone()))
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum ColumnSettingsHelper {
    Simple(ColumnType),
    Detailed {
        #[serde(rename = "type")]
        column_type: ColumnType,
        #[serde(flatten)]
        rules: FieldRules,
    },
}

impl Serialize for ColumnSettings {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if self.rules.is_default() {
            ColumnSettingsHelper::Simple(self.column_type.clone()).serialize(serializer)
        } else {
            ColumnSettingsHelper::Detailed {
                column_type: self.column_type.clone(),
                rules: self.rules.clone(),
            }
            .serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for ColumnSettings {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let helper = ColumnSettingsHelper::deserialize(deserializer)?;
        Ok(match helper {
            ColumnSettingsHelper::Simple(column_type) => ColumnSettings::from_type(column_type),
            ColumnSettingsHelper::Detailed { column_type, rules } => {
                ColumnSettings { column_type, rules }
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ComparableValue {
    Integer(i64),
    Float(f64),
    Decimal(Decimal),
    Boolean(bool),
    Text(String),
    Timestamp(DateTime<Utc>),
    Date(NaiveDate),
    Json(Value),
    Binary(Vec<u8>),
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn vec_is_empty(value: &Vec<String>) -> bool {
    value.is_empty()
}

fn map_is_empty(value: &BTreeMap<String, ColumnSettings>) -> bool {
    value.is_empty()
}

impl ComparableValue {
    fn as_text(&self) -> Option<&str> {
        match self {
            ComparableValue::Text(value) => Some(value.as_str()),
            _ => None,
        }
    }
}

use super::error::{EventError, Result};
use crate::store::payload_to_map;

pub const MAX_EVENT_NOTE_LENGTH: usize = 128;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EventSchema {
    pub fields: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl Default for EventSchema {
    fn default() -> Self {
        Self {
            fields: Vec::new(),
            notes: None,
        }
    }
}

impl EventSchema {
    fn ensure_sorted(&mut self) {
        self.fields.sort();
        self.fields.dedup();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AggregateSchema {
    pub aggregate: String,
    pub snapshot_threshold: Option<u64>,
    pub locked: bool,
    pub field_locks: Vec<String>,
    #[serde(default)]
    pub hidden: bool,
    #[serde(default)]
    pub hidden_fields: Vec<String>,
    #[serde(default)]
    pub column_types: BTreeMap<String, ColumnSettings>,
    pub events: BTreeMap<String, EventSchema>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl AggregateSchema {
    fn ensure_sorted(&mut self) {
        self.field_locks.sort();
        self.field_locks.dedup();
        self.hidden_fields.sort();
        self.hidden_fields.dedup();
        for schema in self.events.values_mut() {
            schema.ensure_sorted();
        }
    }
}

#[derive(Debug)]
pub struct SchemaManager {
    path: PathBuf,
    items: RwLock<BTreeMap<String, AggregateSchema>>,
}

#[derive(Debug)]
pub struct CreateSchemaInput {
    pub aggregate: String,
    pub events: Vec<String>,
    pub snapshot_threshold: Option<u64>,
}

#[derive(Debug, Default)]
pub struct SchemaUpdate {
    pub snapshot_threshold: Option<Option<u64>>,
    pub locked: Option<bool>,
    pub field_lock: Option<(String, bool)>,
    pub event_add_fields: BTreeMap<String, Vec<String>>,
    pub event_remove_fields: BTreeMap<String, Vec<String>>,
    pub event_set_fields: BTreeMap<String, Vec<String>>,
    pub event_notes: BTreeMap<String, Option<String>>,
    pub hidden: Option<bool>,
    pub hidden_field: Option<(String, bool)>,
    pub column_type: Option<(String, Option<ColumnType>)>,
    pub column_rules: Option<(String, Option<FieldRules>)>,
}

impl SchemaManager {
    pub fn load(path: PathBuf) -> Result<Self> {
        if !path.exists() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, "{}")?;
        }

        let contents = fs::read_to_string(&path)?;
        let map: BTreeMap<String, AggregateSchema> = if contents.trim().is_empty() {
            BTreeMap::new()
        } else {
            serde_json::from_str(&contents)?
        };

        Ok(Self {
            path,
            items: RwLock::new(map),
        })
    }

    pub fn create(&self, input: CreateSchemaInput) -> Result<AggregateSchema> {
        if input.aggregate.trim().is_empty() {
            return Err(EventError::InvalidSchema(
                "aggregate name must be provided".into(),
            ));
        }
        if input.events.is_empty() {
            return Err(EventError::InvalidSchema(
                "at least one event must be provided".into(),
            ));
        }

        let mut items = self.items.write();
        let aggregate_key = input.aggregate.clone();
        if items.contains_key(&aggregate_key) {
            return Err(EventError::SchemaExists);
        }

        let now = Utc::now();
        let mut events = BTreeMap::new();
        for event in input.events {
            if event.trim().is_empty() {
                return Err(EventError::InvalidSchema(
                    "event names cannot be empty".into(),
                ));
            }
            events.insert(event, EventSchema::default());
        }

        let mut schema = AggregateSchema {
            aggregate: aggregate_key.clone(),
            snapshot_threshold: input.snapshot_threshold,
            locked: false,
            field_locks: Vec::new(),
            hidden: false,
            hidden_fields: Vec::new(),
            column_types: BTreeMap::new(),
            events,
            created_at: now,
            updated_at: now,
        };
        schema.ensure_sorted();

        items.insert(aggregate_key.clone(), schema.clone());
        self.persist(&items)?;

        Ok(schema)
    }

    pub fn update(&self, aggregate: &str, update: SchemaUpdate) -> Result<AggregateSchema> {
        let mut items = self.items.write();
        {
            let schema = items.get_mut(aggregate).ok_or(EventError::SchemaNotFound)?;

            if let Some(snapshot) = update.snapshot_threshold {
                schema.snapshot_threshold = snapshot;
            }
            if let Some(locked) = update.locked {
                schema.locked = locked;
            }
            if let Some((field, lock)) = update.field_lock {
                if field.trim().is_empty() {
                    return Err(EventError::InvalidSchema(
                        "field name cannot be empty".into(),
                    ));
                }
                if lock {
                    if !schema.field_locks.contains(&field) {
                        schema.field_locks.push(field);
                    }
                } else {
                    schema.field_locks.retain(|item| item != &field);
                }
            }

            if let Some(hidden) = update.hidden {
                schema.hidden = hidden;
            }

            if let Some((field, hide)) = update.hidden_field {
                if field.trim().is_empty() {
                    return Err(EventError::InvalidSchema(
                        "field name cannot be empty".into(),
                    ));
                }
                if hide {
                    if !schema.hidden_fields.contains(&field) {
                        schema.hidden_fields.push(field);
                    }
                } else {
                    schema.hidden_fields.retain(|item| item != &field);
                }
            }

            if let Some((field, data_type)) = update.column_type {
                if field.trim().is_empty() {
                    return Err(EventError::InvalidSchema(
                        "field name cannot be empty".into(),
                    ));
                }
                match data_type {
                    Some(value) => {
                        schema
                            .column_types
                            .insert(field, ColumnSettings::from_type(value));
                    }
                    None => {
                        schema.column_types.remove(&field);
                    }
                }
            }

            if let Some((field, rules)) = update.column_rules {
                if field.trim().is_empty() {
                    return Err(EventError::InvalidSchema(
                        "field name cannot be empty".into(),
                    ));
                }
                if let Some(settings) = schema.column_types.get_mut(&field) {
                    match rules {
                        Some(mut new_rules) => {
                            // preserve existing nested schema unless explicitly provided
                            if new_rules.properties.is_empty() {
                                new_rules.properties =
                                    std::mem::take(&mut settings.rules.properties);
                            }
                            settings.rules = new_rules;
                        }
                        None => {
                            settings.rules = FieldRules::default();
                        }
                    }
                } else {
                    return Err(EventError::InvalidSchema(format!(
                        "cannot update rules for undefined field {}",
                        field
                    )));
                }
            }

            for (event, fields) in update.event_set_fields {
                let schema_event = schema.events.get_mut(&event).ok_or_else(|| {
                    EventError::InvalidSchema(format!(
                        "event {} is not defined for aggregate {}",
                        event, aggregate
                    ))
                })?;
                for field in &fields {
                    if field.trim().is_empty() {
                        return Err(EventError::InvalidSchema(
                            "field names cannot be empty".into(),
                        ));
                    }
                }
                schema_event.fields = fields;
            }

            for (event, fields) in update.event_add_fields {
                let schema_event = schema
                    .events
                    .entry(event.clone())
                    .or_insert_with(EventSchema::default);
                for field in fields {
                    if field.trim().is_empty() {
                        return Err(EventError::InvalidSchema(
                            "field names cannot be empty".into(),
                        ));
                    }
                    schema_event.fields.push(field);
                }
            }

            for (event, fields) in update.event_remove_fields {
                if let Some(schema_event) = schema.events.get_mut(&event) {
                    schema_event
                        .fields
                        .retain(|existing| !fields.contains(existing));
                }
            }

            for (event, note) in update.event_notes {
                let schema_event = schema.events.get_mut(&event).ok_or_else(|| {
                    EventError::InvalidSchema(format!(
                        "event {} is not defined for aggregate {}",
                        event, aggregate
                    ))
                })?;
                if let Some(ref value) = note {
                    if value.chars().count() > MAX_EVENT_NOTE_LENGTH {
                        return Err(EventError::InvalidSchema(format!(
                            "default note for event {} cannot exceed {} characters",
                            event, MAX_EVENT_NOTE_LENGTH
                        )));
                    }
                }
                schema_event.notes = note;
            }

            schema.ensure_sorted();
            schema.updated_at = Utc::now();
        }

        let result = items
            .get(aggregate)
            .cloned()
            .ok_or(EventError::SchemaNotFound)?;
        self.persist(&items)?;

        Ok(result)
    }

    pub fn list(&self) -> Vec<AggregateSchema> {
        self.items.read().values().cloned().collect()
    }

    pub fn get(&self, aggregate: &str) -> Result<AggregateSchema> {
        self.items
            .read()
            .get(aggregate)
            .cloned()
            .ok_or(EventError::SchemaNotFound)
    }

    pub fn should_snapshot(&self, aggregate: &str, version: u64) -> bool {
        if version == 0 {
            return false;
        }
        let items = self.items.read();
        items
            .get(aggregate)
            .and_then(|schema| schema.snapshot_threshold)
            .map(|threshold| threshold > 0 && version % threshold == 0)
            .unwrap_or(false)
    }

    pub fn snapshot(&self) -> BTreeMap<String, AggregateSchema> {
        self.items.read().clone()
    }

    pub fn replace_all(&self, mut items: BTreeMap<String, AggregateSchema>) -> Result<()> {
        for schema in items.values_mut() {
            schema.ensure_sorted();
        }

        {
            let mut guard = self.items.write();
            *guard = items.clone();
        }

        self.persist(&items)?;
        Ok(())
    }

    pub fn validate_event(&self, aggregate: &str, event_type: &str, payload: &Value) -> Result<()> {
        let items = self.items.read();
        let Some(schema) = items.get(aggregate) else {
            return Ok(());
        };

        if schema.locked {
            return Err(EventError::SchemaViolation(format!(
                "aggregate {} is locked for updates",
                aggregate
            )));
        }

        let payload_map = payload_to_map(payload);

        for key in payload_map.keys() {
            if schema.field_locks.contains(key) {
                return Err(EventError::SchemaViolation(format!(
                    "field {} is locked for aggregate {}",
                    key, aggregate
                )));
            }
        }

        let event_schema = schema.events.get(event_type).ok_or_else(|| {
            EventError::SchemaViolation(format!(
                "event {} is not defined for aggregate {}",
                event_type, aggregate
            ))
        })?;

        if !event_schema.fields.is_empty() {
            for required in &event_schema.fields {
                if !payload_map.contains_key(required) {
                    return Err(EventError::SchemaViolation(format!(
                        "missing required field {} for event {}",
                        required, event_type
                    )));
                }
            }

            for key in payload_map.keys() {
                if !event_schema.fields.contains(key) {
                    return Err(EventError::SchemaViolation(format!(
                        "field {} is not permitted for event {}",
                        key, event_type
                    )));
                }
            }
        }

        if !schema.column_types.is_empty() {
            let object = payload.as_object().ok_or_else(|| {
                EventError::SchemaViolation(format!(
                    "aggregate {} expects object payload for validation",
                    aggregate
                ))
            })?;

            validate_columns(
                &schema.column_types,
                object,
                payload,
                &schema.column_types,
                "",
            )?;
        }

        Ok(())
    }

    pub fn normalize_references<F>(
        &self,
        aggregate: &str,
        payload: Value,
        context: ReferenceContext<'_>,
        mut resolver: F,
    ) -> Result<(Value, Vec<ReferenceOutcome>)>
    where
        F: FnMut(&AggregateReference, ReferenceIntegrity) -> Result<ReferenceResolutionStatus>,
    {
        let schema = match self.items.read().get(aggregate) {
            Some(schema) => schema.clone(),
            None => return Ok((payload, Vec::new())),
        };

        let mut normalized = payload;
        let mut outcomes = Vec::new();
        if let Value::Object(ref mut map) = normalized {
            normalize_reference_columns(
                &schema.column_types,
                map,
                context,
                &mut resolver,
                &mut outcomes,
                "",
            )?;
        }

        Ok((normalized, outcomes))
    }

    pub fn collect_references(
        &self,
        aggregate: &str,
        state: &BTreeMap<String, String>,
        context: ReferenceContext<'_>,
    ) -> Result<Vec<LocatedReference>> {
        let items = self.items.read();
        let Some(schema) = items.get(aggregate) else {
            return Ok(Vec::new());
        };

        let mut references = Vec::new();
        collect_reference_columns(&schema.column_types, state, context, &mut references, "")?;
        Ok(references)
    }

    pub fn reference_rules_for_path(&self, aggregate: &str, path: &str) -> Option<ReferenceRules> {
        let items = self.items.read();
        let schema = items.get(aggregate)?;
        let segments: Vec<&str> = path.split('.').collect();
        reference_rules_for_segments(&schema.column_types, &segments)
    }

    pub fn remove_event(&self, aggregate: &str, event: &str) -> Result<AggregateSchema> {
        let mut items = self.items.write();
        let result = {
            let schema = items.get_mut(aggregate).ok_or(EventError::SchemaNotFound)?;

            if !schema.events.contains_key(event) {
                return Err(EventError::SchemaViolation(format!(
                    "event {} is not defined for aggregate {}",
                    event, aggregate
                )));
            }

            if schema.events.len() == 1 {
                return Err(EventError::SchemaViolation(format!(
                    "aggregate {} must define at least one event",
                    aggregate
                )));
            }

            schema.events.remove(event);
            schema.updated_at = Utc::now();
            schema.clone()
        };

        self.persist(&items)?;
        Ok(result)
    }

    fn persist(&self, items: &BTreeMap<String, AggregateSchema>) -> Result<()> {
        let payload = serde_json::to_string_pretty(items)?;
        fs::write(&self.path, payload)?;
        Ok(())
    }
}

fn validate_columns(
    definitions: &BTreeMap<String, ColumnSettings>,
    payload: &JsonMap<String, Value>,
    root_value: &Value,
    root_definitions: &BTreeMap<String, ColumnSettings>,
    prefix: &str,
) -> Result<()> {
    for (field, definition) in definitions {
        let path = join_path(prefix, field);
        let value_opt = payload.get(field);
        definition.validate_with_path(&path, value_opt, root_value, root_definitions)?;
    }
    Ok(())
}

fn normalize_reference_columns<F>(
    definitions: &BTreeMap<String, ColumnSettings>,
    payload: &mut JsonMap<String, Value>,
    context: ReferenceContext<'_>,
    resolver: &mut F,
    outcomes: &mut Vec<ReferenceOutcome>,
    prefix: &str,
) -> Result<()>
where
    F: FnMut(&AggregateReference, ReferenceIntegrity) -> Result<ReferenceResolutionStatus>,
{
    for (field, definition) in definitions {
        let path = join_path(prefix, field);
        let Some(value) = payload.get_mut(field) else {
            continue;
        };
        normalize_reference_value(value, definition, context, resolver, outcomes, &path)?;
    }
    Ok(())
}

fn collect_reference_columns(
    definitions: &BTreeMap<String, ColumnSettings>,
    state: &BTreeMap<String, String>,
    context: ReferenceContext<'_>,
    output: &mut Vec<LocatedReference>,
    prefix: &str,
) -> Result<()> {
    for (field, definition) in definitions {
        let path = join_path(prefix, field);
        let value = state.get(field);
        collect_reference_field(definition, value, context, output, &path)?;
    }
    Ok(())
}

fn collect_reference_field(
    settings: &ColumnSettings,
    value: Option<&String>,
    context: ReferenceContext<'_>,
    output: &mut Vec<LocatedReference>,
    path: &str,
) -> Result<()> {
    match settings.column_type {
        ColumnType::Object => {
            if settings.rules.properties.is_empty() {
                return Ok(());
            }
            let Some(raw) = value else {
                return Ok(());
            };
            let parsed: Value = serde_json::from_str(raw).map_err(|err| {
                EventError::SchemaViolation(format!(
                    "field {} must be valid JSON object: {}",
                    path, err
                ))
            })?;
            let Value::Object(map) = parsed else {
                return Ok(());
            };
            collect_reference_object(&settings.rules.properties, &map, context, output, path)
        }
        _ => {
            if settings.rules.format != Some(FieldFormat::Reference) {
                return Ok(());
            }
            let Some(raw) = value else {
                return Ok(());
            };
            if raw.trim().is_empty() {
                return Ok(());
            }
            let reference = AggregateReference::parse(raw, context)?;
            let integrity = settings
                .rules
                .reference
                .as_ref()
                .map(|rules| rules.integrity)
                .unwrap_or_default();
            output.push(LocatedReference {
                path: path.to_string(),
                reference,
                integrity,
            });
            Ok(())
        }
    }
}

fn collect_reference_object(
    definitions: &BTreeMap<String, ColumnSettings>,
    payload: &JsonMap<String, Value>,
    context: ReferenceContext<'_>,
    output: &mut Vec<LocatedReference>,
    prefix: &str,
) -> Result<()> {
    for (field, definition) in definitions {
        let path = join_path(prefix, field);
        let Some(value) = payload.get(field) else {
            continue;
        };
        match definition.column_type {
            ColumnType::Object => {
                if definition.rules.properties.is_empty() {
                    continue;
                }
                if let Value::Object(map) = value {
                    collect_reference_object(
                        &definition.rules.properties,
                        map,
                        context,
                        output,
                        &path,
                    )?;
                }
            }
            _ => {
                if definition.rules.format != Some(FieldFormat::Reference) {
                    continue;
                }
                let Some(text) = value.as_str() else {
                    continue;
                };
                if text.trim().is_empty() {
                    continue;
                }
                let reference = AggregateReference::parse(text, context)?;
                let integrity = definition
                    .rules
                    .reference
                    .as_ref()
                    .map(|rules| rules.integrity)
                    .unwrap_or_default();
                output.push(LocatedReference {
                    path: path.to_string(),
                    reference,
                    integrity,
                });
            }
        }
    }
    Ok(())
}

fn reference_rules_for_segments(
    definitions: &BTreeMap<String, ColumnSettings>,
    segments: &[&str],
) -> Option<ReferenceRules> {
    let (first, rest) = segments.split_first()?;
    let settings = definitions.get(*first)?;
    if rest.is_empty() {
        return settings.rules.reference.clone();
    }

    match settings.column_type {
        ColumnType::Object => reference_rules_for_segments(&settings.rules.properties, rest),
        _ => None,
    }
}

fn normalize_reference_value<F>(
    value: &mut Value,
    settings: &ColumnSettings,
    context: ReferenceContext<'_>,
    resolver: &mut F,
    outcomes: &mut Vec<ReferenceOutcome>,
    path: &str,
) -> Result<()>
where
    F: FnMut(&AggregateReference, ReferenceIntegrity) -> Result<ReferenceResolutionStatus>,
{
    match settings.column_type {
        ColumnType::Object => {
            let object = value.as_object_mut().ok_or_else(|| {
                EventError::SchemaViolation(format!("field {} must be a JSON object", path))
            })?;
            normalize_reference_columns(
                &settings.rules.properties,
                object,
                context,
                resolver,
                outcomes,
                path,
            )?;
        }
        _ => {
            if settings.rules.format != Some(FieldFormat::Reference) || value.is_null() {
                return Ok(());
            }
            let text = value.as_str().ok_or_else(|| {
                EventError::SchemaViolation(format!(
                    "field {} must be a string to store a reference",
                    path
                ))
            })?;
            let reference = AggregateReference::parse(text, context)?;
            let integrity = settings
                .rules
                .reference
                .as_ref()
                .map(|rules| rules.integrity)
                .unwrap_or_default();
            if let Some(expected_tenant) = settings
                .rules
                .reference
                .as_ref()
                .and_then(|rules| rules.tenant.as_ref())
            {
                if !reference.domain.eq_ignore_ascii_case(expected_tenant) {
                    return Err(EventError::SchemaViolation(format!(
                        "reference {} must target tenant/domain {}",
                        path, expected_tenant
                    )));
                }
            }
            if let Some(expected_aggregate) = settings
                .rules
                .reference
                .as_ref()
                .and_then(|rules| rules.aggregate_type.as_ref())
            {
                if reference.aggregate_type != *expected_aggregate {
                    return Err(EventError::SchemaViolation(format!(
                        "reference {} must target aggregate type {}",
                        path, expected_aggregate
                    )));
                }
            }
            let status = resolver(&reference, integrity)?;

            if integrity == ReferenceIntegrity::Strong {
                match status {
                    ReferenceResolutionStatus::Ok => {}
                    ReferenceResolutionStatus::NotFound => {
                        return Err(EventError::SchemaViolation(format!(
                            "reference {} points to missing aggregate {}",
                            path, reference
                        )));
                    }
                    ReferenceResolutionStatus::Forbidden => {
                        return Err(EventError::SchemaViolation(format!(
                            "reference {} points to an aggregate the caller cannot access ({})",
                            path, reference
                        )));
                    }
                    ReferenceResolutionStatus::Cycle | ReferenceResolutionStatus::DepthExceeded => {
                        return Err(EventError::SchemaViolation(format!(
                            "reference {} could not be validated (status: {:?})",
                            path, status
                        )));
                    }
                }
            }

            *value = Value::String(reference.to_canonical());
            outcomes.push(ReferenceOutcome {
                path: path.to_string(),
                reference,
                status,
            });
        }
    }
    Ok(())
}

fn join_path(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{}.{}", prefix, key)
    }
}

fn coerce_integer(value: &Value, path: &str) -> Result<i64> {
    match value {
        Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                Ok(int)
            } else if let Some(unsigned) = number.as_u64() {
                i64::try_from(unsigned).map_err(|_| {
                    EventError::SchemaViolation(format!(
                        "field {} exceeds signed integer range",
                        path
                    ))
                })
            } else {
                Err(EventError::SchemaViolation(format!(
                    "field {} must be an integer",
                    path
                )))
            }
        }
        Value::String(raw) => raw
            .trim()
            .parse::<i64>()
            .map_err(|_| EventError::SchemaViolation(format!("field {} must be an integer", path))),
        _ => Err(EventError::SchemaViolation(format!(
            "field {} must be an integer",
            path
        ))),
    }
}

fn coerce_float(value: &Value, path: &str) -> Result<f64> {
    match value {
        Value::Number(number) => number
            .as_f64()
            .ok_or_else(|| EventError::SchemaViolation(format!("field {} must be a float", path))),
        Value::String(raw) => raw
            .trim()
            .parse::<f64>()
            .map_err(|_| EventError::SchemaViolation(format!("field {} must be a float", path))),
        _ => Err(EventError::SchemaViolation(format!(
            "field {} must be a float",
            path
        ))),
    }
}

fn coerce_decimal(value: &Value, path: &str) -> Result<Decimal> {
    let raw = match value {
        Value::Number(number) => number.to_string(),
        Value::String(raw) => raw.trim().to_string(),
        _ => {
            return Err(EventError::SchemaViolation(format!(
                "field {} must be a decimal value",
                path
            )));
        }
    };

    Decimal::from_str(&raw)
        .map_err(|_| EventError::SchemaViolation(format!("field {} must be a decimal value", path)))
}

fn coerce_boolean(value: &Value, path: &str) -> Result<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Number(number) => match number.as_i64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(EventError::SchemaViolation(format!(
                "field {} must be a boolean",
                path
            ))),
        },
        Value::String(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            _ => Err(EventError::SchemaViolation(format!(
                "field {} must be a boolean",
                path
            ))),
        },
        _ => Err(EventError::SchemaViolation(format!(
            "field {} must be a boolean",
            path
        ))),
    }
}

fn coerce_text(value: &Value, path: &str) -> Result<String> {
    match value {
        Value::String(raw) => Ok(raw.clone()),
        _ => Err(EventError::SchemaViolation(format!(
            "field {} must be a string",
            path
        ))),
    }
}

fn coerce_timestamp(value: &Value, path: &str) -> Result<DateTime<Utc>> {
    let raw = match value {
        Value::String(raw) => raw,
        _ => {
            return Err(EventError::SchemaViolation(format!(
                "field {} must be an RFC3339 timestamp",
                path
            )));
        }
    };

    DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|_| {
            EventError::SchemaViolation(format!("field {} must be an RFC3339 timestamp", path))
        })
}

fn coerce_date(value: &Value, path: &str) -> Result<NaiveDate> {
    let raw = match value {
        Value::String(raw) => raw,
        _ => {
            return Err(EventError::SchemaViolation(format!(
                "field {} must be a date formatted as YYYY-MM-DD",
                path
            )));
        }
    };

    NaiveDate::parse_from_str(raw, "%Y-%m-%d").map_err(|_| {
        EventError::SchemaViolation(format!(
            "field {} must be a date formatted as YYYY-MM-DD",
            path
        ))
    })
}

fn coerce_binary(value: &Value, path: &str) -> Result<Vec<u8>> {
    let raw = match value {
        Value::String(raw) => raw,
        _ => {
            return Err(EventError::SchemaViolation(format!(
                "field {} must be base64 encoded",
                path
            )));
        }
    };

    BASE64_ENGINE
        .decode(raw.as_bytes())
        .map_err(|_| EventError::SchemaViolation(format!("field {} must be base64 encoded", path)))
}

fn check_decimal_constraints(
    decimal: &Decimal,
    precision: u8,
    scale: u8,
    path: &str,
) -> Result<()> {
    if decimal.scale() as u8 > scale {
        return Err(EventError::SchemaViolation(format!(
            "field {} must have at most {} decimal places",
            path, scale
        )));
    }

    let digits = decimal
        .abs()
        .to_string()
        .chars()
        .filter(|c| c.is_ascii_digit())
        .count();
    if digits > precision as usize {
        return Err(EventError::SchemaViolation(format!(
            "field {} must have precision <= {}",
            path, precision
        )));
    }
    Ok(())
}

fn check_integer_range(path: &str, value: i64, range: &RangeRule) -> Result<()> {
    if let Some(raw) = &range.min {
        let min = raw.trim().parse::<i64>().map_err(|_| {
            EventError::SchemaViolation(format!(
                "field {} range minimum '{}' is not a valid integer",
                path, raw
            ))
        })?;
        if value < min {
            return Err(EventError::SchemaViolation(format!(
                "field {} must be >= {}",
                path, min
            )));
        }
    }
    if let Some(raw) = &range.max {
        let max = raw.trim().parse::<i64>().map_err(|_| {
            EventError::SchemaViolation(format!(
                "field {} range maximum '{}' is not a valid integer",
                path, raw
            ))
        })?;
        if value > max {
            return Err(EventError::SchemaViolation(format!(
                "field {} must be <= {}",
                path, max
            )));
        }
    }
    Ok(())
}

fn check_float_range(path: &str, value: f64, range: &RangeRule) -> Result<()> {
    if let Some(raw) = &range.min {
        let min = raw.trim().parse::<f64>().map_err(|_| {
            EventError::SchemaViolation(format!(
                "field {} range minimum '{}' is not a valid float",
                path, raw
            ))
        })?;
        if value < min {
            return Err(EventError::SchemaViolation(format!(
                "field {} must be >= {}",
                path, min
            )));
        }
    }
    if let Some(raw) = &range.max {
        let max = raw.trim().parse::<f64>().map_err(|_| {
            EventError::SchemaViolation(format!(
                "field {} range maximum '{}' is not a valid float",
                path, raw
            ))
        })?;
        if value > max {
            return Err(EventError::SchemaViolation(format!(
                "field {} must be <= {}",
                path, max
            )));
        }
    }
    Ok(())
}

fn check_decimal_range(path: &str, value: &Decimal, range: &RangeRule) -> Result<()> {
    if let Some(raw) = &range.min {
        let min = Decimal::from_str(raw.trim()).map_err(|_| {
            EventError::SchemaViolation(format!(
                "field {} range minimum '{}' is not a valid decimal",
                path, raw
            ))
        })?;
        if value < &min {
            return Err(EventError::SchemaViolation(format!(
                "field {} must be >= {}",
                path, min
            )));
        }
    }
    if let Some(raw) = &range.max {
        let max = Decimal::from_str(raw.trim()).map_err(|_| {
            EventError::SchemaViolation(format!(
                "field {} range maximum '{}' is not a valid decimal",
                path, raw
            ))
        })?;
        if value > &max {
            return Err(EventError::SchemaViolation(format!(
                "field {} must be <= {}",
                path, max
            )));
        }
    }
    Ok(())
}

fn check_timestamp_range(path: &str, value: &DateTime<Utc>, range: &RangeRule) -> Result<()> {
    if let Some(raw) = &range.min {
        let min = parse_timestamp_literal(raw, path)?;
        if value < &min {
            return Err(EventError::SchemaViolation(format!(
                "field {} must be >= {}",
                path, raw
            )));
        }
    }
    if let Some(raw) = &range.max {
        let max = parse_timestamp_literal(raw, path)?;
        if value > &max {
            return Err(EventError::SchemaViolation(format!(
                "field {} must be <= {}",
                path, raw
            )));
        }
    }
    Ok(())
}

fn check_date_range(path: &str, value: &NaiveDate, range: &RangeRule) -> Result<()> {
    if let Some(raw) = &range.min {
        let min = parse_date_literal(raw, path)?;
        if value < &min {
            return Err(EventError::SchemaViolation(format!(
                "field {} must be on or after {}",
                path, raw
            )));
        }
    }
    if let Some(raw) = &range.max {
        let max = parse_date_literal(raw, path)?;
        if value > &max {
            return Err(EventError::SchemaViolation(format!(
                "field {} must be on or before {}",
                path, raw
            )));
        }
    }
    Ok(())
}

fn parse_timestamp_literal(raw: &str, path: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(raw.trim())
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|_| {
            EventError::SchemaViolation(format!(
                "field {} range boundary '{}' is not a valid RFC3339 timestamp",
                path, raw
            ))
        })
}

fn parse_date_literal(raw: &str, path: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(raw.trim(), "%Y-%m-%d").map_err(|_| {
        EventError::SchemaViolation(format!(
            "field {} range boundary '{}' is not a valid date (YYYY-MM-DD)",
            path, raw
        ))
    })
}

fn validate_reference_shape(path: &str, value: &str) -> Result<()> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(EventError::SchemaViolation(format!(
            "field {} must use reference notation domain#aggregate#id, aggregate#id, or #id",
            path
        )));
    }
    if !trimmed.contains('#') {
        return Err(EventError::SchemaViolation(format!(
            "field {} must use reference notation domain#aggregate#id, aggregate#id, or #id",
            path
        )));
    }
    let parts: Vec<_> = trimmed.split('#').collect();
    if !(parts.len() == 2 || parts.len() == 3) {
        return Err(EventError::SchemaViolation(format!(
            "field {} must use reference notation domain#aggregate#id, aggregate#id, or #id",
            path
        )));
    }
    if parts.last().map(|part| part.is_empty()).unwrap_or(true) {
        return Err(EventError::SchemaViolation(format!(
            "field {} must specify an aggregate id after '#'",
            path
        )));
    }
    Ok(())
}

fn validate_country_code(value: &str) -> bool {
    if value.len() != 2 {
        return false;
    }
    if !value.chars().all(|c| c.is_ascii_uppercase()) {
        return false;
    }
    ISO_COUNTRY_CODES.binary_search(&value).is_ok()
}

fn validate_wgs84(value: &str) -> bool {
    let mut parts = value.split(',');
    let lat_raw = match parts.next() {
        Some(part) => part.trim(),
        None => return false,
    };
    let lon_raw = match parts.next() {
        Some(part) => part.trim(),
        None => return false,
    };
    if parts.next().is_some() || lat_raw.is_empty() || lon_raw.is_empty() {
        return false;
    }

    let lat: f64 = match lat_raw.parse() {
        Ok(val) => val,
        Err(_) => return false,
    };
    let lon: f64 = match lon_raw.parse() {
        Ok(val) => val,
        Err(_) => return false,
    };

    lat.is_finite()
        && lon.is_finite()
        && (-90.0..=90.0).contains(&lat)
        && (-180.0..=180.0).contains(&lon)
}

fn validate_camel_case(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_lowercase() {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric())
}

fn validate_pascal_case(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_uppercase() {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric())
}

fn validate_snake_case(value: &str) -> bool {
    if value.is_empty() {
        return false;
    }
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_lowercase() {
        return false;
    }
    let mut prev_was_separator = false;
    for c in value.chars() {
        match c {
            'a'..='z' | '0'..='9' => prev_was_separator = false,
            '_' => {
                if prev_was_separator {
                    return false;
                }
                prev_was_separator = true;
                continue;
            }
            _ => return false,
        }
    }
    !prev_was_separator
}

fn validate_kebab_case(value: &str) -> bool {
    if value.is_empty() {
        return false;
    }
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_lowercase() {
        return false;
    }
    let mut prev_was_separator = false;
    for c in value.chars() {
        match c {
            'a'..='z' | '0'..='9' => prev_was_separator = false,
            '-' => {
                if prev_was_separator {
                    return false;
                }
                prev_was_separator = true;
                continue;
            }
            _ => return false,
        }
    }
    !prev_was_separator
}

fn validate_upper_case_snake_case(value: &str) -> bool {
    if value.is_empty() {
        return false;
    }
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_uppercase() {
        return false;
    }
    let mut prev_was_separator = false;
    for c in value.chars() {
        match c {
            'A'..='Z' | '0'..='9' => prev_was_separator = false,
            '_' => {
                if prev_was_separator {
                    return false;
                }
                prev_was_separator = true;
                continue;
            }
            _ => return false,
        }
    }
    !prev_was_separator
}

const ISO_COUNTRY_CODES: &[&str] = &[
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS", "AT", "AU", "AW", "AX", "AZ",
    "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BL", "BM", "BN", "BO", "BQ", "BR", "BS",
    "BT", "BV", "BW", "BY", "BZ", "CA", "CC", "CD", "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN",
    "CO", "CR", "CU", "CV", "CW", "CX", "CY", "CZ", "DE", "DJ", "DK", "DM", "DO", "DZ", "EC", "EE",
    "EG", "EH", "ER", "ES", "ET", "FI", "FJ", "FK", "FM", "FO", "FR", "GA", "GB", "GD", "GE", "GF",
    "GG", "GH", "GI", "GL", "GM", "GN", "GP", "GQ", "GR", "GS", "GT", "GU", "GW", "GY", "HK", "HM",
    "HN", "HR", "HT", "HU", "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR", "IS", "IT", "JE", "JM",
    "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN", "KP", "KR", "KW", "KY", "KZ", "LA", "LB", "LC",
    "LI", "LK", "LR", "LS", "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME", "MF", "MG", "MH", "MK",
    "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU", "MV", "MW", "MX", "MY", "MZ", "NA",
    "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP", "NR", "NU", "NZ", "OM", "PA", "PE", "PF", "PG",
    "PH", "PK", "PL", "PM", "PN", "PR", "PS", "PT", "PW", "PY", "QA", "RE", "RO", "RS", "RU", "RW",
    "SA", "SB", "SC", "SD", "SE", "SG", "SH", "SI", "SJ", "SK", "SL", "SM", "SN", "SO", "SR", "SS",
    "ST", "SV", "SX", "SY", "SZ", "TC", "TD", "TF", "TG", "TH", "TJ", "TK", "TL", "TM", "TN", "TO",
    "TR", "TT", "TV", "TW", "TZ", "UA", "UG", "UM", "US", "UY", "UZ", "VA", "VC", "VE", "VG", "VI",
    "VN", "VU", "WF", "WS", "YE", "YT", "ZA", "ZM", "ZW",
];
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::{collections::BTreeMap, str::FromStr};

    #[test]
    fn create_and_update_schema() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schemas.json");
        let manager = SchemaManager::load(path).unwrap();

        let schema = manager
            .create(CreateSchemaInput {
                aggregate: "patient".into(),
                events: vec!["patient-created".into(), "patient-updated".into()],
                snapshot_threshold: Some(10),
            })
            .unwrap();
        assert_eq!(schema.aggregate, "patient");
        assert_eq!(schema.events.len(), 2);

        let updated = manager
            .update(
                "patient",
                SchemaUpdate {
                    locked: Some(true),
                    field_lock: Some(("birthdate".into(), true)),
                    event_add_fields: {
                        let mut map = BTreeMap::new();
                        map.insert(
                            "patient-created".into(),
                            vec!["name".into(), "birthdate".into()],
                        );
                        map
                    },
                    ..SchemaUpdate::default()
                },
            )
            .unwrap();

        assert!(updated.locked);
        assert!(updated.field_locks.contains(&"birthdate".to_string()));
        assert!(
            updated
                .events
                .get("patient-created")
                .unwrap()
                .fields
                .contains(&"name".to_string())
        );
    }

    #[test]
    fn remove_event_from_aggregate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schemas.json");
        let manager = SchemaManager::load(path).unwrap();

        manager
            .create(CreateSchemaInput {
                aggregate: "person".into(),
                events: vec!["person_created".into(), "person_updated".into()],
                snapshot_threshold: None,
            })
            .unwrap();

        let updated = manager.remove_event("person", "person_updated").unwrap();
        assert!(updated.events.contains_key("person_created"));
        assert!(!updated.events.contains_key("person_updated"));

        let err = manager.remove_event("person", "missing").unwrap_err();
        assert!(matches!(err, EventError::SchemaViolation(_)));

        let err = manager
            .remove_event("person", "person_created")
            .unwrap_err();
        assert!(matches!(err, EventError::SchemaViolation(_)));
    }

    #[test]
    fn validates_required_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schemas.json");
        let manager = SchemaManager::load(path).unwrap();

        manager
            .create(CreateSchemaInput {
                aggregate: "person".into(),
                events: vec!["person_created".into()],
                snapshot_threshold: None,
            })
            .unwrap();

        let mut update = SchemaUpdate::default();
        update
            .event_add_fields
            .insert("person_created".into(), vec!["first_name".into()]);
        manager.update("person", update).unwrap();

        let payload = json!({ "first_name": "Alice" });
        manager
            .validate_event("person", "person_created", &payload)
            .unwrap();

        let payload = json!({});
        let err = manager
            .validate_event("person", "person_created", &payload)
            .unwrap_err();
        assert!(matches!(err, EventError::SchemaViolation(_)));
    }

    #[test]
    fn reference_format_allows_shorthand_shapes() {
        let mut rules = FieldRules::default();
        rules.format = Some(FieldFormat::Reference);

        for value in ["#123", "domain#agg#id-1", "agg#123"] {
            rules
                .validate_rules(
                    "ref",
                    &ColumnType::Text,
                    &ComparableValue::Text(value.to_string()),
                )
                .expect("reference shape should be accepted");
        }

        for value in ["missing-delimiter", "too#many#parts#here", "agg#"] {
            let err = rules
                .validate_rules(
                    "ref",
                    &ColumnType::Text,
                    &ComparableValue::Text(value.to_string()),
                )
                .unwrap_err();
            assert!(matches!(err, EventError::SchemaViolation(_)), "{value}");
        }
    }

    #[test]
    fn normalize_references_canonicalizes_values() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schemas.json");
        let manager = SchemaManager::load(path).unwrap();

        manager
            .create(CreateSchemaInput {
                aggregate: "farm".into(),
                events: vec!["created".into()],
                snapshot_threshold: None,
            })
            .unwrap();

        let mut type_update = SchemaUpdate::default();
        type_update.column_type = Some(("address".into(), Some(ColumnType::Text)));
        manager.update("farm", type_update).unwrap();

        let mut rules = FieldRules::default();
        rules.format = Some(FieldFormat::Reference);
        rules.reference = Some(ReferenceRules::default());

        let mut rules_update = SchemaUpdate::default();
        rules_update.column_rules = Some(("address".into(), Some(rules)));
        manager.update("farm", rules_update).unwrap();

        let payload = json!({ "address": "#123" });
        let (normalized, outcomes) = manager
            .normalize_references(
                "farm",
                payload,
                ReferenceContext {
                    domain: "farm",
                    aggregate_type: "farm",
                },
                |reference, _| {
                    assert_eq!(reference.to_string(), "farm#farm#123");
                    Ok(ReferenceResolutionStatus::Ok)
                },
            )
            .expect("normalization should succeed");

        let address = normalized
            .as_object()
            .and_then(|map| map.get("address"))
            .and_then(Value::as_str);
        assert_eq!(address, Some("farm#farm#123"));
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].status, ReferenceResolutionStatus::Ok);
    }

    #[test]
    fn normalize_references_enforces_strong_integrity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schemas.json");
        let manager = SchemaManager::load(path).unwrap();

        manager
            .create(CreateSchemaInput {
                aggregate: "farm".into(),
                events: vec!["created".into()],
                snapshot_threshold: None,
            })
            .unwrap();

        let mut type_update = SchemaUpdate::default();
        type_update.column_type = Some(("address".into(), Some(ColumnType::Text)));
        manager.update("farm", type_update).unwrap();

        let mut rules = FieldRules::default();
        rules.format = Some(FieldFormat::Reference);
        rules.reference = Some(ReferenceRules::default());

        let mut rules_update = SchemaUpdate::default();
        rules_update.column_rules = Some(("address".into(), Some(rules)));
        manager.update("farm", rules_update).unwrap();

        let payload = json!({ "address": "address#missing" });
        let err = manager
            .normalize_references(
                "farm",
                payload,
                ReferenceContext {
                    domain: "farm",
                    aggregate_type: "farm",
                },
                |_reference, _| Ok(ReferenceResolutionStatus::NotFound),
            )
            .unwrap_err();

        assert!(matches!(err, EventError::SchemaViolation(_)));
    }

    #[test]
    fn normalize_references_allows_weak_integrity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schemas.json");
        let manager = SchemaManager::load(path).unwrap();

        manager
            .create(CreateSchemaInput {
                aggregate: "farm".into(),
                events: vec!["created".into()],
                snapshot_threshold: None,
            })
            .unwrap();

        let mut type_update = SchemaUpdate::default();
        type_update.column_type = Some(("address".into(), Some(ColumnType::Text)));
        manager.update("farm", type_update).unwrap();

        let mut rules = FieldRules::default();
        rules.format = Some(FieldFormat::Reference);
        rules.reference = Some(ReferenceRules {
            integrity: ReferenceIntegrity::Weak,
            ..ReferenceRules::default()
        });

        let mut rules_update = SchemaUpdate::default();
        rules_update.column_rules = Some(("address".into(), Some(rules)));
        manager.update("farm", rules_update).unwrap();

        let payload = json!({ "address": "address#missing" });
        let (normalized, outcomes) = manager
            .normalize_references(
                "farm",
                payload,
                ReferenceContext {
                    domain: "farm",
                    aggregate_type: "farm",
                },
                |_reference, _| Ok(ReferenceResolutionStatus::NotFound),
            )
            .expect("weak reference should not fail validation");

        let address = normalized
            .as_object()
            .and_then(|map| map.get("address"))
            .and_then(Value::as_str);
        assert_eq!(address, Some("farm#address#missing"));
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].status, ReferenceResolutionStatus::NotFound);
    }

    #[test]
    fn normalize_references_enforces_tenant_and_aggregate_rules() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schemas.json");
        let manager = SchemaManager::load(path).unwrap();

        manager
            .create(CreateSchemaInput {
                aggregate: "farm".into(),
                events: vec!["created".into()],
                snapshot_threshold: None,
            })
            .unwrap();

        let mut type_update = SchemaUpdate::default();
        type_update.column_type = Some(("address".into(), Some(ColumnType::Text)));
        manager.update("farm", type_update).unwrap();

        let mut rules = FieldRules::default();
        rules.format = Some(FieldFormat::Reference);
        rules.reference = Some(ReferenceRules {
            tenant: Some("geo".into()),
            aggregate_type: Some("address".into()),
            ..ReferenceRules::default()
        });

        let mut rules_update = SchemaUpdate::default();
        rules_update.column_rules = Some(("address".into(), Some(rules)));
        manager.update("farm", rules_update).unwrap();

        let payload = json!({ "address": "geo#address#123" });
        manager
            .normalize_references(
                "farm",
                payload,
                ReferenceContext {
                    domain: "farm",
                    aggregate_type: "farm",
                },
                |_reference, _| Ok(ReferenceResolutionStatus::Ok),
            )
            .expect("matching tenant/aggregate should pass");

        let payload = json!({ "address": "farm#address#123" });
        let err = manager
            .normalize_references(
                "farm",
                payload,
                ReferenceContext {
                    domain: "farm",
                    aggregate_type: "farm",
                },
                |_reference, _| Ok(ReferenceResolutionStatus::Ok),
            )
            .unwrap_err();
        assert!(matches!(err, EventError::SchemaViolation(_)));
    }

    #[test]
    fn snapshot_threshold_triggers_on_expected_versions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schemas.json");
        let manager = SchemaManager::load(path).unwrap();

        manager
            .create(CreateSchemaInput {
                aggregate: "order".into(),
                events: vec!["order-created".into()],
                snapshot_threshold: Some(3),
            })
            .unwrap();

        assert!(!manager.should_snapshot("order", 1));
        assert!(manager.should_snapshot("order", 3));
        assert!(!manager.should_snapshot("order", 4));

        // Unknown aggregate returns false.
        assert!(!manager.should_snapshot("missing", 3));
    }

    #[test]
    fn column_type_from_str_parses_decimal() {
        let ty = ColumnType::from_str("decimal(12, 4)").unwrap();
        assert_eq!(
            ty,
            ColumnType::Decimal {
                precision: 12,
                scale: 4
            }
        );
    }

    #[test]
    fn column_type_round_trip_serialization() {
        let mut map: BTreeMap<String, ColumnSettings> = BTreeMap::new();
        map.insert(
            "amount".to_string(),
            ColumnSettings::from_type(ColumnType::Decimal {
                precision: 8,
                scale: 2,
            }),
        );
        map.insert(
            "flag".to_string(),
            ColumnSettings::from_type(ColumnType::Boolean),
        );

        let json = serde_json::to_string(&map).unwrap();
        assert_eq!(json, r#"{"amount":"decimal(8,2)","flag":"boolean"}"#);

        let decoded: BTreeMap<String, ColumnSettings> = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, map);
    }

    #[test]
    fn update_schema_column_type() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schemas.json");
        let manager = SchemaManager::load(path).unwrap();

        manager
            .create(CreateSchemaInput {
                aggregate: "order".into(),
                events: vec!["order_created".into()],
                snapshot_threshold: None,
            })
            .unwrap();

        let mut update = SchemaUpdate::default();
        update.column_type = Some((
            "total".to_string(),
            Some(ColumnType::Decimal {
                precision: 12,
                scale: 2,
            }),
        ));

        let schema = manager.update("order", update).unwrap();
        let column_type = schema
            .column_types
            .get("total")
            .expect("column type should be recorded");
        assert_eq!(
            column_type.column_type,
            ColumnType::Decimal {
                precision: 12,
                scale: 2
            }
        );

        let mut removal = SchemaUpdate::default();
        removal.column_type = Some(("total".to_string(), None));
        let schema = manager.update("order", removal).unwrap();
        assert!(!schema.column_types.contains_key("total"));
    }

    #[test]
    fn enforces_contains_and_excludes_rules() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schemas.json");
        let manager = SchemaManager::load(path).unwrap();

        manager
            .create(CreateSchemaInput {
                aggregate: "article".into(),
                events: vec!["article_created".into()],
                snapshot_threshold: None,
            })
            .unwrap();

        let mut update = SchemaUpdate::default();
        update.column_type = Some(("description".into(), Some(ColumnType::Text)));
        manager.update("article", update).unwrap();

        let mut rules = FieldRules::default();
        rules.required = true;
        rules.contains.push("foo".into());
        rules.does_not_contain.push("bar".into());

        let mut rules_update = SchemaUpdate::default();
        rules_update.column_rules = Some(("description".into(), Some(rules)));
        manager.update("article", rules_update).unwrap();

        manager
            .validate_event(
                "article",
                "article_created",
                &json!({ "description": "foo baz" }),
            )
            .unwrap();

        let err = manager
            .validate_event(
                "article",
                "article_created",
                &json!({ "description": "baz bar" }),
            )
            .unwrap_err();
        assert!(matches!(err, EventError::SchemaViolation(_)));
    }

    #[test]
    fn enforces_range_rules() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schemas.json");
        let manager = SchemaManager::load(path).unwrap();

        manager
            .create(CreateSchemaInput {
                aggregate: "metrics".into(),
                events: vec!["metrics_recorded".into()],
                snapshot_threshold: None,
            })
            .unwrap();

        let mut update = SchemaUpdate::default();
        update.column_type = Some(("count".into(), Some(ColumnType::Integer)));
        manager.update("metrics", update).unwrap();

        let mut rules = FieldRules::default();
        rules.range = Some(RangeRule {
            min: Some("1".into()),
            max: Some("10".into()),
        });

        let mut rules_update = SchemaUpdate::default();
        rules_update.column_rules = Some(("count".into(), Some(rules)));
        manager.update("metrics", rules_update).unwrap();

        manager
            .validate_event("metrics", "metrics_recorded", &json!({ "count": 5 }))
            .unwrap();

        let err = manager
            .validate_event("metrics", "metrics_recorded", &json!({ "count": 0 }))
            .unwrap_err();
        assert!(matches!(err, EventError::SchemaViolation(_)));
    }

    #[test]
    fn enforces_format_rules() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schemas.json");
        let manager = SchemaManager::load(path).unwrap();

        manager
            .create(CreateSchemaInput {
                aggregate: "contact".into(),
                events: vec!["contact_added".into()],
                snapshot_threshold: None,
            })
            .unwrap();

        let mut update = SchemaUpdate::default();
        update.column_type = Some(("email".into(), Some(ColumnType::Text)));
        manager.update("contact", update).unwrap();

        let mut rules = FieldRules::default();
        rules.format = Some(FieldFormat::Email);
        rules.required = true;

        let mut rules_update = SchemaUpdate::default();
        rules_update.column_rules = Some(("email".into(), Some(rules)));
        manager.update("contact", rules_update).unwrap();

        manager
            .validate_event(
                "contact",
                "contact_added",
                &json!({ "email": "user@example.com" }),
            )
            .unwrap();

        let err = manager
            .validate_event(
                "contact",
                "contact_added",
                &json!({ "email": "not-an-email" }),
            )
            .unwrap_err();
        assert!(matches!(err, EventError::SchemaViolation(_)));
    }

    #[test]
    fn validates_nested_object_properties() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schemas.json");
        let manager = SchemaManager::load(path).unwrap();

        manager
            .create(CreateSchemaInput {
                aggregate: "customer".into(),
                events: vec!["customer_created".into()],
                snapshot_threshold: None,
            })
            .unwrap();

        let mut update = SchemaUpdate::default();
        update.column_type = Some(("address".into(), Some(ColumnType::Object)));
        manager.update("customer", update).unwrap();

        let mut child_rules = FieldRules::default();
        child_rules.required = true;

        let mut rules = FieldRules::default();
        rules.properties.insert(
            "city".into(),
            ColumnSettings {
                column_type: ColumnType::Text,
                rules: child_rules,
            },
        );

        let mut rules_update = SchemaUpdate::default();
        rules_update.column_rules = Some(("address".into(), Some(rules)));
        manager.update("customer", rules_update).unwrap();

        manager
            .validate_event(
                "customer",
                "customer_created",
                &json!({ "address": { "city": "Paris" } }),
            )
            .unwrap();

        let err = manager
            .validate_event("customer", "customer_created", &json!({ "address": {} }))
            .unwrap_err();
        assert!(matches!(err, EventError::SchemaViolation(_)));
    }

    #[test]
    fn date_columns_accept_calendar_strings() {
        let accepted = coerce_date(&json!("2024-01-01"), "birthday").unwrap();
        assert_eq!(accepted.to_string(), "2024-01-01");

        let err_timestamp = coerce_date(&json!("2024-01-01T00:00:00Z"), "birthday").unwrap_err();
        assert!(matches!(err_timestamp, EventError::SchemaViolation(_)));

        let err_malformed = coerce_date(&json!("20240101"), "birthday").unwrap_err();
        assert!(matches!(err_malformed, EventError::SchemaViolation(_)));
    }

    #[test]
    fn country_code_format_requires_known_iso_entries() {
        let mut rules = FieldRules::default();
        rules.format = Some(FieldFormat::CountryCode);

        rules
            .validate_rules(
                "country",
                &ColumnType::Text,
                &ComparableValue::Text("US".to_string()),
            )
            .expect("US should be accepted");

        let err = rules
            .validate_rules(
                "country",
                &ColumnType::Text,
                &ComparableValue::Text("XX".to_string()),
            )
            .unwrap_err();
        assert!(matches!(err, EventError::SchemaViolation(_)));

        let err_lowercase = rules
            .validate_rules(
                "country",
                &ColumnType::Text,
                &ComparableValue::Text("us".to_string()),
            )
            .unwrap_err();
        assert!(matches!(err_lowercase, EventError::SchemaViolation(_)));
    }

    #[test]
    fn iso_8601_format_requires_valid_timestamps() {
        let mut rules = FieldRules::default();
        rules.format = Some(FieldFormat::Iso8601);

        rules
            .validate_rules(
                "timestamp",
                &ColumnType::Text,
                &ComparableValue::Text("2024-05-02T13:45:00Z".to_string()),
            )
            .expect("valid ISO 8601 should pass");

        let err = rules
            .validate_rules(
                "timestamp",
                &ColumnType::Text,
                &ComparableValue::Text("02/05/2024".to_string()),
            )
            .unwrap_err();
        assert!(matches!(err, EventError::SchemaViolation(_)));
    }

    #[test]
    fn wgs84_format_requires_lat_lon_pair() {
        let mut rules = FieldRules::default();
        rules.format = Some(FieldFormat::Wgs84);

        rules
            .validate_rules(
                "location",
                &ColumnType::Text,
                &ComparableValue::Text("37.7749,-122.4194".to_string()),
            )
            .expect("valid coordinates should pass");

        for invalid in [
            "91.0,0.0",
            "10.0,181.0",
            "10.0",
            "lat,lon",
            "37.0,-122.0,5.0",
            "",
        ] {
            let err = rules
                .validate_rules(
                    "location",
                    &ColumnType::Text,
                    &ComparableValue::Text(invalid.to_string()),
                )
                .unwrap_err();
            assert!(matches!(err, EventError::SchemaViolation(_)), "{}", invalid);
        }
    }

    #[test]
    fn camel_case_format_enforces_pattern() {
        let mut rules = FieldRules::default();
        rules.format = Some(FieldFormat::CamelCase);

        rules
            .validate_rules(
                "label",
                &ColumnType::Text,
                &ComparableValue::Text("camelCaseValue1".to_string()),
            )
            .expect("valid camelCase should pass");

        for invalid in [
            "CamelCase",
            "snake_case",
            "kebab-case",
            "camel_case",
            "camel-case",
        ] {
            let err = rules
                .validate_rules(
                    "label",
                    &ColumnType::Text,
                    &ComparableValue::Text(invalid.to_string()),
                )
                .unwrap_err();
            assert!(matches!(err, EventError::SchemaViolation(_)), "{}", invalid);
        }
    }

    #[test]
    fn snake_case_format_enforces_pattern() {
        let mut rules = FieldRules::default();
        rules.format = Some(FieldFormat::SnakeCase);

        rules
            .validate_rules(
                "label",
                &ColumnType::Text,
                &ComparableValue::Text("snake_case_value1".to_string()),
            )
            .expect("valid snake_case should pass");

        for invalid in [
            "Snake_Case",
            "_leading",
            "trailing_",
            "double__underscore",
            "snake-case",
        ] {
            let err = rules
                .validate_rules(
                    "label",
                    &ColumnType::Text,
                    &ComparableValue::Text(invalid.to_string()),
                )
                .unwrap_err();
            assert!(matches!(err, EventError::SchemaViolation(_)), "{}", invalid);
        }
    }

    #[test]
    fn kebab_case_format_enforces_pattern() {
        let mut rules = FieldRules::default();
        rules.format = Some(FieldFormat::KebabCase);

        rules
            .validate_rules(
                "label",
                &ColumnType::Text,
                &ComparableValue::Text("kebab-case-value1".to_string()),
            )
            .expect("valid kebab-case should pass");

        for invalid in [
            "Kebab-Case",
            "-leading",
            "trailing-",
            "double--dash",
            "kebab_case",
        ] {
            let err = rules
                .validate_rules(
                    "label",
                    &ColumnType::Text,
                    &ComparableValue::Text(invalid.to_string()),
                )
                .unwrap_err();
            assert!(matches!(err, EventError::SchemaViolation(_)), "{}", invalid);
        }
    }

    #[test]
    fn pascal_case_format_enforces_pattern() {
        let mut rules = FieldRules::default();
        rules.format = Some(FieldFormat::PascalCase);

        rules
            .validate_rules(
                "label",
                &ColumnType::Text,
                &ComparableValue::Text("PascalCaseValue1".to_string()),
            )
            .expect("valid PascalCase should pass");

        for invalid in [
            "pascalCase",
            "Pascal_Case",
            "Pascal-Case",
            "PascalCase!",
            "",
        ] {
            let err = rules
                .validate_rules(
                    "label",
                    &ColumnType::Text,
                    &ComparableValue::Text(invalid.to_string()),
                )
                .unwrap_err();
            assert!(matches!(err, EventError::SchemaViolation(_)), "{}", invalid);
        }
    }

    #[test]
    fn upper_case_snake_case_format_enforces_pattern() {
        let mut rules = FieldRules::default();
        rules.format = Some(FieldFormat::UpperCaseSnakeCase);

        rules
            .validate_rules(
                "label",
                &ColumnType::Text,
                &ComparableValue::Text("CONST_VALUE_1".to_string()),
            )
            .expect("valid constant should pass");

        for invalid in [
            "Const_Value",
            "_LEADING",
            "TRAILING_",
            "DOUBLE__UNDERSCORE",
            "CONST-VALUE",
        ] {
            let err = rules
                .validate_rules(
                    "label",
                    &ColumnType::Text,
                    &ComparableValue::Text(invalid.to_string()),
                )
                .unwrap_err();
            assert!(matches!(err, EventError::SchemaViolation(_)), "{}", invalid);
        }
    }
}
