use std::{collections::BTreeMap, fmt, num::NonZeroUsize};

use serde_json::Value;
use thiserror::Error;

use lru::LruCache;
use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::store::{AggregateState, EventMetadata, EventRecord, select_state_field};

#[derive(Debug, Clone)]
pub enum FilterExpr {
    And(Vec<FilterExpr>),
    Or(Vec<FilterExpr>),
    Not(Box<FilterExpr>),
    Comparison { field: String, op: ComparisonOp },
}

#[derive(Debug, Clone)]
pub enum ComparisonOp {
    Equals(FilterValue),
    NotEquals(FilterValue),
    GreaterThan(FilterValue),
    LessThan(FilterValue),
    In(Vec<FilterValue>),
    Like(FilterValue),
}

#[derive(Debug, Clone)]
pub enum FilterValue {
    String(String),
    Number(f64),
    Bool(bool),
    Null,
}

#[derive(Debug, Error)]
pub enum FilterParseError {
    #[error("{0}")]
    Message(String),
}

impl FilterExpr {
    pub fn matches_aggregate(&self, aggregate: &AggregateState) -> bool {
        match self {
            FilterExpr::And(children) => children
                .iter()
                .all(|child| child.matches_aggregate(aggregate)),
            FilterExpr::Or(children) => children
                .iter()
                .any(|child| child.matches_aggregate(aggregate)),
            FilterExpr::Not(expr) => !expr.matches_aggregate(aggregate),
            FilterExpr::Comparison { field, op } => {
                let value = resolve_field_value(aggregate, field);
                evaluate_comparison(value, op)
            }
        }
    }

    /// Returns true when evaluating this filter requires reading aggregate state data
    /// beyond the metadata stored alongside the aggregate (type/id/version/timestamps/etc).
    pub fn requires_aggregate_state(&self) -> bool {
        match self {
            FilterExpr::And(children) | FilterExpr::Or(children) => children
                .iter()
                .any(|child| child.requires_aggregate_state()),
            FilterExpr::Not(expr) => expr.requires_aggregate_state(),
            FilterExpr::Comparison { field, .. } => !is_aggregate_meta_field(field),
        }
    }

    pub fn matches_event(&self, event: &EventRecord) -> bool {
        match self {
            FilterExpr::And(children) => children.iter().all(|child| child.matches_event(event)),
            FilterExpr::Or(children) => children.iter().any(|child| child.matches_event(event)),
            FilterExpr::Not(expr) => !expr.matches_event(event),
            FilterExpr::Comparison { field, op } => {
                let value = resolve_event_value(event, field);
                evaluate_comparison(value, op)
            }
        }
    }

    pub fn references_field(&self, target: &str) -> bool {
        match self {
            FilterExpr::And(children) | FilterExpr::Or(children) => {
                children.iter().any(|child| child.references_field(target))
            }
            FilterExpr::Not(expr) => expr.references_field(target),
            FilterExpr::Comparison { field, .. } => field.eq_ignore_ascii_case(target),
        }
    }
}

impl FilterValue {
    fn matches_value(&self, value: &ComparableValue) -> bool {
        match (self, value) {
            (FilterValue::Null, ComparableValue::Null) => true,
            (FilterValue::Bool(lhs), ComparableValue::Bool(rhs)) => lhs == rhs,
            (FilterValue::Number(lhs), ComparableValue::Number(rhs)) => {
                (lhs - rhs).abs() < f64::EPSILON
            }
            (FilterValue::String(lhs), ComparableValue::String(rhs)) => lhs == rhs,
            _ => false,
        }
    }

    pub fn parse_literal(input: &str) -> Self {
        let trimmed = input.trim();
        if trimmed.eq_ignore_ascii_case("null") {
            FilterValue::Null
        } else if trimmed.eq_ignore_ascii_case("true") {
            FilterValue::Bool(true)
        } else if trimmed.eq_ignore_ascii_case("false") {
            FilterValue::Bool(false)
        } else if let Ok(number) = trimmed.parse::<f64>() {
            FilterValue::Number(number)
        } else {
            FilterValue::String(trimmed.to_string())
        }
    }
}

#[derive(Debug)]
enum ComparableValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Unsupported,
}

fn resolve_field_value(aggregate: &AggregateState, field: &str) -> Option<ComparableValue> {
    match field {
        "aggregate_type" | "aggregateType" => {
            Some(ComparableValue::String(aggregate.aggregate_type.clone()))
        }
        "aggregate_id" | "aggregateId" => {
            Some(ComparableValue::String(aggregate.aggregate_id.clone()))
        }
        "merkle_root" | "merkleRoot" => {
            Some(ComparableValue::String(aggregate.merkle_root.clone()))
        }
        "created_at" | "createdAt" => Some(match aggregate.created_at {
            Some(dt) => ComparableValue::Number(dt.timestamp_millis() as f64),
            None => ComparableValue::Null,
        }),
        "updated_at" | "updatedAt" => Some(match aggregate.updated_at {
            Some(dt) => ComparableValue::Number(dt.timestamp_millis() as f64),
            None => ComparableValue::Null,
        }),
        "archived" => Some(ComparableValue::Bool(aggregate.archived)),
        "version" => Some(ComparableValue::Number(aggregate.version as f64)),
        other if other.starts_with("extensions.") => extension_field_value(
            &aggregate.extensions,
            other.trim_start_matches("extensions."),
        ),
        other => select_state_field(&aggregate.state, other)
            .map(|value| json_value_to_comparable(&value))
            .or_else(|| extension_field_value(&aggregate.extensions, other)),
    }
}

fn is_aggregate_meta_field(field: &str) -> bool {
    matches!(
        field,
        "aggregate_type"
            | "aggregateType"
            | "aggregate_id"
            | "aggregateId"
            | "merkle_root"
            | "merkleRoot"
            | "created_at"
            | "createdAt"
            | "updated_at"
            | "updatedAt"
            | "archived"
            | "version"
    )
}

fn resolve_event_value(event: &EventRecord, field: &str) -> Option<ComparableValue> {
    match field {
        "aggregate_type" | "aggregateType" => {
            Some(ComparableValue::String(event.aggregate_type.clone()))
        }
        "aggregate_id" | "aggregateId" => Some(ComparableValue::String(event.aggregate_id.clone())),
        "event_type" | "eventType" => Some(ComparableValue::String(event.event_type.clone())),
        "version" => Some(ComparableValue::Number(event.version as f64)),
        "event_id" | "eventId" => {
            Some(ComparableValue::String(event.metadata.event_id.to_string()))
        }
        "created_at" | "createdAt" => Some(ComparableValue::Number(
            event.metadata.created_at.timestamp_millis() as f64,
        )),
        "merkle_root" | "merkleRoot" => Some(ComparableValue::String(event.merkle_root.clone())),
        "hash" => Some(ComparableValue::String(event.hash.clone())),
        "note" => event
            .metadata
            .note
            .as_ref()
            .map(|note| ComparableValue::String(note.clone())),
        other if other.starts_with("metadata.") => {
            let path = other.trim_start_matches("metadata.");
            metadata_field_value(&event.metadata, path)
        }
        other if other.starts_with("extensions.") => {
            let path = other.trim_start_matches("extensions.");
            event
                .extensions
                .as_ref()
                .and_then(|value| select_json_comparable(value, path))
        }
        other if other.starts_with("payload.") => {
            let path = other.trim_start_matches("payload.");
            select_json_comparable(&event.payload, path)
        }
        other => metadata_field_value(&event.metadata, other)
            .or_else(|| {
                event
                    .extensions
                    .as_ref()
                    .and_then(|value| select_json_comparable(value, other))
            })
            .or_else(|| select_json_comparable(&event.payload, other)),
    }
}

fn metadata_field_value(metadata: &EventMetadata, field: &str) -> Option<ComparableValue> {
    match field {
        "note" => metadata
            .note
            .as_ref()
            .map(|note| ComparableValue::String(note.clone())),
        "event_id" | "eventId" => Some(ComparableValue::String(metadata.event_id.to_string())),
        "created_at" | "createdAt" => Some(ComparableValue::Number(
            metadata.created_at.timestamp_millis() as f64,
        )),
        other if other.starts_with("issued_by.") => {
            let claims = metadata.issued_by.as_ref()?;
            match other.trim_start_matches("issued_by.") {
                "group" | "Group" => Some(ComparableValue::String(claims.group.clone())),
                "user" | "User" => Some(ComparableValue::String(claims.user.clone())),
                _ => None,
            }
        }
        other if other.starts_with("issuedBy.") => {
            let claims = metadata.issued_by.as_ref()?;
            match other.trim_start_matches("issuedBy.") {
                "group" | "Group" => Some(ComparableValue::String(claims.group.clone())),
                "user" | "User" => Some(ComparableValue::String(claims.user.clone())),
                _ => None,
            }
        }
        _ => None,
    }
}

fn extension_field_value(
    extensions: &BTreeMap<String, Value>,
    path: &str,
) -> Option<ComparableValue> {
    if extensions.is_empty() {
        return None;
    }
    let mut map = serde_json::Map::new();
    for (key, value) in extensions {
        map.insert(key.clone(), value.clone());
    }
    select_json_comparable(&Value::Object(map), path)
}

fn select_json_comparable(value: &Value, path: &str) -> Option<ComparableValue> {
    if path.is_empty() {
        return Some(json_value_to_comparable(value));
    }
    let target = select_json_value(value, path)?;
    Some(json_value_to_comparable(&target))
}

fn select_json_value(value: &Value, path: &str) -> Option<Value> {
    if path.is_empty() {
        return Some(value.clone());
    }

    let mut current = value;
    for segment in path.split('.') {
        if segment.is_empty() {
            return None;
        }
        match current {
            Value::Object(object) => {
                current = object.get(segment)?;
            }
            Value::Array(array) => {
                let index = segment.parse::<usize>().ok()?;
                current = array.get(index)?;
            }
            _ => return None,
        }
    }
    Some(current.clone())
}

fn json_value_to_comparable(value: &Value) -> ComparableValue {
    match value {
        Value::Null => ComparableValue::Null,
        Value::Bool(v) => ComparableValue::Bool(*v),
        Value::Number(num) => num
            .as_f64()
            .map(ComparableValue::Number)
            .unwrap_or(ComparableValue::Unsupported),
        Value::String(text) => ComparableValue::String(text.clone()),
        _ => ComparableValue::Unsupported,
    }
}

fn evaluate_comparison(value: Option<ComparableValue>, op: &ComparisonOp) -> bool {
    match value {
        Some(val) => match op {
            ComparisonOp::Equals(expected) => expected.matches_value(&val),
            ComparisonOp::NotEquals(expected) => !expected.matches_value(&val),
            ComparisonOp::GreaterThan(expected) => {
                compare_numbers(&val, expected, Ordering::Greater)
            }
            ComparisonOp::LessThan(expected) => compare_numbers(&val, expected, Ordering::Less),
            ComparisonOp::In(expected) => expected
                .iter()
                .any(|candidate| candidate.matches_value(&val)),
            ComparisonOp::Like(pattern) => match (pattern, val) {
                (FilterValue::String(pat), ComparableValue::String(text)) => {
                    matches_like(&text, pat)
                }
                _ => false,
            },
        },
        None => matches_missing(op),
    }
}

#[derive(Clone, Copy)]
enum Ordering {
    Greater,
    Less,
}

fn compare_numbers(value: &ComparableValue, expected: &FilterValue, ordering: Ordering) -> bool {
    let ComparableValue::Number(lhs) = value else {
        return false;
    };
    let FilterValue::Number(rhs) = expected else {
        return false;
    };
    match ordering {
        Ordering::Greater => *lhs > *rhs,
        Ordering::Less => *lhs < *rhs,
    }
}

const LIKE_REGEX_CACHE_CAPACITY: usize = 128;
const FILTER_EXPR_CACHE_CAPACITY: usize = 256;

#[derive(Clone)]
enum CachedRegex {
    Compiled(regex::Regex),
    Invalid,
}

impl CachedRegex {
    fn as_regex(&self) -> Option<regex::Regex> {
        match self {
            CachedRegex::Compiled(regex) => Some(regex.clone()),
            CachedRegex::Invalid => None,
        }
    }
}

static LIKE_REGEX_CACHE: Lazy<Mutex<LruCache<String, CachedRegex>>> = Lazy::new(|| {
    Mutex::new(LruCache::new(
        NonZeroUsize::new(LIKE_REGEX_CACHE_CAPACITY).expect("LIKE regex cache capacity > 0"),
    ))
});

static FILTER_EXPR_CACHE: Lazy<Mutex<LruCache<String, FilterExpr>>> = Lazy::new(|| {
    Mutex::new(LruCache::new(
        NonZeroUsize::new(FILTER_EXPR_CACHE_CAPACITY)
            .expect("filter expr cache capacity must be > 0"),
    ))
});

fn matches_like(text: &str, pattern: &str) -> bool {
    like_regex(pattern)
        .map(|regex| regex.is_match(text))
        .unwrap_or(false)
}

fn like_regex(pattern: &str) -> Option<regex::Regex> {
    {
        let mut cache = LIKE_REGEX_CACHE.lock();
        if let Some(entry) = cache.get(pattern) {
            return entry.as_regex();
        }
    }

    let regex_source = build_like_regex(pattern);
    let entry = match regex::Regex::new(&regex_source) {
        Ok(regex) => CachedRegex::Compiled(regex),
        Err(_) => CachedRegex::Invalid,
    };
    let result = entry.as_regex();

    {
        let mut cache = LIKE_REGEX_CACHE.lock();
        cache.put(pattern.to_owned(), entry);
    }

    result
}

fn build_like_regex(pattern: &str) -> String {
    let mut regex_pattern = String::from("^");
    for ch in pattern.chars() {
        match ch {
            '%' | '*' => regex_pattern.push_str(".*"),
            '_' | '?' => regex_pattern.push('.'),
            '.' | '+' | '(' | ')' | '|' | '^' | '$' | '{' | '}' | '[' | ']' | '\\' => {
                regex_pattern.push('\\');
                regex_pattern.push(ch);
            }
            other => regex_pattern.push(other),
        }
    }
    regex_pattern.push('$');
    regex_pattern
}

fn matches_missing(op: &ComparisonOp) -> bool {
    matches!(op, ComparisonOp::NotEquals(FilterValue::Null))
}

impl fmt::Display for FilterValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterValue::String(value) => write!(f, "{value}"),
            FilterValue::Number(value) => write!(f, "{value}"),
            FilterValue::Bool(value) => write!(f, "{value}"),
            FilterValue::Null => write!(f, "null"),
        }
    }
}

// === Shorthand Parser ===

pub fn parse_shorthand(input: &str) -> Result<FilterExpr, FilterParseError> {
    let normalized = input.trim();
    if normalized.is_empty() {
        return Err(FilterParseError::Message("filter cannot be empty".into()));
    }

    if let Some(expr) = FILTER_EXPR_CACHE
        .lock()
        .get(normalized)
        .map(|expr| expr.clone())
    {
        return Ok(expr);
    }

    let lexer = Lexer::new(normalized);
    let tokens = lexer.collect::<Result<Vec<_>, _>>()?;
    if tokens.is_empty() {
        return Err(FilterParseError::Message("filter cannot be empty".into()));
    }
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression()?;
    parser.expect_end()?;
    let result = expr.clone();
    {
        let mut cache = FILTER_EXPR_CACHE.lock();
        cache.put(normalized.to_string(), expr);
    }
    Ok(result)
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Identifier(String),
    String(String),
    Number(String),
    Boolean(bool),
    Null,
    And,
    Or,
    Not,
    Operator(Operator),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Operator {
    Eq,
    Neq,
    Gt,
    Lt,
    In,
    Like,
}

struct Lexer<'a> {
    input: &'a str,
    position: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, position: 0 }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Token, FilterParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.consume_whitespace();
        if self.is_eof() {
            return None;
        }

        let ch = self.peek_char()?;
        let token = if is_identifier_start(ch) {
            self.consume_identifier()
        } else if ch == '"' || ch == '\'' {
            self.consume_string()
        } else if ch.is_ascii_digit() || ch == '-' {
            self.consume_number()
        } else {
            match ch {
                '(' => {
                    self.advance();
                    Ok(Token::LParen)
                }
                ')' => {
                    self.advance();
                    Ok(Token::RParen)
                }
                '[' => {
                    self.advance();
                    Ok(Token::LBracket)
                }
                ']' => {
                    self.advance();
                    Ok(Token::RBracket)
                }
                ',' => {
                    self.advance();
                    Ok(Token::Comma)
                }
                '=' => {
                    self.advance();
                    Ok(Token::Operator(Operator::Eq))
                }
                '!' => {
                    self.advance();
                    if self.peek_char() == Some('=') {
                        self.advance();
                        Ok(Token::Operator(Operator::Neq))
                    } else {
                        Err(FilterParseError::Message("expected '=' after '!'".into()))
                    }
                }
                '>' => {
                    self.advance();
                    Ok(Token::Operator(Operator::Gt))
                }
                '<' => {
                    self.advance();
                    Ok(Token::Operator(Operator::Lt))
                }
                _ => Err(FilterParseError::Message(format!(
                    "unexpected character '{}'",
                    ch
                ))),
            }
        };

        Some(token)
    }
}

impl<'a> Lexer<'a> {
    fn consume_whitespace(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn consume_identifier(&mut self) -> Result<Token, FilterParseError> {
        let start = self.position;
        self.advance();
        while let Some(ch) = self.peek_char() {
            if is_identifier_part(ch) {
                self.advance();
            } else {
                break;
            }
        }
        let ident = &self.input[start..self.position];
        let upper = ident.to_ascii_uppercase();
        match upper.as_str() {
            "AND" => Ok(Token::And),
            "OR" => Ok(Token::Or),
            "NOT" => Ok(Token::Not),
            "IN" => Ok(Token::Operator(Operator::In)),
            "LIKE" => Ok(Token::Operator(Operator::Like)),
            "TRUE" => Ok(Token::Boolean(true)),
            "FALSE" => Ok(Token::Boolean(false)),
            "NULL" => Ok(Token::Null),
            _ => Ok(Token::Identifier(ident.to_string())),
        }
    }

    fn consume_string(&mut self) -> Result<Token, FilterParseError> {
        let quote = self.next_char().unwrap();
        let mut value = String::new();
        while let Some(ch) = self.next_char() {
            if ch == quote {
                return Ok(Token::String(value));
            }
            if ch == '\\' {
                if let Some(escaped) = self.next_char() {
                    value.push(escaped);
                } else {
                    return Err(FilterParseError::Message(
                        "unterminated escape sequence".into(),
                    ));
                }
            } else {
                value.push(ch);
            }
        }
        Err(FilterParseError::Message(
            "unterminated string literal".into(),
        ))
    }

    fn consume_number(&mut self) -> Result<Token, FilterParseError> {
        let start = self.position;
        self.advance();
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_digit() || ch == '.' {
                self.advance();
            } else {
                break;
            }
        }
        Ok(Token::Number(self.input[start..self.position].to_string()))
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.position..].chars().next()
    }

    fn next_char(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.position += ch.len_utf8();
        Some(ch)
    }

    fn advance(&mut self) {
        self.next_char();
    }

    fn is_eof(&self) -> bool {
        self.position >= self.input.len()
    }
}

fn is_identifier_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_' || ch == '$' || ch == '@'
}

fn is_identifier_part(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' || ch == '$' || ch == '@'
}

struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    fn parse_expression(&mut self) -> Result<FilterExpr, FilterParseError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<FilterExpr, FilterParseError> {
        let mut expr = self.parse_and()?;
        while self.match_token(TokenKind::Or) {
            self.advance();
            let right = self.parse_and()?;
            expr = match expr {
                FilterExpr::Or(mut nodes) => {
                    nodes.push(right);
                    FilterExpr::Or(nodes)
                }
                _ => FilterExpr::Or(vec![expr, right]),
            };
        }
        Ok(expr)
    }

    fn parse_and(&mut self) -> Result<FilterExpr, FilterParseError> {
        let mut expr = self.parse_not()?;
        while self.match_token(TokenKind::And) {
            self.advance();
            let right = self.parse_not()?;
            expr = match expr {
                FilterExpr::And(mut nodes) => {
                    nodes.push(right);
                    FilterExpr::And(nodes)
                }
                _ => FilterExpr::And(vec![expr, right]),
            };
        }
        Ok(expr)
    }

    fn parse_not(&mut self) -> Result<FilterExpr, FilterParseError> {
        if self.match_token(TokenKind::Not) {
            self.advance();
            let expr = self.parse_not()?;
            Ok(FilterExpr::Not(Box::new(expr)))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<FilterExpr, FilterParseError> {
        if self.match_token(TokenKind::LParen) {
            self.advance();
            let expr = self.parse_expression()?;
            self.expect(TokenKind::RParen, "')'")?;
            Ok(expr)
        } else {
            self.parse_comparison()
        }
    }

    fn parse_comparison(&mut self) -> Result<FilterExpr, FilterParseError> {
        let field = match self.current_token() {
            Some(Token::Identifier(name)) => {
                let value = name.clone();
                self.advance();
                value
            }
            _ => {
                return Err(FilterParseError::Message(
                    "expected field name in comparison".into(),
                ));
            }
        };

        let mut negate = false;
        if self.match_token(TokenKind::Not) {
            self.advance();
            negate = true;
        }

        let operator = match self.current_token() {
            Some(Token::Operator(operator)) => *operator,
            _ => {
                return Err(FilterParseError::Message(
                    "expected comparison operator".into(),
                ));
            }
        };
        self.advance();

        let expr = match operator {
            Operator::In => {
                self.expect(TokenKind::LBracket, "'['")?;
                let mut values = Vec::new();
                while !self.match_token(TokenKind::RBracket) {
                    values.push(self.parse_value()?);
                    if self.match_token(TokenKind::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                self.expect(TokenKind::RBracket, "']'")?;
                FilterExpr::Comparison {
                    field,
                    op: ComparisonOp::In(values),
                }
            }
            Operator::Like => {
                let value = self.parse_value()?;
                FilterExpr::Comparison {
                    field,
                    op: ComparisonOp::Like(value),
                }
            }
            Operator::Eq => {
                let value = self.parse_value()?;
                FilterExpr::Comparison {
                    field,
                    op: ComparisonOp::Equals(value),
                }
            }
            Operator::Neq => {
                let value = self.parse_value()?;
                FilterExpr::Comparison {
                    field,
                    op: ComparisonOp::NotEquals(value),
                }
            }
            Operator::Gt => {
                let value = self.parse_value()?;
                FilterExpr::Comparison {
                    field,
                    op: ComparisonOp::GreaterThan(value),
                }
            }
            Operator::Lt => {
                let value = self.parse_value()?;
                FilterExpr::Comparison {
                    field,
                    op: ComparisonOp::LessThan(value),
                }
            }
        };

        if negate {
            Ok(FilterExpr::Not(Box::new(expr)))
        } else {
            Ok(expr)
        }
    }

    fn parse_value(&mut self) -> Result<FilterValue, FilterParseError> {
        let token = self
            .current_token()
            .ok_or_else(|| FilterParseError::Message("unexpected end of filter".into()))?
            .clone();
        self.advance();
        match token {
            Token::String(value) => Ok(FilterValue::String(value)),
            Token::Number(value) => value
                .parse::<f64>()
                .map(FilterValue::Number)
                .map_err(|err| FilterParseError::Message(format!("invalid number: {err}"))),
            Token::Boolean(value) => Ok(FilterValue::Bool(value)),
            Token::Null => Ok(FilterValue::Null),
            Token::Identifier(value) => Ok(FilterValue::String(value)),
            _ => Err(FilterParseError::Message("invalid value literal".into())),
        }
    }

    fn expect(&mut self, kind: TokenKind, label: &str) -> Result<(), FilterParseError> {
        if self.match_token(kind) {
            self.advance();
            Ok(())
        } else {
            Err(FilterParseError::Message(format!("expected {label}")))
        }
    }

    fn expect_end(&self) -> Result<(), FilterParseError> {
        if self.position >= self.tokens.len() {
            Ok(())
        } else {
            Err(FilterParseError::Message(
                "unexpected tokens after expression".into(),
            ))
        }
    }

    fn current_token(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn advance(&mut self) {
        if self.position < self.tokens.len() {
            self.position += 1;
        }
    }

    fn match_token(&self, kind: TokenKind) -> bool {
        matches!(
            self.current_token(),
            Some(token) if TokenKind::from(token) == kind
        )
    }
}

#[derive(Clone, Copy, PartialEq)]
enum TokenKind {
    Identifier,
    String,
    Number,
    Boolean,
    Null,
    And,
    Or,
    Not,
    Operator,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
}

impl From<&Token> for TokenKind {
    fn from(token: &Token) -> Self {
        match token {
            Token::Identifier(_) => TokenKind::Identifier,
            Token::String(_) => TokenKind::String,
            Token::Number(_) => TokenKind::Number,
            Token::Boolean(_) => TokenKind::Boolean,
            Token::Null => TokenKind::Null,
            Token::And => TokenKind::And,
            Token::Or => TokenKind::Or,
            Token::Not => TokenKind::Not,
            Token::Operator(_) => TokenKind::Operator,
            Token::LParen => TokenKind::LParen,
            Token::RParen => TokenKind::RParen,
            Token::LBracket => TokenKind::LBracket,
            Token::RBracket => TokenKind::RBracket,
            Token::Comma => TokenKind::Comma,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn sample_aggregate() -> AggregateState {
        let mut state = BTreeMap::new();
        state.insert("status".to_string(), "\"pending\"".to_string());
        state.insert("score".to_string(), "42".to_string());
        state.insert(
            "name".to_string(),
            r#"{"first":"John","last":"Thach"}"#.to_string(),
        );

        AggregateState {
            aggregate_type: "order".to_string(),
            aggregate_id: "order-123".to_string(),
            version: 7,
            state,
            extensions: BTreeMap::new(),
            merkle_root: "root".to_string(),
            created_at: None,
            updated_at: None,
            archived: false,
        }
    }

    #[test]
    fn parse_shorthand_handles_basic_comparison() {
        let expr = parse_shorthand(r#"last_name = "thach""#).expect("parse succeeds");
        match expr {
            FilterExpr::Comparison { field, op } => {
                assert_eq!(field, "last_name");
                match op {
                    ComparisonOp::Equals(FilterValue::String(value)) => {
                        assert_eq!(value, "thach");
                    }
                    _ => panic!("unexpected comparison operator"),
                }
            }
            _ => panic!("expected comparison expression"),
        }
    }

    #[test]
    fn parse_shorthand_supports_nested_logic() {
        let expr =
            parse_shorthand(r#"(status = "pending" OR status = "paid") AND archived = false"#)
                .expect("parse succeeds");

        let FilterExpr::And(parts) = expr else {
            panic!("expected AND expression");
        };
        assert_eq!(parts.len(), 2);

        match &parts[0] {
            FilterExpr::Or(branches) => {
                assert_eq!(branches.len(), 2);
                match &branches[0] {
                    FilterExpr::Comparison { field, op } => {
                        assert_eq!(field, "status");
                        match op {
                            ComparisonOp::Equals(FilterValue::String(value)) => {
                                assert_eq!(value, "pending");
                            }
                            _ => panic!("expected equals comparison"),
                        }
                    }
                    _ => panic!("expected comparison branch"),
                }
            }
            _ => panic!("expected OR branch"),
        }

        match &parts[1] {
            FilterExpr::Comparison { field, op } => {
                assert_eq!(field, "archived");
                match op {
                    ComparisonOp::Equals(FilterValue::Bool(value)) => {
                        assert!(!value);
                    }
                    _ => panic!("expected boolean equals"),
                }
            }
            _ => panic!("expected comparison for second part"),
        }
    }

    #[test]
    fn parse_shorthand_rejects_invalid_syntax() {
        assert!(parse_shorthand("").is_err());
        assert!(parse_shorthand(r#"status = "pending" AND"#).is_err());
        assert!(parse_shorthand(r#"status @ "pending""#).is_err());
    }

    #[test]
    fn matches_aggregate_handles_string_comparison() {
        let aggregate = sample_aggregate();
        let expr = parse_shorthand(r#"status = "pending""#).expect("parse succeeds");
        assert!(expr.matches_aggregate(&aggregate));

        let expr = parse_shorthand(r#"status = "cancelled""#).expect("parse succeeds");
        assert!(!expr.matches_aggregate(&aggregate));
    }

    #[test]
    fn matches_aggregate_handles_in_and_like() {
        let aggregate = sample_aggregate();

        let expr = parse_shorthand(r#"status IN ["pending","paid"]"#).expect("parse succeeds");
        assert!(expr.matches_aggregate(&aggregate));

        let expr = parse_shorthand(r#"aggregate_id LIKE "order%""#).expect("parse succeeds");
        assert!(expr.matches_aggregate(&aggregate));
    }

    #[test]
    fn matches_aggregate_handles_numeric_and_bool_fields() {
        let aggregate = sample_aggregate();

        let expr = parse_shorthand(r#"score > 40"#).expect("parse succeeds");
        assert!(expr.matches_aggregate(&aggregate));

        let expr = parse_shorthand(r#"score < 40"#).expect("parse succeeds");
        assert!(!expr.matches_aggregate(&aggregate));

        let expr = parse_shorthand(r#"archived = false"#).expect("parse succeeds");
        assert!(expr.matches_aggregate(&aggregate));

        let expr = parse_shorthand(r#"NOT archived = true"#).expect("parse succeeds");
        assert!(expr.matches_aggregate(&aggregate));
    }

    #[test]
    fn matches_aggregate_handles_nested_fields() {
        let aggregate = sample_aggregate();
        let expr = parse_shorthand(r#"name.first = "John""#).expect("parse succeeds");
        assert!(expr.matches_aggregate(&aggregate));

        let expr = parse_shorthand(r#"name.first = "Jane""#).expect("parse succeeds");
        assert!(!expr.matches_aggregate(&aggregate));
    }

    #[test]
    fn parse_shorthand_allows_at_prefixed_segments() {
        let expr = parse_shorthand(r#"extensions.@trace.correlation_id = "cli-22""#)
            .expect("parse succeeds");
        match expr {
            FilterExpr::Comparison { field, op } => {
                assert_eq!(field, "extensions.@trace.correlation_id");
                match op {
                    ComparisonOp::Equals(FilterValue::String(value)) => {
                        assert_eq!(value, "cli-22");
                    }
                    _ => panic!("expected string equals comparison"),
                }
            }
            _ => panic!("expected comparison expression"),
        }
    }

    #[test]
    fn parse_shorthand_caches_results() {
        {
            let mut cache = FILTER_EXPR_CACHE.lock();
            cache.clear();
        }
        let expr_text = r#"status = "pending""#;
        let expr = parse_shorthand(expr_text).expect("parse succeeds");
        let FilterExpr::Comparison { .. } = expr else {
            panic!("expected comparison expression");
        };

        {
            let cache = FILTER_EXPR_CACHE.lock();
            assert!(
                cache.peek(expr_text).is_some(),
                "expression should be cached"
            );
        }

        // Leading/trailing whitespace should hit the same cache entry (not a new key).
        let spaced = r#"   status = "pending"   "#;
        parse_shorthand(spaced).expect("parse succeeds with whitespace");
        let cache = FILTER_EXPR_CACHE.lock();
        assert!(
            cache.peek(expr_text.trim()).is_some(),
            "trimmed expression should remain cached"
        );
        assert!(
            cache.peek(spaced).is_none(),
            "whitespace variant should not introduce new cache key"
        );
    }
}
