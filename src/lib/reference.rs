use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    error::{EventError, Result},
    schema::SchemaManager,
    store::AggregateState,
    validation::{ensure_aggregate_id, ensure_snake_case},
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AggregateReference {
    pub domain: String,
    pub aggregate_type: String,
    pub aggregate_id: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceResolutionStatus {
    Ok,
    NotFound,
    Forbidden,
    Cycle,
    DepthExceeded,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceCascade {
    None,
    Restrict,
    Nullify,
}

impl Default for ReferenceCascade {
    fn default() -> Self {
        ReferenceCascade::None
    }
}

pub const DEFAULT_RESOLUTION_DEPTH: usize = 2;
pub const MAX_RESOLUTION_DEPTH: usize = 5;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceIntegrity {
    Strong,
    Weak,
}

impl Default for ReferenceIntegrity {
    fn default() -> Self {
        ReferenceIntegrity::Strong
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReferenceRules {
    #[serde(default)]
    pub integrity: ReferenceIntegrity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggregate_type: Option<String>,
    #[serde(default)]
    pub cascade: ReferenceCascade,
}

impl Default for ReferenceRules {
    fn default() -> Self {
        Self {
            integrity: ReferenceIntegrity::Strong,
            tenant: None,
            aggregate_type: None,
            cascade: ReferenceCascade::None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReferenceContext<'a> {
    pub domain: &'a str,
    pub aggregate_type: &'a str,
}

impl AggregateReference {
    pub fn parse(raw: &str, context: ReferenceContext<'_>) -> Result<Self> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(EventError::SchemaViolation(
                "reference value cannot be empty".into(),
            ));
        }

        let parts: Vec<&str> = trimmed.split('#').collect();
        let (domain, aggregate_type, aggregate_id) = match parts.len() {
            3 => (parts[0], parts[1], parts[2]),
            2 => {
                if parts[0].is_empty() {
                    (context.domain, context.aggregate_type, parts[1])
                } else {
                    (context.domain, parts[0], parts[1])
                }
            }
            _ => {
                return Err(EventError::SchemaViolation(
                    "reference must be domain#aggregate#id, aggregate#id, or #id".into(),
                ));
            }
        };

        if aggregate_id.is_empty() || aggregate_type.is_empty() || domain.is_empty() {
            return Err(EventError::SchemaViolation(
                "reference segments cannot be empty".into(),
            ));
        }

        let domain = normalize_domain(domain)?;
        let aggregate_type = normalize_aggregate_type(aggregate_type)?;
        let aggregate_id = normalize_aggregate_id(aggregate_id)?;

        Ok(Self {
            domain,
            aggregate_type,
            aggregate_id,
        })
    }

    pub fn to_canonical(&self) -> String {
        format!(
            "{}#{}#{}",
            self.domain, self.aggregate_type, self.aggregate_id
        )
    }
}

impl fmt::Display for AggregateReference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_canonical())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReferenceOutcome {
    pub path: String,
    pub reference: AggregateReference,
    pub status: ReferenceResolutionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedReference {
    pub path: String,
    pub reference: AggregateReference,
    pub status: ReferenceResolutionStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved: Option<Box<ResolvedAggregate>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Referrer {
    pub aggregate_type: String,
    pub aggregate_id: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedAggregate {
    pub domain: String,
    pub aggregate: AggregateState,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub references: Vec<ResolvedReference>,
}

#[derive(Debug, Clone)]
pub struct ReferenceFetchOutcome {
    pub status: ReferenceResolutionStatus,
    pub aggregate: Option<AggregateState>,
}

fn normalize_domain(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(EventError::SchemaViolation(
            "reference domain cannot be empty".into(),
        ));
    }

    let lower = trimmed.to_ascii_lowercase();
    if !matches!(lower.chars().next(), Some(ch) if ch.is_ascii_alphanumeric()) {
        return Err(EventError::SchemaViolation(
            "reference domain must begin with an ASCII letter or digit".into(),
        ));
    }
    if !lower
        .chars()
        .skip(1)
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        return Err(EventError::SchemaViolation(
            "reference domain may only contain letters, numbers, '-' or '_'".into(),
        ));
    }
    Ok(lower)
}

fn normalize_aggregate_type(value: &str) -> Result<String> {
    ensure_snake_case("aggregate_type", value).map_err(|err| match err {
        EventError::InvalidSchema(message) | EventError::SchemaViolation(message) => {
            EventError::SchemaViolation(message)
        }
        other => other,
    })?;
    Ok(value.to_string())
}

fn normalize_aggregate_id(value: &str) -> Result<String> {
    ensure_aggregate_id(value).map_err(|err| match err {
        EventError::InvalidSchema(message) | EventError::SchemaViolation(message) => {
            EventError::SchemaViolation(message)
        }
        other => other,
    })?;
    Ok(value.to_string())
}

#[derive(Debug, Clone)]
struct VisitedKey {
    domain: String,
    aggregate_type: String,
    aggregate_id: String,
}

impl VisitedKey {
    fn matches(&self, reference: &AggregateReference) -> bool {
        self.domain.eq_ignore_ascii_case(&reference.domain)
            && self.aggregate_type == reference.aggregate_type
            && self.aggregate_id == reference.aggregate_id
    }
}

pub fn resolve_references<F>(
    root_domain: String,
    root: AggregateState,
    schemas: &SchemaManager,
    depth: usize,
    mut fetcher: F,
) -> Result<ResolvedAggregate>
where
    F: FnMut(&AggregateReference) -> Result<ReferenceFetchOutcome>,
{
    let hops = depth.min(MAX_RESOLUTION_DEPTH);
    let mut stack = Vec::new();
    resolve_references_inner(root_domain, root, schemas, hops, &mut fetcher, &mut stack)
}

fn resolve_references_inner<F>(
    domain: String,
    aggregate: AggregateState,
    schemas: &SchemaManager,
    hops_remaining: usize,
    fetcher: &mut F,
    stack: &mut Vec<VisitedKey>,
) -> Result<ResolvedAggregate>
where
    F: FnMut(&AggregateReference) -> Result<ReferenceFetchOutcome>,
{
    let context = ReferenceContext {
        domain: &domain,
        aggregate_type: &aggregate.aggregate_type,
    };
    let located =
        schemas.collect_references(&aggregate.aggregate_type, &aggregate.state, context)?;

    let mut resolved = Vec::new();
    let key = VisitedKey {
        domain: domain.clone(),
        aggregate_type: aggregate.aggregate_type.clone(),
        aggregate_id: aggregate.aggregate_id.clone(),
    };
    stack.push(key);

    for reference in located {
        if hops_remaining == 0 {
            resolved.push(ResolvedReference {
                path: reference.path,
                reference: reference.reference,
                status: ReferenceResolutionStatus::DepthExceeded,
                resolved: None,
            });
            continue;
        }

        if stack
            .iter()
            .any(|entry| entry.matches(&reference.reference))
        {
            resolved.push(ResolvedReference {
                path: reference.path,
                reference: reference.reference,
                status: ReferenceResolutionStatus::Cycle,
                resolved: None,
            });
            continue;
        }

        let outcome = fetcher(&reference.reference)?;
        let (status, aggregate_opt) = match outcome.status {
            ReferenceResolutionStatus::Ok => match outcome.aggregate {
                Some(aggregate) => (ReferenceResolutionStatus::Ok, Some(aggregate)),
                None => (ReferenceResolutionStatus::NotFound, None),
            },
            other => (other, None),
        };

        if status == ReferenceResolutionStatus::Ok {
            let child = aggregate_opt.expect("child aggregate must be present on ok status");
            let next = resolve_references_inner(
                reference.reference.domain.clone(),
                child,
                schemas,
                hops_remaining.saturating_sub(1),
                fetcher,
                stack,
            )?;
            resolved.push(ResolvedReference {
                path: reference.path,
                reference: reference.reference,
                status,
                resolved: Some(Box::new(next)),
            });
        } else {
            resolved.push(ResolvedReference {
                path: reference.path,
                reference: reference.reference,
                status,
                resolved: None,
            });
        }
    }

    stack.pop();
    Ok(ResolvedAggregate {
        domain,
        aggregate,
        references: resolved,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{
        ColumnType, CreateSchemaInput, FieldFormat, FieldRules, SchemaManager, SchemaUpdate,
    };
    use std::collections::BTreeMap;
    use tempfile::tempdir;

    #[test]
    fn parses_canonical_reference() {
        let context = ReferenceContext {
            domain: "default",
            aggregate_type: "farm",
        };
        let reference =
            AggregateReference::parse("geo#address#123", context).expect("reference should parse");

        assert_eq!(reference.domain, "geo");
        assert_eq!(reference.aggregate_type, "address");
        assert_eq!(reference.aggregate_id, "123");
        assert_eq!(reference.to_string(), "geo#address#123");
    }

    #[test]
    fn fills_domain_when_missing() {
        let context = ReferenceContext {
            domain: "farm",
            aggregate_type: "farm",
        };
        let reference =
            AggregateReference::parse("address#123", context).expect("reference should parse");

        assert_eq!(reference.domain, "farm");
        assert_eq!(reference.aggregate_type, "address");
        assert_eq!(reference.aggregate_id, "123");
    }

    #[test]
    fn uses_context_for_shorthand_id() {
        let context = ReferenceContext {
            domain: "farm",
            aggregate_type: "cattle",
        };
        let reference = AggregateReference::parse("#abc-123", context).expect("should parse");

        assert_eq!(reference.domain, "farm");
        assert_eq!(reference.aggregate_type, "cattle");
        assert_eq!(reference.aggregate_id, "abc-123");
    }

    #[test]
    fn rejects_invalid_shapes() {
        let context = ReferenceContext {
            domain: "farm",
            aggregate_type: "cattle",
        };
        for raw in ["", "just-id", "a#b#c#d", "#", "farm##123"] {
            let err = AggregateReference::parse(raw, context).unwrap_err();
            assert!(matches!(err, EventError::SchemaViolation(_)), "{raw}");
        }
    }

    #[test]
    fn resolve_references_resolves_children() {
        let (manager, _guard) = prepare_schema_with_reference("farm", "address");
        manager
            .create(CreateSchemaInput {
                aggregate: "address".into(),
                events: vec!["created".into()],
                snapshot_threshold: None,
            })
            .unwrap();

        let farm = sample_state("farm", "1", &[("address", "default#address#1")]);
        let address = sample_state("address", "1", &[]);

        let mut aggregates = std::collections::HashMap::new();
        aggregates.insert("default#farm#1".to_string(), farm.clone());
        aggregates.insert("default#address#1".to_string(), address.clone());

        let resolved = resolve_references(
            "default".into(),
            farm,
            &manager,
            DEFAULT_RESOLUTION_DEPTH,
            |reference| {
                let key = reference.to_canonical();
                let aggregate = aggregates.get(&key).cloned();
                Ok(ReferenceFetchOutcome {
                    status: if aggregate.is_some() {
                        ReferenceResolutionStatus::Ok
                    } else {
                        ReferenceResolutionStatus::NotFound
                    },
                    aggregate,
                })
            },
        )
        .expect("resolution should succeed");

        assert_eq!(resolved.references.len(), 1);
        let first = &resolved.references[0];
        assert_eq!(first.status, ReferenceResolutionStatus::Ok);
        let child = first
            .resolved
            .as_ref()
            .expect("child should be present")
            .aggregate
            .clone();
        assert_eq!(child.aggregate_type, "address");
        assert_eq!(child.aggregate_id, "1");
    }

    #[test]
    fn resolve_references_detects_cycles() {
        let (manager, _guard) = prepare_schema_with_reference("farm", "address");
        manager
            .create(CreateSchemaInput {
                aggregate: "address".into(),
                events: vec!["created".into()],
                snapshot_threshold: None,
            })
            .unwrap();
        add_reference_field(&manager, "address", "owner");

        let farm = sample_state("farm", "1", &[("address", "default#address#1")]);
        let address = sample_state("address", "1", &[("owner", "default#farm#1")]);

        let mut aggregates = std::collections::HashMap::new();
        aggregates.insert("default#farm#1".to_string(), farm.clone());
        aggregates.insert("default#address#1".to_string(), address.clone());

        let resolved = resolve_references(
            "default".into(),
            farm,
            &manager,
            DEFAULT_RESOLUTION_DEPTH,
            |reference| {
                let key = reference.to_canonical();
                let aggregate = aggregates.get(&key).cloned();
                Ok(ReferenceFetchOutcome {
                    status: if aggregate.is_some() {
                        ReferenceResolutionStatus::Ok
                    } else {
                        ReferenceResolutionStatus::NotFound
                    },
                    aggregate,
                })
            },
        )
        .expect("resolution should succeed");

        let first = resolved
            .references
            .first()
            .and_then(|ref_entry| ref_entry.resolved.as_ref())
            .expect("child resolved");
        let cycle_status = first
            .references
            .first()
            .map(|entry| entry.status)
            .expect("nested reference should exist");
        assert_eq!(cycle_status, ReferenceResolutionStatus::Cycle);
    }

    #[test]
    fn resolve_references_respects_depth() {
        let (manager, _guard) = prepare_schema_with_reference("farm", "address");
        manager
            .create(CreateSchemaInput {
                aggregate: "address".into(),
                events: vec!["created".into()],
                snapshot_threshold: None,
            })
            .unwrap();
        manager
            .create(CreateSchemaInput {
                aggregate: "facility".into(),
                events: vec!["created".into()],
                snapshot_threshold: None,
            })
            .unwrap();
        add_reference_field(&manager, "address", "facility");

        let farm = sample_state("farm", "1", &[("address", "default#address#1")]);
        let address = sample_state("address", "1", &[("facility", "default#facility#9")]);
        let facility = sample_state("facility", "9", &[]);

        let mut aggregates = std::collections::HashMap::new();
        aggregates.insert("default#farm#1".to_string(), farm.clone());
        aggregates.insert("default#address#1".to_string(), address.clone());
        aggregates.insert("default#facility#9".to_string(), facility.clone());

        let resolved = resolve_references("default".into(), farm, &manager, 1, |reference| {
            let key = reference.to_canonical();
            let aggregate = aggregates.get(&key).cloned();
            Ok(ReferenceFetchOutcome {
                status: if aggregate.is_some() {
                    ReferenceResolutionStatus::Ok
                } else {
                    ReferenceResolutionStatus::NotFound
                },
                aggregate,
            })
        })
        .expect("resolution should succeed");

        let first = resolved
            .references
            .first()
            .and_then(|entry| entry.resolved.as_ref())
            .expect("first hop resolved");
        let leaf_status = first
            .references
            .first()
            .map(|entry| entry.status)
            .expect("nested entry should exist");
        assert_eq!(leaf_status, ReferenceResolutionStatus::DepthExceeded);
    }

    #[test]
    fn resolve_references_cross_domain_allows_when_fetcher_permits() {
        let (manager, _guard) = prepare_schema_with_reference("farm", "address");

        let farm = sample_state("farm", "1", &[("address", "geo#address#1")]);
        let address = sample_state("address", "1", &[]);

        let mut aggregates = std::collections::HashMap::new();
        aggregates.insert("geo#address#1".to_string(), address.clone());

        let resolved = resolve_references(
            "default".into(),
            farm,
            &manager,
            DEFAULT_RESOLUTION_DEPTH,
            |reference| {
                let key = reference.to_canonical();
                let aggregate = aggregates.get(&key).cloned();
                Ok(ReferenceFetchOutcome {
                    status: if aggregate.is_some() {
                        ReferenceResolutionStatus::Ok
                    } else {
                        ReferenceResolutionStatus::NotFound
                    },
                    aggregate,
                })
            },
        )
        .expect("resolution should succeed");

        let first = resolved
            .references
            .first()
            .expect("reference should be present");
        assert_eq!(first.status, ReferenceResolutionStatus::Ok);
        assert_eq!(first.reference.domain, "geo");
        let child = first
            .resolved
            .as_ref()
            .expect("resolved child should exist")
            .aggregate
            .clone();
        assert_eq!(child.aggregate_type, "address");
        assert_eq!(child.aggregate_id, "1");
    }

    #[test]
    fn resolve_references_cross_domain_marks_forbidden_when_fetcher_blocks() {
        let (manager, _guard) = prepare_schema_with_reference("farm", "address");

        let farm = sample_state("farm", "1", &[("address", "geo#address#1")]);

        let resolved = resolve_references(
            "default".into(),
            farm,
            &manager,
            DEFAULT_RESOLUTION_DEPTH,
            |_reference| {
                Ok(ReferenceFetchOutcome {
                    status: ReferenceResolutionStatus::Forbidden,
                    aggregate: None,
                })
            },
        )
        .expect("resolution should succeed");

        let first = resolved
            .references
            .first()
            .expect("reference should be present");
        assert_eq!(first.status, ReferenceResolutionStatus::Forbidden);
        assert!(first.resolved.is_none());
    }

    fn prepare_schema_with_reference(
        aggregate: &str,
        field: &str,
    ) -> (SchemaManager, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("schemas.json");
        let manager = SchemaManager::load(path).unwrap();
        manager
            .create(CreateSchemaInput {
                aggregate: aggregate.into(),
                events: vec!["created".into()],
                snapshot_threshold: None,
            })
            .unwrap();
        add_reference_field(&manager, aggregate, field);
        (manager, dir)
    }

    fn add_reference_field(manager: &SchemaManager, aggregate: &str, field: &str) {
        let mut type_update = SchemaUpdate::default();
        type_update.column_type = Some((field.to_string(), Some(ColumnType::Text)));
        manager.update(aggregate, type_update).unwrap();

        let mut rules = FieldRules::default();
        rules.format = Some(FieldFormat::Reference);
        rules.reference = Some(ReferenceRules::default());

        let mut rules_update = SchemaUpdate::default();
        rules_update.column_rules = Some((field.to_string(), Some(rules)));
        manager.update(aggregate, rules_update).unwrap();
    }

    fn sample_state(
        aggregate_type: &str,
        aggregate_id: &str,
        fields: &[(&str, &str)],
    ) -> AggregateState {
        let mut state = BTreeMap::new();
        for (key, value) in fields {
            state.insert((*key).to_string(), (*value).to_string());
        }
        AggregateState {
            aggregate_type: aggregate_type.to_string(),
            aggregate_id: aggregate_id.to_string(),
            version: 1,
            state,
            merkle_root: String::new(),
            created_at: None,
            updated_at: None,
            archived: false,
        }
    }
}
