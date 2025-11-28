@0xc3d1ec2a1e3f5b26;

using Schema = import "schema.capnp";

struct ControlRequest {
  id @0 :UInt64;
  payload :union {
    listAggregates @1 :ListAggregatesRequest;
    getAggregate @2 :GetAggregateRequest;
    listEvents @3 :ListEventsRequest;
    appendEvent @4 :AppendEventRequest;
    verifyAggregate @5 :VerifyAggregateRequest;
    patchEvent @6 :PatchEventRequest;
    selectAggregate @7 :SelectAggregateRequest;
    createAggregate @8 :CreateAggregateRequest;
    setAggregateArchive @9 :SetAggregateArchiveRequest;
    listSchemas @10 :Schema.ListSchemasRequest;
    replaceSchemas @11 :Schema.ReplaceSchemasRequest;
    tenantAssign @12 :Schema.TenantAssignRequest;
    tenantUnassign @13 :Schema.TenantUnassignRequest;
    tenantQuotaSet @14 :Schema.TenantQuotaSetRequest;
    tenantQuotaClear @15 :Schema.TenantQuotaClearRequest;
    tenantQuotaRecalc @16 :Schema.TenantQuotaRecalcRequest;
    tenantReload @17 :Schema.TenantReloadRequest;
    tenantSchemaPublish @18 :Schema.TenantSchemaPublishRequest;
    createSnapshot @19 :CreateSnapshotRequest;
    listSnapshots @20 :ListSnapshotsRequest;
    getSnapshot @21 :GetSnapshotRequest;
    listReferrers @22 :ListReferrersRequest;
  }
}

struct ControlResponse {
  id @0 :UInt64;
  payload :union {
    listAggregates @1 :ListAggregatesResponse;
    getAggregate @2 :GetAggregateResponse;
    listEvents @3 :ListEventsResponse;
    appendEvent @4 :AppendEventResponse;
    verifyAggregate @5 :VerifyAggregateResponse;
    selectAggregate @6 :SelectAggregateResponse;
    error @7 :ControlError;
    createAggregate @8 :CreateAggregateResponse;
    setAggregateArchive @9 :SetAggregateArchiveResponse;
    listSchemas @10 :Schema.ListSchemasResponse;
    replaceSchemas @11 :Schema.ReplaceSchemasResponse;
    tenantAssign @12 :Schema.TenantAssignResponse;
    tenantUnassign @13 :Schema.TenantUnassignResponse;
    tenantQuotaSet @14 :Schema.TenantQuotaSetResponse;
    tenantQuotaClear @15 :Schema.TenantQuotaClearResponse;
    tenantQuotaRecalc @16 :Schema.TenantQuotaRecalcResponse;
    tenantReload @17 :Schema.TenantReloadResponse;
    tenantSchemaPublish @18 :Schema.TenantSchemaPublishResponse;
    createSnapshot @19 :CreateSnapshotResponse;
    listSnapshots @20 :ListSnapshotsResponse;
    getSnapshot @21 :GetSnapshotResponse;
    listReferrers @22 :ListReferrersResponse;
  }
}

struct ControlHello {
  protocolVersion @0 :UInt16;
  token @1 :Text;
  tenantId @2 :Text;
  noNoise @3 :Bool;
}

struct ControlHelloResponse {
  accepted @0 :Bool;
  message @1 :Text;
  noNoise @2 :Bool;
}

struct ListAggregatesRequest {
  cursor @0 :Text;
  hasCursor @1 :Bool;
  take @2 :UInt64;
  hasTake @3 :Bool;
  filter @4 :Text;
  hasFilter @5 :Bool;
  sort @6 :Text;
  hasSort @7 :Bool;
  includeArchived @8 :Bool;
  archivedOnly @9 :Bool;
  token @10 :Text;
  resolve @11 :Bool;
  resolveDepth @12 :UInt32;
  hasResolveDepth @13 :Bool;
}

struct ListAggregatesResponse {
  aggregatesJson @0 :Text;
  nextCursor @1 :Text;
  hasNextCursor @2 :Bool;
  resolvedJson @3 :Text;
  hasResolvedJson @4 :Bool;
}

struct GetAggregateRequest {
  aggregateType @0 :Text;
  aggregateId @1 :Text;
  token @2 :Text;
  resolve @3 :Bool;
  resolveDepth @4 :UInt32;
  hasResolveDepth @5 :Bool;
}

struct GetAggregateResponse {
  found @0 :Bool;
  aggregateJson @1 :Text;
  resolvedJson @2 :Text;
  hasResolvedJson @3 :Bool;
}

struct ListReferrersRequest {
  token @0 :Text;
  aggregateType @1 :Text;
  aggregateId @2 :Text;
}

struct ListReferrersResponse {
  referrersJson @0 :Text;
}

struct ListEventsRequest {
  aggregateType @0 :Text;
  aggregateId @1 :Text;
  cursor @2 :Text;
  hasCursor @3 :Bool;
  take @4 :UInt64;
  hasTake @5 :Bool;
  filter @6 :Text;
  hasFilter @7 :Bool;
  token @8 :Text;
}

struct ListEventsResponse {
  eventsJson @0 :Text;
  nextCursor @1 :Text;
  hasNextCursor @2 :Bool;
}

struct AppendEventRequest {
  token @0 :Text;
  aggregateType @1 :Text;
  aggregateId @2 :Text;
  eventType @3 :Text;
  payloadJson @4 :Text;
  note @5 :Text;
  hasNote @6 :Bool;
  metadataJson @7 :Text;
  hasMetadata @8 :Bool;
  publishTargets @9 :List(PublishTarget);
  hasPublishTargets @10 :Bool;
}

struct AppendEventResponse {
  eventJson @0 :Text;
}

struct PatchEventRequest {
  token @0 :Text;
  aggregateType @1 :Text;
  aggregateId @2 :Text;
  eventType @3 :Text;
  patchJson @4 :Text;
  note @5 :Text;
  hasNote @6 :Bool;
  metadataJson @7 :Text;
  hasMetadata @8 :Bool;
  publishTargets @9 :List(PublishTarget);
  hasPublishTargets @10 :Bool;
}

struct VerifyAggregateRequest {
  aggregateType @0 :Text;
  aggregateId @1 :Text;
}

struct VerifyAggregateResponse {
  merkleRoot @0 :Text;
}

struct SelectAggregateRequest {
  aggregateType @0 :Text;
  aggregateId @1 :Text;
  fields @2 :List(Text);
  token @3 :Text;
}

struct SelectAggregateResponse {
  found @0 :Bool;
  selectionJson @1 :Text;
}

struct CreateAggregateRequest {
  token @0 :Text;
  aggregateType @1 :Text;
  aggregateId @2 :Text;
  eventType @3 :Text;
  payloadJson @4 :Text;
  note @5 :Text;
  hasNote @6 :Bool;
  metadataJson @7 :Text;
  hasMetadata @8 :Bool;
  publishTargets @9 :List(PublishTarget);
  hasPublishTargets @10 :Bool;
}

struct CreateAggregateResponse {
  aggregateJson @0 :Text;
}

struct SetAggregateArchiveRequest {
  token @0 :Text;
  aggregateType @1 :Text;
  aggregateId @2 :Text;
  archived @3 :Bool;
  note @4 :Text;
  hasNote @5 :Bool;
}

struct SetAggregateArchiveResponse {
  aggregateJson @0 :Text;
}

struct CreateSnapshotRequest {
  token @0 :Text;
  aggregateType @1 :Text;
  aggregateId @2 :Text;
  comment @3 :Text;
  hasComment @4 :Bool;
}

struct CreateSnapshotResponse {
  snapshotJson @0 :Text;
}

struct PublishTarget {
  plugin @0 :Text;
  mode @1 :Text;
  hasMode @2 :Bool;
  priority @3 :Text;
  hasPriority @4 :Bool;
}

struct ListSnapshotsRequest {
  token @0 :Text;
  aggregateType @1 :Text;
  aggregateId @2 :Text;
  hasAggregateType @3 :Bool;
  hasAggregateId @4 :Bool;
  version @5 :UInt64;
  hasVersion @6 :Bool;
}

struct ListSnapshotsResponse {
  snapshotsJson @0 :Text;
}

struct GetSnapshotRequest {
  token @0 :Text;
  snapshotId @1 :UInt64;
}

struct GetSnapshotResponse {
  found @0 :Bool;
  snapshotJson @1 :Text;
}

struct ControlError {
  code @0 :Text;
  message @1 :Text;
}
