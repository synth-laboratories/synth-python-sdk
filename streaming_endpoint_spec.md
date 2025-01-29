# Streaming vs Bulk Endpoint Specification

## Current Bulk Endpoint

```
POST /v1/traces/upload
Content-Type: application/json
Authorization: Bearer <api_key>

{
    "dataset": {
        "questions": [
            {
                "id": string,
                "intent": string,
                "criteria": string
            }
        ],
        "reward_signals": [
            {
                "question_id": string,
                "system_instance_id": string,
                "reward": float
            }
        ]
    },
    "traces": [
        {
            "system_name": string,
            "system_id": string,
            "system_instance_id": string,
            "partition": [
                {
                    "partition_index": int,
                    "events": [
                        {
                            "event_type": string,
                            "opened": float,
                            "closed": float,
                            "partition_index": int,
                            "agent_compute_step": ComputeStep,
                            "environment_compute_steps": [ComputeStep]
                        }
                    ]
                }
            ]
        }
    ]
}
```

## New Streaming Endpoint

```
POST /v1/uploads/stream
Content-Type: application/json
Authorization: Bearer <api_key>

{
    "event": {
        "event_type": string,
        "opened": float,
        "closed": float,
        "partition_index": int,
        "agent_compute_step": ComputeStep,
        "environment_compute_steps": [ComputeStep]
    },
    "system_info": {
        "system_name": string,
        "system_id": string,
        "system_instance_id": string
    },
    "timestamp": float,  // When the event was sent
    "sdk_version": string  // Version of the SDK
}
```

## Key Differences

1. **Granularity**:
   - Bulk: Uploads complete traces with multiple events and dataset information
   - Stream: Sends individual events as they occur

2. **Dataset Association**:
   - Bulk: Dataset is included in the upload
   - Stream: Dataset association happens later during reward signal upload

3. **Timing**:
   - Bulk: Events are sent after completion
   - Stream: Events are sent immediately when they close

4. **Payload Size**:
   - Bulk: Large payloads containing multiple events
   - Stream: Small payloads with single events

## Streaming Endpoint Specification

### Request

- **Method**: POST
- **Path**: /v1/uploads/stream
- **Headers**:
  - Content-Type: application/json
  - Authorization: Bearer <api_key>

### Response

```
200 OK
{
    "success": true,
    "event_id": string  // Unique ID assigned to the event
}

400 Bad Request
{
    "error": string,
    "code": string,
    "details": object
}

429 Too Many Requests
{
    "error": "Rate limit exceeded",
    "retry_after": float  // Seconds to wait
}
```

### Rate Limits
- Maximum 100 requests per minute per API key
- Burst limit of 20 requests per second

### Error Handling
1. **Invalid Event Structure**: 400 Bad Request
2. **Rate Limiting**: 429 Too Many Requests with retry_after
3. **Server Errors**: 500 with automatic client retry
4. **Authentication**: 401 Unauthorized

### Implementation Requirements

1. **Event Order Preservation**:
   - Events must be processed in order within each system_instance_id
   - Partition indices must be sequential

2. **Data Consistency**:
   - Events must be associated with existing system_instance_ids
   - Duplicate event detection based on (system_instance_id, event_type, opened)

3. **Performance**:
   - Maximum processing time: 500ms per event
   - 99th percentile latency under 1s

4. **Storage**:
   - Events must be immediately persisted
   - Support for later association with datasets
   - Efficient querying by system_instance_id

5. **Monitoring**:
   - Track success/failure rates
   - Monitor processing latency
   - Alert on high error rates

### Security Requirements

1. **Authentication**:
   - API key validation
   - Rate limiting per API key
   - Scope validation

2. **Data Validation**:
   - Schema validation
   - Timestamp validation (within reasonable range)
   - System info validation

3. **Audit Trail**:
   - Log all streaming attempts
   - Track failed attempts
   - Record IP addresses

### Migration Strategy

1. **Phase 1**: Add streaming endpoint alongside bulk
2. **Phase 2**: Monitor streaming usage and performance
3. **Phase 3**: Gradually transition clients to streaming
4. **Phase 4**: Mark bulk endpoint as legacy

### Client SDK Requirements

1. **Retry Logic**:
   - Exponential backoff
   - Maximum retry attempts
   - Fallback to event store

2. **Batching**:
   - Queue failed events
   - Batch retry of failed events
   - Configurable batch sizes

3. **Monitoring**:
   - Track success/failure rates
   - Measure latency
   - Log detailed errors 