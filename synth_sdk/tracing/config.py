import json
from enum import Enum
from typing import Dict, List, Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)  # Changed from BatchSpanProcessor
from pydantic import BaseModel, Field


class InMemoryExporter(SpanExporter):
    def __init__(self):
        self.spans: List[Dict] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            self.spans.append(
                {
                    "name": span.name,
                    "context": {
                        "trace_id": span.context.trace_id,
                        "span_id": span.context.span_id,
                    },
                    "parent_id": span.parent.span_id if span.parent else None,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "attributes": dict(span.attributes),
                    "events": [
                        {
                            "name": event.name,
                            "timestamp": event.timestamp,
                            "attributes": dict(event.attributes),
                        }
                        for event in span.events
                    ],
                }
            )
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def get_spans(self) -> List[Dict]:
        return self.spans

    def clear(self):
        self.spans = []

    def to_json(self) -> str:
        return json.dumps(self.spans, default=str)


# Initialize the custom exporter
in_memory_exporter = InMemoryExporter()

# Set up the tracer provider
tracer_provider = TracerProvider()
span_processor = SimpleSpanProcessor(
    in_memory_exporter
)  # Use SimpleSpanProcessor for immediate exporting
tracer_provider.add_span_processor(span_processor)
trace.set_tracer_provider(tracer_provider)

# Get a tracer
tracer = trace.get_tracer(__name__)


def shutdown_tracer_provider():
    tracer_provider.shutdown()


# Valid types for tracking
VALID_TYPES = (str, dict, int, float, bool, list, type(None), BaseModel)


class LoggingMode(Enum):
    INSTANT = "instant"
    DEFERRED = "deferred"


class TracingConfig(BaseModel):
    mode: LoggingMode = Field(default=LoggingMode.DEFERRED)
    api_key: str
    base_url: str = Field(default="https://agent-learning.onrender.com")
    max_retries: int = Field(default=3)
    retry_backoff: float = Field(default=1.5)  # exponential backoff multiplier
    batch_size: int = Field(default=1)  # for future batching support
    timeout: float = Field(default=5.0)  # seconds
    sdk_version: str = Field(default="0.1.0")  # Added sdk_version field

    # Connection settings
    max_connections: int = Field(
        default=100, gt=0, description="Maximum number of concurrent connections"
    )
    keepalive_expiry: float = Field(
        default=30.0, gt=0, description="Connection keepalive time in seconds"
    )

    class Config:
        """Pydantic model configuration"""

        validate_assignment = True
        extra = "forbid"  # Prevent additional fields
