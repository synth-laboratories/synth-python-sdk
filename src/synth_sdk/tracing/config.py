from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # Changed from BatchSpanProcessor
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from typing import Sequence, Dict, List
import json

class InMemoryExporter(SpanExporter):
    def __init__(self):
        self.spans: List[Dict] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            #print(f"Exporting span: {span.name}")  # Added print statement
            self.spans.append({
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
                        "attributes": dict(event.attributes)
                    } for event in span.events
                ],
            })
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
span_processor = SimpleSpanProcessor(in_memory_exporter)  # Use SimpleSpanProcessor for immediate exporting
tracer_provider.add_span_processor(span_processor)
trace.set_tracer_provider(tracer_provider)

# Get a tracer
tracer = trace.get_tracer(__name__)

def shutdown_tracer_provider():
    tracer_provider.shutdown()