"""
Synth SDK - A Python SDK for building and managing AI agents
"""

__version__ = "0.3.0"

# For development versions, uncomment and modify this line:
# __version__ = "0.3.0.dev0"

from .tracing import *  # noqa

# Import version from package metadata
from importlib import metadata

try:
    __version__ = metadata.version("synth-sdk")
except metadata.PackageNotFoundError:
    __version__ = "unknown"
# You can also add other package-level imports and initialization here

from synth_sdk.tracing.decorators import (
    trace_event_sync,
    trace_event_async,
    synth_tracker_sync,
    synth_tracker_async,
)
from synth_sdk.tracing.trackers import (
    track_messages_async,
    track_messages_sync,
)  # SynthTrackerSync, SynthTrackerAsync
from synth_sdk.provider_support.openai import (
    AsyncOpenAI,
    AsyncAzureOpenAI,
    OpenAI,
    AzureOpenAI,
)
from synth_sdk.provider_support.anthropic import AsyncAnthropic, Anthropic
from synth_sdk.tracing.upload import upload
