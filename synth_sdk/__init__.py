"""Synth SDK initialization file"""

# Import version from package metadata
from importlib import metadata

try:
    __version__ = metadata.version("synth-sdk")
except metadata.PackageNotFoundError:
    __version__ = "unknown"
# You can also add other package-level imports and initialization here

from synth_sdk.tracing.decorators import trace_system_sync, trace_system_async, synth_tracker_sync, synth_tracker_async
from synth_sdk.tracing.trackers import SynthTrackerSync, SynthTrackerAsync
from synth_sdk.provider_support.openai_lf import AsyncOpenAI, AsyncAzureOpenAI, OpenAI, AzureOpenAI