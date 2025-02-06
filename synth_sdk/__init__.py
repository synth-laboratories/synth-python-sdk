"""
Synth SDK - A Python SDK for building and managing AI agents
"""

from importlib.metadata import version

__version__ = version("synth-sdk")

from .tracing import *  # noqa

# You can also add other package-level imports and initialization here
from synth_sdk.tracing.abstractions import EventPartitionElement, SystemTrace, TrainingQuestion, RewardSignal
from synth_sdk.tracing.decorators import trace_event_async,trace_event_sync
from synth_sdk.provider_support.openai import OpenAI, AsyncOpenAI
from synth_sdk.provider_support.anthropic import Anthropic, AsyncAnthropic