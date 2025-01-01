import copy
from inspect import isclass
import types
import logging

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import openai.resources
from openai._types import NotGiven
from packaging.version import Version
from wrapt import wrap_function_wrapper

from langfuse import Langfuse
from langfuse.client import StatefulGenerationClient
from langfuse.decorators import langfuse_context
from langfuse.utils import _get_timestamp
from langfuse.utils.langfuse_singleton import LangfuseSingleton
from synth_sdk.tracing.trackers import synth_tracker_sync, synth_tracker_async
from pydantic import BaseModel
from synth_sdk.tracing.abstractions import MessageInputs
from synth_sdk.provider_support.suppress_logging import *

try:
    import openai
except ImportError:
    raise ModuleNotFoundError(
        "Please install OpenAI to use this feature: 'pip install openai'"
    )

# CREDIT TO LANGFUSE FOR OPEN-SOURCING THE CODE THAT THIS IS BASED ON
# USING WITH MIT LICENSE PERMISSION
# https://langfuse.com

try:
    from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI  # noqa: F401
except ImportError:
    AsyncAzureOpenAI = None
    AsyncOpenAI = None
    AzureOpenAI = None
    OpenAI = None

#log = logging.getLogger("langfuse")

# Add logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to see all messages


@dataclass
class OpenAiDefinition:
    module: str
    object: str
    method: str
    type: str
    sync: bool
    min_version: Optional[str] = None


OPENAI_METHODS_V0 = [
    OpenAiDefinition(
        module="openai",
        object="ChatCompletion",
        method="create",
        type="chat",
        sync=True,
    ),
    OpenAiDefinition(
        module="openai",
        object="Completion",
        method="create",
        type="completion",
        sync=True,
    ),
]


OPENAI_METHODS_V1 = [
    OpenAiDefinition(
        module="openai.resources.chat.completions",
        object="Completions",
        method="create",
        type="chat",
        sync=True,
    ),
    OpenAiDefinition(
        module="openai.resources.completions",
        object="Completions",
        method="create",
        type="completion",
        sync=True,
    ),
    OpenAiDefinition(
        module="openai.resources.chat.completions",
        object="AsyncCompletions",
        method="create",
        type="chat",
        sync=False,
    ),
    OpenAiDefinition(
        module="openai.resources.completions",
        object="AsyncCompletions",
        method="create",
        type="completion",
        sync=False,
    ),
    OpenAiDefinition(
        module="openai.resources.beta.chat.completions",
        object="Completions",
        method="parse",
        type="chat",
        sync=True,
        min_version="1.50.0",
    ),
    OpenAiDefinition(
        module="openai.resources.beta.chat.completions",
        object="AsyncCompletions",
        method="parse",
        type="chat",
        sync=False,
        min_version="1.50.0",
    ),
]


class OpenAiArgsExtractor:
    def __init__(
        self,
        name=None,
        metadata=None,
        trace_id=None,
        session_id=None,
        user_id=None,
        tags=None,
        parent_observation_id=None,
        langfuse_prompt=None,  # we cannot use prompt because it's an argument of the old OpenAI completions API
        **kwargs,
    ):
        #logger.debug(f"OpenAiArgsExtractor initialized with kwargs: {kwargs}")
        #raise NotImplementedError("This method is not implemented yet")
        self.args = {}
        self.args["name"] = name
        self.args["metadata"] = (
            metadata
            if "response_format" not in kwargs
            else {
                **(metadata or {}),
                "response_format": kwargs["response_format"].model_json_schema()
                if isclass(kwargs["response_format"])
                and issubclass(kwargs["response_format"], BaseModel)
                else kwargs["response_format"],
            }
        )
        self.args["trace_id"] = trace_id
        self.args["session_id"] = session_id
        self.args["user_id"] = user_id
        self.args["tags"] = tags
        self.args["parent_observation_id"] = parent_observation_id
        self.args["langfuse_prompt"] = langfuse_prompt
        self.kwargs = kwargs

    def get_langfuse_args(self):
        return {**self.args, **self.kwargs}

    def get_openai_args(self):
        return self.kwargs


def _langfuse_wrapper(func):
    def _with_langfuse(open_ai_definitions, initialize):
        def wrapper(wrapped, instance, args, kwargs):
            return func(open_ai_definitions, initialize, wrapped, args, kwargs)

        return wrapper

    return _with_langfuse


def _extract_chat_prompt(kwargs: any):
    """Extracts the user input from prompts. Returns an array of messages or dict with messages and functions"""
    prompt = {}

    if kwargs.get("functions") is not None:
        prompt.update({"functions": kwargs["functions"]})

    if kwargs.get("function_call") is not None:
        prompt.update({"function_call": kwargs["function_call"]})

    if kwargs.get("tools") is not None:
        prompt.update({"tools": kwargs["tools"]})

    if prompt:
        # uf user provided functions, we need to send these together with messages to langfuse
        prompt.update(
            {
                "messages": _filter_image_data(kwargs.get("messages", [])),
            }
        )
        return prompt
    else:
        # vanilla case, only send messages in openai format to langfuse
        return _filter_image_data(kwargs.get("messages", []))


def _extract_chat_response(kwargs: any):
    """Extracts the llm output from the response."""
    response = {
        "role": kwargs.get("role", None),
    }

    if kwargs.get("function_call") is not None:
        response.update({"function_call": kwargs["function_call"]})

    if kwargs.get("tool_calls") is not None:
        response.update({"tool_calls": kwargs["tool_calls"]})

    response.update(
        {
            "content": kwargs.get("content", None),
        }
    )
    return response


def _get_langfuse_data_from_kwargs(
    resource: OpenAiDefinition, langfuse: Langfuse, start_time, kwargs
):
    name = kwargs.get("name", "OpenAI-generation")

    if name is None:
        name = "OpenAI-generation"

    if name is not None and not isinstance(name, str):
        raise TypeError("name must be a string")

    decorator_context_observation_id = langfuse_context.get_current_observation_id()
    decorator_context_trace_id = langfuse_context.get_current_trace_id()

    trace_id = kwargs.get("trace_id", None) or decorator_context_trace_id
    if trace_id is not None and not isinstance(trace_id, str):
        raise TypeError("trace_id must be a string")

    session_id = kwargs.get("session_id", None)
    if session_id is not None and not isinstance(session_id, str):
        raise TypeError("session_id must be a string")

    user_id = kwargs.get("user_id", None)
    if user_id is not None and not isinstance(user_id, str):
        raise TypeError("user_id must be a string")

    tags = kwargs.get("tags", None)
    if tags is not None and (
        not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags)
    ):
        raise TypeError("tags must be a list of strings")

    # Update trace params in decorator context if specified in openai call
    if decorator_context_trace_id:
        langfuse_context.update_current_trace(
            session_id=session_id, user_id=user_id, tags=tags
        )

    parent_observation_id = kwargs.get("parent_observation_id", None) or (
        decorator_context_observation_id
        if decorator_context_observation_id != decorator_context_trace_id
        else None
    )
    if parent_observation_id is not None and not isinstance(parent_observation_id, str):
        raise TypeError("parent_observation_id must be a string")
    if parent_observation_id is not None and trace_id is None:
        raise ValueError("parent_observation_id requires trace_id to be set")

    metadata = kwargs.get("metadata", {})

    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")

    model = kwargs.get("model", None) or None

    prompt = None

    if resource.type == "completion":
        prompt = kwargs.get("prompt", None)
    elif resource.type == "chat":
        prompt = _extract_chat_prompt(kwargs)

    is_nested_trace = False
    if trace_id:
        is_nested_trace = True
        langfuse.trace(id=trace_id, session_id=session_id, user_id=user_id, tags=tags)
    else:
        trace_id = (
            decorator_context_trace_id
            or langfuse.trace(
                session_id=session_id,
                user_id=user_id,
                tags=tags,
                name=name,
                input=prompt,
                metadata=metadata,
            ).id
        )

    parsed_temperature = (
        kwargs.get("temperature", 1)
        if not isinstance(kwargs.get("temperature", 1), NotGiven)
        else 1
    )

    parsed_max_tokens = (
        kwargs.get("max_tokens", float("inf"))
        if not isinstance(kwargs.get("max_tokens", float("inf")), NotGiven)
        else float("inf")
    )

    parsed_top_p = (
        kwargs.get("top_p", 1)
        if not isinstance(kwargs.get("top_p", 1), NotGiven)
        else 1
    )

    parsed_frequency_penalty = (
        kwargs.get("frequency_penalty", 0)
        if not isinstance(kwargs.get("frequency_penalty", 0), NotGiven)
        else 0
    )

    parsed_presence_penalty = (
        kwargs.get("presence_penalty", 0)
        if not isinstance(kwargs.get("presence_penalty", 0), NotGiven)
        else 0
    )

    parsed_seed = (
        kwargs.get("seed", None)
        if not isinstance(kwargs.get("seed", None), NotGiven)
        else None
    )

    modelParameters = {
        "temperature": parsed_temperature,
        "max_tokens": parsed_max_tokens,  # casing?
        "top_p": parsed_top_p,
        "frequency_penalty": parsed_frequency_penalty,
        "presence_penalty": parsed_presence_penalty,
    }
    if parsed_seed is not None:
        modelParameters["seed"] = parsed_seed

    langfuse_prompt = kwargs.get("langfuse_prompt", None)

    return {
        "name": name,
        "metadata": metadata,
        "trace_id": trace_id,
        "parent_observation_id": parent_observation_id,
        "user_id": user_id,
        "start_time": start_time,
        "input": prompt,
        "model_parameters": modelParameters,
        "model": model or None,
        "prompt": langfuse_prompt,
    }, is_nested_trace


def _create_langfuse_update(
    completion,
    generation: StatefulGenerationClient,
    completion_start_time,
    model=None,
    usage=None,
):
    update = {
        "end_time": _get_timestamp(),
        "output": completion,
        "completion_start_time": completion_start_time,
    }
    if model is not None:
        update["model"] = model

    if usage is not None:
        update["usage"] = usage

    generation.update(**update)


def _extract_streamed_openai_response(resource, chunks):
    #logger.debug(f"Extracting streamed response for resource type: {resource.type}")
    #logger.debug(f"Number of chunks: {len(chunks)}")
    completion = defaultdict(str) if resource.type == "chat" else ""
    model = None

    for chunk in chunks:
        if _is_openai_v1():
            chunk = chunk.__dict__
        #logger.debug(f"Processing chunk: {chunk}")

        model = model or chunk.get("model", None) or None
        usage = chunk.get("usage", None)
        choices = chunk.get("choices", [])
        #logger.debug(f"Extracted - model: {model}, choices: {choices}")

    #logger.debug(f"Final completion: {completion}")
    return model, completion, usage


def _get_langfuse_data_from_default_response(resource: OpenAiDefinition, response):
    if response is None:
        return None, "<NoneType response returned from OpenAI>", None

    model = response.get("model", None) or None

    completion = None
    if resource.type == "completion":
        choices = response.get("choices", [])
        if len(choices) > 0:
            choice = choices[-1]

            completion = choice.text if _is_openai_v1() else choice.get("text", None)
    elif resource.type == "chat":
        choices = response.get("choices", [])
        if len(choices) > 0:
            choice = choices[-1]
            completion = (
                _extract_chat_response(choice.message.__dict__)
                if _is_openai_v1()
                else choice.get("message", None)
            )

    usage = response.get("usage", None)

    return (
        model,
        completion,
        usage.__dict__ if _is_openai_v1() and usage is not None else usage,
    )


def _is_openai_v1():
    return Version(openai.__version__) >= Version("1.0.0")


def _is_streaming_response(response):
    return (
        isinstance(response, types.GeneratorType)
        or isinstance(response, types.AsyncGeneratorType)
        or (_is_openai_v1() and isinstance(response, openai.Stream))
        or (_is_openai_v1() and isinstance(response, openai.AsyncStream))
    )


@_langfuse_wrapper
def _wrap(open_ai_resource: OpenAiDefinition, initialize, wrapped, args, kwargs):
    new_langfuse: Langfuse = initialize()

    start_time = _get_timestamp()
    arg_extractor = OpenAiArgsExtractor(*args, **kwargs)

    generation, is_nested_trace = _get_langfuse_data_from_kwargs(
        open_ai_resource, new_langfuse, start_time, arg_extractor.get_langfuse_args()
    )
    generation = new_langfuse.generation(**generation)
    try:
        openai_response = wrapped(**arg_extractor.get_openai_args())

        if _is_streaming_response(openai_response):
            return LangfuseResponseGeneratorSync(
                resource=open_ai_resource,
                response=openai_response,
                generation=generation,
                langfuse=new_langfuse,
                is_nested_trace=is_nested_trace,
                kwargs=arg_extractor.get_openai_args()
            )

        else:
            model, completion, usage = _get_langfuse_data_from_default_response(
                open_ai_resource,
                (openai_response and openai_response.__dict__)
                if _is_openai_v1()
                else openai_response,
            )

            # Collect messages
            if open_ai_resource.type == "completion":
                user_prompt = arg_extractor.get_openai_args().get("prompt", "")
                messages = [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": completion},
                ]
                message_input = MessageInputs(messages=messages)
            elif open_ai_resource.type == "chat":
                messages = arg_extractor.get_openai_args().get("messages", [])
                messages.append({"role": "assistant", "content": completion["content"]})
                message_input = MessageInputs(messages=messages)
            else:
                message_input = MessageInputs(messages=[])

            # Use track_lm
            synth_tracker_sync.track_lm(
                messages=message_input.messages,
                model_name=model,
                finetune=False
            )

            generation.update(
                model=model, output=completion, end_time=_get_timestamp(), usage=usage
            )

            # Avoiding the trace-update if trace-id is provided by user.
            if not is_nested_trace:
                new_langfuse.trace(id=generation.trace_id, output=completion)

        return openai_response
    except Exception as ex:
        #log.warning(ex)
        model = kwargs.get("model", None) or None
        generation.update(
            end_time=_get_timestamp(),
            status_message=str(ex),
            level="ERROR",
            model=model,
            usage={"input_cost": 0, "output_cost": 0, "total_cost": 0},
        )
        raise ex


@_langfuse_wrapper
async def _wrap_async(
    open_ai_resource: OpenAiDefinition, initialize, wrapped, args, kwargs
):
    new_langfuse = initialize()
    start_time = _get_timestamp()
    arg_extractor = OpenAiArgsExtractor(*args, **kwargs)

    generation, is_nested_trace = _get_langfuse_data_from_kwargs(
        open_ai_resource, new_langfuse, start_time, arg_extractor.get_langfuse_args()
    )
    generation = new_langfuse.generation(**generation)
    try:
        openai_response = await wrapped(**arg_extractor.get_openai_args())

        if _is_streaming_response(openai_response):
            return LangfuseResponseGeneratorAsync(
                resource=open_ai_resource,
                response=openai_response,
                generation=generation,
                langfuse=new_langfuse,
                is_nested_trace=is_nested_trace,
                kwargs=arg_extractor.get_openai_args()
            )

        else:
            model, completion, usage = _get_langfuse_data_from_default_response(
                open_ai_resource,
                (openai_response and openai_response.__dict__)
                if _is_openai_v1()
                else openai_response,
            )

            # Collect messages
            if open_ai_resource.type == "completion":
                user_prompt = arg_extractor.get_openai_args().get("prompt", "")
                messages = [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": completion},
                ]
                message_input = MessageInputs(messages=messages)
            elif open_ai_resource.type == "chat":
                #THIS IS DANGEROUSs
                messages = arg_extractor.get_openai_args().get("messages", [])
                messages.append({"role": "assistant", "content": completion["content"]})
                message_input = MessageInputs(messages=messages)
            else:
                message_input = MessageInputs(messages=[])

            # Use track_lm
            synth_tracker_async.track_lm(
                messages=message_input.messages,
                model_name=model,
                finetune=False
            )

            generation.update(
                model=model,
                output=completion,
                end_time=_get_timestamp(),
                usage=usage,
            )
            # Avoiding the trace-update if trace-id is provided by user.
            if not is_nested_trace:
                new_langfuse.trace(id=generation.trace_id, output=completion)

        return openai_response
    except Exception as ex:
        model = kwargs.get("model", None) or None
        generation.update(
            end_time=_get_timestamp(),
            status_message=str(ex),
            level="ERROR",
            model=model,
            usage={"input_cost": 0, "output_cost": 0, "total_cost": 0},
        )
        raise ex


class OpenAILangfuse:
    _langfuse: Optional[Langfuse] = None

    def initialize(self):
        self._langfuse = LangfuseSingleton().get(
            public_key=openai.langfuse_public_key,
            secret_key=openai.langfuse_secret_key,
            host=openai.langfuse_host,
            debug=openai.langfuse_debug,
            enabled=openai.langfuse_enabled,
            sdk_integration="openai",
            sample_rate=openai.langfuse_sample_rate,
        )

        return self._langfuse

    def flush(cls):
        cls._langfuse.flush()

    def langfuse_auth_check(self):
        """Check if the provided Langfuse credentials (public and secret key) are valid.

        Raises:
            Exception: If no projects were found for the provided credentials.

        Note:
            This method is blocking. It is discouraged to use it in production code.
        """
        if self._langfuse is None:
            self.initialize()

        return self._langfuse.auth_check()

    def register_tracing(self):
        resources = OPENAI_METHODS_V1 if _is_openai_v1() else OPENAI_METHODS_V0

        for resource in resources:
            if resource.min_version is not None and Version(
                openai.__version__
            ) < Version(resource.min_version):
                continue

            wrap_function_wrapper(
                resource.module,
                f"{resource.object}.{resource.method}",
                _wrap(resource, self.initialize)
                if resource.sync
                else _wrap_async(resource, self.initialize),
            )

        setattr(openai, "langfuse_public_key", None)
        setattr(openai, "langfuse_secret_key", None)
        setattr(openai, "langfuse_host", None)
        setattr(openai, "langfuse_debug", None)
        setattr(openai, "langfuse_enabled", True)
        setattr(openai, "langfuse_sample_rate", None)
        setattr(openai, "langfuse_mask", None)
        setattr(openai, "langfuse_auth_check", self.langfuse_auth_check)
        setattr(openai, "flush_langfuse", self.flush)


modifier = OpenAILangfuse()
modifier.register_tracing()


# DEPRECATED: Use `openai.langfuse_auth_check()` instead
def auth_check():
    if modifier._langfuse is None:
        modifier.initialize()

    return modifier._langfuse.auth_check()


def _filter_image_data(messages: List[dict]):
    """https://platform.openai.com/docs/guides/vision?lang=python

    The messages array remains the same, but the 'image_url' is removed from the 'content' array.
    It should only be removed if the value starts with 'data:image/jpeg;base64,'

    """
    output_messages = copy.deepcopy(messages)

    for message in output_messages:
        content = (
            message.get("content", None)
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )

        if content is not None:
            for index, item in enumerate(content):
                if isinstance(item, dict) and item.get("image_url", None) is not None:
                    url = item["image_url"]["url"]
                    if url.startswith("data:image/"):
                        del content[index]["image_url"]

    return output_messages


class LangfuseResponseGeneratorSync:
    def __init__(
        self,
        *,
        resource,
        response,
        generation,
        langfuse,
        is_nested_trace,
        kwargs,
    ):
        self.items = []
        self.resource = resource
        self.response = response
        self.generation = generation
        self.langfuse = langfuse
        self.is_nested_trace = is_nested_trace
        self.kwargs = kwargs
        self.completion_start_time = None

    def __iter__(self):
        try:
            for i in self.response:
                self.items.append(i)

                if self.completion_start_time is None:
                    self.completion_start_time = _get_timestamp()

                yield i
        finally:
            self._finalize()

    def __next__(self):
        try:
            item = self.response.__next__()
            self.items.append(item)

            if self.completion_start_time is None:
                self.completion_start_time = _get_timestamp()

            return item

        except StopIteration:
            self._finalize()

            raise

    def __enter__(self):
        return self.__iter__()

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _finalize(self):
        model, completion, usage = _extract_streamed_openai_response(
            self.resource, self.items
        )

        # Collect messages
        if self.resource.type == "completion":
            user_prompt = self.kwargs.get("prompt", "")
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": completion},
            ]
            message_input = MessageInputs(messages=messages)
        elif self.resource.type == "chat":
            messages = self.kwargs.get("messages", [])
            messages.append({"role": "assistant", "content": completion["content"]})
            message_input = MessageInputs(messages=messages)
        else:
            message_input = MessageInputs(messages=[])

        # Use track_lm
        synth_tracker_sync.track_lm(
            messages=message_input.messages,
            model_name=model,
            finetune=False
        )

        # Avoiding the trace-update if trace-id is provided by user.
        if not self.is_nested_trace:
            self.langfuse.trace(id=self.generation.trace_id, output=completion)

        _create_langfuse_update(
            completion,
            self.generation,
            self.completion_start_time,
            model=model,
            usage=usage,
        )


class LangfuseResponseGeneratorAsync:
    def __init__(
        self,
        *,
        resource,
        response,
        generation,
        langfuse,
        is_nested_trace,
        kwargs,
    ):
        #logger.debug(f"LangfuseResponseGeneratorAsync initialized with kwargs: {kwargs}")
        #logger.debug(f"Resource type: {resource.type}")
        self.items = []
        self.resource = resource
        self.response = response
        self.generation = generation
        self.langfuse = langfuse
        self.is_nested_trace = is_nested_trace
        self.kwargs = kwargs
        self.completion_start_time = None

    async def __aiter__(self):
        try:
            async for i in self.response:
                self.items.append(i)

                if self.completion_start_time is None:
                    self.completion_start_time = _get_timestamp()

                yield i
        finally:
            await self._finalize()

    async def __anext__(self):
        try:
            item = await self.response.__anext__()
            self.items.append(item)

            if self.completion_start_time is None:
                self.completion_start_time = _get_timestamp()

            return item

        except StopAsyncIteration:
            await self._finalize()

            raise

    async def __aenter__(self):
        return self.__aiter__()

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass

    async def _finalize(self):
        #logger.debug("Starting _finalize in LangfuseResponseGeneratorAsync")
        model, completion, usage = _extract_streamed_openai_response(
            self.resource, self.items
        )
        #logger.debug(f"Extracted response - model: {model}, completion type: {type(completion)}")
        #logger.debug(f"Completion content: {completion}")

        # Collect messages
        #logger.debug(f"Resource type: {self.resource.type}")
        if self.resource.type == "completion":
            #logger.debug("Processing completion type response")
            user_prompt = self.kwargs.get("prompt", "")
            #logger.debug(f"User prompt: {user_prompt}")
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": completion},
            ]
            message_input = MessageInputs(messages=messages)
            #logger.debug(f"Created message input for completion: {messages}")
        elif self.resource.type == "chat":
            #logger.debug("Processing chat type response")
            messages = self.kwargs.get("messages", [])
            #logger.debug(f"Original messages: {messages}")
            
            if isinstance(completion, dict) and "content" in completion:
                messages.append({"role": "assistant", "content": completion["content"]})
                #logger.debug(f"Appended assistant message: {completion['content']}")
            else:
                logger.warning(f"Unexpected completion format: {completion}")
            
            message_input = MessageInputs(messages=messages)
            #logger.debug(f"Final message input for chat: {message_input.messages}")
        else:
            logger.error(f"Unknown resource type: {self.resource.type}")
            message_input = MessageInputs(messages=[])
        # Use track_lm
        #logger.debug(f"About to track_lm with model: {model}")
        #logger.debug(f"Messages to track: {message_input.messages}")
        synth_tracker_async.track_lm(
            messages=message_input.messages,
            model_name=model,
            finetune=False
        )
        #logger.debug("Completed track_lm call")

        # Avoiding the trace-update if trace-id is provided by user.
        if not self.is_nested_trace:
            self.langfuse.trace(id=self.generation.trace_id, output=completion)

        _create_langfuse_update(
            completion,
            self.generation,
            self.completion_start_time,
            model=model,
            usage=usage,
        )

    async def close(self) -> None:
        """Close the response and release the connection.

        Automatically called if the response body is read to completion.
        """
        await self.response.close()
