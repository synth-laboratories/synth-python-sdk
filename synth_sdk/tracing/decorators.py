# synth_sdk/tracing/decorators.py
import inspect
import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Literal

from synth_sdk.tracing.abstractions import (
    AgentComputeStep,
    ArbitraryInputs,
    ArbitraryOutputs,
    EnvironmentComputeStep,
    Event,
    MessageInputs,
    MessageOutputs,
)
from synth_sdk.tracing.config import LoggingMode, TracingConfig
from synth_sdk.tracing.context import get_current_context, trace_context
from synth_sdk.tracing.events.manage import set_current_event
from synth_sdk.tracing.events.store import event_store
from synth_sdk.tracing.immediate_client import (
    AsyncImmediateLogClient,
    ImmediateLogClient,
)
from synth_sdk.tracing.local import (
    _local,
    active_events_var,
    logger,
)
from synth_sdk.tracing.retry_queue import initialize_retry_queue, retry_queue
from synth_sdk.tracing.trackers import (
    synth_tracker_async,
    synth_tracker_sync,
)
from synth_sdk.tracing.utils import get_system_id

logger = logging.getLogger(__name__)


def clear_current_event(event_type: str) -> None:
    """Clear the current event from the appropriate storage based on context.

    Args:
        event_type: The type of event to clear
    """
    try:
        # Check if we're in an async context
        import asyncio

        asyncio.get_running_loop()
        # We're in async context
        active_events = active_events_var.get()
        if event_type in active_events:
            del active_events[event_type]
            active_events_var.set(active_events)
    except RuntimeError:
        # We're in sync context
        if hasattr(_local, "active_events") and event_type in _local.active_events:
            del _local.active_events[event_type]


# Cache the config to avoid repeated env lookups
def get_tracing_config() -> TracingConfig:
    config = TracingConfig(
        mode=LoggingMode.INSTANT
        if os.getenv("SYNTH_LOGGING_MODE") == "instant"
        else LoggingMode.DEFERRED,
        api_key=os.getenv("SYNTH_API_KEY", ""),
        base_url=os.getenv(
            "SYNTH_ENDPOINT_OVERRIDE", "https://agent-learning.onrender.com"
        ),
    )
    # Initialize retry queue with config if needed
    initialize_retry_queue(config)
    return config


def process_retry_queue_sync() -> None:
    """Process the retry queue synchronously."""
    try:
        success, failure = retry_queue.process_sync()
        if success or failure:
            logger.info(f"Processed retry queue: {success} succeeded, {failure} failed")
    except Exception as e:
        logger.error(f"Error processing retry queue: {e}")


async def process_retry_queue_async() -> None:
    """Process the retry queue asynchronously."""
    try:
        success, failure = await retry_queue.process_async()
        if success or failure:
            logger.info(f"Processed retry queue: {success} succeeded, {failure} failed")
    except Exception as e:
        logger.error(f"Error processing retry queue: {e}")


# # This decorator is used to trace synchronous functions
def trace_system_sync(
    origin: Literal["agent", "environment"],
    event_type: str,
    log_result: bool = False,
    manage_event: Literal["create", "end", None] = None,
    increment_partition: bool = False,
    verbose: bool = False,
    finetune_step: bool = True,
) -> Callable:
    """Decorator for tracing synchronous functions.

    Purpose is to keep track of inputs and outputs for compute steps for sync functions.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine the instance (self) if it's a method
            if not hasattr(func, "__self__") or not func.__self__:
                if not args:
                    raise ValueError(
                        "Instance method expected, but no arguments were passed."
                    )
                self_instance = args[0]
            else:
                self_instance = func.__self__

            # Ensure required attributes are present
            required_attrs = ["system_instance_id", "system_name"]
            for attr in required_attrs:
                if not hasattr(self_instance, attr):
                    raise ValueError(
                        f"Instance of class '{self_instance.__class__.__name__}' missing required attribute '{attr}'"
                    )

            # Use context manager for setup/cleanup
            with trace_context(
                system_name=self_instance.system_name,
                system_id=get_system_id(self_instance.system_name),
                system_instance_id=self_instance.system_instance_id,
            ):
                # Initialize Trace
                synth_tracker_sync.initialize()

                event = None
                compute_began = time.time()
                try:
                    if manage_event == "create":
                        # Create new event
                        context = get_current_context()
                        event = Event(
                            system_instance_id=context["system_instance_id"],
                            event_type=event_type,
                            opened=compute_began,
                            closed=None,
                            partition_index=0,
                            agent_compute_step=None,
                            environment_compute_steps=[],
                            system_name=context["system_name"],
                            system_id=context["system_id"],
                        )
                        if increment_partition:
                            event.partition_index = event_store.increment_partition(
                                context["system_name"],
                                context["system_id"],
                                context["system_instance_id"],
                            )
                            logger.debug(
                                f"Incremented partition to: {event.partition_index}"
                            )
                        set_current_event(event, decorator_type="sync")
                        logger.debug(f"Created and set new event: {event_type}")

                    # Automatically trace function inputs
                    bound_args = inspect.signature(func).bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    for param, value in bound_args.arguments.items():
                        if param == "self":
                            continue
                        synth_tracker_sync.track_state(
                            variable_name=param, variable_value=value, origin=origin
                        )

                    # Execute the function
                    result = func(*args, **kwargs)

                    # Automatically trace function output
                    track_result(result, synth_tracker_sync, origin)

                    # Collect traced inputs and outputs
                    traced_inputs, traced_outputs = synth_tracker_sync.get_traced_data()

                    compute_steps_by_origin: Dict[
                        Literal["agent", "environment"], Dict[str, List[Any]]
                    ] = {
                        "agent": {"inputs": [], "outputs": []},
                        "environment": {"inputs": [], "outputs": []},
                    }

                    # Organize traced data by origin
                    for item in traced_inputs:
                        var_origin = item["origin"]
                        if "variable_value" in item and "variable_name" in item:
                            # Standard variable input
                            compute_steps_by_origin[var_origin]["inputs"].append(
                                ArbitraryInputs(
                                    inputs={
                                        item["variable_name"]: item["variable_value"]
                                    }
                                )
                            )
                        elif "messages" in item:
                            # Message input from track_lm
                            compute_steps_by_origin[var_origin]["inputs"].append(
                                MessageInputs(messages=item["messages"])
                            )
                            compute_steps_by_origin[var_origin]["inputs"].append(
                                ArbitraryInputs(
                                    inputs={"model_name": item["model_name"]}
                                )
                            )
                            finetune = item["finetune"] or finetune_step
                            compute_steps_by_origin[var_origin]["inputs"].append(
                                ArbitraryInputs(inputs={"finetune": finetune})
                            )
                        else:
                            logger.warning(f"Unhandled traced input item: {item}")

                    for item in traced_outputs:
                        var_origin = item["origin"]
                        if "variable_value" in item and "variable_name" in item:
                            # Standard variable output
                            compute_steps_by_origin[var_origin]["outputs"].append(
                                ArbitraryOutputs(
                                    outputs={
                                        item["variable_name"]: item["variable_value"]
                                    }
                                )
                            )
                        elif "messages" in item:
                            # Message output from track_lm
                            compute_steps_by_origin[var_origin]["outputs"].append(
                                MessageOutputs(messages=item["messages"])
                            )
                        else:
                            logger.warning(f"Unhandled traced output item: {item}")

                    # Capture compute end time
                    compute_ended = time.time()

                    # Create compute steps grouped by origin
                    for var_origin in ["agent", "environment"]:
                        inputs = compute_steps_by_origin[var_origin]["inputs"]
                        outputs = compute_steps_by_origin[var_origin]["outputs"]
                        if inputs or outputs:
                            event_order = (
                                1 + len(event.environment_compute_steps) + 1
                                if event
                                else 1
                            )
                            compute_step = (
                                AgentComputeStep(
                                    event_order=event_order,
                                    compute_began=compute_began,
                                    compute_ended=compute_ended,
                                    compute_input=inputs,
                                    compute_output=outputs,
                                )
                                if var_origin == "agent"
                                else EnvironmentComputeStep(
                                    event_order=event_order,
                                    compute_began=compute_began,
                                    compute_ended=compute_ended,
                                    compute_input=inputs,
                                    compute_output=outputs,
                                )
                            )
                            if event:
                                if var_origin == "agent":
                                    event.agent_compute_step = compute_step
                                else:
                                    event.environment_compute_steps.append(compute_step)

                    # Optionally log the function result
                    if log_result:
                        logger.info(f"Function result: {result}")

                    # Handle event management after function execution
                    if manage_event == "end":
                        context = get_current_context()
                        current_event = _local.active_events.get(event_type)
                        if current_event:
                            current_event.closed = compute_ended

                            # Get the config to determine logging mode
                            config = get_tracing_config()

                            # If immediate logging is enabled, send the event now
                            if config.mode == LoggingMode.INSTANT:
                                client = ImmediateLogClient(config)
                                client.send_event(current_event, context)

                            # Always store in event_store as backup
                            event_store.add_event(
                                context["system_name"],
                                context["system_id"],
                                context["system_instance_id"],
                                current_event,
                            )
                            del _local.active_events[event_type]

                    # Process retry queue after successful execution
                    process_retry_queue_sync()

                    return result
                except Exception as e:
                    logger.error(f"Exception in traced function '{func.__name__}': {e}")
                    raise

        return wrapper

    return decorator


def trace_system_async(
    origin: Literal["agent", "environment"],
    event_type: str,
    log_result: bool = False,
    manage_event: Literal["create", "end", "create_and_end", "lazy_end", None] = None,
    increment_partition: bool = False,
    verbose: bool = False,
    finetune_step: bool = True,
) -> Callable:
    """Decorator for tracing asynchronous functions.

    Purpose is to keep track of inputs and outputs for compute steps for async functions.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Determine the instance (self) if it's a method
            if not hasattr(func, "__self__") or not func.__self__:
                if not args:
                    raise ValueError(
                        "Instance method expected, but no arguments were passed."
                    )
                self_instance = args[0]
            else:
                self_instance = func.__self__

            # Ensure required attributes are present
            required_attrs = ["system_instance_id", "system_name"]
            for attr in required_attrs:
                if not hasattr(self_instance, attr):
                    raise ValueError(
                        f"Instance of class '{self_instance.__class__.__name__}' missing required attribute '{attr}'"
                    )

            # Use context manager for setup/cleanup
            with trace_context(
                system_name=self_instance.system_name,
                system_id=get_system_id(self_instance.system_name),
                system_instance_id=self_instance.system_instance_id,
            ):
                # Initialize AsyncTrace
                synth_tracker_async.initialize()

                event = None
                compute_began = time.time()
                try:
                    if manage_event in ["create", "create_and_end"]:
                        # Create new event
                        context = get_current_context()
                        event = Event(
                            system_instance_id=context["system_instance_id"],
                            event_type=event_type,
                            opened=compute_began,
                            closed=None,
                            partition_index=0,
                            agent_compute_step=None,
                            environment_compute_steps=[],
                            system_name=context["system_name"],
                            system_id=context["system_id"],
                        )
                        if increment_partition:
                            event.partition_index = event_store.increment_partition(
                                context["system_name"],
                                context["system_id"],
                                context["system_instance_id"],
                            )
                            logger.debug(
                                f"Incremented partition to: {event.partition_index}"
                            )
                        set_current_event(event, decorator_type="async")
                        logger.debug(f"Created and set new event: {event_type}")

                    # Automatically trace function inputs
                    bound_args = inspect.signature(func).bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    for param, value in bound_args.arguments.items():
                        if param == "self":
                            continue
                        synth_tracker_async.track_state(
                            variable_name=param,
                            variable_value=value,
                            origin=origin,
                            io_type="input",
                        )

                    # Execute the coroutine
                    result = await func(*args, **kwargs)

                    # Automatically trace function output
                    track_result(result, synth_tracker_async, origin)

                    # Collect traced inputs and outputs
                    traced_inputs, traced_outputs = (
                        synth_tracker_async.get_traced_data()
                    )

                    compute_steps_by_origin: Dict[
                        Literal["agent", "environment"], Dict[str, List[Any]]
                    ] = {
                        "agent": {"inputs": [], "outputs": []},
                        "environment": {"inputs": [], "outputs": []},
                    }

                    # Organize traced data by origin
                    for item in traced_inputs:
                        var_origin = item["origin"]
                        if "variable_value" in item and "variable_name" in item:
                            # Standard variable input
                            compute_steps_by_origin[var_origin]["inputs"].append(
                                ArbitraryInputs(
                                    inputs={
                                        item["variable_name"]: item["variable_value"]
                                    }
                                )
                            )
                        elif "messages" in item:
                            # Message input from track_lm
                            compute_steps_by_origin[var_origin]["inputs"].append(
                                MessageInputs(messages=item["messages"])
                            )
                            compute_steps_by_origin[var_origin]["inputs"].append(
                                ArbitraryInputs(
                                    inputs={"model_name": item["model_name"]}
                                )
                            )
                            finetune = finetune_step or item["finetune"]
                            compute_steps_by_origin[var_origin]["inputs"].append(
                                ArbitraryInputs(inputs={"finetune": finetune})
                            )
                        else:
                            logger.warning(f"Unhandled traced input item: {item}")

                    for item in traced_outputs:
                        var_origin = item["origin"]
                        if "variable_value" in item and "variable_name" in item:
                            # Standard variable output
                            compute_steps_by_origin[var_origin]["outputs"].append(
                                ArbitraryOutputs(
                                    outputs={
                                        item["variable_name"]: item["variable_value"]
                                    }
                                )
                            )
                        elif "messages" in item:
                            # Message output from track_lm
                            compute_steps_by_origin[var_origin]["outputs"].append(
                                MessageOutputs(messages=item["messages"])
                            )
                        else:
                            logger.warning(f"Unhandled traced output item: {item}")

                    compute_ended = time.time()

                    # Create compute steps grouped by origin
                    for var_origin in ["agent", "environment"]:
                        inputs = compute_steps_by_origin[var_origin]["inputs"]
                        outputs = compute_steps_by_origin[var_origin]["outputs"]
                        if inputs or outputs:
                            event_order = (
                                1 + len(event.environment_compute_steps) + 1
                                if event
                                else 1
                            )
                            compute_step = (
                                AgentComputeStep(
                                    event_order=event_order,
                                    compute_began=compute_began,
                                    compute_ended=compute_ended,
                                    compute_input=inputs,
                                    compute_output=outputs,
                                )
                                if var_origin == "agent"
                                else EnvironmentComputeStep(
                                    event_order=event_order,
                                    compute_began=compute_began,
                                    compute_ended=compute_ended,
                                    compute_input=inputs,
                                    compute_output=outputs,
                                )
                            )
                            if event:
                                if var_origin == "agent":
                                    event.agent_compute_step = compute_step
                                else:
                                    event.environment_compute_steps.append(compute_step)

                    # Optionally log the function result
                    if log_result:
                        logger.info(f"Function result: {result}")

                    # Handle event management after function execution
                    if manage_event in ["end", "create_and_end"]:
                        context = get_current_context()
                        current_event = active_events_var.get().get(event_type)
                        if current_event:
                            current_event.closed = compute_ended

                            # Get the config to determine logging mode
                            config = get_tracing_config()

                            # If immediate logging is enabled, send the event now
                            if config.mode == LoggingMode.INSTANT:
                                client = AsyncImmediateLogClient(config)
                                await client.send_event(current_event, context)

                            # Always store in event_store as backup
                            event_store.add_event(
                                context["system_name"],
                                context["system_id"],
                                context["system_instance_id"],
                                current_event,
                            )
                            active_events = active_events_var.get()
                            del active_events[event_type]
                            active_events_var.set(active_events)

                    # Process retry queue after successful execution
                    await process_retry_queue_async()

                    return result
                except Exception as e:
                    logger.error(f"Exception in traced function '{func.__name__}': {e}")
                    raise

        return async_wrapper

    return decorator


def trace_system(
    origin: Literal["agent", "environment"],
    event_type: str,
    log_result: bool = False,
    manage_event: Literal["create", "end", "lazy_end", None] = None,
    increment_partition: bool = False,
    verbose: bool = False,
) -> Callable:
    """
    Decorator that chooses the correct tracing method (sync or async) based on
    whether the wrapped function is synchronous or asynchronous.

    Purpose is to keep track of inputs and outputs for compute steps for both sync and async functions.
    """

    def decorator(func: Callable) -> Callable:
        # Check if the function is async or sync
        if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
            # Use async tracing
            # logger.debug("Using async tracing")
            async_decorator = trace_system_async(
                origin,
                event_type,
                log_result,
                manage_event,
                increment_partition,
                verbose,
            )
            return async_decorator(func)
        else:
            # Use sync tracing
            # logger.debug("Using sync tracing")
            sync_decorator = trace_system_sync(
                origin,
                event_type,
                log_result,
                manage_event,
                increment_partition,
                verbose,
            )
            return sync_decorator(func)

    return decorator


def track_result(result, tracker, origin):
    # Helper function to track results, including tuple unpacking
    if isinstance(result, tuple):
        # Track each element of the tuple that matches valid types
        for i, item in enumerate(result):
            try:
                tracker.track_state(
                    variable_name=f"result_{i}", variable_value=item, origin=origin
                )
            except Exception as e:
                logger.warning(f"Could not track tuple element {i}: {str(e)}")
    else:
        # Track single result as before
        try:
            tracker.track_state(
                variable_name="result", variable_value=result, origin=origin
            )
        except Exception as e:
            logger.warning(f"Could not track result: {str(e)}")
