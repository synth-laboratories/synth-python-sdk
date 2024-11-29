# synth_sdk/tracing/decorators.py
from typing import Callable, Optional, Set, Literal, Any, Dict, Tuple, Union, List
from functools import wraps
import time
import logging

from synth_sdk.tracing.abstractions import (
    Event,
    AgentComputeStep,
    EnvironmentComputeStep,
)
from synth_sdk.tracing.events.store import event_store
from synth_sdk.tracing.local import _local, logger
from synth_sdk.tracing.trackers import synth_tracker_sync, synth_tracker_async, SynthTracker
from synth_sdk.tracing.events.manage import set_current_event

from typing import Callable, Optional, Set, Literal, Any, Dict, Tuple, Union
from functools import wraps
import time
import logging
from pydantic import BaseModel

from synth_sdk.tracing.abstractions import (
    ArbitraryInputs,
    ArbitraryOutputs,
    MessageInputs,
    MessageOutputs,
    Event,
    AgentComputeStep,
    EnvironmentComputeStep,
)
from synth_sdk.tracing.events.store import event_store
from synth_sdk.tracing.local import system_id_var, active_events_var
from synth_sdk.tracing.trackers import synth_tracker_async
from synth_sdk.tracing.events.manage import set_current_event

import inspect

logger = logging.getLogger(__name__)

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
    """
    Decorator for tracing synchronous method execution in an AI system.
    Handles automatic input/output tracking and event management.

    Args:
        origin (Literal["agent", "environment"]): Source of the computation
            - "agent": For AI/model operations
            - "environment": For external system operations

        event_type (str): Type of event being traced (e.g., "inference", "training")

        log_result (bool, optional): Whether to log the function's return value.
            Defaults to False.

        manage_event (Literal["create", "end", None], optional): 
            Controls the lifecycle of the event associated with the traced function.
            - "create": Start a new event at the beginning of the function execution.
            - "end": Immediately conclude the current event once the function execution completes.
            - None: Do not manage event lifecycle automatically. Event management must be handled manually if needed.
            Defaults to None.

        increment_partition (bool, optional): Whether to increment the trace partition.
            Used to group related events. Defaults to False.

        verbose (bool, optional): Enable detailed logging. Defaults to False.

        finetune_step (bool, optional): Mark this trace as part of fine-tuning.
            Defaults to True.

    Returns:
        Callable: The decorated function with tracing capabilities.

    Raises:
        ValueError: If:
            - The decorated method is called without an instance (`self`).
            - The instance lacks the required `system_id` attribute.
        RuntimeError: If tracing initialization fails.
        TypeError: If tracked values have invalid types.

    Notes:
        - Requires the decorated method to be an instance method with a `system_id` attribute.
        - Automatically tracks method inputs and outputs.
        - Manages thread-local storage for trace data.
        - Ensures cleanup in the `finally` block to maintain trace integrity.
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

            if not hasattr(self_instance, "system_id"):
                raise ValueError("Instance missing required system_id attribute")

            _local.system_id = self_instance.system_id
            #logger.debug(f"Set system_id in thread local: {_local.system_id}")

            # Initialize Trace
            synth_tracker_sync.initialize()

            # Initialize active_events if not present
            if not hasattr(_local, "active_events"):
                _local.active_events = {}
                #logger.debug("Initialized active_events in thread local storage")

            event = None
            compute_began = time.time()
            try:
                if manage_event == "create":
                    #logger.debug("Creating new event")
                    event = Event(
                        system_id=_local.system_id,
                        event_type=event_type,
                        opened=compute_began,
                        closed=None,
                        partition_index=0,
                        agent_compute_steps=[],
                        environment_compute_steps=[],
                    )
                    if increment_partition:
                        event.partition_index = event_store.increment_partition(
                            _local.system_id
                        )
                        # logger.debug(
                        #     f"Incremented partition to: {event.partition_index}"
                        # )

                    set_current_event(event)
                    #logger.debug(f"Created and set new event: {event_type}")

                # Automatically trace function inputs
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                bound_args.apply_defaults()
                for param, value in bound_args.arguments.items():
                    if param == "self":
                        continue
                    synth_tracker_sync.track_state(
                        variable_name=param,
                        variable_value=value,
                        origin=origin
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
                    var_origin = item['origin']
                    if 'variable_value' in item and 'variable_name' in item:
                        # Standard variable input
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            ArbitraryInputs(inputs={item['variable_name']: item['variable_value']})
                        )
                    elif 'messages' in item:
                        # Message input from track_lm
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            MessageInputs(messages=item['messages'])
                        )
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            ArbitraryInputs(inputs={"model_name": item["model_name"]})
                        )
                        finetune = item["finetune"] or finetune_step
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            ArbitraryInputs(inputs={"finetune": finetune})
                        )
                    else:
                        logger.warning(f"Unhandled traced input item: {item}")

                for item in traced_outputs:
                    var_origin = item['origin']
                    if 'variable_value' in item and 'variable_name' in item:
                        # Standard variable output
                        compute_steps_by_origin[var_origin]["outputs"].append(
                            ArbitraryOutputs(outputs={item['variable_name']: item['variable_value']})
                        )
                    elif 'messages' in item:
                        # Message output from track_lm
                        compute_steps_by_origin[var_origin]["outputs"].append(
                            MessageOutputs(messages=item['messages'])
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
                            len(event.agent_compute_steps)
                            + len(event.environment_compute_steps)
                            + 1
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
                                event.agent_compute_steps.append(compute_step)
                            else:
                                event.environment_compute_steps.append(compute_step)
                        # logger.debug(
                        #     f"Added compute step for {var_origin}: {compute_step.to_dict()}"
                        # )

                # Optionally log the function result
                if log_result:
                    logger.info(f"Function result: {result}")

                # Handle event management after function execution
                if (
                    manage_event == "end"
                    and event_type in _local.active_events
                ):
                    current_event = _local.active_events[event_type]
                    current_event.closed = compute_ended
                    # Store the event
                    if hasattr(_local, "system_id"):
                        event_store.add_event(_local.system_id, current_event)
                        # logger.debug(
                        #     f"Stored and closed event {event_type} for system {_local.system_id}"
                        # )
                    del _local.active_events[event_type]

                return result
            except Exception as e:
                logger.error(f"Exception in traced function '{func.__name__}': {e}")
                raise
            finally:
                #synth_tracker_sync.finalize()
                if hasattr(_local, "system_id"):
                    #logger.debug(f"Cleaning up system_id: {_local.system_id}")
                    delattr(_local, "system_id")

        return wrapper

    return decorator

def trace_system_async(
    origin: Literal["agent", "environment"],
    event_type: str,
    log_result: bool = False,
    manage_event: Literal["create", "end", "lazy_end", None] = None,
    increment_partition: bool = False,
    verbose: bool = False,
    finetune_step: bool = True,
) -> Callable:
    """
    Decorator for tracing asynchronous method execution in an AI system.
    Handles automatic input/output tracking and event management in async contexts.

    Args:
        origin (Literal["agent", "environment"]): Source of the computation
            - "agent": For AI/model operations
            - "environment": For external system operations

        event_type (str): Type of event being traced (e.g., "inference", "training")

        log_result (bool, optional): Whether to log the function's return value.
            Defaults to False.

        manage_event (Literal["create", "end", "lazy_end", None], optional): 
            Controls the lifecycle of the event associated with the traced function.
            - "create": Start a new event at the beginning of the function execution.
            - "end": Immediately conclude the current event once the function execution completes.
            - "lazy_end": Mark the event to be concluded after all nested computations and asynchronous tasks have finished. This ensures that all related asynchronous operations are captured within the same event scope.
            - None: Do not manage event lifecycle automatically. Event management must be handled manually if needed.
            Defaults to None.

        increment_partition (bool, optional): Whether to increment the trace partition.
            Used to group related events. Defaults to False.

        verbose (bool, optional): Enable detailed logging. Defaults to False.

        finetune_step (bool, optional): Mark this trace as part of fine-tuning.
            Defaults to True.

    Returns:
        Callable: The decorated asynchronous function with tracing capabilities.

    Raises:
        ValueError: If:
            - The decorated method is called without an instance (`self`).
            - The instance lacks the required `system_id` attribute.
        RuntimeError: If tracing initialization fails.
        TypeError: If tracked values have invalid types.

    Notes:
        - Requires the decorated method to be an instance method with a `system_id` attribute.
        - Automatically tracks method inputs and outputs.
        - Uses `contextvars` for async-safe storage.
        - Ensures cleanup in the `finally` block to maintain trace integrity.
        - Safe for concurrent execution in asynchronous contexts.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # logger.debug(f"Starting async_wrapper for {func.__name__}")
            # logger.debug(f"Args: {args}")
            # logger.debug(f"Kwargs: {kwargs}")
            
            # Automatically trace function inputs
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            #logger.debug(f"Bound args: {bound_args.arguments}")
            
            for param, value in bound_args.arguments.items():
                if param == "self":
                    continue
                #logger.debug(f"Tracking input param: {param} = {value}")
                synth_tracker_async.track_state(
                    variable_name=param,
                    variable_value=value,
                    origin=origin,
                    io_type="input"
                )

            # Determine the instance (self) if it's a method
            if not hasattr(func, "__self__") or not func.__self__:
                if not args:
                    raise ValueError(
                        "Instance method expected, but no arguments were passed."
                    )
                self_instance = args[0]
            else:
                self_instance = func.__self__

            if not hasattr(self_instance, "system_id"):
                raise ValueError("Instance missing required system_id attribute")

            # Set system_id using context variable
            system_id_token = system_id_var.set(self_instance.system_id)
            #logger.debug(f"Set system_id in context vars: {self_instance.system_id}")

            # Initialize AsyncTrace
            synth_tracker_async.initialize()

            # Initialize active_events if not present
            current_active_events = active_events_var.get()
            if not current_active_events:
                active_events_var.set({})
                #logger.debug("Initialized active_events in context vars")

            event = None
            compute_began = time.time()
            try:
                if manage_event == "create":
                    #logger.debug("Creating new event")
                    event = Event(
                        system_id=self_instance.system_id,
                        event_type=event_type,
                        opened=compute_began,
                        closed=None,
                        partition_index=0,
                        agent_compute_steps=[],
                        environment_compute_steps=[],
                    )
                    if increment_partition:
                        event.partition_index = event_store.increment_partition(
                            system_id_var.get()
                        )
                        # logger.debug(
                        #     f"Incremented partition to: {event.partition_index}"
                        # )

                    set_current_event(event)
                    #logger.debug(f"Created and set new event: {event_type}")

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
                        io_type="input"
                    )

                # Execute the coroutine
                result = await func(*args, **kwargs)

                # Automatically trace function output
                track_result(result, synth_tracker_async, origin)

                # Collect traced inputs and outputs
                traced_inputs, traced_outputs = synth_tracker_async.get_traced_data()

                compute_steps_by_origin: Dict[
                    Literal["agent", "environment"], Dict[str, List[Any]]
                ] = {
                    "agent": {"inputs": [], "outputs": []},
                    "environment": {"inputs": [], "outputs": []},
                }

                # Organize traced data by origin
                for item in traced_inputs:
                    var_origin = item['origin']
                    if 'variable_value' in item and 'variable_name' in item:
                        # Standard variable input
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            ArbitraryInputs(inputs={item['variable_name']: item['variable_value']})
                        )
                    elif 'messages' in item:
                        # Message input from track_lm
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            MessageInputs(messages=item['messages'])
                        )
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            ArbitraryInputs(inputs={"model_name": item["model_name"]})
                        )
                        finetune = finetune_step or item["finetune"]
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            ArbitraryInputs(inputs={"finetune": finetune})
                        )
                    else:
                        logger.warning(f"Unhandled traced input item: {item}")

                for item in traced_outputs:
                    var_origin = item['origin']
                    if 'variable_value' in item and 'variable_name' in item:
                        # Standard variable output
                        compute_steps_by_origin[var_origin]["outputs"].append(
                            ArbitraryOutputs(outputs={item['variable_name']: item['variable_value']})
                        )
                    elif 'messages' in item:
                        # Message output from track_lm
                        compute_steps_by_origin[var_origin]["outputs"].append(
                            MessageOutputs(messages=item['messages'])
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
                            len(event.agent_compute_steps)
                            + len(event.environment_compute_steps)
                            + 1
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
                                event.agent_compute_steps.append(compute_step)
                            else:
                                event.environment_compute_steps.append(compute_step)
                        # logger.debug(
                        #     f"Added compute step for {var_origin}: {compute_step.to_dict()}"
                        # )
                # Optionally log the function result
                if log_result:
                    logger.info(f"Function result: {result}")

                # Handle event management after function execution
                if (
                    manage_event == "end"
                    and event_type in active_events_var.get()
                ):
                    current_event = active_events_var.get()[event_type]
                    current_event.closed = compute_ended
                    # Store the event
                    if system_id_var.get():
                        event_store.add_event(system_id_var.get(), current_event)
                        # logger.debug(
                        #     f"Stored and closed event {event_type} for system {system_id_var.get()}"
                        # )
                    active_events = active_events_var.get()
                    del active_events[event_type]
                    active_events_var.set(active_events)

                return result
            except Exception as e:
                logger.error(f"Exception in traced function '{func.__name__}': {e}")
                raise
            finally:
                #synth_tracker_async.finalize()
                # Reset context variable for system_id
                system_id_var.reset(system_id_token)
                #logger.debug("Cleaning up system_id from context vars")
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
    """
    def decorator(func: Callable) -> Callable:
        # Check if the function is async or sync
        if inspect.iscoroutinefunction(func):
            # Use async tracing
            #logger.debug("Using async tracing")
            async_decorator = trace_system_async(
                origin, event_type, log_result, manage_event, increment_partition, verbose
            )
            return async_decorator(func)
        else:
            # Use sync tracing
            #logger.debug("Using sync tracing")
            sync_decorator = trace_system_sync(
                origin, event_type, log_result, manage_event, increment_partition, verbose
            )
            return sync_decorator(func)

    return decorator

def track_result(result, tracker, origin):
    """Helper function to track results, including tuple unpacking"""
    if isinstance(result, tuple):
        # Track each element of the tuple that matches valid types
        for i, item in enumerate(result):
            try:
                tracker.track_state(
                    variable_name=f"result_{i}",
                    variable_value=item,
                    origin=origin
                )
            except Exception as e:
                logger.warning(f"Could not track tuple element {i}: {str(e)}")
    else:
        # Track single result as before
        try:
            tracker.track_state(
                variable_name="result",
                variable_value=result,
                origin=origin
            )
        except Exception as e:
            logger.warning(f"Could not track result: {str(e)}")

