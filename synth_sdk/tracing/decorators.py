from typing import Callable, Optional, Set, Literal, Any, Dict, Tuple, Union
from functools import wraps
import threading
import time
import logging
import inspect
import contextvars
from pydantic import BaseModel

from synth_sdk.tracing.abstractions import (
    Event,
    AgentComputeStep,
    EnvironmentComputeStep,
)
from synth_sdk.tracing.events.store import event_store

logger = logging.getLogger(__name__)

# Thread-local storage for active events and system_id
_local = threading.local()

# Update VALID_TYPES to include NoneType
VALID_TYPES = (BaseModel, str, dict, int, float, bool, list, type(None))


def get_current_event(event_type: str) -> "Event":
    """
    Get the current active event of the specified type.
    Raises ValueError if no such event exists.
    """
    events = getattr(_local, "active_events", {})
    if event_type not in events:
        raise ValueError(f"No active event of type '{event_type}' found")
    return events[event_type]


def set_current_event(event: Optional["Event"]):
    """
    Set the current event, ending any existing events of the same type.
    If event is None, it clears the current event of that type.
    """
    if event is None:
        raise ValueError("Event cannot be None when setting current event.")
    
    logger.debug(f"Setting current event of type {event.event_type}")

    if not hasattr(_local, "active_events"):
        _local.active_events = {}
        logger.debug("Initialized active_events in thread local storage")

    # If there's an existing event of the same type, end it
    if event.event_type in _local.active_events:
        logger.debug(f"Found existing event of type {event.event_type}")
        existing_event = _local.active_events[event.event_type]
        existing_event.closed = time.time()
        logger.debug(f"Closed existing event of type {event.event_type} at {existing_event.closed}")

        # Store the closed event if system_id is present
        if hasattr(_local, "system_id"):
            logger.debug(f"Storing closed event for system {_local.system_id}")
            try:
                event_store.add_event(_local.system_id, existing_event)
                logger.debug("Successfully stored closed event")
            except Exception as e:
                logger.error(f"Failed to store closed event: {str(e)}")
                raise

    else:
        logger.debug(f"No existing event of type {event.event_type}")

    # Set the new event
    _local.active_events[event.event_type] = event
    logger.debug("New event set as current")


def clear_current_event(event_type: str):
    if hasattr(_local, "active_events"):
        _local.active_events.pop(event_type, None)
        logger.debug(f"Cleared current event of type {event_type}")


def end_event(event_type: str) -> Optional[Event]:
    """End the current event and store it."""
    current_event = get_current_event(event_type)
    if current_event:
        current_event.closed = time.time()
        # Store the event
        if hasattr(_local, "system_id"):
            event_store.add_event(_local.system_id, current_event)
        clear_current_event(event_type)
    return current_event


class Trace:
    _local = threading.local()

    @classmethod
    def initialize(cls):
        cls._local.initialized = True
        cls._local.inputs = []  # List of tuples: (origin, var)
        cls._local.outputs = []  # List of tuples: (origin, var)

    @classmethod
    def input(cls, var: Union[BaseModel, str, dict, int, float, bool, list, None], variable_name: str, origin: Literal["agent", "environment"], annotation: Optional[str] = None):
        if not isinstance(var, VALID_TYPES):
            raise TypeError(f"Variable {variable_name} must be one of {VALID_TYPES}, got {type(var)}")
        
        if getattr(cls._local, 'initialized', False):
            # Convert Pydantic models to dict schema
            if isinstance(var, BaseModel):
                var = var.model_dump()
            cls._local.inputs.append((origin, var, variable_name, annotation))
            logger.debug(f"Traced input: origin={origin}, var_name={variable_name}, annotation={annotation}")
        else:
            raise RuntimeError("Trace not initialized. Use within a function decorated with @trace_system_sync.")

    @classmethod
    def output(cls, var: Union[BaseModel, str, dict, int, float, bool, list, None], variable_name: str, origin: Literal["agent", "environment"], annotation: Optional[str] = None):
        if not isinstance(var, VALID_TYPES):
            raise TypeError(f"Variable {variable_name} must be one of {VALID_TYPES}, got {type(var)}")
            
        if getattr(cls._local, 'initialized', False):
            # Convert Pydantic models to dict schema
            if isinstance(var, BaseModel):
                var = var.model_dump()
            cls._local.outputs.append((origin, var, variable_name, annotation))
            logger.debug(f"Traced output: origin={origin}, var_name={variable_name}, annotation={annotation}")
        else:
            raise RuntimeError("Trace not initialized. Use within a function decorated with @trace_system_sync.")

    @classmethod
    def get_traced_data(cls):
        return getattr(cls._local, 'inputs', []), getattr(cls._local, 'outputs', [])

    @classmethod
    def finalize(cls):
        # Clean up the thread-local storage
        cls._local.initialized = False
        cls._local.inputs = []
        cls._local.outputs = []
        logger.debug("Finalized trace data")


# Context variables for asynchronous tracing
trace_inputs_var = contextvars.ContextVar('trace_inputs', default=None)
trace_outputs_var = contextvars.ContextVar('trace_outputs', default=None)
trace_initialized_var = contextvars.ContextVar('trace_initialized', default=False)


class AsyncTrace:
    @classmethod
    def initialize(cls):
        trace_initialized_var.set(True)
        trace_inputs_var.set([])   # List of tuples: (origin, var)
        trace_outputs_var.set([])  # List of tuples: (origin, var)
        logger.debug("AsyncTrace initialized")

    @classmethod
    def input(cls, var: Union[BaseModel, str, dict, int, float, bool, list, None], variable_name: str, origin: Literal["agent", "environment"], annotation: Optional[str] = None):
        if not isinstance(var, VALID_TYPES):
            raise TypeError(f"Variable {variable_name} must be one of {VALID_TYPES}, got {type(var)}")
            
        if trace_initialized_var.get():
            # Convert Pydantic models to dict schema
            if isinstance(var, BaseModel):
                var = var.model_dump()
            trace_inputs = trace_inputs_var.get()
            trace_inputs.append((origin, var, variable_name, annotation))
            trace_inputs_var.set(trace_inputs)
            logger.debug(f"Traced input: origin={origin}, var_name={variable_name}, annotation={annotation}")
        else:
            raise RuntimeError("Trace not initialized. Use within a function decorated with @trace_system_async.")

    @classmethod
    def output(cls, var: Union[BaseModel, str, dict, int, float, bool, list, None], variable_name: str, origin: Literal["agent", "environment"], annotation: Optional[str] = None):
        if not isinstance(var, VALID_TYPES):
            raise TypeError(f"Variable {variable_name} must be one of {VALID_TYPES}, got {type(var)}")
            
        if trace_initialized_var.get():
            # Convert Pydantic models to dict schema
            if isinstance(var, BaseModel):
                var = var.model_dump()
            trace_outputs = trace_outputs_var.get()
            trace_outputs.append((origin, var, variable_name, annotation))
            trace_outputs_var.set(trace_outputs)
            logger.debug(f"Traced output: origin={origin}, var_name={variable_name}, annotation={annotation}")
        else:
            raise RuntimeError("Trace not initialized. Use within a function decorated with @trace_system_async.")

    @classmethod
    def get_traced_data(cls) -> Tuple[list, list]:
        return trace_inputs_var.get(), trace_outputs_var.get()

    @classmethod
    def finalize(cls):
        trace_initialized_var.set(False)
        trace_inputs_var.set([])
        trace_outputs_var.set([])
        logger.debug("Finalized async trace data")


# Make traces available globally
trace = Trace
async_trace = AsyncTrace


def trace_system_sync(
    origin: Literal["agent", "environment"],
    event_type: str,
    log_result: bool = False,
    manage_event: Literal["create", "end", "lazy_end", None] = None,
    increment_partition: bool = False,
    verbose: bool = False,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine the instance (self) if it's a method
            if not hasattr(func, '__self__') or not func.__self__:
                if not args:
                    raise ValueError("Instance method expected, but no arguments were passed.")
                self_instance = args[0]
            else:
                self_instance = func.__self__

            if not hasattr(self_instance, 'system_id'):
                raise ValueError("Instance missing required system_id attribute")

            _local.system_id = self_instance.system_id
            logger.debug(f"Set system_id in thread local: {_local.system_id}")

            # Initialize Trace
            trace.initialize()

            # Initialize active_events if not present
            if not hasattr(_local, 'active_events'):
                _local.active_events = {}
                logger.debug("Initialized active_events in thread local storage")

            event = None
            compute_began = time.time()
            try:
                if manage_event == "create":
                    logger.debug("Creating new event")
                    event = Event(
                        event_type=event_type,
                        opened=compute_began,
                        closed=None,
                        partition_index=0,
                        agent_compute_steps=[],
                        environment_compute_steps=[]
                    )
                    if increment_partition:
                        event.partition_index = event_store.increment_partition(_local.system_id)
                        logger.debug(f"Incremented partition to: {event.partition_index}")

                    set_current_event(event)
                    logger.debug(f"Created and set new event: {event_type}")

                # Automatically trace function inputs
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                bound_args.apply_defaults()
                for param, value in bound_args.arguments.items():
                    if param == 'self':
                        continue
                    trace.input(value, param, origin)

                # Execute the function
                result = func(*args, **kwargs)

                # Automatically trace function output
                if log_result:
                    trace.output(result, "result", origin)

                # Collect traced inputs and outputs
                traced_inputs, traced_outputs = trace.get_traced_data()

                compute_steps_by_origin: Dict[Literal["agent", "environment"], Dict[str, Dict[str, Any]]] = {
                    "agent": {
                        "inputs": {},
                        "outputs": {}
                    },
                    "environment": {
                        "inputs": {},
                        "outputs": {}
                    }
                }

                # Organize traced data by origin
                for var_origin, var, var_name, _ in traced_inputs:
                    compute_steps_by_origin[var_origin]["inputs"][var_name] = var
                for var_origin, var, var_name, _ in traced_outputs:
                    compute_steps_by_origin[var_origin]["outputs"][var_name] = var

                # Capture compute end time
                compute_ended = time.time()

                # Create compute steps grouped by origin
                for var_origin in ["agent", "environment"]:
                    inputs = compute_steps_by_origin[var_origin]["inputs"]
                    outputs = compute_steps_by_origin[var_origin]["outputs"]
                    if inputs or outputs:
                        event_order = (
                            len(event.agent_compute_steps) + len(event.environment_compute_steps) + 1
                            if event else 1
                        )
                        compute_step = AgentComputeStep(
                            event_order=event_order,
                            compute_began=compute_began,
                            compute_ended=compute_ended,
                            compute_input=inputs,
                            compute_output=outputs
                        ) if var_origin == "agent" else EnvironmentComputeStep(
                            event_order=event_order,
                            compute_began=compute_began,
                            compute_ended=compute_ended,
                            compute_input=inputs,
                            compute_output=outputs
                        )
                        if event:
                            if var_origin == "agent":
                                event.agent_compute_steps.append(compute_step)
                            else:
                                event.environment_compute_steps.append(compute_step)
                        logger.debug(f"Added compute step for {var_origin}: {compute_step.to_dict()}")

                # Optionally log the function result
                if log_result:
                    logger.info(f"Function result: {result}")

                # Handle event management after function execution
                if manage_event in ["end", "lazy_end"] and event_type in _local.active_events:
                    current_event = _local.active_events[event_type]
                    current_event.closed = compute_ended
                    # Store the event
                    if hasattr(_local, "system_id"):
                        event_store.add_event(_local.system_id, current_event)
                        logger.debug(f"Stored and closed event {event_type} for system {_local.system_id}")
                    del _local.active_events[event_type]

                return result
            except Exception as e:
                logger.error(f"Exception in traced function '{func.__name__}': {e}")
                raise
            finally:
                trace.finalize()
                if hasattr(_local, 'system_id'):
                    logger.debug(f"Cleaning up system_id: {_local.system_id}")
                    delattr(_local, 'system_id')

        return wrapper

    return decorator


def trace_system_async(
    origin: Literal["agent", "environment"],
    event_type: str,
    log_result: bool = False,
    manage_event: Literal["create", "end", "lazy_end", None] = None,
    increment_partition: bool = False,
    verbose: bool = False,
) -> Callable:
    """Decorator for tracing asynchronous functions."""
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

            if not hasattr(self_instance, "system_id"):
                raise ValueError("Instance missing required system_id attribute")

            _local.system_id = self_instance.system_id
            logger.debug(f"Set system_id in thread local: {_local.system_id}")

            # Initialize AsyncTrace
            async_trace.initialize()

            # Initialize active_events if not present
            if not hasattr(_local, "active_events"):
                _local.active_events = {}
                logger.debug("Initialized active_events in thread local storage")

            event = None
            compute_began = time.time()
            try:
                if manage_event == "create":
                    logger.debug("Creating new event")
                    event = Event(
                        event_type=event_type,
                        opened=compute_began,
                        closed=None,
                        partition_index=0,  # Will be updated if increment_partition is True
                        agent_compute_steps=[],
                        environment_compute_steps=[],
                    )
                    if increment_partition:
                        event.partition_index = event_store.increment_partition(_local.system_id)
                        logger.debug(f"Incremented partition to: {event.partition_index}")

                    set_current_event(event)
                    logger.debug(f"Created and set new event: {event_type}")

                # Automatically trace function inputs
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                bound_args.apply_defaults()
                for param, value in bound_args.arguments.items():
                    if param == 'self':
                        continue
                    async_trace.input(value, param, origin)

                # Execute the coroutine
                result = await func(*args, **kwargs)

                # Automatically trace function output
                if log_result:
                    async_trace.output(result, "result", origin)

                # Collect traced inputs and outputs
                traced_inputs, traced_outputs = async_trace.get_traced_data()

                compute_steps_by_origin: Dict[Literal["agent", "environment"], Dict[str, Dict[str, Any]]] = {
                    "agent": {
                        "inputs": {},
                        "outputs": {}
                    },
                    "environment": {
                        "inputs": {},
                        "outputs": {}
                    }
                }

                # Organize traced data by origin
                for var_origin, var, var_name, _ in traced_inputs:
                    compute_steps_by_origin[var_origin]["inputs"][var_name] = var
                for var_origin, var, var_name, _ in traced_outputs:
                    compute_steps_by_origin[var_origin]["outputs"][var_name] = var

                # Capture compute end time
                compute_ended = time.time()

                # Create compute steps grouped by origin
                for var_origin in ["agent", "environment"]:
                    inputs = compute_steps_by_origin[var_origin]["inputs"]
                    outputs = compute_steps_by_origin[var_origin]["outputs"]
                    print("Var origin:", var_origin)
                    print("Inputs:", inputs)
                    print("Outputs:", outputs)
                    if inputs or outputs:
                        event_order = (
                            len(event.agent_compute_steps) + len(event.environment_compute_steps) + 1
                            if event else 1
                        )
                        compute_step = AgentComputeStep(
                            event_order=event_order,
                            compute_began=compute_began,
                            compute_ended=compute_ended,
                            compute_input=inputs,
                            compute_output=outputs,
                        ) if var_origin == "agent" else EnvironmentComputeStep(
                            event_order=event_order,
                            compute_began=compute_began,
                            compute_ended=compute_ended,
                            compute_input=inputs,
                            compute_output=outputs,
                        )
                        if event:
                            if var_origin == "agent":
                                event.agent_compute_steps.append(compute_step)
                            else:
                                event.environment_compute_steps.append(compute_step)
                        logger.debug(f"Added compute step for {var_origin}: {compute_step.to_dict()}")

                # Optionally log the function result
                if log_result:
                    logger.info(f"Function result: {result}")

                # Handle event management after function execution
                if manage_event in ["end", "lazy_end"] and event_type in _local.active_events:
                    current_event = _local.active_events[event_type]
                    current_event.closed = compute_ended
                    # Store the event
                    if hasattr(_local, "system_id"):
                        event_store.add_event(_local.system_id, current_event)
                        logger.debug(f"Stored and closed event {event_type} for system {_local.system_id}")
                    del _local.active_events[event_type]

                return result
            except Exception as e:
                logger.error(f"Exception in traced function '{func.__name__}': {e}")
                raise
            finally:
                async_trace.finalize()
                if hasattr(_local, 'system_id'):
                    logger.debug(f"Cleaning up system_id: {_local.system_id}")
                    delattr(_local, 'system_id')

        return async_wrapper
    return decorator