from typing import Callable, Optional, Set, Literal, Any, Dict
from functools import wraps
import threading
import time
import logging
import inspect

from synth_sdk.tracing.abstractions import (
    Event,
    AgentComputeStep,
    EnvironmentComputeStep,
)
from synth_sdk.tracing.events.store import event_store

logger = logging.getLogger(__name__)

# Thread-local storage for active events and system_id
_local = threading.local()


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
    def input(cls, var: Any, origin: Literal["agent", "environment"]):
        if getattr(cls._local, 'initialized', False):
            cls._local.inputs.append((origin, var))
            logger.debug(f"Traced input: origin={origin}, var={var}")
        else:
            raise RuntimeError("Trace not initialized. Use within a function decorated with @trace_system.")

    @classmethod
    def output(cls, var: Any, origin: Literal["agent", "environment"]):
        if getattr(cls._local, 'initialized', False):
            cls._local.outputs.append((origin, var))
            logger.debug(f"Traced output: origin={origin}, var={var}")
        else:
            raise RuntimeError("Trace not initialized. Use within a function decorated with @trace_system.")

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


# Make 'trace' available globally
trace = Trace


def trace_system(
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
            try:
                if manage_event == "create":
                    logger.debug("Creating new event")
                    event = Event(
                        event_type=event_type,
                        opened=time.time(),
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
                    trace.input(value, origin)

                # Execute the function
                result = func(*args, **kwargs)

                # Automatically trace function output
                if log_result:
                    trace.output(result, origin)

                # Collect traced inputs and outputs
                traced_inputs, traced_outputs = trace.get_traced_data()

                compute_steps_by_origin: Dict[Literal["agent", "environment"], Dict[str, Any]] = {
                    "agent": {
                        "inputs": [],
                        "outputs": []
                    },
                    "environment": {
                        "inputs": [],
                        "outputs": []
                    }
                }

                # Organize traced data by origin
                for var_origin, var in traced_inputs:
                    compute_steps_by_origin[var_origin]["inputs"].append(var)
                for var_origin, var in traced_outputs:
                    compute_steps_by_origin[var_origin]["outputs"].append(var)

                if log_result and result is not None:
                    # Already handled above
                    pass

                # Create compute steps grouped by origin
                for var_origin in ["agent", "environment"]:
                    inputs = compute_steps_by_origin[var_origin]["inputs"]
                    outputs = compute_steps_by_origin[var_origin]["outputs"]
                    if inputs or outputs:
                        description = f"{event_type} - {var_origin.capitalize()} Data"
                        metadata = {
                            "inputs": inputs,
                            "outputs": outputs
                        }
                        compute_step = AgentComputeStep(
                            description=description,
                            timestamp=time.time(),
                            metadata=metadata
                        ) if var_origin == "agent" else EnvironmentComputeStep(
                            description=description,
                            timestamp=time.time(),
                            metadata=metadata
                        )
                        if event:
                            if var_origin == "agent":
                                event.agent_compute_steps.append(compute_step)
                            else:
                                event.environment_compute_steps.append(compute_step)
                        logger.debug(f"Added compute step for {var_origin}: {metadata}")

                # Optionally log the function result
                if log_result:
                    logger.info(f"Function result: {result}")

                # Handle event management after function execution
                if manage_event in ["end", "lazy_end"] and event_type in _local.active_events:
                    current_event = _local.active_events[event_type]
                    current_event.closed = time.time()
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
