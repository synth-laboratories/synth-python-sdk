from typing import Callable, Optional, Set, Literal, Any
from functools import wraps
import threading
import time
from synth_sdk.tracing.abstractions import (
    Event,
    AgentComputeStep,
    EnvironmentComputeStep,
)
from synth_sdk.tracing.events.store import event_store
import logging
import inspect
import sys

logger = logging.getLogger(__name__)

# Thread-local storage for active events
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


def set_current_event(event: "Event"):
    """
    Set the current event, ending any existing events of the same type.
    """
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

    if event is not None:
        # Set the new event
        _local.active_events[event.event_type] = event
        logger.debug("New event set as current")


def clear_current_event(event_type: str):
    if hasattr(_local, "active_events"):
        _local.active_events.pop(event_type, None)


def end_event(event_type: str) -> Optional[Event]:
    """End the current event and store it."""
    current_event = get_current_event(event_type)
    if current_event:
        current_event.closed = time.time()
        # Store the event
        if hasattr(_local, "system_id"):
            event_store.add_event(_local.system_id, current_event)
        set_current_event(None)
    return current_event


def trace_system(
    origin: Literal["agent", "environment"],
    event_type: str,
    log_vars_input: Optional[Set[str]] = None,
    log_vars_output: Optional[Set[str]] = None,
    log_result: bool = False,
    manage_event: Literal["create", "end", "lazy_end", None] = None,
    increment_partition: bool = False,
    verbose: bool = False,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'system_id'):
                raise ValueError("Instance missing required system_id attribute")
            
            _local.system_id = self.system_id
            logger.debug(f"Set system_id in thread local: {_local.system_id}")

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
                    
                    # Use set_current_event to handle existing events
                    set_current_event(event)
                    logger.debug(f"Created and set new event: {event_type}")
                
                # Log input variables
                if log_vars_input:
                    bound_args = inspect.signature(func).bind(self, *args, **kwargs)
                    bound_args.apply_defaults()
                    inputs = {var: bound_args.arguments[var] for var in log_vars_input if var in bound_args.arguments}
                    if hasattr(event, 'input_vars') and event is not None:
                        event.input_vars = inputs
                        logger.info(f"Logged input variables: {inputs}")
                    else:
                        logger.warning("Event object does not have 'input_vars' attribute or event is None")
                
                # Execute the function
                result = func(self, *args, **kwargs)
                
                # Log output variables
                if log_vars_output:
                    if isinstance(result, (list, tuple)):
                        outputs = {var: result[i] if i < len(result) else None for i, var in enumerate(log_vars_output)}
                    else:
                        outputs = {var: getattr(result, var, None) for var in log_vars_output}
                    
                    if hasattr(event, 'output_vars') and event is not None:
                        event.output_vars = outputs
                        logger.info(f"Logged output variables: {outputs}")
                    else:
                        logger.warning("Event object does not have 'output_vars' attribute or event is None")
                
                if manage_event in ["end", "lazy_end"] and event_type in _local.active_events:
                    current_event = _local.active_events[event_type]
                    current_event.closed = time.time()
                    # No need to re-add the event; it's already in the event store
                    logger.debug(f"Closed event {event_type} for system {_local.system_id}")
                    del _local.active_events[event_type]

                return result
            finally:
                if hasattr(_local, 'system_id'):
                    logger.debug(f"Cleaning up system_id: {_local.system_id}")
                    delattr(_local, 'system_id')
        return wrapper
    return decorator
