from typing import Callable, Optional, Set, Literal, Any
from functools import wraps
import threading
import time
from synth_sdk.tracing.abstractions import Event, AgentComputeStep, EnvironmentComputeStep
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
        # Store the event if we have a system_id
        if hasattr(_local, "system_id"):
            logger.debug(f"Storing existing event for system {_local.system_id}")
            try:
                event_store.add_event(_local.system_id, existing_event)
                logger.debug("Successfully stored existing event")
            except Exception as e:
                logger.error(f"Failed to store existing event: {str(e)}")
                raise
    else:
        logger.debug(f"No existing event of type {event.event_type}")
    
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


def set_system_id(system_id: str):
    """Set the system_id in thread local storage."""
    _local.system_id = system_id

def trace_system(
    origin: Literal["agent", "environment"],
    event_type: str,
    log_vars_input: Optional[Set[str]] = None,
    log_vars_output: Optional[Set[str]] = None,
    log_result: bool = False,
    manage_event: Literal["create", "end", None] = None,
    increment_partition: bool = False,
    verbose: bool = False,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create and set the event before function execution
            if manage_event == "create":
                event = Event(
                    event_type=event_type,
                    opened=time.time(),
                    closed=None,
                    partition_index=0,  # This will be updated if increment_partition is True
                    agent_compute_steps=[],
                    environment_compute_steps=[]
                )
                if not hasattr(_local, "active_events"):
                    _local.active_events = {}
                else:
                    # End any existing event of the same type
                    if event_type in _local.active_events:
                        existing_event = _local.active_events[event_type]
                        existing_event.closed = time.time()
                        if hasattr(self, "system_id"):
                            event_store.add_event(self.system_id, existing_event)
                _local.active_events[event_type] = event
                
                if increment_partition and hasattr(self, "system_id"):
                    event.partition_index = event_store.increment_partition(self.system_id)

            captured_vars = {}
            
            def trace_func(frame, event, arg):
                # Only trace our decorated function
                if frame.f_code == func.__code__:
                    #if verbose:
                        #logger.debug(f"Trace event: {event}")
                        #logger.debug(f"Locals at this point: {frame.f_locals}")
                    
                    # Capture variables on each line execution
                    if event == 'line':
                        if log_vars_output:
                            for var in log_vars_output:
                                if var in frame.f_locals:
                                    captured_vars[var] = frame.f_locals[var]
                                    #if verbose:
                                        #logger.debug(f"Captured {var} = {frame.f_locals[var]}")
                return trace_func
            
            sys.settrace(trace_func)
            try:
                result = func(self, *args, **kwargs)
            finally:
                sys.settrace(None)

 
            # Get current event and add compute step
            current_event = get_current_event(event_type)
            
            # Create compute step
            start_time = time.time()
            compute_input = {}
            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            if log_vars_input:
                compute_input = {var: kwargs.get(var, bound_args.arguments.get(var)) for var in log_vars_input if var in kwargs or var in bound_args.arguments}
            compute_output = captured_vars
            if log_result:
                compute_output['result'] = result

            end_time = time.time()
            
            # Create and append compute step
            if origin == "agent":
                compute_step = AgentComputeStep(
                    event_order=len(current_event.agent_compute_steps),
                    compute_began=start_time,
                    compute_ended=end_time,
                    compute_input=compute_input,
                    compute_output=compute_output,
                )
                current_event.agent_compute_steps.append(compute_step)
            elif origin == "environment":
                compute_step = EnvironmentComputeStep(
                    event_order=len(current_event.environment_compute_steps),
                    compute_began=start_time,
                    compute_ended=end_time,
                    compute_input=compute_input,
                    compute_output=compute_output,
                )
                current_event.environment_compute_steps.append(compute_step)

            # End the event after function execution if specified
            if manage_event == "end":
                current_event = get_current_event(event_type)
                current_event.closed = time.time()
                if hasattr(_local, "system_id"):
                    event_store.add_event(_local.system_id, current_event)
                clear_current_event(event_type)

            return result

        return wrapper

    return decorator

