from typing import Callable, Optional, Set, Literal, Any, Dict, Tuple, Union
from functools import wraps
import threading
import time
import logging
import inspect
import contextvars
from pydantic import BaseModel
from synth_sdk.tracing.local import _local, logger
from synth_sdk.tracing.events.store import event_store
from synth_sdk.tracing.abstractions import Event


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
        logger.debug(
            f"Closed existing event of type {event.event_type} at {existing_event.closed}"
        )

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
