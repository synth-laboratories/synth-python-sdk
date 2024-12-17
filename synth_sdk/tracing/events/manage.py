import time
from typing import Literal, Optional

from synth_sdk.tracing.abstractions import Event
from synth_sdk.tracing.events.store import event_store
from synth_sdk.tracing.local import _local, logger


def get_current_event(event_type: str) -> "Event":
    """
    Get the current active event of the specified type.
    Raises ValueError if no such event exists.
    """
    events = getattr(_local, "active_events", {})
    if event_type not in events:
        raise ValueError(f"No active event of type '{event_type}' found")
    return events[event_type]


def set_current_event(
    event: Optional["Event"], decorator_type: Literal["sync", "async"] = None
):
    """
    Set the current event, ending any existing events of the same type.
    If event is None, it clears the current event of that type.
    """
    if event is None:
        raise ValueError("Event cannot be None when setting current event.")

    # logger.debug(f"Setting current event of type {event.event_type}")

    # Check if we're in an async context
    try:
        import asyncio

        asyncio.get_running_loop()
        is_async = True
    except RuntimeError:
        is_async = False

    if decorator_type == "sync" or not is_async:
        # Original thread-local storage logic
        if not hasattr(_local, "active_events"):
            _local.active_events = {}
            logger.debug("Initialized active_events in thread local storage")

        # If there's an existing event of the same type, end it
        if event.event_type in _local.active_events:
            if (
                _local.active_events[event.event_type].system_instance_id
                == event.system_instance_id
            ):
                logger.debug(f"Found existing event of type {event.event_type}")
                existing_event = _local.active_events[event.event_type]
                existing_event.closed = time.time()
                logger.debug(
                    f"Closed existing event of type {event.event_type} at {existing_event.closed}"
                )

                # Store the closed event if system_instance_id is present
                if hasattr(_local, "system_instance_id"):
                    logger.debug(
                        f"Storing closed event for system {_local.system_instance_id}"
                    )
                    try:
                        event_store.add_event(
                            _local.system_instance_id, _local.system_id, existing_event
                        )
                        logger.debug("Successfully stored closed event")
                    except Exception as e:
                        logger.error(f"Failed to store closed event: {str(e)}")
                        raise

        # Set the new event
        _local.active_events[event.event_type] = event
        # logger.debug("New event set as current in thread local")

    else:
        from synth_sdk.tracing.local import (
            active_events_var,
            system_id_var,
            system_instance_id_var,
        )

        # Get current active events from context var
        active_events = active_events_var.get()

        # If there's an existing event of the same type, end it
        if event.event_type in active_events:
            existing_event = active_events[event.event_type]

            # Check that the active event has the same system_instance_id as the one we're settting
            if existing_event.system_instance_id == event.system_instance_id:
                logger.debug(f"Found existing event of type {event.event_type}")
                existing_event.closed = time.time()
                logger.debug(
                    f"Closed existing event of type {event.event_type} at {existing_event.closed}"
                )

                # Store the closed event if system_instance_id is present
                system_instance_id = system_instance_id_var.get()
                if system_instance_id:
                    logger.debug(
                        f"Storing closed event for system {system_instance_id}"
                    )
                    try:
                        event_store.add_event(
                            system_instance_id, system_id_var.get(), existing_event
                        )
                        logger.debug("Successfully stored closed event")
                    except Exception as e:
                        logger.error(f"Failed to store closed event: {str(e)}")
                        raise

        # Set the new event
        active_events[event.event_type] = event
        active_events_var.set(active_events)
        logger.debug("New event set as current in context vars")


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
        if hasattr(_local, "system_instance_id"):
            event_store.add_event(
                _local.system_instance_id, _local.system_id, current_event
            )
        clear_current_event(event_type)
    return current_event
