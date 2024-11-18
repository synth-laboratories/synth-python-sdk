from contextlib import contextmanager
import time
from synth_sdk.tracing.abstractions import Event
from synth_sdk.tracing.decorators import set_current_event, clear_current_event, _local
from synth_sdk.tracing.events.store import event_store
from synth_sdk.tracing.local import system_id_var


@contextmanager
def event_scope(event_type: str):
    """
    Context manager for creating and managing events.

    Usage:
        with event_scope("my_event_type"):
            # do stuff
    """
    # Check if we're in an async context
    try:
        import asyncio
        asyncio.get_running_loop()
        is_async = True
    except RuntimeError:
        is_async = False

    # Get system_id from appropriate source
    system_id = system_id_var.get() if is_async else getattr(_local, "system_id", None)
    
    event = Event(
        system_id=system_id,
        event_type=event_type,
        opened=time.time(),
        closed=None,
        partition_index=0,
        agent_compute_steps=[],
        environment_compute_steps=[],
    )
    set_current_event(event)

    try:
        yield event
    finally:
        event.closed = time.time()
        clear_current_event(event_type)
        # Store the event if system_id is available
        if system_id:
            event_store.add_event(system_id, event)
