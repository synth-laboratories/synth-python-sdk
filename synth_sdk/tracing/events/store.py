import json
import logging
import time
from threading import RLock  # Change this import
from typing import Dict, List

from synth_sdk.tracing.abstractions import Event, EventPartitionElement, SystemTrace
from synth_sdk.tracing.local import (  # Import context variables
    _local,
    active_events_var,
    instance_system_id_var,
    system_id_var,
    system_name_var,
)

logger = logging.getLogger(__name__)


class EventStore:
    def __init__(self):
        self._traces: Dict[str, SystemTrace] = {}
        self._lock = RLock()  # Use RLock instead of Lock
        self.logger = logging.getLogger(__name__)

    def get_or_create_system_trace(
        self, instance_system_id: str, system_id: str, _already_locked: bool = False
    ) -> SystemTrace:
        """Get or create a SystemTrace for the given instance_system_id."""
        logger = logging.getLogger(__name__)
        # logger.debug(f"Starting get_or_create_system_trace for {instance_system_id}")

        def _get_or_create():
            # logger.debug("Inside _get_or_create")
            if instance_system_id not in self._traces:
                # logger.debug(f"Creating new system trace for {instance_system_id}")
                self._traces[instance_system_id] = SystemTrace(
                    system_id=system_id,
                    instance_system_id=instance_system_id,
                    metadata={},
                    partition=[EventPartitionElement(partition_index=0, events=[])],
                    current_partition_index=0,
                )
            # logger.debug("Returning system trace")
            return self._traces[instance_system_id]

        if _already_locked:
            return _get_or_create()
        else:
            with self._lock:
                # logger.debug("Lock acquired in get_or_create_system_trace")
                return _get_or_create()

    def increment_partition(self, instance_system_id: str, system_id: str) -> int:
        """Increment the partition index for a system and create new partition element."""
        logger = logging.getLogger(__name__)
        # logger.debug(f"Starting increment_partition for system {instance_system_id}")

        with self._lock:
            # logger.debug("Lock acquired in increment_partition")
            system_trace = self.get_or_create_system_trace(
                instance_system_id, system_id, _already_locked=True
            )
            # logger.debug(
            #     f"Got system trace, current index: {system_trace.current_partition_index}"
            # )

            system_trace.current_partition_index += 1
            # logger.debug(
            #     f"Incremented index to: {system_trace.current_partition_index}"
            # )

            system_trace.partition.append(
                EventPartitionElement(
                    partition_index=system_trace.current_partition_index, events=[]
                )
            )
            # logger.debug("Added new partition element")

            return system_trace.current_partition_index

    def add_event(self, instance_system_id: str, system_id: str, event: Event):
        """Add an event to the appropriate partition of the system trace."""
        # self.#logger.debug(f"Adding event type {event.event_type} to system {instance_system_id}")
        # self.#logger.debug(
        #     f"Event details: opened={event.opened}, closed={event.closed}, partition={event.partition_index}"
        # )
        # print("Adding event to partition")

        # try:
        if not self._lock.acquire(timeout=5):
            self.logger.error("Failed to acquire lock within timeout period")
            return

        try:
            system_trace = self.get_or_create_system_trace(
                instance_system_id, system_id
            )
            # self.#logger.debug(
            #     f"Got system trace with {len(system_trace.partition)} partitions"
            # )

            current_partition = next(
                (
                    p
                    for p in system_trace.partition
                    if p.partition_index == event.partition_index
                ),
                None,
            )

            if current_partition is None:
                self.logger.error(
                    f"No partition found for index {event.partition_index} - existing partitions: {set([p.partition_index for p in system_trace.partition])}"
                )
                raise ValueError(
                    f"No partition found for index {event.partition_index}"
                )

            current_partition.events.append(event)
            # self.#logger.debug(
            #     f"Added event to partition {event.partition_index}. Total events: {len(current_partition.events)}"
            # )
        finally:
            self._lock.release()
        # except Exception as e:
        #     self.logger.error(f"Error in add_event: {str(e)}", exc_info=True)
        #     raise

    def get_system_traces(self) -> List[SystemTrace]:
        """Get all system traces."""
        with self._lock:
            self.end_all_active_events()

            return list(self._traces.values())

    def end_all_active_events(self):
        """End all active events and store them."""
        # self.#logger.debug("Ending all active events")

        # For synchronous code
        if hasattr(_local, "active_events"):
            active_events = _local.active_events
            instance_system_id = getattr(_local, "instance_system_id", None)
            system_id = getattr(_local, "system_id", None)
            if active_events:  # and instance_system_id:
                for event_type, event in list(active_events.items()):
                    if event.closed is None:
                        event.closed = time.time()
                        self.add_event(event.instance_system_id, system_id, event)
                        # self.#logger.debug(f"Stored and closed event {event_type}")
                _local.active_events.clear()

        else:
            # For asynchronous code
            active_events_async = active_events_var.get()

            

            if active_events_async:  # and instance_system_id_async:
                for event_type, event in list(active_events_async.items()):
                    system_id = system_id_var.get()
                    if event.closed is None:
                        event.closed = time.time()
                        self.add_event(event.instance_system_id, system_id, event)
                        # self.#logger.debug(f"Stored and closed event {event_type}")
                active_events_var.set({})

    def get_system_traces_json(self) -> str:
        """Get all system traces as JSON."""
        with self._lock:
            return json.dumps(
                [
                    {
                        "instance_system_id": trace.instance_system_id,
                        "current_partition_index": trace.current_partition_index,
                        "partition": [
                            {
                                "partition_index": p.partition_index,
                                "events": [
                                    self._event_to_dict(event) for event in p.events
                                ],
                            }
                            for p in trace.partition
                        ],
                    }
                    for trace in self._traces.values()
                ],
                default=str,
            )

    def _event_to_dict(self, event: Event) -> dict:
        """Convert an Event object to a dictionary."""
        return {
            "event_type": event.event_type,
            "opened": event.opened,
            "closed": event.closed,
            "partition_index": event.partition_index,
            "agent_compute_steps": [
                {
                    "event_order": step.event_order,
                    "compute_began": step.compute_began,
                    "compute_ended": step.compute_ended,
                    "compute_input": step.compute_input,
                    "compute_output": step.compute_output,
                }
                for step in event.agent_compute_steps
            ],
            "environment_compute_steps": [
                {
                    "event_order": step.event_order,
                    "compute_began": step.compute_began,
                    "compute_ended": step.compute_ended,
                    "compute_input": step.compute_input,
                    "compute_output": step.compute_output,
                }
                for step in event.environment_compute_steps
            ],
        }


# Global event store instance
event_store = EventStore()
