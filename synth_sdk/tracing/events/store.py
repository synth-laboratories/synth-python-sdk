import json
import threading
import logging
from typing import Dict, List, Optional
from synth_sdk.tracing.abstractions import SystemTrace, EventPartitionElement, Event
from synth_sdk.tracing.config import tracer  # Update this import line
from threading import RLock  # Change this import


logger = logging.getLogger(__name__)

class EventStore:
    def __init__(self):
        self._traces: Dict[str, SystemTrace] = {}
        self._lock = RLock()  # Use RLock instead of Lock
        self.logger = logging.getLogger(__name__)

    def get_or_create_system_trace(self, system_id: str, _already_locked: bool = False) -> SystemTrace:
        """Get or create a SystemTrace for the given system_id."""
        logger = logging.getLogger(__name__)
        logger.debug(f"Starting get_or_create_system_trace for {system_id}")
        
        def _get_or_create():
            logger.debug("Inside _get_or_create")
            if system_id not in self._traces:
                logger.debug(f"Creating new system trace for {system_id}")
                self._traces[system_id] = SystemTrace(
                    system_id=system_id,
                    partition=[EventPartitionElement(partition_index=0, events=[])],
                    current_partition_index=0,
                )
            logger.debug("Returning system trace")
            return self._traces[system_id]
        
        if _already_locked:
            return _get_or_create()
        else:
            with self._lock:
                logger.debug("Lock acquired in get_or_create_system_trace")
                return _get_or_create()

    def increment_partition(self, system_id: str) -> int:
        """Increment the partition index for a system and create new partition element."""
        logger = logging.getLogger(__name__)
        logger.debug(f"Starting increment_partition for system {system_id}")
        
        with self._lock:
            logger.debug("Lock acquired in increment_partition")
            system_trace = self.get_or_create_system_trace(system_id, _already_locked=True)
            logger.debug(f"Got system trace, current index: {system_trace.current_partition_index}")
            
            system_trace.current_partition_index += 1
            logger.debug(f"Incremented index to: {system_trace.current_partition_index}")
            
            system_trace.partition.append(
                EventPartitionElement(
                    partition_index=system_trace.current_partition_index, events=[]
                )
            )
            logger.debug("Added new partition element")
            
            return system_trace.current_partition_index

    def add_event(self, system_id: str, event: Event):
        """Add an event to the appropriate partition of the system trace."""
        self.logger.debug(f"Attempting to add event for system {system_id}")
        
        try:
            # Use a timeout on the lock to prevent deadlocks
            if not self._lock.acquire(timeout=5):  # 5 second timeout
                self.logger.error("Failed to acquire lock within timeout period")
                return
                
            try:
                self.logger.debug("Lock acquired")
                system_trace = self.get_or_create_system_trace(system_id)
                self.logger.debug(f"Got system trace for {system_id}")

                # Find the current partition element
                current_partition = next(
                    (
                        p
                        for p in system_trace.partition
                        if p.partition_index == event.partition_index
                    ),
                    None,
                )
                self.logger.debug(f"Found partition: {current_partition is not None}")

                if current_partition is None:
                    self.logger.error(f"No partition found for index {event.partition_index}")
                    raise ValueError(
                        f"No partition found for index {event.partition_index} "
                        f"in system {system_id}"
                    )

                current_partition.events.append(event)
                self.logger.debug(f"Event appended to partition. Total events: {len(current_partition.events)}")

            finally:
                self._lock.release()
                self.logger.debug("Lock released")

            # Create a span for the event outside the lock
            self.logger.debug("About to create span")
            with tracer.start_as_current_span(event.event_type) as span:
                self.logger.debug("Span created")
                span.set_attribute("system.id", system_id)
                span.set_attribute("event.opened", event.opened)
                span.set_attribute("event.closed", event.closed)
                span.set_attribute("event.partition_index", event.partition_index)
                self.logger.debug("Span attributes set")

                for step in event.agent_compute_steps:
                    self.logger.debug(f"Adding compute step {step.event_order}")
                    span.add_event(
                        "agent_compute",
                        {
                            "order": step.event_order,
                            "began": step.compute_began,
                            "ended": step.compute_ended,
                            "input": step.compute_input,
                            "output": step.compute_output,
                        },
                    )
                self.logger.debug("Finished processing span")

        except Exception as e:
            self.logger.error(f"Error in add_event: {str(e)}", exc_info=True)
            raise

    def get_system_traces(self) -> List[SystemTrace]:
        """Get all system traces."""
        with self._lock:
            return list(self._traces.values())

    def get_system_traces_json(self) -> str:
        """Get all system traces as JSON."""
        with self._lock:
            return json.dumps(
                [
                    {
                        "system_id": trace.system_id,
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
