import asyncio
import logging
import time
from typing import Dict

from synth_sdk.tracing.abstractions import Event
from synth_sdk.tracing.client_manager import ClientManager
from synth_sdk.tracing.config import TracingConfig
from synth_sdk.tracing.log_client_base import BaseAsyncLogClient, BaseLogClient

logger = logging.getLogger(__name__)


class ImmediateLogClient(BaseLogClient):
    """Synchronous client for immediate logging of events"""

    def __init__(self, config: TracingConfig):
        super().__init__(config)
        self.client_manager = ClientManager.initialize(config)

    def send_event(self, event: Event, system_info: Dict[str, str]) -> bool:
        """Send a single event with retries and fallback"""
        from synth_sdk.tracing.retry_queue import (
            retry_queue,  # Import here to avoid circular import
        )

        payload = self._prepare_payload(event, system_info)
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                response = self.client_manager.sync_client.post(
                    f"{self.config.base_url}/v1/traces/stream",
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                response_data = response.json()
                event.id = response_data.get("event_id")  # Store the event_id
                self._handle_success()
                return True
            except Exception as e:
                last_exception = e
                status_code = getattr(e, "response", None)
                if status_code:
                    status_code = status_code.status_code

                if not self._should_retry(attempt, status_code):
                    break

                backoff = self.client_manager.calculate_backoff(attempt)
                time.sleep(backoff)

        # If we get here, all immediate retries failed
        self._handle_failure(event, system_info, last_exception)

        # Add to retry queue for later processing
        retry_queue.add_failed_event(event, system_info)
        return False


class AsyncImmediateLogClient(BaseAsyncLogClient):
    """Asynchronous client for immediate logging of events"""

    def __init__(self, config: TracingConfig):
        super().__init__(config)
        self.client_manager = ClientManager.initialize(config)

    async def send_event(self, event: Event, system_info: Dict[str, str]) -> bool:
        """Send a single event with retries and fallback (async version)"""
        from synth_sdk.tracing.retry_queue import retry_queue

        payload = self._prepare_payload(event, system_info)
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.client_manager.async_client as client:
                    response = await client.post(
                        f"{self.config.base_url}/v1/traces/stream",
                        json=payload,
                        timeout=self.config.timeout,
                    )
                    response.raise_for_status()
                    response_data = await response.json()
                    event.id = response_data.get("event_id")  # Store the event_id
                    return True

            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    backoff = self.client_manager.calculate_backoff(attempt)
                    await asyncio.sleep(backoff)

        # Only reach here if all retries failed
        retry_queue.add_failed_event(event, system_info)
        return False
