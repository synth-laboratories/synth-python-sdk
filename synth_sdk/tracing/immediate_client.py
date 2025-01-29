import asyncio
import time
from typing import Dict

from synth_sdk.tracing.abstractions import Event
from synth_sdk.tracing.base_client import BaseAsyncLogClient, BaseLogClient
from synth_sdk.tracing.client_manager import ClientManager
from synth_sdk.tracing.config import TracingConfig


class ImmediateLogClient(BaseLogClient):
    """Synchronous client for immediate logging of events"""

    def __init__(self, config: TracingConfig):
        super().__init__(config)
        self.client_manager = ClientManager.initialize(config)

    def send_event(self, event: Event, system_info: Dict[str, str]) -> bool:
        """Send a single event with retries and fallback"""
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

        self._handle_failure(event, system_info, last_exception)
        return False


class AsyncImmediateLogClient(BaseAsyncLogClient):
    """Asynchronous client for immediate logging of events"""

    def __init__(self, config: TracingConfig):
        super().__init__(config)
        self.client_manager = ClientManager.initialize(config)

    async def send_event(self, event: Event, system_info: Dict[str, str]) -> bool:
        """Send a single event with retries and fallback (async version)"""
        payload = self._prepare_payload(event, system_info)
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client_manager.async_client.post(
                    f"{self.config.base_url}/v1/traces/stream",
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
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
                await asyncio.sleep(backoff)

        self._handle_failure(event, system_info, last_exception)
        return False
