import random

import httpx
import requests
from requests.adapters import HTTPAdapter

from synth_sdk.tracing.config import TracingConfig


class ClientManager:
    """Singleton manager for HTTP clients used in logging"""

    _instance = None
    _sync_client = None
    _async_client = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, config: TracingConfig):
        """Initialize or return the singleton instance with given config"""
        if cls._instance is None:
            cls._instance = cls()
        cls._instance._config = config
        return cls._instance

    @property
    def sync_client(self) -> requests.Session:
        """Get or create the synchronous HTTP client"""
        if self._sync_client is None:
            self._sync_client = requests.Session()
            self._sync_client.headers["Authorization"] = (
                f"Bearer {self._config.api_key}"
            )

            # Configure the adapter with connection pooling
            adapter = HTTPAdapter(
                pool_connections=100,
                pool_maxsize=100,
                max_retries=0,  # We handle retries ourselves
                pool_block=True,
            )
            self._sync_client.mount("https://", adapter)
            self._sync_client.mount("http://", adapter)

        return self._sync_client

    @property
    def async_client(self) -> httpx.AsyncClient:
        """Get or create the asynchronous HTTP client"""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self._config.timeout,
                headers={"Authorization": f"Bearer {self._config.api_key}"},
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=30.0,
                ),
            )
        return self._async_client

    def calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter for retries

        Args:
            attempt: The current retry attempt number (0-based)

        Returns:
            The number of seconds to wait before the next retry
        """
        # Calculate base delay with exponential backoff
        base_delay = min(self._config.retry_backoff**attempt, 60)  # Cap at 60 seconds

        # Add random jitter (±10% of base_delay)
        jitter = random.uniform(-0.1 * base_delay, 0.1 * base_delay)

        return max(0.0, base_delay + jitter)  # Ensure non-negative delay

    async def aclose(self):
        """Close the async client (call this during cleanup)"""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    def close(self):
        """Close the sync client (call this during cleanup)"""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
