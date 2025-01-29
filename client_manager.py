from __future__ import annotations

import asyncio
import random
from typing import ClassVar, Dict, Optional

import httpx

from config import TracingConfig


class ClientManager:
    _instance: ClassVar[Optional[ClientManager]] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._config: Optional[TracingConfig] = None
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._credentials_cache: Dict[str, str] = {}

    @classmethod
    async def get_instance(cls) -> ClientManager:
        if not cls._instance:
            async with cls._lock:
                if not cls._instance:
                    cls._instance = ClientManager()
        return cls._instance

    def configure(self, config: TracingConfig) -> None:
        """Configure the client manager with new settings."""
        self._config = config
        self._credentials_cache = {
            "api_key": config.api_key,
            "api_secret": config.api_secret,
        }

        # Reset clients to apply new configuration
        self._close_clients()

    def _close_clients(self) -> None:
        """Close existing clients if they exist."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
        if self._async_client:
            asyncio.create_task(self._async_client.aclose())
            self._async_client = None

    def get_sync_client(self) -> httpx.Client:
        """Get or create synchronized HTTP client with connection pooling."""
        if not self._sync_client and self._config:
            limits = httpx.Limits(
                max_connections=self._config.connection.pool_size,
                max_keepalive_connections=self._config.connection.pool_size,
            )
            timeout = httpx.Timeout(
                connect=self._config.connection.connect_timeout_seconds,
                read=self._config.connection.read_timeout_seconds,
            )
            self._sync_client = httpx.Client(
                base_url=self._config.base_url, timeout=timeout, limits=limits
            )
        return self._sync_client

    async def get_async_client(self) -> httpx.AsyncClient:
        """Get or create asynchronous HTTP client with connection pooling."""
        if not self._async_client and self._config:
            limits = httpx.Limits(
                max_connections=self._config.connection.pool_size,
                max_keepalive_connections=self._config.connection.pool_size,
            )
            timeout = httpx.Timeout(
                connect=self._config.connection.connect_timeout_seconds,
                read=self._config.connection.read_timeout_seconds,
            )
            self._async_client = httpx.AsyncClient(
                base_url=self._config.base_url, timeout=timeout, limits=limits
            )
        return self._async_client

    def calculate_backoff(self, retry_number: int) -> float:
        """Calculate backoff time with exponential backoff and jitter."""
        if not self._config:
            raise RuntimeError("ClientManager not configured")

        if retry_number <= 0:
            return 0

        if not self._config.retry.exponential_backoff:
            return self._config.retry.base_delay_seconds

        # Calculate exponential backoff
        delay = min(
            self._config.retry.base_delay_seconds * (2 ** (retry_number - 1)),
            self._config.retry.max_delay_seconds,
        )

        # Add jitter (±25% of delay)
        jitter = delay * 0.25
        delay = random.uniform(delay - jitter, delay + jitter)

        return delay

    @property
    def config(self) -> Optional[TracingConfig]:
        """Get the current configuration."""
        return self._config
