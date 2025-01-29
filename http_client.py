from __future__ import annotations

import ssl
from typing import Any, Dict, Optional

import httpx
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from client_manager import ClientManager
from config import TracingConfig


class HTTPClient:
    def __init__(self, config: TracingConfig):
        self.config = config
        self._client_manager = None
        self._session = None
        self.ssl_context = self._create_ssl_context()

    async def _get_client_manager(self) -> ClientManager:
        if not self._client_manager:
            self._client_manager = await ClientManager.get_instance()
            self._client_manager.configure(self.config)
        return self._client_manager

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create a secure SSL context with modern settings."""
        context = ssl.create_default_context()
        # Use highest available SSL/TLS version
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        # Set secure cipher suites
        context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20")
        context.options |= ssl.OP_NO_COMPRESSION  # Disable compression (CRIME attack)
        return context

    def _create_sync_session(self) -> requests.Session:
        """Create a requests Session with retry logic and connection pooling."""
        if not self._session:
            session = requests.Session()

            # Configure retries
            retry_strategy = Retry(
                total=self.config.retry.max_retries,
                backoff_factor=self.config.retry.base_delay_seconds,
                status_forcelist=[408, 429, 500, 502, 503, 504],
                allowed_methods=[
                    "HEAD",
                    "GET",
                    "PUT",
                    "DELETE",
                    "OPTIONS",
                    "TRACE",
                    "POST",
                ],
            )

            # Configure connection pooling
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=self.config.connection.pool_size,
                pool_maxsize=self.config.connection.pool_size,
                pool_block=True,
            )

            session.mount("http://", adapter)
            session.mount("https://", adapter)

            # Set default headers and timeouts
            session.headers.update(
                {
                    "User-Agent": "SynthSDK/1.0",
                    "Accept": "application/json",
                }
            )

            if self.config.api_key:
                session.headers["Authorization"] = f"Bearer {self.config.api_key}"

            self._session = session

        return self._session

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get configured async client from client manager."""
        client_manager = await self._get_client_manager()
        return await client_manager.get_async_client()

    async def async_request(
        self,
        method: str,
        endpoint: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> httpx.Response:
        """Make an async HTTP request."""
        client = await self._get_async_client()

        merged_headers = {
            "User-Agent": "SynthSDK/1.0",
            "Accept": "application/json",
        }
        if self.config.api_key:
            merged_headers["Authorization"] = f"Bearer {self.config.api_key}"
        if headers:
            merged_headers.update(headers)

        response = await client.request(
            method=method,
            url=endpoint,
            json=json,
            params=params,
            headers=merged_headers,
            verify=self.ssl_context,
        )
        response.raise_for_status()
        return response

    def sync_request(
        self,
        method: str,
        endpoint: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        """Make a synchronous HTTP request."""
        session = self._create_sync_session()

        merged_headers = session.headers.copy()
        if headers:
            merged_headers.update(headers)

        response = session.request(
            method=method,
            url=f"{self.config.base_url}{endpoint}",
            json=json,
            params=params,
            headers=merged_headers,
            timeout=(
                self.config.connection.connect_timeout_seconds,
                self.config.connection.read_timeout_seconds,
            ),
            verify=True,  # Uses system CA certificates
        )
        response.raise_for_status()
        return response

    async def close(self) -> None:
        """Close all clients and sessions."""
        if self._session:
            self._session.close()
            self._session = None

        if self._client_manager:
            client_manager = await self._get_client_manager()
            client_manager._close_clients()
