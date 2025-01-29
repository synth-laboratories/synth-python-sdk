from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LoggingMode(Enum):
    INSTANT = "instant"
    DEFERRED = "deferred"


@dataclass
class ConnectionSettings:
    pool_size: int = 10
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 30.0


@dataclass
class RetrySettings:
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_backoff: bool = True


@dataclass
class TracingConfig:
    logging_mode: LoggingMode
    connection: ConnectionSettings = ConnectionSettings()
    retry: RetrySettings = RetrySettings()
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: str = "https://api.tracing-service.com/v1"

    def __post_init__(self):
        if not isinstance(self.connection, ConnectionSettings):
            self.connection = ConnectionSettings(**self.connection)
        if not isinstance(self.retry, RetrySettings):
            self.retry = RetrySettings(**self.retry)
