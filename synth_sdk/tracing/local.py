from typing import Callable, Optional, Set, Literal, Any, Dict, Tuple, Union
from functools import wraps
import threading
import time
import logging
import inspect
import contextvars
from contextvars import ContextVar

from pydantic import BaseModel


logger = logging.getLogger(__name__)

# Thread-local storage for active events and system_id
# Used for synchronous tracing
_local = threading.local()
# Used for asynchronous tracing
system_id_var: ContextVar[str] = ContextVar("system_id")
active_events_var: ContextVar[dict] = ContextVar("active_events", default={})
