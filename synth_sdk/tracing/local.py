import logging
import threading
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Thread-local storage for active events and system_instance_id
# Used for synchronous tracing
_local = threading.local()
# Used for asynchronous tracing
system_name_var: ContextVar[str] = ContextVar("system_name", default=None)
system_id_var: ContextVar[str] = ContextVar("system_id", default=None)
system_instance_id_var: ContextVar[str] = ContextVar("system_instance_id", default=None)
active_events_var: ContextVar[dict] = ContextVar("active_events", default={})
