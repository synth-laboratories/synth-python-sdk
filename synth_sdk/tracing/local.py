from typing import Callable, Optional, Set, Literal, Any, Dict, Tuple, Union
from functools import wraps
import threading
import time
import logging
import inspect
import contextvars
from pydantic import BaseModel


logger = logging.getLogger(__name__)

# Thread-local storage for active events and system_id
_local = threading.local()
