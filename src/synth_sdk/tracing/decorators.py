import functools
from opentelemetry import trace
from typing import Any, Callable
import asyncio
from ..tracing.config import tracer  # Import the tracer from your config
#from src.exfil.ot_config import tracer  # Import the tracer from your config

def trace_lm_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to trace LM calls and log history steps.
    Includes agent_id to associate spans with agents.
    Supports both synchronous and asynchronous functions.
    """
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            agent_id = getattr(self, 'agent_id', 'unknown_agent')  # Retrieve agent_id from the instance
            with tracer.start_as_current_span(func.__name__) as span:
                span.set_attribute("function", func.__name__)
                span.set_attribute("agent.id", agent_id)
                try:
                    result = await func(self, *args, **kwargs)
                    
                    if hasattr(self, 'history') and self.history.steps and len(self.history.steps) > 1:
                        span.add_event("Last Step", {"history": str(self.history.steps[-2].to_dict())})
                    
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    print(f"Span for function {func.__name__} encountered an error: {e}")
                    raise e
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            agent_id = getattr(self, 'agent_id', 'unknown_agent')  # Retrieve agent_id from the instance
            with tracer.start_as_current_span(func.__name__) as span:
                span.set_attribute("function", func.__name__)
                span.set_attribute("agent.id", agent_id)
                try:
                    result = func(self, *args, **kwargs)
                    
                    if hasattr(self, 'history') and self.history.steps and len(self.history.steps) > 1:
                        span.add_event("Last Step", {"history": str(self.history.steps[-2].to_dict())})
                    
                    #print(f"Finished span for function: {func.__name__}")
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    print(f"Span for function {func.__name__} encountered an error: {e}")
                    raise e
        return sync_wrapper