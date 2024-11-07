from typing import Union, Optional, Tuple, Literal
import asyncio
import threading, contextvars
import contextvars
from pydantic import BaseModel
from synth_sdk.tracing.local import logger, _local
from synth_sdk.tracing.config import VALID_TYPES

# This tracker ought to be used for synchronous tracing
class SynthTrackerSync:
    _local = _local

    @classmethod
    def initialize(cls):
        cls._local.initialized = True
        cls._local.inputs = []  # List of tuples: (origin, var)
        cls._local.outputs = []  # List of tuples: (origin, var)

    @classmethod
    def track_input(
        cls,
        var: Union[BaseModel, str, dict, int, float, bool, list, None],
        variable_name: str,
        origin: Literal["agent", "environment"],
        annotation: Optional[str] = None,
    ):
        if not isinstance(var, VALID_TYPES):
            raise TypeError(
                f"Variable {variable_name} must be one of {VALID_TYPES}, got {type(var)}"
            )

        if getattr(cls._local, "initialized", False):
            # Convert Pydantic models to dict schema
            if isinstance(var, BaseModel):
                var = var.model_dump()
            cls._local.inputs.append((origin, var, variable_name, annotation))
            logger.debug(
                f"Traced input: origin={origin}, var_name={variable_name}, annotation={annotation}"
            )
        else:
            raise RuntimeError(
                "Trace not initialized. Use within a function decorated with @trace_system_sync."
            )

    @classmethod
    def track_output(
        cls,
        var: Union[BaseModel, str, dict, int, float, bool, list, None],
        variable_name: str,
        origin: Literal["agent", "environment"],
        annotation: Optional[str] = None,
    ):
        if not isinstance(var, VALID_TYPES):
            raise TypeError(
                f"Variable {variable_name} must be one of {VALID_TYPES}, got {type(var)}"
            )

        if getattr(cls._local, "initialized", False):
            # Convert Pydantic models to dict schema
            if isinstance(var, BaseModel):
                var = var.model_dump()
            cls._local.outputs.append((origin, var, variable_name, annotation))
            logger.debug(
                f"Traced output: origin={origin}, var_name={variable_name}, annotation={annotation}"
            )
        else:
            raise RuntimeError(
                "Trace not initialized. Use within a function decorated with @trace_system_sync."
            )

    @classmethod
    def get_traced_data(cls):
        return getattr(cls._local, "inputs", []), getattr(cls._local, "outputs", [])

    @classmethod
    def finalize(cls):
        # Clean up the thread-local storage
        cls._local.initialized = False
        cls._local.inputs = []
        cls._local.outputs = []
        logger.debug("Finalized trace data")


# Context variables for asynchronous tracing
trace_inputs_var = contextvars.ContextVar("trace_inputs", default=None)
trace_outputs_var = contextvars.ContextVar("trace_outputs", default=None)
trace_initialized_var = contextvars.ContextVar("trace_initialized", default=False)


# This tracker ought to be used for asynchronous tracing
class SynthTrackerAsync:
    @classmethod
    def initialize(cls):
        trace_initialized_var.set(True)
        trace_inputs_var.set([])  # List of tuples: (origin, var)
        trace_outputs_var.set([])  # List of tuples: (origin, var)
        logger.debug("AsyncTrace initialized")

    @classmethod
    def track_input(
        cls,
        var: Union[BaseModel, str, dict, int, float, bool, list, None],
        variable_name: str,
        origin: Literal["agent", "environment"],
        annotation: Optional[str] = None,
    ):
        if not isinstance(var, VALID_TYPES):
            raise TypeError(
                f"Variable {variable_name} must be one of {VALID_TYPES}, got {type(var)}"
            )

        if trace_initialized_var.get():
            # Convert Pydantic models to dict schema
            if isinstance(var, BaseModel):
                var = var.model_dump()
            trace_inputs = trace_inputs_var.get()
            trace_inputs.append((origin, var, variable_name, annotation))
            trace_inputs_var.set(trace_inputs)
            logger.debug(
                f"Traced input: origin={origin}, var_name={variable_name}, annotation={annotation}"
            )
        else:
            raise RuntimeError(
                "Trace not initialized. Use within a function decorated with @trace_system_async."
            )

    @classmethod
    def track_output(
        cls,
        var: Union[BaseModel, str, dict, int, float, bool, list, None],
        variable_name: str,
        origin: Literal["agent", "environment"],
        annotation: Optional[str] = None,
    ):
        if not isinstance(var, VALID_TYPES):
            raise TypeError(
                f"Variable {variable_name} must be one of {VALID_TYPES}, got {type(var)}"
            )

        if trace_initialized_var.get():
            # Convert Pydantic models to dict schema
            if isinstance(var, BaseModel):
                var = var.model_dump()
            trace_outputs = trace_outputs_var.get()
            trace_outputs.append((origin, var, variable_name, annotation))
            trace_outputs_var.set(trace_outputs)
            logger.debug(
                f"Traced output: origin={origin}, var_name={variable_name}, annotation={annotation}"
            )
        else:
            raise RuntimeError(
                "Trace not initialized. Use within a function decorated with @trace_system_async."
            )

    @classmethod
    def get_traced_data(cls) -> Tuple[list, list]:
        return trace_inputs_var.get(), trace_outputs_var.get()

    @classmethod
    def finalize(cls):
        trace_initialized_var.set(False)
        trace_inputs_var.set([])
        trace_outputs_var.set([])
        logger.debug("Finalized async trace data")

# Make traces available globally
synth_tracker_sync = SynthTrackerSync
synth_tracker_async = SynthTrackerAsync

# Generalized SynthTracker class, depending on if an event loop is running (called from async)
# & if the specified tracker is initalized will determine the appropriate tracker to use
class SynthTracker:
    def is_called_by_async():
        try:
            asyncio.get_running_loop()  # Attempt to get the running event loop
            return True  # If successful, we are in an async context
        except RuntimeError:
            return False  # If there's no running event loop, we are in a sync context
        
    # SynthTracker Async & Sync are initalized by the decorators that wrap the 
    # respective async & sync functions
    @classmethod
    def initialize(cls):
        pass

    @classmethod
    def track_input(
        cls,
        var: Union[BaseModel, str, dict, int, float, bool, list, None],
        variable_name: str,
        origin: Literal["agent", "environment"],
        annotation: Optional[str] = None,
        async_sync: Literal["async", "sync", ""] = "", # Force the tracker to be async or sync
    ):
        
        if async_sync == "async" or cls.is_called_by_async() and trace_initialized_var.get():
            logger.debug("Using async tracker to track input")
            synth_tracker_async.track_input(var, variable_name, origin, annotation)

        # don't want to add the same event to both trackers
        elif async_sync == "sync" or hasattr(synth_tracker_sync._local, "initialized"):
            logger.debug("Using sync tracker to track input")
            synth_tracker_sync.track_input(var, variable_name, origin, annotation)
        else:
            raise RuntimeError("track_input() \n Trace not initialized. Use within a function decorated with @trace_system_async or @trace_system_sync.")
    
    @classmethod
    def track_output(
        cls,
        var: Union[BaseModel, str, dict, int, float, bool, list, None],
        variable_name: str,
        origin: Literal["agent", "environment"],
        annotation: Optional[str] = None,
        async_sync: Literal["async", "sync", ""] = "", # Force the tracker to be async or sync
    ):
        if async_sync == "async" or cls.is_called_by_async() and trace_initialized_var.get():
            logger.debug("Using async tracker to track output")
            synth_tracker_async.track_output(var, variable_name, origin, annotation)

        # don't want to add the same event to both trackers
        elif async_sync == "sync" or hasattr(synth_tracker_sync._local, "initialized"):
            logger.debug("Using sync tracker to track output")
            synth_tracker_sync.track_output(var, variable_name, origin, annotation)
        else:
            raise RuntimeError("track_output() \n Trace not initialized. Use within a function decorated with @trace_system_async or @trace_system_sync.")

        
    # if both trackers have been used, want to return both sets of data
    @classmethod
    def get_traced_data(
        cls,
        async_sync: Literal["async", "sync", ""] = "", # Force only async or sync data to be returned
    ) -> Tuple[list, list]:

        traced_inputs, traced_outputs = [], []

        if async_sync == "async" or async_sync == "":
            # Attempt to get the traced data from the async tracker
            logger.debug("Getting traced data from async tracker")
            traced_inputs_async, traced_outputs_async = synth_tracker_async.get_traced_data()
            traced_inputs.extend(traced_inputs_async)
            traced_outputs.extend(traced_outputs_async)

        if async_sync == "sync" or async_sync == "":
            # Attempt to get the traced data from the sync tracker
            logger.debug("Getting traced data from sync tracker")
            traced_inputs_sync, traced_outputs_sync = synth_tracker_sync.get_traced_data()
            traced_inputs.extend(traced_inputs_sync)
            traced_outputs.extend(traced_outputs_sync)

        # TODO: Test that the order of the inputs and outputs is correct wrt 
        # the order of events since we are combining the two trackers
        return traced_inputs, traced_outputs
        
    # Finalize both trackers
    @classmethod
    def finalize(
        cls,
        async_sync: Literal["async", "sync", ""] = "",         
    ):
        if async_sync == "async" or async_sync == "":
            logger.debug("Finalizing async tracker")
            synth_tracker_async.finalize()

        if async_sync == "sync" or async_sync == "":
            logger.debug("Finalizing sync tracker")
            synth_tracker_sync.finalize()
