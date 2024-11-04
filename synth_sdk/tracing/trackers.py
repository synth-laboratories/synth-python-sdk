from typing import Union, Optional, Tuple, Literal
import threading
import contextvars
from pydantic import BaseModel
from synth_sdk.tracing.local import logger
from synth_sdk.tracing.config import VALID_TYPES

# This tracker ought to be used for synchronous tracing
class SynthTrackerSync:
    _local = threading.local()

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
