# import logging
# import uuid

# import pytest
# from pydantic import BaseModel
# from zyk import LM

# from synth_sdk.tracing.abstractions import (
#     ArbitraryInputs,
#     MessageInputs,
# )
# from synth_sdk.tracing.decorators import trace_system_async
# from synth_sdk.tracing.trackers import synth_tracker_async
# from synth_sdk.tracing.utils import get_system_id

# # Override logging configuration
# logger = logging.getLogger("synth_sdk")
# #logger.setLevel(logging.DEBUG)
# logger.propagate = True

# # Add a console handler if none exists
# if not logger.handlers:
#     ch = logging.StreamHandler()
#     #ch.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)


# class ArbitraryResponseModel(BaseModel):
#     result: list[str]


# class TestEmptyComputeInputs:
#     def __init__(self):
#         self.system_instance_id = str(uuid.uuid4())
#         self.system_name = "test_agent"
#         #self.system_id = get_system_id(self.system_name)
#         # logger.debug(
#         #     f"Initialized TestEmptyComputeInputs with system_instance_id: {self.system_instance_id}"
#         # )
#         self.lm = LM(
#             model_name="gpt-4o-mini",
#             formatting_model_name="gpt-4o-mini",
#             synth_logging=True,
#             temperature=0.0,
#         )

#     @trace_system_async(
#         origin="agent",
#         event_type="test",
#         manage_event="create",
#         increment_partition=True,
#     )
#     async def action_with_empty_inputs(self):
#         # logger.debug("Entering action_with_empty_inputs")
#         # Get traced data before return
#         inputs, outputs = synth_tracker_async.get_traced_data()
#         # logger.debug(f"Current traced inputs: {inputs}")
#         # logger.debug(f"Current traced outputs: {outputs}")

#         return await self.lm.respond_async(
#             system_message="You are a helpful assistant.",
#             user_message="Say hello!",
#             response_model=ArbitraryResponseModel,
#         )


# @pytest.mark.asyncio
# async def test_empty_compute_inputs_detection():
#     # Create test agent
#     agent = TestEmptyComputeInputs()

#     # Execute test action
#     response = await agent.action_with_empty_inputs()

#     # Get the event from the event store
#     from synth_sdk.tracing.events.store import event_store

#     traces = event_store.get_system_traces()

#     # There should be exactly one trace
#     assert len(traces) == 1
#     trace = traces[0]

#     # There should be one partition
#     assert len(trace.partition) == 2
#     partition = trace.partition[1]

#     # There should be one event
#     assert len(partition.events) == 1
#     event = partition.events[0]

#     # The event should have agent compute steps
#     assert len(event.agent_compute_steps) > 0

#     # The compute step should have inputs
#     compute_step = event.agent_compute_steps[0]
#     assert len(compute_step.compute_input) > 0, "Compute inputs should not be empty"

#     # Verify there is at least one input item
#     print(compute_step.compute_input)
#     print(compute_step.compute_output)
#     input_found = False
#     for input_item in compute_step.compute_input:
#         if isinstance(input_item, (MessageInputs, ArbitraryInputs)):
#             input_found = True
#             break

#     assert input_found, "No valid input items found in compute_input"


# class TestAgent:
#     def __init__(self):
#         self.system_instance_id = "test_agent"
#         self.system_instance_id = "test_instance"
#         self.system_name = "test_agent"
