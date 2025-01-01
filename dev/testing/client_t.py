# import pytest
# from pydantic import BaseModel
# from zyk import LM

# from synth_sdk.tracing.abstractions import (
#     ArbitraryOutputs,
#     MessageInputs,
#     MessageOutputs,
# )
# from synth_sdk.tracing.decorators import trace_system_async


# class ArbitraryResponseModel(BaseModel):
#     response: str


# class TestAgent:
#     def __init__(self):
#         self.instance_system_id = "test_agent"
#         self.system_instance_id = "test_instance"
#         self.system_name = "test_agent"
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
#     async def test_action(self):
#         # This should be tracked as a compute input
#         response = await self.lm.respond_async(
#             system_message="You are a helpful assistant.",
#             user_message="Say hello!",
#             response_model=ArbitraryResponseModel,
#         )
#         return response


# @pytest.mark.asyncio
# async def test_compute_inputs_tracking():
#     # Create test agent
#     agent = TestAgent()

#     # Execute test action
#     response = await agent.test_action()

#     # Verify response model
#     assert isinstance(response, ArbitraryResponseModel)

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
#     assert len(compute_step.compute_input) > 0, "No compute inputs were tracked"

#     # Verify the input contains the message and model info
#     input_found = False
#     print(compute_step.compute_input)
#     for input_item in compute_step.compute_input:
#         if isinstance(input_item, MessageInputs):
#             # Check that the roles match
#             assert input_item.messages[0]["role"] == "system"
#             assert input_item.messages[1]["role"] == "user"

#             # Check that the expected content is contained within the actual messages
#             assert "You are a helpful assistant." in input_item.messages[0]["content"]
#             assert input_item.messages[1]["content"] == "Say hello!"
#             input_found = True
#             break

#     assert input_found, "MessageInputs not found in compute_input"

#     # Verify the output contains the response model data
#     output_found = False
#     print("Compute outputs:", compute_step.compute_output)
#     for output_item in compute_step.compute_output:
#         print(f"Checking output item of type: {type(output_item)}")
#         if isinstance(output_item, MessageOutputs):
#             # The response should be in the last message
#             assert isinstance(output_item.messages[-1]["content"], str)
#             output_found = True
#             break
#         elif isinstance(output_item, ArbitraryOutputs):
#             # Check the response in the arbitrary outputs format
#             assert "result" in output_item.outputs
#             assert "response" in output_item.outputs["result"]
#             assert isinstance(output_item.outputs["result"]["response"], str)
#             output_found = True
#             break

#     assert output_found, "No valid output format found in compute_output"
