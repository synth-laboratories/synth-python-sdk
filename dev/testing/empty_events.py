import asyncio

import pytest

from synth_sdk.tracing.abstractions import Dataset, RewardSignal, TrainingQuestion
from synth_sdk.tracing.decorators import trace_system_async
from synth_sdk.tracing.events.store import event_store


@pytest.mark.asyncio
async def test_single_async_event():
    # Clear any old data
    event_store.__init__()

    class AsyncTestAgent:
        def __init__(self):
            self.system_instance_id = "agent_1"
            self.system_name = "TestAsyncAgent"

        @trace_system_async(
            origin="agent",
            event_type="test_event",
            manage_event="create",  # Notice only create, never “end”
            increment_partition=True,
            verbose=True,
        )
        async def do_something(self):
            return "Async result"

    # Create and use one event
    agent = AsyncTestAgent()
    await agent.do_something()

    # Prepare minimal dataset
    dataset = Dataset(
        questions=[TrainingQuestion(id="q1", intent="Test intent", criteria="Test")],
        reward_signals=[
            RewardSignal(
                question_id="q1",
                system_instance_id=agent.system_instance_id,
                reward=1.0,
            )
        ],
    )

    # Perform upload
    # (We expect it to close out the one active event. Before the fix, this fails.)
    # await upload(dataset=dataset)

    # Now check if at least 1 event is recorded
    traces = event_store.get_system_traces()
    print(traces)
    # We expect exactly 1 trace with at least 1 event recorded
    assert len(traces) == 1, "No traces found, the single event never got stored!"
    assert (
        len(traces[0].partition[1].events) == 1
    ), "Expected 1 event; event is missing!"


if __name__ == "__main__":
    asyncio.run(test_single_async_event())
