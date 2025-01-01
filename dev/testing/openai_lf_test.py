import asyncio
from uuid import uuid4

import pytest

from synth_sdk.provider_support.openai_lf import AsyncOpenAI as SynthAsyncOpenAI
from synth_sdk.tracing.abstractions import Dataset, RewardSignal, TrainingQuestion
from synth_sdk.tracing.decorators import trace_system_async
from synth_sdk.tracing.events.store import event_store
from synth_sdk.tracing.upload import upload
import os
from dotenv import load_dotenv
load_dotenv()

class AsyncOpenAIAgent:
    def __init__(self):
        # Use the Synth-wrapped AsyncOpenAI client directly
        self.client = SynthAsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Replace with your key
        self.system_instance_id = str(uuid4())
        self.system_name = "TestAsyncOpenAIAgent"

    @trace_system_async(
        origin="agent",
        event_type="plan",
        manage_event="create",
        increment_partition=True,
        verbose=True,
    )
    async def plan(self, question: str) -> str:
        response = await self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Plan how to solve: {question}"},
            ],
        )
        return response.choices[0].message.content

    @trace_system_async(
        origin="agent",
        event_type="execute",
        manage_event="create",
        increment_partition=True,
        verbose=True,
    )
    async def execute(self, plan: str) -> str:
        response = await self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Execute this plan: {plan}"},
            ],
        )
        return response.choices[0].message.content


@pytest.mark.asyncio
async def test_openai_lf_events():
    # Clear any old data
    event_store.__init__()

    # Create and use agent
    agent = AsyncOpenAIAgent()

    # Single calls to each method
    plan = await agent.plan("What's 2+2?")
    solution = await agent.execute(plan)

    # Prepare minimal dataset
    dataset = Dataset(
        questions=[
            TrainingQuestion(
                id="q1", intent="Test OpenAI LF instrumentation", criteria="Test"
            )
        ],
        reward_signals=[
            RewardSignal(
                question_id="q1",
                system_instance_id=agent.system_instance_id,
                reward=1.0,
            )
        ],
    )

    # Upload and check traces
    #upload(dataset=dataset)

    # Now check if events were recorded
    traces = event_store.get_system_traces()
    print("Traces:", traces)

    # We expect exactly 1 trace with events in partition[1] and partition[2]
    assert len(traces) == 1, "No traces found!"
    assert len(traces[0].partition[1].events) == 1, "Plan event missing!"
    assert len(traces[0].partition[2].events) == 1, "Execute event missing!"


if __name__ == "__main__":
    asyncio.run(test_openai_lf_events())
