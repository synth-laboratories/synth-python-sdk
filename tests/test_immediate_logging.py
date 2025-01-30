import logging
import os
from uuid import uuid4

from dotenv import load_dotenv

from synth_sdk.provider_support.openai_lf import AsyncOpenAI as SynthAsyncOpenAI
from synth_sdk.tracing.client_manager import ClientManager
from synth_sdk.tracing.decorators import get_tracing_config, trace_system_async
from synth_sdk.tracing.events.store import event_store
from synth_sdk.tracing.retry_queue import retry_queue

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()


class AsyncOpenAIStreamingAgent:
    def __init__(self):
        # Use the Synth-wrapped AsyncOpenAI client directly
        self.client = SynthAsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_instance_id = str(uuid4())
        self.system_name = "TestAsyncOpenAIStreamingAgent"

    @trace_system_async(
        origin="agent",
        event_type="plan",
        manage_event="create_and_end",
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
        manage_event="create_and_end",
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


async def setup():
    """Setup function to prepare the test environment."""
    logger.info("=== SETUP STARTING ===")

    # Setup - runs before each test
    logger.info("Clearing event store and retry queue")
    event_store.__init__()
    retry_queue.queue.clear()

    # Reset environment to default state
    logger.info("Setting up environment variables")
    os.environ["SYNTH_LOGGING_MODE"] = "instant"
    os.environ["SYNTH_ENDPOINT_OVERRIDE"] = "https://agent-learning.onrender.com"
    # Force new config to be read from env
    logger.info("Initializing client manager")
    config = get_tracing_config()
    logger.info(f"Logging mode set to: {config.mode}")
    logger.info(f"API Key present: {bool(config.api_key)}")  # Log API key presence
    logger.info(f"Base URL: {config.base_url}")
    ClientManager.initialize(config)
    logger.info("Setup complete")


async def teardown():
    """Teardown function to clean up after tests."""
    logger.info("=== TEARDOWN STARTING ===")

    # Close the async client to ensure all tasks complete
    logger.info("Closing async client")
    manager = ClientManager.initialize(get_tracing_config())
    await manager.aclose()
    logger.info("Async client closed")

    # Clear the stores
    logger.info("Clearing stores")
    event_store.__init__()
    retry_queue.queue.clear()

    # Reset environment back to default state
    logger.info("Resetting environment variables")
    os.environ["SYNTH_LOGGING_MODE"] = "deferred"
    os.environ["SYNTH_ENDPOINT_OVERRIDE"] = "http://localhost:8000"
    logger.info("Teardown complete")


async def test_immediate_logging():
    logger.info("=== TEST_IMMEDIATE_LOGGING STARTING ===")

    await setup()

    try:
        # Create and use agent
        agent = AsyncOpenAIStreamingAgent()

        # Verify event store is empty at start
        initial_traces = event_store.get_system_traces()
        logger.info(f"Initial event store state: {len(initial_traces)} traces")
        assert len(initial_traces) == 0, "Event store should be empty at test start"

        # Single calls to each method
        logger.info("Making plan call")
        plan = await agent.plan("What's 2+2?")
        logger.info("Making execute call")
        solution = await agent.execute(plan)

        # Check if events were recorded in event_store
        traces = event_store.get_system_traces()
        logger.info(f"Final event store state: {len(traces)} traces")
        assert len(traces) == 1, f"Expected exactly 1 trace, but found {len(traces)}"
        assert (
            len(traces[0].partition[1].events) == 1
        ), "Plan event missing from event store!"
        assert (
            len(traces[0].partition[2].events) == 1
        ), "Execute event missing from event store!"

        # Check retry queue is empty
        logger.info(f"Retry queue state: {len(retry_queue.queue)} events")
        assert len(retry_queue.queue) == 0, "Events found in retry queue!"

    finally:
        await teardown()


# Add a main block to run the test
if __name__ == "__main__":
    import asyncio

    asyncio.run(test_immediate_logging())
