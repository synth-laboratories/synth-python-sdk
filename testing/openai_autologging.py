import asyncio
import logging
import json
import os
from typing import List
from synth_sdk.provider_support.openai_lf import AsyncOpenAI
from synth_sdk.tracing.decorators import trace_system
from synth_sdk.tracing.events.store import event_store
from synth_sdk.tracing.abstractions import Event, SystemTrace

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class OpenAIAgent:
    def __init__(self):
        self.system_id = "openai_agent_async_test"
        logger.debug("Initializing OpenAIAgent with system_id: %s", self.system_id)
        self.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Replace with your actual API key
        load_dotenv()
    @trace_system(
        origin="agent",
        event_type="openai_completion",
        manage_event="create",
        increment_partition=True,
        verbose=True,
    )
    async def get_completion(self, prompt: str) -> str:
        logger.debug("Sending prompt to OpenAI: %s", prompt)
        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages = [{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            completion_text = response.choices[0].message.content
            logger.debug("Received completion: %s", completion_text)
            return completion_text
        except Exception as e:
            logger.error("Error during OpenAI call: %s", str(e), exc_info=True)
            raise

async def run_test():
    logger.info("Starting OpenAI Agent Async Test")
    agent = OpenAIAgent()
    prompt = "Explain the theory of relativity in simple terms."

    try:
        completion = await agent.get_completion(prompt)
        print(f"OpenAI Completion:\n{completion}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Retrieve and display traces from the event store
    logger.info("Retrieving system traces from event store")
    traces: List[SystemTrace] = event_store.get_system_traces()
    print("\nRetrieved System Traces:")
    for trace in traces:
        print(json.dumps(trace.to_dict(), indent=2))

if __name__ == "__main__":
    asyncio.run(run_test())
