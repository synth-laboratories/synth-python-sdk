import asyncio
import logging
import os
import uuid  # Added import for UUID generation

from dotenv import load_dotenv

from synth_sdk import AsyncOpenAI, trace_event_async, upload
from synth_sdk.tracing.abstractions import Dataset, RewardSignal, TrainingQuestion

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


os.environ["SYNTH_LOGGING_MODE"] = "deferred"


class MathAgent:
    def __init__(self):
        self.system_instance_id = "math_agent_async"
        self.system_name = "math_agent"
        self.client = AsyncOpenAI()
        logger.debug("OpenAI client initialized")

    @trace_event_async(
        event_type="math_solution",
    )
    async def solve_math_problem(self, problem: str) -> str:
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a math problem solver. Provide step-by-step solutions.",
                },
                {"role": "user", "content": problem},
            ],
            temperature=0,
        )

        solution = response.choices[0].message.content
        return solution


async def run_test():
    agent = MathAgent()

    try:
        # Test math problem
        problem = "If x + 5 = 12, what is x?"

        # Solve the problem
        solution = await agent.solve_math_problem(problem)
        print(f"\nProblem: {problem}")
        print(f"Solution: {solution}")

        # Generate unique UUID for the training question
        question_uuid = str(uuid.uuid4())

        # Create dataset for upload
        dataset = Dataset(
            questions=[
                TrainingQuestion(
                    id=question_uuid,  # Using UUID for unique id
                    intent="Solve algebraic equation",
                    criteria="Provide correct step-by-step solution",
                    question_id="math_q1",
                )
            ],
            reward_signals=[
                RewardSignal(
                    question_id="math_q1",
                    system_instance_id=agent.system_instance_id,
                    reward=1.0,
                    annotation="Math solution provided",
                )
            ],
        )

        # Upload traces
        response, questions_json, reward_signals_json, traces_json = upload(
            dataset=dataset, verbose=True
        )
        print("Upload successful!")

    except Exception as e:
        logger.error("Error during execution: %s", str(e), exc_info=True)
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_test())
