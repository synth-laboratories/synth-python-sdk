import logging
import os

from dotenv import load_dotenv

from synth_sdk.provider_support.anthropic import Anthropic
from synth_sdk.tracing.decorators import trace_event_sync

# Load environment variables
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Set instant logging mode
os.environ["SYNTH_LOGGING_MODE"] = "instant"

# Configure logging
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MathAgent:
    def __init__(self):
        self.system_instance_id = "math_agent_sync"
        self.system_name = "math_agent"
        self.client = Anthropic(api_key=anthropic_api_key)
        logger.debug("Anthropic client initialized")

    @trace_event_sync(
        event_type="math_solution",
    )
    def solve_math_problem(self, problem: str) -> str:
        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            system="You are a math problem solver. Provide step-by-step solutions.",
            messages=[
                {"role": "user", "content": problem},
            ],
            temperature=0,
        )

        solution = response.content[0].text
        return solution


def run_test():
    agent = MathAgent()

    try:
        # Test math problem
        problem = "If x + 5 = 12, what is x?"

        # Solve the problem
        solution = agent.solve_math_problem(problem)
        print(f"\nProblem: {problem}")
        print(f"Solution: {solution}")

    except Exception as e:
        logger.error("Error during execution: %s", str(e), exc_info=True)
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    run_test()
