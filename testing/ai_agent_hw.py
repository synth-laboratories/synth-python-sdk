from zyk import LM
from synth_sdk.tracing.decorators import trace_system, _local
from synth_sdk.tracing.upload import upload
from synth_sdk.tracing.abstractions import TrainingQuestion, RewardSignal, Dataset
from synth_sdk.tracing.events.store import event_store
import asyncio
import time
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.CRITICAL,  # Change this to WARNING or ERROR to reduce verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestAgent:
    def __init__(self):
        self.system_id = "test_agent_1"
        logger.debug("Initializing TestAgent with system_id: %s", self.system_id)
        self.lm = LM(
            model_name="gpt-4o-mini-2024-07-18",
            formatting_model_name="gpt-4o-mini-2024-07-18",
            temperature=1,
        )
        logger.debug("LM initialized")

    @trace_system(
        origin="agent",
        event_type="lm_call",
        manage_event="create",
        increment_partition=True,
        log_vars_input={"user_message"},
        log_vars_output={"response"},
        verbose=True
    )
    def make_lm_call(self, user_message: str) -> str:
        logger.debug("Starting LM call with message: %s", user_message)
        response = self.lm.respond_sync(
            system_message="You are a helpful assistant.",
            user_message=user_message
        )
        logger.debug("LM response received: %s", response)
        time.sleep(0.1)  # Simulate some processing time
        return response

async def run_test():
    logger.info("Starting run_test")
    # Create test agent
    agent = TestAgent()
    
    try:
        # List of test questions
        questions = [
            "What's the capital of France?",
            "What's 2+2?",
            "Who wrote Romeo and Juliet?",
        ]
        logger.debug("Test questions initialized: %s", questions)
        
        # Make multiple LM calls
        responses = []
        for i, question in enumerate(questions):
            logger.info("Processing question %d: %s", i, question)
            #print("Running question:")
            try:
                logger.debug("Making LM call for question: %s", question)
                response = agent.make_lm_call(question)
                responses.append(response)
                logger.debug("Response received and stored: %s", response)
                #print(f"Q: {question}\nA: {response}\n")
            except Exception as e:
                logger.error("Error during LM call: %s", str(e), exc_info=True)
                #print(f"Error during LM call: {str(e)}")
                continue

        logger.info("Creating dataset for upload")
        # Create dataset for upload
        dataset = Dataset(
            questions=[
                TrainingQuestion(
                    intent="Test question",
                    criteria="Testing tracing functionality",
                    question_id=f"q{i}"
                )
                for i in range(len(questions))
            ],
            reward_signals=[
                RewardSignal(
                    question_id=f"q{i}",
                    agent_id=agent.system_id,
                    reward=1.0,
                    annotation="Test reward"
                )
                for i in range(len(questions))
            ]
        )
        logger.debug("Dataset created with %d questions and %d reward signals", 
                    len(dataset.questions), len(dataset.reward_signals))

        # Upload traces
        try:
            logger.info("Attempting to upload traces")
            response = await upload(dataset=dataset, verbose=True)
            logger.info("Upload successful!")
            print("Upload successful!")
        except Exception as e:
            logger.error("Upload failed: %s", str(e), exc_info=True)
            print(f"Upload failed: {str(e)}")
            
            # Print debug information
            traces = event_store.get_system_traces()
            logger.debug("Retrieved %d system traces", len(traces))
            print("\nTraces:")
            print(json.dumps([trace.to_dict() for trace in traces], indent=2))
            
            print("\nDataset:")
            print(json.dumps(dataset.to_dict(), indent=2))
    finally:
        logger.info("Starting cleanup")
        # Cleanup
        if hasattr(_local, "active_events"):
            for event_type, event in _local.active_events.items():
                logger.debug("Cleaning up event: %s", event_type)
                if event.closed is None:
                    event.closed = time.time()
                    if hasattr(_local, "system_id"):
                        try:
                            event_store.add_event(_local.system_id, event)
                            logger.debug("Successfully cleaned up event: %s", event_type)
                        except Exception as e:
                            logger.error("Error during cleanup of event %s: %s", 
                                       event_type, str(e), exc_info=True)
                            print(f"Error during cleanup of event {event_type}: {str(e)}")
        logger.info("Cleanup completed")

if __name__ == "__main__":
    logger.info("Starting main execution")
    asyncio.run(run_test())
    logger.info("Main execution completed")
