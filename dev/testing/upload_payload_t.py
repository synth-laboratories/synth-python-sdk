import asyncio
import logging
import time
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import requests

from synth_sdk.tracing.abstractions import (
    Dataset,
    RewardSignal,
    SystemTrace,
    TrainingQuestion,
)
from synth_sdk.tracing.decorators import _local, trace_system
from synth_sdk.tracing.events.store import event_store, instance_system_id_var
from synth_sdk.tracing.trackers import SynthTracker
from synth_sdk.tracing.upload import createPayload, upload
from synth_sdk.tracing.utils import get_instance_system_id

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed from CRITICAL to DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Unit Test Configuration:
# ===============================
questions = ["What's the capital of France?"]
mock_llm_response = "The capital of France is Paris."


# This function generates a payload from the data in the dataset to compare the sent payload against
def generate_payload_from_data(dataset: Dataset, traces: List[SystemTrace]) -> Dict:
    payload = {
        "traces": [trace.to_dict() for trace in traces],
        "dataset": dataset.to_dict(),
    }
    return payload


def createPayload_wrapper(
    dataset: Dataset, traces: str, base_url: str, api_key: str
) -> Dict:
    payload = createPayload(dataset, traces)
    response = requests.Response()
    response.status_code = 200
    return response, payload


# ===============================
# Utility Functions
def createUploadDataset(agent):
    dataset = Dataset(
        questions=[
            TrainingQuestion(
                id=f"q{i}",
                intent="Test question",
                criteria="Testing tracing functionality",
            )
            for i in range(len(questions))
        ],
        reward_signals=[
            RewardSignal(
                question_id=f"q{i}",
                system_instance_id=agent.system_instance_id,
                reward=1.0,
                annotation="Test reward",
            )
            for i in range(len(questions))
        ],
    )
    logger.debug(
        "Dataset created with %d questions and %d reward signals",
        len(dataset.questions),
        len(dataset.reward_signals),
    )
    return dataset


def ask_questions(agent):
    # Make multiple LM calls with environment processing
    responses = []
    for i, question in enumerate(questions):
        logger.info("Processing question %d: %s", i, question)
        env_result = agent.process_environment(question)
        logger.debug("Environment processing result: %s", env_result)
        response = agent.make_lm_call(question)
        responses.append(response)
        logger.debug("Response received and stored: %s", response)
    return responses


# ===============================


class TestAgent:
    def __init__(self):
        self.instance_system_id = "test_instance_system_id"
        self.system_instance_id = "test_instance"
        self.system_name = "test_agent"

        # Initialize instance_system_id_var context variable
        instance_system_id = get_instance_system_id(self.system_name)
        instance_system_id_var.set(instance_system_id)

        # Set thread local storage
        _local.instance_system_id = self.instance_system_id
        _local.system_instance_id = self.system_instance_id
        _local.system_name = self.system_name

        logger.debug(
            f"Initialized TestAgent with instance_system_id: {self.instance_system_id}"
        )
        self.lm = MagicMock()
        self.lm.model_name = "gpt-4o-mini-2024-07-18"
        self.lm.respond_sync.return_value = mock_llm_response
        logger.debug("LM initialized")

    @trace_system(
        origin="agent",
        event_type="lm_call",
        manage_event="create",
        increment_partition=True,
        verbose=False,
    )
    def make_lm_call(self, user_message: str) -> str:
        # Track LM interaction using the new SynthTracker form
        SynthTracker.track_lm(
            messages=[{"role": "user", "content": user_message}],
            model_name=self.lm.model_name,
            finetune=False,
        )

        logger.debug("Starting LM call with message: %s", user_message)
        response = self.lm.respond_sync(
            system_message="You are a helpful assistant.", user_message=user_message
        )

        # Track state using the new SynthTracker form
        SynthTracker.track_state(
            variable_name="response",
            variable_value={"role": "assistant", "content": response},
            origin="agent",
            annotation="LLM response",
        )

        logger.debug("LM response received: %s", response)
        return response

    @trace_system(
        origin="environment",
        event_type="environment_processing",
        manage_event="create",
        verbose=False,
    )
    def process_environment(self, input_data: str) -> dict:
        # Track state using the new SynthTracker form
        SynthTracker.track_state(
            variable_name="input_data",
            variable_value=input_data,
            origin="environment",
            annotation="Environment input data",
        )

        result = {"processed": input_data, "timestamp": time.time()}

        # Track state using the new SynthTracker form
        SynthTracker.track_state(
            variable_name="result",
            variable_value=result,
            origin="environment",
            annotation="Environment processing result",
        )
        return result


@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Cleanup after each test
    event_store.__init__()
    if hasattr(_local, "instance_system_id"):
        delattr(_local, "instance_system_id")
    if hasattr(_local, "system_instance_id"):
        delattr(_local, "system_instance_id")
    if hasattr(_local, "system_name"):
        delattr(_local, "system_name")


@pytest.mark.asyncio
@patch("synth_sdk.tracing.upload.send_system_traces", side_effect=createPayload_wrapper)
async def test_upload_async(mock_send_system_traces):
    logger.info("Starting test_upload_async")
    agent = TestAgent()

    # Ask questions
    responses = ask_questions(agent)

    logger.info("Creating dataset for upload")
    dataset = createUploadDataset(agent)

    # Upload traces
    response, questions_json, reward_signals_json, traces_json = await upload(
        dataset=dataset, verbose=True, show_payload=True
    )

    # Assertions
    assert response.status_code == 200
    assert isinstance(questions_json, list)
    assert isinstance(reward_signals_json, list)
    assert isinstance(traces_json, list)


@patch("synth_sdk.tracing.upload.send_system_traces", side_effect=createPayload_wrapper)
def test_upload_sync(mock_send_system_traces):
    logger.info("Starting test_upload_sync")
    agent = TestAgent()

    # Ask questions
    responses = ask_questions(agent)

    logger.info("Creating dataset for upload")
    dataset = createUploadDataset(agent)

    # Upload traces
    logger.info("Attempting to upload traces, non-async version")
    response, questions_json, reward_signals_json, traces_json = upload(
        dataset=dataset, verbose=True, show_payload=True
    )
    logger.info("Upload successful!")

    # Assertions
    assert response.status_code == 200
    assert isinstance(questions_json, list)
    assert isinstance(reward_signals_json, list)
    assert isinstance(traces_json, list)

    logger.info("Resetting event store")
    event_store.__init__()


# Run the tests
if __name__ == "__main__":
    logger.info("Starting main execution")
    asyncio.run(test_upload_async())
    logger.info("Async test completed")
    print("=============================================")
    print("=============================================")
    test_upload_sync()
    logger.info("Sync test completed")
    logger.info("Main execution completed")
