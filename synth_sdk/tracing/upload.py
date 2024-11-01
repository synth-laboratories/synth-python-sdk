from typing import List, Union, Optional
from pydantic import BaseModel
import requests
import logging
import os
import time
from synth_sdk.tracing.events.store import event_store
import json

class TrainingQuestion(BaseModel):
    intent: str
    criteria: str
    question_id: Optional[str] = None

    def to_dict(self):
        return {
            "intent": self.intent,
            "criteria": self.criteria,
        }


class RewardSignal(BaseModel):
    question_id: Optional[str] = None
    system_id: str
    reward: Union[float, int, bool]
    annotation: Optional[str] = None

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "system_id": self.system_id,
            "reward": self.reward,
            "annotation": self.annotation,
        }


class Dataset(BaseModel):
    questions: List[TrainingQuestion]
    reward_signals: List[RewardSignal]

    def to_dict(self):
        return {
            "questions": [question.to_dict() for question in self.questions],
            "reward_signals": [signal.to_dict() for signal in self.reward_signals],
        }

def validate_json(data: dict) -> None:
    """
    Validate that a dictionary contains only JSON-serializable values.

    Args:
        data: Dictionary to validate for JSON serialization

    Raises:
        ValueError: If the dictionary contains non-serializable values
    """
    try:
        json.dumps(data)
    except (TypeError, OverflowError) as e:
        raise ValueError(f"Contains non-JSON-serializable values: {e}. {data}")


def send_system_traces(
    dataset: Dataset, base_url: str, api_key: str
) -> requests.Response:
    """Send all system traces and dataset metadata to the server."""
    # Get the token using the API key
    token_url = f"{base_url}/token"
    token_response = requests.get(
        token_url, headers={"customer_specific_api_key": api_key}
    )
    token_response.raise_for_status()
    access_token = token_response.json()["access_token"]
    traces = event_store.get_system_traces()
    #print("Traces: ", traces)
    # Send the traces with the token
    api_url = f"{base_url}/upload/"
    
    payload = {
        "traces": [trace.to_dict() for trace in traces],  # Convert SystemTrace objects to dicts
        "dataset": dataset.to_dict()
    }
    
    validate_json(payload)  # Validate the entire payload

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Upload ID: {response.json().get('upload_id')}")
        return response
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        raise
    except Exception as err:
        logging.error(f"An error occurred: {err}")
        raise


async def upload(dataset: Dataset, verbose: bool = False):
    """Upload all system traces and dataset to the server."""
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        raise ValueError("SYNTH_API_KEY environment variable not set")

    # End all active events before uploading
    from synth_sdk.tracing.decorators import _local
    if hasattr(_local, "active_events"):
        for event_type, event in _local.active_events.items():
            if event and event.closed is None:
                event.closed = time.time()
                if hasattr(_local, "system_id"):
                    try:
                        event_store.add_event(_local.system_id, event)
                        if verbose:
                            print(f"Closed and stored active event: {event_type}")
                    except Exception as e:
                        logging.error(f"Failed to store event {event_type}: {str(e)}")
        _local.active_events.clear()

    try:
        response = send_system_traces(
            dataset=dataset, base_url="https://agent-learning.onrender.com", api_key=api_key
        )

        if verbose:
            print("Response status code:", response.status_code)
            if response.status_code == 202:
                traces = event_store.get_system_traces()
                print(f"Upload successful - sent {len(traces)} system traces.")
                print(
                    f"Dataset included {len(dataset.questions)} questions and {len(dataset.reward_signals)} reward signals."
                )

        return response
    except requests.exceptions.HTTPError as e:
        if verbose:
            print("HTTP error occurred:", e)
            traces = event_store.get_system_traces()
            print("\nTraces:")
            print(json.dumps([trace.to_dict() for trace in traces], indent=2))
            print("\nDataset:")
            print(json.dumps(dataset.to_dict(), indent=2))
        raise e
