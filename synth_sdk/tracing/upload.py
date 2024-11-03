from typing import List, Dict, Any
from pydantic import BaseModel, validator
import requests
import logging
import os
import time
from synth_sdk.tracing.events.store import event_store
from synth_sdk.tracing.abstractions import Dataset
import json


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
    # print("Traces: ", traces)
    # Send the traces with the token
    api_url = f"{base_url}/upload/"

    payload = {
        "traces": [
            trace.to_dict() for trace in traces
        ],  # Convert SystemTrace objects to dicts
        "dataset": dataset.to_dict(),
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
        logging.error(
            f"HTTP error occurred: {http_err} - Response Content: {response.text}"
        )
        raise
    except Exception as err:
        logging.error(f"An error occurred: {err}")
        raise


class UploadValidator(BaseModel):
    traces: List[Dict[str, Any]]
    dataset: Dict[str, Any]

    @validator("traces")
    def validate_traces(cls, traces):
        if not traces:
            raise ValueError("Traces list cannot be empty")

        for trace in traces:
            # Validate required fields in each trace
            if "system_id" not in trace:
                raise ValueError("Each trace must have a system_id")
            if "partition" not in trace:
                raise ValueError("Each trace must have a partition")

            # Validate partition structure
            partition = trace["partition"]
            if not isinstance(partition, list):
                raise ValueError("Partition must be a list")

            for part in partition:
                if "partition_index" not in part:
                    raise ValueError(
                        "Each partition element must have a partition_index"
                    )
                if "events" not in part:
                    raise ValueError("Each partition element must have an events list")

                # Validate events
                events = part["events"]
                if not isinstance(events, list):
                    raise ValueError("Events must be a list")

                for event in events:
                    required_fields = [
                        "event_type",
                        "opened",
                        "closed",
                        "partition_index",
                    ]
                    missing_fields = [f for f in required_fields if f not in event]
                    if missing_fields:
                        raise ValueError(
                            f"Event missing required fields: {missing_fields}"
                        )

        return traces

    @validator("dataset")
    def validate_dataset(cls, dataset):
        required_fields = ["questions", "reward_signals"]
        missing_fields = [f for f in required_fields if f not in dataset]
        if missing_fields:
            raise ValueError(f"Dataset missing required fields: {missing_fields}")

        # Validate questions
        questions = dataset["questions"]
        if not isinstance(questions, list):
            raise ValueError("Questions must be a list")

        for question in questions:
            if "intent" not in question or "criteria" not in question:
                raise ValueError("Each question must have intent and criteria")

        # Validate reward signals
        reward_signals = dataset["reward_signals"]
        if not isinstance(reward_signals, list):
            raise ValueError("Reward signals must be a list")

        for signal in reward_signals:
            required_signal_fields = ["question_id", "system_id", "reward"]
            missing_fields = [f for f in required_signal_fields if f not in signal]
            if missing_fields:
                raise ValueError(
                    f"Reward signal missing required fields: {missing_fields}"
                )

        return dataset


def validate_upload(traces: List[Dict[str, Any]], dataset: Dict[str, Any]):
    """
    Validate the upload format before sending to server.
    Raises ValueError if validation fails.
    """
    try:
        UploadValidator(traces=traces, dataset=dataset)
        return True
    except ValueError as e:
        raise ValueError(f"Upload validation failed: {str(e)}")


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

    # Also close any unclosed events in existing traces
    traces = event_store.get_system_traces()
    current_time = time.time()
    for trace in traces:
        for partition in trace.partition:
            for event in partition.events:
                if event.closed is None:
                    event.closed = current_time
                    event_store.add_event(trace.system_id, event)
                    if verbose:
                        print(f"Closed existing unclosed event: {event.event_type}")

    try:
        # Get traces and convert to dict format
        traces = event_store.get_system_traces()
        traces_dict = [trace.to_dict() for trace in traces]
        dataset_dict = dataset.to_dict()

        # Validate upload format
        if verbose:
            print("Validating upload format...")
        validate_upload(traces_dict, dataset_dict)
        if verbose:
            print("Upload format validation successful")

        # Send to server
        response = send_system_traces(
            dataset=dataset,
            base_url="https://agent-learning.onrender.com",
            api_key=api_key,
        )

        if verbose:
            print("Response status code:", response.status_code)
            if response.status_code == 202:
                print(f"Upload successful - sent {len(traces)} system traces.")
                print(
                    f"Dataset included {len(dataset.questions)} questions and {len(dataset.reward_signals)} reward signals."
                )

        return response
    except ValueError as e:
        if verbose:
            print("Validation error:", str(e))
            print("\nTraces:")
            print(json.dumps(traces_dict, indent=2))
            print("\nDataset:")
            print(json.dumps(dataset_dict, indent=2))
        raise
    except requests.exceptions.HTTPError as e:
        if verbose:
            print("HTTP error occurred:", e)
            print("\nTraces:")
            print(json.dumps(traces_dict, indent=2))
            print("\nDataset:")
            print(json.dumps(dataset_dict, indent=2))
        raise
