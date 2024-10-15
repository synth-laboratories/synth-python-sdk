from ..tracing.config import in_memory_exporter
from ..tracing.abstractions import TrajectoryTrace, Dataset
from typing import Dict, List
import requests
import logging

def send_trajectories(traces: List[TrajectoryTrace], dataset: Dataset, base_url: str, api_key: str) -> requests.Response:
    # Get the token using the API key
    # token_url = f"{base_url}/token"
    # token_response = requests.post(token_url, headers={"Authorization": f"Bearer {api_key}"})
    # token_response.raise_for_status()
    # access_token = token_response.json()["access_token"]
    token_url = f"{base_url}/token"
    token_response = requests.get(token_url, headers={"customer_specific_api_key": api_key})
    token_response.raise_for_status()
    access_token = token_response.json()["access_token"]


    # Send the traces with the token
    api_url = f"{base_url}/upload/"
    traces_dict = [trace.to_dict() for trace in traces]
    dataset_dict = dataset.to_dict()
    payload = {"traces": traces_dict, "dataset": dataset_dict}
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}'
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
    import os
    spans = in_memory_exporter.get_spans()
    trajectory_traces: Dict[str, TrajectoryTrace] = {}
    
    for span in spans:
        agent_id = span.get("attributes", {}).get("agent.id")
        if agent_id:
            if agent_id not in trajectory_traces:
                trajectory_traces[agent_id] = TrajectoryTrace(agent_id=agent_id, spans=[])
            trajectory_traces[agent_id].add_span(span)
    trajectory_traces_list = list(trajectory_traces.values())
    response = send_trajectories(traces=trajectory_traces_list, dataset=dataset, base_url="https://agent-learning.onrender.com", api_key=os.getenv("SYNTH_API_KEY"))
    
    if verbose:
        print("Response status code:", response.status_code)
        if response.status_code == 202:
            print("Upload successful")
    return response
