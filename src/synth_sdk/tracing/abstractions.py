from typing import Dict, List
from pydantic import BaseModel
import requests
import logging


class TrajectoryTrace(BaseModel):
    agent_id: str
    spans: List[Dict] = []

    def add_span(self, span: Dict):
        assert isinstance(span, dict)
        self.spans.append(span)

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "spans": self.spans
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)

def send_trajectory_traces(traces: List[TrajectoryTrace], base_url: str, api_key: str) -> requests.Response:
    # Get the token using the API key
    token_url = f"{base_url}/token"
    token_response = requests.post(token_url, headers={"Authorization": f"Bearer {api_key}"})
    token_response.raise_for_status()
    access_token = token_response.json()["access_token"]

    # Send the traces with the token
    api_url = f"{base_url}/traces/"
    traces_dict = [trace.to_dict() for trace in traces]
    payload = {"traces": traces_dict}
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        logging.info(f"Response status code: {response.status_code}")
        return response
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        raise
    except Exception as err:
        logging.error(f"An error occurred: {err}")
        raise