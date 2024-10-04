from ..tracing.config import in_memory_exporter
from ..tracing.abstractions import TrajectoryTrace, send_trajectory_traces
from typing import Dict


async def upload(verbose: bool = False):
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
    api_key = os.getenv("SYNTH_API_KEY")
    response = send_trajectory_traces(trajectory_traces_list, "https://agent-learning.onrender.com", api_key)
    
    if verbose:
        print("Response status code:", response.status_code)
        if response.status_code == 202:
            print("Upload successful")
