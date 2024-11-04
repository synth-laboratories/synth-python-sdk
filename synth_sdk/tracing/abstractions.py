from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union
from pydantic import BaseModel
import logging
from synth_sdk.tracing.config import VALID_TYPES

logger = logging.getLogger(__name__)


@dataclass
class ComputeStep:
    event_order: int
    compute_ended: Any  # time step
    compute_began: Any  # time step
    compute_input: Dict[str, Any]  # {variable_name: value}
    compute_output: Dict[str, Any]  # {variable_name: value}

    def to_dict(self):
        # Define serializable types
        #serializable_types = (str, int, float, bool, list, dict, type(None))

        # Filter compute_input
        serializable_input = {}
        for name, value in self.compute_input.items():
            if isinstance(value, VALID_TYPES):
                serializable_input[name] = value
            else:
                logger.warning(f"Skipping non-serializable input: {name}={value}")

        # Filter compute_output
        serializable_output = {}
        for name, value in self.compute_output.items():
            if isinstance(value, VALID_TYPES):
                serializable_output[name] = value
            else:
                logger.warning(f"Skipping non-serializable output: {name}={value}")

        return {
            "event_order": self.event_order,
            "compute_ended": self.compute_ended,
            "compute_began": self.compute_began,
            "compute_input": serializable_input,
            "compute_output": serializable_output,
        }


class AgentComputeStep(ComputeStep):
    pass


class EnvironmentComputeStep(ComputeStep):
    pass


@dataclass
class Event:
    event_type: str
    opened: Any  # time stamp
    closed: Any  # time stamp
    partition_index: int  # New field
    agent_compute_steps: List[AgentComputeStep]
    environment_compute_steps: List[EnvironmentComputeStep]

    def to_dict(self):
        return {
            "event_type": self.event_type,
            "opened": self.opened,
            "closed": self.closed,
            "partition_index": self.partition_index,
            "agent_compute_steps": [
                step.to_dict() for step in self.agent_compute_steps
            ],
            "environment_compute_steps": [
                step.to_dict() for step in self.environment_compute_steps
            ],
        }


@dataclass
class EventPartitionElement:
    partition_index: int
    events: List[Event]

    def to_dict(self):
        return {
            "partition_index": self.partition_index,
            "events": [event.to_dict() for event in self.events],
        }


@dataclass
class SystemTrace:
    system_id: str
    partition: List[EventPartitionElement]
    current_partition_index: int = 0  # Track current partition

    def to_dict(self):
        return {
            "system_id": self.system_id,
            "partition": [element.to_dict() for element in self.partition],
            "current_partition_index": self.current_partition_index,
        }


class TrainingQuestion(BaseModel):
    intent: str
    criteria: str
    question_id: Optional[str] = None

    def to_dict(self):
        return {
            "intent": self.intent,
            "criteria": self.criteria,
            "question_id": self.question_id,
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
