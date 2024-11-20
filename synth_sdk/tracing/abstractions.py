from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union
from datetime import datetime
from pydantic import BaseModel
import logging
from synth_sdk.tracing.config import VALID_TYPES

logger = logging.getLogger(__name__)


@dataclass
class MessageInputs:
    messages: List[Dict[str, str]]  # {"role": "", "content": ""}


@dataclass
class ArbitraryInputs:
    inputs: Dict[str, Any]


@dataclass
class MessageOutputs:
    messages: List[Dict[str, str]]


@dataclass
class ArbitraryOutputs:
    outputs: Dict[str, Any]


@dataclass
class ComputeStep:
    event_order: int
    compute_ended: datetime  # time step
    compute_began: datetime  # time step
    compute_input: Dict[str, Any]  # {variable_name: value}
    compute_output: Dict[str, Any]  # {variable_name: value}

    def to_dict(self):
        # Serialize compute_input
        serializable_input = [
            input_item.__dict__ for input_item in self.compute_input
            if isinstance(input_item, (MessageInputs, ArbitraryInputs))
        ]

        # Serialize compute_output
        serializable_output = [
            output_item.__dict__ for output_item in self.compute_output
            if isinstance(output_item, (MessageOutputs, ArbitraryOutputs))
        ]

        # Warn about non-serializable inputs/outputs
        for item in self.compute_input:
            if not isinstance(item, (MessageInputs, ArbitraryInputs)):
                logger.warning(f"Skipping non-serializable input: {item}")
        for item in self.compute_output:
            if not isinstance(item, (MessageOutputs, ArbitraryOutputs)):
                logger.warning(f"Skipping non-serializable output: {item}")

        return {
            "event_order": self.event_order,
            "compute_ended": self.compute_ended.isoformat() if isinstance(self.compute_ended, datetime) else self.compute_ended,
            "compute_began": self.compute_began.isoformat() if isinstance(self.compute_began, datetime) else self.compute_began,
            "compute_input": serializable_input,
            "compute_output": serializable_output,
        }


@dataclass
class AgentComputeStep(ComputeStep):
    model_name: Optional[str] = None
    compute_input: List[Union[MessageInputs, ArbitraryInputs]]
    compute_output: List[Union[MessageOutputs, ArbitraryOutputs]]


@dataclass
class EnvironmentComputeStep(ComputeStep):
    compute_input: List[ArbitraryInputs]
    compute_output: List[ArbitraryOutputs]


@dataclass
class Event:
    system_id: str
    event_type: str
    opened: Any  # timestamp
    closed: Any  # timestamp
    partition_index: int
    agent_compute_steps: List[AgentComputeStep]
    environment_compute_steps: List[EnvironmentComputeStep]

    def to_dict(self):
        return {
            "event_type": self.event_type,
            "opened": self.opened.isoformat() if isinstance(self.opened, datetime) else self.opened,
            "closed": self.closed.isoformat() if isinstance(self.closed, datetime) else self.closed,
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
    """
    A training question for the system.

    Attributes:
        intent (str): The intended purpose or goal of the question
        criteria (str): The evaluation criteria for the question
        question_id (Optional[str]): Unique identifier for the question
    
    Example:
        ```python
        question = TrainingQuestion(
            intent="Test basic addition",
            criteria="Check if the system can add two numbers correctly",
            question_id="math-001"
        )
        ```
    """
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
    """
    A reward signal indicating system performance.

    Attributes:
        question_id (Optional[str]): Reference to the associated training question
        system_id (str): Identifier for the system being evaluated
        reward (Union[float, int, bool]): Performance metric/score
        annotation (Optional[str]): Additional notes about the reward

    Example:
        ```python
        signal = RewardSignal(
            question_id="math-001",
            system_id="calc-v1",
            reward=1.0,
            annotation="Correct addition performed"
        )
        ```
    """
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
    """
    A collection of training questions and reward signals.

    Attributes:
        questions (List[TrainingQuestion]): List of training questions
        reward_signals (List[RewardSignal]): List of associated reward signals

    Example:
        ```python
        dataset = Dataset(
            questions=[TrainingQuestion(...)],
            reward_signals=[RewardSignal(...)]
        )
        ```
    """
    questions: List[TrainingQuestion]
    reward_signals: List[RewardSignal]

    def to_dict(self):
        return {
            "questions": [question.to_dict() for question in self.questions],
            "reward_signals": [signal.to_dict() for signal in self.reward_signals],
        }
