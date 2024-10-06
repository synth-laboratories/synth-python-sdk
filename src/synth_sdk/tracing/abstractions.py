from typing import Dict, List
from pydantic import BaseModel

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

class TrainingQuestion(BaseModel):
    intent: str
    criteria: str

    def to_dict(self):
        return {
            "intent": self.intent,
            "criteria": self.criteria
        }

class Dataset(BaseModel):
    questions: List[TrainingQuestion]

    def to_dict(self):
        return {
            "questions": [question.to_dict() for question in self.questions]
        }