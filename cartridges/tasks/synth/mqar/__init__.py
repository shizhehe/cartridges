from cartridges.context import BaseContextConfig
from cartridges.structs import Context
from pydantic import BaseModel
from typing import Dict
from cartridges.tasks.synth.mqar.mqar_data import MQARConfig, MQARData
from cartridges.tasks.synth.mqar.generator import MQARGenerator


class MQARQuestion(BaseModel):
    question_id: str
    question: str
    answer: str
    explanation: str


class MQARDataset(BaseModel):
    dataset_id: str
    questions: Dict[str, MQARQuestion]

__all__ = ["MQARConfig", "MQARData", "MQARGenerator"]


class MQARContextConfig(BaseContextConfig):
    def instantiate(self) -> Context:
        return Context(
            title="MQAR Synthetic Data",
            sections=[],
            text=None
        )
