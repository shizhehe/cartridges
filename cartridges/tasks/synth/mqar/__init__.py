from capsules.generate.run import BaseContextConfig
from capsules.generate.structs import Context
from pydantic import BaseModel
from typing import Dict
from capsules.tasks.synth.mqar.mqar_data import MQARConfig, MQARData
from capsules.tasks.synth.mqar.generator import MQARGenerator


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
