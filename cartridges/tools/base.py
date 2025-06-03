from __future__ import annotations


from typing import Any, List
import abc


from pydantic import BaseModel
from pydrantic import ObjectConfig

from capsules.context import StructuredContext
from transformers import AutoTokenizer


class Tool(abc.ABC):

    class Config(ObjectConfig):
        _pass_as_config = True

    class Input(BaseModel):
        pass


    def __call__(
        input: Input,
    ):
        raise NotImplementedError()
    
    def batch_call(
        self,
        inputs: List[Input],
    ) -> List[Any]:
        outputs = []
        for input in inputs:
            try:
                out = self(input)
                outputs.append({
                    "success": True,
                    "input": input,
                    "tool_response": out,
                    "error": None,
                })
            except Exception as e:
                outputs.append({
                    "success": False,
                    "input": input,
                    "tool_response": None,
                    "error": str(e),
                })
                
        return outputs
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    


