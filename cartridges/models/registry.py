from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from cartridges.models.helpers import ModelHelper

@dataclass
class ModelSpec:
    model_name: str
    model_cls: type[nn.Module]
    helper_cls: type[ModelHelper]
    helper_kwargs: dict[str, Any]

class ModelRegistry:
    def __init__(self):
        self.models = {}
    
    def __getitem__(self, model_name: str) -> ModelSpec:
        if model_name.lower() not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        return self.models[model_name.lower()]
    
    def __contains__(self, model_name: str) -> bool:
        return model_name.lower() in self.models

    def register(
        self, 
        model_name: str, 
        model_cls: type[nn.Module], 
        helper_cls: type[ModelHelper], **kwargs
    ):

        self.models[model_name.lower()] = ModelSpec(
            model_name=model_name, 
            model_cls=model_cls,
            helper_cls=helper_cls, 
            helper_kwargs=kwargs,
        )

    def get_model_helper(self, model_name: str) -> ModelHelper:
        if model_name.lower() not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        spec = self.models[model_name.lower()]
        return spec.helper_cls(model_name=model_name, **spec.helper_kwargs)

MODEL_REGISTRY = ModelRegistry()