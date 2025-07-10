from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from pydrantic import ObjectConfig
from pydantic import BaseModel
from cartridges.clients.usage import Usage
from dataclasses import dataclass, asdict


@dataclass(slots=True)
class ClientResponse:
    samples: List[ClientSample]
    usage: Usage
    
    timings: Optional[List[Dict[str, Any]]] = None
    def to_dict(self):
        return asdict(self)


@dataclass(slots=True)
class ClientSample:
    text: str
    token_ids: Optional[List[int]] = None

    top_logprobs: Optional[TopLogprobs] = None

@dataclass(slots=True)
class TopLogprobs:
    logprobs: np.ndarray  # [num_tokens, num_top_logprobs]
    token_ids: np.ndarray  # [num_tokens, num_top_logprobs]


class ClientConfig(ObjectConfig):
    _pass_as_config: bool = True

    model_name: str

    show_progress_bar: bool = False

    def instantiate(self, *args, **kwargs) -> "Client":
        return super().instantiate(*args, **kwargs)


class Client(ABC):
    def __init__(self, config: ClientConfig):
        self.config = config

    @abstractmethod
    async def chat(
        self, 
        chats: List[List[Dict[str, Any]]], 
        temperature: float = 0.6, 
        stop: List[str] = [], 
        max_completion_tokens: Optional[int] = None,
        frequency_penalty: float = 0.0,
        top_logprobs: int = 1,
        logprobs_start_message: Optional[int] = None,
        modal_upstream_id: Optional[str] = None,
    ) -> ClientResponse:
        raise NotImplementedError
