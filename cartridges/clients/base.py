from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from pydrantic import ObjectConfig
from pydantic import BaseModel
from cartridges.clients.usage import Usage
from dataclasses import dataclass, asdict


@dataclass(slots=True)
class TopToken:
    # TODO (SE): text is temporarily optional because the tokasaurus Cartridges api
    # does not return it.
    logprob: float
    text: Optional[str] = None

    # id is optional because some clients (e.g. OpenAI) do not return it.
    id: Optional[int] = None
@dataclass(slots=True)
class SelectedToken: 
    # TODO (SE): text is temporarily optional because the tokasaurus Cartridges api
    # does not return it.
    text: Optional[str] = None

    # TODO (SE): making this optional is a temporary solution to support
    # tokasaurus Cartridges api which does not return logprob for the selected token
    logprob: Optional[float] = None

    # id is optional because some clients (e.g. OpenAI) do not return it.
    id: Optional[int] = None

    top_logprobs: Optional[List[TopToken]] = None

@dataclass(slots=True)
class InputToken: 
    # TODO (SE): text is temporarily optional because the tokasaurus Cartridges api
    # does not return it.
    text: Optional[str] = None

    # id is optional because some clients (e.g. OpenAI) do not return it.
    id: Optional[int] = None

    top_logprobs: Optional[List[TopToken]] = None


@dataclass(slots=True)
class Sample:
    text: str  # Does NOT include eos_token
    tokens: List[SelectedToken] # Includes eos_token
    stop_reason: Literal["max_tokens", "stop", "length", "error"]
    input_tokens: Optional[List[InputToken]] = None # Does NOT include eos_token


@dataclass(slots=True)
class ClientResponse:
    samples: List[Sample]
    usage: Usage
    
    timings: Optional[List[Dict[str, Any]]] = None
    def to_dict(self):
        return asdict(self)
    
@dataclass(slots=True)
class CartridgesConvoWithLogprobs:
    num_output_tokens: int

    token_ids: np.ndarray
    top_logprob_logprobs: np.ndarray
    top_logprob_ids: np.ndarray
    assistant_text: str


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
    def complete(
        self, 
        prompts: List[Union[str, List[int]]], 
        **kwargs
    ) -> ClientResponse:
        raise NotImplementedError

    @abstractmethod
    def chat(
        self, 
        chats: List[List[Dict[str, Any]]], 
        temperature: float = 0.6, 
        stop: List[str] = [], 
        max_completion_tokens: Optional[int] = None,
        frequency_penalty: float = 0.0,
        top_logprobs: int = 1,
    ) -> ClientResponse:
        raise NotImplementedError

    def chat_with_logprobs(
        self,
        chats: List[list[Dict[str, Any]]],
        max_completion_tokens: int,
        temperature: float = 0.6,
        stop: Optional[List[str]] = None,
        top_logprobs: Optional[int] = None,
        routing_tag: Optional[str] = None,
        **kwargs,
    ) -> list[CartridgesConvoWithLogprobs]:
        raise NotImplementedError(
            "The `chat_with_logprobs` method is not yet supported for this client. See the `TokasaurusBatchClient` for an example."
        )
