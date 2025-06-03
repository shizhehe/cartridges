from dataclasses import dataclass, fields
import gc
import pickle
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Union
import uuid
import requests

import numpy as np
from pydantic import BaseModel
from transformers import AutoTokenizer

from cartridges.clients.base import (
    Client,
    Sample,
    ClientConfig,
    ClientResponse,
    CartridgesConvoWithLogprobs,
)
from cartridges.clients.usage import Usage
from cartridges.utils import get_logger


logger = get_logger(__name__)


class CapulesMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class CartridgesConvo(BaseModel):
    routing_tag: str | None
    messages: list[CapulesMessage]


class CartridgesBatchRequest(BaseModel):
    elements: list[CartridgesConvo]
    temperature: float
    max_tokens: int



class TokasaurusBatchClient(Client):

    class Config(ClientConfig):
        """Configuration options for the TogetherClient."""

        model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
        url: str
        ports: Optional[list[int]] = None
        
        # SE(04/17): sometimes Modal servers just hang weirdly, so I set a timeout
        max_retries: int = 10
        timeout: int = 360  # in seconds
        on_failure: Literal["raise", "continue"] = "raise"


    def __init__(self, config: Config):
        """
        Initialize the Together client with the provided config.
        If config.api_key is set, it will override TOGETHER_API_KEY in the environment.
        """
        if config.ports is None:
            # No ports provided for use with Modal
            config.ports = [None]
            
        self.config = config
        self.logger = get_logger("TogetherClient")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tag_to_port = {}
        self.active_per_port = {port: 0 for port in self.config.ports}

    def complete(
        self,
        prompts: List[Union[str, List[int]]],
        temperature: float = 0.6,
        stop: Optional[List[str]] = None,
        max_completion_tokens: int = 1,
        **kwargs,
    ) -> Sample:
        """
        Tokasaurus does not directly support a `complete` API.
        """
        raise NotImplementedError(
            "The `complete` method is not yet supported by Tokasuarus Client."
        )

    def chat(
        self,
        chats: List[Dict[str, Any]],
        max_completion_tokens: int,
        temperature: float = 0.6,
        stop: Optional[List[str]] = None,
        top_logprobs: Optional[int] = None,
        routing_tag: Optional[str] = None,
        **kwargs,
    ) -> ClientResponse:
        raise NotImplementedError("Please use chat_custom")

    def chat_with_logprobs(
        self,
        chats: List[Dict[str, Any]],
        max_completion_tokens: int,
        temperature: float = 0.6,
        stop: Optional[List[str]] = None,
        top_logprobs: Optional[int] = None,
        routing_tag: Optional[str] = None,
        **kwargs,
    ) -> list[CartridgesConvoWithLogprobs]:
        assert (
            stop is None
        ), "stop is not supported by Tokasaurus batch Cartridges endpoint"
        t0 = time.time()
        port = None

        request = CartridgesBatchRequest(
            elements=[
                CartridgesConvo(
                    routing_tag=routing_tag,
                    messages=chat,  # type: ignore
                )
                for chat in chats
            ],
            max_tokens=max_completion_tokens,
            temperature=temperature,
        )
        request_json = request.model_dump()
        request_json["password"] = str(uuid.uuid4())


        response = None
        for retry_idx in range(self.config.max_retries):
            try:
                if port is None:
                    url = f"{self.config.url}/capules/batch"
                else:
                    url = f"{self.config.url}:{port}/capules/batch"
                t1 = time.time()
                response = requests.post(url, json=request_json, timeout=self.config.timeout * (2 ** (retry_idx / 10)))
                t2 = time.time()

                if response.status_code != 200:
                    raise Exception(f"Error: {response.status_code} {response.text}")
                
            except Exception as e:
                logger.warning(f"The following error occurred when sending a request to the server. Retrying ({retry_idx + 1}/{self.config.max_retries})... \n{e}")
                response = None
                time.sleep(1)
                continue
            else:
                break
    
        if response is None:
            logger.error(f"No more retries left. Failed to get a response from the server.")
            if self.config.on_failure == "raise":
                raise Exception(f"Failed to get a response from the server.")
            else:
                return [CartridgesConvoWithLogprobs(
                    num_output_tokens=0,
                    token_ids=None,
                    assistant_text="",
                    top_logprob_logprobs=None,
                    top_logprob_ids=None,
                )] * len(chats)

        elements = []

        batch_elements = pickle.loads(response.content)
        for elem in batch_elements:
            (num_tokens,) = elem["token_ids"].shape
            assert len(elem["top_logprob_logprobs"].shape) == 2
            assert elem["top_logprob_logprobs"].shape == elem["top_logprob_ids"].shape
            # assert elem["top_logprob_logprobs"].shape[0] == num_tokens - 1

            elements.append(
                CartridgesConvoWithLogprobs(
                    num_output_tokens=elem["num_output_tokens"],
                    token_ids=elem["token_ids"],
                    assistant_text=elem["assistant_text"],
                    top_logprob_logprobs=elem["top_logprob_logprobs"],
                    top_logprob_ids=elem["top_logprob_ids"],
                )
            )
        # logger.info(f"Chat logprobs full time: {time.time() - t0} seconds -- {t2 - t1} seconds for requests")

        assert len(elements) == len(chats)
        return elements
