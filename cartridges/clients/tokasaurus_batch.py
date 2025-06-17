import asyncio
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
    ClientSample,
    ClientConfig,
    ClientResponse,
    TopLogprobs,
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
    ) -> ClientSample:
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
        logprobs_start_message: Optional[int] = None,
        **kwargs,
    ) -> ClientResponse:
        assert (
            stop is None
        ), "stop is not supported by Tokasaurus batch Cartridges endpoint"
        t0 = time.time()
        port = None

        request = CartridgesBatchRequest(
            elements=[
                CartridgesConvo(
                    routing_tag=None,
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
                return ClientResponse(
                    samples=[
                        ClientSample(assistant_text="", num_output_tokens=0) for _ in chats
                    ], 
                    usage=Usage(num_input_tokens=0, num_output_tokens=0)
                )    
                    
                

        samples = []
        usage = Usage(prompt_tokens=0, completion_tokens=0)
        batch_elements = pickle.loads(response.content)
        for elem in batch_elements:
            (num_tokens,) = elem["token_ids"].shape
            assert len(elem["top_logprob_logprobs"].shape) == 2
            assert elem["top_logprob_logprobs"].shape == elem["top_logprob_ids"].shape
            # assert elem["top_logprob_logprobs"].shape[0] == num_tokens - 1

            usage += Usage(
                prompt_tokens=len(elem["token_ids"]) - elem["num_output_tokens"],
                completion_tokens=elem["num_output_tokens"],
            )
            logprobs = TopLogprobs(                   
                num_input_tokens=len(elem["token_ids"]) - elem["num_output_tokens"],
                token_ids=elem["token_ids"],
                top_logprobs=elem["top_logprob_logprobs"],
                top_ids=elem["top_logprob_ids"],
            )
            if logprobs_start_message is not None and logprobs_start_message > 0:
                logprobs = self._trim_logprobs(logprobs, logprobs_start_message)

            samples.append(
                ClientSample(
                    num_output_tokens=elem["num_output_tokens"],
                    output_text=elem["assistant_text"],
                    top_logprobs=logprobs,
                )
            )
        # logger.info(f"Chat logprobs full time: {time.time() - t0} seconds -- {t2 - t1} seconds for requests")

        assert len(samples) == len(chats)
        return ClientResponse(samples=samples, usage=usage)


    async def chat_async(
        self,
        chats: List[Dict[str, Any]],
        max_completion_tokens: int,
        temperature: float = 0.6,
        stop: Optional[List[str]] = None,
        top_logprobs: Optional[int] = None,
        logprobs_start_message: Optional[int] = None,
        **kwargs,
    ) -> ClientResponse:
        assert (
            stop is None
        ), "stop is not supported by Tokasaurus batch Cartridges endpoint"
        t0 = time.time()
        port = None

        request = CartridgesBatchRequest(
            elements=[
                CartridgesConvo(
                    routing_tag=None,
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
                import aiohttp
                from types import SimpleNamespace

                timeout = aiohttp.ClientTimeout(
                    total=self.config.timeout * (2 ** (retry_idx / 10))
                )
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=request_json) as resp:
                        content = await resp.read()
                        response = pickle.loads(content)
                t2 = time.time()
                
            except Exception as e:
                logger.warning(f"The following error occurred when sending a request to the server. Retrying ({retry_idx + 1}/{self.config.max_retries})... \n{e}")
                response = None
                await asyncio.sleep(1)
                continue
            else:
                break
    
        if response is None:
            logger.error(f"No more retries left. Failed to get a response from the server.")
            if self.config.on_failure == "raise":
                raise Exception(f"Failed to get a response from the server.")
            else:
                return ClientResponse(
                    samples=[
                        ClientSample(assistant_text="", num_output_tokens=0) for _ in chats
                    ], 
                    usage=Usage(num_input_tokens=0, num_output_tokens=0)
                )    
                    
                

        samples = []
        usage = Usage(prompt_tokens=0, completion_tokens=0)
        batch_elements = response
        for elem in batch_elements:
            (num_tokens,) = elem["token_ids"].shape
            assert len(elem["top_logprob_logprobs"].shape) == 2
            assert elem["top_logprob_logprobs"].shape == elem["top_logprob_ids"].shape
            # assert elem["top_logprob_logprobs"].shape[0] == num_tokens - 1

            usage += Usage(
                prompt_tokens=len(elem["token_ids"]) - elem["num_output_tokens"],
                completion_tokens=elem["num_output_tokens"],
            )
            logprobs = TopLogprobs(                   
                num_input_tokens=len(elem["token_ids"]) - elem["num_output_tokens"],
                token_ids=elem["token_ids"],
                top_logprobs=elem["top_logprob_logprobs"],
                top_ids=elem["top_logprob_ids"],
            )
            if logprobs_start_message is not None and logprobs_start_message > 0:
                logprobs = self._trim_logprobs(logprobs, logprobs_start_message)

            samples.append(
                ClientSample(
                    num_output_tokens=elem["num_output_tokens"],
                    output_text=elem["assistant_text"],
                    top_logprobs=logprobs,
                )
            )
        # logger.info(f"Chat logprobs full time: {time.time() - t0} seconds -- {t2 - t1} seconds for requests")

        assert len(samples) == len(chats)
        return ClientResponse(samples=samples, usage=usage)


    def _trim_logprobs(
        self, 
        logprobs: TopLogprobs, 
        logprobs_start_message: int
    ) -> TopLogprobs:
        assert logprobs_start_message == 1

        header_locations = np.where(logprobs.token_ids == 128006)[
            0
        ].tolist()

        assert len(header_locations) > 1, "There should be at least two messages in the inputs to use trim logprobs"

        prefix_end_idx = header_locations[1]
        token_ids = logprobs.token_ids[prefix_end_idx:]

        assert (
            logprobs.top_logprobs.shape
            == logprobs.top_ids.shape
        )
        assert (
            logprobs.top_logprobs.shape[0] == len(token_ids) - 1
        ), "You probably need to pull down on tokasaurus server"

        return TopLogprobs(
            num_input_tokens=logprobs.num_input_tokens,
            token_ids=token_ids,
            top_logprobs=logprobs.top_logprobs,
            top_ids=logprobs.top_ids,
        )