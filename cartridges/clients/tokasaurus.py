from dataclasses import dataclass
from collections import defaultdict
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from readline import get_history_item
from typing import TypeVar, Generic, Callable
import uuid
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from capsules.clients.base import Client, Sample, SelectedToken, TopToken, ClientConfig, ClientResponse
from capsules.clients.usage import Usage
from capsules.utils import get_logger



from tqdm import tqdm

@dataclass
class BatchConfig:
    batch_size: int = 100
    num_threads: int = 10
    initial_wait: int = 5
    check_interval: int = 1 
    max_tokens: int = 1024
    temperature: float = 0.0

T = TypeVar('T')

@dataclass
class Response(Generic[T]):
    item: T
    response: str

def load_responses[T](get_item_file: Callable[[T], Path], items: list[T]):
    results = [
        Response(
            item,
            get_item_file(item).read_text(encoding="utf-8")
        )
        for item in items
    ]
    return results

Message = dict

@dataclass
class BatchProcessor(Generic[T]):
    """Generic batch processor for OpenAI API calls"""
    items: list[T]
    get_prompt: Callable[[T], str]
    config: BatchConfig
    get_item_file: Callable[[T], Path]

BATCH_INITIAL_WAIT = 0.5
BATCH_CHECK_INTERVAL = 0.5
REQUEST_TIMEOUT = 1024
MAX_RETRIES = 10
RETRY_WAIT = 0.5

class TokasaurusClient(Client):

    class Config(ClientConfig):
        """Configuration options for the TogetherClient."""

        url: str="https://scalingintelligence--tokasaurus-llama-3b-serve.modal.run/v1"
        use_modal_endpoint: bool = False

    def __init__(self, config: Config):
        """
        Initialize the Together client with the provided config.
        If config.api_key is set, it will override TOGETHER_API_KEY in the environment.
        """
        self.config = config
        self.logger = get_logger("TokasaurusClient")
        
        # Initialize the Together API client (replace with actual Together client initialization if available)
        self.client = OpenAI(
            base_url=self.config.url,
            api_key='unused',
        )

    def complete(
        self,
        prompts: List[Union[str, List[int]]],
        temperature: float = 0.6,
        stop: Optional[List[str]] = None,
        max_completion_tokens: int = 1,
        **kwargs
    ) -> Sample:
        """
        Together does not directly support a `complete` API. Raise a `NotImplementedError` as a placeholder.
        """
        raise NotImplementedError("The `complete` method is not yet supported by Tokasuarus Client.")

    def chat(
        self,
        chats: List[Dict[str, Any]],
        max_completion_tokens: int,
        temperature: float = 0.6,
        stop: Optional[List[str]] = None,
        top_logprobs: Optional[int] = None,
        **kwargs
    ) -> ClientResponse:
        """
        Handle chat completions using the Together API.
        """
        assert len(chats) > 0, "Messages cannot be empty."

        if self.config.use_modal_endpoint:
            fn = self._run_batch_modal
        else:
            fn = self._run_batch

        return fn(
            chats=chats,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            stop=stop,
            top_logprobs=top_logprobs,
        )
    
    def _run_batch_modal(
        self,
        chats: list[list[Message]],
        max_completion_tokens: int,
        temperature: float = 0.6,
        stop: Optional[List[str]] = None,
        top_logprobs: Optional[int] = None,
    ) -> list[ClientResponse]:
        import requests

        url = self.config.url
        if url.endswith("/v1"):
            url = url[:-3] 
        url += "/batch"

        obj = {
            "chats": chats,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "stop": stop,
            "top_logprobs": top_logprobs,
        }
        self.logger.info(f"Running batch with {len(chats)} chats")

        for i in range(MAX_RETRIES):
            try:
                response = requests.post(
                    url, 
                    json=obj, 
                    timeout=REQUEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    break
                else:
                    error_message = f"Batch failed with status code {response.status_code}. Response: {response.text}"
                    self.logger.error(error_message)
                    raise Exception(error_message)
            except Exception as e:
                self.logger.error(f"Batch failed with error {e}")
                if i == MAX_RETRIES - 1:
                    raise e
                time.sleep(RETRY_WAIT)

        return self._parse_batch_response(response.json())

    def _run_batch(
        self,
        chats: list[list[Message]],
        max_completion_tokens: int,
        temperature: float = 0.6,
        stop: Optional[List[str]] = None,
        top_logprobs: Optional[int] = None,
    ) -> list[ClientResponse]:
        endpoint = "/v1/chat/completions"
        
        id_to_item = {}
        file_lines = []

        for chat in chats:
            item_id = str(uuid.uuid4())
            body = {
                    "model": "todo",  # Set by tokasarus
                    "messages": chat,
                    "max_tokens": max_completion_tokens,
                    "temperature": temperature,
                }
            
            # if top_logprobs is not None:
            #     body["logprobs"] = top_logprobs

            assert stop is None

            file_lines.append(json.dumps({
                "custom_id": item_id,
                "method": "POST",
                "url": endpoint,
                "body": body
            }))
            id_to_item[item_id] = chat

        batch_input_file = self.client.files.create(
            file="\n".join(file_lines).encode(),
            purpose="batch"
        )
        
        batch_info = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint=endpoint,
            completion_window="24h",
        )
        
        time.sleep(BATCH_INITIAL_WAIT)
        
        while True:
            batch_info = self.client.batches.retrieve(batch_info.id)
            print("Batch status:", batch_info.id, batch_info.status)
            if batch_info.status == 'completed':
                break
            elif batch_info.status == 'in_progress':
                time.sleep(BATCH_CHECK_INTERVAL)
                continue
            else:
                raise ValueError(f"Invalid batch status: {batch_info.status}")

        file_content = self.client.files.content(batch_info.output_file_id)
        samples = []
        for line in file_content.text.splitlines():
            data = json.loads(line)
            samples.append(data)
        return self._parse_batch_response(response=samples)

    def _parse_batch_response(
        self,
        response: list[dict],
    ) -> list[ClientResponse]:
        samples: list[Sample] = []
        usage = Usage()
        
        for data in response:
            choices = data['response']['body']['choices']
            assert len(choices) == 1
            
            choice = choices[0]
            text = choice['message']['content']
            
            # Extract token information for SelectedToken objects
            tokens = []
            for token_info in choice['logprobs']['content']:
                token = SelectedToken(
                    text=token_info['token'],
                    id=token_info["bytes"][0] if len(token_info["bytes"]) > 0 else None, # FIXME: Tokasaurus with rse's hack returns the id here
                    logprob=token_info['logprob'],
                    top_logprobs=[
                        TopToken(
                            id=int(top['token']),
                            text="",  # FIXME: Tokasaurus with rse's hack doesn't return text, eventually should fix
                            logprob=top['logprob']
                        )
                        for top in token_info.get('top_logprobs', [])
                    ]
                )
                tokens.append(token)
            
            # Get stop reason from finish_reason
            stop_reason_map = {
                "stop": "stop",
                "length": "length",
                "max_tokens": "max_tokens"
            }
            stop_reason = stop_reason_map.get(choice['finish_reason'], "error")
            
            # Create Sample object
            sample = Sample(
                text=text,  # Does not include eos_token
                tokens=tokens,  # Includes eos_token
                stop_reason=stop_reason
            )
            
            samples.append(sample)

            usage_data = data['response']['body']['usage']
            usage += Usage(
                completion_tokens=usage_data.get('completion_tokens', 0),
                prompt_tokens=usage_data.get('prompt_tokens', 0),
                cached_prompt_tokens=usage_data.get('cached_prompt_tokens', 0),
                seen_prompt_tokens=usage_data.get('seen_prompt_tokens', 0)
            )

        return ClientResponse(
            samples=samples,
            usage=usage,
        )
