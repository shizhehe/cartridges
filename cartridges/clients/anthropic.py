from collections import defaultdict
import os
from typing import Any, Dict, List, Literal, Optional, Type, Union
import openai
from openai.types.chat.chat_completion import ChatCompletion
import asyncio

from pydrantic import ObjectConfig
import tiktoken
from cartridges.clients.base import (
    Client,
    Sample,
    SelectedToken,
    TopToken,
    ClientConfig,
    ClientResponse,
)
from cartridges.clients.usage import Usage, num_tokens_from_messages_openai
from cartridges.utils import get_logger

# import anthropic


class AnthropicClient(Client):
    """Client for interacting with Anthropic's Claude API."""

    class Config(ClientConfig):
        """Configuration options for the AnthropicClient."""

        _pass_as_config: bool = True

        model_name: str = "claude-3-7-sonnet-20250219"
        api_key: Optional[str] = None

    def __init__(self, config: Config):
        """
        Initialize the Anthropic client with the provided config.
        If config.api_key is set, it will override ANTHROPIC_API_KEY in the environment.
        """
        self.config = config

        # Import anthropic here to avoid requiring it as a dependency for the entire project
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=(
                config.api_key if config.api_key else os.getenv("ANTHROPIC_API_KEY")
            ),
        )
        self.logger = get_logger("AnthropicClient")

    def complete(
        self,
        prompts: List[Union[str, List[int]]],
        temperature: float = 0.6,
        stop: List[str] = [],
        max_completion_tokens: int = 1,
        **kwargs,
    ) -> ClientResponse:
        raise NotImplementedError("Anthropic does not support a completion API.")

    def chat(
        self,
        chats: List[List[Dict[str, Any]]],
        temperature: float = 0.6,
        stop: List[str] = [],
        max_completion_tokens: Optional[int] = None,
        frequency_penalty: float = 0.0,
        top_logprobs: int = 1,
        **kwargs,
    ) -> ClientResponse:
        """
        Calls the Anthropic Messages API to generate responses to the provided chats.
        """
        assert len(chats) > 0

        assert not stop
        assert frequency_penalty == 0.0
        assert top_logprobs == 1.0

        responses = []

        for i, chat in enumerate(chats):
            system_content: str | list = ""
            anthropic_messages = []

            # Process messages and convert to Anthropic format
            for message in chat:
                role = message["role"]
                content = message["content"]

                if role == "system":
                    # TODO(add prompt caching)

                    assert system_content == ""

                    extra_kwargs = (
                        {"cache_control": message["cache_control"]}
                        if "cache_control" in message
                        else {}
                    )
                    system_content = [
                        {
                            "type": "text",
                            "text": content,
                            **extra_kwargs,
                        }
                    ]
                elif role == "user":
                    anthropic_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    anthropic_messages.append({"role": "assistant", "content": content})
                # Ignore other roles

            # Call Anthropic API
            response = self.client.messages.create(
                model=self.config.model_name,
                messages=anthropic_messages,
                system=system_content,
                max_tokens=(
                    max_completion_tokens if max_completion_tokens is not None else 8192
                ),
                temperature=temperature,
            )

            # Extract text from the response
            # Anthropic returns content as a list of ContentBlocks
            response_text = ""
            for content_block in response.content:
                if content_block.type == "text":
                    response_text += content_block.text

            # Map Anthropic stop reason to our format
            if response.stop_reason == "stop_sequence":
                stop_reason = "stop"
            elif response.stop_reason == "max_tokens":
                stop_reason = "max_tokens"
            else:
                stop_reason = "stop"  # Default

            # Create a single token for the entire response
            tokens = [
                SelectedToken(
                    text="not implemented yet, bug RE to do it", logprob=0.0, id=None
                )
            ]

            responses.append(
                Sample(text=response_text, tokens=tokens, stop_reason=stop_reason)
            )

            print("Usage", response.usage)

        # Create a minimal usage object (not tracking usage as requested)
        usage = Usage(prompt_tokens=0, completion_tokens=0)


        return ClientResponse(samples=responses, usage=usage)
