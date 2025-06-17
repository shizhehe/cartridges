from __future__ import annotations
from abc import ABC
from dataclasses import field
import math
from typing import List, Literal, Optional
import concurrent.futures

import os

import pandas as pd
import torch

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer
import wandb

import pydrantic
import torch.nn as nn

# Cartridges specific imports
from cartridges.datasets import CartridgeGenerateDatasetElement

from cartridges.structs import Context
from cartridges.generate_baseline import BaselineGenerator, GenerateBaselineResponse
from cartridges.clients.base import Client, ClientConfig, ClientResponse
from cartridges.utils import get_logger


logger = get_logger(__name__)


    

class ReglabHousingGoldPassageBaseline(BaselineGenerator):

    class Config(BaselineGenerator.Config):
        client: ClientConfig

        # used to count number of tokens in the prompt
        tokenizer: str = "meta-llama/Llama-3.2-3B-Instruct"

        # The system prompt template which should contain {title} and {content}
        # variables. 
        system_prompt_template: str = "{title}\n\n{content}"
        max_completion_tokens: int = 384
        temperature: float = 0.0
    def __init__(
        self, config: Config, context: Context
    ):
        self.config = config
        self.client = config.client.instantiate()
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        self.context = context


    def generate(
        self, 
        elements: List[CartridgeGenerateDatasetElement]
    ) -> List[GenerateBaselineResponse]:
        chats = []
        for element in elements:
            # TODO (SE): Add support for full gold statute 
            gold_statutes = element.metadata["statutes"]

            context = ""
            for statute in gold_statutes:
                context += f"{statute['citation']}: {statute['excerpt']}\n\n"

            system_prompt = self.config.system_prompt_template.format(
                title=self.context.title,
                content=context,
            )
            chats.append(
                [
                    {
                        "role": "system", 
                        "content": system_prompt,
                    },
                    {"role": "user", "content": element.prompt}
                ]
            )
        
        response: ClientResponse = self.client.chat(
            chats=chats,
            max_completion_tokens=self.config.max_completion_tokens,
            temperature=self.config.temperature,
        )
            
        results = []
        for sample, messages, element in zip(response.samples, chats, elements):
            num_prompt_tokens = len(self.tokenizer.apply_chat_template(
                messages,
            ))

            results.append(
                GenerateBaselineResponse(
                    prompt_messages=messages,
                    num_prompt_tokens=num_prompt_tokens,
                    text=sample.text,
                )
            )
        return results

