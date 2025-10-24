from __future__ import annotations
from copy import deepcopy
from functools import partial
import os
from typing import Callable, Dict, List, Literal, Optional, Any, TypedDict
from abc import abstractmethod
from collections import deque
import json
import pickle
from pathlib import Path
import random
from dataclasses import dataclass

from torch.utils.data import Dataset
import torch
from transformers import PreTrainedTokenizerFast
from pydrantic import ObjectConfig, BaseConfig
import numpy as np

from cartridges.clients.base import FlatTopLogprobs
from cartridges.structs import Conversation, read_conversations, MessageDict
from cartridges.utils import get_logger
from cartridges.utils.hf import read_conversations_from_hf
from cartridges.utils.wandb import read_conversations_from_wandb

# SE(04/02): required to silence tokenizer warnings when using dataloders with
# multiple worker processes
os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = get_logger(__name__)

@dataclass
class TokenCounts:
    num_system_and_user_tokens: int = 0
    num_assistant_tokens: int = 0

    @property
    def num_tokens(self):
        return self.num_system_and_user_tokens + self.num_assistant_tokens

    def __add__(self, other: "TokenCounts"):
        return TokenCounts(
            num_system_and_user_tokens=self.num_system_and_user_tokens
            + other.num_system_and_user_tokens,
            num_assistant_tokens=self.num_assistant_tokens + other.num_assistant_tokens,
        )

def _base_convert_messages_to_element_retokenize(
    messages: List[Conversation.Message],
    tokenizer: PreTrainedTokenizerFast,
    message_start_tokens: dict[str, list[int]],
    message_end_tokens: dict[str, list[int]],
    message_extra_end_tokens: dict[str, list[int]],
    drop_thinking_fn:  Callable[[Conversation.Message], Conversation.Message] = lambda x: x,
) -> DatasetElement:
    input_ids, topk_token_ids, topk_logprobs, topk_token_idxs = [], [], [], []
    token_counts = TokenCounts()

    for i, message in enumerate(messages):
        if message.role == "assistant":
            message = drop_thinking_fn(message)

        token_ids = tokenizer.encode(message.content, add_special_tokens=False)
        token_ids += message_end_tokens[message.role] + message_extra_end_tokens[message.role]
        msg_input_ids = message_start_tokens[message.role] + token_ids

        if message.role == "assistant":
            topk_token_ids.append(token_ids)
            topk_logprobs.append(np.zeros(len(token_ids), dtype=np.float16))
            topk_token_idxs.append(np.arange(len(token_ids), dtype=np.int32) + len(input_ids) + len(message_start_tokens[message.role]))

        input_ids.extend(msg_input_ids)
    
        # FIXME: this is broken in the case that we truncate the input ids
        token_counts += TokenCounts(
            num_system_and_user_tokens=len(input_ids) if message.role == "user" else 0,
            num_assistant_tokens=len(input_ids) if message.role == "assistant" else 0,
        )

    return DatasetElement(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        topk_token_ids=torch.from_numpy(np.concatenate(topk_token_ids)),
        topk_logprobs=torch.from_numpy(np.concatenate(topk_logprobs)),
        topk_token_idxs=torch.from_numpy(np.concatenate(topk_token_idxs)),
        metadata=[],
        token_counts=token_counts,
    )



def _base_convert_messages_to_element(
    messages: List[Conversation.Message],
    message_start_tokens: dict[str, list[int]],
    message_end_tokens: dict[str, list[int]],
    message_extra_end_tokens: dict[str, list[int]],
    tokenizer: PreTrainedTokenizerFast,
    drop_thinking_fn:  Callable[[Conversation.Message], Conversation.Message] = lambda x: x,
) -> DatasetElement:
    input_ids, topk_token_ids, topk_logprobs, topk_token_idxs = [], [], [], []
    token_counts = TokenCounts()

    for i, message in enumerate(messages):
        if message.token_ids is None:
            # Some synthesizers may not return token ids for some messages, in which case
            # we actually need to do the retokenization. 
            msg_token_ids = tokenizer.encode(message.content, add_special_tokens=False)
        else:
            msg_token_ids = list(message.token_ids)
        msg_input_ids = message_start_tokens[message.role] + msg_token_ids

        if message.role == "assistant":
            message = drop_thinking_fn(message)

        end_tokens = message_end_tokens[message.role]
        # usually, messages will end with some tokenizer-specific "end" token(s) (e.g. <|endoftext|>)
        ends_with_eot = len(msg_input_ids) >= len(end_tokens) and msg_input_ids[-len(end_tokens):] == end_tokens
        if (not ends_with_eot and i < len(messages) - 1):
            # (1) if it does not, this means that the generation hit the max token limit. In 
            # this case, if we're not the last message in the conversation, we add the end 
            # tokens in, so that the next message starts with the correct tokens.
            # Otherwise, we can just leave it as is.
            msg_input_ids += end_tokens + message_extra_end_tokens[message.role]
        elif ends_with_eot and i < len(messages) - 1:
            # (2) if it does end with the eot tokens, then we just need to add any 
            # extra end tokens that come after the eot tokens and before the next 
            # message starts.
            msg_input_ids += message_extra_end_tokens[message.role]
        elif ends_with_eot and i == len(messages) - 1:
            # (3) if we're the last message in the conversation, then we don't need to 
            # add anything.
            pass

        if message.top_logprobs is not None:
            topk_token_ids.append(message.top_logprobs.token_id)
            topk_logprobs.append(message.top_logprobs.logprobs)
            topk_token_idxs.append(message.top_logprobs.token_idx + len(input_ids) + len(message_start_tokens[message.role]))

        input_ids.extend(msg_input_ids)
    
        # FIXME: this is broken in the case that we truncate the input ids
        token_counts += TokenCounts(
            num_system_and_user_tokens=len(input_ids) if message.role == "user" else 0,
            num_assistant_tokens=len(input_ids) if message.role == "assistant" else 0,
        )

    return DatasetElement(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        topk_token_ids=torch.from_numpy(np.concatenate(topk_token_ids)),
        topk_logprobs=torch.from_numpy(np.concatenate(topk_logprobs)),
        topk_token_idxs=torch.from_numpy(np.concatenate(topk_token_idxs)),
        metadata=[],
        token_counts=token_counts,
    )


@dataclass
class DatasetElement:
    input_ids: torch.Tensor

    metadata: list[dict[str, Any]]
    token_counts: TokenCounts

    topk_logprobs: Optional[torch.Tensor] = None
    topk_token_ids: Optional[torch.Tensor] = None
    topk_token_idxs: Optional[torch.Tensor] = None

@dataclass
class DatasetBatch:
    input_ids: torch.Tensor
    element_ids: torch.Tensor
    position_ids: torch.Tensor

    metadata: list[dict[str, Any]]
    token_counts: TokenCounts
    loss_weight: Optional[torch.Tensor] = None

    topk_logprobs: Optional[torch.Tensor] = None
    topk_token_ids: Optional[torch.Tensor] = None
    topk_token_idxs: Optional[torch.Tensor] = None


def msg(content, role: Literal["user"] | Literal["assistant"] | Literal["system"]):
    return {"content": content, "role": role}


class DataSource(BaseConfig):
    path: str
    type: Literal["local", "wandb", "hf"]
    limit: int | None = None

def _prepare_data_source(source: str | DataSource) -> list[Conversation]:
    if isinstance(source, str):
        is_local = ".pkl" in source or ".parquet" in source
        source = DataSource(path=source, type="local" if is_local else "wandb")
    
    if source.type == "local":
        data = read_conversations(source.path)
    elif source.type == "wandb":
        data = read_conversations_from_wandb(source.path)
    elif source.type == "hf":
        data = read_conversations_from_hf(source.path)
    else:
        raise ValueError(f"Unsupported data source type: {source.type}")

    return data[:source.limit] if source.limit is not None else data


class TrainDataset(Dataset):
    """ This dataset 

    - In "truncate" mode, if adding an element to the current batch would exceed the `seq_length`, the element 
    is added to a new batch instead. However, if an individual element's sequence length exceeds `seq_length`, 
    it is forced to be truncated and added to a new batch on its own.
    
    - In "pad" mode, elements are added to the current batch until adding another would exceed `seq_length`. 
    The current batch is then finalized, and a new batch is started. The final batch is padded to ensure 
    consistent batch sizes.

    Args:
        packing_mode (Literal["truncate", "pad"]): The mode of operation for batching, either "truncate" or "pad".
        packed_seq_length (int): The maximum allowed sequence length for each batch.
        shuffle (bool): Whether to shuffle the batches. Currently, only shuffling is supported.
    """

    class Config(ObjectConfig):
        _pass_as_config = True
        data_sources: list[str | DataSource]  
        is_wandb: bool = False
        top_k_logits: int = 20
        targets: Literal["logits", "tokens"] = "logits"

        prob_drop_thinking: float = 1.0

        packing_mode: Literal["truncate", "pad"]="pad"
        packed_seq_length: int = 2048

        user_prompt_prefix: list[str] | None = None


    def __init__(self, config: Config, model_helper: ModelHelper, seed: int):

        self.config = config
        self.model_helper = model_helper
        self.tokenizer = model_helper.tokenizer
        
        self.elements: List[DatasetElement] = self._prepare_elements()
        # each batch is a list of element indices
        self.batches: List[List[int]] = self._prepare_batches(seed=seed)  

    def _prepare_elements(self) -> list[DatasetElement]:
        data = []
        for source in self.config.data_sources:
            data.extend(_prepare_data_source(source))

        elements = []
        for row in data:
            elements.append(self.model_helper.messages_to_element(
                row.messages,
                retokenize=self.config.targets == "tokens",
                tokenizer=self.tokenizer,
                prob_drop_thinking=self.config.prob_drop_thinking,
            ))

        return elements
    
    def _prepare_batches(self, seed: int) -> List[List[int]]:
        """
        This function attempts to create batches of dataset elements such that the total sequence length of each batch 
        does not exceed a specified limit (`seq_length`). The batching process can operate in two modes: "truncate" 
        and "pad". 

        Note that this function does not actually handle the truncation or padding, this is left to the collate function.
        Which is applied the fly in the dataloader worker. 
        """
        batches = [] 
        elem_idxs = random.Random(seed).sample(range(len(self.elements)), len(self.elements))
        queue = deque(elem_idxs)

        curr_batch, curr_seq_len = [], 0
        while queue:
            idx = queue[0]
            elem: DatasetElement = self._get_element(idx)
            
            if curr_seq_len == 0 and len(elem.input_ids) > self.config.packed_seq_length:
                # if the current element is by itself longer than the sequence length,
                # then the only option is to truncate it. So we just add it to the batch
                # and start a new batch.
                curr_batch.append(queue.popleft())
                batches.append(curr_batch)
                curr_batch, curr_seq_len = [], 0
            elif curr_seq_len + len(elem.input_ids) > self.config.packed_seq_length:
                # when the current batch would be too long if we add the current element,
                # if we are in truncate mode, then we just add the current element to the batch
                # and start a new batch. Otherwise, we just start a new batch.
                if self.config.packing_mode == "truncate":
                    curr_batch.append(queue.popleft())
                batches.append(curr_batch)
                curr_batch, curr_seq_len = [], 0
            else:
                # otherwise, we just add the current element to the batch and continue
                curr_batch.append(queue.popleft())
                curr_seq_len += len(elem.input_ids)
        
        if curr_batch:
            # need to add the last batch
            batches.append(curr_batch)
        
        return batches

    def __len__(self):
        return len(self.batches)
    
    def _get_element(self, elem_idx: int) -> DatasetElement:
        return self.elements[elem_idx]
    
    def _get_batch(self, batch_idx: int):
        elem_idxs = self.batches[batch_idx]
        elements = [self._get_element(elem_idx) for elem_idx in elem_idxs]
        return self.collate(elements)

    def __getitem__(self, index: int) -> DatasetBatch:
        return self._get_batch(index)
        
    def reload(self):
        # Check if dataset has data_source_indices attribute
        if hasattr(self, "data_source_indices") and self.data_source_indices:
            combined = list(zip(self.data, self.data_source_indices))
            random.shuffle(combined)
            self.data, self.data_source_indices = zip(*combined)
            self.data = list(self.data)
            self.data_source_indices = list(self.data_source_indices)
        else:
            # Just shuffle the data if no data_source_indices
            random.shuffle(self.data)

    def collate(
        self,
        batch: list[DatasetElement]
    ) -> DatasetBatch:
        """
        Collate a list of dataset elements into a single sequence of length `self.config.packed_seq_length`.
        The elements are packed into a single sequence 
        """
        if not batch:
            raise ValueError("Empty batch provided to collate function")

        input_ids, element_ids, position_ids = [], [], []
        topk_token_ids, topk_logprobs, topk_token_idxs = [], [], []
        metadatas = []
        token_counts = TokenCounts()
        curr_token_idx = 0
        for element_id, element in enumerate(batch):
            input_ids.append(element.input_ids)
            element_ids.append(torch.full_like(element.input_ids, element_id, dtype=torch.long))
            position_ids.append(torch.arange(len(element.input_ids), dtype=torch.long))
            topk_token_ids.append(element.topk_token_ids)
            topk_logprobs.append(element.topk_logprobs)
            topk_token_idxs.append(element.topk_token_idxs + curr_token_idx)
            metadatas.append(element.metadata)
            token_counts += element.token_counts
            curr_token_idx += len(element.input_ids)
        
        input_ids = torch.cat(input_ids, dim=0)
        element_ids = torch.cat(element_ids, dim=0)
        position_ids = torch.cat(position_ids, dim=0)
        topk_token_ids = torch.cat(topk_token_ids, dim=0)
        topk_logprobs = torch.cat(topk_logprobs, dim=0)
        topk_token_idxs = torch.cat(topk_token_idxs, dim=0)

        if len(input_ids) > self.config.packed_seq_length:
            # if the input ids are longer than the sequence length, 
            # we need to truncate them
            input_ids = input_ids[:self.config.packed_seq_length]
            element_ids = element_ids[:self.config.packed_seq_length]
            position_ids = position_ids[:self.config.packed_seq_length]

            # we also need to filter out any targets that are from the truncated part of
            # the input ids
            mask = topk_token_idxs < self.config.packed_seq_length
            topk_token_ids = topk_token_ids[mask]
            topk_logprobs = topk_logprobs[mask]
            topk_token_idxs = topk_token_idxs[mask]

        elif len(input_ids) < self.config.packed_seq_length:
            # if the input ids are shorter than the sequence length, we need to pad them
            # it is critical that the sequence length stays constant to avoid 
            # flex attention recompiles.
            padding = torch.full((self.config.packed_seq_length - len(input_ids),), 0, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
            element_ids = torch.cat([element_ids, padding])
            position_ids = torch.cat([position_ids, padding])

        return DatasetBatch(
            input_ids=input_ids,
            element_ids=element_ids,
            position_ids=position_ids,
            topk_token_ids=topk_token_ids,
            topk_logprobs=topk_logprobs,
            topk_token_idxs=topk_token_idxs,
            metadata=metadatas,
            token_counts=token_counts,
        )

class LossEvalDataset(TrainDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        
        # path to a file containing conversations 
        # accepted formats are: *.jsonl, *.parquet, *.pkl, a wandb artifact id, or a huggingface repo id
        data_source: str | DataSource

        is_wandb: bool = False
        targets: Literal["tokens"] = "tokens"

        packing_mode: Literal["truncate", "pad"]="pad"
        packed_seq_length: int = 2048

        system_prompt: str | None = None
        

        user_prompt_prefix: list[str] | None = None


    def _prepare_elements(self) -> list[Conversation]:
        data: list[Conversation] = []
        data = _prepare_data_source(self.config.data_source)

        elements = []
        for row in data:
            if self.config.system_prompt is not None:
                messages = [
                    Conversation.Message(role="system", content=self.config.system_prompt, token_ids=None),
                    *row.messages,
                ]
            else:
                messages = row.messages

            elements.append(self.model_helper.messages_to_element(
                messages,
                retokenize=self.config.targets == "tokens",
                tokenizer=self.tokenizer,
            ))
        
        return elements
    

@dataclass
class GenerateEvalDatasetElement:
    input_ids: torch.Tensor

    # messages to be used as the prompt
    prompt: List[Dict[str, str]]

    answer: Optional[str]
    metadata: dict[str, Any]
    convo_id: Optional[str] = None


@dataclass
class GenerateEvalDatasetElement:
    input_ids: torch.Tensor

    # messages to be used as the prompt
    prompt: List[MessageDict] | str

    answer: Optional[str]
    metadata: dict[str, Any]
    convo_id: Optional[str] = None

class GenerateEvalDataset(Dataset):
    class Config(ObjectConfig):
        _pass_as_config = True

        data_source: Optional[str | DataSource] = None
        cot: bool = False
    
    def __init__(self, config: Config, model_helper: ModelHelper, seed: int):
        self.config = config
        self.model_helper = model_helper
        self.tokenizer = model_helper.tokenizer
        self.seed = seed

        assert self.config.data_source is not None, "data_source is required for GenerateEvalDataset"
        self.data: list[Conversation] = _prepare_data_source(self.config.data_source)

    @abstractmethod
    def __getitem__(self, index: int) -> GenerateEvalDatasetElement:
        convo: Conversation = self.data[index]
        assert len(convo.messages) > 1, "Conversation must have at least 2 messages"
        assert convo.messages[-1].role == "assistant", "Last message must be assistant"


        kwargs = self.model_helper.get_apply_chat_template_kwargs(self.config.cot)
        input_ids = self.tokenizer.apply_chat_template(
            [
                {"role": msg.role, "content": msg.content}
                for msg in convo.messages[:-1]
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=self.model_helper.get_chat_template(),
            **kwargs,
        )

        return GenerateEvalDatasetElement(
            input_ids=input_ids,    
            prompt=[
                {"role": msg.role, "content": msg.content}
                for msg in convo.messages[:-1]
            ],
            answer=convo.messages[-1].content,
            convo_id=str(index),
            metadata={"idx": index}
        )
    
    def __len__(self):
        return len(self.data)

        