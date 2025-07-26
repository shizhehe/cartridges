from __future__ import annotations
from abc import abstractmethod
from collections import deque
import pickle
from pathlib import Path
import random

from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Any
import torch
import os

from transformers import PreTrainedTokenizerFast
from pydrantic import ObjectConfig
import numpy as np

from cartridges.structs import TrainingExample
from torch.utils.data import BatchSampler, Sampler
from cartridges.utils import get_logger, wandb

# SE(04/02): required to silence tokenizer warnings when using dataloders with
# multiple worker processes
os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = get_logger(__name__)

BOS_TOKEN_ID = 128000
EOS_TOKEN_ID = 128009
START_HEADER_ID = 128006
END_HEADER_ID = 128007
USER_TOKEN_ID = 882
ASSISTANT_TOKEN_ID = 78191

# SE (03/08): We need to add {% generation %} to the template to get it to work
# with `return_assistant_tokens_mask=True`.
# Tested that the template is equivalent in scratch/sabri/m03d08_dev_tokenizer_tempalte.ipynb
# RE(03/15): there were a few things that were not equivalent.
TEMPLATE = """\
{%- for message in messages %}
    {%- if  (message.role == 'assistant') %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}{% generation %}{{- message['content'] | trim + '<|eot_id|>' }}{% endgeneration %}

    {%- else %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
        
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""


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


def _base_convert_messages_to_element(
    messages: List[TrainingExample.Message],
    message_start_tokens: dict[str, list[int]],
    message_end_tokens: dict[str, list[int]],
) -> CartridgeDatasetElement:
    input_ids, topk_token_ids, topk_logprobs, topk_token_idxs = [], [], [], []
    token_counts = TokenCounts()

    for message in messages:
        msg_input_ids = message_start_tokens[message.role] + message.token_ids + message_end_tokens[message.role]

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

    return CartridgeDatasetElement(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        topk_token_ids=torch.from_numpy(np.concatenate(topk_token_ids)),
        topk_logprobs=torch.from_numpy(np.concatenate(topk_logprobs)),
        topk_token_idxs=torch.from_numpy(np.concatenate(topk_token_idxs)),
        metadata=[],
        token_counts=token_counts,
    )

def qwen_messages_to_element(
    messages: List[TrainingExample.Message],
) -> CartridgeDatasetElement:

    return _base_convert_messages_to_element(
        messages,
        message_start_tokens={
            "user": [151644, 872,198],
            "assistant": [151644, 77091,198],
        },
        message_end_tokens={
            "user": [151645, 198],
            "assistant": [151645, 198],
        },

    )

def llama_messages_to_element(
    messages: List[TrainingExample.Message],
) -> CartridgeDatasetElement:
    return _base_convert_messages_to_element(
        messages,
        message_start_tokens={
            "user": [128006, 882,128007],
            "assistant": [128006, 78191,128007],
        },
        message_end_tokens={
            "user": [128009],
            "assistant": [128009],
        },
    )

MODEL_TO_MESSAGE_CONVERTER = {
    "Qwen/Qwen3-4b": qwen_messages_to_element,
    "meta-llama/Llama-3.2-3B-Instruct": llama_messages_to_element,
}
MODEL_TO_MESSAGE_CONVERTER = {k.lower(): v for k, v in MODEL_TO_MESSAGE_CONVERTER.items()}


@dataclass
class CartridgeDatasetElement:
    input_ids: torch.Tensor
    
    topk_logprobs: torch.Tensor
    topk_token_ids: torch.Tensor
    topk_token_idxs: torch.Tensor

    metadata: list[dict[str, Any]]
    token_counts: TokenCounts
    
    context_ids: torch.Tensor|None = None
    context_mask: torch.Tensor|None = None


@dataclass
class CartridgeDatasetBatch:
    input_ids: torch.Tensor
    element_ids: torch.Tensor
    position_ids: torch.Tensor

    topk_logprobs: torch.Tensor
    topk_token_ids: torch.Tensor
    topk_token_idxs: torch.Tensor

    metadata: list[dict[str, Any]]
    token_counts: TokenCounts
    loss_weight: Optional[torch.Tensor] = None
    context_ids: Optional[torch.Tensor] = None
    context_mask: Optional[torch.Tensor] = None


def msg(content, role: Literal["user"] | Literal["assistant"] | Literal["system"]):
    return {"content": content, "role": role}


@dataclass
class TokenData:
    token_id: int
    top_tokens: list[int]
    top_logprobs: list[float]
    apply_loss: bool


class CartridgeTrainDataset(Dataset):
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
        data_sources: list[tuple[str, int | None],]  # path, limit
        is_wandb: bool = False
        top_k_logits: int = 20

        packing_mode: Literal["truncate", "pad"]="pad"
        packed_seq_length: int = 2048

        dataset_weights: Optional[list[float]] = None
        user_prompt_prefix: list[str] | None = None


    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):

        self.config = config
        self.tokenizer = tokenizer
        
        self.elements: List[TrainingExample] = self._prepare_elements()
        # each batch is a list of element indices
        self.batches: List[List[int]] = self._prepare_batches(seed=seed)  
    
    def _prepare_elements(self) -> list[TrainingExample]:
        data = []
        for source, limit in self.config.data_sources:
            if source.endswith(".pkl"):
                pkl_path = source
                if not Path(pkl_path).exists():
                    modal_pkl_path = os.path.join("/root/cartridges-datasets", os.path.relpath(source, "/"))
                    if not Path(modal_pkl_path).exists():
                        raise FileNotFoundError(f"File {source} not found either locally or in modal")
                    pkl_path = modal_pkl_path

            else:
                dataset_dir = (
                    wandb.get_artifact_dir(source) / "dataset"
                    if self.config.is_wandb
                    else Path(source)
                )

                if not dataset_dir.exists():
                    wandb.download_artifact(source)
                pkl_path = dataset_dir / "dataset.pkl"

            with open(pkl_path, "rb") as f:
                data_dict = pickle.load(f)
            assert data_dict.keys() == {"rows"}

            data += data_dict["rows"][:limit]

        return data
    
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
            elem: CartridgeDatasetElement = self._get_element(idx)
            
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
    
    def _get_element(self, elem_idx: int) -> CartridgeDatasetElement:
        row = self.elements[elem_idx]
        return MODEL_TO_MESSAGE_CONVERTER[self.tokenizer.name_or_path.lower()](row.messages)
    
    def _get_batch(self, batch_idx: int):
        elem_idxs = self.batches[batch_idx]
        elements = [self._get_element(elem_idx) for elem_idx in elem_idxs]
        return self.collate(elements)

    def __getitem__(self, index: int) -> CartridgeDatasetBatch:
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
        batch: (
            list[CartridgeDatasetElement]
        ),
    ) -> CartridgeDatasetBatch:
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
            # if the input ids are longer than the sequence length, we need to truncate 
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

        return CartridgeDatasetBatch(
            input_ids=input_ids,
            element_ids=element_ids,
            position_ids=position_ids,
            topk_token_ids=topk_token_ids,
            topk_logprobs=topk_logprobs,
            topk_token_idxs=topk_token_idxs,
            metadata=metadatas,
            token_counts=token_counts,
        )


@dataclass
class CartridgeGenerateDatasetElement:
    input_ids: torch.Tensor
    prompt: str

    answer: Optional[str]
    metadata: dict[str, Any]
    convo_id: Optional[str] = None
    
    # this is needed for some datasets, like MMLU, where the in context examples
    # are structured as prior messages
    prompt_messages: Optional[List[Dict[str,str]]] = None

class CartridgeGenerateDataset(Dataset):
    class Config(ObjectConfig):
        _pass_as_config = True
    
    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):
        self.config = config
        self.tokenizer = tokenizer
        self.seed = seed

    @abstractmethod
    def __getitem__(self, index: int) -> CartridgeGenerateDatasetElement:
        raise NotImplementedError
        