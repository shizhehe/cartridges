from __future__ import annotations
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
) -> CartridgeDatasetElementLogitLabels:
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

    return CartridgeDatasetElementLogitLabels(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        topk_token_ids=torch.from_numpy(np.concatenate(topk_token_ids)),
        topk_logprobs=torch.from_numpy(np.concatenate(topk_logprobs)),
        topk_token_idxs=torch.from_numpy(np.concatenate(topk_token_idxs)),
        metadata=[],
        token_counts=token_counts,
    )

def qwen_messages_to_element(
    messages: List[TrainingExample.Message],
) -> CartridgeDatasetElementLogitLabels:

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

MODEL_TO_MESSAGE_CONVERTER = {
    "Qwen/Qwen3-4b": qwen_messages_to_element,
}


@dataclass
class CartridgeDatasetElementLogitLabels:
    input_ids: torch.Tensor
    
    topk_logprobs: torch.Tensor
    topk_token_ids: torch.Tensor
    topk_token_idxs: torch.Tensor

    metadata: list[dict[str, Any]]
    token_counts: TokenCounts
    
    context_ids: torch.Tensor|None = None
    context_mask: torch.Tensor|None = None


@dataclass
class CartridgeDatasetBatchLogitLabels:
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
    
@dataclass
class CartridgeDatasetElementTokenLabels:
    input_ids: torch.Tensor
    labels: torch.Tensor
    metadata: dict[str, Any]
    token_counts: List[TokenCounts]
    mask: torch.Tensor


@dataclass
class CartridgeDatasetBatchTokenLabels:
    input_ids: torch.Tensor
    labels: torch.Tensor
    metadata: list[dict[str, Any]]
    token_counts: List[TokenCounts]
    mask: torch.Tensor


def msg(content, role: Literal["user"] | Literal["assistant"] | Literal["system"]):
    return {"content": content, "role": role}



@dataclass
class TokenData:
    token_id: int
    top_tokens: list[int]
    top_logprobs: list[float]
    apply_loss: bool


class CartridgeTrainDataset(Dataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        data_sources: list[tuple[str, int | None],]  # path, limit
        is_wandb: bool = False
        label_type: Literal["tokens", "logits"] = "logits"
        top_k_logits: int = 20
        dataset_weights: Optional[list[float]] = None
        user_prompt_prefix: list[str] | None = None
        # system_prompt: str | None = None

    data: list[TrainingExample]

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):

        self.config = config

        self.data: List[TrainingExample] = []
        self.datasets = []

        for source, limit in config.data_sources:
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
                    if config.is_wandb
                    else Path(source)
                )

                if not dataset_dir.exists():
                    wandb.download_artifact(source)
                pkl_path = dataset_dir / "dataset.pkl"

            with open(pkl_path, "rb") as f:
                data_dict = pickle.load(f)
            assert data_dict.keys() == {"rows"}

            self.data += data_dict["rows"][:limit]

        self.tokenizer = tokenizer
    
    def _getitem_tokens(
        self,
        index: int,
        row: TrainingExample,
    ) -> CartridgeDatasetElementTokenLabels:
        # TODO (SE): Support unique token ids
        # if row.token_ids is not None:
        #     token_ids = torch.tensor(row.token_ids)
        #     input_ids = token_ids
        # else: 
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in row.messages],
            add_generation_prompt=False,
            return_tensors="pt",
            chat_template=TEMPLATE,
        ).squeeze(0)
        token_labels = input_ids

        mask = torch.ones_like(input_ids, dtype=torch.bool)
        # SE(04/03): need to ensure that this is a boolean mask
        assert mask.dtype == torch.bool

        token_counts = TokenCounts(
            num_system_and_user_tokens=(~mask).sum(),
            num_assistant_tokens=mask.sum(),
        )
        return CartridgeDatasetElementTokenLabels(
            input_ids=input_ids,
            labels=token_labels,
            metadata=[],
            token_counts=token_counts,
            mask=mask,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> CartridgeDatasetElementLogitLabels:
        row = self.data[index]
        if self.config.label_type == "tokens":
            return self._getitem_tokens(index, row)

        assert isinstance(row, TrainingExample)
        return MODEL_TO_MESSAGE_CONVERTER[self.tokenizer.name_or_path](row.messages)

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
            list[CartridgeDatasetElementLogitLabels]
            | list[CartridgeDatasetElementTokenLabels]
        ),
        packed_seq_length: int,
    ) -> CartridgeDatasetBatchLogitLabels | CartridgeDatasetBatchTokenLabels:
        """
        Collate a list of dataset elements into a single sequence of length `seq_length`.
        The elements are packed into a single sequence 
        """
        if not batch:
            raise ValueError("Empty batch provided to collate function")

        # Determine the batch type based on the first element
        if isinstance(batch[0], CartridgeDatasetElementTokenLabels):
            raise NotImplementedError("Token labels are not supported yet.")

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

        if len(input_ids) > packed_seq_length:
            # if the input ids are longer than the sequence length, we need to truncate 
            # we need to truncate them
            input_ids = input_ids[:packed_seq_length]
            element_ids = element_ids[:packed_seq_length]
            position_ids = position_ids[:packed_seq_length]

            # we also need to filter out any targets that are from the truncated part of
            # the input ids
            mask = topk_token_idxs < packed_seq_length
            topk_token_ids = topk_token_ids[mask]
            topk_logprobs = topk_logprobs[mask]
            topk_token_idxs = topk_token_idxs[mask]

        elif len(input_ids) < packed_seq_length:
            # if the input ids are shorter than the sequence length, we need to pad them
            # it is critical that the sequence length stays constant to avoid 
            # flex attention recompiles.
            padding = torch.full((packed_seq_length - len(input_ids),), 0, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
            element_ids = torch.cat([element_ids, padding])
            position_ids = torch.cat([position_ids, padding])

        return CartridgeDatasetBatchLogitLabels(
            input_ids=input_ids,
            element_ids=element_ids,
            position_ids=position_ids,
            topk_token_ids=topk_token_ids,
            topk_logprobs=topk_logprobs,
            topk_token_idxs=topk_token_idxs,
            metadata=metadatas,
            token_counts=token_counts,
        )


class PackedBatchSampler(BatchSampler):
    """
    A sampler that organizes dataset elements into batches with a focus on sequence length constraints.

    This class attempts to create batches of dataset elements such that the total sequence length of each batch
    does not exceed a specified limit (`seq_length`). The batching process can operate in two modes: "truncate" 
    and "pad". 

    - In "truncate" mode, if adding an element to the current batch would exceed the `seq_length`, the element 
        is added to a new batch instead. However, if an individual element's sequence length exceeds `seq_length`, 
        it is forced to be truncated and added to a new batch on its own.
    
    - In "pad" mode, elements are added to the current batch until adding another would exceed `seq_length`. 
        The current batch is then finalized, and a new batch is started. The final batch is padded to ensure 
        consistent batch sizes.
    
    Note that the sampler does not actually handle the truncation or padding, this is 
    left to the collate function.

    Args:
        sampler (Sampler): A sampler that provides indices of dataset elements.
        dataset (CartridgeTrainDataset): The dataset containing elements to be batched.
        mode (Literal["truncate", "pad"]): The mode of operation for batching, either "truncate" or "pad".
        packed_seq_length (int): The maximum allowed sequence length for each batch.
        shuffle (bool): Whether to shuffle the batches. Currently, only shuffling is supported.
    """
    def __init__(
        self, 
        sampler: Sampler,
        dataset: CartridgeTrainDataset,
        packing_mode: Literal["truncate", "pad"]="pad",
        packed_seq_length: int = 2048,
        shuffle: bool = True,
    ):
        assert shuffle, "We only support shuffling for now."
        self.batches = [] 
        queue = deque(sampler)
        
        curr_batch, curr_seq_len = [], 0
        while queue:
            idx = queue[0]
            elem: CartridgeDatasetElementLogitLabels = dataset[idx]
            
            if curr_seq_len == 0 and len(elem.input_ids) > packed_seq_length:
                # if the current element is by itself longer than the sequence length,
                # then the only option is to truncate it. So we just add it to the batch
                # and start a new batch.
                curr_batch.append(queue.popleft())
                self.batches.append(curr_batch)
                curr_batch, curr_seq_len = [], 0
            elif curr_seq_len + len(elem.input_ids) > packed_seq_length:
                # when the current batch would be too long if we add the current element,
                # if we are in truncate mode, then we just add the current element to the batch
                # and start a new batch. Otherwise, we just start a new batch.
                if packing_mode == "truncate":
                    curr_batch.append(queue.popleft())
                self.batches.append(curr_batch)
                curr_batch, curr_seq_len = [], 0
            else:
                # otherwise, we just add the current element to the batch and continue
                curr_batch.append(queue.popleft())
                curr_seq_len += len(elem.input_ids)
        
        if curr_batch:
            # need to add the last batch
            self.batches.append(curr_batch)
        
        random.shuffle(self.batches)
    
    def __iter__(self):
        for batch_idxs in self.batches:
            yield batch_idxs

    def __len__(self):
        return len(self.batches)



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

class CartridgeGenerateDataset(CartridgeTrainDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        data_sources: list[tuple[str, int | None],]  # path, limit
        is_wandb: bool = False
        label_type: Literal["tokens"] | Literal["logits"]
        top_k_logits: int = 20
        dataset_weights: Optional[list[float]] = None
        user_prompt_prefix: list[str] | None = None
        # system_prompt: str | None = None

    def __getitem__(self, index: int) -> CartridgeGenerateDatasetElement:
        convo: ContextConvo = self.data[index]

        prompt = convo.messages[0].content

        input_ids = self.tokenizer.apply_chat_template(
            (
                # [{"role": "system", "content": self.config.system_prompt}]
                # if self.config.system_prompt is not None
                # else []
                []
            )
            + [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=TEMPLATE,
        )

        return CartridgeGenerateDatasetElement(
            input_ids=input_ids,
            prompt=prompt,
            answer=convo.messages[1].content,
            convo_id=convo.id,
            metadata={
                "idx": index,
            },
        )