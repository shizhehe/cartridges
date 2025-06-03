from collections import defaultdict
import pickle
import pyarrow as pa
from pathlib import Path
import random
import pandas as pd
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Any
import torch
import os

from transformers import PreTrainedTokenizerFast
from pydrantic import ObjectConfig

from cartridges.context import StructuredContext
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


@dataclass
class CartridgeDatasetElementLogitLabels:
    input_ids: torch.Tensor
    

    topk_logprobs: torch.Tensor
    topk_tokens: torch.Tensor

    mask: torch.Tensor
    metadata: list[dict[str, Any]]
    token_counts: TokenCounts
    
    context_ids: torch.Tensor|None = None
    context_mask: torch.Tensor|None = None


@dataclass
class CartridgeDatasetBatchLogitLabels:
    input_ids: torch.Tensor

    topk_logprobs: torch.Tensor
    topk_tokens: torch.Tensor

    mask: torch.Tensor
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


SENTINEL_MESSAGE = "sentinel"


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
        max_sequence_length: int | None = None
        is_wandb: bool = False
        label_type: Literal["tokens"] | Literal["logits"] | Literal["online"]
        top_k_logits: int = 20
        dataset_weights: Optional[list[float]] = None
        user_prompt_prefix: list[str] | None = None
        # system_prompt: str | None = None

    data: list[TrainingExample]
    context: StructuredContext

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):

        self.config = config

        self.data = []
        self.datasets = []
        context = None

        for source, limit in config.data_sources:
            if source.endswith(".pkl"):
                pkl_path = source
                if not Path(pkl_path).exists():
                    modal_pkl_path = os.path.join("/root/Cartridges-datasets", os.path.relpath(source, "/"))
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
            assert data_dict.keys() == {"rows", "context"}

            self.data += data_dict["rows"][:limit]
            if context is None:
                context = data_dict["context"]

        assert context is not None
        self.context = context

        self.tokenizer = tokenizer
    
    def _getitem_tokens(
        self,
        index: int,
        row: TrainingExample,
    ) -> CartridgeDatasetElementTokenLabels:
        token_ids = torch.tensor(row.token_ids)
        input_ids = token_ids
        token_labels = token_ids

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
        
        # truncate the input ids
        if (
            self.config.max_sequence_length is not None
            and len(row.token_ids) > self.config.max_sequence_length
        ):
            row.token_ids = row.token_ids[:self.config.max_sequence_length + 1]
            row.top_logprob_ids = row.top_logprob_ids[:self.config.max_sequence_length]
            row.top_logprob_logprobs = row.top_logprob_logprobs[:self.config.max_sequence_length]

        element = CartridgeDatasetElementLogitLabels(
            input_ids=torch.from_numpy(row.token_ids[:-1]),
            topk_tokens=torch.from_numpy(row.top_logprob_ids),
            topk_logprobs=torch.from_numpy(row.top_logprob_logprobs),
            mask=torch.full_like(torch.from_numpy(row.token_ids[:-1]), True),
            metadata=[],
            
            # FIXME: this is broken in the case that we truncate the input ids
            token_counts=TokenCounts(
                # -1 because we last token is jut output (not input)
                num_assistant_tokens=row.num_output_tokens - 1,
                num_system_and_user_tokens=len(row.token_ids)
                - row.num_output_tokens
                - 1,
            ),
        )

        assert len(element.topk_logprobs.shape) == 2
        assert element.topk_logprobs.shape == element.topk_tokens.shape
        assert element.topk_logprobs.shape[1] == self.config.top_k_logits

        assert len(element.input_ids.shape) == 1
        assert element.input_ids.shape == element.mask.shape

        assert element.topk_logprobs.shape[0] == element.input_ids.shape[0]

        return element

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
    ) -> CartridgeDatasetBatchLogitLabels | CartridgeDatasetBatchTokenLabels:
        """
        Collate a list of dataset elements into a batch.

        Args:
            batch: A list of dataset elements, either logit labels or token labels

        Returns:
            A batch object containing the batched input_ids, labels, and mask (for logit labels)

        Raises:
            ValueError: If the batch contains mixed element types
        """
        # TODO: get rid of this hack
        # batch = [i for i in batch if i is not None]
        if not batch:
            raise ValueError("Empty batch provided to collate function")

        # Ensure all elements are of the same type
        first_element_type = type(batch[0])
        if not all(isinstance(element, first_element_type) for element in batch):
            raise ValueError(
                f"All elements in the batch must be of the same type. "
                f"Expected {first_element_type.__name__}, but found mixed types."
            )

        # Copy over input IDs
        max_len = max(element.input_ids.size(0) for element in batch)
        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_len), EOS_TOKEN_ID, dtype=torch.long)
        metadata = [elem.metadata for elem in batch]
        token_counts = [elem.token_counts for elem in batch]

        # Determine the batch type based on the first element
        if isinstance(batch[0], CartridgeDatasetElementTokenLabels):
            # Create tensors to hold the batched data
            labels = torch.full(
                (batch_size, max_len), -100, dtype=torch.long
            )  # -100 is ignored in loss calculation

            mask = torch.zeros_like(input_ids, dtype=torch.bool)

            for i, element in enumerate(batch):
                assert isinstance(element, CartridgeDatasetElementTokenLabels)
                seq_len = element.input_ids.shape[0]
                input_ids[i, :seq_len] = element.input_ids
                labels[i, : element.input_ids.shape[0]] = element.labels
                mask[i, :seq_len] = element.mask

            return CartridgeDatasetBatchTokenLabels(
                input_ids,
                labels,
                metadata=metadata,
                token_counts=token_counts,
                mask=mask,
            )

        try:
            assert isinstance(batch[0], CartridgeDatasetElementLogitLabels)
        except:
            breakpoint()

        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        k = batch[0].topk_tokens.shape[-1]

        topk_tokens = torch.full(
            (batch_size, max_len, k),
            fill_value=EOS_TOKEN_ID,
            dtype=torch.long,
        )
        topk_logprobs = torch.full(
            (batch_size, max_len, k),
            fill_value=-k,  # HACK: prevent overflows
            dtype=torch.float,
        )

        # Fill the tensors with data from each element
        for i, element in enumerate(batch):
            assert isinstance(element, CartridgeDatasetElementLogitLabels)

            assert element.topk_tokens.shape == element.topk_logprobs.shape
            assert len(element.topk_tokens.shape) == 2

            seq_len = element.input_ids.shape[0]

            input_ids[i, :seq_len] = element.input_ids
            topk_tokens[i, :seq_len] = element.topk_tokens
            topk_logprobs[i, :seq_len] = element.topk_logprobs
            mask[i, :seq_len] = element.mask

        loss_weight = None
        if (
            hasattr(self, "data_source_indices")
            and len(self.data_source_indices) > 0
            and len(self.data_source_indices) >= len(batch)
        ):
            loss_weight = torch.ones_like(input_ids, dtype=torch.float)
            for i, element in enumerate(batch):
                seq_len = element.input_ids.shape[0]
                dataset_idx = self.data_source_indices[i]
                loss_weight[i, :seq_len] = (
                    self.dataset_weights[dataset_idx]
                    if hasattr(self, "dataset_weights")
                    else 1.0
                )
        return CartridgeDatasetBatchLogitLabels(
            input_ids=input_ids,
            topk_tokens=topk_tokens,
            topk_logprobs=topk_logprobs,
            mask=mask,
            loss_weight=loss_weight,
            metadata=metadata,
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


class CartridgeDatasetBatchSampler(BatchSampler):
    def __init__(
        self, 
        sampler: Sampler,
        dataset: CartridgeTrainDataset,
        batch_size: int, 
        shuffle: bool = True,
    ):
        self.batches = [] 
        assert shuffle
        
        # get lengths of each element 
        lengths = []
        for idx in sampler:
            elem: CartridgeDatasetElementLogitLabels = dataset[idx]
            lengths.append((len(elem.input_ids), idx))
        sorted_idxs = sorted(lengths, key=lambda x: x[0])

        self.idx_batches = [
            [idx for _, idx in sorted_idxs[i:i + batch_size]] 
            for i in range(0, len(sorted_idxs), batch_size) 
        ]
        
        random.shuffle(self.idx_batches)
    
    def __iter__(self):
        for batch_idxs in self.idx_batches:
            yield batch_idxs

    def __len__(self):
        return len(self.idx_batches)

