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

from capsules.generate.run import BaseContextConfig
from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.generate.structs import (
    Context,
    ContextConvoDataset,
    ContextConvo,
    Message,
    TrainingExample,
)
from torch.utils.data import BatchSampler, Sampler
import torch.distributed as dist
from capsules.transforms import ConvoTransform, ConvoTransformConfig
from capsules.utils import get_logger, wandb
import numpy as np

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
class CapsuleDatasetElementLogitLabels:
    input_ids: torch.Tensor
    

    topk_logprobs: torch.Tensor
    topk_tokens: torch.Tensor

    mask: torch.Tensor
    metadata: list[dict[str, Any]]
    token_counts: TokenCounts
    
    context_ids: torch.Tensor|None = None
    context_mask: torch.Tensor|None = None


@dataclass
class CapsuleDatasetBatchLogitLabels:
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
class CapsuleDatasetElementTokenLabels:
    input_ids: torch.Tensor
    labels: torch.Tensor
    metadata: dict[str, Any]
    token_counts: List[TokenCounts]
    mask: torch.Tensor


@dataclass
class CapsuleDatasetBatchTokenLabels:
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


class CapsuleDataset(Dataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        data_sources: list[tuple[str, int | None],]  # path, limit
        is_wandb: bool = False
        label_type: Literal["tokens"] | Literal["logits"] | Literal["online"]
        top_k_logits: int = 20
        dataset_weights: Optional[list[float]] = None
        user_prompt_prefix: list[str] | None = None
        # system_prompt: str | None = None
        convo_transforms: list[ConvoTransformConfig] | None = None

    data: list[ContextConvo]
    context: Context
    convo_transforms: list[ConvoTransform] = None

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config

        self.data = []
        self.datasets = []
        self.data_source_indices = []

        assert config.data_sources
        context: Optional[Context] = None
        if config.dataset_weights:
            assert len(config.dataset_weights) == len(
                config.data_sources
            ), "Must provide weight for each dataset"
            self.dataset_weights = config.dataset_weights
        else:
            self.dataset_weights = [1.0] * len(config.data_sources)

        for i, (path, limit) in enumerate(config.data_sources):
            logger.info(f"Loading dataset {i + 1}/{len(config.data_sources)}")
            dataset = ContextConvoDataset.load(
                path,
                is_wandb=config.is_wandb,
            )

            if i == 0:
                assert context is None
                context = dataset.context
            else:
                assert context is not None
                # assert dataset.context.to_string() == context.to_string()
            self.datasets.append(dataset)
            try:
                dataset_rows = dataset.rows['rows'] if limit is None else dataset.rows['rows'][:limit]
            except:
                dataset_rows = dataset.rows if limit is None else dataset.rows[:limit]
            self.data += dataset_rows
            self.data_source_indices.extend([i] * len(dataset_rows))

        if self.config.user_prompt_prefix is None:
            self.config.user_prompt_prefix = [""] * len(self.datasets)

        assert context is not None
        self.context = context

        if self.config.convo_transforms is not None:
            self.convo_transforms = [
                transform.instantiate(tokenizer=tokenizer)
                for transform in self.config.convo_transforms
            ]
        else:
            self.convo_transforms = []

        self.tokenizer = tokenizer
        logger.info("Datasets loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> CapsuleDatasetElementLogitLabels | CapsuleDatasetElementTokenLabels:
        convo: ContextConvo = self.data[index]

        if self.convo_transforms:
            for transform in self.convo_transforms:
                convo = transform(convo)

        if self.config.label_type == "tokens":
            return self._getitem_tokens(index, convo=convo)
        elif self.config.label_type == "logits":
            return self._getitem_logits(index, convo=convo)
        else:
            raise ValueError(f"Invalid label type: {self.config.label_type}")

    def _getitem_tokens(
        self,
        index: int,
        convo: ContextConvo,
    ) -> CapsuleDatasetElementTokenLabels:
        apply_chat_template = lambda messages: self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            chat_template=TEMPLATE,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )

        input_ids_list: dict[str, list[int]] = apply_chat_template(
            [
                msg(
                    (
                        self.config.user_prompt_prefix[self.data_source_indices[index]]
                        if m.role == "user" and hasattr(self, "data_source_indices")
                        else ""
                    )
                    + m.content,
                    m.role,
                )
                for m in convo.messages
            ]
        )  # type: ignore

        mask = torch.tensor(input_ids_list["assistant_masks"], dtype=torch.bool)
        input_ids = torch.tensor(input_ids_list["input_ids"])
        token_labels = torch.full_like(input_ids, -100)
        token_labels[mask == 1] = input_ids[mask == 1]

        # SE(04/03): need to ensure that this is a boolean mask
        assert mask.dtype == torch.bool

        token_counts = TokenCounts(
            num_system_and_user_tokens=(~mask).sum(),
            num_assistant_tokens=mask.sum(),
        )
        return CapsuleDatasetElementTokenLabels(
            input_ids,
            token_labels,
            metadata=convo.metadata,
            token_counts=token_counts,
            mask=mask,
        )
        

    def _getitem_logits(
        self,
        index: int,
        convo: ContextConvo,
    ) -> CapsuleDatasetElementLogitLabels:
        apply_chat_template = lambda message: self.tokenizer.apply_chat_template(
            [
                msg(
                    (
                        self.config.user_prompt_prefix[self.data_source_indices[index]]
                        if message.role == "user"
                        and hasattr(self, "data_source_indices")
                        else ""
                    )
                    + message.content,
                    message.role,
                )
            ],
            add_generation_prompt=False,
            chat_template=TEMPLATE,
        )

        mask: List[bool] = []
        input_ids: List[int] = []
        topk_tokens: List[List[int]] = []
        topk_logprobs: List[List[float]] = []
        token_counts: Dict[str, int] = defaultdict(int)

        # Placeholder for messages without logprobs
        top_tokens_placeholder = [EOS_TOKEN_ID] * self.config.top_k_logits
        top_logprobs_placeholder = [
            -float(self.config.top_k_logits)
        ] * self.config.top_k_logits
        for message in convo.messages:

            if message.logprobs is not None and message.token_ids is not None:
                # if there are logprobs in the message, we set the mask to True
                # (i.e. we apply loss for these tokens) and fill the topk
                # with the logprobs

                # (a) first we add the input ids for the message header
                header_ids = self.tokenizer.apply_chat_template(
                    [{"role": message.role, "content": ""}],
                    add_generation_prompt=False,
                    chat_template=TEMPLATE,
                )[
                    :-1
                ]  # we remove the eot token; it is already in the message token ids
                assert len(header_ids) == 4  # start, role, end, newlines
                input_ids.extend(header_ids)

                # (b) next we
                num_ignored_labels = len(header_ids) - 1
                mask.extend([False] * num_ignored_labels)
                topk_tokens.extend([top_tokens_placeholder] * num_ignored_labels)
                topk_logprobs.extend([top_logprobs_placeholder] * num_ignored_labels)

                # (b) then we add the ids for the message content and the eot token
                for token_id, logprobs in zip(message.token_ids, message.logprobs):
                    mask.append(True)
                    input_ids.append(token_id)
                    topk_tokens.append([l.token_id for l in logprobs])
                    topk_logprobs.append([l.logprob for l in logprobs])

                mask.append(False)
                topk_tokens.append(top_tokens_placeholder)
                topk_logprobs.append(top_logprobs_placeholder)

                token_counts[message.role] += len(message.token_ids) + len(header_ids)

            else:
                # if there are no logprobs in the message, we set the mask to False
                # (i.e. we don't apply loss for these tokens) and fill the topk
                # with the placeholder values
                curr_input_ids = apply_chat_template(message)

                mask.extend([False] * len(curr_input_ids))
                topk_tokens.extend([top_tokens_placeholder] * len(curr_input_ids))
                topk_logprobs.extend([top_logprobs_placeholder] * len(curr_input_ids))
                input_ids.extend(curr_input_ids)

                token_counts[message.role] += len(curr_input_ids)

        element = CapsuleDatasetElementLogitLabels(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            topk_tokens=torch.tensor(topk_tokens, dtype=torch.long),
            topk_logprobs=torch.tensor(topk_logprobs, dtype=torch.float),
            mask=torch.tensor(mask, dtype=torch.bool),
            metadata=convo.metadata,
            token_counts=TokenCounts(
                num_system_and_user_tokens=token_counts["user"]
                + token_counts["system"],
                num_assistant_tokens=token_counts["assistant"],
            ),
        )

        assert len(element.input_ids) == len(element.mask)
        assert (
            element.topk_tokens.shape[0]
            == element.topk_logprobs.shape[0]
            == len(element.input_ids)
        )
        assert (
            element.topk_tokens.shape[1]
            == element.topk_logprobs.shape[1]
            == self.config.top_k_logits
        )
        assert element.topk_tokens.shape[0] == element.input_ids.shape[0]

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
            list[CapsuleDatasetElementLogitLabels]
            | list[CapsuleDatasetElementTokenLabels]
        ),
    ) -> CapsuleDatasetBatchLogitLabels | CapsuleDatasetBatchTokenLabels:
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
        if isinstance(batch[0], CapsuleDatasetElementTokenLabels):
            # Create tensors to hold the batched data
            labels = torch.full(
                (batch_size, max_len), -100, dtype=torch.long
            )  # -100 is ignored in loss calculation

            mask = torch.zeros_like(input_ids, dtype=torch.bool)

            for i, element in enumerate(batch):
                assert isinstance(element, CapsuleDatasetElementTokenLabels)
                seq_len = element.input_ids.shape[0]
                input_ids[i, :seq_len] = element.input_ids
                labels[i, : element.input_ids.shape[0]] = element.labels
                mask[i, :seq_len] = element.mask

            return CapsuleDatasetBatchTokenLabels(
                input_ids,
                labels,
                metadata=metadata,
                token_counts=token_counts,
                mask=mask,
            )

        try:
            assert isinstance(batch[0], CapsuleDatasetElementLogitLabels)
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
            assert isinstance(element, CapsuleDatasetElementLogitLabels)

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
        return CapsuleDatasetBatchLogitLabels(
            input_ids=input_ids,
            topk_tokens=topk_tokens,
            topk_logprobs=topk_logprobs,
            mask=mask,
            loss_weight=loss_weight,
            metadata=metadata,
            token_counts=token_counts,
        )



@dataclass
class CapsuleGenerateDatasetElement:
    input_ids: torch.Tensor
    prompt: str

    answer: Optional[str]
    metadata: dict[str, Any]
    convo_id: Optional[str] = None
    
    # this is needed for some datasets, like MMLU, where the in context examples
    # are structured as prior messages
    prompt_messages: Optional[List[Dict[str,str]]] = None

class CapsuleGenerateDataset(CapsuleDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        data_sources: list[tuple[str, int | None],]  # path, limit
        is_wandb: bool = False
        label_type: Literal["tokens"] | Literal["logits"]
        top_k_logits: int = 20
        dataset_weights: Optional[list[float]] = None
        user_prompt_prefix: list[str] | None = None
        # system_prompt: str | None = None
        convo_transforms: list[ConvoTransformConfig] | None = None

    def __getitem__(self, index: int) -> CapsuleGenerateDatasetElement:
        convo: ContextConvo = self.data[index]

        for transform in self.convo_transforms:
            convo = transform(convo)

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

        return CapsuleGenerateDatasetElement(
            input_ids=input_ids,
            prompt=prompt,
            answer=convo.messages[1].content,
            convo_id=convo.id,
            metadata={
                "idx": index,
            },
        )


class MultipleChoiceGenerateDataset(Dataset):
    class Config(ObjectConfig):
        _pass_as_config = True

        # path to the json file
        path: str

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config

        self.data = pd.read_json(config.path).to_dict(orient="records")

        self.tokenizer = tokenizer
        logger.info("Datasets loaded")

    def __getitem__(self, index: int) -> CapsuleGenerateDatasetElement:
        convo: ContextConvo = self.data[index]

        prompt = convo.messages[0].content

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=TEMPLATE,
        )

        return CapsuleGenerateDatasetElement(
            input_ids=input_ids,
            prompt=prompt,
            answer=convo.messages[1].content,
            convo_id=convo.id,
            metadata={
                "idx": index,
            },
        )

    def score():
        pass


class RefDist:
    # todo: maybe switch these away from data classes?
    @dataclass
    class Logprob:
        token_id: int
        logprob: float

    @dataclass
    class Token:
        input_token_id: int
        top_logprobs: list["RefDist.Logprob"]

    @dataclass
    class TrainingExample:
        original_dataset_index: int
        tokens: list["RefDist.Token"]

    @dataclass
    class LazyTrainingExample:
        def __init__(self, table: pa.Table, index: int):
            self._table = table
            self._index = index

            # self._row = None
            assert set(table.schema.names) == {"original_dataset_index", "tokens"}

        def __getitem__(self, name):
            assert name in {"original_dataset_index", "tokens"}
            return self._table.column(name)[self._index].as_py()

        # def materialize(self):
        #     if self._row is None:
        #         self._row = {
        #             name: self._table.column(j)[self._index].as_py()
        #             for j, name in enumerate(self._table.schema.names)
        #         }

        #     return self._row


class CapsuleDatasetWithRefDist(CapsuleDataset):
    class Config(CapsuleDataset.Config):
        _pass_as_config = True
        data_sources: list[tuple[str, int | None]]  # type: ignore
        context: BaseContextConfig

        is_wandb: bool = False

        user_prompt_prefix: list[str] | None = None

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        assert config.label_type == "logits"

        self.config = config

        self.data = []
        self.datasets = []

        self.context = self.config.context.instantiate()

        for source, limit in config.data_sources:
            dataset_dir = (
                wandb.get_artifact_dir(source) / "dataset"
                if config.is_wandb
                else Path(source)
            )

            if not dataset_dir.exists() and config.is_wandb:
                wandb.download_artifact(source)

            with pa.memory_map(
                str((dataset_dir / "final_output.feather").absolute()), "r"
            ) as pa_source:
                reader = pa.ipc.open_file(pa_source)
                table = reader.read_all()

            logger.info(f"Loaded dataset {source} with {table.shape[0]} elements")

            to_add = [
                RefDist.LazyTrainingExample(table, index)
                for index in range(table.shape[0])
            ]

            if limit is not None:
                to_add = to_add[:limit]

            self.data += to_add

        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> CapsuleDatasetElementLogitLabels:
        row = self.data[index]
        assert self.config.label_type == "logits"

        token_data = []
        for token in row["tokens"]:
            token_data.append(
                TokenData(
                    token_id=token["input_token_id"],
                    top_tokens=[
                        top_logprob["token_id"] for top_logprob in token["top_logprobs"]
                    ],
                    top_logprobs=[
                        top_logprob["logprob"] for top_logprob in token["top_logprobs"]
                    ],
                    # TODO: make it so we can pick which part of the loss we apply the mask too
                    apply_loss=True,
                )
            )

        element = CapsuleDatasetElementLogitLabels(
            input_ids=torch.tensor([token.token_id for token in token_data]),
            topk_tokens=torch.tensor([token.top_tokens for token in token_data]),
            topk_logprobs=torch.tensor([token.top_logprobs for token in token_data]),
            mask=torch.tensor([token.apply_loss for token in token_data]),
            metadata=[],
            token_counts=TokenCounts(),
        )

        assert len(element.input_ids) == len(element.mask)
        assert (
            element.topk_tokens.shape[0]
            == element.topk_logprobs.shape[0]
            == len(element.input_ids)
        )
        assert (
            element.topk_tokens.shape[1]
            == element.topk_logprobs.shape[1]
            == self.config.top_k_logits
        )
        assert element.topk_tokens.shape[0] == element.input_ids.shape[0]

        return element


class CapsuleDatasetLatest(CapsuleDataset):
    class Config(CapsuleDataset.Config):
        _pass_as_config = True
        data_sources: list[tuple[str, int | None]]  # type: ignore
        max_sequence_length: int | None = None

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):

        self.config = config

        self.data = []
        self.datasets = []
        context = None

        for source, limit in config.data_sources:
            if source.endswith(".pkl"):
                pkl_path = source
                if not Path(pkl_path).exists():
                    modal_pkl_path = os.path.join("/root/capsules-datasets", os.path.relpath(source, "/"))
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
    ) -> CapsuleDatasetElementTokenLabels:
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
        return CapsuleDatasetElementTokenLabels(
            input_ids=input_ids,
            labels=token_labels,
            metadata=[],
            token_counts=token_counts,
            mask=mask,
        )

    def __getitem__(self, index: int) -> CapsuleDatasetElementLogitLabels:
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

        element = CapsuleDatasetElementLogitLabels(
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


class CapsuleDatasetBatchSampler(BatchSampler):
    def __init__(
        self, 
        sampler: Sampler,
        dataset: CapsuleDatasetLatest,
        batch_size: int, 
        shuffle: bool = True,
    ):
        self.batches = [] 
        assert shuffle
        
        # get lengths of each element 
        lengths = []
        for idx in sampler:
            elem: CapsuleDatasetElementLogitLabels = dataset[idx]
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


@dataclass
class CapsuleDatasetOnlineElement:
    context_ids: list[int]
    input_ids: list[int]


USER_HEADER = [128006, 882, 128007, 271]


@dataclass
class CapsuleDatasetOnlineBatch:
    input_ids_with_context: torch.Tensor
    input_ids: torch.Tensor

    context_start_end: list[tuple[int, int]]
    input_ids_mask: torch.Tensor
    

    token_counts: list[TokenCounts]

    def assert_shapes(self):
        assert self.input_ids.shape == self.input_ids_mask.shape
        assert len(self.input_ids.shape) == 2
        assert self.input_ids.shape[0] == self.context_start_end


class CapsuleDatasetOnline(CapsuleDataset):
    class Config(CapsuleDataset.Config):
        _pass_as_config = True
        data_sources: list[tuple[str, int | None]]  # type: ignore
        max_sequence_length: int | None = None
        include_header: bool = False
        label_type:  Literal["online"] = "online"
    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        assert config.label_type == "online"

        self.config = config

        self.data = []
        self.datasets = []
        context = None

        for source, limit in config.data_sources:
            if source.endswith(".pkl"):
                pkl_path = source
                if not Path(pkl_path).exists():
                    modal_pkl_path = os.path.join("/root/capsules-datasets", os.path.relpath(source, "/"))
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

    def __getitem__(self, index: int) -> CapsuleDatasetElementLogitLabels:
        row: TrainingExample = self.data[index]
        assert isinstance(row, TrainingExample)
        
        context_ids = self.tokenizer.encode(row.context, add_special_tokens=False)

        if self.config.include_header:
            sys_ids = self.tokenizer.apply_chat_template([dict(role='user', content='')])[:-5]
            context_ids = sys_ids + context_ids

        element = CapsuleDatasetOnlineElement(
                context_ids=context_ids,
                input_ids=row.token_ids.tolist(),
        )   
        return element


    def collate(
        self,
        batch: list[CapsuleDatasetOnlineElement],
    ) -> CapsuleDatasetOnlineBatch:
        max_len_contexts = max([len(x.context_ids) + len(x.input_ids) for x in batch])
        max_len_inputs = max([len(x.input_ids) + len(USER_HEADER) for x in batch])

        input_ids_with_context = torch.zeros(
            (len(batch), max_len_contexts), dtype=torch.long
        )
        input_ids = torch.zeros((len(batch), max_len_inputs), dtype=torch.long)
        mask = torch.zeros((len(batch), max_len_inputs), dtype=torch.long)

        token_counts = []

        context_start_end = []
        for i, element in enumerate(batch):

            input_ids_with_context_tensor = torch.tensor(
                element.context_ids + element.input_ids
            )

            input_ids_with_context[i, : len(input_ids_with_context_tensor)] = (
                input_ids_with_context_tensor
            )
            token_counts.append(
                TokenCounts(num_assistant_tokens=len(element.input_ids))
            )

            input_ids[i, :  len(USER_HEADER) + len(element.input_ids)] = torch.tensor(
                USER_HEADER + element.input_ids
            )
            mask[i, : len(USER_HEADER) + len(element.input_ids)] = 1
            context_start_end.append(
                (
                    len(element.context_ids),
                    len(element.context_ids) + len(element.input_ids),
                )
            )
        return CapsuleDatasetOnlineBatch(
            input_ids_with_context=input_ids_with_context,
            input_ids=input_ids,
            context_start_end=context_start_end,
            input_ids_mask=mask,
            token_counts=token_counts,
        )
