from dataclasses import dataclass
import random
from textwrap import dedent
from typing import Literal
import torch

from capsules.datasets import CapsuleDataset, TokenCounts
from capsules.tasks.codehop.code_hop_synth import (
    CodeHopSynthConfig,
    make_code_hop,
    serialize_file,
)


@dataclass
class CapsuleDatasetCodeHopElement:
    context_ids: list[int]
    input_ids: list[int]


USER_HEADER = [128006, 882, 128007, 271]


@dataclass
class CapsuleDatasetCodeHopBatch:
    input_ids_with_context: torch.Tensor
    input_ids: torch.Tensor

    context_start_end: list[tuple[int, int]]
    input_ids_mask: torch.Tensor

    token_counts: list[TokenCounts]

    def assert_shapes(self):
        assert self.input_ids.shape == self.input_ids_mask.shape
        assert len(self.input_ids.shape) == 2
        assert self.input_ids.shape[0] == self.context_start_end


class CodeHopDataset(CapsuleDataset):
    class Config(CapsuleDataset.Config):
        _pass_as_config = True

        code_hop_config: CodeHopSynthConfig
        include_header: bool = False

        data_sources: list[tuple[str, int | None],] = []
        is_wandb: bool = False
        label_type: Literal["tokens"] | Literal["logits"] | Literal["online"] = "online"


    def __init__(self, config: Config, tokenizer):
        self.config = config

        code_hop = make_code_hop(config.code_hop_config)
        self.data = []

        for file in code_hop.files:

            context = USER_HEADER.copy()
            context += tokenizer.encode(
                f"""
Here is a python file named {file.name}:
<{file.name}.py>
{serialize_file(file)}
</{file.name}.py>
""",
                add_special_tokens=False,
            )

            for method in file.methods:
                for input_vocab in code_hop.input_vocab:
                    for output_vocab in code_hop.output_vocab:
                        input_tokens = tokenizer.encode(
                            dedent(f"""\
                                Please tell me the result of running the following python code.
                                Respond with just the output and no other text.

                                ```
                                import {file.name}

                                print({file.name}.{method.name}("{input_vocab}"))
                                ```                            
                                """, 
                            ),
                            add_special_tokens=False

                        )

                        output_tokens = tokenizer.encode(output_vocab, add_special_tokens=False)

                        input_ids = [
                            *input_tokens,
                            128009,
                            128006,
                            78191,
                            128007,
                            271,
                            *output_tokens,
                            128009,
                        ]
                        self.data.append(
                            CapsuleDatasetCodeHopElement(
                                context_ids=context,
                                input_ids=input_ids,
                            )
                        )

        if config.include_header:
            sys_ids = tokenizer.apply_chat_template([dict(role='user', content='')])[:-5]
            for item in self.data:
                item.context_ids = sys_ids + item.context_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(
        self,
        batch: list[CapsuleDatasetCodeHopElement],
    ) -> CapsuleDatasetCodeHopBatch:
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
        return CapsuleDatasetCodeHopBatch(
            input_ids_with_context=input_ids_with_context,
            input_ids=input_ids,
            context_start_end=context_start_end,
            input_ids_mask=mask,
            token_counts=token_counts,
        )
