from __future__ import annotations
import random
import time
from typing import List
import uuid
from collections import deque
from dataclasses import dataclass, field
import random
import numpy as np

from capsules.clients.tokasaurus_batch import (
    CapsulesConvoWithLogprobs,
    TokasaurusBatchClient,
)
from capsules.data.mtob import load_test_ek, load_test_ke, load_train_examples
from capsules.datasets import TEMPLATE, msg
from capsules.generate.generators.base import (
    ContextConvoGenerator,
    get_subcontext,
    responses_and_chats_to_training_examples,
)
from capsules.generate.structs import Context, TrainingExample
from capsules.generate.subcontext import SubcontextGenerator

from capsules.tasks.mtob import prompt
from capsules.utils import get_logger
from capsules.generate.structs import Context

from transformers import AutoTokenizer

logger = get_logger(__name__)


class TestExamplesMemorize(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):
        tokenizer_name: str

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config

        self.examples = [
            (
                prompt(train_example["original"], "Kalamang", "English"),
                train_example["ground_truth"],
            )
            for train_example in load_test_ke()
        ] + [
            (
                prompt(train_example["original"], "English", "Kalamang"),
                train_example["ground_truth"],
            )
            for train_example in load_test_ek()
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name,
        )

    def sample_convos(
        self, batch_idx: int, num_convos: int, total_batches: int
    ) -> list[TrainingExample]:
        examples = []
        for (question, answer) in self.examples:
            messages = [
                msg(role="user", content=question),
                msg(role="assistant", content=answer),
            ]

            tokens = self.tokenizer.apply_chat_template(
                messages,
                chat_template=TEMPLATE,
            )
            assert isinstance(tokens, list)

            examples.append(
                TrainingExample(
                    messages=[TrainingExample.Message(**message) for message in messages],
                    top_logprob_ids=np.array([[token] for token in tokens[1:]]),
                    top_logprob_logprobs=np.array([[0.0] for token in tokens[1:]]).astype(np.float32),  # We can convert to float32 to save space in the file
                    token_ids=np.array(tokens),
                    num_output_tokens=0,
                    type="todo",
                    metadata={},
                )
            )

        return examples
