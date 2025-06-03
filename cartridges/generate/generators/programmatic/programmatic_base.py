from __future__ import annotations
import math
import random
import time
from typing import List, Literal
import uuid
from collections import deque
from dataclasses import dataclass, field
import random
import threading
from transformers import AutoTokenizer

from capsules.clients.tokasaurus_batch import (
    CapsulesConvoWithLogprobs,
    TokasaurusBatchClient,
)
from capsules.datasets import (
    ASSISTANT_TOKEN_ID,
    END_HEADER_ID,
    START_HEADER_ID,
    TEMPLATE,
    msg,
)
from capsules.generate.generators.base import (
    ContextConvoGenerator,
    get_subcontext,
)
from capsules.generate.generators.programmatic.text_prediction import (
    next_passage_prediction,
    previous_passage_prediction,
    unshuffle,
    fill_in_the_blank_masked_words,
)
from capsules.generate.structs import Context, TrainingExample
from capsules.generate.subcontext import SubcontextGenerator

from capsules.utils import get_logger
from capsules.generate.structs import Context
import numpy as np

logger = get_logger(__name__)


class ProgrammaticConvoGenerator(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        convo_type: Literal[
            "next_passage", "previous_passage", "between_passage", "unshuffle", "unmask"
        ]

        num_top_logprobs: int = 20
        lines_per_passage: int = 15

        tokenizer_name: str

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,
        )

        if self.config.convo_type == "between_passage":
            assert False
            fn = between_passage_prediction
        elif self.config.convo_type == "next_passage":
            fn = next_passage_prediction
        elif self.config.convo_type == "previous_passage":
            fn = previous_passage_prediction
        elif self.config.convo_type == "unshuffle":
            fn = unshuffle
        elif self.config.convo_type == "unmask":
            fn = fill_in_the_blank_masked_words
        else:
            raise ValueError("Invalid convo type")

        self.fn = fn

    def sample_convos(
        self, batch_idx: int, num_convos: int, total_batches: int
    ) -> list[TrainingExample]:
        chats = self.fn(
            document_title=self.context.title,
            content="\n".join([str(section) for section in self.context.sections]),
            lines_per_passage=self.config.lines_per_passage,
        )
        logger.info(
            f"ProgrammaticConvoGenerator: there are {len(chats)} messages generated programatically"
        )

        examples = []
        for messages in chats:
            tokens = self.tokenizer.apply_chat_template(
                messages,
                chat_template=TEMPLATE,
            )
            token_ids = np.array(tokens)
            assert isinstance(tokens, list)

            header_locations = np.where(token_ids == START_HEADER_ID)[0].tolist()
            assert len(header_locations) == 2
            assert header_locations[0] == 0
            user_start_idx, asst_start_idx = header_locations
            assert isinstance(asst_start_idx, int)

            assert tokens[asst_start_idx] == START_HEADER_ID
            assert tokens[asst_start_idx + 1] == ASSISTANT_TOKEN_ID
            assert tokens[asst_start_idx + 2] == END_HEADER_ID

            start_of_data = asst_start_idx + 3

            mask = np.array(
                [False] * (start_of_data)
                + (
                    [True]
                    *
                    # -1 because last token isn't trained on
                    (len(tokens) - start_of_data - 1)
                )
            )

            top_logprob_ids = []
            top_logprob_logprobs = []

            for token in tokens[1:]:
                top_logprob_ids.append(
                    [token] + [0] * (self.config.num_top_logprobs - 1)
                )
                top_logprob_logprobs.append(
                    [0.0] + [-1000.0] * (self.config.num_top_logprobs - 1)
                )

            examples.append(
                TrainingExample(
                    messages=[
                        TrainingExample.Message(**message) for message in messages
                    ],
                    top_logprob_ids=np.array(top_logprob_ids),
                    top_logprob_logprobs=np.array(top_logprob_logprobs).astype(
                        np.float32
                    ),  # We can convert to float32 to save space in the file
                    token_ids=token_ids,
                    num_output_tokens=0,
                    type="programmatic",
                    metadata={},
                    mask=mask,
                )
            )

        # --- end construct convos ---
        return examples
