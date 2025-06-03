from __future__ import annotations
import random
import time
from typing import List
import uuid
from collections import deque
from dataclasses import dataclass, field
import random
import threading

from capsules.clients.tokasaurus_batch import (
    CapsulesConvoWithLogprobs,
    TokasaurusBatchClient,
)
from capsules.data.mtob import load_test_ek, load_test_ke, load_train_examples
from capsules.generate.generators.base import (
    ContextConvoGenerator,
    get_subcontext,
    responses_and_chats_to_training_examples,
)
from capsules.generate.structs import Context, TrainingExample
from capsules.generate.subcontext import SubcontextGenerator

from capsules.utils import get_logger
from capsules.generate.structs import Context

logger = get_logger(__name__)


class TestExamples(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        answer_client: TokasaurusBatchClient.Config
        answer_temperature: float = 0.0
        answer_max_completion_tokens: int = 384
        # The system prompt for the answer generator. It can contain
        # variables for `subcontext`.
        answer_system_prompt_template: str = "{subcontext}"
        # This will be used to format the answer prompt. It must contain
        # the {question} variable.

        num_top_logprobs: int = 20

        # This will be used to format the question. It can contain
        # the {content} variable.
        lines_per_passage: int = 10

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config

        self.answer_client = config.answer_client.instantiate()

        self.samples_by_subcontext = {}

        self.context_str = "\n".join(
            [section.content for section in self.context.sections]
        )

        self.requests = [
            (
                f"Please translate the following passage from Kalamang to English: {train_example["original"]}",
                train_example["ground_truth"],
            )
            for train_example in load_test_ke()
        ] + [
            (
                f"Please translate the following passage from English to Kalamang: {train_example["original"]}",
                train_example["ground_truth"],
            )
            for train_example in load_test_ek()
        ]

    def sample_convos(
        self, batch_idx: int, num_convos: int, total_batches: int
    ) -> list[TrainingExample]:

        routing_tag = str(uuid.uuid4())
        # (1) sample a subcontext to provide to the question generators
        # --- begin get subcontext ---
        t0 = time.time()
        assert total_batches == 1, "Only one batch is supported"

        logger.info(f"Time taken to get passage: {time.time() - t0} seconds")
        # --- end get subcontext ---

        # (2) sample questions
        # --- begin sample questions ---
        t0 = time.time()
        questions = self.requests
        logger.info(f"Time taken to sample questions: {time.time() - t0} seconds")
        # --- end sample questions ---

        # (3) Sample answers
        # --- begin sample answers ---
        t0 = time.time()

        answer_chats = [
            [
                {
                    "role": "system",
                    "content": self.config.answer_system_prompt_template.format(
                        context=self.context_str,
                        title=self.context.title,
                    ),
                },
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "user",
                    "content": assistant,
                },
            ]
            for (question, answer) in questions
        ]

        convos: List[CapsulesConvoWithLogprobs] = self.answer_client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.answer_temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.answer_max_completion_tokens,
            routing_tag=routing_tag,
        )
        logger.info(f"Time taken to sample answers: {time.time() - t0} seconds")
        # --- end sample answers ---

        # (4) Construct convos
        # --- begin construct convos ---
        t0 = time.time()
        examples = responses_and_chats_to_training_examples(convos, answer_chats)

        logger.info(
            f"Time taken to construct {len(convos)} convos: {time.time() - t0} seconds"
        )

        # --- end construct convos ---
        return examples
