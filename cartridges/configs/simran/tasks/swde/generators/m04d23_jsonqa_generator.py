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


@dataclass
class SectionInfo:
    start: int
    end: int


@dataclass
class FairSectionSampler:
    chunk_size_range: tuple[int, int]
    content: str

    sections: deque[SectionInfo] | None = None
    _lock: threading.Lock = field(default_factory=lambda: threading.Lock())

    def get_section(self) -> tuple[str, SectionInfo]:
        with self._lock:
            assert len(self.content)
            if self.sections is None or len(self.sections) == 0:
                # print("Fair section sampler looping")
                sections = []

                current_index = 0
                while current_index < len(self.content):
                    end = current_index + random.randint(
                        self.chunk_size_range[0], self.chunk_size_range[1]
                    )
                    assert end > current_index

                    sections.append(SectionInfo(current_index, end))
                    current_index = end

                random.shuffle(sections)
                self.sections = deque(sections)

            info = self.sections.pop()
            return self.content[info.start : info.end], info


QUESTION_TEMPLATE = """You pre-processing this passage to strip boiler-plate that we never need (ad scaffolding, inline JS, whitespace, etc.). Output the cleaned post-processed text.

<passage>
{content}
</passage>

Output only the cleaned text:
"""


class LocateFairSectionGenerator(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        answer_client: TokasaurusBatchClient.Config
        answer_temperature: float = 0.0
        answer_max_completion_tokens: int = 384
        # The system prompt for the answer generator. It can contain
        # variables for `subcontext`.
        answer_system_prompt_template: str = "{subcontext}"
        # This will be used to format the answer prompt. It must contain
        # the {question} variable.
        answer_prompt_template: str = "{question}"

        num_top_logprobs: int = 20

        # This will be used to format the question. It can contain
        # the {content} variable.
        question_template: str = QUESTION_TEMPLATE

        chunk_size_range: tuple[int, int] = (500, 2000)
        subcontext_generator: SubcontextGenerator.Config

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config

        self.answer_client = config.answer_client.instantiate()

        self.subcontexts = config.subcontext_generator.instantiate(
            context=context,
        )
        self.samples_by_subcontext = {}

    def sample_convos(
        self, batch_idx: int, num_convos: int, total_batches: int
    ) -> list[TrainingExample]:
        routing_tag = str(uuid.uuid4())

        # (1) sample a subcontext to provide to the question generators
        t0 = time.time()
        subcontext = get_subcontext(batch_idx, total_batches, self.subcontexts)
        logger.info(f"Time taken to get subcontext: {time.time() - t0} seconds")

        # (2) sample questions
        t0 = time.time()
        questions = self._get_questions(subcontext, num_convos)
        logger.info(f"Time taken to sample questions: {time.time() - t0} seconds")

        # (3) Sample answers
        t0 = time.time()
        answer_chats = [
            [
                {
                    "role": "system",
                    "content": self.config.answer_system_prompt_template.format(
                        subcontext=subcontext,
                        title=self.context.title,
                    ),
                },
                {
                    "role": "user",
                    "content": self.config.answer_prompt_template.format(
                        question=question
                    ),
                },
            ]
            for question in questions
        ]

        convos: List[CapsulesConvoWithLogprobs] = self.answer_client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.answer_temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.answer_max_completion_tokens,
            routing_tag=routing_tag,
        )
        logger.info(f"Time taken to sample answers: {time.time() - t0} seconds")


        # (4) Construct convos
        t0 = time.time()
        examples = responses_and_chats_to_training_examples(convos, answer_chats)
        logger.info(f"Time taken to construct {len(convos)} convos: {time.time() - t0} seconds")
        return examples

    def _get_questions(
        self,
        subcontext: str,
        num_samples: int,
    ) -> list[str]:
        questions = []

        for _ in range(num_samples):
            content, section_info = self._get_fair_section_sampler(
                subcontext
            ).get_section()
            question = self.config.question_template.format(content=content)
            questions.append(question)
        return questions

    def _get_fair_section_sampler(self, subcontext: str) -> FairSectionSampler:
        assert isinstance(self.config, LocateFairSectionGenerator.Config)
        if subcontext not in self.samples_by_subcontext:
            self.samples_by_subcontext[subcontext] = FairSectionSampler(
                chunk_size_range=self.config.chunk_size_range,
                content=subcontext,
            )

        return self.samples_by_subcontext[subcontext]


