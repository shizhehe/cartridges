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



class ReWriteForTranslation(ContextConvoGenerator):
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
        question_template: str
        lines_per_passage: int = 10

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config

        self.answer_client = config.answer_client.instantiate()

        self.samples_by_subcontext = {}

        passages = []
        for section in self.context.sections:
            section_lines = section.content.splitlines()
            for i in range(0, len(section_lines), self.config.lines_per_passage):
                lines = section_lines[i : i + self.config.lines_per_passage]
                if lines:
                    passages.append("\n".join(lines))

        self.passages = passages

        self.context_str = "\n".join(
            [section.content for section in self.context.sections]
        )
        print("")

    def sample_convos(
        self, batch_idx: int, num_convos: int, total_batches: int
    ) -> list[TrainingExample]:

        routing_tag = str(uuid.uuid4())
        # (1) sample a subcontext to provide to the question generators
        # --- begin get subcontext ---
        t0 = time.time()
        assert total_batches >= len(self.passages)

        passage = get_subcontext(batch_idx, total_batches, self.passages)
        logger.info(f"Time taken to get passage: {time.time() - t0} seconds")
        # --- end get subcontext ---

        # (2) sample questions
        # --- begin sample questions ---
        t0 = time.time()
        questions = [
            self.config.question_template.format(
                passage=passage
            )
        ] * num_convos
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
