from __future__ import annotations

import time
import uuid


from capsules.clients.base import ClientConfig
from capsules.clients.tokasaurus_batch import CapsulesConvoWithLogprobs
from capsules.generate.chunk import Chunker
from capsules.generate.generators.base import (
    ContextConvoGenerator,
    get_subcontext,
    responses_and_chats_to_training_examples,
)
from capsules.generate.structs import (
    Context,
    Document,
    Section,
    TrainingExample,
)
from capsules.generate.subcontext import SubcontextGenerator
from capsules.utils import get_logger

logger = get_logger(__name__)


class SimpleQuestionFromChunk(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        answer_client: ClientConfig
        answer_temperature: float = 0.0
        answer_max_completion_tokens: int = 512
        # The system prompt for the answer generator. It can contain
        # variables for `subcontext`.
        answer_system_prompt_template: str = "{subcontext}"
        # This will be used to format the answer prompt. It must contain
        # the {question} variable.
        answer_prompt_template: str = "{question}"

        num_top_logprobs: int = 20

        question_client: ClientConfig
        question_temperature: float = 0.0
        question_max_completion_tokens: int = 512
        # The system prompt for the question generator. It can contain
        # variables for `subcontext`.
        question_system_prompt_template: str = "{subcontext}"
        # This will be used to format the question prompt. It can contain
        # the {chunk} variable.
        question_prompt_template: str = (
            "Please generate a challenging question about the following excerpt from a document. \n\n{chunk}"
        )

        # We use the chunker when generating questions. Each question get's a different
        # chunk of the subcontext to improve coverage.
        chunker: Chunker.Config
        subcontext_generator: SubcontextGenerator.Config

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config

        self.chunker = config.chunker.instantiate()

        self.question_client = config.question_client.instantiate()
        self.answer_client = config.answer_client.instantiate()

        self.subcontexts = config.subcontext_generator.instantiate(
            context=context,
        )

    def sample_convos(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
    ) -> list[TrainingExample]:
        routing_tag = str(uuid.uuid4())
        # (1) sample a subcontext to provide to the question generators
        # --- begin get subcontext ---
        t0 = time.time()
        subcontext = get_subcontext(batch_idx, total_batches, self.subcontexts)
        logger.info(f"Time taken to get subcontext: {time.time() - t0} seconds")
        # --- end get subcontext ---

        # (2) sample questions
        # --- begin sample questions ---
        t0 = time.time()
        question_prompts = [
            self._sample_generate_question_prompt(subcontext) for _ in range(num_convos)
        ]

        question_chats = [
            [
                {
                    "role": "system",
                    "content": self.config.question_system_prompt_template.format(
                        subcontext=subcontext,
                        title=self.context.title,
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            for prompt in question_prompts
        ]

        question_responses: list[CapsulesConvoWithLogprobs] = (
            self.question_client.chat_with_logprobs(
                chats=question_chats,
                temperature=self.config.question_temperature,
                max_completion_tokens=self.config.question_max_completion_tokens,
                routing_tag=routing_tag,
            )
        )

        questions = [response.assistant_text for response in question_responses]
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

        convos: list[CapsulesConvoWithLogprobs] = self.answer_client.chat_with_logprobs(
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

    def _sample_generate_question_prompt(self, subcontext: str):
        section = Section(desc=self.context.title, path="", content=subcontext)
        if not len(section.content):
            breakpoint()

        chunk = self.chunker(
            # Hack
            Context(
                title=self.context.title,
                sections=[section],
            )
        )

        return self.config.question_prompt_template.format(chunk=chunk)
