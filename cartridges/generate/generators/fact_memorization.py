from __future__ import annotations

import time
import uuid


from capsules.clients.base import ClientConfig
from capsules.clients.tokasaurus_batch import CapsulesConvoWithLogprobs

from capsules.generate.generators.base import (
    ContextConvoGenerator,
    get_subcontext,
    responses_and_chats_to_training_examples,
)
from capsules.generate.structs import (
    Context,
    TrainingExample,
)
from capsules.generate.subcontext import SubcontextGenerator
from capsules.utils import get_logger

logger = get_logger(__name__)


class FactMemorization(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        answer_client: ClientConfig
        answer_temperature: float = 0.0
        answer_max_completion_tokens: int = 1024
        # The system prompt for the answer generator. It can contain
        # variables for `subcontext`.
        answer_system_prompt_template: str = "{subcontext}"
        # This will be used to format the answer prompt. It must contain
        # the {question} variable.
        answer_prompt_template: str = "{question}"

        num_top_logprobs: int = 20

        question_client: ClientConfig
        question_temperature: float = 0.2
        question_max_completion_tokens: int = 1024
        # The system prompt for the question generator. It can contain
        # variables for `subcontext`.
        question_system_prompt_template: str = "{subcontext}"
        # This will be used to format the question prompt. It can contain
        # the {chunk} variable.
        question_prompt_template: str = (
            """Please generate questions based on the following passage.
Please generate one question per piece of specific factual information in this passage.
The answer to the question should be objective based off of the content in the passage and the information in the system prompt.
Use the information in the system prompt to understand the context of this passage.
<passage>
{passage}
</passage>
Each question must be self-contained and independently understandable. The questions will be asked without the passage or system prompt in context.
Include enough information within the question itself so that someone without with the passage or document can understand what is being asked.
The answerer of the question will not have access to passage, so qualify your questions properly: DO NOT USE phrases like "according to the passage" or "as stated in the passage".

Please generate one question per line. Do not generate answers or number the questions."""
        )

        num_lines_per_passage: int = 10

        subcontext_generator: SubcontextGenerator.Config
        microbatch_size: int = 2048

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config

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
        # --- end get subcontext ---

        # (2) sample questions
        # --- begin sample questions ---
        t0 = time.time()

        passages = []

        subcontext_lines = [
            line
            for line in subcontext.splitlines()
            if line.strip() != ""
            and not line.startswith("<section desc=")
            and line != "</section>"
        ]

        for line_index in range(
            0, len(subcontext_lines), self.config.num_lines_per_passage
        ):
            passage = "\n".join(
                subcontext_lines[
                    line_index : line_index + self.config.num_lines_per_passage
                ]
            )
            passages.append(passage)

        question_prompts = [
            self.config.question_prompt_template.format(passage=passage)
            for passage in passages
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

        questions = []

        for question_response in question_responses:
            questions += [
                question
                for question in question_response.assistant_text.splitlines()
                if question.strip() != ""
            ]

        breakpoint()

        logger.info(
            f"Time taken to sample {len(questions)} questions from {len(passages)} passages: {time.time() - t0} seconds"
        )

        # --- end sample questions ---

        # (3) Sample answers
        # --- begin sample answers ---
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

        convos: list[CapsulesConvoWithLogprobs] = []
        for index in range(0, len(answer_chats), self.config.microbatch_size):
            t0 = time.time()
            logger.info(f"starting to sample batch of answers")
            convos += self.answer_client.chat_with_logprobs(
                chats=answer_chats[index : index + self.config.microbatch_size],
                temperature=self.config.answer_temperature,
                top_logprobs=self.config.num_top_logprobs,
                max_completion_tokens=self.config.answer_max_completion_tokens,
                routing_tag=routing_tag,
            )
            logger.info(f"finished to sampling batch of answers: {time.time() - t0} seconds")

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
