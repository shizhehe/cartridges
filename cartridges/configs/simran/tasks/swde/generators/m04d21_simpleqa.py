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

        self.subcontexts = config.subcontext_generator.instantiate(context=context,)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    def sample_convos(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
    ) -> list[TrainingExample]:
        
        routing_tag = str(uuid.uuid4())

        # (1) sample a subcontext to provide to the question generators
        t0 = time.time()
        subcontext = get_subcontext(batch_idx, total_batches, self.subcontexts)
        logger.info(f"Time taken to get subcontext: {time.time() - t0} seconds")

        # (2) sample questions
        question_prompts = [self._sample_generate_question_prompt(subcontext) for _ in range(num_convos)]
        question_chats = []
        for prompt in question_prompts:
            chat = [
                {
                    "role": "system",
                    "content": self.config.question_system_prompt_template.format(
                        subcontext=subcontext,
                        title=self.context.title,
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            question_chats.append(chat)
        question_responses: list[CapsulesConvoWithLogprobs] = (
            self.question_client.chat_with_logprobs(
                chats=question_chats,
                temperature=self.config.question_temperature,
                max_completion_tokens=self.config.question_max_completion_tokens,
                routing_tag=routing_tag,
            )
        )

        unprocessed_questions = [response.assistant_text for response in question_responses]
        questions = []
        for q in unprocessed_questions:
            # Remove the <start> and <end> tags from the question
            q = q.replace("<start>", "").replace("</end>", "")
            if q.startswith("<"):
                q = q[1:]
            q = q.strip()
            if q.endswith(">"):
                q = q[:-1]
            questions.append(q)

        # (3) Sample answers
        answer_chats = []
        for question in questions:
            chat = [
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
            answer_chats.append(chat)

        

        convos: list[CapsulesConvoWithLogprobs] = self.answer_client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.answer_temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.answer_max_completion_tokens,
            routing_tag=routing_tag,
        )

        # (4) Construct convos
        examples = responses_and_chats_to_training_examples(convos, answer_chats)
        logger.info(
            f"Time taken to construct {len(convos)} convos: {time.time() - t0} seconds"
        )

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

        prompt = self.config.question_prompt_template.format(chunk=chunk.chunk)
        return prompt
