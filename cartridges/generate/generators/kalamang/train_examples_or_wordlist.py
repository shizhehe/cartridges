from __future__ import annotations

import time
from typing import Literal
import uuid


from capsules.clients.base import ClientConfig
from capsules.clients.tokasaurus_batch import CapsulesConvoWithLogprobs
from capsules.data.mtob import (
    load_book_long,
    load_book_medium,
    load_train_examples,
    load_wordlist,
)
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


def question_prompt_parallel_sentence(
    kalamang_sentence,
    english_sentence,
    direction: Literal["Kalamang to English", "English to Kalamang"],
):
    return f"""Please generate a question that's focused on translating from {direction}.

Your should question should be based on the following pair of equivalent Kalamang and English sentences:
English: {english_sentence}
Kalamang: {kalamang_sentence}
It can also include details from the grammar textbook above.

The question should be open-ended and require translating from {direction}.

Respond with the question only, do not include any other text.
"""


def question_prompt_word_definition(
    kalamang_word,
    english_word,
    direction: Literal["Kalamang to English", "English to Kalamang"],
):
    return f"""Please generate a question that's focused on translating from {direction}.

Your question should be based on the following pair of equivalent Kalamang and English words:
English: {english_word}
Kalamang: {kalamang_word}
It can also include details from the grammar textbook above.

The question should be open-ended and require translating from {direction}.

Respond with the question only, do not include any other text.
"""


class ParallelSentencesOrWordlist(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        answer_client: ClientConfig
        answer_temperature: float = 0.0
        answer_max_completion_tokens: int = 512
        num_top_logprobs: int = 20

        question_client: ClientConfig
        question_temperature: float = 0.0
        question_max_completion_tokens: int = 512

        source: str = "training_examples"

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config

        self.question_client = config.question_client.instantiate()
        self.answer_client = config.answer_client.instantiate()

        # book = load_book_long()
        # book = book[book.index("\n") + 1 :]
        book = load_book_medium()
        book = book[book.index("\n") + 1 :]

        self.system_prompt = f"Here is a grammar book explaining how to translate between the language Kalamang and English: \nBEGIN BOOK{book}END BOOK\n\n"

        if config.source == "training_examples":
            train_examples = load_train_examples()

            self.question_prompts = [
                (
                    question_prompt_parallel_sentence(
                        example["translation"],
                        example["original"],
                        direction="Kalamang to English",
                    ),
                    example,
                )
                for example in train_examples
            ] + [
                (
                    question_prompt_parallel_sentence(
                        example["translation"],
                        example["original"],
                        direction="English to Kalamang",
                    ),
                    example,
                )
                for example in train_examples
            ]
        elif config.source == "wordlist":
            wordlist = load_wordlist()
            word_ke = wordlist["ke"]
            word_ek = wordlist["ek"]
            self.question_prompts = [
                (
                    question_prompt_word_definition(
                        kalamang,
                        english,
                        "Kalamang to English",
                    ),
                    (kalamang, english),
                )
                for kalamang, (_, english) in word_ke.items()
            ] + [
                (
                    question_prompt_word_definition(
                        kalamang,
                        english,
                        "English to Kalamang",
                    ),
                    (kalamang, english),
                )
                for english, kalamang in word_ek.items()
            ]

    def sample_convos(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
    ) -> list[TrainingExample]:
        question_chats = []
        routing_tag = str(batch_idx)

        for index in range(num_convos):
            question_prompt_idx = batch_idx * num_convos + index
            question_prompt, question_data = self.question_prompts[question_prompt_idx]

            if self.config.source == "training_examples":
                system_prompt = f"""{self.system_prompt}
    Here's an example of a Kalamang sentence and its English translation:

    Kalamang Sentence: {question_data['translation']}
    Equivalent English Sentence: {question_data['original']}

    """
            else:
                system_prompt = f"""{self.system_prompt}

    Here is a Kalamang word and its English translation:
    Kalamang Word: {question_data[0]}
    English Word: {question_data[1]}
    """

            question_chats.append(
                [
                    dict(role="system", content=system_prompt),
                    dict(role="user", content=question_prompt),
                ]
            )

        t0 = time.time()

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
                source_chat[0],
                dict(role="user", content=question),
            ]
            for question, source_chat in zip(questions, question_chats, strict=True)
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
