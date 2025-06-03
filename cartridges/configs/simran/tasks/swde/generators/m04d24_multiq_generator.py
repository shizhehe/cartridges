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


handcrafted_questions = {
    "release date": "What is the release data of this movie?",
    "quotes": "What are the key quotes from this movie?",
    "recommendations": "What are the listed recommendations based on this movie?",
    "sound mix": "What is the sound mix of this movie?",
    "color": "What is listed as the color attribute in the html?",
    "frequently asked questions": "What are some of the frequently asked questions for this movie?",
    "filming locations": "What are the filming locations for this movie?",
    "title": "What is the title of this movie?",
    "stars": "Who are the stars of this movie?",
    "genres": "What are the listed genre(s) for this movie?",
    "rating": "What is the rating of this movie?",
    "cast": "Who are the cast members of this movie?",
    "trivia": "What are some mentioned trivia for this movie?",
    "topic_entity_name": "What is the topic entity name for this movie?",
    "motion picture rating": "What is the motion picture rating for this movie?",
    "goofs": "What are the listed goofs for this movie?",
    "release date": "What is the release date of this movie?",
    "plot keywords": "What are the listed plot keywords for this movie?",
    "soundtracks": "What are the soundtracks for this movie?",
    "production co": "What are the production companies for this movie?",
    "opening weekend": "What are the opening weekend details for this movie?",
    "budget": "What is the budget for this movie?",
    "connections": "What are some of the listed connections for this movie?",
    "users": "What are the 'users' details for this movie?",
    "runtime" : "What is the runtime of this movie?",
    "critics": "What are the 'critics' details for this movie?",
    "language": "In what language(s) is this movie available?",
    "gross": "What is the gross amount for this movie?",
    "related news": "What are some of the related news listed for this movie?",
    "also known as": "What is this movie 'also known as'?",
    "writers": "Who are the writers of this movie?",
    "storyline": "What is the 'storyline' of this movie?",
    "message boards": "What is listed on the message boards for this movie?",
    "related lists": "What are some of the related lists for this movie?",
    "aspect ratio": "What is the aspect ratio of this movie?",
    "moviemeter": "What is reported for 'moviemeter' in this webpage?",
    "year": "In what year was this movie released?",
    "country": "Which country is this movie from?",
    "director": "Who are the director(s) of this movie?",
    "taglines": "What is the tagline of this movie?",
}

attributes = handcrafted_questions.keys()


class SimpleQuestionFromChunk(ContextConvoGenerator):

    class Config(ContextConvoGenerator.Config):

        answer_client: ClientConfig
        answer_temperature: float = 0.0
        answer_max_completion_tokens: int = 512
        # The system prompt for the answer generator. It can contain variables for `subcontext`.
        answer_system_prompt_template: str = "{subcontext}"
        # This will be used to format the answer prompt. It must contain the {question} variable.
        answer_prompt_template: str = "{question}"
        num_top_logprobs: int = 20
        question_client: ClientConfig
        question_temperature: float = 0.0
        question_max_completion_tokens: int = 512
        # The system prompt for the question generator. It can contain variables for `subcontext`.
        question_system_prompt_template: str = "{subcontext}"
        # This will be used to format the question prompt. It can contain the {chunk} variable.
        question_prompt_template: str = (
            "Please generate a challenging question about the following excerpt from a document. \n\n{chunk}"
        )

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config
        self.question_client = config.question_client.instantiate()
        self.answer_client = config.answer_client.instantiate()
        self.context = context

    def sample_convos(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
    ) -> list[TrainingExample]:
        
        routing_tag = str(uuid.uuid4())

        # (1) sample a subcontext to provide to the question generators
        num_context_sections = len(self.context.sections)
        context_idx = batch_idx % num_context_sections
        subcontext = self.context.sections[context_idx].content

        for attribute in attributes:
            if attribute in subcontext.lower():
                print(f"Found {attribute} in subcontext {context_idx} of {num_context_sections}")
        print("-------"*10)

        # (2) sample questions
        question_prompts = [
            # self._sample_generate_question_prompt(subcontext) 
            self.config.question_prompt_template.format(chunk=subcontext)
            for _ in range(num_convos)
        ]
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

        # import re
        # for new_q in unprocessed_questions:
        #     # Remove the <start> and <end> tags from the question
        #     pattern = r'<question\s*\d+\s*>(.*?)</question\s*\d+\s*>'
        #     new_qs = [q.strip()                       # remove leading / trailing whitespace
        #                 for q in re.findall(pattern,
        #                                     new_q,
        #                                     flags=re.I | re.S)]  # re.I = case-insensitive, re.S = dot matches newlines
        #     questions.extend(new_qs)
        # print(f"Added {len(questions)} questions to the batch")

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

        return examples

