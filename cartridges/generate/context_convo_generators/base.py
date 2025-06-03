import abc
from dataclasses import dataclass
import random
from typing import Any, Dict, Optional
import uuid

from capsules.clients.base import Client, Sample, ClientConfig

from capsules.generate.structs import Context, ContextConvo, Message
from pydrantic import ObjectConfig

"""
I think the generic interface is 

(document) -> (question)

this can be implemented under the hood as 

(document) -> (model) -> question

or (document) -> (chunk) -> question
"""


class ContextConvoGenerator(abc.ABC):

    class Config(ObjectConfig):
        _pass_as_config: bool = True

    def __init__(self, config: Config, context: Context):
        self.config = config
        self.context = context

    @abc.abstractmethod
    def sample_convos(self, start_idx: int, end_idx: int) -> list[ContextConvo]:
        raise NotImplementedError()


MetadataT = dict


@dataclass
class QuestionData:
    question: str
    sample: Optional[Sample]
    metadata: MetadataT
    chunk: Optional[str]=None


class QuestionSystemPromptGenerator(abc.ABC):

    class Config(ObjectConfig):
        _pass_as_config: bool = True

    def __init__(self, config: Config):
        self.config = config

    @abc.abstractmethod
    def __call__(self, context: Context, metadata: MetadataT) -> str:
        raise NotImplementedError()


class AnswerSystemPromptGenerator(abc.ABC):

    class Config(ObjectConfig):
        _pass_as_config: bool = True

    def __init__(self, config: Config):
        self.config = config

    @abc.abstractmethod
    def __call__(self, context: Context, question_data: QuestionData) -> str:
        raise NotImplementedError()


class ConvoGeneratorWithLLMAnswer(ContextConvoGenerator, abc.ABC):

    class Config(ContextConvoGenerator.Config):
        answer_client: ClientConfig
        answer_temperature: float = 0.0
        answer_max_completion_tokens: int = 512

        answer_system_prompt_generator: AnswerSystemPromptGenerator.Config
        
        # This will be used to format the answer prompt. It must contain
        # the {question} variable. If not provided, the question without any
        # formatting will be used.
        answer_prompt_template: str = "{question}"
        num_top_logprobs: int = 20

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config

        self.answer_client: Client = config.answer_client.instantiate()
        self.answer_system_prompt_generator: AnswerSystemPromptGenerator = (
            config.answer_system_prompt_generator.instantiate()
        )

    @abc.abstractmethod
    def get_questions(self, num_samples: int) -> list[QuestionData]:
        raise NotImplementedError()

    def sample_convos(
        self,
        start_idx: int,
        end_idx: int,
    ) -> list[ContextConvo]:
        # (1) Sample questions
        num_convos = end_idx - start_idx
        assert num_convos > 0

        questions = self.get_questions(num_convos)
        try:
            assert len(questions) == num_convos
        except:
            breakpoint()

        # (2) Sample answers
        answer_chats = [
            [
                {
                    "role": "system",
                    "content": self.answer_system_prompt_generator(self.context, question_data),
                },
                {
                    "role": "user",
                    "content": self.config.answer_prompt_template.format(question=question_data.question),
                },
            ]
            for question_data in questions
        ]
        # breakpoint() # self.config.answer_prompt_template.format(question=questions[0].question)
        answer_samples = self.answer_client.chat(
            chats=answer_chats,
            temperature=self.config.answer_temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.answer_max_completion_tokens,
        ).samples

        # (3) Construct convos
        convos = [
            ContextConvo(
                messages=[
                    Message(
                        content=question.question,
                        sample=question.sample,
                        role="user",
                    ),
                    Message(
                        content=answer_sample.text,
                        role="assistant",
                        sample=answer_sample,
                    ),
                ],
                type=self.__class__.__name__,
                metadata=question.metadata,
                id=str(uuid.uuid4()),  # SE(03/10): Added this to make sure every sample has a unique id
            )
            for question, answer_sample in zip(questions, answer_samples, strict=True)
        ]

        return convos


class ConvoGeneratorWithLLMQuestionAndAnswer(ConvoGeneratorWithLLMAnswer, abc.ABC):

    class Config(ConvoGeneratorWithLLMAnswer.Config):
        question_client: ClientConfig
        question_temperature: float = 0.0
        question_max_completion_tokens: int = 512

        question_system_prompt_generator: QuestionSystemPromptGenerator.Config

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config
        self.question_client: Client = config.question_client.instantiate()

        self.question_system_prompt_generator: QuestionSystemPromptGenerator = (
            config.question_system_prompt_generator.instantiate()
        )

    @abc.abstractmethod
    def sample_generate_question_prompt(self) -> tuple[str, MetadataT]:
        raise NotImplementedError()

    def get_questions(self, num_samples: int) -> list[QuestionData]:
        question_prompt_metadata_pair = [
            self.sample_generate_question_prompt() for _ in range(num_samples)
        ]

        chats=[
            [
                {
                    "role": "system",
                    "content": self.question_system_prompt_generator(
                        self.context,
                        metadata,
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            for (prompt, metadata, _) in question_prompt_metadata_pair
        ]
        

        response = self.question_client.chat(
            chats=chats,
            temperature=self.config.question_temperature,
            max_completion_tokens=self.config.question_max_completion_tokens,
        )

        questions = [
            QuestionData(
                question=self._extract_question(sample),
                sample=sample,
                metadata=metadata,
                chunk=chunk,

            )
            for sample, (_, metadata, chunk) in zip(
                response.samples, question_prompt_metadata_pair, strict=True
            )
        ]

        return questions

    def _extract_question(self, sample: Sample) -> str:
        return sample.text
    

    
