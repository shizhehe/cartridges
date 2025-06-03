import abc
import json
from pathlib import Path
import random
from capsules.generate.context_convo_generators.base import (
    ConvoGeneratorWithLLMAnswer,
    QuestionData,
)
from capsules.generate.structs import Context


class ConvoGeneratorWithLLMAnswerQuestionsFromFile(ConvoGeneratorWithLLMAnswer):

    class Config(ConvoGeneratorWithLLMAnswer.Config):
        file: Path

    candidate_questions: list[str]
    q_index: int

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)

        self.candidate_questions = json.load(config.file.open())
        assert isinstance(self.candidate_questions, list)
        for q in self.candidate_questions:
            assert isinstance(q, str)

        self.q_index = 0

    def get_questions(self, num_samples: int) -> list[QuestionData]:
        questions = []
        for _ in range(num_samples):
            questions.append(
                QuestionData(
                    self.candidate_questions[self.q_index],
                    sample=None,
                    metadata={"question_index": self.q_index},
                )
            )
            self.q_index += 1

        return questions
