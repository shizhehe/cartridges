import threading
from capsules.generate.context_convo_generators.base import (
    ConvoGeneratorWithLLMAnswer,
    QuestionData,
)
from capsules.generate.structs import Context, ContextConvoDataset


class GenerateAnswersToExistingQuestions(ConvoGeneratorWithLLMAnswer):
    class Config(ConvoGeneratorWithLLMAnswer.Config):
        directory_or_artifact: str
        is_wandb: bool

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)

        self.lock = threading.Lock()
        self.index = 0

        self.dataset = ContextConvoDataset.load(
            directory_or_artifact=config.directory_or_artifact,
            is_wandb=config.is_wandb,
        )

    def get_questions(self, num_samples: int) -> list[QuestionData]:
        dataset_indexes = []

        with self.lock:
            for sample_index in range(num_samples):
                if self.index >= len(self.dataset.rows):
                    raise ValueError("Asking to generate too many questions")

                dataset_indexes.append(self.index)
                self.index += 1

        questions = []

        for idx in dataset_indexes:
            row = self.dataset.rows[idx]

            assert row.messages[0].role == "user"
            questions.append(
                QuestionData(
                    question=row.messages[0].content,
                    sample=None,
                    metadata=row.metadata,
                )
            )

        return questions
