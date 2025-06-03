import abc
from collections import deque
from dataclasses import dataclass, field
import random
import threading
from capsules.generate.context_convo_generators.base import (
    ConvoGeneratorWithLLMAnswer,
    QuestionData,
)

from capsules.generate.chunk import Chunker
from capsules.generate.structs import Context


class ConvoGeneratorWithLLMAnswerUnbatchedQuestions(ConvoGeneratorWithLLMAnswer):

    @abc.abstractmethod
    def get_question(self) -> QuestionData:
        raise NotImplementedError()

    def get_questions(self, num_samples) -> list[QuestionData]:
        return [self.get_question() for _ in range(num_samples)]


class NextSectionPrediction(ConvoGeneratorWithLLMAnswerUnbatchedQuestions):
    class Config(ConvoGeneratorWithLLMAnswer.Config):
        before_chunker: Chunker.Config

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.before_chunker: Chunker = config.before_chunker.instantiate()

    def get_question(self) -> QuestionData:
        chunk = self.before_chunker(self.context)

        question = f"""Here is a section of the document:
<section>
{chunk.chunk}
</section>

What comes after this in the document? Be specific, quoting the document directly.
"""

        return QuestionData(
            question=question,
            metadata={"chunk_metadata": chunk.metadata},
            sample=None,
        )


class PreviousSectionPrediction(ConvoGeneratorWithLLMAnswerUnbatchedQuestions):
    class Config(ConvoGeneratorWithLLMAnswer.Config):
        after_chunker: Chunker.Config

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.after_chunker: Chunker = config.after_chunker.instantiate()

    def get_question(self) -> QuestionData:
        chunk = self.after_chunker(self.context)

        question = f"""Here is a section of the document:
<section>
{chunk.chunk}
</section>

What content came before this document? Be specific, quoting the document directly.
"""

        return QuestionData(
            question=question,
            metadata={"chunk_metadata": chunk.metadata},
            sample=None,
        )


class BetweenSectionPrediction(ConvoGeneratorWithLLMAnswerUnbatchedQuestions):
    class Config(ConvoGeneratorWithLLMAnswer.Config):
        section_1_chunker: Chunker.Config
        section_2_chunker: Chunker.Config

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.section_1_chunker: Chunker = config.section_1_chunker.instantiate()
        self.section_2_chunker: Chunker = config.section_1_chunker.instantiate()


    def get_question(self) -> QuestionData:
        chunk_1 = self.section_1_chunker(self.context)
        chunk_2 = self.section_2_chunker(self.context)

        question = f"""Here are two sections of the document:

<section>
{chunk_1.chunk}
</section>

<section>
{chunk_2.chunk}
</section>

Which section is first? What content is between these two sections?
Be specific, quoting the document directly.
"""
        return QuestionData(
            question=question,
            metadata={
                "chunk_1_metadata": chunk_1.metadata,
                "chunk_2_metadata": chunk_2.metadata,
            },
            sample=None,
        )

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
                    end = current_index + random.randint(self.chunk_size_range[0], self.chunk_size_range[1])
                    assert end > current_index

                    sections.append(SectionInfo(current_index, end))
                    current_index = end

                random.shuffle(sections)
                self.sections = deque(sections)

            info = self.sections.pop()
            return self.content[info.start:info.end], info

            


class FairNextSectionPrediction(ConvoGeneratorWithLLMAnswerUnbatchedQuestions):
    class Config(ConvoGeneratorWithLLMAnswer.Config):
        chunk_size_range: tuple[int, int]

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        assert 0 < config.chunk_size_range[0] <= config.chunk_size_range[1]

        self.section_sampler = FairSectionSampler(chunk_size_range=config.chunk_size_range, content=context.to_string())

    def get_question(self) -> QuestionData:
        content, section_info = self.section_sampler.get_section()

        question = f"""Here is a some content from of the document titled {self.context.title}:
<content>
{content}
</content>

What comes after this in the document? Be specific, quoting the document directly.
"""

        return QuestionData(
            question=question,
            metadata={"section_metadata": section_info.__dict__},
            sample=None,
        )

class FairNextSectionPredictionTrans(ConvoGeneratorWithLLMAnswerUnbatchedQuestions):
    class Config(ConvoGeneratorWithLLMAnswer.Config):
        chunk_size_range: tuple[int, int]
        lang: str

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        assert 0 < config.chunk_size_range[0] <= config.chunk_size_range[1]

        self.section_sampler = FairSectionSampler(chunk_size_range=config.chunk_size_range, content=context.to_string())

    def get_question(self) -> QuestionData:
        content, section_info = self.section_sampler.get_section()

        question = f"""Here is a some content from of the document titled {self.context.title}:
<content>
{content}
</content>

What comes after this in the document? Be specific, and express your answer translated into {self.config.lang}.
"""

        return QuestionData(
            question=question,
            metadata={"section_metadata": section_info.__dict__},
            sample=None,
        )

class FairNextSectionPredictionAlsoTrans(ConvoGeneratorWithLLMAnswerUnbatchedQuestions):
    class Config(ConvoGeneratorWithLLMAnswer.Config):
        chunk_size_range: tuple[int, int]
        lang: str

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        assert 0 < config.chunk_size_range[0] <= config.chunk_size_range[1]

        self.section_sampler = FairSectionSampler(chunk_size_range=config.chunk_size_range, content=context.to_string())

    def get_question(self) -> QuestionData:
        content, section_info = self.section_sampler.get_section()

        question = f"""Here is a some content from of the document titled {self.context.title}:
<content>
{content}
</content>

What comes after this in the document? Be specific, quoting the document directly. After your answer, repeat the same paragraph and translate it three times into French, Chinese, and simple English.
"""

        return QuestionData(
            question=question,
            metadata={"section_metadata": section_info.__dict__},
            sample=None,
        )


class FairPreviousSectionPrediction(ConvoGeneratorWithLLMAnswerUnbatchedQuestions):
    class Config(ConvoGeneratorWithLLMAnswer.Config):
        chunk_size_range: tuple[int, int]

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        assert 0 < config.chunk_size_range[0] <= config.chunk_size_range[1]

        self.section_sampler = FairSectionSampler(chunk_size_range=config.chunk_size_range, content=context.to_string())

    def get_question(self) -> QuestionData:
        content, section_info = self.section_sampler.get_section()

        question = f"""Here is a some content from of the document titled {self.context.title}:
<content>
{content}
</content>

What comes before this in the document? Be specific, quoting the document directly.
"""

        return QuestionData(
            question=question,
            metadata={"section_metadata": section_info.__dict__},
            sample=None,
        )
    


class FairSectionLocator(ConvoGeneratorWithLLMAnswerUnbatchedQuestions):
    class Config(ConvoGeneratorWithLLMAnswer.Config):
        chunk_size_range: tuple[int, int]

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        assert 0 < config.chunk_size_range[0] <= config.chunk_size_range[1]

        self.section_sampler = FairSectionSampler(chunk_size_range=config.chunk_size_range, content=context.to_string())

    def get_question(self) -> QuestionData:
        content, section_info = self.section_sampler.get_section()

        question = f"""Here is a passage from a document.
<passage>
{content}
</passage>
     

Please do the following three things:
1. Tell me what document is this from and where is it located in the document.
2. Provide context on what the surrounding text is about?
3. Restate the context before the passage, the passage, and the context after the passage verbatim from the document. The content between <passage> and </passage> tags should be a subset of what you output.
"""

        return QuestionData(
            question=question,
            metadata={"section_metadata": section_info.__dict__},
            sample=None,
        )
    

