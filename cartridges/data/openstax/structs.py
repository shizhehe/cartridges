from dataclasses import dataclass
from typing import Literal


# url types in comments
ReviewQuestionType = Literal[
    'multiple-choice', # "1-multiple-choice"
    # 'fill-in-the-blank', # "1-fill-in-the-blank"
    'review-questions', # "1-review-questions"
    'assessment-questions', # "1-review-questions"

]

@dataclass
class TextbookMetadata:
    github_repo: str 
    collection_name: str
    openstax_url: str # https://openstax.org/books/microbiology/pages
    review_questions_types: list[ReviewQuestionType]

@dataclass
class TextbookSection:
    title: str
    content: str

@dataclass
class QuestionAndAnswer:
    number: int
    question: str
    answer: str | None

@dataclass
class TextbookChapter:
    sections: list[TextbookSection]
    questions: list[QuestionAndAnswer]

@dataclass
class ChapterMetadata:
    number: int
    modules_ids: list[str]
    title: str

