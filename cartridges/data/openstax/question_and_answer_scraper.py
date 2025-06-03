from dataclasses import dataclass
import requests
from capsules.data.openstax.structs import QuestionAndAnswer, TextbookMetadata
from bs4 import BeautifulSoup


def get_bs4(url: str) -> BeautifulSoup:
    return BeautifulSoup(requests.get(url).text, "html.parser")


@dataclass
class Question:
    question_number: str
    content: str


def extract_question_number_and_text(problem_element) -> Question:
    problem_number = int(
        problem_element.select_one("a.os-number, span.os-number").text.strip()
    )
    problem_container = problem_element.select_one(".os-problem-container")

    p = problem_container.select_one("p")
    if p is not None:
        # eg business textbook
        question = problem_container.select_one("p").text.strip()
    else:
        # eg pharm textbook
        q_container = problem_container.select_one("div[data-type='question-stem']")
        assert q_container is not None
        question = q_container.text.strip()

    ol_element = problem_container.select_one("ol")
    use_letters = ol_element.get("type").lower() == "a" if ol_element else False

    choices = problem_container.select("ol > li")
    formatted_choices = []

    for i, choice in enumerate(choices):
        if use_letters:
            # Convert index to letter (0->A, 1->B, etc.)
            prefix = chr(65 + i)  # ASCII: 65 is 'A'
        else:
            # Use regular numbers (1, 2, 3...)
            prefix = str(i + 1)

        formatted_choices.append(f"{prefix}. {choice.text.strip()}")

    result = question + "\n" + "\n".join(formatted_choices)
    return Question(problem_number, result)


def select_one(el: BeautifulSoup, selector: str):
    options = el.select(selector)
    if len(options) != 1:
        breakpoint()

    return options[0]


def get_questions(
    metadata: TextbookMetadata, chapter_num: int
) -> list[QuestionAndAnswer]:
    assert len(metadata.review_questions_types) == 1

    question_html = get_bs4(
        f"{metadata.openstax_url}/{chapter_num}-{metadata.review_questions_types[0]}"
    )
    # questions_section = select_one(question_html, ".os-review-questions-container")
    questions = [
        extract_question_number_and_text(el)
        for el in question_html.select("section > div[data-type='problem'], div[data-type='exercise-question'], div[data-formats='multiple-choice']")
    ]

    # weird url, I know
    answer_html = get_bs4(f"{metadata.openstax_url}/chapter-{chapter_num}")
    # answers_section = select_one(answer_html, ".os-solution-container")

    # TODO: clean this up
    if "pharmacology" in metadata.openstax_url:
        labels = [
            el for el in answer_html.select("span.os-title-label") if el.text == "Review Questions"
        ]
        assert len(labels) == 1
        answer_container = labels[0].parent.parent

        section = answer_container.select("div[data-type='question-solution']")

        answers_by_number = {
            int(
                answer_section.select_one("a.os-number, span.os-number").text.strip()
            ): answer_section.select_one("div.os-solution-container").text.strip()
            for answer_section in answer_container.select("div[data-type='question-solution']")
        }
    else:
        answers_by_number = {
            int(
                answer_section.select_one("a.os-number, span.os-number").text.strip()
            ): answer_section.select_one("div.os-solution-container").text.strip()
            for answer_section in answer_html.select("div[data-type='solution']")
        }

    used_answers = set()

    q_and_answers = []
    for question in questions:
        assert question.question_number not in used_answers

        answer = answers_by_number.get(question.question_number, None)
        if answer is not None:
            used_answers.add(question.question_number)

        q_and_answers.append(
            QuestionAndAnswer(
                question=question.content,
                number=f"chap-{chapter_num}-q-{question.question_number}",
                answer=answers_by_number.get(question.question_number, None),
            )
        )

    # right now this assert isn't always true because we don't scrape all the questions
    # assert len(used_answers) == len(answers_by_number)

    return q_and_answers

