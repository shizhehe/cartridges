
from capsules.clients.base import Client
from capsules.data.openstax.structs import QuestionAndAnswer


def eval_q_and_a(client: Client, questions_and_answers: list[QuestionAndAnswer]):
    return client.chat(
        [
            [
                {
                    "role": "user",
                    "content": f"Please answer this multiple choice question, giving only your answer: {question.question}",
                },
            ]
            for question in questions_and_answers
        ],
        max_completion_tokens=5,
        temperature=0.0,
    )
