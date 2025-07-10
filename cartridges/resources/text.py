
import os
import random
from .base import Resource
from typing import List


class StaticTextResource(Resource):

    class Config(Resource.Config):
        path: str = "database/static_text"
        seed: int = 42
        text: str = ""
        prompts: List[str] = []

    def __init__(self, config: Config):
        self.config = config

        random.seed(self.config.seed)
        files = os.listdir(self.config.path)
        file = random.choice(files)
        with open(f"{self.config.path}/{file}", "r") as f:
            self.text = f.read()

        self.prompts = STATIC_TEXT_PROMPTS

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        return self.text, random.choices(self.prompts, k=batch_size)


STATIC_TEXT_PROMPTS = [
        (
            "Generate a question for an LLM that will test its knowledge of the information in the corpus above. "
            "In your question be sure to include details (ids, names, titles, dates, etc.) that make it clear what you are asking about. "
            "Output only a single question. Do NOT include any other text or explanation other than the question."
        ),
        (
            "Generate a message for an LLM that will test its knowledge of the information in the corpus above."
            "Be sure to include details (ids, names, titles, dates, etc.) in the question so that it can be answered without access to the corpus (i.e. closed-book setting). "
            "Output only a single question. Do NOT include any other text or explanation other than the question."
        ),
        (
            "You are helping to quiz a user about the information in the corpus. "
            "Please generate a question about the subsection of the corpus above. "
            "Be sure to include details (ids, names, titles, dates, etc.) in the question to make it clear what you are asking about. "
            "Answer only with the question, do not include any other text."
        ),
        (
            "You are working to train a language model on the information in the following corpus. "
            "Your primary goal is to think about practical, real-world tasks or applications that someone could achieve using the knowledge contained within this corpus. "
            "Consider how a user might want to apply this information, not just recall it. "
            "After considering potential use cases, your task will be to generate a sample question that reflects one of these downstream applications. "
            "This question/instruction/task should be something a user, who has access to this corpus, might ask when trying to accomplish their specific goal. "
            "Output only a single question. Do NOT include any other text or explanation other than the question."
        ),
        (
            "Please generate a single chat message instructing an LLM to summarize part of the corpus. "
            "Make sure the instruction is very explicit about the section of the corpus that you want to summarize. "
            "Include details (ids, names, titles, dates, etc.) that make it clear what you are asking about. "
        ),
        (
            "Please generate a single chat message instructing an LLM to summarize a section. "
            "Make sure the instruction is explicit about the section that should be summarized and the document it is from."
        ),
]


