
import os
import random
from .base import Resource
from typing import List


class ThunderkittensResource(Resource):

    class Config(Resource.Config):
        path: str = "database/thunderkittens"
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

        prompts = THUNDERKITTENS_PROMPTS
        self.prompts = [prompt.format(filename=file) for prompt in prompts]

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        return self.text, random.choices(self.prompts, k=batch_size)


THUNDERKITTENS_PROMPTS = [
    (
        "Here is some information about the ThunderKittens repository filename {filename}. "
        "Please generate a single chat message instructing an LLM to summarize a section. "
        "Make sure the instruction is explicit about the section that should be summarized and the document it is from. "
    ),
    (
        "Here is some information about the ThunderKittens repository filename {filename}. "
        "Please generate a question that can be used to test a ThunderKittens developer's knowledge of the repository. "
        "Make sure the question is clear and concise. "
    ),
    (
        "Here is some information about the ThunderKittens repository filename {filename}. "
        "Please generate a question that can be used to test a ThunderKittens developer's knowledge of the contents of this file. "
        "Make sure the question is clear and concise. "
    ),
    (
        "Here is some information about the ThunderKittens repository filename {filename}. "
        "Please generate a question that can be used to test a ThunderKittens developer's knowledge of data types or functions in this file and the syntax needed to use them. "
        "Make sure the question is clear and concise. "
    ),
]


