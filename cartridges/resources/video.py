
import os
import random
from typing import List
from .base import Resource


class VideoResource(Resource):

    class Config(Resource.Config):
        path: str = "database/video"
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

        self.prompts = VIDEO_PROMPTS


    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        return self.text, self.prompts


VIDEO_PROMPTS = [
    (
        "You are analyzing the conversation between someone learning about AMD GPUs and AMD experts."
        "Please identify a key insight about AMD GPUs or AMD GEMM kernels that was shared by the experts in the meeting."
        "Please generate a question that can be used to test the user's understanding of the insight."
        "Be sure to include details (ids, concepts, names, titles, dates, etc.) in the question to make it clear what you are asking about."
        "Answer only with the question, do not include any other text."
    ),
    (
        "You are analyzing the conversation between someone learning about AMD GPUs and AMD experts."
        "Please identify a the key topics discussed in the meeting."
        "Please generate a question that can be used to test the user's understanding of one of these key topics."
        "Be sure to include details (ids, concepts, names, titles, dates, etc.) in the question to make it clear what you are asking about."
        "Answer only with the question, do not include any other text."
    )
]



