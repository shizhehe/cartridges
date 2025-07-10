"""
User interactions with an assistive AI model.
"""

import os
import random
from typing import List
from .base import Resource


class ChatLogResource(Resource):

    class Config(Resource.Config):
        path: str = "database/chatlogs"
        seed: int = 42
        text: str = ""
        prompts: List[str] = []


    def __init__(self, config: Config):
        self.config = config

        random.seed(self.config.seed)
        file_paths = os.listdir(self.config.path)
        self.files = []
        for file_path in file_paths:
            with open(f"{self.config.path}/{file_path}", "r") as f:
                self.files.append(f.read())

        self.prompts = CHATLOG_PROMPTS


    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        text = random.choice(self.files)
        prompts = random.choices(CHATLOG_PROMPTS, k=batch_size)
        return text, prompts


CHATLOG_PROMPTS = [
    (
        "You are analyzing the user's conversation history with a machine learning model assistant."
        "Please determine an instance where the model was helpful."
        "Please generate a single chat message instructing an LLM to explain the key insight that the model provided to the user."
    ),
    (
        "You are analyzing the user's conversation history with a machine learning model assistant."
        "Please determine an instance where the model was not helpful."
        "Please generate a single chat message instructing an LLM to explain what the model did wrong and what it should have done instead."
    ),
    (
        "You are analyzing the user's conversation history with a machine learning model assistant."
        "Identify the most frequently discussed topics in the conversation."
        "Please generate a question that can be used to test the user's understanding of the most frequently discussed topics."
        "Be sure to include details (ids, concepts, names, titles, dates, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    ),
    (
        "You are analyzing the user's conversation history with a machine learning model assistant."
        "Please identify an instance where the model did not provide a helpful response."
        "Please generate a question that can be asked to an expert to help the user understand the instance where the model did not provide a helpful response."
        "Be sure to include details (ids, concepts, names, titles, dates, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    )
]





