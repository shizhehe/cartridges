from dataclasses import field
import random
from typing import Literal
import datasets
import torch
from cartridges.datasets import (
    CartridgeDataset,
    CartridgeGenerateDataset,
    CartridgeGenerateDatasetElement,
)
from transformers import PreTrainedTokenizerFast
from pydantic import Field

from cartridges.structs import ContextConvo, Message
from cartridges.tasks.mmlu.generate_hf_dataset import DATASET_NAME
from cartridges.train import GenerateDatasetConfig


class MMLUEvalDataset(CartridgeDataset):
    class Config(CartridgeDataset.Config):
        num_samples: int
        seed: int = 42
        label_type: Literal["tokens", "logits"] = "tokens"
        data_sources: list[tuple[str, int | None],] = Field(  # path, limit
            default_factory=list
        )

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        self.tokenizer = tokenizer

        data = [
            ContextConvo(
                id=None,
                messages=[
                    Message(
                        content=row["prompt"],
                        role="user",
                    ),
                    Message(
                        content=row["answer"],
                        role="assistant",
                    ),
                ],
                type="mmlu",
                metadata={},
            )
            for row in datasets.load_dataset(DATASET_NAME)["test"]
        ]

        random.seed(config.seed)
        random.shuffle(data)

        self.data = data[: self.config.num_samples]


def parse_chat_messages(text):
    """
    Parse text with format <|start_header_id|>role<|end_header_id|>content<|eot_id|>
    into a list of message dictionaries.
    """
    import re
    messages = []
    
    # Pattern to match each message block
    pattern = r'<\|start_header_id\|>(.*?)<\|end_header_id\|>(.*?)(?=<\|eot_id\|>)'
    
    # Find all matches
    matches = re.findall(pattern, text, re.DOTALL)
    
    for role, content in matches:
        # Clean up the content (remove leading/trailing whitespace and newlines)
        content = content.strip()
        
        # Skip empty messages
        if content:
            messages.append({
                "role": role,
                "content": content
            })
    
    return messages

class MMLUGenerateDataset(CartridgeGenerateDataset):
    class Config(CartridgeGenerateDataset.Config):
        num_problems: int = 200
        seed: int = 47

        label_type: str = "tokens"
        data_sources: list = []


    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        self.tokenizer = tokenizer

        dataset = datasets.load_dataset(
            "meta-llama/Llama-3.1-8B-Instruct-evals",
            "Llama-3.1-8B-Instruct-evals__mmlu__details",
        )["latest"]

        random.seed(config.seed)
        indexes = list(range(len(dataset)))
        random.shuffle(indexes)
        indexes = indexes[: self.config.num_problems]

        data = []
        for index in indexes:
            row = dataset[index]
            (prompt,) = row["input_final_prompts"]
            (answer,) = row["input_correct_responses"]

            prompt_messages = parse_chat_messages(
                prompt[:-len("<|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is")]
            )

            data.append(
                CartridgeGenerateDatasetElement(
                    input_ids=torch.tensor([tokenizer.encode(prompt)]),
                    prompt=prompt,
                    prompt_messages=prompt_messages,
                    answer=answer,
                    metadata={},
                )
            )

        assert (
            len(data) == self.config.num_problems
        ), f"Expected {self.config.num_problems} problems, but got {len(data)}"

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def score(
        self, pred: str, answer: str, convo_id: str
    ) -> tuple[bool, dict[str, str | None]]:
        if len(pred.strip()) == 0:
            return False, {}
    
        if pred.strip().startswith("The best answer is"):
            pred = pred.strip()[len("The best answer is"):].strip()
        return pred.strip()[0] == answer.strip()[0], {}


def mmlu_subset():
    return GenerateDatasetConfig(
        name_for_wandb="mmlu-subset",
        dataset=MMLUGenerateDataset.Config(),
        override_max_tokens=1,
        batch_size=1
    )