from __future__ import annotations

import random
import uuid

from capsules.clients.base import ClientConfig
from capsules.clients.tokasaurus_batch import CapsulesConvoWithLogprobs
from capsules.generate.generators.base import (
    ContextConvoGenerator,
    responses_and_chats_to_training_examples,
)
from capsules.generate.structs import (
    Context,
    TrainingExample,
)
from capsules.utils import get_logger

logger = get_logger(__name__)



DATA_FORMATS = [
    "JSON",
    "YAML",
    "TOML",
    "INI",
    "XML",
]


INSTRUCTION_TEMPLATES = [
    (
        "Can you write a summary of all of the information related to the company in the following 10k statement? "
        "Be sure to include all information related to dates, times, and numerical values. "
    ),
    (
        "Please extract and organize all information about the company from this statement. "
    ),
    (
        "Summarize the company details."
    ),
    (
        "What are the key details related to this company's 10k sstatement? "
        "Please provide a comprehensive analysis with all relevant details."
    ),
    (
        "Extract all numerical values from this 10k statement. "
        "Organize them."
    ),
] + [ 
    (
        "Can you structure all of the information in the following 10k statement "
        f"in the following format: {data_format}? "
        "Be sure to include all information related to the company and any dates, times, and numerical values."
    )
    for data_format in DATA_FORMATS
]


PROMPT_TEMPLATE = """{instruction}{additional_instructions}"""


ADDITIONAL_INSTRUCTIONS = [
    (
        "If helpful, you can think before responding. Put your thinking between <thinking> and </thinking> tags."
        "Then, provide your final response between <response> and </response> tags."
    ), 
    (
        "If helpful, you can think before responding. Put your thinking between <thinking> and </thinking> tags."
        "Then, provide your final response between <response> and </response> tags."
    ), 
    (
        "Respond in the following format: <thinking>...</thinking> <response>...</response>"
    ),
    (
        "Explain your reasoning before providing your final response."
    ),
    (
        "Explain your reasonining between <reasoning> and </reasoning> tags."
    ),
    (
        "Provide your final answer within <answer>...</answer> tags. Optionally, you can explain your reasoning between <reasoning> and </reasoning> tags."
    ),
    (
        ""
    )
]

SYSTEM_PROMPT_TEMPLATE = """You are a financial analyst. 
Please reference the following financial statement when answering questions.

<statement>
{context}
</statement>
"""


class ReformatGenerator(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        client: ClientConfig
        temperature: float = 0.0
        max_completion_tokens: int = 512
        # The system prompt for the answer generator. It can contain variables for `{instruction}` and `{additional_instructions}`.
        prompt_template: str = PROMPT_TEMPLATE
        # The system prompt for the answer generator. It can contain variables for `{context}`.
        system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE
        num_top_logprobs: int = 20

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config
        self.client = config.client.instantiate()

    def sample_convos(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
    ) -> list[TrainingExample]:
        routing_tag = str(uuid.uuid4())
        
        # (1) sample a subcontext  provide to the question generators
        sample_idx = random.randint(0, len(self.context.sections) - 1)
        section = self.context.sections[sample_idx]
            
        # (2) sample instructions
        instruction_templates = random.choices(INSTRUCTION_TEMPLATES, k=num_convos )
        additional_instructions = random.choices(ADDITIONAL_INSTRUCTIONS, k=num_convos)
        prompts = []
        for instruction_template, additional_instruction in zip(instruction_templates, additional_instructions):
            prompts.append(
                self.config.prompt_template.format(
                    instruction=instruction_template,
                    additional_instructions=additional_instruction,
                )
            )

        # (3) Sample answers
        system_prompt = self.config.system_prompt_template.format(context=section.to_string())
        answer_chats = [
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            for prompt in prompts
        ]
        convos: list[CapsulesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )

        # (4) Construct convos
        examples = responses_and_chats_to_training_examples(convos, answer_chats)
        return examples


